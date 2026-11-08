"""Optional Anthropic Messages API adapter implementing :class:`~harness_core.types.ModelClient`.

``pip install anthropic`` is required only when this class is constructed without a ``client``;
the import is lazy. The model id is always supplied by the caller (configuration), never defaulted.

What the adapter guarantees (reference report section 2.7 step 6, section 2.8; current API facts):

* the assistant turn comes back as the *full* ``response.content`` translated with
  :func:`~harness_core.types.block_from_api`; thinking, redacted-thinking, compaction and
  server-tool blocks become :class:`~harness_core.types.OpaqueBlock` and are sent back unchanged;
* reasoning is configured once per client (``thinking={"type": "adaptive"}``, optional
  ``output_config.effort``) so it never changes mid-session and never busts the cache;
* parameters rejected by current models are refused up front with
  :class:`~harness_core.errors.ConfigError`: ``budget_tokens``, ``temperature``, ``top_p``,
  ``top_k``, and forced ``tool_choice`` (``any``/``tool``) - use ``auto``;
* beta features are routed through ``client.beta.messages.create``; ``context_management`` edits
  add their beta header automatically (``clear_tool_uses_20250919`` ->
  ``context-management-2025-06-27``, ``compact_20260112`` -> ``compact-2026-01-12``);
* SDK exceptions are classified into the core error classes so the loop's retry policy applies.
  A client constructed here uses ``max_retries=0`` so retries are not doubled.
"""

from __future__ import annotations

from typing import Any, Sequence

from .errors import ConfigError, ContextWindowExceededError, ModelError, RateLimitError, TerminalModelError, TransientModelError
from .types import ModelRequest, ModelResponse, StopReason, Usage, block_from_api

FORBIDDEN_PARAMS = frozenset({"temperature", "top_p", "top_k", "budget_tokens"})
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EDIT_BETAS = {
    "clear_tool_uses_20250919": "context-management-2025-06-27",
    "clear_thinking_20251015": "context-management-2025-06-27",
    "compact_20260112": "compact-2026-01-12",
}
_CONTEXT_MARKERS = ("prompt is too long", "context window", "context_window", "too many tokens", "maximum context")


class AnthropicModelClient:
    """``ModelClient`` over ``anthropic.Anthropic().messages.create``.

    :param model: model id (required).
    :param client: an existing SDK client (or any object with the same ``messages.create`` shape).
    :param effort: ``output_config.effort`` (``low``/``medium``/``high``/``xhigh``/``max``); ``None``
        leaves the model default.
    :param thinking: thinking config; default adaptive. ``None`` omits it.
    :param betas: beta headers sent on every call.
    :param extra: extra ``messages.create`` keyword arguments sent on every call.
    """

    def __init__(self, model: str, *, client: Any | None = None, effort: str | None = None,
                 thinking: dict[str, Any] | None = None, adaptive_thinking: bool = True, betas: Sequence[str] = (),
                 extra: dict[str, Any] | None = None, client_kwargs: dict[str, Any] | None = None) -> None:
        if not model or not isinstance(model, str):
            raise ConfigError("model id is required (pass it from configuration)")
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ConfigError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("AnthropicModelClient needs the 'anthropic' package: pip install anthropic") from exc
            client = anthropic.Anthropic(**{"max_retries": 0, **(client_kwargs or {})})
        self.client = client
        self.model = model
        self.effort = effort
        self.thinking = thinking if thinking is not None else ({"type": "adaptive"} if adaptive_thinking else None)
        self.betas = list(betas)
        self.extra = dict(extra or {})
        _validate_params(self.extra)
        if self.thinking:
            _validate_params({"thinking": self.thinking})

    # ------------------------------------------------------------------ request translation
    def build_params(self, request: ModelRequest) -> tuple[dict[str, Any], list[str]]:
        """Translate a :class:`ModelRequest` into ``messages.create`` kwargs plus beta headers."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [m.to_api() for m in request.messages],
        }
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [t.to_api() for t in request.tools]
            if all(t.defer_loading for t in request.tools):
                raise ConfigError("never defer all tools: at least one tool (e.g. the tool-search tool) must be loaded")
        if self.thinking:
            params["thinking"] = dict(self.thinking)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        extra = {**self.extra, **request.extra}
        betas = list(self.betas) + list(extra.pop("betas", []) or [])
        # Per-request effort (e.g. a sub-agent's ``extra["effort"]``) is not a top-level API
        # parameter: it belongs in ``output_config.effort``.
        effort = extra.pop("effort", None)
        if effort is not None:
            if effort not in EFFORT_LEVELS:
                raise ConfigError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
            params["output_config"] = {**params.get("output_config", {}), "effort": effort}
        if "output_config" in extra:
            params["output_config"] = {**params.get("output_config", {}), **extra.pop("output_config")}
        params.update(extra)
        _validate_params(params)
        for edit in (params.get("context_management") or {}).get("edits", []) or []:
            beta = EDIT_BETAS.get(str(edit.get("type", "")))
            if beta and beta not in betas:
                betas.append(beta)
        return params, betas

    def create(self, request: ModelRequest) -> ModelResponse:
        params, betas = self.build_params(request)
        try:
            if betas:
                resp = self.client.beta.messages.create(betas=betas, **params)
            else:
                resp = self.client.messages.create(**params)
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            raise classify_exception(exc) from exc
        return response_from_api(resp)


def _validate_params(params: dict[str, Any]) -> None:
    bad = sorted(FORBIDDEN_PARAMS & set(params))
    if bad:
        raise ConfigError(f"parameters not accepted by current models: {bad}")
    thinking = params.get("thinking")
    if isinstance(thinking, dict) and ("budget_tokens" in thinking or thinking.get("type") == "enabled"):
        raise ConfigError("manual thinking budgets are rejected; use thinking={'type': 'adaptive'} plus output_config.effort")
    tc = params.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") in ("any", "tool"):
        raise ConfigError("forced tool_choice ('any'/'tool') returns 400 on the newest models; use {'type': 'auto'}")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def response_from_api(resp: Any) -> ModelResponse:
    """Translate an SDK ``Message`` (or an equivalent dict) into a :class:`ModelResponse`."""
    raw_usage = _get(resp, "usage")
    usage = Usage(
        input_tokens=int(_get(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(_get(raw_usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(_get(raw_usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(_get(raw_usage, "cache_creation_input_tokens", 0) or 0),
    ) if raw_usage is not None else Usage()
    return ModelResponse(
        content=[block_from_api(b) for b in (_get(resp, "content") or [])],
        stop_reason=StopReason.parse(_get(resp, "stop_reason")),
        usage=usage,
        model=str(_get(resp, "model", "") or ""),
        id=str(_get(resp, "id", "") or ""),
        raw=resp,
    )


def _retry_after(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_exception(exc: Exception) -> ModelError:
    """Map an SDK / transport exception onto the core retry classes."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    message = str(exc)
    lowered = message.lower()
    if status == 429 or name == "RateLimitError":
        return RateLimitError(message, retry_after=_retry_after(exc))
    if name in ("APIConnectionError", "APITimeoutError") or isinstance(exc, (ConnectionError, TimeoutError)):
        return TransientModelError(message)
    if isinstance(status, int):
        if status in (408, 409) or status >= 500:
            return TransientModelError(message)
        if status in (400, 413) and any(m in lowered for m in _CONTEXT_MARKERS):
            return ContextWindowExceededError(message)
        return TerminalModelError(f"HTTP {status}: {message}")
    if name in ("InternalServerError", "OverloadedError", "ServiceUnavailableError"):
        return TransientModelError(message)
    return TerminalModelError(f"{name}: {message}")
