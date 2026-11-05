"""Content policy: structure-only telemetry by default, opt-in content capture, secrets always redacted.

Research basis (reference §10.3 "Structure-only production tracing", checklist item 2):

* OTel GenAI treats message content as **opt-in**; Anthropic's production research agent was
  debugged from "agent decision patterns and interaction structures" without reading
  conversation contents.
* The default is therefore *structure only*: tool names, argument **hashes**, token counts, error
  classes and timings. Content attributes are replaced by ``<key>.sha256`` + ``<key>.length``.
* With capture enabled (eval runs, dogfooding, opted-in tenants) content is still passed through
  :class:`harness_memory.SecretRedactor` and truncated before it reaches any exporter.

Hashes of short content can be reversed by dictionary attack ("yes", an e-mail address...). Pass
``hash_key`` to switch to HMAC-SHA256 with a per-deployment secret.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from harness_memory import SecretRedactor

from ._util import sha256_hex

#: Span attributes that carry message/tool content (OTel GenAI names + our ``harness.*`` extras).
DEFAULT_CONTENT_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.prompt",
        "gen_ai.completion",
        "harness.content",
    }
)

#: Keys inside :class:`~harness_core.types.Event` ``data`` that carry content.
DEFAULT_CONTENT_EVENT_KEYS: frozenset[str] = frozenset(
    {"text", "content", "prompt", "messages", "input", "output", "arguments", "result", "system", "completion", "query", "body"}
)

#: Environment switches. ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is the variable the
#: OpenTelemetry Python GenAI instrumentations use; ``HARNESS_CAPTURE_CONTENT`` is ours.
CAPTURE_ENV_VARS = ("HARNESS_CAPTURE_CONTENT", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")

AttributeValue = Any


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ContentPolicy:
    """Decides what content leaves the process. Immutable; share one instance per tracer/sink."""

    capture_content: bool = False
    max_content_chars: int = 16_384
    max_attribute_chars: int = 4_096
    redactor: SecretRedactor = field(default_factory=SecretRedactor, compare=False)
    content_attributes: frozenset[str] = DEFAULT_CONTENT_ATTRIBUTES
    content_event_keys: frozenset[str] = DEFAULT_CONTENT_EVENT_KEYS
    hash_key: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_content_chars < 1 or self.max_attribute_chars < 1:
            raise ValueError("max_content_chars and max_attribute_chars must be >= 1")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> "ContentPolicy":
        """Capture is on only if one of :data:`CAPTURE_ENV_VARS` is truthy."""
        env = os.environ if env is None else env
        capture = any(_truthy(env.get(name)) for name in CAPTURE_ENV_VARS)
        return cls(capture_content=overrides.pop("capture_content", capture), **overrides)

    # ------------------------------------------------------------------ primitives

    def digest(self, text: str) -> str:
        return sha256_hex(text, self.hash_key)

    def redact(self, text: str, limit: int | None = None) -> str:
        """Redact secrets, then truncate to ``limit`` (default: ``max_attribute_chars``)."""
        out = self.redactor.redact(text)
        limit = self.max_attribute_chars if limit is None else limit
        if len(out) > limit:
            out = out[:limit] + f"...[{len(out) - limit} chars truncated]"
        return out

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))

    # ------------------------------------------------------------------ span attributes

    def sanitize_attribute(self, key: str, value: Any) -> dict[str, AttributeValue]:
        """Return the attribute(s) to record for ``key=value`` (possibly none, possibly two)."""
        if value is None:
            return {}
        if key in self.content_attributes:
            text = self._as_text(value)
            if not self.capture_content:
                return {f"{key}.sha256": self.digest(text), f"{key}.length": len(text)}
            return {key: self.redact(text, self.max_content_chars)}
        return {key: self._coerce(value)}

    def _coerce(self, value: Any) -> AttributeValue:
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, (list, tuple)):
            items = list(value)
            if all(isinstance(v, bool) for v in items) or all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in items):
                return items
            return [self.redact(v if isinstance(v, str) else self._as_text(v)) for v in items]
        if hasattr(value, "value") and isinstance(value.value, (str, int)):  # Enum
            return self._coerce(value.value)
        return self.redact(self._as_text(value))

    def sanitize_attributes(self, attributes: Mapping[str, Any]) -> dict[str, AttributeValue]:
        out: dict[str, AttributeValue] = {}
        for k, v in attributes.items():
            out.update(self.sanitize_attribute(k, v))
        return out

    # ------------------------------------------------------------------ event payloads

    def sanitize_event_data(self, data: Any, _depth: int = 0) -> Any:
        """Recursively strip content keys from an Event payload (replaced by a digest) and redact strings."""
        if _depth > 12:
            return "[max depth]"
        if isinstance(data, Mapping):
            out: dict[str, Any] = {}
            for k, v in data.items():
                key = str(k)
                if key in self.content_event_keys and v is not None and not isinstance(v, (bool, int, float)):
                    text = self._as_text(v)
                    if self.capture_content:
                        out[key] = self.redact(text, self.max_content_chars) if isinstance(v, str) else self._redact_tree(v, _depth + 1)
                    else:
                        out[key] = {"sha256": self.digest(text), "chars": len(text)}
                else:
                    out[key] = self.sanitize_event_data(v, _depth + 1)
            return out
        if isinstance(data, (list, tuple)):
            return [self.sanitize_event_data(v, _depth + 1) for v in data]
        if isinstance(data, str):
            return self.redact(data)
        return data

    def _redact_tree(self, data: Any, depth: int) -> Any:
        if depth > 12:
            return "[max depth]"
        if isinstance(data, Mapping):
            return {str(k): self._redact_tree(v, depth + 1) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._redact_tree(v, depth + 1) for v in data]
        if isinstance(data, str):
            return self.redact(data, self.max_content_chars)
        return data
