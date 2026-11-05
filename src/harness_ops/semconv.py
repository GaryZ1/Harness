"""OpenTelemetry GenAI semantic-convention names, behind a one-file mapping layer.

Research basis (reference §10.3 "OTel GenAI span tree", checklist item 1, pitfall "Pinning
dashboards to one OTel release"):

* The span tree is ``invoke_agent`` (root, per run) -> ``chat`` (per model call) ->
  ``execute_tool`` (per tool call), plus ``create_agent``. The root span is named
  ``invoke_agent {gen_ai.agent.name}`` when the name is known.
* The conventions are still in *Development* (v1.41 vs v1.42 conflict; repo move mid-2026). The
  cache-token names (``gen_ai.usage.cache_read.input_tokens``...) are **unverified**. Every
  instrumentation call in this package therefore writes the names below, and
  :class:`SemConv` renames them at record time, so a spec rename is a change to one mapping.
* Attributes the spec does not cover use the ``harness.*`` namespace from the research's
  minimum span schema (version, prompts/tools SHA, termination, cost, TTFT, retry count...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Op(str, Enum):
    """``gen_ai.operation.name`` values used by the harness."""

    INVOKE_AGENT = "invoke_agent"
    CREATE_AGENT = "create_agent"
    CHAT = "chat"
    EXECUTE_TOOL = "execute_tool"


class HarnessOp(str, Enum):
    """Custom harness spans the GenAI spec does not define (``harness.operation``)."""

    COMPACTION = "compaction"
    SUBAGENT_SPAWN = "subagent_spawn"
    APPROVAL_WAIT = "approval_wait"
    DURABLE_STEP = "durable_step"


class A:
    """Attribute names as written by the instrumentation (pre-mapping)."""

    # --- OTel GenAI (Development status) ---------------------------------------------------
    OPERATION_NAME = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"
    CONVERSATION_ID = "gen_ai.conversation.id"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"  # unverified name
    USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"  # unverified name
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"  # content: opt-in
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"  # content: opt-in
    INPUT_MESSAGES = "gen_ai.input.messages"  # content: opt-in
    OUTPUT_MESSAGES = "gen_ai.output.messages"  # content: opt-in
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"  # content: opt-in
    ERROR_TYPE = "error.type"

    # --- harness.* (research minimum schema; not spec names) -------------------------------
    OPERATION = "harness.operation"
    VERSION = "harness.version"
    PROMPTS_TOOLS_SHA = "harness.prompts_tools_sha"
    EFFORT = "harness.effort"
    SESSION_ID = "harness.session_id"
    TERMINATION = "harness.termination"
    TOTAL_COST_USD = "harness.total_cost_usd"
    NUM_TURNS = "harness.num_turns"
    COST_USD = "harness.cost_usd"
    TTFT_MS = "harness.ttft_ms"
    LATENCY_MS = "harness.latency_ms"
    STOP_REASON = "harness.stop_reason"
    RETRY_COUNT = "harness.retry_count"
    FALLBACK_FROM = "harness.fallback_from"
    TOTAL_INPUT_TOKENS = "harness.usage.total_input_tokens"
    CACHE_HIT_RATIO = "harness.cache.hit_ratio"
    CACHE_OBSERVATION = "harness.cache.observation"
    CACHE_EXPECTED_REBUILD = "harness.cache.expected_rebuild"
    TOOL_RESULT_TOKENS = "harness.tool.result_tokens"
    TOOL_TRUNCATED = "harness.tool.truncated"
    TOOL_IS_ERROR = "harness.tool.is_error"
    DURATION_MS = "harness.duration_ms"
    IDEMPOTENCY_KEY = "harness.idempotency_key"
    SANDBOX_ID = "harness.sandbox_id"
    COMPACTION_TOKENS_BEFORE = "harness.compaction.tokens_before"
    COMPACTION_TOKENS_AFTER = "harness.compaction.tokens_after"
    SUBAGENT_PARENT_TOOL_CALL_ID = "harness.subagent.parent_tool_call_id"
    SUBAGENT_DEPTH = "harness.subagent.depth"
    APPROVAL_DECISION = "harness.approval.decision"
    APPROVAL_TOOL = "harness.approval.tool"
    STEP_KEY = "harness.step.key"
    STEP_REPLAYED = "harness.step.replayed"
    CONTENT = "harness.content"  # content: opt-in


@dataclass(frozen=True)
class SemConv:
    """Attribute rename layer. ``overrides`` maps a name written by the code to the wire name.

    >>> SemConv({"gen_ai.usage.cache_read.input_tokens": "gen_ai.usage.cache_read_input_tokens"}).name(A.USAGE_CACHE_READ_INPUT_TOKENS)
    'gen_ai.usage.cache_read_input_tokens'

    Map a name to ``""`` to drop the attribute entirely.
    """

    overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def name(self, key: str) -> str:
        return self.overrides.get(key, key)

    def apply(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in attributes.items():
            # ``<content key>.sha256`` / ``.length`` follow their base key's rename.
            base, dot, suffix = k.rpartition(".")
            if k not in self.overrides and suffix in ("sha256", "length") and base in self.overrides:
                mapped_base = self.overrides[base]
                if mapped_base:
                    out[f"{mapped_base}{dot}{suffix}"] = v
                continue
            mapped = self.name(k)
            if mapped:
                out[mapped] = v
        return out


def agent_span_name(agent_name: str | None) -> str:
    """``invoke_agent {gen_ai.agent.name}`` when the name is known, else ``invoke_agent``."""
    return f"{Op.INVOKE_AGENT.value} {agent_name}" if agent_name else Op.INVOKE_AGENT.value


def chat_span_name(model: str | None) -> str:
    return f"{Op.CHAT.value} {model}" if model else Op.CHAT.value


def tool_span_name(tool_name: str | None) -> str:
    return f"{Op.EXECUTE_TOOL.value} {tool_name}" if tool_name else Op.EXECUTE_TOOL.value
