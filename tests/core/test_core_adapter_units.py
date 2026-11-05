"""Anthropic adapter (with a fake SDK client), stuck detector and context helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness_core import (
    AgentLoop,
    CompactionError,
    ConfigError,
    ContextWindowExceededError,
    FunctionToolExecutor,
    Message,
    ModelRequest,
    OpaqueBlock,
    RateLimitError,
    StopReason,
    StuckConfig,
    StuckDetector,
    TerminalModelError,
    TextBlock,
    ToolCall,
    ToolOutcome,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    TransientModelError,
    validate_tool_pairing,
)
from harness_core.anthropic_adapter import AnthropicModelClient, classify_exception, response_from_api


class SdkBlock:
    """Mimics an SDK pydantic block (``model_dump``)."""

    def __init__(self, **kw):
        self._d = kw

    def model_dump(self):
        return dict(self._d)


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def sdk_reply(*blocks, stop_reason="end_turn", model="test-model"):
    return SimpleNamespace(id="msg_1", model=model, stop_reason=stop_reason, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=11, output_tokens=7, cache_read_input_tokens=5,
                                                 cache_creation_input_tokens=None))


def fake_client(*replies, beta_replies=()):
    return SimpleNamespace(messages=FakeMessages(replies), beta=SimpleNamespace(messages=FakeMessages(beta_replies)))


def test_model_id_is_required():
    with pytest.raises(ConfigError):
        AnthropicModelClient("", client=fake_client())


def test_build_params_uses_adaptive_thinking_effort_and_native_shapes():
    c = AnthropicModelClient("test-model", client=fake_client(), effort="high")
    req = ModelRequest(system="sys", messages=[Message.user("hi"), Message.system("operator note")],
                       tools=[ToolSpec("read", "Read", read_only=True), ToolSpec("memory", "", provider_type="memory_20250818")],
                       max_tokens=1234)
    params, betas = c.build_params(req)
    assert params["model"] == "test-model" and params["max_tokens"] == 1234 and params["system"] == "sys"
    assert params["thinking"] == {"type": "adaptive"} and params["output_config"] == {"effort": "high"}
    assert params["messages"][1] == {"role": "system", "content": [{"type": "text", "text": "operator note"}]}
    assert params["tools"][1] == {"type": "memory_20250818", "name": "memory"}
    assert betas == []
    for forbidden in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert forbidden not in params


@pytest.mark.parametrize("extra", [{"temperature": 0.2}, {"top_k": 5}, {"thinking": {"type": "enabled", "budget_tokens": 1000}},
                                   {"tool_choice": {"type": "any"}}, {"tool_choice": {"type": "tool", "name": "x"}}])
def test_forbidden_parameters_are_refused(extra):
    c = AnthropicModelClient("test-model", client=fake_client())
    with pytest.raises(ConfigError):
        c.build_params(ModelRequest(system="", messages=[Message.user("x")], extra=extra))


def test_forbidden_parameters_in_constructor_extra_are_refused():
    with pytest.raises(ConfigError):
        AnthropicModelClient("test-model", client=fake_client(), extra={"temperature": 1})
    with pytest.raises(ConfigError):
        AnthropicModelClient("test-model", client=fake_client(), effort="extreme")


def test_never_defer_all_tools():
    c = AnthropicModelClient("test-model", client=fake_client())
    with pytest.raises(ConfigError, match="never defer all tools"):
        c.build_params(ModelRequest(system="", messages=[Message.user("x")], tools=[ToolSpec("a", "a", defer_loading=True)]))


def test_context_management_routes_to_beta_with_header():
    client = fake_client(beta_replies=[sdk_reply(SdkBlock(type="text", text="ok"))])
    c = AnthropicModelClient("test-model", client=client)
    req = ModelRequest(system="", messages=[Message.user("x")],
                       extra={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}, {"type": "compact_20260112"}]},
                              "output_config": {"effort": "low"}})
    resp = c.create(req)
    call = client.beta.messages.calls[0]
    assert call["betas"] == ["context-management-2025-06-27", "compact-2026-01-12"]
    assert call["output_config"] == {"effort": "low"} and resp.text == "ok"
    assert client.messages.calls == []


def test_response_translation_keeps_opaque_blocks_and_usage():
    reply = sdk_reply(SdkBlock(type="thinking", thinking="", signature="SIG", citations=None),
                      SdkBlock(type="text", text="hi", citations=None),
                      SdkBlock(type="tool_use", id="tu1", name="read", input={"p": 1}), stop_reason="tool_use")
    resp = response_from_api(reply)
    assert resp.stop_reason is StopReason.TOOL_USE and resp.model == "test-model"
    assert isinstance(resp.content[0], OpaqueBlock) and resp.content[0].to_api() == {"type": "thinking", "thinking": "", "signature": "SIG"}
    assert isinstance(resp.content[2], ToolUseBlock)
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.cache_read_input_tokens, resp.usage.cache_creation_input_tokens) == (11, 7, 5, 0)
    assert response_from_api(SimpleNamespace(content=[], stop_reason="weird_new_reason")).stop_reason is StopReason.OTHER


def test_adapter_drives_the_loop_and_replays_thinking_unchanged():
    client = fake_client(
        sdk_reply(SdkBlock(type="thinking", thinking="", signature="SIG"), SdkBlock(type="tool_use", id="tu1", name="read", input={}),
                  stop_reason="tool_use"),
        sdk_reply(SdkBlock(type="text", text="done")),
    )
    ex = FunctionToolExecutor()
    ex.tool("read", "read", read_only=True)(lambda i, c: "R")
    result = AgentLoop(AnthropicModelClient("test-model", client=client), ex).run("go")
    assert result.ok
    second = client.messages.calls[1]["messages"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "SIG"}
    assert second[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "R"}]}


class _Exc(Exception):
    def __init__(self, msg, status=None, headers=None):
        super().__init__(msg)
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


def _named(name, msg, **kw):
    return type(name, (_Exc,), {})(msg, **kw)


def test_exception_classification():
    rl = classify_exception(_Exc("slow down", 429, {"retry-after": "3"}))
    assert isinstance(rl, RateLimitError) and rl.retry_after == 3.0
    assert isinstance(classify_exception(_Exc("overloaded", 529)), TransientModelError)
    assert isinstance(classify_exception(_Exc("bad gateway", 502)), TransientModelError)
    assert isinstance(classify_exception(_named("APIConnectionError", "reset")), TransientModelError)
    assert isinstance(classify_exception(_Exc("prompt is too long: 250000 tokens", 400)), ContextWindowExceededError)
    assert isinstance(classify_exception(_Exc("invalid x-api-key", 401)), TerminalModelError)
    assert isinstance(classify_exception(_Exc("bad field", 400)), TerminalModelError)
    assert isinstance(classify_exception(ValueError("?")), TerminalModelError)


def test_adapter_raises_classified_errors():
    client = fake_client(_Exc("overloaded", 529))
    with pytest.raises(TransientModelError):
        AnthropicModelClient("test-model", client=client).create(ModelRequest(system="", messages=[Message.user("x")]))


# ----------------------------------------------------------------------------- stuck detector


def test_stuck_detector_identical_calls_and_patterns():
    d = StuckDetector(StuckConfig(identical_call_threshold=3, pattern_repeats=3))
    a, b = ToolCall("1", "read", {"p": "a"}), ToolCall("2", "read", {"p": "b"})
    assert d.observe_calls([a]) is None and d.observe_calls([a]) is None
    sig = d.observe_calls([a])
    assert sig is not None and sig.kind == "identical_call" and sig.hard
    d.reset()
    signals = [d.observe_calls([x]) for x in [a, b, a, b, a, b]]
    assert signals[-1] is not None and signals[-1].kind == "repeating_pattern"
    assert all(s is None for s in signals[:-1])


def test_stuck_detector_identical_errors_and_key_order_insensitivity():
    d = StuckDetector(StuckConfig(identical_call_threshold=None, pattern_repeats=None, identical_error_threshold=2))
    c1, c2 = ToolCall("1", "run", {"a": 1, "b": 2}), ToolCall("2", "run", {"b": 2, "a": 1})
    err = ToolOutcome("boom", is_error=True)
    assert d.observe_results([c1], [err]) is None
    sig = d.observe_results([c2], [err])
    assert sig is not None and sig.kind == "identical_error"
    assert d.observe_results([c1], [ToolOutcome("fine")]) is None


def test_validate_tool_pairing():
    use = Message.assistant([ToolUseBlock("t1", "read", {})])
    ok = Message.user([ToolResultBlock("t1", "x")])
    validate_tool_pairing([Message.user("hi"), use, ok, Message.assistant([TextBlock("done")])])
    validate_tool_pairing([Message.user("s"), use])  # trailing open batch is allowed (in progress)
    with pytest.raises(CompactionError):
        validate_tool_pairing([Message.user([ToolResultBlock("t9", "x")])])
    with pytest.raises(CompactionError):
        validate_tool_pairing([use, Message.user("forgot the result")])
