import pytest
from context_fakes import TOOLS

from harness_context import ConfigurationError, HeuristicTokenCounter, PromptLayer, PromptStack, SkillEntry
from harness_context.layers import LayerKind, normalize_text
from harness_context.tokens import AnthropicTokenCounter
from harness_core.types import Message, ModelRequest


def test_heuristic_counter_is_conservative_and_structural():
    c = HeuristicTokenCounter()
    assert c.count_text("") == 0
    assert c.count_text("abcd") == 1 and c.count_text("abcde") == 2
    req = ModelRequest(system="s" * 400, messages=[Message.user("u" * 400)], tools=TOOLS)
    total = c.count_request(req)
    assert total > 200  # 100 system + ~100 message + tool definitions
    assert c.count_request(ModelRequest(system="", messages=[], tools=[])) == 0


class _FakeMessages:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def count_tokens(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("network down")
        return type("R", (), {"input_tokens": 1234})()


class _FakeAnthropic:
    def __init__(self, fail=False):
        self.messages = _FakeMessages(fail)


def test_anthropic_counter_uses_count_tokens_and_falls_back():
    client = _FakeAnthropic()
    counter = AnthropicTokenCounter("test-model", client=client)
    req = ModelRequest(system=[{"type": "text", "text": "sys"}], messages=[Message.user("hi")], tools=TOOLS)
    assert counter.count_request(req) == 1234
    call = client.messages.calls[0]
    assert call["model"] == "test-model" and call["messages"][0]["role"] == "user" and len(call["tools"]) == 2
    failing = AnthropicTokenCounter("test-model", client=_FakeAnthropic(fail=True))
    assert failing.count_request(req) == HeuristicTokenCounter().count_request(req)
    with pytest.raises(RuntimeError):
        AnthropicTokenCounter("test-model", client=_FakeAnthropic(fail=True), strict=True).count_request(req)
    with pytest.raises(ValueError):
        AnthropicTokenCounter("", client=client)


def test_stack_renders_fixed_order_and_strips_comments():
    stack = PromptStack()
    stack.add_instructions("Use pytest.\n<!-- maintainer note: secret plan -->\nKeep diffs small.   \r\n", source="AGENTS.md")
    stack.add(PromptLayer("style", "Be concise.", LayerKind.SYSTEM))
    stack.add_skills_index([SkillEntry("zeta", "Last skill"), SkillEntry("alpha", "First skill"), SkillEntry("hidden", "x", model_invocable=False)])
    out = stack.render("You are a coding agent.")
    texts = [b.text for b in out.blocks]
    assert texts[0] == "You are a coding agent." and texts[1] == "Be concise."  # SYSTEM layers before PROJECT
    assert "maintainer note" not in texts[2] and texts[2].startswith('<instructions source="AGENTS.md">')
    assert "Keep diffs small.\n" in texts[2]
    assert texts[3].index("alpha") < texts[3].index("zeta") and "hidden" not in texts[3]
    assert [b.text for b in stack.render("You are a coding agent.").blocks] == texts  # byte-stable


def test_stack_lints_and_budgets():
    long_instr = "\n".join(f"rule {i}" for i in range(250))
    stack = PromptStack(project_byte_budget=600)
    stack.add_instructions(long_instr)
    stack.add_memory_index("- memory line\n" * 10)
    out = stack.render("base")
    msgs = [la.message for la in out.lints]
    assert any("250 lines" in m for m in msgs)
    assert any("truncated" in m for m in msgs)
    assert any("dropped" in m for m in msgs)  # memory index did not fit the remaining budget
    assert "[... truncated: over the" in out.blocks[1].text


def test_memory_index_line_cap_and_layer_validation():
    stack = PromptStack(project_byte_budget=None)
    stack.add_memory_index("\n".join(f"- m{i}" for i in range(300)))
    text = stack.render("").blocks[0].text
    assert "- m199" in text and "- m200" not in text and "100 more lines not loaded" in text
    with pytest.raises(ConfigurationError):
        stack.add_memory_index("again")  # duplicate name
    with pytest.raises(ConfigurationError):
        PromptLayer("bad name!", "x")
    assert normalize_text("a  \r\nb\r\n\n") == "a\nb"


def test_reference_index_layer_is_project_context():
    from harness_context import ContextReference

    stack = PromptStack().add_reference_index([ContextReference("file", "docs/api.md", "API notes")])
    out = stack.render("base")
    assert out.kinds[-1] is LayerKind.PROJECT and "docs/api.md" in out.blocks[-1].text
