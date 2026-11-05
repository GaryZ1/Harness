from types import SimpleNamespace as NS

import pytest

from harness_memory import AnthropicLLM, MemoryToolHandler, run_memory_agent
from harness_memory.llm import LLMRefusal


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def client_with(*responses):
    return NS(messages=FakeMessages(responses))


def test_anthropic_llm_params_and_text():
    c = client_with(NS(stop_reason="end_turn", content=[NS(type="thinking", thinking=""), NS(type="text", text="{}")]))
    llm = AnthropicLLM("model-under-test", client=c, effort="medium")
    assert llm.complete(system="sys", prompt="hi") == "{}"
    call = c.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"} and call["output_config"] == {"effort": "medium"}
    assert call["system"] == "sys" and call["messages"] == [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError):
        AnthropicLLM("", client=c)


def test_anthropic_llm_refusal():
    c = client_with(NS(stop_reason="refusal", content=[]))
    with pytest.raises(LLMRefusal):
        AnthropicLLM("m", client=c).complete(system="s", prompt="p")


def test_memory_agent_loop(store):
    tool_use = NS(
        type="tool_use",
        id="tu_1",
        name="memory",
        input={"command": "create", "path": "/memories/progress.md", "file_text": "step 1 done\n"},
    )
    view = NS(type="tool_use", id="tu_2", name="memory", input={"command": "view", "path": "/memories"})
    c = client_with(
        NS(stop_reason="tool_use", content=[tool_use, view]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="saved")]),
    )
    messages = [{"role": "user", "content": "work"}]
    final = run_memory_agent(c, model="m", system="s", messages=messages, handler=MemoryToolHandler(store))
    assert final.content[0].text == "saved"
    assert store.read("progress.md").content == "step 1 done\n"
    results = messages[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]  # all results in ONE user message
    assert c.messages.calls[0]["tools"][0] == {"type": "memory_20250818", "name": "memory"}
