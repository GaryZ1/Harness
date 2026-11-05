"""End-to-end runs: kernel tools in a tmp workspace, bash through the sandbox runner, memory, telemetry, resume."""

from __future__ import annotations

import logging

from integration_support import (
    approve_all_fixture,  # noqa: F401 - fixture
    build,
    events_of,
    make_config,
    read_jsonl,
    text_response,
    tool_response,
    tool_results,
)

from harness_core.anthropic_adapter import AnthropicModelClient
from harness_core.loop import idempotency_key
from harness_core.results import ResultSubtype
from harness_core.types import EventType, RunContext, ToolCall, ToolOutcome, ToolSpec
from harness_ops.idempotency import idempotency_key as ops_idempotency_key
from harness_sandbox import BaseRunner

logging.getLogger("harness_sandbox").setLevel(logging.ERROR)


def test_full_run_reads_edits_and_runs_bash_through_the_sandbox_runner(tmp_path, approve_all):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "app.py"
    target.write_text("GREETING = 'hello'\n")
    script = [
        tool_response(("read", {"path": str(target)})),
        tool_response(("edit", {"path": str(target), "old_string": "'hello'", "new_string": "'hello, world'"})),
        tool_response(("bash", {"command": "cat app.py && pwd"})),
        text_response("Updated the greeting and verified it."),
    ]
    h, client = build(tmp_path, script, approver=approve_all)
    result = h.run("Change the greeting to 'hello, world' and show the file.")

    assert result.subtype is ResultSubtype.SUCCESS
    assert target.read_text() == "GREETING = 'hello, world'\n"
    results = tool_results(result)
    assert [r.is_error for r in results] == [False, False, False]
    assert "hello, world" in results[2].text and str(ws.resolve()) in results[2].text
    # bash went through the harness_sandbox runner (adapted to the CommandRunner protocol), not the tools default
    assert isinstance(h.kernel.runner.inner, BaseRunner)
    assert h.runner.choice.backend in ("bubblewrap", "seatbelt", "local")
    # the model saw the kernel + memory tools, the sandbox description and the memory context in the system prompt
    req = client.requests[0]
    assert {"read", "write", "edit", "bash", "grep", "glob", "memory"} <= {t.name for t in req.tools}
    system_text = "\n".join(b["text"] for b in req.system)
    assert "# Sandbox" in system_text and "<memory-protocol>" in system_text
    h.end_session()


def test_memory_tool_create_and_citation_recorded(tmp_path):
    script = [
        tool_response(("memory", {"command": "create", "path": "/memories/feedback_style.md", "file_text": "Prefers short answers.\n"})),
        text_response("Noted.\n<memory-citation>\nfeedback_style.md:1-1|note=[style preference]\n</memory-citation>"),
    ]
    h, client = build(tmp_path, script)
    result = h.run("Remember that I prefer short answers.")

    assert result.subtype is ResultSubtype.SUCCESS
    assert result.final_text == "Noted."  # citation trailer stripped by Session.finish_reply
    assert "<memory-citation>" in h.last_raw_reply
    store_root = h.memory_session.memory_tool.store.root
    assert (store_root / "feedback_style.md").read_text().strip().endswith("Prefers short answers.")
    rows = h.memory_session.usage.rows()
    assert [(r["path"], r["start"], r["session"]) for r in rows] == [("feedback_style.md", 1, h.memory_session.session_id)]
    # the memory tool is exposed as the provider-defined memory_20250818 type
    spec = next(t for t in client.requests[0].tools if t.name == "memory")
    assert spec.provider_type == "memory_20250818" and spec.to_api() == {"type": "memory_20250818", "name": "memory"}
    h.end_session()
    meta = h.memory.registry.get(h.memory_session.session_id)
    assert meta.ended


def test_metrics_traces_and_events_are_written(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("alpha\n")
    h, _ = build(tmp_path, [tool_response(("read", {"path": str(ws / "a.txt")})), text_response("done")])
    h.run("Read a.txt")
    h.end_session("finished")

    tel = h.telemetry_dir
    span_names = {s["name"] for s in read_jsonl(tel / "spans.jsonl")}
    assert {"invoke_agent", "chat", "execute_tool", "context.prepare"} <= span_names
    events = read_jsonl(tel / "events.jsonl")
    assert {"session_start", "model_response", "tool_result", "permission", "stop", "session_end"} <= {e["type"] for e in events}
    tool_ev = next(e for e in events if e["type"] == "tool_result")
    assert tool_ev["data"]["tool"] == "read"  # OpsEventAdapter filled the tool name
    assert read_jsonl(tel / "audit.jsonl")  # permission engine audit trail
    prom = (tel / "metrics.prom").read_text()
    assert 'harness_tool_calls_total{tool="read",outcome="ok"} 1' in prom
    assert 'harness_stop_reasons_total{reason="success"} 1' in prom
    dash = h.metrics()
    assert dash["task_success_rate"] == 1.0 and dash["stop_reasons"] == {"success": 1.0}
    assert dash["permission_rates"]["allow"] == 1.0 and dash["turns_per_task_p50"] is not None


def test_durable_session_resumes_from_the_log(tmp_path):
    cfg = make_config(tmp_path)
    h1, _ = build(tmp_path, [text_response("first answer")], config=cfg, session_id="resume-me")
    assert h1.run("first question").final_text == "first answer"
    h1.end_session()

    h2, client2 = build(tmp_path, [text_response("second answer")], config=cfg, session_id="resume-me")
    assert h2.session.id == "resume-me" and h2.store is not None
    assert [m.text for m in h2.session.messages()] == ["first question", "first answer"]
    h2.run("second question")
    sent = [m.text for m in client2.requests[0].messages]
    assert sent[:2] == ["first question", "first answer"] and "second question" in sent[2]
    assert h2.memory_session.session_id == "resume-me-r1"  # memory transcripts are per process run
    h2.end_session()


def test_tool_context_metadata_is_populated(tmp_path):
    seen: list[dict] = []

    class Probe:
        def specs(self):
            return [ToolSpec("probe", "Report the context.", read_only=True, metadata={"permission_kind": "internal"})]

        def execute(self, call: ToolCall, ctx: RunContext) -> ToolOutcome:
            seen.append(dict(ctx.metadata))
            return ToolOutcome("ok")

    script = [tool_response(("probe", {}, "tu_probe")), text_response("done")]
    cfg = make_config(tmp_path)
    cfg.permissions.mode = "accept-edits"
    h, _ = build(tmp_path, script, config=cfg, tool_executor=Probe())
    h.loop.ctx.metadata["rollout_id"] = "rollout_1"
    h.run("probe")
    md = seen[0]
    step = md["step_idx"]
    assert step == next(e.seq for e in events_of(h, EventType.MODEL_RESPONSE) if "message" in e.data)
    assert md["idempotency_key"] == idempotency_key(h.session.id, step, "tu_probe")
    assert ops_idempotency_key(h.session.id, step, "tu_probe").startswith(md["idempotency_key"])  # same formula as harness_ops
    assert md["tool_use_id"] == "tu_probe" and md["interactive"] is True and md["permission_mode"] == "accept-edits"
    assert md["rollout_id"] == "rollout_1"
    h.end_session()


def test_large_tool_output_is_offloaded_to_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    big = ws / "big.log"
    big.write_text("".join(f"line {i:05d} " + "x" * 30 + "\n" for i in range(1500)))
    h, _ = build(tmp_path, [tool_response(("read", {"path": str(big)})), text_response("done")])
    result = h.run("Read big.log")
    text = tool_results(result)[0].text
    assert text.startswith("<offloaded-tool-output path=")
    path = text.split('path="', 1)[1].split('"', 1)[0]
    assert path.startswith(str((ws / ".harness" / "tool-outputs").resolve())) and "line 01499" in open(path).read()
    h.end_session()


def test_default_model_client_is_the_anthropic_adapter_with_configured_model(tmp_path, monkeypatch):
    created = {}

    def fake_init(self, model, **kw):
        created.update(model=model, **kw)
        self.model = model

    monkeypatch.setattr(AnthropicModelClient, "__init__", fake_init)
    from harness import build_harness

    cfg = make_config(tmp_path, effort="low")
    h = build_harness(cfg)
    assert isinstance(h.model_client, AnthropicModelClient)
    assert created == {"model": "test-model", "effort": "low"}
    h.end_session()
