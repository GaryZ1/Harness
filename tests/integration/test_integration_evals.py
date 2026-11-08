"""harness_evals / harness_rl over the real loop: LoopAgentUnderTest builds a fresh harness per trial."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from integration_support import build, text_response, tool_response

from harness import HarnessConfig, LoopAgentUnderTest, trajectory_from_session
from harness.config import FeaturesConfig, PermissionsConfig
from harness_core.testing import ScriptedModelClient
from harness_core.types import RunContext, ToolSpec
from harness_evals import AgentTask, EvalRunner, Limits, ResultStore, RunConditions, RunConfig, Suite, parse_task
from harness_rl import FunctionToolExecutor, RolloutRunner, TrajectoryExporter, load_jsonl

logging.getLogger("harness_sandbox").setLevel(logging.ERROR)

BASE = HarnessConfig(model="test-model", features=FeaturesConfig(memory_git=False), permissions=PermissionsConfig(mode="accept-edits"))


def _write_task(content: str):
    return parse_task({
        "id": "write-greeting", "instruction": "Create hello.txt containing exactly 'hello world'.", "suite": "capability",
        "graders": [{"kind": "file_equals", "path": "hello.txt", "content": "hello world"}], "trials": 2,
        "limits": {"max_turns": 5, "max_tool_calls": 4, "timeout_s": 60},
    }), content


def _runner(tmp_path: Path, agent) -> EvalRunner:
    return EvalRunner(agent, store=ResultStore(tmp_path / "evals"), conditions=RunConditions(harness_version="harness@0.1.0", model="test-model"),
                      config=RunConfig(concurrency=2, workspace_root=tmp_path / "trials", keep_workspaces="always"))


def _agent(content: str, seen: list) -> LoopAgentUnderTest:
    def factory(task, ctx):
        seen.append(dict(ctx.metadata))
        ws = Path(ctx.cwd)
        return ScriptedModelClient([tool_response(("write", {"path": str(ws / "hello.txt"), "content": content})),
                                    text_response("Created hello.txt")])
    return LoopAgentUnderTest(BASE, model_client_factory=factory)


def test_eval_trial_passes_end_state_grader(tmp_path):
    task, content = _write_task("hello world\n")
    seen: list = []
    result = _runner(tmp_path, _agent(content, seen)).run(Suite("integration", "1", (task,)), run_id="pass")
    recs = result.records
    assert [r["status"] for r in recs] == ["completed", "completed"] and all(r["passed"] for r in recs)
    assert all(r["tool_calls"] == 1 and r["turns"] == 2 for r in recs)
    assert {r["grades"][0]["kind"] for r in recs} == {"file_equals"}
    assert result.summary().tiers["capability"].pass1 == 1.0
    # each trial got its own memory root and seed; memory lived only under ctx.metadata["memory_dir"]
    assert len({md["memory_dir"] for md in seen}) == 2 and len({md["seed"] for md in seen}) == 2
    assert all((Path(md["memory_dir"]) / "sessions").is_dir() for md in seen)


def test_eval_trial_fails_when_the_end_state_is_wrong(tmp_path):
    task, content = _write_task("goodbye\n")
    result = _runner(tmp_path, _agent(content, [])).run(Suite("integration", "1", (task,)), run_id="fail")
    assert all(r["status"] == "completed" and not r["passed"] for r in result.records)


def test_cancel_event_interrupts_the_loop(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    cancel = threading.Event()

    def slow_then_cancel(request):
        cancel.set()  # the eval runner's watchdog fires while the model is "thinking"
        import time

        time.sleep(0.3)
        return tool_response(("write", {"path": str(ws / "late.txt"), "content": "x"}))

    client = ScriptedModelClient([slow_then_cancel, text_response("never")])
    agent = LoopAgentUnderTest(replace_state(BASE, tmp_path), model_client=client)
    ctx = RunContext(session_id="trial.cancel", cwd=str(ws), metadata={"cancel_event": cancel, "seed": 1})
    traj = agent.run(AgentTask("t", "do it", Limits(timeout_s=30)), ws, ctx)
    assert traj.metadata["result_subtype"] == "interrupted"
    assert not (ws / "late.txt").exists() and client.calls == 1


def replace_state(cfg: HarnessConfig, tmp_path: Path) -> HarnessConfig:
    from dataclasses import replace

    return replace(cfg, state_dir=str(tmp_path / "state"), base_dir=str(tmp_path / "base"))


def test_rl_rollout_routes_every_call_through_the_backend(tmp_path):
    tests = {"tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"}
    task = parse_task({"id": "calc", "instruction": "Make add() correct.", "setup": {"files": {"calc.py": "def add(a, b):\n    return 0\n"}},
                       "hidden_files": tests, "limits": {"max_tool_calls": 3},
                       "graders": [{"kind": "unit_tests", "fail_to_pass": ["tests/test_calc.py::test_add"]}]})

    def fs_tools(sandbox):
        ws = sandbox.workspace

        def write(inp, ctx):
            (ws / inp["path"]).write_text(inp["content"])
            return f"wrote {inp['path']}"

        schema = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
        return FunctionToolExecutor({"write_file": (ToolSpec("write_file", "Write a file.", schema), write)})

    def factory(t, ctx):
        assert ctx.metadata["tools"] is not None
        return ScriptedModelClient([tool_response(("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})),
                                    text_response("fixed")])

    exporter = TrajectoryExporter(tmp_path / "train.jsonl")
    eps = RolloutRunner(LoopAgentUnderTest(BASE, model_client_factory=factory), tools_factory=fs_tools, sandbox_root=tmp_path / "sb",
                        exporter=exporter, concurrency=1).run([task], n_per_task=1, seed=3)
    ep = eps[0]
    assert ep.status == "completed" and ep.passed and ep.reward.total == 1.0
    assert ep.record.metadata["tool_stats"]["calls"] == 1  # the rollout-scoped backend executed the call
    recs = list(load_jsonl(tmp_path / "train.jsonl"))
    assert len(recs) == 1 and recs[0].messages[1]["content"][0]["name"] == "write_file"


def test_trajectory_from_session_uses_documented_payloads(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("alpha\n")
    h, _ = build(tmp_path, [tool_response(("read", {"path": str(ws / "a.txt")})), text_response("alpha it is")])
    result = h.run("What is in a.txt?")
    traj = trajectory_from_session(h.session, result, system=h.loop.system, model="test-model")
    assert [m.role.value for m in traj.messages] == ["user", "assistant", "user", "assistant"]  # no duplicated tool results
    assert traj.turns == 2 and traj.n_tool_calls == 1 and traj.tool_calls[0].name == "read" and "alpha" in traj.tool_calls[0].output
    assert traj.final_text == "alpha it is" and traj.usage.input_tokens == 200 and traj.stop_reason == "end_turn"
    assert traj.model == "test-model" and traj.system.startswith("You are") and traj.compaction_events == 0
    assert traj.metadata["result_subtype"] == "success"
    h.end_session()
