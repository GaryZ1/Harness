from __future__ import annotations

import threading

import pytest

from harness_core.types import Message, RunContext, TextBlock, ToolCall, ToolResultBlock, ToolSpec, ToolUseBlock
from harness_evals import CallableAgent, InfraError, Trajectory, parse_task
from harness_rl import (
    EnvironmentStateError,
    FunctionToolExecutor,
    ProfileRegistry,
    RewardConfigError,
    RolloutRunner,
    RolloutToolBackend,
    TaskEnvironment,
    TrajectoryExporter,
    load_jsonl,
)

TESTS = {"tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"}
TASK = parse_task({"id": "calc", "instruction": "Make add() correct.", "setup": {"files": {"calc.py": "def add(a, b):\n    return 0\n"}},
                   "hidden_files": TESTS, "limits": {"max_tool_calls": 3},
                   "graders": [{"kind": "unit_tests", "fail_to_pass": ["tests/test_calc.py::test_add"]}]})
PROFILE = ProfileRegistry.from_mapping({"profiles": [{"name": "p", "match": ["test-*"], "edit_tool": "write_file", "tools": ["write_file", "read_file"],
                                                      "tool_aliases": {"apply_patch": "write_file"}}]}).select("test-model").profile


def fs_tools(sandbox):
    ws = sandbox.workspace

    def write(inp, ctx):
        (ws / inp["path"]).parent.mkdir(parents=True, exist_ok=True)
        (ws / inp["path"]).write_text(inp["content"])
        return f"wrote {inp['path']}"

    def read(inp, ctx):
        return (ws / inp["path"]).read_text()

    return FunctionToolExecutor({"write_file": (ToolSpec("write_file", "write"), write), "read_file": (ToolSpec("read_file", "read"), read)})


FIX = {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}


def test_reset_step_finish_rewards_correct_fix(tmp_path):
    env = TaskEnvironment(TASK, tools_factory=fs_tools, profile=PROFILE, sandbox_root=tmp_path, model="test-model")
    with pytest.raises(EnvironmentStateError):
        env.step(ToolCall("x", "read_file", {"path": "calc.py"}))
    obs = env.reset(seed=1)
    assert [t.name for t in obs.tools] == ["write_file", "read_file"] and not (obs.workspace / "tests").exists()
    assert "return 0" in env.step(ToolCall("c1", "read_file", {"path": "calc.py"})).outcome.content
    env.step(ToolCall("c2", "write_file", FIX))
    ep = env.finish(final_text="fixed")
    assert ep.passed and ep.reward.total == 1.0 and ep.record.reward == 1.0
    rec = ep.record.validate()
    assert rec.rollout_id == obs.rollout_id and rec.profile == "p" and rec.profile_hash and rec.model == "test-model"
    assert all(b["type"] == "tool_result" for m in rec.messages[1:] if m["role"] == "user" for b in m["content"])
    env.close()


def test_tampering_with_tests_is_penalised(tmp_path):
    env = TaskEnvironment(TASK, tools_factory=fs_tools, sandbox_root=tmp_path)
    env.reset(seed=2)
    env.step(ToolCall("c1", "write_file", {"path": "tests/test_calc.py", "content": "def test_add():\n    pass\n"}))
    ep = env.finish()
    assert ep.reward.total == -1.0 and ep.reward.hack_detected and not ep.passed
    assert ep.record.hack_signals[0]["kind"] == "protected_created"


def test_backend_caps_calls_rejects_cross_rollout_and_mismatch(tmp_path):
    env = TaskEnvironment(TASK, tools_factory=fs_tools, profile=PROFILE, sandbox_root=tmp_path)
    env.reset(seed=3)
    mism = env.step(ToolCall("m", "apply_patch", {"patch": "..."}))
    assert mism.outcome.is_error and "Use 'write_file' instead" in mism.outcome.content
    for i in range(3):
        assert not env.step(ToolCall(f"c{i}", "read_file", {"path": "calc.py"})).outcome.is_error
    over = env.step(ToolCall("c9", "read_file", {"path": "calc.py"}))
    assert over.outcome.is_error and over.done and "budget exhausted" in over.outcome.content
    stats = env.backend.stats()
    assert stats["calls"] == 3 and stats["profile_mismatch"] == {"apply_patch": 1} and stats["rejected"]["budget"] == 1
    backend = RolloutToolBackend(fs_tools(env.sandbox), rollout_id="rollout_a")
    out = backend.execute(ToolCall("x", "read_file", {"path": "calc.py"}), RunContext("s", metadata={"rollout_id": "rollout_b"}))
    assert out.is_error and backend.rejected == {"cross_rollout": 1}
    env.close()


def test_env_refuses_hackable_task(tmp_path):
    judge_only = parse_task({"id": "j", "instruction": "x", "graders": [{"kind": "transcript", "max_turns": 3}]})
    with pytest.raises(RewardConfigError):
        TaskEnvironment(judge_only, sandbox_root=tmp_path)
    TaskEnvironment(judge_only, sandbox_root=tmp_path, require_rl_ready=False)


def agent_fn(task, ws, ctx):
    tools = ctx.metadata["tools"]
    seed = ctx.metadata["seed"]
    msgs = [Message.user(task.instruction)]
    content = FIX["content"] if seed % 2 == 0 else "def add(a, b):\n    return a - b\n"
    out = tools.execute(ToolCall("t1", "write_file", {"path": "calc.py", "content": content}), ctx)
    msgs += [Message.assistant([ToolUseBlock("t1", "write_file", {"path": "calc.py", "content": content})]),
             Message.user([ToolResultBlock("t1", out.content, out.is_error)]), Message.assistant([TextBlock("done")])]
    return Trajectory.from_messages(msgs, model="test-model", metadata={"rollout_details": {"token_ids": [1, 2, 3]}})


def test_rollout_runner_seeds_export_and_determinism(tmp_path):
    exp = TrajectoryExporter(tmp_path / "train.jsonl")
    runner = RolloutRunner(CallableAgent(agent_fn), tools_factory=fs_tools, profile=PROFILE, sandbox_root=tmp_path / "sb", exporter=exp, concurrency=3)
    eps = runner.run([TASK], n_per_task=6, seed=11)
    again = RolloutRunner(CallableAgent(agent_fn), tools_factory=fs_tools, sandbox_root=tmp_path / "sb2").run([TASK], n_per_task=6, seed=11)
    assert [e.seed for e in eps] == [e.seed for e in again] and len({e.seed for e in eps}) == 6
    assert [e.passed for e in eps] == [e.seed % 2 == 0 for e in eps]
    assert len({e.rollout_id for e in eps}) == 6
    recs = list(load_jsonl(tmp_path / "train.jsonl"))
    assert len(recs) == 6 and all(r.rollout_details == {"token_ids": [1, 2, 3]} for r in recs)
    assert {r.reward for r in recs} <= {0.0, 1.0}


def test_rollout_runner_infra_retry_and_timeout(tmp_path):
    seen = []

    def flaky(task, ws, ctx):
        seen.append(ctx.metadata["rollout_id"])
        if len(seen) == 1:
            raise InfraError("sandbox lost")
        return agent_fn(task, ws, ctx)

    ep = RolloutRunner(CallableAgent(flaky), tools_factory=fs_tools, sandbox_root=tmp_path, max_infra_retries=1).run([TASK], seed=0)[0]
    assert ep.status == "completed" and len(set(seen)) == 2  # retried in a fresh rollout

    def always_infra(task, ws, ctx):
        raise InfraError("down")

    bad = RolloutRunner(CallableAgent(always_infra), tools_factory=fs_tools, sandbox_root=tmp_path, max_infra_retries=1).run([TASK])[0]
    assert bad.status == "infra_error" and bad.record.metadata["exclude_from_training"] and bad.reward.total == 0.0

    stop = threading.Event()

    def slow(task, ws, ctx):
        ctx.metadata["cancel_event"].wait(5)
        stop.set()

    to = RolloutRunner(CallableAgent(slow), tools_factory=fs_tools, sandbox_root=tmp_path, timeout_s=0.2).run([TASK])[0]
    assert to.status == "timeout" and to.reward.total == 0.0 and to.record.metadata["truncated"] and stop.wait(2)
