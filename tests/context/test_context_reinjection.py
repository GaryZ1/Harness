from harness_context import FileActivityTracker, RecentFilesSource, ReinjectionBuilder, SkillBodiesSource, TextSource
from harness_core.types import Message, RunContext, ToolCall, ToolUseBlock

CTX = RunContext("s")


def test_tracker_orders_by_recency_and_ignores_non_file_tools():
    tr = FileActivityTracker(capacity=3)
    tr.observe_call(ToolCall("1", "read_file", {"path": "a.py"}))
    tr.observe_call(ToolCall("2", "bash", {"path": "ignored.py"}))
    tr.observe_messages([Message.assistant([ToolUseBlock("3", "edit_file", {"file_path": "b.py"}), ToolUseBlock("4", "Read", {"path": "a.py"})])])
    tr.observe_call(ToolCall("5", "write_file", {"path": "c.py"}))
    tr.observe_call(ToolCall("6", "write_file", {"path": "d.py"}))
    assert tr.recent() == ["d.py", "c.py", "a.py"]  # capacity 3 evicts b.py


def test_recent_files_reread_small_and_reference_large(tmp_path):
    (tmp_path / "small.py").write_text("print('hi')\n")
    (tmp_path / "big.py").write_text("x = 1\n" * 10_000)
    tr = FileActivityTracker()
    for p in ("small.py", "big.py", "missing.py", "../outside.py"):
        tr.touch(p)
    text = RecentFilesSource(tr, tmp_path, max_files=5).render(CTX)
    assert '<file path="small.py">\nprint(\'hi\')\n</file>' in text
    assert '<referenced-file path="big.py" tokens="~' in text and "x = 1" not in text
    assert '<referenced-file path="missing.py" status="unavailable"/>' in text
    assert '<referenced-file path="../outside.py" status="unavailable"/>' in text  # sandboxed


def test_skill_bodies_capped_oldest_dropped_truncated_from_end():
    skills = [("old", "o" * 40_000), ("mid", "m" * 30_000), ("new", "n" * 100)]
    src = SkillBodiesSource(lambda: skills, per_skill_tokens=5_000, total_tokens=5_020)
    text = src.render(CTX)
    assert '<skill name="new">' in text and '<skill name="mid">' not in text and '<skill name="old">' not in text
    roomy = SkillBodiesSource(lambda: skills, per_skill_tokens=5_000, total_tokens=25_000).render(CTX)
    assert roomy.index('name="old"') < roomy.index('name="mid"') < roomy.index('name="new"')  # invocation order
    assert "skill body truncated" in roomy and len(roomy) < 45_000


def test_builder_order_failures_and_background_jobs():
    def boom():
        raise RuntimeError("disk gone")

    b = ReinjectionBuilder(
        [TextSource("instructions", "Use pytest"), TextSource("git", boom), TextSource("empty", lambda: None)],
        background_jobs=lambda: ["npm run dev (pid 42)"],
    )
    bundle = b.build(CTX)
    assert bundle.sections == ["instructions", "git", "background-jobs"]
    assert "could not re-load git: RuntimeError" in bundle.text and "do not start them again" in bundle.text
    assert ReinjectionBuilder([]).build(CTX).text == ""
    ctx_src = TextSource("who", lambda ctx: ctx.agent_name, pass_context=True)
    assert ctx_src.render(CTX) == "main"
