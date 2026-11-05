"""End-to-end walkthrough of every memory type. Runs offline (a scripted fake model).

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_memory import CallableLLM, MemorySystem  # noqa: E402
from harness_memory._time import utcnow  # noqa: E402


class Clock:
    def __init__(self):
        self.now = utcnow()

    def __call__(self):
        return self.now


def fake_model(system: str, prompt: str) -> str:
    """Stands in for AnthropicLLM: Phase-1 extraction, then Phase-2 consolidation."""
    if "Session id:" in prompt:
        return json.dumps({
            "rollout_summary": "User asked to switch the project to pnpm; CI fast target must run first.",
            "rollout_slug": "pnpm-and-ci",
            "raw_memory": "- User prefers pnpm over npm (explicit correction)\n- CI: run `make test-fast` before the full suite",
            "outcome": "success",
        })
    return json.dumps({
        "ops": [
            {"op": "remember", "type": "user", "slug": "pnpm", "description": "Prefers pnpm over npm",
             "body": "Use pnpm for every install/run command.", "sessions": ["s_demo_1"], "epistemic": "verified"},
            {"op": "playbook", "deltas": [{"op": "add", "section": "Strategies", "text": "Run make test-fast before the full suite"}]},
        ],
        "summary": {"User preferences": "- pnpm over npm"},
        "commit": {"type": "feat", "subject": "record pnpm preference and CI ordering"},
    })


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="harness-memory-demo-"))
    repo = base / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# Project rules\n- Two-space indentation.\n")
    clock = Clock()

    ms = MemorySystem(base=base / "harness", user_id="demo-user", repo_root=repo, clock=clock, context_window_tokens=200_000)

    # Task state (initializer session).
    ms.task.initialize("Ship the demo", [{"id": "f1", "category": "core", "description": "Home page renders"}], "echo smoke ok")

    # Session 1: inline memory tool + working memory + episodic transcript.
    s = ms.start_session("s_demo_1")
    s.working.goal = "Switch tooling to pnpm"
    s.working.set_plan(["Replace npm scripts", "Run fast tests"])
    s.record("user", "Use pnpm, not npm. And CI must run make test-fast first.")
    print(s.memory_tool.handle({"command": "create", "path": "/memories/feedback_ci-order.md",
                                "file_text": "---\ntype: feedback\ndescription: CI runs make test-fast first\n---\nRun `make test-fast` before the full suite.\n"}).content)
    print("\n--- session-start context (excerpt) ---\n" + ms.session_start_context(s)[:900] + "\n...")
    shown = s.finish_reply("Switched to pnpm.\n<memory-citation>\nfeedback_ci-order.md:1-6|note=[CI order]\n</memory-citation>")
    print("\nreply shown to user:", shown)
    ms.end_session(s, progress_summary="Switched scripts to pnpm")

    # Background pipeline after the idle window.
    clock.now += timedelta(hours=13)
    results, report = ms.run_background(CallableLLM(fake_model))
    print("\nphase 1:", [(r.session_id, r.outcome.value) for r in results])
    print("phase 2 applied:", report.applied, "commit:", report.commit[:8] if report.commit else None)

    # Retrieval and procedural memory.
    print("\nsearch 'package manager pnpm':", [h.doc_id for h in ms.search("package manager pnpm")])
    print("playbook top:", [b.render() for b in ms.playbook.top(3)])
    print("memory repo log:", ms.project_store.versioner.log())
    print(f"\nEverything lives under {base}")


if __name__ == "__main__":
    main()
