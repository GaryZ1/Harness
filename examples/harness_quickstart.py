"""End-to-end harness quickstart. Runs offline by default (a scripted model; no API key needed).

    python examples/harness_quickstart.py                 # offline, ScriptedModelClient with "test-model"
    HARNESS_MODEL=<model id> python examples/harness_quickstart.py --real "Summarise README.md"

The offline run builds every package into one harness (kernel tools in a temp workspace, bash through
the sandbox runner, the permission engine with an approver, memory, context assembly, telemetry), runs
one scripted task and prints what happened. ``--real`` uses the same configuration with the Anthropic
adapter (``pip install anthropic``; the model id comes only from ``HARNESS_MODEL``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import HarnessConfig, build_harness  # noqa: E402
from harness_core.testing import ScriptedModelClient, text_response, tool_response  # noqa: E402

CONFIG_YAML = """
model: ${HARNESS_MODEL:-test-model}     # required; never defaulted in code (the offline run supplies test-model)
effort: ${HARNESS_EFFORT:-}
workspace: ws
base_dir: home
context_window_tokens: 200000
permissions:
  interactive: true                     # ASK decisions go to the approver below
  policy_files: [harness-policy.yaml]
features:
  memory: true
  memory_git: false
budgets:
  max_turns: 20
"""

POLICY_YAML = """
rules:
  deny: ["Bash(rm -rf *)"]
  ask: ["Bash(python3 *)"]
"""


def approver(call, spec, ctx, decision) -> bool:
    print(f"  [approval] {call.name} {call.input} -> approved ({decision.reason})")
    return True


def offline_script(ws: Path) -> ScriptedModelClient:
    app = ws / "app.py"
    return ScriptedModelClient([
        tool_response(("read", {"path": str(app)})),
        tool_response(("edit", {"path": str(app), "old_string": "return a - b", "new_string": "return a + b"})),
        tool_response(("bash", {"command": "python3 -c 'import app; assert app.add(2, 3) == 5; print(\"ok\")'"})),
        tool_response(("bash", {"command": "rm -rf build"})),
        tool_response(("memory", {"command": "create", "path": "/memories/feedback_tests.md",
                                  "file_text": "Verify fixes with a one-line python3 assertion before finishing.\n"})),
        text_response("Fixed add() in app.py and verified it (prints ok).\n"
                      "<memory-citation>\nfeedback_tests.md:1-1|note=[verification habit]\n</memory-citation>"),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="use the Anthropic adapter (model id from HARNESS_MODEL)")
    parser.add_argument("prompt", nargs="?", default="add() in app.py is wrong; fix it and verify.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    root = Path(tempfile.mkdtemp(prefix="harness-quickstart-"))
    ws = root / "ws"
    ws.mkdir()
    (ws / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (ws / "AGENTS.md").write_text("# Project rules\nKeep changes minimal and verify them.\n")
    (root / "harness-policy.yaml").write_text(POLICY_YAML)
    (root / "harness.yaml").write_text(CONFIG_YAML)

    if args.real and not os.environ.get("HARNESS_MODEL"):
        print("--real needs HARNESS_MODEL=<model id> (the harness never defaults a model id)")
        return 2
    cfg = HarnessConfig.from_yaml(root / "harness.yaml")
    client = None if args.real else offline_script(ws)   # None -> AnthropicModelClient(cfg.model, effort=cfg.effort)
    print(f"workspace: {cfg.workspace}\nmodel: {cfg.model} ({'real' if args.real else 'scripted, offline'})")

    with build_harness(cfg, model_client=client, approver=approver) as h:
        print(f"sandbox: {h.runner.choice.backend} (auto-allow in boundary: {h.runner.choice.auto_allow_in_boundary})")
        result = h.run(args.prompt)
        print(f"\nresult: {result.subtype.value} after {result.num_turns} model calls")
        for m in result.messages:
            for b in m.tool_uses:
                print(f"  tool_use    {b.name} {b.input}")
            for r in m.tool_results:
                print(f"  tool_result {'ERROR ' if r.is_error else ''}{r.text.splitlines()[0][:110] if r.text else ''}")
        print(f"\nfinal reply (citations stripped): {result.final_text}")
        if not args.real:
            assert (ws / "app.py").read_text() == "def add(a, b):\n    return a + b\n", "edit not applied"
            assert any(r.is_error and "deny rule" in r.text for m in result.messages for r in m.tool_results), "deny not applied"
        if h.memory_session is not None:
            print(f"memory citations recorded: {len(h.memory_session.usage.rows())}")
        dash = h.metrics()
        print(f"metrics: stop_reasons={dash['stop_reasons']} tool_error_rate={dash['tool_error_rate']} "
              f"permission_rates={dash['permission_rates']}")
        print(f"telemetry: {sorted(p.name for p in h.telemetry_dir.iterdir() if not p.name.startswith('.'))} in {h.telemetry_dir}")
        print(f"session log: {h.store.root / (h.session.id + '.jsonl') if h.store else '(in memory)'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
