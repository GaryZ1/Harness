# Harness

This repository holds research and a production implementation for a state-of-the-art agentic harness. It has ten packages built against one shared contract (`harness_core/types.py`) and a top-level `harness` package that wires them into one working harness.

Start with the [harness overview](docs/README.md), which covers the architecture, the turn lifecycle and the configuration reference. Then read [docs/integration.md](docs/integration.md) for how the packages are joined.

| Package | Contents | Docs |
|---|---|---|
| [`src/harness/`](src/harness/) | **Integration**: `HarnessConfig` (YAML), `build_harness`, `Harness`, `LoopAgentRunner`, `LoopAgentUnderTest` | [docs/README.md](docs/README.md), [docs/integration.md](docs/integration.md) |
| [`src/harness_core/`](src/harness_core/) | Shared types, `AgentLoop`, durable session log, hooks, budgets, retries, Anthropic adapter, test doubles | [docs/core/](docs/core/README.md) |
| [`src/harness_context/`](src/harness_context/) | `LayeredContextAssembler`, prompt layers, caching, clearing, compaction, re-injection, offloading | [docs/context/](docs/context/README.md) |
| [`src/harness_memory/`](src/harness_memory/) | Instruction, working, semantic, episodic and procedural memory; task state; the `memory_20250818` tool | [docs/memory/](docs/memory/README.md) |
| [`src/harness_tools/`](src/harness_tools/) | `ToolRegistry`, kernel tools, tool catalog and search, MCP client | [docs/tools/](docs/tools/README.md) |
| [`src/harness_sandbox/`](src/harness_sandbox/) | Sandbox policies, OS runners, egress proxy, snapshots | [docs/sandbox/](docs/sandbox/README.md) |
| [`src/harness_permissions/`](src/harness_permissions/) | `PolicyEngine`, shell analysis, taint and injection defence, `ApprovalBroker`, policy files | [docs/permissions/](docs/permissions/README.md) |
| [`src/harness_multiagent/`](src/harness_multiagent/) | `SubagentManager`, `AgentRunner` protocol, leases, worktrees, workflows | [docs/multiagent/](docs/multiagent/README.md) |
| [`src/harness_evals/`](src/harness_evals/) | `EvalRunner`, task format, isolation, graders, metrics, reports | [docs/evals/](docs/evals/README.md) |
| [`src/harness_rl/`](src/harness_rl/) | RL environments, rewards, trajectory export, the self-improvement loop | [docs/rl/](docs/rl/README.md) |
| [`src/harness_ops/`](src/harness_ops/) | Tracing, metrics, pricing, durable steps, idempotency, deploys | [docs/ops/](docs/ops/README.md) |

Other paths:

| Path | Contents |
|---|---|
| [`reports/`](reports/) | The harness build plan and the full SOTA reference: architecture, loop, context, memory, tools, sandboxing, permissions, multi-agent, evals, observability, RL. |
| [`research_notes/`](research_notes/) | The source-level research notes behind the reports. |
| [`tests/`](tests/) | The pytest suite, one directory per package plus `tests/integration/`. It runs offline and needs no API key. |
| [`examples/harness_quickstart.py`](examples/harness_quickstart.py) | The whole harness end to end. It runs offline with a scripted model; pass `--real` to use `HARNESS_MODEL`. |
| [`examples/quickstart.py`](examples/quickstart.py) | A walkthrough of every memory type. |

## Quick start

```bash
pip install -e ".[dev]"                  # add ,anthropic for real model calls; ,otel for OTLP export
pytest
python examples/harness_quickstart.py    # offline
HARNESS_MODEL=<model id> python examples/harness_quickstart.py --real "Summarise README.md"
```

```python
from harness import HarnessConfig, build_harness

cfg = HarnessConfig.from_yaml("harness.yaml")      # model: ${HARNESS_MODEL} (required, never defaulted)
with build_harness(cfg) as h:
    result = h.run("Fix the failing test in tests/test_parse.py")
    print(result.subtype.value, result.final_text, h.metrics()["cache_hit_rate"])
```

Requirements: Python ≥ 3.10, PyYAML, and `git` on `PATH`. The optional SDKs are imported lazily: `anthropic` for `AnthropicModelClient` and `AnthropicLLM`, and `opentelemetry-sdk` with the OTLP exporter for `harness_ops.otlp_exporter`.
