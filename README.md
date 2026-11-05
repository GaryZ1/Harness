# Harness

Research and a production implementation for a state-of-the-art agentic harness.

| Path | Contents |
|---|---|
| [`reports/`](reports/) | The harness build plan and the full SOTA reference (architecture, loop, context, memory, tools, sandboxing, permissions, multi-agent, evals, observability, RL). |
| [`research_notes/`](research_notes/) | Source-level research notes behind the reports. |
| [`src/harness_memory/`](src/harness_memory/) | **`harness-memory`**: the production Python memory subsystem, with one module per memory type. |
| [`docs/memory/`](docs/memory/) | Documentation for every memory type and the write, read and security paths. |
| [`tests/`](tests/) | pytest suite (offline; no API key needed). |
| [`examples/quickstart.py`](examples/quickstart.py) | Runnable end-to-end walkthrough. |

## harness-memory at a glance

| Memory type | Module | Docs |
|---|---|---|
| Instruction memory (`AGENTS.md` / `CLAUDE.md`) | `instructions.py` | [instruction-memory.md](docs/memory/instruction-memory.md) |
| Working memory (plan, notes, context budget) | `working.py` | [working-memory.md](docs/memory/working-memory.md) |
| Semantic memory (facts, preferences, decisions) | `semantic.py`, `index.py` | [semantic-memory.md](docs/memory/semantic-memory.md) |
| Episodic memory (transcripts, rollout summaries) | `episodic.py` | [episodic-memory.md](docs/memory/episodic-memory.md) |
| Procedural memory (skills, playbook) | `procedural.py` | [procedural-memory.md](docs/memory/procedural-memory.md) |
| Long-running task state | `task_state.py` | [task-state.md](docs/memory/task-state.md) |
| Inline tool + background pipeline | `tool.py`, `pipeline/` | [write-pipeline.md](docs/memory/write-pipeline.md) |
| Retrieval, citations, forgetting | `retrieval.py`, `usage.py`, `forgetting.py` | [retrieval-and-forgetting.md](docs/memory/retrieval-and-forgetting.md) |
| Sandbox, redaction, taint, scopes, ops | `security.py`, `store.py`, `scopes.py` | [security-and-operations.md](docs/memory/security-and-operations.md) |

Start with the [memory overview](docs/memory/README.md).

```bash
pip install -e ".[dev]"          # add ,anthropic for the Anthropic adapter
pytest
python examples/quickstart.py
```

Requires Python ≥ 3.10, PyYAML and `git` on `PATH`. The optional `anthropic` extra enables `AnthropicLLM` and `run_memory_agent`.
