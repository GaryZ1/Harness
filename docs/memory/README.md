# Memory subsystem (`harness_memory`)

Production Python implementation of the memory design in
[`reports/SOTA agentic harness reference.md` §4](../../reports/SOTA%20agentic%20harness%20reference.md) and
[`research_notes/.../memory_systems.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/memory_systems.md).
Each memory type has its own module and its own page:

| Memory type | Function (what it holds) | Loading policy | Module | Doc |
|---|---|---|---|---|
| **Instruction memory** | Human-written rules (`AGENTS.md` / `CLAUDE.md`) | Always loaded | `instructions.py` | [instruction-memory.md](instruction-memory.md) |
| **Working memory** | Current goal, plan/todos, scratch notes, context budget | In context; recited at the end; mirrored to disk | `working.py` | [working-memory.md](working-memory.md) |
| **Semantic memory** | Facts, preferences, corrections, decisions, references | Index always loaded; topic files on demand | `semantic.py`, `index.py` | [semantic-memory.md](semantic-memory.md) |
| **Episodic memory** | What happened: transcripts, per-session summaries | Evidence only, searched on demand | `episodic.py` | [episodic-memory.md](episodic-memory.md) |
| **Procedural memory** | Skills, workflows, lessons (playbook) | Skill catalog always; bodies on demand | `procedural.py` | [procedural-memory.md](procedural-memory.md) |
| **Task state** | Per-task feature list, progress log, `init.sh`, git | Re-read at session start and after compaction | `task_state.py` | [task-state.md](task-state.md) |

Cross-cutting pages:

* [write-pipeline.md](write-pipeline.md) covers the inline `memory_20250818` tool (channel A), Phase-1 extraction and Phase-2 consolidation (channel B).
* [retrieval-and-forgetting.md](retrieval-and-forgetting.md) covers the quick memory pass, hybrid retrieval, citations and usage telemetry, decay and deletion.
* [security-and-operations.md](security-and-operations.md) covers the path sandbox, secret redaction, taint tracking, scopes and mounts, concurrency, versioning, configuration defaults and evaluation.

## Why this shape

The research converges on a few points. This package encodes each one in code, not in prompts.

1. **File-first, model-curated memory with progressive disclosure.** A small always-loaded index or summary, plus on-demand files read with ordinary tools. Claude Code, Codex, the Agents SDK and Letta Code all converged on this, and a plain filesystem agent scored 74.0% on LoCoMo against Mem0g's 68.5%.
2. **Two write channels.** An inline, sandboxed, size-capped tool handles explicit "remember X". An asynchronous, idle-gated, two-phase background pipeline handles everything else.
3. **ADD-only, supersede-not-delete.** Mem0 v3 dropped LLM-decided UPDATE/DELETE, and Graphiti invalidates old facts instead of deleting them. "Current truth" is resolved at read time.
4. **Itemised delta updates.** Consolidation uses delta edits, never whole-file rewrites, which avoids ACE's "brevity bias" and "context collapse".
5. **Verified procedural memory only.** Curated skills added +16 pp on SkillsBench; self-generated skills added about 0. Generated skills start as candidates behind a promotion gate.
6. **Memory is an attack surface.** The defences are read-only shared mounts, taint tracking (`memory_mode=polluted`), secret redaction, path sandboxing and versioned rollback.
7. **Four persistence layers stay separate:** instructions, learned memory, task state and evidence. Mixing them causes bloat and cross-task leakage.

## Architecture

```text
                    ┌─────────────────────────── session (MemorySystem.start_session) ───────────────────────────┐
 always loaded ───▶ │ instructions (AGENTS.md…)  memory_summary.md / MEMORY.md  skill catalog  task orientation    │
                    │ READ_PROTOCOL + citation instructions                                                          │
 in context ──────▶ │ WorkingMemory.recitation()  (appended at the end, never in the cached prefix)                 │
                    │                                                                                               │
 tools ───────────▶ │ memory tool (view/create/str_replace/insert/delete/rename)  ── channel A (inline writes)     │
                    │ other tools ──▶ TaintTracker (untrusted output ⇒ memory_mode=polluted, writes tainted)        │
 evidence ────────▶ │ TranscriptLog sessions/<id>.jsonl (append-only, hash-chained, redacted)                       │
                    └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                         │ end_session / idle ≥ 12 h
                                                         ▼
          Phase 1 Extractor (leases, strict JSON, no-op preferred, skips sub-agent/ephemeral/polluted)
               ├─▶ raw_memories/<id>.json            └─▶ rollout_summaries/<id>_<slug>.md   (episodic)
                                                         ▼
          Phase 2 Consolidator (global lock, single writer, delta ops only, quarantines tainted inputs)
               ├─▶ semantic topic files + MEMORY.md (remember / supersede / mark_stale)
               ├─▶ playbook.md bullets (add / tag / prune)              ─┐ procedural
               ├─▶ skills/<name>/SKILL.md candidates (≤ 1 per run)      ─┘
               ├─▶ usage (.usage.jsonl ⇒ use_count/last_used) ⇒ forgetting (archive / forget_derived)
               └─▶ memory_summary.md (v1) ⇒ git commit "feat|fix|chore(memory): …" ⇒ baseline ref
```

## On-disk layout

```text
~/.harness/
  org/                                  # READ-ONLY mount (org standards, shared skills)
  users/<user_id>/memory/               # git repo, read_write
  projects/<repo_id>/memory/            # git repo, read_write; repo_id shared across worktrees
    MEMORY.md                           # index: one line per memory; 200 lines / 25 KB cap
    memory_summary.md                   # "v1" + 4 sections, injected at session start
    <type>_<slug>.md                    # semantic / lesson / workflow topic files (+ --vN versions)
    playbook.md                         # ACE-style bullets with helpful/harmful counters
    skills/<name>/SKILL.md              # procedural memory
    rollout_summaries/<sid>_<slug>.md   # episodic summaries
    raw_memories/<sid>.json             # Phase-1 outputs
    inbox/<timestamp>-<slug>.md         # user-requested notes (restricted mode)
    archive/                            # expired memories (still in git history)
    .usage.jsonl                        # citation telemetry
  sessions/<sid>.jsonl + .meta.json     # immutable transcripts + session flags
  state/memory_jobs.sqlite              # Phase-1 job leases

<repo>/.harness/task/                   # task state lives WITH the work, under git
  feature_list.json  progress.md  init.sh  NOTES.md
```

## Quick start

```bash
pip install -e ".[anthropic,dev]"
pytest
```

```python
from harness_memory import AnthropicLLM, MemorySystem

ms = MemorySystem(base="~/.harness", user_id="u_42", repo_root=".", context_window_tokens=200_000)
session = ms.start_session("s_2026_11_05_001")
system_prompt = BASE_PROMPT + "\n\n" + ms.session_start_context(session)

# In your agent loop:
#   memory tool calls  -> session.memory_tool.handle(tool_use.input).to_block(tool_use.id)
#   other tool results -> session.record_tool_output(name, text, untrusted=is_open_world)
#   final answers      -> shown = session.finish_reply(model_text)
#   after compaction   -> re-inject ms.post_compaction_context(session)

ms.end_session(session, progress_summary="Implemented login; e2e green")

# On idle, from a background worker:
llm = AnthropicLLM(model=os.environ["HARNESS_MEMORY_MODEL"], effort="high")
extraction_results, consolidation_report = ms.run_background(llm)
```

A complete runnable walkthrough that needs no API key is in [`examples/quickstart.py`](../../examples/quickstart.py).
