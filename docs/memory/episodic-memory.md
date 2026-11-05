# Episodic memory: `harness_memory.episodic`

**Function:** what happened, when, and in which session.
**Loading policy:** *evidence only*. It is never auto-loaded. The quick memory pass opens a rollout summary when the index points at one, and searches raw transcripts only for exact evidence.

## Tier 1: raw transcripts (`TranscriptLog`)

`sessions/<session_id>.jsonl`, one event per line:

```json
{"seq": 3, "ts": "2026-11-03T10:00:00Z", "kind": "tool_result", "payload": {"tool": "bash", "content": "..."},
 "prev_hash": "<sha256 of event 2>", "hash": "<sha256 of this event>"}
```

* **Append-only.** Writes use `O_APPEND` plus `fsync` under a cross-process lock. Nothing rewrites history ("Raw rollouts are immutable evidence").
* **Tamper-evident.** Each event hashes `(seq, ts, kind, payload, prev_hash)`, and `verify()` recomputes the chain.
* **Redacted.** `SessionRegistry.transcript()` attaches the secret redactor, so credentials never reach disk.
* `search(regex, kinds=…)` is the exact-evidence step of the memory pass. `last_activity` drives idle gating.

## Session flags (`SessionRegistry`, `SessionMeta`)

`sessions/<id>.meta.json` records `ephemeral`, `subagent`, `memory_enabled`, `memory_mode` (`clean` | `polluted`), `pollution_reasons`, `parent_session`, `user_id`, `project_id`, `started` and `ended`.

`generation_block_reason()` returns why memory must **not** be generated from a session:

| Flag | Reason |
|---|---|
| `subagent=True` | sub-agent output duplicates and pollutes parent memory (Codex never runs the pipeline for sub-agents) |
| `ephemeral=True` | one-off / incognito sessions |
| `memory_enabled=False` | user or org opt-out |
| `memory_mode="polluted"` | the session ingested untrusted content: web, email, third-party MCP (Codex `disable_on_external_context`) |

```python
reg = SessionRegistry("~/.harness")
reg.start("s_1", user_id="u_42", project_id=repo_id)
log = reg.transcript("s_1")
log.append("user", "Deploy to staging")
reg.apply_taint("s_1", tracker)      # persist TaintTracker result
reg.end("s_1")
```

## Tier 2: rollout summaries (`RolloutSummaryStore`)

Phase-1 extraction writes one distilled page per session to `rollout_summaries/<session_id>_<slug>.md` in the learned-memory store. The page has frontmatter `session_id`, `slug`, `outcome` (`success | partial | uncertain | fail`), `created` and `tainted`. Consolidation cites these files from `MEMORY.md` and topic-file provenance (`source.evidence`). `delete_for_session(sid)` supports "forget this session", and the next consolidation cascades the deletion (see [retrieval-and-forgetting.md](retrieval-and-forgetting.md)).

## Why two tiers

Raw transcripts are complete but expensive to read. Summaries are cheap but lossy. Keeping both, plus provenance pointers from semantic and procedural memory into the summaries, lets every learned memory be traced back to evidence. Graphiti's "episodes" and Codex's `rollout_summaries/` follow the same principle.
