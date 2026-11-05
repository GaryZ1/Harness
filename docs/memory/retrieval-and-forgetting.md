# Retrieval, usage telemetry and forgetting

## Read path: the quick memory pass

`MemorySystem.session_start_context()` builds the stable, cache-friendly context:

1. instruction files;
2. `READ_PROTOCOL`, which tells the agent to do a quick memory pass in at most 4–6 steps: skim the summary or index, then grep the index, then open one or two topic files, skills or rollout summaries, then search transcripts only for exact evidence. It also says memory is guidance and **the current environment wins**: verify cheap, drift-prone facts and label anything else "from memory, may be stale";
3. the citation instructions;
4. the mount description;
5. `memory_summary.md` (or the bounded `MEMORY.md`) plus the user index;
6. the skill catalog;
7. task orientation.

After compaction, `post_compaction_context()` re-injects all of this plus the working-memory recitation.

## Hybrid retrieval (`harness_memory.retrieval`)

Retrieval is an optional sidecar *behind* the file abstraction. Files remain the source of truth.

| Signal | Implementation |
|---|---|
| Lexical | Incremental Okapi BM25 (`k1=1.5`, `b=0.75`). The tokenizer keeps dotted and hyphenated identifiers *and* their parts. |
| Semantic | Optional `embedder(texts) -> vectors`, compared by cosine similarity. Without one, retrieval is BM25 + entity. |
| Entity | Overlap between query tokens and the `entities` frontmatter (or description keywords). |
| Recency | Applied only for queries with cues such as *latest/current/now/recent/still*, and only to documents that already matched something (time-aware query expansion). |
| Fusion | Reciprocal-rank fusion (`k=60`), normalised to `[0, 1]`. Defaults follow Mem0 v3: `top_k=20`, `threshold=0.1`, no rerank. |
| Time | `as_of` keeps items with `timestamp ≤ as_of < valid_to`. Superseded memories carry `valid_to`, so point-in-time questions work without destructive updates. |

`grep(lines, pattern)` is the agentic-search fallback.

## Citations and usage (`harness_memory.usage`)

Answers that used memory end with:

```text
<memory-citation>
feedback_ci.md:3-7|note=[CI order]
rollout_summaries/s1_fix-ci.md:12-30
<session_ids>s1, s2</session_ids>
</memory-citation>
```

* `Session.finish_reply(reply)` parses the block, appends rows to `.usage.jsonl`, logs the reply and returns it with the block stripped.
* `UsageLog.stats()` aggregates per path, plus `rollout:<sid>` for cited sessions.
* `UsageLog.apply_to_store()` runs during consolidation, not mid-session, so the cached prompt prefix never changes. It updates `use_count` and `last_used` while preserving `modified`.

## Forgetting (`harness_memory.forgetting`)

| Mechanism | Default | Behaviour |
|---|---|---|
| Phase-2 input selection | `max_unused_days=90`, `max_raw_memories=256` | Raw memories unused for 90 days drop out. Never-used ones age from `generated_at`. Ranked by `usage_count`, then recency. |
| Topic-file expiry | `expire_after_days=180` | Active root-level memories untouched for 180 days are **archived** (moved to `archive/` and removed from the index; git keeps history), not hard-deleted. |
| Explicit deletion | n/a | `SemanticMemory.forget(path)`; `MemorySystem.forget_session(sid)` removes the raw memory and rollout summary; the next consolidation runs `forget_derived`. |
| `forget_derived(store, index, sessions)` | n/a | Deletes memories supported **only** by those sessions and trims them from shared provenance ("delete only memory supported by deleted inputs"). |

Deleting a transcript does **not** silently delete derived memories. Use the explicit cascade above.
