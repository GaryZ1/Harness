# Semantic memory: `harness_memory.semantic` + `harness_memory.index`

**Function:** durable, non-derivable, behaviour-changing facts.

| `type` | Holds | Example |
|---|---|---|
| `user` | role, expertise, preferences | "Prefers pnpm over npm" |
| `feedback` | corrections, confirmed approaches | "CI runs `make test-fast` first; full suite only on main" |
| `project` | decisions and deadlines **not derivable from code or git** | "Release freeze moved to 2026-11-27" |
| `reference` | where to find external info | "Latency dashboards: Grafana → SRE/latency" |
| `lesson` / `workflow` | experience-derived (see [procedural](procedural-memory.md)); **provenance required** | "Retry the flaky integration job once" |

**Loading policy:** one line per memory in `MEMORY.md`, which is always loaded (first **200 lines / 25 KB**). Topic files `<type>_<slug>.md` load on demand. After consolidation, `memory_summary.md` (first line `v1`, four sections: *User Profile*, *User preferences*, *General Tips*, *What's in Memory*) is injected instead of the raw index.

## File format

```yaml
---
type: feedback
scope: project              # org | user | project | agent | local
status: active              # active | superseded | stale
description: CI runs make test-fast first     # becomes the index line
supersedes: null
source: {sessions: [s_8f2c], evidence: ["rollout_summaries/s_8f2c_fix-ci.md:12-30"], epistemic: verified}
created: '2026-10-30T12:00:00Z'
modified: '2026-11-02T09:14:00Z'   # harness-stamped on every write
last_used: '2026-11-03T08:00:00Z'  # from citations, folded in by consolidation
use_count: 4
tainted: false                     # true if written by a polluted session
---
# CI runs make test-fast first
CI runs `make test-fast` first; full suite only on main. Never edit `ci/generated/*`.
```

## Write semantics: ADD-only, supersede, never destructive

```python
from harness_memory import SemanticMemory, Epistemic

sm = SemanticMemory(project_store)
res = sm.remember("project", "Release freeze date", "Freeze starts 2026-11-20.", sessions=["s1"])
#   -> project_release-freeze-date.md + index line; AlreadyExistsError if an active one exists

new = sm.supersede(res.path, "Freeze moved to 2026-11-27.", reason="PM moved it",
                   sessions=["s2"], epistemic=Epistemic.VERIFIED)
#   -> project_release-freeze-date--v2.md (supersedes: old)
#   -> old file kept: status=superseded, superseded_by, valid_to=now, conflict_note
#   -> index now points at --v2

sm.current(res.path)      # follows superseded_by to the live version
sm.history(new.path)      # oldest -> newest chain
sm.mark_stale(path, "dashboard was renamed")   # drops out of default search, stays readable
sm.forget(path)           # explicit user-requested deletion (git keeps the audit trail)
```

Guards:

* `reject_if=looks_derivable` (the default) refuses bodies that are just a file path, a commit SHA or pasted code. Store the *insight*, not what `git` or `grep` can re-derive. Pass `reject_if=None` to disable it.
* Descriptions must be single-line because they become index lines.
* `lesson` and `workflow` require `sessions` or `evidence`.

## Reads

```python
hits = sm.search("release freeze", top_k=20, threshold=0.1)          # active + untainted only
hits = sm.search("release freeze", as_of=datetime(2026, 11, 21, tzinfo=UTC))   # point-in-time
hits = sm.search("dashboards", types=["reference"], include_stale=True, include_tainted=True)
for h in hits:
    print(h.doc_id, round(h.score, 2), h.signals)   # signals: bm25 / semantic / entity / recency
```

Retrieval details (BM25, optional embeddings, entity match, recency boost for "latest/current" queries, validity windows) are in [retrieval-and-forgetting.md](retrieval-and-forgetting.md).

## `MemoryIndex`

* `upsert(IndexEntry(title, target, note))` and `remove(target)` edit only the matching `- [title](target) — note` line, preserving headings and free text.
* `status()` reports `ok`, `near_cap` (≥ 90%; returns Claude Code's reminder *"keep one line per entry, move detail into topic files, and merge or drop stale entries"*) or `over_cap`. Past the cap, writes are **accepted** but the memory tool returns an error demanding an index rewrite.
* `for_prompt()` returns exactly what is loaded at session start. `dangling()` lists entries whose files are gone.
