# Security and operations

## Threat model

Memory turns a one-time prompt injection into a persistent one. A poisoned memory is read as *trusted* in every later session (Managed Agents warning, MINJA, MemoryGraft, OWASP ASI06). The defences below are all enforced in code.

| Control | Where | Behaviour |
|---|---|---|
| Path sandbox | `security.PathSandbox` | Canonicalises with `resolve()` + `relative_to()`. Rejects `../`, `..\`, URL-encoded and double-encoded traversal (`%2e%2e%2f`, `%252e…`), absolute and drive paths, NUL bytes and symlink escapes. `.git` and lock files are protected. |
| Secret redaction | `security.SecretRedactor` | Applied on every store write and transcript append. Covers private keys, Anthropic/OpenAI-style keys, GitHub/Slack/AWS/Google tokens, JWTs, bearer tokens, URL credentials and `*password/secret/token/api_key*=` assignments. Replaced with `[REDACTED_SECRET]`. Extend with `add_pattern`. |
| Taint tracking | `security.TaintTracker`, `Session.record_tool_output` | Output from any tool outside `trusted_tools` (or with `untrusted=True`) marks the session `polluted`. Later writes are stamped `tainted: true`, tainted memories are excluded from default search, polluted sessions are never extracted, tainted raw memories are quarantined from consolidation, and tainted skill candidates need human approval. |
| Read-only scopes | `store.Access`, `scopes.MountSet` | Org memory is always read-only. Sub-agents get read-only project memory. At most 8 mounts per session, frozen at session start, with mount instructions ≤ 4,096 chars. |
| Caps | `store.StoreLimits`, `tool`, `index` | 100 kB per file, 10,000 files per store, `view` ≤ 16,000 chars, index 200 lines / 25 KB. |
| Treat-as-data prompts | `pipeline/*` | Transcripts and memory state are fenced `trust="data-only"`, and the system prompts say so explicitly. |
| Audit and rollback | `versioning.GitVersioner` | One git repo per writable scope. Typed commits. A baseline ref (`refs/harness/memory-baseline`). Roll back with `git revert`. For PII in history, rewrite history or rotate the repo, then archive the old one read-only. |
| Derivability guard | `semantic.looks_derivable` | Refuses to store bare paths, SHAs and code. |

## Concurrency

* `FileMemoryStore` writes are atomic (temp file + `fsync` + `os.replace`) under a re-entrant cross-process `flock`.
* Read-modify-write paths use `content_sha256` preconditions and raise `ConcurrencyError` on a lost update. Callers re-read and retry.
* Phase 1 uses SQLite leases. Phase 2 uses a single non-blocking global lock per repository.
* Transcripts use `O_APPEND` plus a lock, and each event's `seq` is derived under the lock.

## Configuration defaults

| Setting | Default | Anchor |
|---|---|---|
| Index cap | 200 lines / 25 KB | Claude Code auto memory |
| View truncation | 16,000 chars | Anthropic memory tool |
| Per-file / per-store | 100 kB / 10,000 | Managed Agents memory stores |
| Mounts per session | 8 | Managed Agents |
| Phase-1 idle gate | 12 h | Codex (">12h recommended") |
| Phase-1 batch | 16 sessions | our choice (Codex allows 1–128) |
| Rate-limit headroom | ≥ 20% | our choice (`min_rate_limit_remaining_percent`) |
| Raw memory cap | 256 | Agents SDK default |
| Unused window | 90 days | our choice (Codex `max_unused_days`; value not public) |
| Topic expiry | 180 days → archive | our choice (Anthropic: "expire files not accessed in a long time") |
| Skill promotion | verified evidence, human approval, or ≥ 2 independent successful reuses | our choice, motivated by SkillsBench/EvoClawBench |
| Retrieval | `top_k=20`, `threshold=0.1`, rerank off | Mem0 v3 |

## Operations checklist

* Run `MemorySystem.run_background(llm)` from a worker on a schedule or at session close, never on the hot path. Pass the provider's real rate-limit headroom.
* Back up `~/.harness/` (the memory repos are plain git, so `git push` to a private remote works). Never sync memory that may hold PII to shared remotes without review.
* Alert on growing `ConsolidationReport.errors` (invalid ops, quarantined inputs) and on Phase-1 jobs stuck in `failed`.
* Prune before hard limits. The tool's index warnings and `expire_unused` handle steady state.

## Evaluating memory

* Measure **cross-session task outcomes**: workflow knowledge, environment gotchas and premise awareness, in the style of MemoryArena or LongMemEval-V2. Track tokens and latency per query too. Treat LoCoMo and LongMemEval scores as regression checks only.
* Track citation-based usage per memory (`UsageLog.stats()`). Memories that are never cited are candidates for removal.
* **Reset or freeze memory for every eval trial** (fresh `base` directory). Shared memory and git history make trials non-independent.
