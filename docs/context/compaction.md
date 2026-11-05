# Compaction and re-injection: `harness_context.compaction`, `harness_context.reinjection`

**Purpose:** when clearing is not enough, summarize older history into a structured handoff, keep the recent tail verbatim and re-load durable state. All of this happens without editing the log. These are ladder rungs L3 and L4.

**Research basis:** reference report §3.3 techniques 17, 19 and 20, §3.7 steps 11–12, and the §3.8 "Compaction errors". The summary schema merges the Anthropic SDK and Codex prompts and adds an explicit "do not call any tools", because without it the model sometimes calls a tool instead of summarizing. Other sources:

* **Keep tail:** the reference loop keeps the last 2 turns, and the cut never splits a `tool_use` from its `tool_result`.
* **Codex fallback:** keep the user's own messages, newest first, up to 20k tokens.
* **Re-injection list (Claude Code):** root instructions, memory, plan, git status, active skill bodies (5k tokens each, 25k total, oldest dropped first) and up to 5 recent files (over 5k tokens becomes a path only).
* **Fit check:** the summarization call itself must fit, and an overflow returns HTTP 200 with empty content.
* **Concurrency:** never run two compactions at once.

## Summary schema

`SUMMARY_PROMPT` requires these headings, in this order:

| Key | Heading | Required |
|---|---|---|
| `task_overview` | Task overview | yes |
| `user_preferences` | User preferences and learnings | yes |
| `current_state` | Current state | yes |
| `files_touched` | Files touched | yes (also harness-merged) |
| `decisions` | Decisions and rationale | yes |
| `errors` | Errors and failed approaches | no |
| `open_tasks` | Open tasks and next steps | yes |
| `references` | Critical references | no |

`parse_summary(text)` behaves as follows:

* It accepts `<summary>…</summary>`, with or without a closing tag, or a bare Markdown body with recognizable `#` or `**bold**` headings, matched by keyword.
* It raises `CompactionError` when neither is present.
* Required sections that are missing are listed in `missing_sections`.
* `render()` always outputs every heading and writes "None recorded." under empty ones.
* `merge_files(paths)` appends harness-observed paths that the model omitted. The paths come from `path`, `file_path`, `paths` and similar keys in the inputs of every `tool_use` in the summarized range, plus the previous marker's list.

## Compactor

```python
comp = Compactor(model_client, config=CompactionConfig(), counter=counter, reinjection=builder, events=sink, tracer=tracer, clock=clock)
result = comp.compact(system=system_blocks, history=window, log=durable_log, tools=tools, ctx=ctx,
                      max_request_tokens=effective_window, focus="the auth bug", trigger="manual")
durable_log.append(result.marker)       # the assembler does this for you
```

1. Take the per-session lock without blocking. If it is held, raise `CompactionInProgressError`.
2. Pick the cut with `choose_cut`: the start of the `keep_recent_turns`-th last real user turn, falling back to the latest safe boundary that leaves ≥4 tail messages. Raise `NothingToCompactError` if there is none. Every cut must satisfy `is_safe_cut`.
3. Build the request with the **same `system` and `tools` as the session**, so the call reads the session cache. Messages are the head (with any previous marker first) plus the instructions, `max_tokens = summary_max_tokens`, and `extra = request_extra` for pinned effort and thinking.
4. Fit check: if the estimate plus `summary_max_tokens` exceeds `max_request_tokens`, replace every head tool result with a placeholder. If it still does not fit, raise `CompactionError` ("threshold too close to the window").
5. If the model returns `tool_use`, retry once with a reinforced "respond with text only". If it fails again, or the stop reason is anything other than `end_turn`/`stop_sequence`, or the text is empty, raise `CompactionError`.
6. Build the **marker** as a user message with these blocks: `SUMMARY_PREFIX` plus `<summary>` (schema-rendered), `<prior-user-messages>` (≤20k tokens, oldest first, chained from the previous marker), `<operator-instructions>` (mid-conversation `system` messages from the summarized range, which would otherwise stop applying) and `<post-compaction-context>` (the re-injection bundle).

The marker metadata, under `metadata["harness_context"]`, holds:

* `kind="compaction"`, `compaction_id`, `mode`, `trigger`, `previous` and `created` (in metadata only, never in the text);
* `covers_through` and `covers_log_index`;
* `prefix_sha256`, a hash chain over the covered log prefix;
* `summarized_count`, `user_messages`, `operator_instructions`, `files_touched`, `missing_sections`, `reinjected` and `usage`;
* `tokens_before` and `tokens_after`.

A COMPACTION event (`client_compaction`) is emitted, and an OTel-style span `context.compaction` wraps the call.

The assembler projects `[marker] + log messages after covers_through`. Compacting again summarizes the old marker together with everything after it.

| Setting | Default | Source |
|---|---|---|
| `keep_recent_turns` | 2 | reference loop |
| `min_tail_messages` | 4 | reference loop (4 messages from the end) |
| `retain_user_message_tokens` | 20,000 | Codex `COMPACT_USER_MESSAGE_MAX_TOKENS` |
| `summary_max_tokens` | 8,000 | research example ("several thousand") |
| `send_tools` | `True` | vendor guidance (same system and tools) |
| `carry_operator_instructions` | `True` | our choice (mitigates a documented loss) |

## Re-injection sources

```python
builder = ReinjectionBuilder([
    TextSource("instructions", lambda: loader.load().render()),
    TextSource("memory", lambda: memory_system.post_compaction_context(session)),   # harness_memory integration
    TextSource("git", git_status_short),
    SkillBodiesSource(lambda: active_skills),          # [(name, body)] oldest first; 5k each / 25k total
    RecentFilesSource(tracker, repo_root),             # ≤5 files; >5k tokens → <referenced-file path=… tokens=…>
], background_jobs=lambda: ["npm run dev (pid 42)"])
```

Sections are rendered in the configured order inside `<post-compaction-context>`. A failing source becomes "(could not re-load X: …)" and never aborts compaction. `RecentFilesSource` is sandboxed to its root. `SkillBodiesSource` keeps the newest skills, truncates each one from the end and drops the oldest first. `FileActivityTracker.observe_call(...)` records read and edit tool paths.

## Invariants

* The marker is appended, and the covered messages are never edited or deleted. `LogIntegrityError` is raised when they change (checked by the assembler) or when `covers_through` is not in the log.
* The bundle is rendered once and stored in the marker, so later windows carry identical bytes.
* Compaction is single-flight per session, and other sessions are not blocked.
