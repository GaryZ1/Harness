# Prompt layout: `harness_context.layers`, `dynamic`, `references`

**Purpose:** render the stable layers [1] and [2] of the fixed request layout. Route everything time-varying through a single dynamic channel at the end of the window. Keep lightweight references in context in place of the data.

**Research basis:**

* Reference report §3.3, techniques 1, 3, 5 and 6, and §3.7 steps 2, 4, 8 and 9.
* "Right altitude" static prompts with no timestamps, IDs or environment state.
* Root-to-cwd instruction files of ≤200 lines, with block HTML comments stripped.
* A skills index of name plus description, about 100 tokens per skill.
* A memory index of ≤200 lines and 25 KB, and a repo map of about 1k tokens.
* Turn-scoped dynamic content (`clear_at: "next_user_message"`) and recitation at the end of context (Manus `todo.md`).
* Just-in-time identifiers in place of pre-loading (Anthropic), also known as restorable compression (Manus).

## Layers

```python
stack = (PromptStack(project_byte_budget=32 * 1024)
         .add(PromptLayer("style", "Prefer small, reviewed diffs.", LayerKind.SYSTEM))
         .add_instructions(loaded.render(), source="AGENTS.md")       # harness_memory.InstructionLoader output
         .add_skills_index([SkillEntry("pdf", "Fill PDF forms"), SkillEntry("deploy", "…", model_invocable=False)])
         .add_memory_index(memory_index_text)
         .add_repo_map(repo_map_text)
         .add_reference_index([ContextReference("file", "docs/api.md", "API notes", how="read_file(path, offset, limit)")]))
rendered = stack.render(BASE_SYSTEM_PROMPT)    # RenderedPrompt(blocks, lints, layer_bytes, kinds)
```

Render order is fixed. The base prompt comes first, then every `SYSTEM` layer, then every `PROJECT` layer, each group in insertion order. Text is normalized: `\r\n` becomes `\n`, trailing whitespace is stripped, and block HTML comments are removed. Truncation is deterministic and ends with a marker that names the source to read. The same inputs therefore always render to the same bytes.

| Setting | Default | Source |
|---|---|---|
| Instruction lint | > 200 lines → warning | vendor guidance ("<200 lines") |
| Memory index | first 200 lines, 25,000 bytes | vendor default (Claude Code `MEMORY.md`) |
| Skill description | 400 chars (≈100 tokens), sorted by name; `model_invocable=False` skills are omitted | vendor guidance (`disable-model-invocation`) |
| Repo map | 4,096 bytes (≈1k tokens) | Aider `map_tokens=1024` |
| Project byte budget | 32 KiB across all PROJECT layers; the overflowing layer is truncated and later layers are dropped | our choice, modeled on Codex `project_doc_max_bytes` |

File discovery is deliberately out of scope here: root-to-cwd search, `@imports`, overrides and lazy nested files. Use `harness_memory.InstructionLoader`, or any other loader, and pass in the text. When a nested instruction file is first touched, append it with `assembler.append_reminder(...)` and do not rebuild the prefix.

## Dynamic channel

```python
dyn = DynamicContext(role=Role.SYSTEM)                  # Role.USER → <system-reminder> merged into the last user turn
dyn.register("git", lambda ctx: git_status_short())     # evaluated every window
dyn.register_recitation(working_memory.recitation)      # rendered last (priority 1000)
dyn.push("file-changed", "src/app.py changed on disk")  # next window only
```

* **Turn-scoped** items (`collect` / `render`) form one tail message with metadata `{"kind": "dynamic", "turn_scoped": True, "clear_at": "next_user_message"}`. The message is **never stored in the log**, is placed after the moving breakpoint and is rebuilt on every call. Ordering is by `(priority, key)`. The assembler adds the pre-compaction warning (50), the loop check (60) and the budget note (900).
* **Persistent** items (`persistent_message`) are reminders that must stay. They are de-duplicated by key and content hash. Append them through `assembler.append_reminder`, which refuses to append between a `tool_use` and its `tool_result`.
* A failing provider renders as `(<key> unavailable: <Error>)`. It never breaks the request.

## References

`extract_locators(tool_input)` returns the re-fetchable identifiers in priority order: `path`, `file_path`, …, `url`, `uri`, `query`, `pattern`, `glob`, `command`, `id`. Values are truncated to 200 chars. Clearing placeholders use these locators. `render_reference_index(refs)` renders a sorted, de-duplicated `<references>` block.

## Invariants and failure modes

* Layer names are unique and match `[A-Za-z0-9_.:-]{1,64}`. Violations raise `ConfigurationError`.
* Nothing in `layers` reads the clock or the environment. The cache checker ([caching.md](caching.md)) flags any timestamps or ids the caller passes in.
* Over-budget content is truncated or dropped with a `LayerLint`, never silently, and the lints appear in the `ContextReport`.
