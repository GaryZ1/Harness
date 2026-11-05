# Instruction memory: `harness_memory.instructions`

**Function:** human-written, always-true rules for the agent, such as conventions, commands and guardrails.
**Loading policy:** always loaded at session start. Subdirectory files load lazily.
**Writer:** humans only. When the agent is asked to "add this to AGENTS.md", that is an instruction edit and needs user approval. It is not a memory write.

Instructions are *context, not enforced configuration*. Anything that must hold, such as "never edit `ci/generated/*`" or "do not delete tests", belongs in a hook. See `task_state.guard_tool_call` for an example.

## Loading rules (Claude Code semantics)

| Rule | Implementation |
|---|---|
| Managed file first, then user file (`~/.harness/AGENTS.md`), then project files from the project root down to `cwd` (closest file read last) | `InstructionLoader.load(cwd)` |
| `CLAUDE.md` or `.claude/CLAUDE.md` per level. `AGENTS.md` is used only when no `CLAUDE.md` exists (`mode="claude-md-or-agents-md"`). Other modes: `claude-md-and-agents-md`, `agents-md-only` | `_files_at` |
| `CLAUDE.local.md` / `AGENTS.local.md` is appended after its sibling | `_files_at` |
| `@path` imports expand recursively up to **4 hops**, with cycles ignored | `_load_file(depth)` |
| Imports that leave the project root need a one-time approval callback (`approve_external`) | `_approved` cache |
| Files over **4 MiB** are skipped. Files over **200 lines** produce a warning | `MAX_FILE_BYTES`, `RECOMMENDED_MAX_LINES` |
| Subdirectory instruction files load lazily when a file under them is touched | `InstructionLoader.for_path(path)` |

## Usage

```python
from harness_memory import InstructionLoader

loader = InstructionLoader(repo_root, user_file="~/.harness/AGENTS.md",
                           approve_external=lambda p: ask_user(f"Load {p}?"))
loaded = loader.load()
system_prompt += loaded.render()          # <instructions source="..."> blocks
for warning in loaded.warnings:
    log.warning(warning)

# When the agent reads or edits pkg/payments/api.py:
extra = loader.for_path("pkg/payments/api.py")   # append as a message, not into the cached prefix
```

## Guidance

* Keep each file under 200 lines and include only always-true facts. Move procedures into skills and learned facts into semantic memory.
* Contradictory instruction files make the model pick a side arbitrarily. The loader preserves order (closest file last) so the most specific rule is read last.
* Edits to instruction files take effect on the next session or after compaction. Do not rewrite the cached prompt prefix mid-session.
