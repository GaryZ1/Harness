# Harness profiles: `harness_rl.profiles`

## Purpose

A profile presents each model with the tool set, edit format, prompt style, reasoning-replay protocol, tool-result role and compaction protocol it was trained with. Every scaffold sits behind a flag so it can be ablated per model.

## Research basis

* **Harness-only deltas** (§11.4): +7.2 points (Codex CLI vs Terminus-2), +5.2 (TB 2.1), +20.5% (Claude Code SDK vs a generalist scaffold), +13.7 (LangChain).
* **Mismatch failures**: OpenAI recommends its exact `apply_patch` implementation, and GPT-5.3-Codex calls a non-existent `apply_patch` in harnesses that lack it.
* **Recommended profiles** (§11.7 step 1):
  * OpenAI Codex models: `apply_patch` (V4A) plus `shell`/`exec_command` and `update_plan`;
  * Claude: str_replace-style edit, bash and read/glob/grep (inferred, not confirmed by Anthropic);
  * DeepSeek: the reference encoders.
* **Compaction** (§11.7 step 4): Codex-family models use a 90% trigger and a handoff prompt. A reset-plus-handoff mode suits models with "context anxiety". Every scaffold should be a flag.
* **Pitfalls** (§11.8, §11.7 step 15): one generic tool set for every model, and stale scaffolds kept past their model generation. Ablate at every model upgrade.

## Configuration

```yaml
# config/profiles.yaml - patterns are globs over model ids from YOUR deployment; no ids live in code
fallback: generic            # optional; used only with select(..., allow_fallback=True)
profiles:
  - name: patch-native
    family: codex
    match: ["<codex-family-glob>"]
    edit_format: apply_patch_v4a
    edit_tool: apply_patch
    tools:
      - {name: apply_patch, description: "Apply a V4A patch (*** Begin Patch … *** End Patch)"}
      - exec_command
      - write_stdin
      - update_plan
    prompt_style: vendor_prompt_file
    prompt_file: prompts/codex_family.md
    reasoning_replay: openai_encrypted_reasoning
    tool_result_role: tool
    compaction: {mode: client, trigger_ratio: 0.9, prompt_file: prompts/handoff.md}
    scaffolds: {todo: false, planner: false, evaluator_gate: true}
  - name: edit-native
    family: claude
    match: ["<claude-family-glob>"]
    edit_format: str_replace
    edit_tool: edit
    tools: [read, edit, write, bash, glob, grep]
    reasoning_replay: anthropic_thinking
    tool_result_role: tool_result_block
    tool_aliases: {apply_patch: edit}       # mismatch hint: "Use 'edit' instead."
    scaffolds: {todo: false, think: false, evaluator_gate: true, context_resets: false}
  - name: generic
    match: ["__explicit-fallback-only__"]
    edit_tool: write
    tools: [read, write, bash]
```

## API

```python
from harness_rl import ProfileRegistry, ProfileMismatchMonitor
reg = ProfileRegistry.from_file("config/profiles.yaml")
sel = reg.select(model_id)                       # ProfileSelection(profile, model, matched_pattern, fallback=False)
sel = reg.select(model_id, allow_fallback=True)  # fallback=True is reported, never silent
p = sel.profile
p.tool_names, p.edit_tool, p.reasoning_replay, p.tool_result_role, p.compaction.trigger_ratio
p.scaffold("todo"); p.with_scaffold("todo", True); p.ablations()   # [(flag, profile_with_flag_off), …]
p.content_hash()                                 # stamped into every TrainingRecord (profile_hash)
mon = ProfileMismatchMonitor(p); mon.check("apply_patch")   # None, or an actionable ToolOutcome(is_error=True)
```

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| `tool_result_role` | `tool_result_block` | Anthropic-native; `tool` for chat-completions-style providers |
| `reasoning_replay` | `none` | Set per family: `anthropic_thinking` (signed blocks), `openai_encrypted_reasoning`, `deepseek_reasoning_content` |
| `compaction` | `client`, trigger 0.9, no reset-handoff | 0.9 is the Codex-family vendor default; others our choice |
| Scaffold flags | off unless configured | Our choice: newest models moved todo behaviour into the model |
| Selection | most specific glob wins; ties raise; no match raises | Our choice, from "never fall back silently" |

## Invariants

* At least one `match` glob. Tool names are unique, and `edit_tool` is one of the tools.
* `tool_result_role` ∈ {`tool`, `tool_result_block`}. `user_text` is rejected.
* `reasoning_replay` comes from the known set, and scaffold values are booleans.
* `compaction.trigger_ratio` ∈ [0.1, 1.0].
* Profile names are unique, and the fallback must be a defined profile.

## Failure modes

* **`ProfileNotFoundError` at startup.** Add a profile for the new model family. Do not paper over it with a fallback.
* **`ProfileAmbiguityError`.** Two profiles have equally specific globs. Make one more specific.
* **A rising `profile_mismatch` metric.** The model expects tools the profile lacks. Add the native tool, or alias it.
* **A stale scaffold.** Run `ablations()` through `harness_rl.AblationProposer` / the improvement loop at every model upgrade.
