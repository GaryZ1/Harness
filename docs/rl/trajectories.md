# Trajectory schema and export: `harness_rl.trajectory`

## Purpose

This module turns what the harness already records (an eval `Trajectory`, a `harness_core` `Event` stream, or a `Message` list) into validated, redacted training records that RL trainers and fine-tuning pipelines can consume.

## Research basis

* **Native tool roles** (§11.7 step 3; §11.8): sending tool results as user messages breaks DeepSeek's reasoning retention (Terminus and Roo Code are named) and native tool-role training generally.
* **Rollout identity** (agent RFT practice): every rollout has a unique id attached to its tool calls.
* **Token-level data**: Harbor's `collect_rollout_details` captures token ids and logprobs. Routing all model calls through one client lets a black-box trainer attach later (§11.7 step 13).
* **Never store secrets**: `harness_memory.SecretRedactor` runs on every exported string.

## Schema (`harness_rl.trajectory/v1`)

| Field | Meaning |
|---|---|
| `rollout_id`, `task_id`, `split`, `seed`, `episode_index` | Identity and reproducibility |
| `model`, `profile`, `profile_hash`, `harness_version` | Which harness and model produced it |
| `system`, `messages` | Anthropic-shape messages (`tool_use` in assistant, `tool_result` blocks in the following user turn) |
| `tools` | Tool specs offered (API shape) |
| `tool_calls` | Flattened `{id, name, input, output, is_error}` |
| `reward`, `reward_components`, `hack_signals`, `outcome` | From `harness_rl.rewards` / `harness_evals` grading |
| `usage`, `turns` | Efficiency |
| `rollout_details` | Token ids / logprobs passthrough (optional) |
| `metadata` | e.g. `truncated`, `tool_stats`, `exclude_from_training`, `flags` |
| `created_at`, `schema` | |

## API

```python
from harness_rl import from_trajectory, from_events, from_messages, TrajectoryExporter, load_jsonl

rec = from_events(session_events, rollout_id="rollout_…", task_id="fix-001", reward=1.0, split="train")
rec.validate()                           # raises TrajectorySchemaError
rec.to_openai_chat()                     # assistant tool_calls + role "tool" results; reasoning blocks kept verbatim
TrajectoryExporter("out.jsonl").write(records)                    # validate all, then atomic replace
TrajectoryExporter("out.jsonl", fmt="openai_chat").append(rec)    # locked append
list(load_jsonl("out.jsonl"))
```

`tool_results_in_user_text(messages)` is a regression check for loop authors. It returns the indices of user messages that carry text while a tool call is still unanswered, which is the "tool output pasted as user text" bug.

## Defaults

| Setting | Default | Vendor default vs our choice |
|---|---|---|
| Canonical shape | Anthropic content blocks | Our choice (mirrors `harness_core.types`) |
| Export format | `native`; `openai_chat` optional | Our choice |
| Redaction | `SecretRedactor()` default patterns on every string | Our choice; Codex/Letta "never store secrets" |
| Opaque keys exempt from redaction | `signature`, `encrypted_content`, `data` | Our choice: provider reasoning payloads must round-trip byte-exact |

## Invariants (`validate()`)

* `rollout_id` and `task_id` are present, and the schema version is known.
* Roles are `user`, `assistant` or `system`.
* `tool_use` blocks appear only in assistant messages, and their ids are unique.
* A `tool_result` sits only in the user/tool position and answers exactly one earlier open `tool_use`.
* Every `tool_use` gets a result, unless `metadata.truncated` is set (timeouts, budget cuts).
* `write()` validates every record before touching the file, so a bad record leaves the previous file intact.

## Failure modes

* **Exporting a timed-out rollout without `truncated`.** `TaskEnvironment.finish` sets it automatically.
* **A custom redaction pattern that matches signatures.** Keep signatures under the opaque keys.
* **Hand-rolled chat templates for open models.** Out of scope here. Use the vendors' reference encoders downstream of this JSONL.
