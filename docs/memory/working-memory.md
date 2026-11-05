# Working memory: `harness_memory.working`

**Function:** the scratchpad for the current task: goal, plan/todos, short notes and the context-window budget.
**Loading policy:** lives in context, recited at the *end* of the context, and mirrored to disk so it survives compaction and context resets ("ASSUME INTERRUPTION").

## Model

| Element | Behaviour |
|---|---|
| `TodoItem` | `pending` → `in_progress` → `completed` / `blocked`. **At most one `in_progress`** (enforced). |
| `notes` | Bounded FIFO (`max_notes=50`, `max_note_chars=2000`), timestamped. Long material belongs in files. |
| `recitation()` | A compact `<working-memory>` block with the goal, the plan with progress, and the last *n* notes. Append it as a message near the end of the context (Manus "recitation" against lost-in-the-middle). Never put it in the cached system prefix. |
| `ContextBudget` | `window_tokens`, `compaction_threshold=0.8`, `warn_margin=0.05`. |
| `observe_usage(tokens)` | Returns `PRE_COMPACTION_WARNING` **once** as usage approaches the threshold, telling the model to save findings before old tool results are cleared. |
| `after_compaction()` | Resets the warning latch. Re-inject `recitation()` together with `MemorySystem.post_compaction_context()`. |
| `save()/load()` | Atomic JSON persistence. `to_markdown()` renders a human-readable `NOTES.md`. |

## Usage

```python
from harness_memory import WorkingMemory
from harness_memory.working import ContextBudget

wm = WorkingMemory(goal="Add OAuth login", budget=ContextBudget(window_tokens=200_000))
wm.set_plan(["Read auth module", "Add provider config", "E2E test login"])
wm.start_next()
wm.note("Provider secrets come from $OAUTH_CLIENT_SECRET (never inline)")

if (warning := wm.observe_usage(response.usage.input_tokens)):
    messages.append({"role": "user", "content": warning})

messages.append({"role": "user", "content": wm.recitation()})   # end of context
wm.save(".harness/task/working.json")
```

## Design notes

* Working memory is per session. Durable learnings move into semantic or procedural memory through the memory tool or the background pipeline. Per-task progress belongs in [task state](task-state.md).
* Gate scaffolds per model. Newer models need fewer todo or reset scaffolds, and Manus reportedly spent about a third of its actions maintaining `todo.md`. Keep recitation cheap and optional.
