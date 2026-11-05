# Session log: `harness_core.session`

**Purpose:** the durable, append-only record of a session. Every prompt is built by projecting it,
and any harness process can resume a session from it after a crash.

## Research basis

* **Managed Agents** (Anthropic, April 2026): a session is "an append-only event log"; events persist via `emitEvent(id, event)`; `getEvents()` returns positional slices; after a harness crash a replacement calls `wake(sessionId)` and resumes "from the last event". Decoupling the log from the container is what made containers disposable.
* **OpenHands** records condensation as `Condensation` events "similar to tombstones", and computes the LLM-visible `View` by applying them. **Cline** stores compaction state separately and reuses it on resume only after validating a hash of the canonical prefix.
* **Codex** and **Claude Code** persist sessions as append-only JSONL with resume and fork.
* Checklist §1.7 step 1: persist every user message, model message (with reasoning blocks), tool call, tool result, hook decision, compaction and checkpoint; never mutate the log; record compaction as a new event.

## API

```python
from harness_core import SessionStore, Session, Message, EventType

store = SessionStore("~/.harness/sessions")       # <id>.jsonl, <id>.meta.json, <id>.lease
s = store.create("sess-42", metadata={"user": "u1"})
s.append_message(Message.user("hello"))           # one durable event with data["message"]
s.emit(EventType.CUSTOM, {"note": "anything"})    # observability only, not part of the transcript
s.checkpoint("before-refactor")                   # named fork/rewind target

s2 = store.wake("sess-42")                        # reopen and rebuild state from the log
s2.messages()      # canonical history (never affected by compaction)
s2.view()          # what the model sees: latest compaction view + messages appended after it
s2.events(10, 20)  # positional slice by seq (start inclusive, end exclusive)

s2.record_compaction([Message.user("<summary>")], summary="...", reason="threshold")
fork = store.fork("sess-42", at_seq=s.find_checkpoint("before-refactor"))   # conversation-only rewind
with s2.lease():   # exclusive right to drive the session (AgentLoop.run takes it for you)
    ...
```

| Class / function | Role |
|---|---|
| `EventLog` (protocol) | `emit(event) -> event`, `events(start, end)`, `last_seq`, `session_id` |
| `JsonlEventLog` | Durable implementation: one `Event.to_json()` line per event |
| `InMemoryEventLog` | Non-durable implementation for tests and ephemeral sub-agents (`Session.in_memory()`) |
| `FanoutSink` | Forward events to several `EventSink`s |
| `Session` | Log + `SessionInfo` + replayed messages + compaction projection + lease |
| `SessionStore` | `create`, `open`, `wake`, `exists`, `list`, `fork` |
| `PendingToolBatch` | From `Session.pending_tool_batch()`: tool calls of the last assistant turn without a result message, split into `completed`, `started` and `not_started` |
| `message_to_dict` / `message_from_dict` / `history_hash` | Lossless message JSON and the canonical prefix hash |

### Replay rule

A message is part of the transcript if, and only if, its event's `data` has a `"message"` key.
The loop writes the prompt, each assistant turn, each tool-result message, steering and
continuation messages that way. Everything else (`model_request`, per-call `tool_call` /
`tool_result`, `permission`, `evaluation`, `error`, `stop`) is observability and resume metadata.
A `compaction` event with a `view` key is applied on replay only if its `prefix_hash` equals the
hash of the canonical messages up to its `through_seq`.

### Session status

`SessionInfo.status` (in `<id>.meta.json`, written atomically) is `active` → `running` (while a
loop holds the lease) → `idle` (success), `paused` (waiting for approval), `interrupted`, `failed`
(any `error_*` result) or `completed` (`AgentLoop.end_session`). A crashed run leaves `running`;
that is expected, and `wake` + `run()` repairs it.

## Defaults

| Setting | Default | Source |
|---|---|---|
| `fsync` on every append | on | our choice (durability over throughput; disable for tests) |
| Session id pattern | `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`, no `..` | our choice (safe file names) |
| Generated ids | `sess_<20 hex>` | `harness_core.types.new_id` |
| Fork refuses to split a tool batch | yes | our choice (resume would silently re-run those tools) |

## Invariants (enforced in code)

* Events are only appended. `seq` is contiguous from 0, assigned under an exclusive `flock` on `.<file>.lock`; a writer re-reads the tail if another process appended since its last write.
* Each event is written with one `write()` call on an `O_APPEND` descriptor and `fsync`-ed before `emit` returns.
* On open, a torn final line (no trailing newline, or invalid JSON) is truncated (`repaired_bytes` reports how much). Any other malformed line, or a `seq` gap, raises `CorruptLogError` instead of being skipped.
* Events for another session id are rejected (`ValueError`).
* A session is driven by at most one harness: `Session.lease()` takes a non-blocking `flock` on `<id>.lease` (released by the OS if the holder dies) and raises `SessionLockedError` otherwise. In-memory sessions use a thread lock.
* Compaction never rewrites the log; `messages()` is always the full canonical history.

## Failure modes

| Situation | Behaviour |
|---|---|
| Crash mid-append | Torn line truncated on next open; earlier events intact |
| Hand-edited or corrupted log | `CorruptLogError` with file and line number |
| Two harnesses wake the same session | Second gets `SessionLockedError` |
| Metadata file lost but log present | `open` reconstructs `SessionInfo` with status `active` |
| Compaction event whose prefix no longer matches | Ignored on replay; the full canonical history is used |
| Reading while another process appends | A line without a trailing newline is treated as not yet committed |
