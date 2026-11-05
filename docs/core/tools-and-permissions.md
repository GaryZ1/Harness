# Tools and permissions: `harness_core.tools`, `harness_core.permissions`

**Purpose:** put the "hands" behind the shared `ToolExecutor` protocol so the loop never imports a
sandbox, and decide per call whether it may run.

## Research basis

* **Managed Agents:** tools dispatch through `execute(name, input) → string`; sandboxes are provisioned lazily, "only when a tool call needs one" (about −60% p50 and −90% p95 time-to-first-token); when a sandbox dies the failure returns to the model as a tool error it can retry, and the session survives.
* **OpenAI Agents SDK:** `tool_not_found_behavior="return_error_to_model"` turns a hallucinated tool name into a correctable error.
* **Claude Agent SDK:** read-only tools run concurrently, custom tools are sequential unless marked read-only; `default` permission mode asks a `canUseTool` callback and denies when there is none; a denied call comes back "as the tool result".
* Checklist §2.7 step 5: run tools concurrently only when *trusted* metadata marks them read-only (treat untrusted MCP `readOnlyHint` as false), serialize writes, cap concurrency, and preserve call/result pairing.

## Executors

```python
from harness_core import FunctionToolExecutor, CompositeExecutor, LazyExecutor, ToolSpec, ToolOutcome

local = FunctionToolExecutor()

@local.tool("grep", "Search files", {"type": "object", "properties": {"pattern": {"type": "string"}}}, read_only=True)
def grep(inp, ctx):                     # return ToolOutcome | str | JSON-able
    return ToolOutcome("3 matches", metadata={"bytes": 120})

sandbox = LazyExecutor(
    [ToolSpec("bash", "Run a shell command", {...})],   # declared up front: listing tools provisions nothing
    factory=lambda: MySandboxExecutor(),                  # called on the first bash call
    on_dispose=lambda ex: ex.close(),
)
executor = CompositeExecutor(local, sandbox)              # routes by tool name; duplicate names rejected
```

| Executor | Behaviour |
|---|---|
| `FunctionToolExecutor` | Plain callables `fn(input, ctx)`. Unknown name → error outcome; a tool exception → error outcome `Error: <Type>: <msg>`; `SandboxError` is re-raised so a wrapping `LazyExecutor` can react |
| `CompositeExecutor` | Routes by name; `cancel()` fans out to children |
| `LazyExecutor` | Provisions on first `execute`; inner `SandboxError` → error outcome "the sandbox died…", instance disposed, next call re-provisions; factory failure → error outcome "could not be provisioned" |

The loop adds a second guard: any exception from `execute` becomes an `is_error` result.
Executors must be safe to call concurrently for read-only tools. Each call's `RunContext` carries
`metadata["tool_use_id"]` and `metadata["idempotency_key"]` (`sha256(session_id:step:tool_use_id)[:32]`,
where `step` is the `seq` of the assistant event), so side-effecting executors can de-duplicate
retries after a crash.

`ToolSpec.open_world=True`, or an outcome with `metadata["untrusted"]=True`, sets
`RunContext.tainted` for the rest of the run (and emits one `custom` event), for downstream
policies such as memory-write restrictions.

## Permission pipeline (per call, in order)

1. **`pre_tool` hooks** (in registration order; each sees the input as rewritten by the previous `updated_input`). `continue_=False` halts the run. Combined decision: deny beats ask beats allow; a `block` counts as deny.
2. Hook **deny** → denied.
3. **Policy** `check(call, spec, ctx)`; its `updated_input` is applied; policy **deny** → denied (a hook allow cannot override a policy deny).
4. **Ask** if a hook said ask, or the policy said ask and no hook said allow:
   * an entry in `run(approvals={...})` answers it (resuming a pause);
   * no approver → denied ("approval required but no approver is configured");
   * else `approver(call, spec, ctx, decision)` inside an `approval_wait` span, returning `bool` or a `PermissionDecision` (whose `updated_input` is applied), or raising `ApprovalRequired` to pause durably.
5. Every outcome is a `permission` event (`behavior`, `source`, `reason`). A denial returns `Permission denied: <reason>` as an `is_error` tool result and counts toward the denial limits.

### Durable approval pauses

```python
def approver(call, spec, ctx, decision):
    notify_reviewer(call)              # e.g. post to a review queue
    raise ApprovalRequired("waiting for reviewer")

r = loop.run("deploy")                 # -> paused_for_approval; r.pending_approvals[0].tool_use_id
# ... later, possibly in a new process:
r = AgentLoop(model, executor, session=store.wake(r.session_id), approver=approver).run(approvals={tool_use_id: True})
```

Calls in the batch that already ran are not re-run; the pending call and any later ones then run.

## Policies

| Policy | Decision |
|---|---|
| `ReadOnlyPolicy` (loop default) | allow if the spec is `read_only`, else ask |
| `AllowAllPolicy` | allow everything (isolated environments only) |
| `DenyAllPolicy` | deny everything (dry runs, judges) |
| `AskAllPolicy` | ask for everything |
| `CallbackPolicy(fn)` | adapt `fn(call, spec, ctx) -> PermissionDecision` |

The layered permission package (modes, `Tool(pattern)` rules, classifier) implements the same
`PermissionPolicy` protocol and replaces these.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Default policy | `ReadOnlyPolicy` | Agent SDK `default` mode without a callback (deny) plus read-only auto-allow (ours) |
| Parallel read-only tools | on, at most 8 at once | Claude Agent SDK rule; cap is ours |
| Unknown tool | error result listing available tools | Agents SDK `return_error_to_model` |
| Denial limits | 3 consecutive / 20 total | Claude Code auto mode |

## Failure modes

| Failure | Result returned to the model |
|---|---|
| Unknown tool name | `Error: unknown tool 'x'. Available tools: …` |
| Tool raises | `Error: RuntimeError: …` |
| Sandbox dies | `Error: the sandbox died while running bash (…)`; next call re-provisions |
| Sandbox cannot start | `Error: sandbox could not be provisioned (…); retry later` |
| Denied | `Permission denied: <reason>`; after 3 in a row the run ends `error_permission_denials` |
