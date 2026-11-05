# Tool layer (`harness_tools`)

Production Python implementation of the tool design in
[`reports/SOTA agentic harness reference.md` §5](../../reports/SOTA%20agentic%20harness%20reference.md) (checklist §5.7, pitfalls §5.8),
[`research_notes/.../tools_sandboxing_safety.md`](../../research_notes/SOTA%20agentic%20harness%20build%20plan/tools_sandboxing_safety.md) §1–4 and the build plan's
"Tools and protocols: a five-tool kernel, a searchable catalog, open standards". Each component has its own module and its own page:

| Component | What it does | Module | Doc |
|---|---|---|---|
| **Registry / executor** | Implements `harness_core.types.ToolExecutor`: register functions or handlers, validate input, run with timeouts, convert every failure into an `is_error` result, cap output, mark taint, run read-only calls in parallel, record telemetry | `registry.py`, `output.py` | [registry.md](registry.md) |
| **JSON Schema** | Stdlib validator (2020-12 subset), schema-from-signature, per-provider strict compiler and lint | `schema.py` | [schemas.md](schemas.md) |
| **Kernel tools** | `read`, `write`, `edit` (exact unique replace), `bash`, `grep`, `glob`, confined to a workspace | `kernel.py`, `workspace.py`, `descriptions.py` | [kernel.md](kernel.md) |
| **Command runner** | `CommandRunner` protocol behind `bash`; default local subprocess runner with timeout, process-group kill, output caps, env allowlist | `runner.py` | [kernel.md#command-runner](kernel.md#command-runner) |
| **Tool catalog** | Deferred loading behind a client-side `tool_search` (BM25 / regex), tiering, server-side tool-search definitions | `catalog.py` | [catalog.md](catalog.md) |
| **MCP client** | stdio JSON-RPC 2.0 client (`server/discover` → `initialize` fallback, `tools/list`, `tools/call`, `input_required`), adapter into the registry, server-side connector params | `mcp.py` | [mcp.md](mcp.md) |
| **Design hygiene** | Definition-token budgets, description lint, description pinning (rug-pull defence) | `design.py` | [design.md](design.md) |

## Why this shape

The research converges on a few points. This package encodes each one in code, not in prompts.

1. **A small kernel, not an API mirror.** Every leading harness ships a shell, file read/search, one structured edit primitive and a whole-file write; Pi ships four tools, and mini-swe-agent gets >74% on SWE-bench Verified with bash alone. The kernel here is six tools, with Claude Code names available as a profile.
2. **Poka-yoke arguments.** File tools take absolute paths only, because relative paths after `cd` were the classic SWE-bench failure. `edit` fails on zero or multiple matches and says how to fix the call (add surrounding lines). Read-before-overwrite stops blind clobbering.
3. **Errors are tool results.** Unknown tools, schema violations, permission denials, timeouts and handler crashes come back as `is_error: true` results with a corrected example, so the model can self-correct (MCP SEP-1303). After 3 consecutive or 20 total failures the result is flagged for escalation (auto-mode precedent).
4. **Bounded output.** Every result is capped at 25,000 tokens (Claude Code's default). The cap is never silent: the full payload is offloaded to a workspace file and the model gets a preview, the path and a "narrow your query" hint.
5. **Definitions are a context budget.** Past 10 tools or 10K definition tokens, all but 3–5 hot tools are deferred behind `tool_search`. That cuts definition tokens by 85% and raised Opus 4.5 MCP-eval accuracy from 79.5% to 88.1%. The search tool is never deferred, and the tool array is deterministic and frozen per session so the prompt cache survives.
6. **Concurrency classes.** Read-only tools run in parallel and mutating tools run alone, in order (Claude Agent SDK rule). Results always come back in call order, so the loop answers every `tool_use` in ONE user message.
7. **Sandbox the shell, not each tool.** `bash` runs through one `CommandRunner` protocol, so a hardened runner can be swapped in without touching the tools. Keyword-style sandbox runners are adapted automatically. The default local runner is explicitly *not* a sandbox.
8. **MCP is the tool bus, and its servers are untrusted.** MCP tools are namespaced `mcp__server__tool`, deferred, `open_world` (their output taints the session) and annotation-blind unless the server is vetted. Approved descriptions can be pinned so a later "rug pull" is refused.

## Architecture

```text
                 model turn: [tool_use a][tool_use b][tool_use c]
                                     │
                                     ▼
 ToolRegistry.run_tool_uses ─▶ execute_batch: partition by concurrency class
        │                      ┌───────────────┬──────────────┬───────────────┐
        │                      │ read-only ∥   │ mutating     │ read-only ∥   │  (ordered)
        ▼                      └───────────────┴──────────────┴───────────────┘
 execute(call, ctx)  — never raises
   lookup ─▶ unknown? "did you mean…/use tool_search"         ┐
   coerce JSON object ─▶ validate(schema) ─▶ apply defaults   │ any failure ⇒
   PermissionPolicy.check (allow / deny / ask→approver)       │ ToolOutcome(is_error=True,
   run handler in worker thread with per-tool timeout         │ corrected example)
   normalise ─▶ OutputPolicy (25k tokens; offload + preview)  ┘
   open_world? ⇒ metadata.untrusted, ctx.tainted, TaintTracker
   FailureTracker (3 consecutive / 20 total ⇒ escalate) ─▶ ToolCallRecord ─▶ on_record
        │
        ▼
 one user message: [tool_result a][tool_result b, is_error][tool_result c]

 tools behind the registry:
   kernel ── read/write/edit/grep/glob ──▶ Workspace (absolute paths, no escapes, .git protected,
         │                                  read-before-write, Checkpointer, atomic writes)
         └─ bash ──▶ CommandRunner (LocalSubprocessRunner | sandbox package runner)
   catalog ── tool_search (BM25 | regex) ──▶ tool_reference blocks (Anthropic) | revealed definitions
   mcp ── MCPToolset ──▶ MCPClient ──stdio JSON-RPC──▶ server  (deferred, open_world, pinned)
```

## Quick start

```python
from harness_core.types import ModelRequest, RunContext
from harness_tools import (
    MCPServerConfig, MemoryCheckpointer, ToolRegistry, Workspace,
    connect_mcp_servers, enable_tool_search, register_kernel,
)

registry = ToolRegistry()                                   # 25k-token cap, 3/20 escalation
ws = Workspace("/abs/path/to/repo", checkpointer=MemoryCheckpointer())
register_kernel(registry, ws)                               # read/write/edit/bash/grep/glob
enable_tool_search(registry)                                # client-side tool_search (never deferred)
clients = connect_mcp_servers(registry, [MCPServerConfig("tracker", "tracker-mcp", ("--stdio",))])

@registry.tool(read_only=True, defer_loading=True)
def search_customers(query: str, limit: int = 10) -> list[dict]:
    """Search customers by name or email. Returns name, email and plan for each match."""
    ...

registry.freeze()                                           # tool array is now cache-stable
ctx = RunContext(session_id="s_001")
request = ModelRequest(system=SYSTEM, messages=messages, tools=registry.specs())
# ... model call ...
messages.append(registry.run_tool_uses(response, ctx))      # ONE user message, every tool_use answered
```

Tests: `python3 -m pytest -p no:cacheprovider tests/tools` (83 tests, offline). The MCP tests drive a fake stdio server, `tests/tools/tools_fake_mcp_server.py`.
