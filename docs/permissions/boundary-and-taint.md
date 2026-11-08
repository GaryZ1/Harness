# Boundary classification and session taint

Modules: `boundary.py` (`classify`, `ActionProfile`, `Crossing`, `ToolKind`), `paths.py` (`Workspace`, `PathPattern`, `canonicalize`), `network.py` (`host_from_url`, `is_private_host`, `domain_matches`) and `taint.py` (`SessionTaint`, `TaintSource`, `sources_for`).

## Purpose

"Auto-allow inside the boundary, classify every crossing." `classify(call, spec, ctx, workspace, config)` is the checklist's `normalize(call)` step. It is pure and builds an `ActionProfile` with:

* the tool kind;
* every touched path, canonicalised;
* every network host, normalised;
* the parsed shell command, with a profile per sub-command;
* the MCP server and trusted annotations;
* a map of `Crossing → reasons`.

Session taint then decides which crossings need a human.

## Research basis

* Sandboxing "safely reduces permission prompts by 84%". The boundary is the auto-mode *Environment* slot, by default the current git repo.
* Hard read-deny of `~/.ssh` and `**/.env`. `.git/hooks` and agent config dirs stay read-only even under a writable root (Codex `read_only_subpaths`). "Private/LAN destinations are always blocked."
* The lethal trifecta (private data, untrusted content, external communication): "once a session ingests untrusted or open-world content, tighten policy for actions with exfiltration potential" (technique 22). `openWorldHint` defaults to `true`. Taint "should never auto-lower restrictions. Clearing taint requires a new session or explicit user action."
* "Mark remote side effects as non-rewindable and gate them *before* execution" (checklist step 13). Checkpoints "cannot undo remote side effects".

## Crossings

| Group (auto-mode block group) | Crossing | Raised by (examples) |
|---|---|---|
| cross trust boundaries | `read_outside_workspace`, `write_outside_workspace` | canonical path outside `roots` (+ `read_roots`, `temp_roots`); unresolvable paths (`$FILE`, relative after a dynamic `cd`) |
| | `network_egress`, `new_egress_domain` | any destination not on `network.allowed_domains` or approved for the session; destination unknown |
| | `external_tool` | every MCP tool |
| | `subagent_spawn` | `Agent` / `Task` |
| destroy or exfiltrate | `credential_access` **(hard deny)** | built-in credential paths (`~/.ssh/**`, `~/.aws/**`, `**/.env`, `**/*.pem`, `/proc/*/environ`…) plus `workspace.credential_paths`; `.env.example` and similar are exempt |
| | `environment_dump` | bare `env`, `printenv`, `set`, `export` |
| | `secret_in_input` | the input contains a credential-like string (`harness_memory.SecretRedactor` patterns) |
| | `network_write` | `curl -d/-F/-T/-X POST`, `wget --post-*`, httpie items, `nc`/`ssh`/`scp`/`rsync host:`, `gh api -X POST`, message tools |
| | `message_send` | mail CLIs, `gh … comment/create/review`, tools or MCP tools named send/post/reply/email/message… |
| | `irreversible` | `rm -r`, `shred`, `truncate`, `find -delete`, `git reset --hard`, `git clean -f`, `git checkout -- .`, `branch -D`, `stash drop`, force/delete pushes, `gh pr merge`, `crontab -r`, destructive (or unvetted) MCP tools |
| | `catastrophic` **(hard deny)** | recursive delete of `/`, `~`, system dirs, or a workspace root or its ancestor; `mkfs*`, `fdisk`, `wipefs`; `dd of=/dev/…`; fork bomb |
| | `private_network` **(hard deny)** | loopback, RFC 1918, link-local (incl. `169.254.169.254`), CGNAT, ULA, `*.localhost`/`*.internal`/`*.local`, metadata names; numeric IPv4 forms (`2130706433`, `0x7f.1`) |
| bypass review / affect others | `git_push_session_branch`, `git_push_non_session_branch` | `git push`, classified against `workspace.session_branch` (or `ctx.metadata["session_branch"]`) |
| | `deploy` | `kubectl apply/delete…`, `helm install/upgrade`, `terraform apply/destroy`, `npm/twine/cargo publish`, `docker push`, `fly/vercel/netlify/firebase deploy`, cloud CLI write verbs |
| | `memory_write` | memory tool commands other than `view` (boundary-neutral; matters only when tainted) |
| degrade security posture | `protected_path` **(hard deny)** | writes to `.git/**`, `.harness/**`, `.claude/settings*.json`, `.claude/hooks/**`, `.codex/**`, `harness-policy.*`, `~/.harness/**`, `~/.codex/**`, `/etc/harness/**`, every loaded policy file, `workspace.protected_paths` |
| | `privilege_escalation` | `sudo`/`doas`/`su`/`pkexec`, `chown root` |
| | `degrade_security_posture` | `iptables`, `ufw`, `setenforce`, `chmod 777`/setuid, `git config --global` or code-running keys, erasing history |
| | `self_harm` **(hard deny)** | `pkill bash`, `kill -9 -1`, `kill $$`, `killall python`, `shutdown`/`reboot`. "One agent ran `pkill -9 bash` and killed its own loop." |
| not statically analysable | `opaque_execution` | see [shell-analysis.md](shell-analysis.md) |
| | `code_execution` | interpreters running scripts, `source`, package installs, programs invoked by a non-system path (`./git`) |
| | `dangerous_env` | `PATH`, `LD_*`, `DYLD_*`, `PYTHONPATH`, `NODE_OPTIONS`, `BASH_ENV`, `GIT_*`, `PAGER`, `EDITOR`… assignments |
| | `unknown_effects`, `suspicious_input` | unknown tools without `read_only`; invisible or look-alike characters |

`HARD_DENY`, `OUTWARD`, `BOUNDARY_NEUTRAL` and `WILDCARD_UNCOVERABLE` are exported sets.

## Paths

`Workspace(roots, read_roots, temp_roots=["/tmp"], protected_paths=…, credential_paths=…)`. `Workspace.from_cwd()` picks the enclosing git repository. Every path is canonicalised with `realpath` against the call's `cwd` (also tracked across `cd`/`pushd` inside a command), so `src/../../etc/passwd` and symlinks pointing outside resolve to where they really go. Glob patterns (`*.py`, `{a,b}`) are reduced to their literal directory prefix. Protected and credential matching is fuzzy: case-folded, NFKC and invisible characters stripped.

## Network

`host_from_url()` treats backslashes as slashes and drops userinfo, so `https://github.com@evil.example/` → `evil.example` and `https://evil.example\@github.com` → `evil.example`. Hosts are lower-cased, trailing dots are stripped and hosts are IDNA-encoded, so a Cyrillic `gіthub.com` never matches `github.com`. `domain_matches("a.b.com", "*.b.com")` matches strict subdomains, and `b.com` matches only itself. DNS rebinding (a public name resolving to a private IP) can only be caught by the egress proxy, which resolves names. That stays in the sandbox layer.

## Taint

```python
out = engine.record_result(call, spec, ctx, outcome)   # marks taint, probes, annotates; sets ctx.tainted
engine.session(ctx).taint.tainted, engine.session(ctx).taint.reasons()
engine.clear_taint(sid, confirmed_by_user=True, ctx=ctx)   # explicit user action only (TaintClearRefused otherwise)
```

| Source (`taint.sources`) | Fires on |
|---|---|
| `web_fetch` | WebFetch / WebSearch; shell commands with network crossings (`curl`) |
| `browser` | browser tools |
| `mcp:openWorld` | MCP tools whose effective `openWorldHint` is true (always, for unvetted servers) |
| `email` | tools whose name mentions mail / inbox / imap / smtp / outlook |
| `out_of_repo_files` | reads outside the workspace |
| `subagent_report` | sub-agent results, `subagent_finished()` |
| `memory_read` (off by default) | memory `view` |
| always on | `ToolSpec.open_world`, `outcome.metadata["untrusted"]`, an injection-probe hit, an inherited parent taint, `RunContext.tainted` |

While a session is tainted, any crossing in `taint.strict_actions` gets layer `L6_taint`. The defaults are the schema's `network_write`, `message_send`, `git_push_non_session_branch`, `new_egress_domain` and `memory_write`, plus our `deploy` and `irreversible`. These actions **ask a human even if an allow rule covers them**. With `taint.on_strict = "strict_review"` the stage-1-only reviewer may approve instead, except for `new_egress_domain`, which is always human-only.

## Defaults

| Setting | Default | Source |
|---|---|---|
| Boundary | enclosing git repo | vendor (auto-mode Environment) |
| `temp_roots` | `["/tmp"]` | vendor (Codex workspace-write) |
| Private destinations | hard deny | vendor (Codex network proxy / srt) |
| Taint sources | schema list | research schema |
| Strict actions | schema list + `deploy`, `irreversible` | schema + our choice (task: outward/irreversible actions ask when tainted) |
| `on_strict` | `ask` | our choice (checklist allows strict classifier) |

## Invariants

* Hard-deny crossings are denied in every mode.
* Paths are compared only after canonicalisation. Fuzzy matching is used only on the deny side.
* Taint is monotonic within a session. It is inherited by children and propagated from children to the parent.
* Taint never relaxes a decision.

## Failure modes

* A path that cannot be resolved (variables, unknown cwd) counts as **outside** the workspace.
* Unknown tools, and MCP annotations from unvetted servers, are treated as having side effects (`destructiveHint` assumed true).
