# Shell analyzer

Module: `shell.py`. Entry point: `analyze(command) -> ShellAnalysis`. It never raises, never executes anything and never expands variables.

## Purpose

This module turns a shell string into **every simple command that would run**, with its argv (quotes removed), its leading `NAME=value` assignments, its redirections, the wrappers peeled off it, where it came from (`top`, `subshell`, `command_substitution`, `process_substitution`, `bash -c`, `eval`, `su -c`, `env -S`, `find -exec`, `watch`), and whether it can be resolved statically. The policy engine requires *every* command to be allowed. Deny rules are checked against each command, and boundary classification runs per command.

## Research basis

* Prefix rules "cannot see through `bash -c`, `eval`, pipes into interpreters, or scripts written to disk" (§7.3 technique 5).
* Checklist step 5 says: "Parse compound commands (pipes, `&&`, `bash -c`, subshells) with a shell AST parser, as OpenHands does, and evaluate every segment; if parsing fails, fall through to the classifier or the human."
* The basename-only pitfall: "a repo-local shim named `git` matches 'allow git'". The analyzer keeps `executable_path` so policy can tell `./git` from `git`.

## API

```python
from harness_permissions import analyze

a = analyze("FOO=1 sudo -u root bash -c 'cd /tmp && rm -rf x' | tee log && echo $(id)")
[c.argv for c in a.commands]
# [('bash', '-c', 'cd /tmp && rm -rf x'), ('cd', '/tmp'), ('rm', '-rf', 'x'), ('tee', 'log'), ('echo', '$(id)'), ('id',)]
a.commands[0].wrappers, a.commands[0].env[0][0], a.commands[0].transparent   # (['sudo'], 'FOO', True)
a.commands[3].piped_input                                                     # True
a.errors, a.opaque_reasons, a.suspicious, a.fully_resolved
```

`SimpleCommand` properties: `argv`, `executable` (basename), `executable_path` (when invoked by path), `text` (space-joined argv, which rules match against), `privileged` (sudo/doas/su/pkexec/run0), `dynamic_command`, `code_execution`, `dynamic_args` (xargs, find -exec) and `transparent` (a wrapper whose payload was parsed: `sh -c`, `eval`, `su -c`, `env -S`, `watch`).

## What is handled

| Construct | Handling |
|---|---|
| `&&` `\|\|` `;` `\|` `\|&` `&` newline | split; `piped_input` recorded; a dangling operator is an error |
| `( … )`, `{ …; }`, `if/then/else/fi`, `while/until/do/done` | parsed recursively; reserved words skipped |
| `$( … )`, backticks (also inside `"…"` and unquoted heredocs), `$(( … ))` | inner commands parsed recursively |
| `<( … )`, `>( … )` | parsed recursively |
| `'…'`, `"…"`, `$'…'` (with `\xHH`, octal and `\u` decoding), `\x` escapes | removed, so `'r'm`, `\rm`, `r''m` and `$'\x72m'` all become `rm` |
| `>`, `>>`, `&>`, `>\|`, `<>`, `<`, `N>&M`, `<<EOF` (quoted delimiter disables substitution), `<<<` | recorded; `Redirect.writes` excludes fd duplication and `/dev/null` |
| `NAME=value cmd` | moved to `env` |
| `sudo`/`doas`/`pkexec`/`run0` (options skipped), `su -c`, `env [-i] [-u] [VAR=…] [-S str]`, `nohup`, `time`, `nice`, `ionice`, `timeout`, `stdbuf`, `chrt`, `taskset`, `command`, `builtin`, `exec`, `unbuffer`, `xargs`, `watch` | peeled; the inner command is analysed |
| `sh`/`bash`/`zsh`/… `-c STR` (also `-lc`, `-ec`), `eval STR`, `find … -exec/-execdir/-ok … ;` | payload parsed recursively |
| `# comments`, `\` line continuation | skipped |

## Opaque execution and errors

`opaque_reasons` (statically unresolvable, so never rule-allowed or boundary-allowed) covers:

* dynamic command names (`$CMD`);
* `eval`, `sh -c`, `su -c` or `env -S` with a dynamic string;
* piping into a shell or interpreter (`curl … | sh`, `base64 -d | bash`);
* inline interpreter code (`python -c`, `node -e`, `perl -e`, `ruby -e`, `php -r`, `osascript -e`, `pwsh -c`…);
* `git -c` / `--config-env`, which can set `core.sshCommand`, `alias.*` or `core.pager` to run code;
* `sudo -i` / `-s` and `su` without `-c` (interactive shells).

`errors` (unparsable, so the engine treats the command as opaque, and deny rules fall back to a raw-substring check) covers unterminated quotes or substitutions, unbalanced parentheses, `case`/`function`/`select`/`coproc`, function definitions, dangling operators, nesting deeper than 8 levels, commands over 100,000 characters, NUL bytes, and fork bombs (also a hard deny).

`suspicious` lists zero-width or bidi characters, NFKC look-alikes (full-width `ｒｍ`), carriage returns and terminal escapes. The engine maps these to the `suspicious_input` crossing, which no rule can cover.

## Defaults

| Setting | Value | Source |
|---|---|---|
| `MAX_NESTING` | 8 | our choice |
| `MAX_COMMAND_CHARS` | 100,000 | our choice |
| Unsupported grammar | error, so the reviewer or a human decides | research ("if parsing fails, fall through") |

## Invariants

* Every command the shell would execute appears in `commands`, including those inside substitutions, subshells, heredocs, `-c` payloads, `eval` strings and `find -exec`. Anything the analyzer cannot see is reported in `opaque_reasons` or `errors`, never silently dropped.
* Wrappers propagate into nested payloads (`sudo bash -c 'x'` marks `x` as privileged).
* Single quotes and quoted heredoc delimiters suppress substitution, exactly as in bash.

## Failure modes and limits

* The analyzer is a conservative subset of bash. Unusual but legal syntax becomes an error, which fails *closed* to the reviewer or a human.
* Commands written to disk and run later (`echo … > x.sh; bash x.sh`) are flagged as `code_execution` but cannot be inspected. The OS sandbox is the backstop.
* Aliases, functions defined elsewhere and `PATH` lookups are not resolved. That is why dangerous env assignments (`PATH=`, `LD_PRELOAD=`…) are uncoverable crossings and path-invoked binaries need pinning.
