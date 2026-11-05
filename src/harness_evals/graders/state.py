"""End-state graders: verify what the agent *produced*, not what it said.

Reference §9.7 step 5 and §9.8: grade the outcome ("check whether the reservation exists in the
DB, not whether the agent claimed to book it"), run tests from outside the agent's sandbox, make
graders tolerant of legitimate format variation (CORE-Bench rejected "96.12" against
"96.124991..."), and do not trust exit codes alone (``sys.exit(0)``-style hacks).

All file paths are task-relative and traversal-checked; graders read the *grading copy*, where
hidden tests have already been overlaid.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import sys
from typing import Any, Mapping, Sequence

from ..errors import GraderConfigError
from ..isolation import run_command
from ..task import safe_relpath
from .base import GradeContext, Grader, GraderDeps, GradeResult, common_kwargs, register_grader

_NUMBER = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _read(ctx: GradeContext, rel: str) -> str | None:
    p = ctx.workspace / safe_relpath(rel, what="grader path")
    try:
        p.resolve().relative_to(ctx.workspace.resolve())
    except ValueError:
        return None  # symlink escape: treat as missing
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None


def glob_match(path: str, patterns: Sequence[str]) -> bool:
    """Glob match where ``dir/**`` and ``dir/`` mean "anything under dir"."""
    for pat in patterns:
        pat = pat.rstrip("/") + "/**" if pat.endswith("/") else pat
        if fnmatch.fnmatchcase(path, pat) or (pat.endswith("/**") and (path == pat[:-3] or path.startswith(pat[:-2]))):
            return True
    return False


@register_grader("file_exists")
class FileExistsGrader(Grader):
    kind = "file_exists"

    def __init__(self, path: str, *, exists: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.path = safe_relpath(path, what="file_exists.path")
        self.exists = exists

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "FileExistsGrader":
        return cls(spec["path"], exists=bool(spec.get("exists", True)), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        present = _read(ctx, self.path) is not None
        return self.result(present == self.exists, details={"path": self.path, "present": present})


@register_grader("file_regex")
class FileRegexGrader(Grader):
    kind = "file_regex"

    def __init__(self, path: str, pattern: str, *, must_match: bool = True, ignore_case: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.path = safe_relpath(path, what="file_regex.path")
        try:
            self.regex = re.compile(pattern, re.M | (re.I if ignore_case else 0))
        except re.error as exc:
            raise GraderConfigError(f"file_regex: bad pattern {pattern!r}: {exc}") from exc
        self.must_match = must_match

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "FileRegexGrader":
        return cls(spec["path"], spec["pattern"], must_match=bool(spec.get("must_match", True)), ignore_case=bool(spec.get("ignore_case", False)), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        text = _read(ctx, self.path)
        if text is None:
            return self.result(False, details={"path": self.path, "reason": "file missing"})
        m = self.regex.search(text)
        return self.result(bool(m) == self.must_match, details={"path": self.path, "matched": bool(m)}, evidence=[m.group(0)[:200]] if m else [])


@register_grader("file_equals")
class FileEqualsGrader(Grader):
    kind = "file_equals"

    def __init__(self, path: str, expected: str, *, normalize_whitespace: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        self.path = safe_relpath(path, what="file_equals.path")
        self.expected = expected
        self.normalize = normalize_whitespace

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "FileEqualsGrader":
        return cls(spec["path"], str(spec["content"]), normalize_whitespace=bool(spec.get("normalize_whitespace", True)), **common_kwargs(spec))

    def _norm(self, s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines()) if self.normalize else s

    def grade(self, ctx: GradeContext) -> GradeResult:
        text = _read(ctx, self.path)
        if text is None:
            return self.result(False, details={"path": self.path, "reason": "file missing"})
        return self.result(self._norm(text) == self._norm(self.expected), details={"path": self.path})


def json_pointer(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer (``/a/0/b``); raises ``KeyError`` when absent."""
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        tok = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            if not tok.isdigit() or int(tok) >= len(cur):
                raise KeyError(pointer)
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise KeyError(pointer)
            cur = cur[tok]
        else:
            raise KeyError(pointer)
    return cur


def numbers_close(a: float, b: float, *, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _check_assertion(doc: Any, a: Mapping[str, Any], rel_tol: float, abs_tol: float) -> tuple[bool, str]:
    ptr = str(a.get("pointer", a.get("path", "")))
    try:
        val = json_pointer(doc, ptr)
    except KeyError:
        return (a.get("exists") is False), f"{ptr}: absent"
    if "exists" in a:
        return bool(a["exists"]), f"{ptr}: present"
    if "approx" in a:
        try:
            ok = numbers_close(float(val), float(a["approx"]), rel_tol=float(a.get("rel_tol", rel_tol)), abs_tol=float(a.get("abs_tol", abs_tol)))
        except (TypeError, ValueError):
            ok = False
        return ok, f"{ptr}: {val!r} ~ {a['approx']!r}"
    if "equals" in a:
        return val == a["equals"], f"{ptr}: {val!r} == {a['equals']!r}"
    if "contains" in a:
        return (a["contains"] in val) if isinstance(val, (str, list, dict)) else False, f"{ptr}: contains {a['contains']!r}"
    if "type" in a:
        types = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict, "null": type(None)}
        return isinstance(val, types.get(str(a["type"]), object)), f"{ptr}: type {a['type']}"
    if "regex" in a:
        return bool(re.search(str(a["regex"]), str(val))), f"{ptr}: ~ /{a['regex']}/"
    raise GraderConfigError(f"json_assert: assertion for {ptr!r} has no operator")


@register_grader("json_assert")
class JsonAssertGrader(Grader):
    """Assertions over a JSON file: ``exists``, ``equals``, ``approx`` (with tolerance), ``contains``, ``type``, ``regex``."""

    kind = "json_assert"

    def __init__(self, path: str, assertions: Sequence[Mapping[str, Any]], *, rel_tol: float = 1e-6, abs_tol: float = 1e-9, **kw: Any) -> None:
        super().__init__(**kw)
        if not assertions:
            raise GraderConfigError("json_assert needs at least one assertion")
        self.path = safe_relpath(path, what="json_assert.path")
        self.assertions = [dict(a) for a in assertions]
        self.rel_tol, self.abs_tol = rel_tol, abs_tol

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "JsonAssertGrader":
        return cls(spec["path"], spec["assertions"], rel_tol=float(spec.get("rel_tol", 1e-6)), abs_tol=float(spec.get("abs_tol", 1e-9)), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        text = _read(ctx, self.path)
        if text is None:
            return self.result(False, 0.0, details={"path": self.path, "reason": "file missing"})
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            return self.result(False, 0.0, details={"path": self.path, "reason": f"invalid JSON: {exc}"})
        outcomes = [_check_assertion(doc, a, self.rel_tol, self.abs_tol) for a in self.assertions]
        n_ok = sum(ok for ok, _ in outcomes)
        return self.result(n_ok == len(outcomes), n_ok / len(outcomes), details={"checks": [{"ok": ok, "check": c} for ok, c in outcomes]})


def parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s.strip()) else s.strip()
    try:
        return float(s)
    except ValueError:
        return None


@register_grader("answer")
class AnswerGrader(Grader):
    """Check the final answer (or a file) with format tolerance.

    ``match``: ``numeric`` (default tolerance rel 1e-3 so "96.12" matches "96.124991"), ``exact``
    (whitespace/case-normalised), ``contains`` or ``regex``. ``numeric_pick`` picks which number in
    the text is the answer: ``last`` (default), ``only`` (exactly one number) or ``any``.
    """

    kind = "answer"

    def __init__(self, expected: Any, *, match: str = "numeric", source: str = "final_text", rel_tol: float = 1e-3, abs_tol: float = 1e-9,
                 numeric_pick: str = "last", case_sensitive: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        if match not in ("numeric", "exact", "contains", "regex"):
            raise GraderConfigError(f"answer.match must be numeric|exact|contains|regex, got {match!r}")
        if numeric_pick not in ("last", "only", "any"):
            raise GraderConfigError("answer.numeric_pick must be last|only|any")
        if source != "final_text" and not source.startswith("file:"):
            raise GraderConfigError("answer.source must be 'final_text' or 'file:<path>'")
        self.expected, self.match, self.source = expected, match, source
        self.rel_tol, self.abs_tol, self.pick, self.case_sensitive = rel_tol, abs_tol, numeric_pick, case_sensitive

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "AnswerGrader":
        keys = ("match", "source", "rel_tol", "abs_tol", "numeric_pick", "case_sensitive")
        return cls(spec["expected"], **{k: spec[k] for k in keys if k in spec}, **common_kwargs(spec))

    def _text(self, ctx: GradeContext) -> str | None:
        if self.source == "final_text":
            return ctx.trajectory.final_text
        return _read(ctx, self.source[5:])

    def grade(self, ctx: GradeContext) -> GradeResult:
        text = self._text(ctx)
        if text is None:
            return self.result(False, details={"reason": f"{self.source} missing"})
        if self.match == "numeric":
            exp = float(self.expected)
            found = [n for n in (parse_number(m.group(0)) for m in _NUMBER.finditer(text)) if n is not None]
            if not found or (self.pick == "only" and len(found) != 1):
                return self.result(False, details={"reason": "no unambiguous number", "found": found[:10]})
            cands = found if self.pick == "any" else [found[-1]]
            ok = any(numbers_close(v, exp, rel_tol=self.rel_tol, abs_tol=self.abs_tol) for v in cands)
            return self.result(ok, details={"expected": exp, "got": cands[-1] if cands else None})
        norm = (lambda s: " ".join(str(s).split())) if self.case_sensitive else (lambda s: " ".join(str(s).split()).lower())
        if self.match == "exact":
            ok = norm(text) == norm(self.expected)
        elif self.match == "contains":
            ok = norm(self.expected) in norm(text)
        else:
            ok = re.search(str(self.expected), text, 0 if self.case_sensitive else re.I) is not None
        return self.result(ok, details={"match": self.match})


@register_grader("command", "state_check")
class CommandGrader(Grader):
    """Run a check command in the grading copy (outside the agent sandbox).

    Exit codes alone are hackable, so ``require_output`` (regex that must appear) and
    ``forbid_output`` are available; prefer :class:`UnitTestsGrader` for test suites.
    """

    kind = "command"

    def __init__(self, command: str | Sequence[str], *, expect_exit: int = 0, require_output: str | None = None, forbid_output: str | None = None,
                 timeout_s: float | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        if not command:
            raise GraderConfigError("command grader needs a command")
        self.command = command if isinstance(command, str) else [str(c) for c in command]
        self.expect_exit = int(expect_exit)
        self.require = re.compile(require_output, re.M) if require_output else None
        self.forbid = re.compile(forbid_output, re.M) if forbid_output else None
        self.timeout_s = timeout_s

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "CommandGrader":
        cmd = spec.get("command", spec.get("check"))
        return cls(cmd, expect_exit=int(spec.get("expect_exit", 0)), require_output=spec.get("require_output"), forbid_output=spec.get("forbid_output"),
                   timeout_s=spec.get("timeout_s"), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        r = run_command(self.command, cwd=ctx.workspace, env=ctx.env, timeout_s=float(self.timeout_s or ctx.timeout_s))
        reasons = []
        if r.timed_out:
            reasons.append("timed out")
        if r.exit_code != self.expect_exit:
            reasons.append(f"exit {r.exit_code} != {self.expect_exit}")
        if self.require and not self.require.search(r.output):
            reasons.append("required output missing")
        if self.forbid and self.forbid.search(r.output):
            reasons.append("forbidden output present")
        return self.result(not reasons, details={"exit_code": r.exit_code, "reasons": reasons, "duration_s": round(r.duration_s, 3), "output_tail": r.output[-2000:]})


_PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.M)


@register_grader("unit_tests")
class UnitTestsGrader(Grader):
    """SWE-bench-style fail-to-pass / pass-to-pass grading via ``pytest -rA``.

    Every ``fail_to_pass`` id must be reported ``PASSED``; every ``pass_to_pass`` entry (an id or a
    path prefix) must have at least one ``PASSED`` and no ``FAILED``/``ERROR``. Because verdicts
    come from per-test report lines, an exit code of 0 without those lines (``sys.exit(0)`` in a
    conftest, collection sabotage) is a failure. Score = fraction of fail-to-pass ids passing.
    """

    kind = "unit_tests"

    def __init__(self, fail_to_pass: Sequence[str] = (), pass_to_pass: Sequence[str] = (), *, command: Sequence[str] | None = None,
                 timeout_s: float | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        if not fail_to_pass and not pass_to_pass:
            raise GraderConfigError("unit_tests needs fail_to_pass and/or pass_to_pass")
        self.f2p = [str(t) for t in fail_to_pass]
        self.p2p = [str(t) for t in pass_to_pass]
        self.command = list(command) if command else [sys.executable, "-m", "pytest", "-rA", "-q", "-p", "no:cacheprovider"]
        self.timeout_s = timeout_s

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "UnitTestsGrader":
        return cls(spec.get("fail_to_pass") or (), spec.get("pass_to_pass") or (), command=spec.get("command"), timeout_s=spec.get("timeout_s"), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        targets = sorted({t.split("::")[0] for t in self.f2p} | {t.split("::")[0] for t in self.p2p})
        r = run_command([*self.command, *targets], cwd=ctx.workspace, env=ctx.env, timeout_s=float(self.timeout_s or ctx.timeout_s))
        status: dict[str, str] = {}
        for m in _PYTEST_LINE.finditer(r.output):
            status[m.group(2)] = m.group(1)
        f2p_ok = [t for t in self.f2p if status.get(t) == "PASSED"]
        p2p_bad = []
        for prefix in self.p2p:
            under = {k: v for k, v in status.items() if k == prefix or k.startswith(prefix.rstrip("/") + "/") or k.startswith(prefix + "::")}
            if not any(v == "PASSED" for v in under.values()) or any(v in ("FAILED", "ERROR") for v in under.values()):
                p2p_bad.append(prefix)
        passed = len(f2p_ok) == len(self.f2p) and not p2p_bad and not r.timed_out
        details = {
            "exit_code": r.exit_code, "fail_to_pass_passed": f2p_ok, "fail_to_pass_missing": [t for t in self.f2p if t not in f2p_ok],
            "pass_to_pass_broken": p2p_bad, "timed_out": r.timed_out, "output_tail": r.output[-2000:],
        }
        if r.exit_code == 0 and not status:
            details["suspicious"] = "exit 0 without per-test report lines (possible sys.exit/collection hack)"
            passed = False
        score = (len(f2p_ok) / len(self.f2p)) if self.f2p else (1.0 if passed else 0.0)
        if p2p_bad:
            score = min(score, 0.5) if self.f2p else 0.0
        return self.result(passed, score, details=details)


@register_grader("diff_scope")
class DiffScopeGrader(Grader):
    """Constrain *where* the agent changed files (e.g. "touches only src/tz/*")."""

    kind = "diff_scope"

    def __init__(self, *, allowed: Sequence[str] = (), forbidden: Sequence[str] = (), require_change: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        if not allowed and not forbidden and not require_change:
            raise GraderConfigError("diff_scope needs allowed, forbidden or require_change")
        self.allowed, self.forbidden, self.require_change = list(allowed), list(forbidden), require_change

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "DiffScopeGrader":
        return cls(allowed=spec.get("allowed") or (), forbidden=spec.get("forbidden") or (), require_change=bool(spec.get("require_change", False)), **common_kwargs(spec))

    def grade(self, ctx: GradeContext) -> GradeResult:
        touched = ctx.diff.touched
        out_of_scope = [p for p in touched if self.allowed and not glob_match(p, self.allowed)]
        forbidden = [p for p in touched if glob_match(p, self.forbidden)]
        ok = not out_of_scope and not forbidden and (bool(touched) or not self.require_change)
        return self.result(ok, details={"touched": list(touched), "out_of_scope": out_of_scope, "forbidden": forbidden})


__all__ = [
    "AnswerGrader",
    "CommandGrader",
    "DiffScopeGrader",
    "FileEqualsGrader",
    "FileExistsGrader",
    "FileRegexGrader",
    "JsonAssertGrader",
    "UnitTestsGrader",
    "glob_match",
    "json_pointer",
    "numbers_close",
]
