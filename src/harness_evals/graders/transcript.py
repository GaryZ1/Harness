"""Transcript graders: constraints on the path, used only where policy requires them.

Anthropic calls rigid step-by-step trajectory grading "overly brittle" (reference §9.8), so these
graders check *bounded* properties - turn and tool-call budgets, required / forbidden tools,
forbidden command patterns (``rm -rf``), argument constraints such as ``refund.amount <= 100``
(Anthropic's conversational example) - not the exact sequence of steps.
"""

from __future__ import annotations

import json
import operator
import re
from typing import Any, Mapping, Sequence

from ..errors import GraderConfigError
from .base import GradeContext, Grader, GraderDeps, GradeResult, common_kwargs, register_grader
from .state import json_pointer

_OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq, "!=": operator.ne}


@register_grader("transcript", "tool_calls")
class TranscriptGrader(Grader):
    kind = "transcript"

    def __init__(
        self,
        *,
        max_turns: int | None = None,
        max_tool_calls: int | None = None,
        required_tools: Sequence[str] = (),
        forbidden_tools: Sequence[str] = (),
        forbid: Sequence[str] = (),
        tool_constraints: Sequence[Mapping[str, Any]] = (),
        final_regex: str | None = None,
        max_tool_errors: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.max_turns, self.max_tool_calls, self.max_tool_errors = max_turns, max_tool_calls, max_tool_errors
        self.required, self.forbidden_tools = list(required_tools), set(forbidden_tools)
        try:
            self.forbid = [re.compile(p[3:]) if p.startswith("re:") else re.compile(re.escape(p)) for p in forbid]
            self.final_regex = re.compile(final_regex, re.S) if final_regex else None
        except re.error as exc:
            raise GraderConfigError(f"transcript: bad pattern: {exc}") from exc
        for c in tool_constraints:
            if "tool" not in c or c.get("op") not in _OPS or "field" not in c or "value" not in c:
                raise GraderConfigError(f"transcript.tool_constraints entry needs tool/field/op/value with op in {sorted(_OPS)}: {dict(c)}")
        self.constraints = [dict(c) for c in tool_constraints]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], deps: GraderDeps) -> "TranscriptGrader":
        constraints = list(spec.get("tool_constraints") or [])
        # Anthropic YAML shorthand: {kind: tool_calls, refund: {amount: "<=100"}}
        for tool, fields in spec.items():
            if tool in ("kind", "name", "weight", "required", "max_turns", "max_tool_calls", "required_tools", "forbidden_tools", "forbid",
                        "tool_constraints", "final_regex", "max_tool_errors", "dimension"):
                continue
            if isinstance(fields, Mapping):
                for fld, expr in fields.items():
                    m = re.match(r"^\s*(<=|>=|==|!=|<|>)\s*(.+)$", str(expr))
                    if not m:
                        raise GraderConfigError(f"tool_calls shorthand {tool}.{fld}: expected '<op><value>', got {expr!r}")
                    constraints.append({"tool": tool, "field": fld, "op": m.group(1), "value": _coerce(m.group(2))})
        forbid = spec.get("forbid") or ()
        return cls(
            max_turns=spec.get("max_turns"), max_tool_calls=spec.get("max_tool_calls"), required_tools=spec.get("required_tools") or (),
            forbidden_tools=spec.get("forbidden_tools") or (), forbid=[forbid] if isinstance(forbid, str) else forbid,
            tool_constraints=constraints, final_regex=spec.get("final_regex"), max_tool_errors=spec.get("max_tool_errors"), **common_kwargs(spec),
        )

    def grade(self, ctx: GradeContext) -> GradeResult:
        t = ctx.trajectory
        violations: list[str] = []
        if self.max_turns is not None and (t.turns or 0) > self.max_turns:
            violations.append(f"turns {t.turns} > {self.max_turns}")
        if self.max_tool_calls is not None and t.n_tool_calls > self.max_tool_calls:
            violations.append(f"tool calls {t.n_tool_calls} > {self.max_tool_calls}")
        if self.max_tool_errors is not None and sum(t.tool_errors_by_tool().values()) > self.max_tool_errors:
            violations.append(f"tool errors > {self.max_tool_errors}")
        used = {c.name for c in t.tool_calls}
        violations += [f"required tool not used: {n}" for n in self.required if n not in used]
        violations += [f"forbidden tool used: {n}" for n in sorted(used & self.forbidden_tools)]
        evidence: list[str] = []
        for c in t.tool_calls:
            blob = json.dumps(c.input, ensure_ascii=False, sort_keys=True)
            for rx in self.forbid:
                if rx.search(blob):
                    violations.append(f"forbidden pattern {rx.pattern!r} in {c.name} call")
                    evidence.append(blob[:200])
            for con in self.constraints:
                if con["tool"] != c.name:
                    continue
                try:
                    val = json_pointer(c.input, "/" + str(con["field"]).replace(".", "/"))
                    ok = _OPS[con["op"]](_coerce(val), con["value"])
                except (KeyError, TypeError):
                    ok = False
                if not ok:
                    violations.append(f"{c.name}.{con['field']} violates {con['op']} {con['value']!r}")
                    evidence.append(blob[:200])
        if self.final_regex and not self.final_regex.search(t.final_text or ""):
            violations.append("final message does not match final_regex")
        return self.result(not violations, details={"violations": violations, "turns": t.turns, "tool_calls": t.n_tool_calls}, evidence=evidence)


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().strip("'\"")
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    return v


__all__ = ["TranscriptGrader"]
