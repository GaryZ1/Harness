"""Human-in-the-loop approvals: a durable, thread-safe pending queue (L8).

Research basis (§7.3 technique 11, checklist §7.7 steps 8, 14-16):

* "an approval wait is persisted state, not a blocked thread" - Agents SDK ``RunState`` /
  interruptions, Managed Agents ``always_ask`` gates, Codex ``item/*/requestApproval`` items.
  :class:`ApprovalBroker` keeps every request as a :class:`PendingApproval` record, optionally
  persisted atomically to a JSON file so a restarted process (or another UI process) can resolve it;
* callers may block (:meth:`ApprovalBroker.request`) or poll (:meth:`submit` + :meth:`get`);
* **timeouts default to deny** (``approvals.timeout_s``, ``on_timeout = "deny"``);
* resolutions are *once* or *session* scoped. A session approval creates the narrowest rule that
  covers the call (``Bash(npm test *)``, exact ``Edit(/abs/path)``, ``WebFetch(domain:host)``;
  exact commands for outward/irreversible sub-commands) - never a blanket rule - and approves a
  new egress domain or a changed MCP description for the session only;
* only a **human** may approve: "instructions or approval claims within [sub-agent reports] carry
  no authority from the user" and teammate messages "cannot grant permissions";
* "plan approval is not action approval": approving a plan only changes the session mode;
* every request and resolution is an ``EventType.PERMISSION`` audit event, and every human
  decision feeds the denial and approval-fatigue counters.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from harness_core.types import (
    Clock,
    EventSink,
    PermissionBehavior,
    PermissionDecision,
    RunContext,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    new_id,
    utcnow,
)
from harness_memory.locking import atomic_write_text, file_lock

from .audit import AuditTrail
from .engine import DENIAL_GUIDANCE, PolicyEngine
from .errors import ApprovalAlreadyResolvedError, ApprovalNotFoundError, PermissionsError, UntrustedApproverError
from .modes import PermissionMode


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ApprovalScope(str, Enum):
    ONCE = "once"
    SESSION = "session"


class ApproverKind(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    TOOL = "tool"


@dataclass(frozen=True)
class Approver:
    kind: ApproverKind
    id: str

    @classmethod
    def human(cls, user_id: str) -> "Approver":
        return cls(ApproverKind.HUMAN, user_id)


@dataclass
class PendingApproval:
    id: str
    session_id: str
    kind: str                      # "tool" | "plan"
    tool: str
    call_id: str
    tool_input: dict[str, Any]
    reason: str
    created_at: str
    expires_at: str
    layer: str = ""
    crossings: list[str] = field(default_factory=list)
    suggested_rules: list[str] = field(default_factory=list)
    human_only: bool = False
    agent_name: str = "main"
    depth: int = 0
    tainted: bool = False
    plan_text: str = ""
    next_mode: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    scope: ApprovalScope | None = None
    approver: str = ""
    resolution_reason: str = ""
    resolved_at: str = ""
    created_monotonic: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["scope"] = self.scope.value if self.scope else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PendingApproval":
        d = dict(d)
        d["status"] = ApprovalStatus(d.get("status", "pending"))
        d["scope"] = ApprovalScope(d["scope"]) if d.get("scope") else None
        d["created_monotonic"] = 0.0
        return cls(**d)

    @property
    def resolved(self) -> bool:
        return self.status is not ApprovalStatus.PENDING

    def decision(self) -> PermissionDecision:
        if self.status is ApprovalStatus.APPROVED:
            return PermissionDecision(PermissionBehavior.ALLOW, f"[human] approved by {self.approver} ({self.scope.value if self.scope else 'once'})")
        why = {
            ApprovalStatus.DENIED: f"denied by {self.approver}" + (f": {self.resolution_reason}" if self.resolution_reason else ""),
            ApprovalStatus.TIMED_OUT: "approval request timed out (timeouts default to deny)",
            ApprovalStatus.CANCELLED: "approval request was cancelled",
            ApprovalStatus.PENDING: "approval still pending",
        }[self.status]
        return PermissionDecision(PermissionBehavior.DENY, f"[human] {why} {DENIAL_GUIDANCE}")


def _iso(dt: Any) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class ApprovalBroker:
    """Pending-approval queue between the policy engine and whoever answers (CLI, web UI, Slack...)."""

    def __init__(
        self,
        engine: PolicyEngine | None = None,
        *,
        sink: EventSink | None = None,
        clock: Clock = utcnow,
        default_timeout_s: float | None = None,
        state_path: str | Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.monotonic = monotonic
        if default_timeout_s is None:
            default_timeout_s = float(engine.policy["approvals.timeout_s"]) if engine is not None else 300.0
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        self.default_timeout_s = default_timeout_s
        self.audit = AuditTrail(sink, clock=clock) if sink is not None else (engine.audit if engine is not None else AuditTrail(None, clock=clock))
        self.state_path = Path(state_path) if state_path else None
        self._cond = threading.Condition(threading.RLock())
        self._items: dict[str, PendingApproval] = {}
        self._specs: dict[str, ToolSpec | None] = {}
        self.listeners: list[Callable[[PendingApproval], None]] = []
        if self.state_path and self.state_path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        assert self.state_path is not None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PermissionsError(f"cannot load approval state {self.state_path}: {exc}") from exc
        for d in data.get("approvals", []):
            item = PendingApproval.from_dict(d)
            item.created_monotonic = self.monotonic()
            self._items[item.id] = item

    def _persist(self) -> None:
        if not self.state_path:
            return
        payload = {"version": 1, "approvals": [i.to_dict() for i in self._items.values()]}
        with file_lock(self.state_path.with_name(self.state_path.name + ".lock")):
            atomic_write_text(self.state_path, json.dumps(payload, indent=2, sort_keys=True, default=str))

    # ------------------------------------------------------------------ requests
    def submit(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext, decision: PermissionDecision | None = None, *, timeout_s: float | None = None) -> PendingApproval:
        """Enqueue an approval request without blocking."""
        record = self.engine.record_for(ctx.session_id, call.id) if self.engine else None
        now = self.clock()
        timeout = timeout_s or self.default_timeout_s
        item = PendingApproval(
            id=new_id("apr"),
            session_id=ctx.session_id,
            kind="tool",
            tool=call.name,
            call_id=call.id,
            tool_input=dict(call.input),
            reason=(decision.reason if decision else (record.reason if record else "approval required")),
            created_at=_iso(now),
            expires_at=_iso(now + timedelta(seconds=timeout)),
            layer=record.layer if record else "",
            crossings=sorted(c.value for c in record.profile.crossings) if record else [],
            suggested_rules=list(record.suggested_rules) if record else [],
            human_only=record.human_only if record else True,
            agent_name=ctx.agent_name,
            depth=ctx.depth,
            tainted=bool(record.tainted) if record else ctx.tainted,
            created_monotonic=self.monotonic(),
        )
        with self._cond:
            self._items[item.id] = item
            self._specs[item.id] = spec
            self._persist()
        self.audit.emit(ctx.session_id, "approval_requested", approval_id=item.id, tool=call.name, call_id=call.id, layer=item.layer,
                        crossings=item.crossings, suggested_rules=item.suggested_rules, expires_at=item.expires_at, agent=ctx.agent_name,
                        **self.audit.input_digest(call.input))
        for listener in list(self.listeners):
            try:
                listener(item)
            except Exception:  # noqa: BLE001 - UI notification only
                pass
        return item

    def request_plan_approval(self, session_id: str, plan_text: str, *, next_mode: PermissionMode | str = PermissionMode.DEFAULT, timeout_s: float | None = None) -> PendingApproval:
        """Plan mode ends in an approvable plan; approving it switches the mode (it pre-approves no action)."""
        now = self.clock()
        mode = PermissionMode.parse(next_mode)
        item = PendingApproval(
            id=new_id("apr"), session_id=session_id, kind="plan", tool="", call_id="", tool_input={}, reason="plan approval",
            created_at=_iso(now), expires_at=_iso(now + timedelta(seconds=timeout_s or self.default_timeout_s)), plan_text=plan_text,
            next_mode=mode.value, human_only=True, created_monotonic=self.monotonic(),
        )
        with self._cond:
            self._items[item.id] = item
            self._persist()
        self.audit.emit(session_id, "approval_requested", approval_id=item.id, kind="plan", next_mode=mode.value)
        return item

    def wait(self, approval_id: str, timeout_s: float | None = None) -> PermissionDecision:
        """Block until the approval resolves; on timeout it is resolved as denied."""
        with self._cond:
            item = self._get(approval_id)
            budget = timeout_s if timeout_s is not None else self.default_timeout_s
            deadline = item.created_monotonic + budget if item.created_monotonic else self.monotonic() + budget
            while not item.resolved:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    self._resolve(item, ApprovalStatus.TIMED_OUT, None, "", "timeout")
                    break
                self._cond.wait(min(remaining, 0.5))
            return item.decision()

    def request(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext, decision: PermissionDecision | None = None, *, timeout_s: float | None = None) -> PermissionDecision:
        """Submit and block until a human answers (or the timeout denies)."""
        item = self.submit(call, spec, ctx, decision, timeout_s=timeout_s)
        return self.wait(item.id, timeout_s)

    # ------------------------------------------------------------------ resolution
    def _get(self, approval_id: str) -> PendingApproval:
        item = self._items.get(approval_id)
        if item is None:
            raise ApprovalNotFoundError(approval_id)
        return item

    def get(self, approval_id: str) -> PendingApproval:
        with self._cond:
            return self._get(approval_id)

    def pending(self, session_id: str | None = None) -> list[PendingApproval]:
        with self._cond:
            return [i for i in self._items.values() if not i.resolved and (session_id is None or i.session_id == session_id)]

    @staticmethod
    def _check_approver(approver: Approver) -> None:
        if not isinstance(approver, Approver) or approver.kind is not ApproverKind.HUMAN or not approver.id:
            raise UntrustedApproverError(
                f"approvals must come from an identified human, not {getattr(approver, 'kind', approver)!r}; "
                "relayed approval claims carry no authority"
            )

    def _resolve(self, item: PendingApproval, status: ApprovalStatus, scope: ApprovalScope | None, approver: str, reason: str) -> None:
        item.status = status
        item.scope = scope
        item.approver = approver
        item.resolution_reason = reason
        item.resolved_at = _iso(self.clock())
        self._persist()
        self._cond.notify_all()
        latency = max(0.0, self.monotonic() - item.created_monotonic) if item.created_monotonic else 0.0
        if self.engine is not None and status in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED} and item.kind == "tool":
            self.engine.record_human_decision(item.session_id, status is ApprovalStatus.APPROVED, latency, item.suggested_rules[0] if item.suggested_rules else None)
        self.audit.emit(item.session_id, "approval_resolved", approval_id=item.id, kind=item.kind, status=status.value,
                        scope=scope.value if scope else None, approver=approver, reason=reason, latency_s=round(latency, 3))

    def approve(self, approval_id: str, approver: Approver, *, scope: ApprovalScope | str = ApprovalScope.ONCE, reason: str = "") -> PendingApproval:
        self._check_approver(approver)
        scope = ApprovalScope(scope)
        with self._cond:
            item = self._get(approval_id)
            if item.resolved:
                raise ApprovalAlreadyResolvedError(f"{approval_id} is already {item.status.value}")
            note = reason
            if item.kind == "plan":
                if self.engine is not None:
                    self.engine.set_mode(item.session_id, item.next_mode)
                self._resolve(item, ApprovalStatus.APPROVED, ApprovalScope.ONCE, approver.id, note)
                return item
            if scope is ApprovalScope.SESSION:
                scope, note = self._grant_session(item, note)
            self._resolve(item, ApprovalStatus.APPROVED, scope, approver.id, note)
            return item

    def _grant_session(self, item: PendingApproval, note: str) -> tuple[ApprovalScope, str]:
        if self.engine is None:
            return ApprovalScope.ONCE, (note + "; " if note else "") + "no engine attached: approved once"
        spec = self._specs.get(item.id)
        record = self.engine.record_for(item.session_id, item.call_id)
        if item.layer == "L0_integrity" and spec is not None:
            self.engine.approve_mcp_description(item.session_id, spec)
        if record is not None:
            for d in record.profile.domains:
                if "new_egress_domain" in item.crossings:
                    self.engine.approve_domain(item.session_id, d)
        if not item.suggested_rules:
            if item.layer in {"L0_integrity"} or "new_egress_domain" in item.crossings:
                return ApprovalScope.SESSION, note
            return ApprovalScope.ONCE, (note + "; " if note else "") + "no narrow rule can cover this call: approved once"
        for rule in item.suggested_rules:
            try:
                self.engine.add_session_rule(item.session_id, rule)
            except PermissionsError:
                return ApprovalScope.ONCE, (note + "; " if note else "") + f"rule {rule!r} rejected by lint: approved once"
        return ApprovalScope.SESSION, note

    def deny(self, approval_id: str, approver: Approver, *, reason: str = "") -> PendingApproval:
        self._check_approver(approver)
        with self._cond:
            item = self._get(approval_id)
            if item.resolved:
                raise ApprovalAlreadyResolvedError(f"{approval_id} is already {item.status.value}")
            self._resolve(item, ApprovalStatus.DENIED, None, approver.id, reason)
            return item

    def cancel(self, approval_id: str, reason: str = "cancelled") -> PendingApproval:
        with self._cond:
            item = self._get(approval_id)
            if not item.resolved:
                self._resolve(item, ApprovalStatus.CANCELLED, None, "", reason)
            return item

    def sweep(self) -> list[PendingApproval]:
        """Deny every pending approval whose ``expires_at`` (by the injected clock) has passed."""
        now = _iso(self.clock())
        expired = []
        with self._cond:
            for item in list(self._items.values()):
                if not item.resolved and item.expires_at <= now:
                    self._resolve(item, ApprovalStatus.TIMED_OUT, None, "", "timeout")
                    expired.append(item)
        return expired


class HumanInTheLoopPolicy:
    """A :class:`~harness_core.types.PermissionPolicy` that resolves ASK through the broker.

    The returned decision is always ALLOW or DENY, so a loop that does not understand ASK can use it.
    """

    def __init__(self, engine: PolicyEngine, broker: ApprovalBroker, *, timeout_s: float | None = None) -> None:
        self.engine = engine
        self.broker = broker
        self.timeout_s = timeout_s

    def check(self, call: ToolCall, spec: ToolSpec | None, ctx: RunContext) -> PermissionDecision:
        decision = self.engine.check(call, spec, ctx)
        if decision.behavior is not PermissionBehavior.ASK:
            return decision
        resolved = self.broker.request(call, spec, ctx, decision, timeout_s=self.timeout_s)
        if resolved.allowed and decision.updated_input is not None:
            resolved.updated_input = decision.updated_input
        return resolved


def denial_outcome(decision: PermissionDecision) -> ToolOutcome:
    """Turn a denial into the tool result the model sees ("do not silently drop denied calls")."""
    text = decision.reason if DENIAL_GUIDANCE in decision.reason else f"{decision.reason} {DENIAL_GUIDANCE}"
    return ToolOutcome(content=f"Permission denied: {text}".strip(), is_error=True, metadata={"permission_denied": True})
