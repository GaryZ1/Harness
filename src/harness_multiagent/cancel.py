"""Hierarchical cooperative cancellation with deadlines.

The harness, not the prompt, owns termination (reference section 8.7 item 4). A run owns one root
:class:`CancellationToken`; every subagent gets a *child* token whose deadline is the earlier of its
own timeout and its parent's. Cancelling a parent cancels every descendant (push propagation), so a
run-level budget hit or ``cancel_all`` stops background agents, as Claude Code does on
``max_budget_usd``. Checks are cooperative: runners call :meth:`CancellationToken.raise_if_cancelled`
between model calls and tool calls. A model call already in flight cannot be interrupted, which is
why the manager additionally bounds each subagent with a wall-clock join (see ``manager.py``).
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import Callable

from .errors import AgentTimeout, Cancelled

Monotonic = Callable[[], float]

#: Stable cancellation reasons; mapped onto :class:`~harness_multiagent.result.SubagentStatus`.
CANCELLED = "cancelled"
TIMEOUT = "timeout"
BUDGET = "budget"


class CancellationToken:
    """Thread-safe cancellation flag with an optional monotonic deadline and parent linkage."""

    def __init__(
        self,
        *,
        deadline: float | None = None,
        parent: "CancellationToken | None" = None,
        monotonic: Monotonic = time.monotonic,
    ) -> None:
        self._monotonic = monotonic
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._detail = ""
        self._children: "weakref.WeakSet[CancellationToken]" = weakref.WeakSet()
        if parent is not None and parent.deadline is not None:
            deadline = parent.deadline if deadline is None else min(deadline, parent.deadline)
        self.deadline = deadline
        self.parent = parent
        if parent is not None:
            parent._adopt(self)

    @classmethod
    def with_timeout(cls, seconds: float | None, *, monotonic: Monotonic = time.monotonic) -> "CancellationToken":
        return cls(deadline=None if seconds is None else monotonic() + seconds, monotonic=monotonic)

    def child(self, timeout_s: float | None = None) -> "CancellationToken":
        """A token cancelled with this one; its deadline is ``min(now + timeout_s, self.deadline)``."""
        deadline = None if timeout_s is None else self._monotonic() + timeout_s
        return CancellationToken(deadline=deadline, parent=self, monotonic=self._monotonic)

    def _adopt(self, child: "CancellationToken") -> None:
        with self._lock:
            self._children.add(child)
            reason, detail = self._reason, self._detail
        if reason is not None:
            child.cancel(reason, detail)

    # ------------------------------------------------------------------ state
    def cancel(self, reason: str = CANCELLED, detail: str = "") -> bool:
        """Cancel this token and all descendants. Returns ``False`` if it was already cancelled."""
        with self._lock:
            if self._reason is not None:
                return False
            self._reason, self._detail = reason, detail
            children = list(self._children)
        self._event.set()
        for c in children:
            c.cancel(reason, detail)
        return True

    def _check_deadline(self) -> None:
        if self._reason is None and self.deadline is not None and self._monotonic() >= self.deadline:
            self.cancel(TIMEOUT, "deadline exceeded")

    @property
    def cancelled(self) -> bool:
        self._check_deadline()
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        self._check_deadline()
        return self._reason

    @property
    def detail(self) -> str:
        return self._detail

    def remaining(self) -> float | None:
        """Seconds until the deadline (``0.0`` once passed), or ``None`` when unbounded."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self._monotonic())

    def raise_if_cancelled(self) -> None:
        self._check_deadline()
        if self._reason is None:
            return
        if self._reason == TIMEOUT:
            raise AgentTimeout(self._detail)
        raise Cancelled(self._reason, self._detail)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until cancelled or ``timeout`` elapses (deadline-aware). Returns ``self.cancelled``."""
        rem = self.remaining()
        if rem is not None:
            timeout = rem if timeout is None else min(timeout, rem)
        self._event.wait(timeout)
        return self.cancelled


__all__ = ["CancellationToken", "CANCELLED", "TIMEOUT", "BUDGET", "Monotonic"]
