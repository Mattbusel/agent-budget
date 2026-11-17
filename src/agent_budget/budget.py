"""Budgets, reservations and child scopes.

The semantics match llm-budget's (Rust) ``AgentBudget::child``:

* :meth:`Budget.reserve` holds headroom before a call; refused holds mean the
  call is never made. :meth:`Reservation.settle` books what the call really
  cost, even past the cap (the money is already spent).
* :meth:`Budget.child` reserves a slice of a budget for a sub-agent before it
  starts, so parallel children can never be admitted for more than the
  parent has left. Everything a child books is booked on every ancestor at
  the same moment, out of the child's hold. Closing the child (``with``
  block exit, :meth:`Scope.close`) refunds the unused part.

Amounts are kept as whole nano-units, like llm-budget. One lock per budget
tree makes every operation atomic across threads; nothing awaits, so the
same objects are safe to use from asyncio tasks.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from typing import Optional

from ._money import AmountError, nanos_to_usd, usd_to_nanos

DEFAULT_MAX_DEPTH = 8

_current: contextvars.ContextVar[Optional["Budget"]] = contextvars.ContextVar("agent_budget_current", default=None)


def current_budget() -> Optional["Budget"]:
    """The innermost budget or scope entered with ``with`` in this context
    (thread or asyncio task), or ``None``."""
    return _current.get()


class BudgetError(Exception):
    """Base class of every error this package raises for a refused call."""


class BudgetExhausted(BudgetError):
    """The amount does not fit in what the budget has left."""

    def __init__(self, scope: str, requested_usd: float, remaining_usd: float):
        super().__init__(
            f"budget '{scope}' cannot hold ${requested_usd:.9f}: ${remaining_usd:.9f} left"
        )
        self.scope = scope
        self.requested_usd = requested_usd
        self.remaining_usd = remaining_usd


class ScopeClosed(BudgetError):
    """The scope (or an ancestor) was closed; it admits nothing new."""

    def __init__(self, scope: str):
        super().__init__(f"budget scope '{scope}' is closed")
        self.scope = scope


class BudgetKilled(BudgetError):
    """The kill switch of the budget (or an ancestor) is on."""

    def __init__(self, scope: str, reason: str):
        super().__init__(f"budget '{scope}' was killed: {reason}")
        self.scope = scope
        self.reason = reason


class DepthExceeded(BudgetError):
    """A child scope would be deeper than the tree allows."""

    def __init__(self, scope: str, depth: int, max_depth: int):
        super().__init__(f"scope '{scope}' cannot be created at depth {depth}: the limit is {max_depth}")
        self.scope = scope
        self.depth = depth
        self.max_depth = max_depth


@dataclass(frozen=True)
class Outcome:
    """What a settlement booked."""

    recorded_usd: float
    total_spent_usd: float
    over_cap: bool
    """True when spend plus open holds is above the cap after this booking."""


@dataclass(frozen=True)
class ScopeReceipt:
    """What a child scope did, returned by :meth:`Scope.close`.

    ``own_cost_usd + child_cost_usd == spent_usd``, the split the A2A budget
    draft reports as ``incurred.cost`` and ``incurred.aggregatedChildCost``.
    """

    scope: str
    depth: int
    held_usd: float
    spent_usd: float
    own_cost_usd: float
    child_cost_usd: float
    released_usd: float
    in_flight_usd: float
    over_hold: bool


class Budget:
    """A hard cap in USD with reserve/settle accounting and child scopes.

    >>> b = Budget(1.00, name="orchestrator")
    >>> with b.child("researcher", usd=0.60) as r:
    ...     _ = r.reserve(0.10).settle(0.07)
    ...     round(b.spent_usd, 9)
    0.07
    >>> round(b.remaining_usd, 9)
    0.93
    """

    def __init__(self, limit_usd: float, name: str = "root", *, max_depth: int = DEFAULT_MAX_DEPTH):
        limit = usd_to_nanos(limit_usd)
        if limit <= 0:
            raise AmountError(f"limit must be > 0, got {limit_usd!r}")
        if max_depth < 0:
            raise AmountError(f"max_depth must be >= 0, got {max_depth}")
        self._init(name, limit, None, 0, max_depth, threading.RLock())

    @classmethod
    def _from_nanos(cls, name: str, limit: int, parent: "Budget", depth: int) -> "Scope":
        obj = Scope.__new__(Scope)
        obj._init(name, limit, parent, depth, parent.max_depth, parent._lock)
        return obj

    def _init(self, name, limit, parent, depth, max_depth, lock) -> None:
        self.name = name
        self._limit = limit
        self._committed = 0  # spend + open holds
        self._reserved = 0  # open holds (part of committed)
        self._child_spend = 0  # spend forwarded from descendants
        self._delegations = 0  # direct children created
        self._parent: Optional[Budget] = parent
        self._hold = 0  # for a child: its unconverted hold in the parent
        self._closed = False
        self._killed: Optional[str] = None
        self.depth = depth
        self.max_depth = max_depth
        self._lock = lock
        self._tokens: list[contextvars.Token] = []

    # ── reading ──────────────────────────────────────────────────────────
    @property
    def limit_usd(self) -> float:
        return nanos_to_usd(self._limit)

    @property
    def spent_usd(self) -> float:
        """Everything booked: own calls and descendants'."""
        with self._lock:
            return nanos_to_usd(self._committed - self._reserved)

    @property
    def reserved_usd(self) -> float:
        """Held by open reservations and child scopes."""
        with self._lock:
            return nanos_to_usd(self._reserved)

    @property
    def remaining_usd(self) -> float:
        """Left to spend or reserve."""
        with self._lock:
            return nanos_to_usd(max(0, self._limit - self._committed))

    @property
    def own_cost_usd(self) -> float:
        with self._lock:
            return nanos_to_usd(self._committed - self._reserved - self._child_spend)

    @property
    def child_cost_usd(self) -> float:
        with self._lock:
            return nanos_to_usd(self._child_spend)

    @property
    def delegations(self) -> int:
        """How many child scopes were created directly under this one."""
        with self._lock:
            return self._delegations

    @property
    def utilization(self) -> float:
        with self._lock:
            return self._committed / self._limit

    @property
    def parent(self) -> Optional["Budget"]:
        return self._parent

    @property
    def closed(self) -> bool:
        return self._closed

    # Nano-unit views, exact.
    @property
    def spent_nanos(self) -> int:
        with self._lock:
            return self._committed - self._reserved

    @property
    def own_cost_nanos(self) -> int:
        with self._lock:
            return self._committed - self._reserved - self._child_spend

    @property
    def child_cost_nanos(self) -> int:
        with self._lock:
            return self._child_spend

    @property
    def remaining_nanos(self) -> int:
        with self._lock:
            return max(0, self._limit - self._committed)

    # ── admission ────────────────────────────────────────────────────────
    def _check_alive(self) -> None:
        node: Optional[Budget] = self
        while node is not None:
            if node._closed:
                raise ScopeClosed(node.name)
            if node._killed is not None:
                raise BudgetKilled(node.name, node._killed)
            node = node._parent

    def _admit(self, nanos: int) -> None:
        if self._committed + nanos > self._limit:
            raise BudgetExhausted(self.name, nanos_to_usd(nanos), nanos_to_usd(max(0, self._limit - self._committed)))
        self._committed += nanos
        self._reserved += nanos

    def reserve(self, estimate_usd: float) -> "Reservation":
        """Hold ``estimate_usd`` for a call about to be made.

        Raises :class:`BudgetExhausted`, :class:`ScopeClosed` or
        :class:`BudgetKilled`; nothing is held then.
        """
        nanos = usd_to_nanos(estimate_usd)
        with self._lock:
            self._check_alive()
            self._admit(nanos)
        return Reservation(self, nanos)

    def child(
        self,
        name: str,
        *,
        usd: Optional[float] = None,
        fraction: Optional[float] = None,
    ) -> "Scope":
        """Reserve a slice of this budget for a sub-agent: a fixed ``usd``
        amount, or a ``fraction`` (0, 1] of what is left now.

        Use the result as a context manager (``with b.child(...) as s:``) so
        it closes, and refunds what it did not spend, when the block exits,
        also on an exception or a cancelled task.
        """
        if (usd is None) == (fraction is None):
            raise AmountError("give exactly one of usd= or fraction=")
        with self._lock:
            self._check_alive()
            depth = self.depth + 1
            if depth > self.max_depth:
                raise DepthExceeded(name, depth, self.max_depth)
            if usd is not None:
                nanos = usd_to_nanos(usd)
                if nanos <= 0:
                    raise AmountError(f"usd must be > 0, got {usd!r}")
            else:
                f = float(fraction)  # type: ignore[arg-type]
                if not (0.0 < f <= 1.0):
                    raise AmountError(f"fraction must be in (0, 1], got {fraction!r}")
                nanos = int(max(0, self._limit - self._committed) * f)
                if nanos <= 0:
                    raise BudgetExhausted(self.name, 0.0, nanos_to_usd(max(0, self._limit - self._committed)))
            self._admit(nanos)
            self._delegations += 1
            scope = Budget._from_nanos(name, nanos, self, depth)
            scope._hold = nanos
            scope._held = nanos
        return scope

    # ── booking ──────────────────────────────────────────────────────────
    def charge_force(self, cost_usd: float) -> Outcome:
        """Book spend that already happened, whatever the cap or the kill
        switch say."""
        nanos = usd_to_nanos(cost_usd)
        with self._lock:
            self._book(0, nanos)
            return self._outcome(nanos)

    record = charge_force

    def _outcome(self, nanos: int) -> Outcome:
        return Outcome(
            recorded_usd=nanos_to_usd(nanos),
            total_spent_usd=nanos_to_usd(self._committed - self._reserved),
            over_cap=self._committed > self._limit,
        )

    def _book(self, released: int, actual: int) -> None:
        """Release ``released`` nanos of hold here and book ``actual``."""
        # Forward first, while the hold here still covers the reservation.
        self._forward(actual)
        self._reserved -= released
        self._committed = self._committed - released + actual
        self._trim_if_closed()

    def _forward(self, actual: int) -> None:
        parent = self._parent
        if parent is None or actual == 0:
            return
        take = min(self._hold, actual)
        self._hold -= take
        parent._forward(actual)
        parent._reserved -= take
        parent._committed = parent._committed - take + actual
        parent._child_spend += actual
        parent._trim_if_closed()

    def _release(self, nanos: int) -> None:
        self._reserved -= nanos
        self._committed -= nanos
        self._trim_if_closed()

    def _trim_if_closed(self) -> int:
        if self._parent is None or not self._closed:
            return 0
        keep = self._reserved
        if self._hold <= keep:
            return 0
        released = self._hold - keep
        self._hold = keep
        self._parent._release(released)
        return released

    def kill(self, reason: str = "killed") -> None:
        """Refuse every later reservation, charge and child, here and below."""
        with self._lock:
            self._killed = reason

    # ── context ──────────────────────────────────────────────────────────
    def __enter__(self) -> "Budget":
        self._tokens.append(_current.set(self))
        return self

    def __exit__(self, *exc) -> None:
        _current.reset(self._tokens.pop())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, limit_usd={self.limit_usd}, "
            f"spent_usd={self.spent_usd}, reserved_usd={self.reserved_usd})"
        )


class Scope(Budget):
    """A child budget; see :meth:`Budget.child`.

    Entering it with ``with`` also makes it :func:`current_budget` for the
    block; leaving the block closes it.
    """

    _held: int

    @property
    def held_usd(self) -> float:
        return nanos_to_usd(self._held)

    def close(self) -> ScopeReceipt:
        """Close the scope: it admits nothing new, and the part of its hold
        no call in flight needs goes back to the parent now. Calling it again
        returns the current figures and releases nothing more."""
        with self._lock:
            self._closed = True
            released = self._trim_if_closed()
            spent = self._committed - self._reserved
            return ScopeReceipt(
                scope=self.name,
                depth=self.depth,
                held_usd=nanos_to_usd(self._held),
                spent_usd=nanos_to_usd(spent),
                own_cost_usd=nanos_to_usd(spent - self._child_spend),
                child_cost_usd=nanos_to_usd(self._child_spend),
                released_usd=nanos_to_usd(released),
                in_flight_usd=nanos_to_usd(self._hold),
                over_hold=spent > self._held,
            )

    settle = close

    def __exit__(self, *exc) -> None:
        try:
            self.close()
        finally:
            super().__exit__(*exc)


class Reservation:
    """Headroom held for one call. Settle it with the real cost, or release
    it. As a context manager it is released on exit unless settled, which is
    the reliable way to release it when a task is cancelled; a reservation
    garbage-collected unsettled is also released, when the collector gets to
    it."""

    __slots__ = ("_budget", "_nanos", "_open")

    def __init__(self, budget: Budget, nanos: int):
        self._budget = budget
        self._nanos = nanos
        self._open = True

    @property
    def estimate_usd(self) -> float:
        return nanos_to_usd(self._nanos)

    def settle(self, actual_usd: float) -> Outcome:
        """Release the hold and book ``actual_usd`` in its place, even past
        the cap. Settling twice raises ``RuntimeError``."""
        nanos = usd_to_nanos(actual_usd)
        b = self._budget
        with b._lock:
            if not self._open:
                raise RuntimeError("reservation already settled or released")
            self._open = False
            b._book(self._nanos, nanos)
            return b._outcome(nanos)

    def release(self) -> None:
        """Release the hold without booking anything (idempotent)."""
        b = self._budget
        with b._lock:
            if self._open:
                self._open = False
                b._release(self._nanos)

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self) -> None:
        # Like dropping a Rust Reservation: a hold nobody can settle any more
        # (its task was cancelled, its frame unwound) is released.
        try:
            if self._open:
                self.release()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def __repr__(self) -> str:
        return f"Reservation(budget={self._budget.name!r}, estimate_usd={self.estimate_usd}, open={self._open})"
