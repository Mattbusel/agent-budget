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

Fail closed (0.2): a budget needs a hard cap, :meth:`Budget.reserve_for` and
:meth:`Budget.record_usage` refuse a call they cannot price exactly
(:mod:`agent_budget.pricing`), and a call that does not fit is refused before
it is made. ``allow_unpriced=True`` and ``unsafe_soft_limit=True`` are the
explicit ways out; each issues an
:class:`~agent_budget.pricing.UnsafeBudgetWarning` on every call it lets
through.
"""

from __future__ import annotations

import contextvars
import threading
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple, Union

from ._money import MAX_NANOS, AmountError, nanos_to_usd, usd_to_nanos

if TYPE_CHECKING:  # pragma: no cover
    import llm_prices

    from .pricing import Pricer

    UsageArg = Union["llm_prices.Usage", Mapping[str, Any]]

DEFAULT_MAX_DEPTH = 8

_current: contextvars.ContextVar[Optional["Budget"]] = contextvars.ContextVar("agent_budget_current", default=None)
# The ``with`` entries open in this context (thread or asyncio task), innermost
# last. Kept per context, not on the budget, so one budget can be entered by
# many tasks and threads at once.
_entries: contextvars.ContextVar[Tuple[Tuple[Budget, contextvars.Token[Optional[Budget]]], ...]] = contextvars.ContextVar("agent_budget_entries", default=())


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
class _Safety:
    """The fail-closed settings of one budget tree, shared by its scopes."""

    pricer: Optional["Pricer"] = None
    allow_unpriced: bool = False
    soft_limit: bool = False
    events: List[str] = field(default_factory=list, compare=False)

    def get_pricer(self) -> "Pricer":
        if self.pricer is not None:
            return self.pricer
        from .pricing import Pricer

        p = Pricer()
        object.__setattr__(self, "pricer", p)
        return p

    def unsafe(self, message: str) -> None:
        from .pricing import warn_unsafe

        self.events.append(message)
        warn_unsafe(message, stacklevel=4)


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

    Args:
        limit_usd: The hard cap. Required: ``None``, zero, negative, NaN and
            infinite caps raise :class:`AmountError`.
        name: The budget's name in errors and receipts.
        max_depth: The deepest child scope the tree allows.
        pricer: Prices for :meth:`reserve_for`, :meth:`record_usage` and
            :meth:`Reservation.settle_usage`; default
            :class:`agent_budget.pricing.Pricer` (llm-prices, fail closed).
        allow_unpriced: **Unsafe.** Let calls to models that cannot be priced
            through at $0 instead of refusing them, with an
            :class:`~agent_budget.pricing.UnsafeBudgetWarning` every time.
        unsafe_soft_limit: **Unsafe.** Admit reservations that do not fit
            under the cap instead of refusing them, the way a soft limit
            does, with a warning every time. Kills and closed scopes still
            refuse; creating a child scope still needs room.

    Child scopes share the pricer and both opt-outs.
    """

    def __init__(
        self,
        limit_usd: float,
        name: str = "root",
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        pricer: Optional["Pricer"] = None,
        allow_unpriced: bool = False,
        unsafe_soft_limit: bool = False,
    ):
        if limit_usd is None or isinstance(limit_usd, bool):
            raise AmountError("a budget needs a hard cap: limit_usd must be a positive USD amount")
        try:
            limit = usd_to_nanos(limit_usd)
        except AmountError as e:
            raise AmountError(f"a budget needs a hard cap: {e}") from e
        if limit <= 0:
            raise AmountError(f"a budget needs a hard cap: limit must be > 0, got {limit_usd!r}")
        if max_depth < 0:
            raise AmountError(f"max_depth must be >= 0, got {max_depth}")
        self._init(name, limit, None, 0, max_depth, threading.RLock())
        self._safety = _Safety(pricer=pricer, allow_unpriced=allow_unpriced, soft_limit=unsafe_soft_limit)

    @classmethod
    def _empty(cls, name: str) -> "Budget":
        """A budget of $0: refuses every reservation and child above $0, and
        still books spend that already happened (``charge_force``)."""
        obj = Budget.__new__(Budget)
        obj._init(name, 0, None, 0, DEFAULT_MAX_DEPTH, threading.RLock())
        obj._safety = _Safety()
        return obj

    @classmethod
    def _from_nanos(cls, name: str, limit: int, parent: "Budget", depth: int) -> "Scope":
        obj = Scope.__new__(Scope)
        obj._init(name, limit, parent, depth, parent.max_depth, parent._lock)
        obj._safety = parent._safety
        return obj

    def _init(self, name: str, limit: int, parent: Optional[Budget], depth: int, max_depth: int,
              lock: threading.RLock) -> None:
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
            if self._limit == 0:
                return 1.0
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

    _safety: _Safety

    def _admit(self, nanos: int, *, soft: bool = False) -> None:
        if self._committed + nanos > self._limit:
            if not (soft and self._safety.soft_limit):
                raise BudgetExhausted(self.name, nanos_to_usd(nanos), nanos_to_usd(max(0, self._limit - self._committed)))
            self._safety.unsafe(
                f"budget {self.name!r}: soft limit let a call through over the ${nanos_to_usd(self._limit):.6f} cap "
                f"(committed ${nanos_to_usd(self._committed + nanos):.6f})"
            )
        self._committed += nanos
        self._reserved += nanos

    @property
    def unsafe_events(self) -> int:
        """How many calls an opt-out (``allow_unpriced``,
        ``unsafe_soft_limit``) let through in this budget's tree."""
        return len(self._safety.events)

    @property
    def pricer(self) -> "Pricer":
        """The prices this budget's tree uses."""
        return self._safety.get_pricer()

    def _price_or_unsafe(self, model: str, usage: "UsageArg") -> int:
        """The nano-dollar cost of ``usage`` on ``model``, or 0 with a warning
        when it cannot be priced and the tree allows unpriced calls."""
        from .pricing import PriceError, StalePrice

        try:
            return usd_to_nanos(self.pricer.cost(model, usage))
        except StalePrice:
            raise
        except PriceError as e:
            if not self._safety.allow_unpriced:
                raise
            self._safety.unsafe(f"budget {self.name!r}: {e}; let through at $0")
            return 0

    def reserve(self, estimate_usd: float) -> "Reservation":
        """Hold ``estimate_usd`` for a call about to be made.

        Raises :class:`BudgetExhausted`, :class:`ScopeClosed` or
        :class:`BudgetKilled`; nothing is held then.
        """
        nanos = usd_to_nanos(estimate_usd)
        with self._lock:
            self._check_alive()
            self._admit(nanos, soft=True)
        return Reservation(self, nanos)

    def reserve_for(self, model: str, estimate: "UsageArg") -> "Reservation":
        """Hold what a call to ``model`` of size ``estimate`` (prompt tokens
        plus the output you allow, as a provider usage block or
        :class:`llm_prices.Usage`) costs at this budget's prices. Settle it
        with :meth:`Reservation.settle_usage`.

        >>> b = Budget(0.10)
        >>> hold = b.reserve_for("gpt-4o", {"prompt_tokens": 8000, "completion_tokens": 1000})
        >>> round(hold.estimate_usd, 6)
        0.03
        >>> _ = hold.settle_usage("gpt-4o", {"prompt_tokens": 7412, "completion_tokens": 230})
        >>> b.reserve_for("gtp-4o", {"prompt_tokens": 10, "completion_tokens": 1})
        Traceback (most recent call last):
        ...
        agent_budget.pricing.UnknownModel: no price known for model 'gtp-4o'; refusing to guess

        Raises a :class:`~agent_budget.pricing.PriceError` when the call
        cannot be priced (unless ``allow_unpriced``), and otherwise what
        :meth:`reserve` raises.
        """
        nanos = self._price_or_unsafe(model, estimate)
        hold = self.reserve(nanos_to_usd(nanos))
        hold._model = model
        return hold

    def record_usage(self, model: str, usage: "UsageArg") -> Outcome:
        """Book a finished call from its reported usage, priced at this
        budget's prices, even past the cap (the money is spent).

        Raises a :class:`~agent_budget.pricing.PriceError` (booking nothing)
        when the call cannot be priced, unless ``allow_unpriced``.
        """
        return self._charge_force_nanos(self._price_or_unsafe(model, usage))

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
            self._admit(nanos, soft=False)
            self._delegations += 1
            scope = Budget._from_nanos(name, nanos, self, depth)
            scope._hold = nanos
            scope._held = nanos
        return scope

    # ── booking ──────────────────────────────────────────────────────────
    def charge_force(self, cost_usd: float) -> Outcome:
        """Book spend that already happened, whatever the cap or the kill
        switch say."""
        return self._charge_force_nanos(usd_to_nanos(cost_usd))

    def _charge_force_nanos(self, nanos: int) -> Outcome:
        """:meth:`charge_force` in nano-units, saturating at the largest
        amount the wire format carries (as llm-budget's counters do)."""
        nanos = max(0, min(int(nanos), MAX_NANOS))
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
        token = _current.set(self)
        _entries.set(_entries.get() + ((self, token),))
        return self

    def __exit__(self, *exc: object) -> None:
        entries = _entries.get()
        for i in range(len(entries) - 1, -1, -1):
            if entries[i][0] is self:
                token = entries[i][1]
                _entries.set(entries[:i] + entries[i + 1:])
                try:
                    _current.reset(token)
                except ValueError:
                    # Exited in a context copied from the one it was entered
                    # in: restore what was current before the entry.
                    old = token.old_value
                    _current.set(None if old is contextvars.Token.MISSING else old)
                return

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

    def __exit__(self, *exc: object) -> None:
        try:
            self.close()
        finally:
            super().__exit__(*exc)

    def __del__(self) -> None:
        # Like dropping a Rust ScopeGuard: a scope nobody closed refunds its
        # unused hold when it is collected. Holds still taken by open
        # reservations keep the scope alive, so they are never cut short.
        try:
            if not self._closed and self._parent is not None:
                self.close()
                warnings.warn(
                    f"budget scope {self.name!r} was never closed; closed when collected. "
                    "Use `with budget.child(...)` or call close().",
                    ResourceWarning,
                    stacklevel=2,
                )
        except Exception:  # pragma: no cover - interpreter shutdown, or warnings as errors
            pass


class Reservation:
    """Headroom held for one call. Settle it with the real cost, or release
    it. As a context manager it is released on exit unless settled, which is
    the reliable way to release it when a task is cancelled; a reservation
    garbage-collected unsettled is also released, when the collector gets to
    it."""

    __slots__ = ("_budget", "_nanos", "_open", "_model")

    def __init__(self, budget: Budget, nanos: int):
        self._budget = budget
        self._nanos = nanos
        self._open = True
        self._model: Optional[str] = None

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

    def settle_usage(self, model: str, usage: "UsageArg") -> Outcome:
        """Release the hold and book what ``usage`` on ``model`` costs at the
        budget's prices, even past the cap.

        The call has happened, so this never books less than is known: if
        ``model`` cannot be priced (a dated id the provider echoes back, say),
        the model given to :meth:`Budget.reserve_for` is tried, and failing
        that the held amount is booked, never $0.
        """
        from .pricing import PriceError

        pricer = self._budget.pricer
        for m in (model, self._model):
            if m is None:
                continue
            try:
                return self.settle(pricer.cost(m, usage))
            except PriceError:
                continue
        return self.settle(self.estimate_usd)

    def release(self) -> None:
        """Release the hold without booking anything (idempotent)."""
        b = self._budget
        with b._lock:
            if self._open:
                self._open = False
                b._release(self._nanos)

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, *exc: object) -> None:
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
