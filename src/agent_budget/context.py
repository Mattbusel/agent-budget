"""``BudgetContext``: the budget a caller hands down and the cost a callee
reports back, in the shape of the A2A budget extension draft
(https://github.com/a2aproject/A2A/issues/2121).

Field names follow the draft: ``budget{cost, tokens, deadline}`` going down,
``incurred{cost, aggregatedChildCost, tokens, delegations}`` coming back,
plus an optional ``reservation{id, amount, currency, expires}`` object
this package adds for the slice a parent reserved for the child.

Rules from the draft thread, applied here:

* Absent is not zero: a field nobody measured is omitted, never filled
  with 0, and reads back as ``None``.
* Money is a decimal string (``"0.25"``); JSON numbers are accepted on
  input.
* ``incurred.cost`` is the reporter's own spend only;
  ``aggregatedChildCost`` is everything its delegates reported.
* In a stream, ``incurred`` is cumulative, never a delta.

The JSON produced here is identical, key for key, to llm-budget's (Rust)
``context::BudgetContext``; both test suites read the same golden fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .budget import Budget, Outcome

from ._money import MAX_NANOS, AmountError, format_nanos, nanos_to_usd, parse_amount, parse_count, parse_currency

EXTENSION_URI = "https://a2a-protocol.org/extensions/budget/v1"
"""The extension URI proposed in a2aproject/A2A#2121 (not yet reserved by the
A2A project; see the module docs of :mod:`agent_budget.a2a`)."""

COST_SOURCES = ("provider", "local")


class ContextError(ValueError):
    """A budget context that cannot be read."""


def _obj(v: Any, where: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise ContextError(f"{where} must be an object")
    return v


def _str(v: Any, where: str, max_len: int = 256) -> str:
    """A non-empty string of at most ``max_len`` UTF-8 bytes (the limits are
    in bytes, as in llm-budget)."""
    if not isinstance(v, str) or not v or len(v.encode("utf-8", "surrogatepass")) > max_len:
        raise ContextError(f"{where} must be a non-empty string of at most {max_len} bytes")
    return v


@dataclass(frozen=True)
class Money:
    """An amount in whole nano-units of ``currency``."""

    nanos: int
    currency: str = "USD"

    @classmethod
    def usd(cls, amount: float) -> "Money":
        from ._money import usd_to_nanos

        return cls(usd_to_nanos(amount), "USD")

    @property
    def amount(self) -> float:
        return nanos_to_usd(self.nanos)

    def to_dict(self) -> Dict[str, Any]:
        return {"amount": format_nanos(self.nanos), "currency": self.currency}

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Money":
        d = _obj(d, where)
        try:
            return cls(parse_amount(d.get("amount")), parse_currency(d.get("currency")))
        except AmountError as e:
            raise ContextError(f"{where}: {e}") from e


@dataclass(frozen=True)
class Cost:
    """A reported cost: an amount, its currency and where it came from
    (``provider`` when it was in a response, ``local`` when computed from
    prices; ``None`` when the reporter did not say)."""

    nanos: int
    currency: str = "USD"
    source: Optional[str] = None

    @property
    def amount(self) -> float:
        return nanos_to_usd(self.nanos)

    def to_dict(self) -> Dict[str, Any]:
        d = {"amount": format_nanos(self.nanos), "currency": self.currency}
        if self.source is not None:
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "Cost":
        m = Money.from_dict(d, where)
        source = d.get("source")
        if source is not None and source not in COST_SOURCES:
            raise ContextError(f"{where}.source must be 'provider' or 'local', got {source!r}")
        return cls(m.nanos, m.currency, source)


@dataclass(frozen=True)
class TokenCounts:
    """Token counts. ``total`` is what the provider reported, never derived."""

    input: Optional[int] = None
    output: Optional[int] = None
    cache_read: Optional[int] = None
    cache_write: Optional[int] = None
    total: Optional[int] = None

    _WIRE = (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"), ("cache_write", "cacheWrite"), ("total", "total"))

    def to_dict(self) -> Dict[str, Any]:
        return {wire: getattr(self, attr) for attr, wire in self._WIRE if getattr(self, attr) is not None}

    @classmethod
    def from_dict(cls, d: Any, where: str) -> "TokenCounts":
        d = _obj(d, where)
        kw = {}
        for attr, wire in cls._WIRE:
            if attr != wire and wire in d and attr in d:
                # llm-budget (serde) rejects a field given under both names.
                raise ContextError(f"{where}.{wire} is given twice (as {wire} and {attr})")
            v = d.get(wire, d.get(attr))  # accept snake_case too
            if v is not None:
                try:
                    kw[attr] = parse_count(v, f"{where}.{wire}")
                except AmountError as e:
                    raise ContextError(str(e)) from e
        return cls(**kw)


@dataclass(frozen=True)
class TokenLimit:
    """The most tokens the callee may use (``None`` when not given)."""

    limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {} if self.limit is None else {"limit": self.limit}

    @classmethod
    def from_dict(cls, d: Any) -> "TokenLimit":
        d = _obj(d, "budget.tokens")
        if d.get("limit") is None:
            return cls()
        try:
            return cls(parse_count(d["limit"], "budget.tokens.limit"))
        except AmountError as e:
            raise ContextError(str(e)) from e


@dataclass(frozen=True)
class BudgetLimit:
    """What the caller allows: money, tokens and a deadline (all optional)."""

    cost: Optional[Money] = None
    tokens: Optional[TokenLimit] = None
    deadline: Optional[str] = None

    @property
    def token_limit(self) -> Optional[int]:
        return None if self.tokens is None else self.tokens.limit

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.cost is not None:
            d["cost"] = self.cost.to_dict()
        if self.tokens is not None:
            d["tokens"] = self.tokens.to_dict()
        if self.deadline is not None:
            d["deadline"] = self.deadline
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetLimit":
        d = _obj(d, "budget")
        cost = Money.from_dict(d["cost"], "budget.cost") if d.get("cost") is not None else None
        tokens = TokenLimit.from_dict(d["tokens"]) if d.get("tokens") is not None else None
        deadline = _str(d["deadline"], "budget.deadline", 64) if d.get("deadline") is not None else None
        return cls(cost, tokens, deadline)


@dataclass(frozen=True)
class Incurred:
    """What the callee spent: its own ``cost``, the ``aggregated_child_cost``
    of everything it delegated, token counts and how many delegations it
    made."""

    cost: Optional[Cost] = None
    aggregated_child_cost: Optional[Cost] = None
    tokens: Optional[TokenCounts] = None
    delegations: Optional[int] = None

    def total_nanos(self, currency: str = "USD") -> Optional[int]:
        """``cost`` plus ``aggregated_child_cost`` in ``currency``, or
        ``None`` if neither is reported in it (absent is not zero)."""
        parts = [c.nanos for c in (self.cost, self.aggregated_child_cost) if c is not None and c.currency == currency]
        return min(sum(parts), MAX_NANOS) if parts else None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.cost is not None:
            d["cost"] = self.cost.to_dict()
        if self.aggregated_child_cost is not None:
            d["aggregatedChildCost"] = self.aggregated_child_cost.to_dict()
        if self.tokens is not None:
            d["tokens"] = self.tokens.to_dict()
        if self.delegations is not None:
            d["delegations"] = self.delegations
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "Incurred":
        d = _obj(d, "incurred")
        cost = Cost.from_dict(d["cost"], "incurred.cost") if d.get("cost") is not None else None
        agg = (
            Cost.from_dict(d["aggregatedChildCost"], "incurred.aggregatedChildCost")
            if d.get("aggregatedChildCost") is not None
            else None
        )
        tokens = TokenCounts.from_dict(d["tokens"], "incurred.tokens") if d.get("tokens") is not None else None
        delegations = None
        if d.get("delegations") is not None:
            try:
                delegations = parse_count(d["delegations"], "incurred.delegations")
            except AmountError as e:
                raise ContextError(str(e)) from e
        return cls(cost, agg, tokens, delegations)


@dataclass(frozen=True)
class ReservationInfo:
    """The slice the caller reserved for this delegation: an opaque ``id``
    (unique per reservation, never shared between siblings), the amount and
    an optional RFC 3339 ``expires``."""

    id: str
    nanos: int
    currency: str = "USD"
    expires: Optional[str] = None

    @property
    def amount(self) -> float:
        return nanos_to_usd(self.nanos)

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "amount": format_nanos(self.nanos), "currency": self.currency}
        if self.expires is not None:
            d["expires"] = self.expires
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "ReservationInfo":
        d = _obj(d, "reservation")
        rid = _str(d.get("id"), "reservation.id", 128)
        try:
            nanos = parse_amount(d.get("amount"))
            currency = parse_currency(d.get("currency"))
        except AmountError as e:
            raise ContextError(f"reservation: {e}") from e
        expires = _str(d["expires"], "reservation.expires", 64) if d.get("expires") is not None else None
        return cls(rid, nanos, currency, expires)


@dataclass(frozen=True)
class BudgetContext:
    """The whole extension payload: any of ``budget``, ``incurred`` and
    ``reservation``."""

    budget: Optional[BudgetLimit] = None
    incurred: Optional[Incurred] = None
    reservation: Optional[ReservationInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.budget is not None:
            d["budget"] = self.budget.to_dict()
        if self.incurred is not None:
            d["incurred"] = self.incurred.to_dict()
        if self.reservation is not None:
            d["reservation"] = self.reservation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetContext":
        """Read a context; unknown keys are ignored, malformed known ones
        raise :class:`ContextError`."""
        d = _obj(d, "budget context")
        try:
            return cls(
                BudgetLimit.from_dict(d["budget"]) if d.get("budget") is not None else None,
                Incurred.from_dict(d["incurred"]) if d.get("incurred") is not None else None,
                ReservationInfo.from_dict(d["reservation"]) if d.get("reservation") is not None else None,
            )
        except ContextError:
            raise
        except (ValueError, TypeError, OverflowError, RecursionError) as e:
            # Untrusted input: whatever slipped past a field check is still
            # a ContextError, never anything else.
            raise ContextError(f"budget context: {e}") from e

    def budget_nanos(self, currency: str = "USD") -> Optional[int]:
        """The money this context allows in ``currency``: ``budget.cost``,
        else the reservation amount, else ``None``."""
        if self.budget is not None and self.budget.cost is not None and self.budget.cost.currency == currency:
            return self.budget.cost.nanos
        if self.reservation is not None and self.reservation.currency == currency:
            return self.reservation.nanos
        return None


# ── building contexts from budgets ─────────────────────────────────────────


def _new_id() -> str:
    import secrets

    return "rsv_" + secrets.token_hex(16)


def outbound(scope: Budget, *, deadline: Optional[str] = None, token_limit: Optional[int] = None,
             expires: Optional[str] = None) -> BudgetContext:
    """The context to send with a delegation made from ``scope`` (usually a
    fresh child scope): ``budget.cost`` is what the scope has left and the
    reservation is its hold in the parent."""
    from .budget import Scope

    tokens = TokenLimit(token_limit) if token_limit is not None else None
    budget = BudgetLimit(Money(scope.remaining_nanos, "USD"), tokens, deadline)
    reservation = None
    if isinstance(scope, Scope):
        reservation = ReservationInfo(_new_id(), scope._held, "USD", expires)
    return BudgetContext(budget=budget, reservation=reservation)


def incurred_of(budget: Budget, *, source: Optional[str] = "local", tokens: Optional[TokenCounts] = None) -> Incurred:
    """What ``budget`` reports back: its own spend as ``cost``, its
    descendants' as ``aggregatedChildCost`` (only when it delegated), and
    the delegation count."""
    if source is not None and source not in COST_SOURCES:
        raise ContextError(f"source must be 'provider' or 'local', got {source!r}")
    delegations = budget.delegations
    agg = Cost(budget.child_cost_nanos, "USD", None) if delegations else None
    return Incurred(Cost(budget.own_cost_nanos, "USD", source), agg, tokens, delegations)


class IncurredBooker:
    """Books a remote callee's reported cost on a local budget exactly once,
    even when the report arrives many times (stream updates, a resent final
    task): ``incurred`` is cumulative, so only the growth is booked.

    ``cap_nanos`` (or :meth:`capped_at`) bounds what the booker will ever
    book: a callee that reports more than it was given is booked at the cap,
    and the part above it is kept in :attr:`excess_nanos` so the caller can
    flag it. Without a cap every reported nano is booked (it is money the
    callee says it already spent). Amounts saturate at the largest amount
    the wire format carries, like llm-budget's counters; a report never
    makes ``book`` raise.
    """

    def __init__(self, currency: str = "USD", *, cap_nanos: Optional[int] = None):
        self.currency = currency
        self.booked_nanos = 0
        self.cap_nanos = None if cap_nanos is None else max(0, min(int(cap_nanos), MAX_NANOS))
        self.excess_nanos = 0

    @classmethod
    def capped_at(cls, scope: Budget, currency: str = "USD") -> "IncurredBooker":
        """A booker that books at most what ``scope`` holds in its parent
        (its reservation), or its limit for a root budget."""
        held = getattr(scope, "_held", None)
        return cls(currency, cap_nanos=held if held is not None else scope._limit)

    def book(self, budget: Budget, incurred: Optional[Incurred]) -> Optional[Outcome]:
        """Book the part of ``incurred``'s total not booked yet; returns the
        :class:`~agent_budget.budget.Outcome`, or ``None`` when nothing new
        was reported."""
        if incurred is None:
            return None
        total = incurred.total_nanos(self.currency)
        if total is None:
            return None
        if self.cap_nanos is not None and total > self.cap_nanos:
            self.excess_nanos = max(self.excess_nanos, total - self.cap_nanos)
            total = self.cap_nanos
        if total <= self.booked_nanos:
            return None
        delta = total - self.booked_nanos
        self.booked_nanos = total
        return budget._charge_force_nanos(delta)


__all__ = [
    "EXTENSION_URI",
    "BudgetContext",
    "BudgetLimit",
    "ContextError",
    "Cost",
    "Incurred",
    "IncurredBooker",
    "Money",
    "ReservationInfo",
    "TokenCounts",
    "TokenLimit",
    "incurred_of",
    "outbound",
]

