"""Carry a :class:`~agent_budget.context.BudgetContext` across a boundary.

* **A2A**: the whole context under the extension URI in message or task
  metadata (:func:`to_a2a_metadata`, :func:`from_a2a_metadata`).
* **W3C baggage**: the outbound part (budget and reservation) as
  ``llm.budget.*`` members of a ``baggage`` header (:func:`to_baggage`,
  :func:`from_baggage`, :func:`merge_baggage`). Anything that already
  propagates OpenTelemetry baggage carries it with no other change.
* **MCP**: MCP reserves the ``baggage`` key of ``params._meta`` for W3C
  baggage (SEP-414, in the specification since February 2026), so the same
  members ride on any MCP request (:func:`inject_mcp_meta`,
  :func:`extract_mcp_meta`).

Inputs from the wire are untrusted: every reader rejects malformed values
with :class:`~agent_budget.context.ContextError` and never raises anything
else on bad input.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

from ._money import AmountError, format_nanos, parse_amount, parse_count, parse_currency
from .context import (
    EXTENSION_URI,
    BudgetContext,
    BudgetLimit,
    ContextError,
    Money,
    ReservationInfo,
    TokenLimit,
)

PREFIX = "llm.budget."
MAX_BAGGAGE_BYTES = 8192

# Emission order, and the only keys this package writes.
COST_AMOUNT = "llm.budget.cost.amount"
COST_CURRENCY = "llm.budget.cost.currency"
TOKENS_LIMIT = "llm.budget.tokens.limit"
DEADLINE = "llm.budget.deadline"
RESERVATION_ID = "llm.budget.reservation.id"
RESERVATION_AMOUNT = "llm.budget.reservation.amount"
RESERVATION_CURRENCY = "llm.budget.reservation.currency"
RESERVATION_EXPIRES = "llm.budget.reservation.expires"

# W3C baggage-octet: %x21 / %x23-2B / %x2D-3A / %x3C-5B / %x5D-7E, minus "%"
# (which starts an escape).
_SAFE = frozenset(
    [0x21, *range(0x23, 0x2C), *range(0x2D, 0x3B), *range(0x3C, 0x5C), *range(0x5D, 0x7F)]
) - {0x25}
_TOKEN = frozenset(b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _encode(value: str) -> str:
    return "".join(chr(b) if b in _SAFE else f"%{b:02X}" for b in value.encode("utf-8"))


def _decode(value: str, key: str) -> str:
    raw = value.encode("utf-8", "surrogatepass") if value.isascii() else None
    if raw is None:
        raise ContextError(f"baggage value of {key} is not ASCII")
    out = bytearray()
    i = 0
    while i < len(raw):
        b = raw[i]
        if b == 0x25:
            hexpart = raw[i + 1 : i + 3]
            if len(hexpart) != 2 or not all(c in b"0123456789abcdefABCDEF" for c in hexpart):
                raise ContextError(f"bad percent escape in baggage value of {key}")
            out.append(int(hexpart, 16))
            i += 3
        elif b in _SAFE:
            out.append(b)
            i += 1
        else:
            raise ContextError(f"character {chr(b)!r} not allowed in baggage value of {key}")
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ContextError(f"baggage value of {key} is not UTF-8") from e


def baggage_members(ctx: BudgetContext) -> list[tuple[str, str]]:
    """The ``llm.budget.*`` members for ``ctx`` (budget and reservation; the
    ``incurred`` report travels back in A2A metadata, not in baggage)."""
    out: list[tuple[str, str]] = []
    b = ctx.budget
    if b is not None:
        if b.cost is not None:
            out.append((COST_AMOUNT, format_nanos(b.cost.nanos)))
            out.append((COST_CURRENCY, b.cost.currency))
        if b.tokens is not None and b.tokens.limit is not None:
            out.append((TOKENS_LIMIT, str(b.tokens.limit)))
        if b.deadline is not None:
            out.append((DEADLINE, b.deadline))
    r = ctx.reservation
    if r is not None:
        out.append((RESERVATION_ID, r.id))
        out.append((RESERVATION_AMOUNT, format_nanos(r.nanos)))
        out.append((RESERVATION_CURRENCY, r.currency))
        if r.expires is not None:
            out.append((RESERVATION_EXPIRES, r.expires))
    return out


def to_baggage(ctx: BudgetContext) -> str:
    """A ``baggage`` header value holding only the budget members."""
    return ",".join(f"{k}={_encode(v)}" for k, v in baggage_members(ctx))


def merge_baggage(existing: Optional[str], ctx: BudgetContext) -> str:
    """``existing`` with every ``llm.budget.*`` member replaced by ``ctx``'s.
    Other members are kept as they were."""
    kept = []
    for member in (existing or "").split(","):
        m = member.strip()
        if not m:
            continue
        key = m.split("=", 1)[0].strip()
        if key.startswith(PREFIX):
            continue
        kept.append(m)
    ours = to_baggage(ctx)
    return ",".join([*kept, ours] if ours else kept)


def _parse_members(header: str) -> dict[str, str]:
    if len(header.encode("utf-8", "surrogatepass")) > MAX_BAGGAGE_BYTES:
        raise ContextError(f"baggage header longer than {MAX_BAGGAGE_BYTES} bytes")
    found: dict[str, str] = {}
    for member in header.split(","):
        head = member.split(";", 1)[0]
        if "=" not in head:
            continue  # not ours to judge
        key, value = head.split("=", 1)
        key = key.strip(" \t")
        if not key.startswith(PREFIX):
            continue
        if not all(ord(c) < 128 and ord(c) in _TOKEN for c in key):
            raise ContextError(f"baggage key {key!r} is not a token")
        if key in found:
            raise ContextError(f"baggage key {key} appears twice")
        found[key] = _decode(value.strip(" \t"), key)
    return found


def from_baggage(header: Optional[str]) -> Optional[BudgetContext]:
    """Read the budget members of a ``baggage`` header. ``None`` when it has
    none; :class:`ContextError` when they are malformed."""
    if not header:
        return None
    if not isinstance(header, str):
        raise ContextError("baggage header must be a string")
    m = _parse_members(header)
    if not m:
        return None
    try:
        cost = None
        if COST_AMOUNT in m or COST_CURRENCY in m:
            if COST_AMOUNT not in m or COST_CURRENCY not in m:
                raise ContextError("llm.budget.cost.amount and llm.budget.cost.currency go together")
            cost = Money(parse_amount(m[COST_AMOUNT]), parse_currency(m[COST_CURRENCY]))
        limit = parse_count(_int_text(m[TOKENS_LIMIT]), TOKENS_LIMIT) if TOKENS_LIMIT in m else None
        deadline = _bounded(m.get(DEADLINE), DEADLINE, 64) or None
        tokens = TokenLimit(limit) if limit is not None else None
        budget = BudgetLimit(cost, tokens, deadline) if (cost or tokens or deadline) else None
        reservation = None
        if any(k in m for k in (RESERVATION_ID, RESERVATION_AMOUNT, RESERVATION_CURRENCY, RESERVATION_EXPIRES)):
            for k in (RESERVATION_ID, RESERVATION_AMOUNT, RESERVATION_CURRENCY):
                if k not in m:
                    raise ContextError(f"{k} is missing from the reservation")
            reservation = ReservationInfo(
                _bounded(m[RESERVATION_ID], RESERVATION_ID, 128) or "",
                parse_amount(m[RESERVATION_AMOUNT]),
                parse_currency(m[RESERVATION_CURRENCY]),
                _bounded(m.get(RESERVATION_EXPIRES), RESERVATION_EXPIRES, 64) or None,
            )
            if not reservation.id:
                raise ContextError("llm.budget.reservation.id is empty")
    except AmountError as e:
        raise ContextError(str(e)) from e
    if budget is None and reservation is None:
        return None
    return BudgetContext(budget=budget, reservation=reservation)


def _int_text(s: str) -> int:
    if not s.isascii() or not s.isdigit() or len(s) > 20:
        raise ContextError(f"not a whole number: {s!r}")
    return int(s)


def _bounded(v: Optional[str], key: str, max_len: int) -> Optional[str]:
    if v is None:
        return None
    if len(v.encode("utf-8", "surrogatepass")) > max_len:
        raise ContextError(f"{key} is longer than {max_len} bytes")
    return v


# ── A2A ────────────────────────────────────────────────────────────────────


def to_a2a_metadata(ctx: BudgetContext) -> dict:
    """``{EXTENSION_URI: ctx}``, ready to merge into message, task or
    request metadata."""
    return {EXTENSION_URI: ctx.to_dict()}


def from_a2a_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[BudgetContext]:
    """The context under the extension URI, or ``None`` if there is none."""
    if not metadata:
        return None
    try:
        payload = metadata.get(EXTENSION_URI)
    except AttributeError as e:
        raise ContextError("metadata must be a mapping") from e
    if payload is None:
        return None
    return BudgetContext.from_dict(payload)


# ── MCP ────────────────────────────────────────────────────────────────────


def inject_mcp_meta(meta: Optional[MutableMapping[str, Any]], ctx: BudgetContext) -> MutableMapping[str, Any]:
    """Put ``ctx`` into an MCP ``_meta`` object's ``baggage`` (merged with
    baggage already there). Returns the object (a new dict if ``meta`` is
    ``None``)."""
    meta = {} if meta is None else meta
    merged = merge_baggage(meta.get("baggage"), ctx)
    if merged:
        meta["baggage"] = merged
    return meta


def extract_mcp_meta(meta: Optional[Mapping[str, Any]]) -> Optional[BudgetContext]:
    """The budget members of an MCP ``_meta`` object's ``baggage``."""
    if not meta:
        return None
    return from_baggage(meta.get("baggage"))


__all__ = [
    "extract_mcp_meta",
    "from_a2a_metadata",
    "from_baggage",
    "inject_mcp_meta",
    "merge_baggage",
    "to_a2a_metadata",
    "to_baggage",
]
