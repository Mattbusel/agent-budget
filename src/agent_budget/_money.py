"""Exact amounts: whole nano-units (1 USD = 1,000,000,000) and canonical
decimal strings.

Every amount in this package is held as an ``int`` of nano-units, the same
representation llm-budget (Rust) uses, so the two libraries agree to the
last digit. On the wire an amount is a decimal string such as ``"0.25"``:
digits, an optional point and at most nine fractional digits, no sign and
no exponent, with trailing zeros dropped.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

NANOS_PER_UNIT = 1_000_000_000
MAX_NANOS = 2**64 - 1

_DECIMAL = re.compile(r"\A([0-9]{1,20})(?:\.([0-9]{1,9}))?\Z")
_CURRENCY = re.compile(r"\A[A-Z]{3}\Z")


class AmountError(ValueError):
    """An amount or currency that cannot be read exactly."""


def usd_to_nanos(usd: float | int | Decimal) -> int:
    """Round a non-negative number to whole nano-units.

    Rounds half up, as llm-budget does. Negative, NaN and infinite values
    raise :class:`AmountError`.
    """
    if isinstance(usd, bool):
        raise AmountError("an amount cannot be a boolean")
    if isinstance(usd, Decimal):
        if not usd.is_finite() or usd < 0:
            raise AmountError(f"amount must be a finite number >= 0, got {usd}")
        nanos = int((usd * NANOS_PER_UNIT).to_integral_value(rounding="ROUND_HALF_UP"))
    else:
        x = float(usd)
        if not math.isfinite(x) or x < 0:
            raise AmountError(f"amount must be a finite number >= 0, got {usd!r}")
        nanos = math.floor(x * NANOS_PER_UNIT + 0.5)
    if nanos > MAX_NANOS:
        raise AmountError(f"amount too large: {usd}")
    return nanos


def nanos_to_usd(nanos: int) -> float:
    """Nano-units as a float (for display and arithmetic, not for the wire)."""
    return nanos / NANOS_PER_UNIT


def format_nanos(nanos: int) -> str:
    """The canonical decimal string for ``nanos`` nano-units.

    >>> format_nanos(500_000_000)
    '0.5'
    >>> format_nanos(3 * 10**9)
    '3'
    """
    if nanos < 0 or nanos > MAX_NANOS:
        raise AmountError(f"nanos out of range: {nanos}")
    whole, frac = divmod(nanos, NANOS_PER_UNIT)
    if frac == 0:
        return str(whole)
    return f"{whole}.{frac:09d}".rstrip("0")


def parse_amount(value: object) -> int:
    """Read a wire amount into nano-units.

    Accepts a decimal string (the canonical form) or a JSON number (the
    form in the first A2A draft, and what a protobuf ``Struct`` turns every
    number into). Strings with more than nine fractional digits, signs,
    exponents or anything else raise :class:`AmountError`.
    """
    if isinstance(value, bool):
        raise AmountError("an amount cannot be a boolean")
    if isinstance(value, str):
        m = _DECIMAL.match(value)
        if not m:
            raise AmountError(f"not a decimal amount: {value!r}")
        whole = int(m.group(1))
        frac = int((m.group(2) or "").ljust(9, "0"))
        nanos = whole * NANOS_PER_UNIT + frac
        if nanos > MAX_NANOS:
            raise AmountError(f"amount too large: {value!r}")
        return nanos
    if isinstance(value, (int, float)):
        return usd_to_nanos(value)
    raise AmountError(f"not an amount: {value!r}")


def parse_currency(value: object) -> str:
    """An ISO 4217 alphabetic code: three upper-case letters."""
    if not isinstance(value, str) or not _CURRENCY.match(value):
        raise AmountError(f"not an ISO 4217 currency code: {value!r}")
    return value


def parse_count(value: object, what: str) -> int:
    """A non-negative whole number. Floats are accepted when they are whole
    (protobuf ``Struct`` stores every number as a double)."""
    if isinstance(value, bool):
        raise AmountError(f"{what} cannot be a boolean")
    if isinstance(value, int):
        n = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        n = int(value)
    else:
        raise AmountError(f"{what} must be a whole number >= 0, got {value!r}")
    if n < 0 or n > MAX_NANOS:
        raise AmountError(f"{what} out of range: {value!r}")
    return n


def to_decimal(nanos: int) -> Decimal:
    """Nano-units as an exact :class:`~decimal.Decimal`."""
    try:
        return Decimal(format_nanos(nanos))
    except InvalidOperation as e:  # pragma: no cover - format_nanos is always valid
        raise AmountError(str(e)) from e
