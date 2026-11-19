"""Spend audit: replay a log of LLM calls and show where a soft limit would
have let money through that a hard cap refuses.

Most budgets in agent frameworks are soft limits: they check what was spent
*before* a call, book the cost *after* it, price unknown models at a default
(often $0), and price cached tokens as plain input. Each of those lets money
past the cap. :func:`replay` runs the same calls two ways:

* **soft limit**: a call goes through while the *tracked* spend is under the
  cap; the tracker books what the log says it booked (``booked_usd``), or the
  true cost when the log has none, and $0 for a model it cannot price;
* **hard cap, fail closed** (this package's default): a call is admitted only
  if its true cost fits in what is left, and refused if its price is unknown,
  missing a cache rate, or too old;

and reports the true cost of what the soft limit let through, the call at
which real spend crossed the cap, and the money past it. Same semantics and
output as ``llm-budget audit`` (Rust).

Command line::

    agent-budget audit calls.jsonl --cap 5      # or: python -m agent_budget audit ...

Input: one JSON object per line with a ``model`` and the provider's usage
(OpenAI, Anthropic or Gemini; the whole response or the usage block),
optionally ``booked_usd`` (or ``cost_usd``, ``cost``): what your current
tracker booked. Exit status 2 when money got past the cap or a call had no
price, 1 on bad input, 0 otherwise.

>>> calls = parse_jsonl('''
... {"model": "gpt-4o", "usage": {"prompt_tokens": 100000, "completion_tokens": 2000}}
... {"model": "gpt-4o", "usage": {"prompt_tokens": 100000, "completion_tokens": 2000}}
... ''')
>>> r = replay(calls, cap_usd=0.30, today="2026-11-19")
>>> r.soft.calls_through, round(r.soft.money_past_cap_usd, 2), r.hard.admitted
(2, 0.24, 1)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import llm_prices

from .pricing import (
    DEFAULT_MAX_PRICE_AGE_DAYS,
    CachePriceMissing,
    PriceError,
    Pricer,
    StalePrice,
    usage_from,
)

__all__ = [
    "AuditReport",
    "Call",
    "HardCapResult",
    "ModelDrift",
    "SoftLimitResult",
    "Unpriced",
    "main",
    "parse_jsonl",
    "replay",
]


@dataclass(frozen=True)
class Call:
    """One call from the log."""

    line: int
    model: str
    usage: llm_prices.Usage
    booked_usd: Optional[float] = None


class AuditInputError(ValueError):
    """A line of the log could not be read."""


def parse_jsonl(text: str) -> List[Call]:
    """Parse a JSON-lines log. Raises :class:`AuditInputError` naming the
    first bad line."""
    calls: List[Call] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError as e:
            raise AuditInputError(f"line {i}: not JSON: {e}") from e
        if not isinstance(v, dict) or not isinstance(v.get("model"), str):
            raise AuditInputError(f'line {i}: no string "model" field')
        try:
            usage = usage_from(v)
        except (ValueError, TypeError) as e:
            raise AuditInputError(f"line {i}: {e}") from e
        booked = next((v[k] for k in ("booked_usd", "cost_usd", "cost") if isinstance(v.get(k), (int, float))), None)
        if booked is not None and (isinstance(booked, bool) or not math.isfinite(booked) or booked < 0):
            booked = None
        calls.append(Call(line=i, model=v["model"], usage=usage, booked_usd=None if booked is None else float(booked)))
    return calls


@dataclass
class SoftLimitResult:
    """How the soft limit did on the log."""

    calls_through: int = 0
    fired_at_call: Optional[int] = None
    tracked_usd: float = 0.0
    true_usd: float = 0.0
    unpriced_calls_through: int = 0
    crossed_cap_at_call: Optional[int] = None
    calls_past_cap: int = 0
    money_past_cap_usd: float = 0.0


@dataclass
class HardCapResult:
    """How the fail-closed hard cap did on the log."""

    admitted: int = 0
    spent_usd: float = 0.0
    refused_over_cap: int = 0
    refused_unpriced: int = 0
    refused_cache_price: int = 0
    refused_stale: int = 0


@dataclass
class ModelDrift:
    """Booked versus true cost for one model."""

    model: str
    calls: int = 0
    booked_usd: float = 0.0
    true_usd: float = 0.0

    @property
    def ratio(self) -> Optional[float]:
        """``booked / true``: below 1 the tracker under-books."""
        return self.booked_usd / self.true_usd if self.true_usd > 0 else None


@dataclass
class Unpriced:
    """A model with no usable price."""

    model: str
    reason: str
    calls: int = 0


@dataclass
class AuditReport:
    """The result of :func:`replay`."""

    cap_usd: float
    calls: int = 0
    true_total_usd: float = 0.0
    priced_calls: int = 0
    soft: SoftLimitResult = field(default_factory=SoftLimitResult)
    hard: HardCapResult = field(default_factory=HardCapResult)
    drift: List[ModelDrift] = field(default_factory=list)
    unpriced: List[Unpriced] = field(default_factory=list)
    stale: Dict[str, str] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        """True when the soft limit let money past the cap, or a call had no price."""
        return self.soft.money_past_cap_usd > 0 or self.soft.unpriced_calls_through > 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for m, row in zip(self.drift, d["drift"]):
            row["ratio"] = m.ratio
        return d

    def __str__(self) -> str:
        cap = self.cap_usd
        s, h = self.soft, self.hard
        out = [
            f"{self.calls} calls, cap ${cap:.2f}",
            f"True cost of every call: ${self.true_total_usd:.4f} ({self.priced_calls} priced, "
            f"{self.calls - self.priced_calls} without a usable price)",
            "",
            "Soft limit (checks spend before each call, books after, $0 for unpriced models):",
        ]
        if s.fired_at_call is not None:
            out.append(f"  stopped the run at call {s.fired_at_call}, after letting {s.calls_through} calls through")
        else:
            out.append(f"  never stopped the run: all {s.calls_through} calls went through")
        out.append(f"  tracker booked ${s.tracked_usd:.4f}; those calls really cost ${s.true_usd:.4f}")
        if s.money_past_cap_usd > 0:
            n = "?" if s.crossed_cap_at_call is None else s.crossed_cap_at_call
            out.append(
                f"  MONEY PAST THE CAP: ${s.money_past_cap_usd:.4f} ({s.true_usd / cap:.2f}x the cap); real spend "
                f"crossed ${cap:.2f} at call {n}, then {s.calls_past_cap} more calls ran"
            )
        else:
            out.append("  no money past the cap")
        if s.unpriced_calls_through:
            out.append(f"  plus {s.unpriced_calls_through} calls with no price, booked at $0: their cost is unknown")
        out += [
            "",
            "Hard cap, fail closed (agent-budget default):",
            f"  admitted {h.admitted} calls, spent ${h.spent_usd:.4f} of ${cap:.2f}",
            f"  refused {h.refused_over_cap} over the cap, {h.refused_unpriced} unpriced, "
            f"{h.refused_cache_price} missing a cache price, {h.refused_stale} with a stale price",
        ]
        if self.drift:
            out += ["", "Booked by your tracker vs true cost:"]
            for d in self.drift[:15]:
                out.append(
                    f"  {d.model:<32} {d.calls:>6} calls  booked ${d.booked_usd:>10.4f}  "
                    f"true ${d.true_usd:>10.4f}  {d.ratio or 0.0:>6.2f}x"
                )
        if self.unpriced:
            out += ["", "No usable price (a soft limit books these at $0):"]
            out += [f"  {u.model} ({u.calls} calls): {u.reason}" for u in self.unpriced]
        if self.stale:
            out += ["", "Prices older than the limit (pip install -U llm-prices):"]
            out += [f"  {m}: last checked {d}" for m, d in sorted(self.stale.items())]
        return "\n".join(out) + "\n"


def replay(
    calls: Sequence[Call],
    cap_usd: float,
    *,
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
    today: Optional[str] = None,
    pricer: Optional[Pricer] = None,
) -> AuditReport:
    """Replay ``calls`` under a soft limit and under a fail-closed hard cap of
    ``cap_usd``. Prices come from ``pricer`` (default: llm-prices); its age
    limit is ``max_price_age_days`` as of ``today``."""
    if not (isinstance(cap_usd, (int, float)) and math.isfinite(cap_usd) and cap_usd > 0):
        raise ValueError(f"a cap is required: cap_usd must be a positive amount, got {cap_usd!r}")
    strict = pricer or Pricer(max_price_age_days=max_price_age_days, today=today)
    loose = Pricer(max_price_age_days=10**9, prices=getattr(strict, "_prices", None), lookup=strict._lookup)
    report = AuditReport(cap_usd=cap_usd, calls=len(calls))
    soft, hard = report.soft, report.hard
    unpriced: Dict[str, Unpriced] = {}
    drift: Dict[str, ModelDrift] = {}
    stopped = False
    for n, call in enumerate(calls, start=1):
        true_cost: Optional[float]
        refusal: Optional[str] = None
        try:
            true_cost = loose.cost(call.model, call.usage)
        except PriceError as e:
            true_cost = None
            refusal = "cache" if isinstance(e, CachePriceMissing) else "unpriced"
            u = unpriced.setdefault(call.model, Unpriced(call.model, str(e)))
            u.calls += 1
        else:
            report.priced_calls += 1
            report.true_total_usd += true_cost
            try:
                strict.price(call.model)
            except StalePrice as e:
                refusal = "stale"
                report.stale[call.model] = e.checked
        if true_cost is not None and call.booked_usd is not None:
            d = drift.setdefault(call.model, ModelDrift(call.model))
            d.calls += 1
            d.booked_usd += call.booked_usd
            d.true_usd += true_cost

        if not stopped:
            if soft.tracked_usd >= cap_usd:
                stopped = True
                soft.fired_at_call = n
            else:
                soft.calls_through += 1
                if soft.true_usd > cap_usd:
                    soft.calls_past_cap += 1
                booked = call.booked_usd if call.booked_usd is not None else true_cost
                soft.tracked_usd += booked or 0.0
                if true_cost is None:
                    soft.unpriced_calls_through += 1
                else:
                    soft.true_usd += true_cost
                    if soft.crossed_cap_at_call is None and soft.true_usd > cap_usd:
                        soft.crossed_cap_at_call = n

        if refusal == "unpriced":
            hard.refused_unpriced += 1
        elif refusal == "cache":
            hard.refused_cache_price += 1
        elif refusal == "stale":
            hard.refused_stale += 1
        elif true_cost is not None and hard.spent_usd + true_cost <= cap_usd + 1e-12:
            hard.admitted += 1
            hard.spent_usd += true_cost
        else:
            hard.refused_over_cap += 1
    soft.money_past_cap_usd = max(0.0, soft.true_usd - cap_usd)
    report.unpriced = list(unpriced.values())
    report.drift = sorted(
        (d for d in drift.values() if d.ratio is not None and abs(d.ratio - 1.0) > 0.01),
        key=lambda d: -abs(math.log(d.ratio)) if d.ratio else 0.0,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``agent-budget audit <calls.jsonl | -> --cap USD [--json]``."""
    parser = argparse.ArgumentParser(
        prog="agent-budget",
        description="Where would a soft limit have let money past your cap?",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="replay a JSON-lines log of LLM calls")
    a.add_argument("path", help="calls.jsonl, or - for standard input")
    a.add_argument("--cap", required=True, help="the hard cap in USD (required: a budget without a cap is not a budget)")
    a.add_argument("--max-price-age", type=int, default=DEFAULT_MAX_PRICE_AGE_DAYS, help="refuse prices checked longer ago (days)")
    a.add_argument("--today", help="judge price ages as of YYYY-MM-DD")
    a.add_argument("--json", action="store_true", help="print the report as JSON")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code in (0, None) else 1
    try:
        cap = float(str(args.cap).lstrip("$"))
        if not (math.isfinite(cap) and cap > 0):
            raise ValueError
    except ValueError:
        print(f"agent-budget: --cap must be a positive amount, got {args.cap}", file=sys.stderr)
        return 1
    try:
        text = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
        calls = parse_jsonl(text)
        report = replay(calls, cap, max_price_age_days=args.max_price_age, today=args.today)
    except (OSError, ValueError) as e:
        print(f"agent-budget: {args.path}: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"agent-budget audit: {args.path}")
        print(report, end="")
    return 2 if report.leaked else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
