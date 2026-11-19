"""Fail-closed prices for model calls, from llm-prices.

A budget that guesses a price is not a cap. Cost trackers keep making the
same three guesses: an unknown model priced at $0 or a placeholder, cached
tokens billed as plain input, and a price table nobody updates. A
:class:`Pricer` refuses each of them instead:

* a model llm-prices does not list raises :class:`UnknownModel`;
* cache reads or writes on a price with no cache rate raise
  :class:`CachePriceMissing` (llm-prices itself bills them as plain input);
* a price last checked against the provider's page more than
  ``max_price_age_days`` ago (default 90) raises :class:`StalePrice`;
* a model listed without a price the call needs raises :class:`Unpriced`.

>>> p = Pricer(today="2026-11-19")
>>> round(p.cost("claude-sonnet-4-6", {"input_tokens": 2048, "output_tokens": 1024,
...     "cache_read_input_tokens": 45000, "cache_creation_input_tokens": 3500}), 9)
0.048129
>>> p.cost("gtp-4o", {"prompt_tokens": 10, "completion_tokens": 1})
Traceback (most recent call last):
...
agent_budget.pricing.UnknownModel: no price known for model 'gtp-4o'; refusing to guess

Usage can be an :class:`llm_prices.Usage` or a provider usage block (OpenAI
Chat Completions or Responses, Anthropic, Gemini; the whole response or just
its usage object); :func:`usage_from` normalizes them.

The ways out are explicit and loud: ``allow_stale=True`` here, and
``allow_unpriced=True`` / ``unsafe_soft_limit=True`` on
:class:`~agent_budget.Budget`, each issue an :class:`UnsafeBudgetWarning`
on every call they let through.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Union

import llm_prices

from .budget import BudgetError

__all__ = [
    "DEFAULT_MAX_PRICE_AGE_DAYS",
    "CachePriceMissing",
    "Pricer",
    "PriceError",
    "StalePrice",
    "UnknownModel",
    "Unpriced",
    "UnsafeBudgetWarning",
    "UsageLike",
    "usage_from",
    "warn_unsafe",
]

DEFAULT_MAX_PRICE_AGE_DAYS = 90

log = logging.getLogger("agent_budget")


class UnsafeBudgetWarning(UserWarning):
    """A call the fail-closed default would have refused was let through by an
    explicit opt-out. Issued on every such call, never only once."""


# Shown every time, not once per call site: an opt-out stays loud.
warnings.filterwarnings("always", category=UnsafeBudgetWarning)


def warn_unsafe(message: str, *, stacklevel: int = 3) -> None:
    """Issue an :class:`UnsafeBudgetWarning` and log it at WARNING level."""
    log.warning("UNSAFE: %s", message)
    warnings.warn(message, UnsafeBudgetWarning, stacklevel=stacklevel)


class PriceError(BudgetError):
    """A call could not be priced exactly, so it is refused."""

    model: str


class UnknownModel(PriceError):
    """No price source lists the model."""

    def __init__(self, model: str):
        super().__init__(f"no price known for model {model!r}; refusing to guess")
        self.model = model


class Unpriced(PriceError):
    """The model is listed without a price this call needs (an output, batch
    or long-context rate)."""

    def __init__(self, model: str, reason: str):
        super().__init__(f"model {model!r} has no usable price for this call: {reason}")
        self.model = model
        self.reason = reason


class CachePriceMissing(PriceError):
    """The call read from (``kind == "read"``) or wrote to (``"write"``) the
    prompt cache, and the price lists no rate for that."""

    def __init__(self, model: str, kind: str):
        super().__init__(
            f"model {model!r} has no cache-{kind} price, and this call has cache-{kind} tokens; refusing to guess"
        )
        self.model = model
        self.kind = kind


class StalePrice(PriceError):
    """The price was last checked against its source longer ago than allowed."""

    def __init__(self, model: str, checked: str, age_days: int, max_age_days: int):
        super().__init__(
            f"price of {model!r} was last checked {checked} ({age_days} days ago), past the "
            f"{max_age_days}-day limit; upgrade llm-prices (pip install -U llm-prices) or pass allow_stale=True"
        )
        self.model = model
        self.checked = checked
        self.age_days = age_days
        self.max_age_days = max_age_days


UsageLike = Union[llm_prices.Usage, Mapping[str, Any]]


def _n(d: Mapping[str, Any], key: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v < 0:
        return 0
    return int(v)


def usage_from(usage: UsageLike) -> llm_prices.Usage:
    """Normalize a provider usage block to :class:`llm_prices.Usage`
    (``input`` = uncached input tokens).

    Reads OpenAI Chat Completions (``prompt_tokens`` with
    ``prompt_tokens_details.cached_tokens``), OpenAI Responses
    (``input_tokens`` with ``input_tokens_details``), Anthropic (cache reads
    and writes on top of ``input_tokens``) and Gemini (``usageMetadata``), the
    whole response or just the usage object, and a mapping of
    :class:`llm_prices.Usage` fields (``{"input": ..., "cache_read": ...}``).

    >>> usage_from({"usage": {"prompt_tokens": 1000, "completion_tokens": 50,
    ...     "prompt_tokens_details": {"cached_tokens": 800}}})
    Usage(input=200, output=50, cache_read=800, cache_write_5m=0, cache_write_1h=0, batch=False)
    """
    if isinstance(usage, llm_prices.Usage):
        return usage
    if not isinstance(usage, Mapping):
        raise TypeError(f"expected a usage mapping or llm_prices.Usage, not {type(usage).__name__}")
    u: Mapping[str, Any] = usage
    for key in ("usage", "usageMetadata", "usage_metadata"):
        inner = usage.get(key)
        if isinstance(inner, Mapping):
            u = inner
            break
    if "prompt_tokens" in u:
        details = u.get("prompt_tokens_details") or {}
        prompt = _n(u, "prompt_tokens")
        cached = min(_n(details, "cached_tokens"), prompt)
        return llm_prices.Usage(input=prompt - cached, output=_n(u, "completion_tokens"), cache_read=cached)
    if "promptTokenCount" in u or "candidatesTokenCount" in u:
        prompt = _n(u, "promptTokenCount")
        cached = min(_n(u, "cachedContentTokenCount"), prompt)
        out = _n(u, "candidatesTokenCount") + _n(u, "thoughtsTokenCount")
        return llm_prices.Usage(input=prompt - cached, output=out, cache_read=cached)
    if "input_tokens" in u or "output_tokens" in u:
        details = u.get("input_tokens_details")
        if isinstance(details, Mapping):
            prompt = _n(u, "input_tokens")
            cached = min(_n(details, "cached_tokens"), prompt)
            return llm_prices.Usage(input=prompt - cached, output=_n(u, "output_tokens"), cache_read=cached)
        creation = u.get("cache_creation")
        w1h = _n(creation, "ephemeral_1h_input_tokens") if isinstance(creation, Mapping) else 0
        w5m = _n(creation, "ephemeral_5m_input_tokens") if isinstance(creation, Mapping) else 0
        if not w1h and not w5m:
            w5m = _n(u, "cache_creation_input_tokens")
        return llm_prices.Usage(
            input=_n(u, "input_tokens"),
            output=_n(u, "output_tokens"),
            cache_read=_n(u, "cache_read_input_tokens"),
            cache_write_5m=w5m,
            cache_write_1h=w1h,
        )
    fields = {"input", "output", "cache_read", "cache_write_5m", "cache_write_1h", "batch"}
    if set(u) & {"input", "output"} and set(u) <= fields:
        return llm_prices.Usage(**u)
    raise ValueError("no usage block found (expected OpenAI, Anthropic or Gemini usage fields)")


def _days(date: str) -> int:
    return datetime.date.fromisoformat(date).toordinal()


PriceSource = Callable[[str], Optional[llm_prices.Price]]


class Pricer:
    """Price model calls from llm-prices, refusing every guess.

    Args:
        max_price_age_days: Refuse a price checked longer ago than this
            (default :data:`DEFAULT_MAX_PRICE_AGE_DAYS`).
        allow_stale: Use stale prices anyway, with an
            :class:`UnsafeBudgetWarning` on every call. Unsafe.
        today: Judge price ages as of this ``YYYY-MM-DD`` date (replays,
            tests). Default: today.
        prices: Your own prices by model id (negotiated rates, fine-tunes,
            self-hosted models), as :class:`llm_prices.Price`. They win over
            the table and are judged for age only when their ``checked`` is set.
        lookup: The table lookup; default :func:`llm_prices.lookup`.
    """

    def __init__(
        self,
        *,
        max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
        allow_stale: bool = False,
        today: Optional[str] = None,
        prices: Optional[Mapping[str, llm_prices.Price]] = None,
        lookup: PriceSource = llm_prices.lookup,
    ) -> None:
        if max_price_age_days < 0:
            raise ValueError(f"max_price_age_days must be >= 0, got {max_price_age_days}")
        self.max_price_age_days = max_price_age_days
        self.allow_stale = allow_stale
        self._today = None if today is None else _days(today)
        self._prices: Dict[str, llm_prices.Price] = dict(prices or {})
        self._lookup = lookup

    def price(self, model: str) -> llm_prices.Price:
        """The price row for ``model``, checked for age.

        Raises :class:`UnknownModel` or :class:`StalePrice`.
        """
        p = self._prices.get(model) or self._lookup(model)
        if p is None:
            raise UnknownModel(model)
        self.check_fresh(model, p.checked)
        return p

    def check_fresh(self, model: str, checked: str) -> None:
        """Raise :class:`StalePrice` if a price checked on ``checked``
        (``YYYY-MM-DD``) is older than the limit; warn instead with
        ``allow_stale``. An empty or unreadable date is not judged."""
        if not checked:
            return
        try:
            age = (self._today or datetime.date.today().toordinal()) - _days(checked)
        except ValueError:
            return
        if age <= self.max_price_age_days:
            return
        if self.allow_stale:
            warn_unsafe(f"price of {model!r} was last checked {checked} ({age} days ago); used anyway")
            return
        raise StalePrice(model, checked, age, self.max_price_age_days)

    def cost(self, model: str, usage: UsageLike) -> float:
        """USD cost of one call. Raises a :class:`PriceError` when it cannot be
        priced exactly."""
        return self._cost(model, self.price(model), usage_from(usage))

    @staticmethod
    def _cost(model: str, p: llm_prices.Price, u: llm_prices.Usage) -> float:
        if u.cache_read and p.cache_read is None:
            raise CachePriceMissing(model, "read")
        if (u.cache_write_5m and "5m" not in p.cache_write) or (u.cache_write_1h and "1h" not in p.cache_write):
            raise CachePriceMissing(model, "write")
        if p.output is None and u.output == 0:
            p = dataclasses.replace(p, output=0.0)  # an embedding model
        c = p.cost_usage(u)
        if c is None:
            what = "batch" if u.batch else "input/output or long-context"
            raise Unpriced(model, f"no {what} price listed")
        return float(c.total)

    def worst_case(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """The most a call with this prompt size and output cap can cost: every
        prompt token at the highest input-side rate the model lists (input,
        cache read or cache write), every output token at the output price,
        long-context prices when the prompt is over the threshold."""
        p = self.price(model)
        lc = p.long_context
        rates = p
        if lc is not None and input_tokens > lc.threshold_tokens:
            rates = lc  # type: ignore[assignment]
        inp = rates.input if rates.input is not None else p.input
        out = rates.output if rates.output is not None else p.output
        if inp is None or (out is None and output_tokens):
            raise Unpriced(model, "no input/output price listed")
        per_input = max([inp, *(r for r in (rates.cache_read, *rates.cache_write.values()) if r is not None)])
        return (input_tokens * per_input + output_tokens * (out or 0.0)) / 1e6
