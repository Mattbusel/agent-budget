"""Untrusted amounts, saturating bookings, concurrent ``with`` entries and
scopes nobody closed: the 0.1.1 fixes, each pinned by a test."""

import asyncio
import gc
import threading
import time
import warnings

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_budget import (
    Budget,
    BudgetContext,
    BudgetExhausted,
    ContextError,
    Incurred,
    IncurredBooker,
    current_budget,
    from_a2a_metadata,
)
from agent_budget._money import MAX_NANOS, AmountError, parse_amount, usd_to_nanos
from agent_budget.context import Cost

MAX_AMOUNT = "18446744073.709551615"  # MAX_NANOS as a wire amount


def near(a, b):
    return abs(a - b) < 1e-9


# ── amounts from untrusted JSON ──────────────────────────────────────────────


@pytest.mark.parametrize("value", [1e300, 1.8446744073709552e10 * 2, 10**400, 2**64, 18446744074, float("inf")])
def test_huge_amounts_are_amount_errors_not_overflow(value):
    with pytest.raises(AmountError):
        parse_amount(value)
    with pytest.raises(ContextError):
        BudgetContext.from_dict({"budget": {"cost": {"amount": value, "currency": "USD"}}})
    with pytest.raises(ContextError):
        BudgetContext.from_dict({"incurred": {"cost": {"amount": value, "currency": "USD"}}})


def test_integer_amounts_are_exact():
    # Through float, 18446744073 * 1e9 comes out as 18446744072.999999488.
    assert parse_amount(18446744073) == 18446744073 * 10**9
    ctx = BudgetContext.from_dict({"budget": {"cost": {"amount": 18446744073, "currency": "USD"}}})
    assert ctx.to_dict()["budget"]["cost"]["amount"] == "18446744073"
    assert usd_to_nanos(3) == 3 * 10**9


def test_a_count_given_under_both_names_is_rejected():
    for camel, snake in (("cacheRead", "cache_read"), ("cacheWrite", "cache_write")):
        with pytest.raises(ContextError):
            BudgetContext.from_dict({"incurred": {"tokens": {camel: 1, snake: 2}}})
        # Either name alone is fine.
        assert BudgetContext.from_dict({"incurred": {"tokens": {snake: 2}}}).incurred.tokens is not None


big_numbers = st.one_of(
    st.floats(min_value=1e9),
    st.floats(),
    st.integers(min_value=-(10**30), max_value=10**500),
)


@settings(max_examples=1000, deadline=None)
@given(big_numbers)
def test_readers_only_raise_context_error_on_any_number(v):
    for doc in (
        {"budget": {"cost": {"amount": v, "currency": "USD"}}},
        {"incurred": {"cost": {"amount": v, "currency": "USD"}, "aggregatedChildCost": {"amount": v, "currency": "USD"}}},
        {"incurred": {"tokens": {"input": v}, "delegations": v}},
        {"reservation": {"id": "r", "amount": v, "currency": "USD"}},
        {"budget": {"tokens": {"limit": v}}},
    ):
        try:
            BudgetContext.from_dict(doc)
        except ContextError:
            pass
        try:
            from_a2a_metadata({"https://a2a-protocol.org/extensions/budget/v1": doc})
        except ContextError:
            pass


# ── IncurredBooker saturates, and can cap at the reservation ─────────────────


def _incurred(cost, child=None):
    return Incurred(
        Cost(cost, "USD", "local"),
        None if child is None else Cost(child, "USD", None),
    )


def test_booker_saturates_on_the_largest_valid_report():
    b = Budget(1.0)
    booker = IncurredBooker()
    report = BudgetContext.from_dict({"incurred": {"cost": {"amount": MAX_AMOUNT, "currency": "USD"}}})
    out = booker.book(b, report.incurred)
    assert out is not None and out.over_cap
    assert b.spent_nanos == MAX_NANOS
    # A repeat books nothing, and nothing raises.
    assert booker.book(b, _incurred(MAX_NANOS)) is None


def test_booker_saturates_when_cost_plus_child_cost_is_above_the_max():
    b = Budget(1.0)
    booker = IncurredBooker()
    ten_billion = 10_000_000_000 * 10**9
    out = booker.book(b, _incurred(ten_billion, ten_billion))
    assert out is not None and out.over_cap
    assert booker.booked_nanos == MAX_NANOS
    assert b.spent_nanos == MAX_NANOS


def test_booker_can_cap_reports_at_the_reservation():
    root = Budget(1.0)
    with root.child("remote", usd=0.10) as s:
        booker = IncurredBooker.capped_at(s)
        booker.book(s, _incurred(40_000_000))
        assert booker.excess_nanos == 0
        out = booker.book(s, _incurred(5 * 10**9))  # a buggy callee reports $5
        assert out is not None and not out.over_cap
        assert booker.booked_nanos == 100_000_000
        assert booker.excess_nanos == 5 * 10**9 - 100_000_000
    assert near(root.spent_usd, 0.10)
    assert root.reserved_usd == 0


# ── one budget entered by many tasks and threads ────────────────────────────


async def test_one_budget_entered_by_concurrent_tasks():
    shared = Budget(1.0, name="shared")
    seen = []

    async def handler(delay):
        with shared:
            await asyncio.sleep(delay)
            seen.append(current_budget() is shared)
            with shared.child("sub", usd=0.01) as s:
                await asyncio.sleep(delay / 2)
                seen.append(current_budget() is s)
            seen.append(current_budget() is shared)
        seen.append(current_budget() is None)

    await asyncio.gather(*(handler(d) for d in (0.03, 0.01, 0.02, 0.0)))
    assert seen and all(seen)
    assert current_budget() is None


def test_one_budget_entered_by_concurrent_threads():
    shared = Budget(1.0, name="shared")
    start = threading.Barrier(8)
    errors = []

    def worker():
        try:
            start.wait()
            for _ in range(200):
                with shared:
                    time.sleep(0)  # let another thread enter in between
                    assert current_budget() is shared
                    with shared:
                        assert current_budget() is shared
                    assert current_budget() is shared
                assert current_budget() is None
        except Exception as e:  # pragma: no cover - the failure being tested
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_nested_entries_of_one_budget_unwind_in_order():
    a, b = Budget(1.0, name="a"), Budget(1.0, name="b")
    with a:
        with b:
            with a:
                assert current_budget() is a
            assert current_budget() is b
        assert current_budget() is a
    assert current_budget() is None


# ── a scope nobody closed ────────────────────────────────────────────────────


def test_an_unclosed_scope_refunds_when_collected():
    root = Budget(1.0)

    def leaky():
        s = root.child("leak", usd=0.5)
        s.reserve(0.1).settle(0.05)
        raise RuntimeError("the sub-agent failed before closing its scope")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            leaky()
        gc.collect()
    assert near(root.spent_usd, 0.05)
    assert root.reserved_usd == 0
    assert near(root.remaining_usd, 0.95)
    assert any(issubclass(w.category, ResourceWarning) for w in caught)


def test_an_open_reservation_keeps_its_scope_open():
    root = Budget(1.0)
    s = root.child("c", usd=0.5)
    hold = s.reserve(0.2)
    del s
    gc.collect()
    # The reservation holds the scope, so its hold is not cut short.
    assert near(root.reserved_usd, 0.5)
    hold.settle(0.2)
    assert near(root.spent_usd, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        del hold
        gc.collect()
    assert root.reserved_usd == 0
    assert near(root.remaining_usd, 0.8)


# ── a budget of $0 ──────────────────────────────────────────────────────────


def test_empty_budget_refuses_but_books():
    b = Budget._empty("exhausted-caller")
    with pytest.raises(BudgetExhausted):
        b.reserve(0.01)
    with pytest.raises(BudgetExhausted):
        b.child("c", usd=0.01)
    assert b.utilization == 1.0
    assert b.charge_force(0.02).over_cap
