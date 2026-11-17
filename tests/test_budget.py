"""Reserve/settle and child scopes: the same cases as llm-budget's
tests/scopes.rs, plus Python-specific threading and asyncio checks."""

import asyncio
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_budget import (
    AmountError,
    Budget,
    BudgetExhausted,
    BudgetKilled,
    DepthExceeded,
    ScopeClosed,
    current_budget,
)


def near(a, b):
    return abs(a - b) < 1e-9


def test_child_reserves_and_refunds():
    r = Budget(1.0)
    c = r.child("c", usd=0.4)
    assert near(r.remaining_usd, 0.6) and near(r.reserved_usd, 0.4) and near(r.spent_usd, 0)
    c.reserve(0.1).settle(0.08)
    assert near(r.spent_usd, 0.08) and near(r.reserved_usd, 0.32)
    rec = c.close()
    assert near(rec.spent_usd, 0.08) and near(rec.released_usd, 0.32) and near(rec.own_cost_usd, 0.08)
    assert not rec.over_hold
    assert near(r.remaining_usd, 0.92) and near(r.reserved_usd, 0)
    assert near(r.child_cost_usd, 0.08) and near(r.own_cost_usd, 0)


def test_fraction_is_of_what_is_left():
    r = Budget(1.0)
    r.charge_force(0.2)
    c = r.child("c", fraction=0.5)
    assert near(c.limit_usd, 0.4)
    d = r.child("d", fraction=1.0)
    assert near(d.limit_usd, 0.4)
    with pytest.raises(BudgetExhausted):
        r.child("e", fraction=0.5)
    c.close()
    d.close()


@pytest.mark.parametrize("kw", [dict(usd=0), dict(usd=-1), dict(usd=float("nan")), dict(fraction=0), dict(fraction=1.5), {}, dict(usd=1, fraction=0.5)])
def test_bad_limits(kw):
    r = Budget(1.0)
    with pytest.raises(AmountError):
        r.child("x", **kw)
    assert near(r.remaining_usd, 1.0)


def test_too_big_a_slice_is_refused_up_front():
    r = Budget(1.0)
    with pytest.raises(BudgetExhausted):
        r.child("big", usd=1.01)
    assert near(r.remaining_usd, 1.0)


def test_depth_limit():
    r = Budget(1.0, max_depth=2)
    a = r.child("a", fraction=0.5)
    b = a.child("b", fraction=0.5)
    with pytest.raises(DepthExceeded) as e:
        b.child("c", fraction=0.5)
    assert (e.value.depth, e.value.max_depth) == (3, 2)
    assert near(b.remaining_usd, 0.25)


def test_closed_scope_refuses_new_work_but_settles_in_flight():
    r = Budget(1.0)
    c = r.child("c", usd=0.5)
    hold = c.reserve(0.2)
    rec = c.close()
    assert near(rec.in_flight_usd, 0.2) and near(rec.released_usd, 0.3)
    with pytest.raises(ScopeClosed):
        c.reserve(0.01)
    with pytest.raises(ScopeClosed):
        c.child("g", usd=0.01)
    assert near(r.reserved_usd, 0.2)
    hold.settle(0.15)
    assert near(r.spent_usd, 0.15) and near(r.reserved_usd, 0) and near(r.remaining_usd, 0.85)


def test_released_in_flight_reservation_after_close_is_refunded():
    r = Budget(1.0)
    with r.child("c", usd=0.5) as c:
        hold = c.reserve(0.2)
    assert near(r.reserved_usd, 0.2)
    hold.release()
    assert near(r.reserved_usd, 0) and near(r.remaining_usd, 1.0)


def test_overrun_is_booked_up_every_level():
    r = Budget(1.0)
    a = r.child("a", usd=0.5)
    b = a.child("b", usd=0.2)
    out = b.reserve(0.2).settle(0.35)
    assert out.over_cap
    assert near(b.spent_usd, 0.35) and near(a.spent_usd, 0.35) and near(r.spent_usd, 0.35)
    assert b.close().over_hold
    ra = a.close()
    assert near(ra.child_cost_usd, 0.35) and near(ra.own_cost_usd, 0) and near(ra.released_usd, 0.15)
    assert near(r.remaining_usd, 0.65)


def test_own_and_child_cost_split():
    r = Budget(2.0)
    a = r.child("a", usd=1.0)
    a.charge_force(0.1)
    b = a.child("b", usd=0.5)
    b.charge_force(0.2)
    c = b.child("c", usd=0.1)
    c.charge_force(0.05)
    assert near(a.own_cost_usd, 0.1) and near(a.child_cost_usd, 0.25)
    assert near(b.own_cost_usd, 0.2) and near(b.child_cost_usd, 0.05)
    assert near(r.spent_usd, 0.35) and near(r.child_cost_usd, 0.35)
    assert (r.delegations, a.delegations, b.delegations, c.delegations) == (1, 1, 1, 0)


def test_killing_an_ancestor_stops_descendants():
    r = Budget(1.0)
    b = r.child("a", usd=0.5).child("b", usd=0.2)
    r.kill("operator stop")
    with pytest.raises(BudgetKilled) as e:
        b.reserve(0.01)
    assert e.value.scope == "root"
    assert not b.charge_force(0.01).over_cap  # spend that happened is still booked
    assert near(r.spent_usd, 0.01)


def test_settle_twice_is_an_error():
    r = Budget(1.0)
    h = r.reserve(0.1)
    h.settle(0.1)
    with pytest.raises(RuntimeError):
        h.settle(0.1)
    h.release()  # no-op
    assert near(r.spent_usd, 0.1)


def test_current_budget_follows_with_blocks():
    r = Budget(1.0)
    assert current_budget() is None
    with r:
        assert current_budget() is r
        with r.child("c", usd=0.1) as c:
            assert current_budget() is c
        assert current_budget() is r
        assert c.closed
    assert current_budget() is None


def test_exception_inside_scope_still_refunds():
    r = Budget(1.0)
    with pytest.raises(ValueError):
        with r.child("flaky", usd=0.5) as c:
            c.reserve(0.1).settle(0.1)
            raise ValueError("sub-agent crashed")
    assert near(r.spent_usd, 0.1) and near(r.reserved_usd, 0) and near(r.remaining_usd, 0.9)


def test_propagated_remainder_overshoots_with_two_parallel_delegates():
    parent = Budget(1.00)
    barrier = threading.Barrier(2)

    def delegate():
        remaining = parent.remaining_usd  # both read $1.00 before either spends
        barrier.wait()
        own = Budget(remaining)
        for _ in range(4):
            own.reserve(0.20).settle(0.20)
            parent.charge_force(0.20)

    ts = [threading.Thread(target=delegate) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert near(parent.spent_usd, 1.60)


def test_child_scopes_hold_the_cap_with_two_parallel_delegates():
    parent = Budget(1.00)
    barrier = threading.Barrier(2)
    calls = []

    def delegate(i):
        barrier.wait()
        try:
            scope = parent.child(f"d{i}", usd=0.80)
        except BudgetExhausted:
            calls.append(0)
            return
        n = 0
        with scope:
            while True:
                try:
                    scope.reserve(0.20).settle(0.20)
                except BudgetExhausted:
                    break
                n += 1
        calls.append(n)

    ts = [threading.Thread(target=delegate, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(calls) == [0, 4]
    assert parent.spent_usd <= 1.0 + 1e-9 and near(parent.reserved_usd, 0)


def test_sixty_four_threads_never_overcommit():
    parent = Budget(10.0)
    stop = threading.Event()
    seen = []

    def monitor():
        while not stop.is_set():
            seen.append(parent.utilization)

    def child(i):
        try:
            scope = parent.child(f"c{i}", usd=0.25)
        except BudgetExhausted:
            return
        with scope:
            for k in range(20):
                try:
                    h = scope.reserve(0.01)
                except BudgetExhausted:
                    break
                h.settle(0.005 + ((i + k) % 5) * 0.001)
            try:
                g = scope.child(f"g{i}", fraction=0.5)
            except BudgetExhausted:
                return
            with g:
                while True:
                    try:
                        g.reserve(0.02).settle(0.015)
                    except BudgetExhausted:
                        break

    m = threading.Thread(target=monitor)
    m.start()
    ts = [threading.Thread(target=child, args=(i,)) for i in range(64)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    stop.set()
    m.join()
    assert seen and max(seen) <= 1.0 + 1e-12
    assert parent.spent_usd <= 10.0 + 1e-9 and near(parent.reserved_usd, 0)
    assert near(parent.child_cost_usd, parent.spent_usd)


async def test_asyncio_fan_out_with_a_cancelled_task():
    parent = Budget(1.00, name="orchestrator")

    async def sub(i):
        with parent.child(f"sub-{i}", usd=0.25) as scope:
            assert current_budget() is scope
            while True:
                try:
                    hold = scope.reserve(0.04)
                except BudgetExhausted:
                    break
                with hold:  # released on exit unless settled, also on cancellation
                    await asyncio.sleep(0)
                    if i == 3:
                        await asyncio.Event().wait()  # never finishes; cancelled from outside
                    hold.settle(0.03)
            return scope.close()

    tasks = [asyncio.create_task(sub(i)) for i in range(4)]
    await asyncio.sleep(0.01)
    tasks[3].cancel()
    receipts = await asyncio.gather(*tasks[:3])
    with pytest.raises(asyncio.CancelledError):
        await tasks[3]
    for r in receipts:
        assert near(r.spent_usd, 0.24) and near(r.released_usd, 0.01)
    # The cancelled task's hold was released by its with block.
    assert near(parent.spent_usd, 0.72)
    assert near(parent.reserved_usd, 0)


ops = st.lists(
    st.one_of(
        st.tuples(st.just("child"), st.integers(0, 50), st.integers(1, 60), st.booleans()),
        st.tuples(st.just("reserve"), st.integers(0, 50), st.integers(0, 30)),
        st.tuples(st.just("settle"), st.integers(0, 50), st.integers(0, 130)),
        st.tuples(st.just("release"), st.integers(0, 50)),
        st.tuples(st.just("force"), st.integers(0, 50), st.integers(0, 10)),
        st.tuples(st.just("close"), st.integers(0, 50)),
    ),
    max_size=80,
)


@settings(max_examples=300, deadline=None)
@given(ops)
def test_books_stay_exact(seq):
    root = Budget(1.00, max_depth=4)
    scopes = [root]
    holds = []
    booked = 0
    overrun = False
    for op in seq:
        kind = op[0]
        if kind == "child":
            _, p, cents, frac = op
            try:
                kw = dict(fraction=cents / 100) if frac else dict(usd=cents / 100)
                scopes.append(scopes[p % len(scopes)].child(f"s{len(scopes)}", **kw))
            except (BudgetExhausted, ScopeClosed, DepthExceeded, AmountError):
                pass
        elif kind == "reserve":
            try:
                holds.append(scopes[op[1] % len(scopes)].reserve(op[2] / 100))
            except (BudgetExhausted, ScopeClosed):
                pass
        elif kind == "settle" and holds:
            h = holds[op[1] % len(holds)]
            if h._open:
                if op[2] > 100:
                    overrun = True
                out = h.settle(h.estimate_usd * op[2] / 100)
                booked += round(out.recorded_usd * 1e9)
        elif kind == "release" and holds:
            holds[op[1] % len(holds)].release()
        elif kind == "force":
            if op[2]:
                overrun = True
            out = scopes[op[1] % len(scopes)].charge_force(op[2] / 100)
            booked += round(out.recorded_usd * 1e9)
        elif kind == "close" and len(scopes) > 1:
            rec = scopes[1 + op[1] % (len(scopes) - 1)].close()
            assert abs(rec.own_cost_usd + rec.child_cost_usd - rec.spent_usd) < 1e-9
        assert root.spent_nanos == booked
        for s in scopes:
            assert s.own_cost_nanos + s.child_cost_nanos == s.spent_nanos
            if not overrun:
                assert s.utilization <= 1.0
    for h in holds:
        h.release()
    for s in scopes[1:]:
        s.close()
    assert root.reserved_usd == 0
    assert root.spent_nanos == booked


def test_unreferenced_reservation_is_released():
    r = Budget(1.0)
    r.reserve(0.3)  # dropped at once: CPython frees it, and its hold, right here
    import gc

    gc.collect()
    assert near(r.reserved_usd, 0)
