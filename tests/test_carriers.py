"""Carriers: baggage merge, MCP _meta, building contexts from budgets, and
fuzzing of every reader that takes untrusted input."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_budget import (
    Budget,
    BudgetContext,
    ContextError,
    IncurredBooker,
    extract_mcp_meta,
    from_a2a_metadata,
    from_baggage,
    incurred_of,
    inject_mcp_meta,
    merge_baggage,
    outbound,
    to_a2a_metadata,
    to_baggage,
)
from agent_budget._money import format_nanos, parse_amount
from agent_budget.carriers import MAX_BAGGAGE_BYTES


def ctx_usd(amount):
    return BudgetContext.from_dict({"budget": {"cost": {"amount": amount, "currency": "USD"}}})


def test_merge_keeps_other_members_and_replaces_ours():
    merged = merge_baggage("a=1, llm.budget.cost.amount=9 ,b=2;p", ctx_usd("1"))
    assert merged == "a=1,b=2;p,llm.budget.cost.amount=1,llm.budget.cost.currency=USD"
    meta = {"baggage": "tenant=acme", "traceparent": "00-x"}
    inject_mcp_meta(meta, ctx_usd("1"))
    assert meta["baggage"] == "tenant=acme,llm.budget.cost.amount=1,llm.budget.cost.currency=USD"
    assert extract_mcp_meta(meta) == ctx_usd("1")
    assert extract_mcp_meta({}) is None
    assert inject_mcp_meta(None, BudgetContext()) == {}


def test_oversized_baggage_is_refused():
    with pytest.raises(ContextError):
        from_baggage("llm.budget.tokens.limit=1,pad=" + "x" * MAX_BAGGAGE_BYTES)


def test_delegation_round_trip():
    parent = Budget(1.00, name="orchestrator")
    slice_ = parent.child("remote", usd=0.40)
    out = outbound(slice_, deadline="2026-12-01T00:00:00Z")
    assert out.reservation.id.startswith("rsv_") and len(out.reservation.id) == 36
    assert out.reservation.nanos == 400_000_000
    assert outbound(slice_).reservation.id != out.reservation.id

    got = from_a2a_metadata(json.loads(json.dumps(to_a2a_metadata(out))))
    callee = Budget(got.budget_nanos() / 1e9, name="callee")
    callee.reserve(0.10).settle(0.07)
    with callee.child("sub", usd=0.10) as sub:
        sub.reserve(0.05).settle(0.05)

    report = BudgetContext(incurred=incurred_of(callee))
    d = report.to_dict()["incurred"]
    assert d["cost"] == {"amount": "0.07", "currency": "USD", "source": "local"}
    assert d["aggregatedChildCost"] == {"amount": "0.05", "currency": "USD"}
    assert d["delegations"] == 1

    booker = IncurredBooker()
    assert booker.book(slice_, report.incurred) is not None
    assert booker.book(slice_, report.incurred) is None  # resent final task: booked once
    assert abs(parent.spent_usd - 0.12) < 1e-12
    assert abs(slice_.close().released_usd - 0.28) < 1e-12


def test_report_without_delegations_has_no_aggregate():
    b = Budget(1.0)
    b.charge_force(0.25)
    d = BudgetContext(incurred=incurred_of(b, source="provider")).to_dict()
    assert d == {"incurred": {"cost": {"amount": "0.25", "currency": "USD", "source": "provider"}, "delegations": 0}}
    with pytest.raises(ContextError):
        incurred_of(b, source="estimate")


def test_booker_ignores_other_currencies_and_absent_costs():
    b = Budget(1.0)
    booker = IncurredBooker()
    eur = BudgetContext.from_dict({"incurred": {"cost": {"amount": "1", "currency": "EUR"}}}).incurred
    assert booker.book(b, eur) is None
    assert booker.book(b, BudgetContext.from_dict({"incurred": {"delegations": 2}}).incurred) is None
    assert booker.book(b, None) is None
    assert b.spent_usd == 0


@settings(max_examples=2000, deadline=None)
@given(st.text(max_size=300))
def test_baggage_reader_only_raises_context_error(s):
    for header in (s, f"llm.budget.cost.amount={s},llm.budget.cost.currency=USD", f"llm.budget.reservation.id={s}"):
        try:
            from_baggage(header)
        except ContextError:
            pass


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=20),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=12), children, max_size=4),
    max_leaves=20,
)


@settings(max_examples=2000, deadline=None)
@given(json_values)
def test_json_reader_only_raises_context_error(v):
    for doc in (
        v,
        {"budget": v},
        {"budget": {"cost": {"amount": v, "currency": "USD"}}},
        {"incurred": {"cost": v, "tokens": v, "delegations": v}},
        {"reservation": {"id": v, "amount": v, "currency": v}},
    ):
        try:
            BudgetContext.from_dict(doc)
        except ContextError:
            pass


amounts = st.integers(0, 2**64 - 1)


@given(amounts)
def test_amounts_print_and_parse_back_exactly(n):
    assert parse_amount(format_nanos(n)) == n


ascii_text = st.text(st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=1, max_size=64)


@given(
    st.one_of(st.none(), st.integers(0, 10**12).map(format_nanos)),
    st.one_of(st.none(), st.integers(0, 2**32)),
    st.one_of(st.none(), ascii_text),
    st.one_of(st.none(), st.tuples(st.text(min_size=1, max_size=40), st.integers(0, 10**12).map(format_nanos), st.one_of(st.none(), ascii_text))),
)
def test_outbound_contexts_round_trip(cost, limit, deadline, res):
    budget = {}
    if cost is not None:
        budget["cost"] = {"amount": cost, "currency": "USD"}
    if limit is not None:
        budget["tokens"] = {"limit": limit}
    if deadline is not None:
        budget["deadline"] = deadline
    doc = {}
    if budget:
        doc["budget"] = budget
    if res is not None:
        r = {"id": res[0], "amount": res[1], "currency": "USD"}
        if res[2] is not None:
            r["expires"] = res[2]
        doc["reservation"] = r
    try:
        ctx = BudgetContext.from_dict(doc)
    except ContextError:
        # ids longer than 128 bytes once UTF-8 encoded are refused by Rust;
        # Python counts characters, so only check what both accept.
        return
    assert ctx.to_dict() == doc
    back = from_baggage(to_baggage(ctx)) or BudgetContext()
    assert back == ctx
