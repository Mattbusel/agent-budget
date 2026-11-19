"""The 0.2 defaults: every call the budget cannot price exactly is refused,
and every opt-out warns on every call it lets through."""

import math
import warnings

import llm_prices
import pytest

from agent_budget import (
    AmountError,
    Budget,
    BudgetExhausted,
    BudgetKilled,
    CachePriceMissing,
    Pricer,
    StalePrice,
    UnknownModel,
    Unpriced,
    UnsafeBudgetWarning,
    usage_from,
)

U = {"prompt_tokens": 1_000, "completion_tokens": 100}


@pytest.mark.parametrize("limit", [None, 0, -1, math.nan, math.inf, True])
def test_a_budget_without_a_hard_cap_does_not_build(limit):
    with pytest.raises(AmountError, match="hard cap"):
        Budget(limit)


def test_unknown_models_are_refused():
    b = Budget(1.0)
    with pytest.raises(UnknownModel):
        b.reserve_for("gtp-4o", U)
    with pytest.raises(UnknownModel):
        b.record_usage("gtp-4o", U)
    assert b.spent_usd == 0 and b.reserved_usd == 0 and b.unsafe_events == 0


def test_priced_calls_book_the_real_cost():
    b = Budget(1.0)
    hold = b.reserve_for("gpt-4o", {"prompt_tokens": 8_000, "completion_tokens": 1_000})
    assert hold.estimate_usd == pytest.approx(0.03)
    hold.settle_usage("gpt-4o", {"prompt_tokens": 7_412, "completion_tokens": 230})
    assert b.spent_usd == pytest.approx((7_412 * 2.5 + 230 * 10.0) / 1e6)
    b.record_usage("gpt-4o", U)
    assert b.reserved_usd == 0


def test_cache_tokens_need_an_explicit_cache_price():
    no_cache = llm_prices.Price(provider="self", model="no-cache", input=1.0, output=2.0, checked="2026-11-19")
    pricer = Pricer(prices={"no-cache": no_cache}, today="2026-11-19")
    b = Budget(10.0, pricer=pricer)
    b.reserve_for("no-cache", {"input": 1_000, "output": 10}).release()
    with pytest.raises(CachePriceMissing) as e:
        b.reserve_for("no-cache", {"input": 100, "cache_read": 900, "output": 10})
    assert e.value.kind == "read"
    with pytest.raises(CachePriceMissing) as e:
        b.record_usage("no-cache", {"input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 500})
    assert e.value.kind == "write"
    assert "refusing to guess" in str(e.value)


def test_listed_cache_prices_are_used():
    usage = {"input_tokens": 2_048, "output_tokens": 1_024, "cache_read_input_tokens": 45_000,
             "cache_creation_input_tokens": 3_500}
    assert Pricer().cost("claude-sonnet-4-6", usage) == pytest.approx(0.048129)


def test_allow_unpriced_warns_on_every_call():
    b = Budget(1.0, allow_unpriced=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            b.reserve_for("in-house-70b", U).release()
        b.record_usage("in-house-70b", U)
    unsafe = [w for w in caught if issubclass(w.category, UnsafeBudgetWarning)]
    assert len(unsafe) == 4
    assert b.unsafe_events == 4
    assert "let through at $0" in str(unsafe[0].message)


def test_unsafe_warnings_are_not_deduplicated_by_default():
    # The package registers an "always" filter for its category, so a
    # default warnings setup still shows every unsafe call.
    b = Budget(1.0, allow_unpriced=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=UnsafeBudgetWarning)
        for _ in range(2):
            b.reserve_for("in-house-70b", U).release()
    assert sum(issubclass(w.category, UnsafeBudgetWarning) for w in caught) == 2


def test_stale_prices_are_refused_until_updated_or_waived():
    future = Pricer(today="2099-01-01")
    with pytest.raises(StalePrice) as e:
        Budget(1.0, pricer=future).reserve_for("gpt-4o", U)
    assert e.value.age_days > 20_000 and e.value.max_age_days == 90
    assert "pip install -U llm-prices" in str(e.value)
    Budget(1.0, pricer=Pricer(today="2099-01-01", max_price_age_days=100_000)).reserve_for("gpt-4o", U).release()
    loud = Budget(1.0, pricer=Pricer(today="2099-01-01", allow_stale=True))
    with pytest.warns(UnsafeBudgetWarning, match="used anyway"):
        loud.reserve_for("gpt-4o", U).release()


def test_todays_prices_are_fresh():
    Budget(1.0).reserve_for("claude-sonnet-4-6", U).release()


def test_hard_cap_refuses_and_soft_limit_admits_loudly():
    hard = Budget(1.0)
    hard.reserve(0.9).settle(0.9)
    with pytest.raises(BudgetExhausted):
        hard.reserve(0.2)
    soft = Budget(1.0, unsafe_soft_limit=True)
    soft.reserve(0.9).settle(0.9)
    with pytest.warns(UnsafeBudgetWarning, match="soft limit"):
        out = soft.reserve(0.2).settle(0.2)
    assert out.over_cap
    assert soft.spent_usd == pytest.approx(1.1)
    assert soft.unsafe_events == 1


def test_soft_limit_still_stops_on_kill_and_children_still_need_room():
    soft = Budget(1.0, unsafe_soft_limit=True)
    with pytest.raises(BudgetExhausted):
        soft.child("worker", usd=2.0)
    soft.kill("stop")
    with pytest.raises(BudgetKilled):
        soft.reserve(2.0)


def test_children_share_the_settings():
    root = Budget(1.0, allow_unpriced=True, unsafe_soft_limit=True)
    with root.child("worker", usd=0.1) as w:
        with pytest.warns(UnsafeBudgetWarning):
            w.reserve_for("in-house", U).release()
        with pytest.warns(UnsafeBudgetWarning):
            w.reserve(0.5).settle(0.5)
    assert root.unsafe_events == 2
    assert root.spent_usd == pytest.approx(0.5)


def test_settlement_never_books_zero_for_a_call_that_happened():
    b = Budget(1.0)
    hold = b.reserve_for("gpt-4o", U)
    held = hold.estimate_usd
    # The provider echoes a dated id llm-prices does not know: the
    # reservation's model prices it.
    hold.settle_usage("gpt-4o-unknown-snapshot", U)
    assert b.spent_usd == pytest.approx(held)
    hold = Budget(1.0).reserve(0.07)
    assert hold.settle_usage("nobody-knows", U).recorded_usd == pytest.approx(0.07)


def test_usage_shapes_normalize():
    assert usage_from({"usageMetadata": {"promptTokenCount": 100, "cachedContentTokenCount": 40,
                                         "candidatesTokenCount": 5, "thoughtsTokenCount": 7}}) == llm_prices.Usage(
        input=60, cache_read=40, output=12)
    assert usage_from({"input_tokens": 100, "output_tokens": 1,
                       "input_tokens_details": {"cached_tokens": 30}}) == llm_prices.Usage(input=70, cache_read=30, output=1)
    assert usage_from({"input_tokens": 10, "output_tokens": 1, "cache_creation": {
        "ephemeral_1h_input_tokens": 5}}) == llm_prices.Usage(input=10, output=1, cache_write_1h=5)
    assert usage_from({"input": 3, "output": 4}) == llm_prices.Usage(input=3, output=4)
    with pytest.raises(ValueError):
        usage_from({"tokens": 3})


def test_embedding_models_price_without_output():
    p = Pricer()
    emb = next((r for r in llm_prices.all_prices() if r.output is None and r.input is not None), None)
    if emb is None:
        pytest.skip("no embedding rows in this llm-prices")
    assert p.cost(emb.model, {"input": 1_000_000, "output": 0}) == pytest.approx(emb.input)
    with pytest.raises(Unpriced):
        p.cost(emb.model, {"input": 1, "output": 1})


def test_worst_case_uses_the_highest_input_side_rate():
    p = Pricer()
    row = llm_prices.lookup("claude-sonnet-4-6")
    assert row is not None
    top = max([row.input, row.cache_read or 0, *row.cache_write.values()])
    assert p.worst_case("claude-sonnet-4-6", 1_000, 100) == pytest.approx((1_000 * top + 100 * row.output) / 1e6)
