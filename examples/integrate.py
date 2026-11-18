"""The integration from docs/integrating.md, runnable and type-checked.

CI runs it and checks it with ``mypy --strict``, so the snippet in the docs
cannot drift from the API. Run: ``python examples/integrate.py``.
"""

from __future__ import annotations

from agent_budget import Budget, BudgetExhausted


def call_model(prompt: str) -> float:
    """Stand-in for any client: returns what the call really cost in USD
    (from the provider's usage report and your price table)."""
    return 0.0042


def main() -> None:
    # The five lines.
    budget = Budget(5.00, name="my-agent")             # $5 hard cap
    with budget.reserve(0.01) as hold:                 # raises BudgetExhausted if it would not fit
        cost = call_model("...")                       # the real call, with any client
        hold.settle(cost)                              # books what it really cost

    assert abs(budget.spent_usd - 0.0042) < 1e-12
    assert budget.reserved_usd == 0

    # A sub-agent gets a slice reserved up front; the unused part comes back.
    with budget.child("researcher", usd=0.25) as research:
        research.reserve(0.10).settle(0.08)
    assert abs(budget.remaining_usd - (5.00 - 0.0042 - 0.08)) < 1e-9

    # Over the cap, the call is never made.
    small = Budget(0.01, name="small")
    try:
        small.reserve(0.02)
    except BudgetExhausted as e:
        print(f"refused before the call: {e}")

    print(f"spent ${budget.spent_usd:.4f} of ${budget.limit_usd:.2f}")


if __name__ == "__main__":
    main()
