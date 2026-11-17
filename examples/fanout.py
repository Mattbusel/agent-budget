"""Four sub-agents in parallel under one $1.00 cap, offline.

    python examples/fanout.py

Each sub-agent wants eight gpt-4o calls, reserved at $0.06 (20,000 prompt
tokens + 1,000 output) and settled at $0.052 (18,000 in, 700 out).

1. Passing the remainder down: every sub-agent is told what the parent has
   left and enforces that on its own. All four start together, all four see
   $1.00, and together they spend $1.664.
2. Child scopes: each sub-agent's $0.25 slice is reserved from the parent
   before it starts. Total spend stays under the cap.
"""

import asyncio

from agent_budget import Budget, BudgetExhausted

RESERVE, ACTUAL, CALLS = 0.06, 0.052, 8


async def sub_agent(budget):
    made = 0
    for _ in range(CALLS):
        try:
            hold = budget.reserve(RESERVE)  # refused before the call: nothing spent
        except BudgetExhausted:
            break
        await asyncio.sleep(0)  # the model call would go here
        hold.settle(ACTUAL)
        made += 1
    return made


async def main():
    # 1. Each sub-agent gets its own budget equal to the parent's remainder.
    parent = Budget(1.00, name="orchestrator")
    remaining = parent.remaining_usd

    async def naive(i):
        own = Budget(remaining, name=f"sub-{i}")
        calls = await sub_agent(own)
        parent.charge_force(own.spent_usd)
        return calls

    calls = sum(await asyncio.gather(*(naive(i) for i in range(4))))
    over = (parent.spent_usd / parent.limit_usd - 1) * 100
    print(f"remainder passed down: {calls} calls, spent ${parent.spent_usd:.3f} of ${parent.limit_usd:.2f} ({over:.0f}% over the cap)")

    # 2. Child scopes.
    parent = Budget(1.00, name="orchestrator")

    async def scoped(i):
        with parent.child(f"sub-{i}", usd=0.25) as scope:
            n = await sub_agent(scope)
            return n, scope.close()

    results = await asyncio.gather(*(scoped(i) for i in range(4)))
    for n, r in results:
        print(f"  {r.scope}: {n} calls, spent ${r.spent_usd:.3f} of its ${r.held_usd:.2f}, ${r.released_usd:.3f} back to the parent")
    calls = sum(n for n, _ in results)
    print(f"child scopes:          {calls} calls, spent ${parent.spent_usd:.3f} of ${parent.limit_usd:.2f}, ${parent.reserved_usd:.3f} still held")
    assert parent.spent_usd <= parent.limit_usd


if __name__ == "__main__":
    asyncio.run(main())
