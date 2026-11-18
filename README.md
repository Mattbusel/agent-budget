# agent-budget

Hard USD budgets for agents that delegate.

When an orchestrator hands work to several sub-agents at once, each one
checks its spending against the same "remaining" figure, none of them sees
the others, and together they spend a multiple of what was left. agent-budget
fixes that at the delegation boundary: each sub-agent's slice is **reserved
from the parent before it starts**, every amount it books is booked on the
parent at the same moment, and the unused part goes back when it finishes.

```bash
pip install agent-budget            # no dependencies
pip install "agent-budget[a2a]"     # plus the a2a-python plugins
```

Until the first upload reaches PyPI, install from GitHub:
`pip install "agent-budget[a2a] @ git+https://github.com/Mattbusel/agent-budget"`.

## Child scopes

```python
from agent_budget import Budget

orchestrator = Budget(1.00, name="orchestrator")

with orchestrator.child("researcher", usd=0.60) as research:
    hold = research.reserve(0.06)        # before the model call; refused if it does not fit
    # ... call the model ...
    hold.settle(0.052)                   # what it really cost

    with research.child("fetcher", fraction=0.5) as fetch:   # half of what research has left
        fetch.reserve(0.01).settle(0.01)

print(orchestrator.spent_usd)            # 0.062: every level, as it happens
print(orchestrator.remaining_usd)        # 0.938: the unused slice came back on exit
```

- A slice that does not fit raises `BudgetExhausted` before the sub-agent
  spends anything.
- A closed scope (and everything below it) raises `ScopeClosed`; calls
  already in flight still settle.
- `kill()` stops a scope and its whole subtree. `max_depth` (default 8)
  raises `DepthExceeded`.
- `own_cost_usd` + `child_cost_usd` == `spent_usd` on every scope.
- Thread-safe (one lock per tree) and safe in asyncio tasks;
  `current_budget()` is the innermost scope entered with `with` in the
  current thread or task. One budget can be entered by many threads and
  tasks at once.

## Across agents: A2A, W3C baggage, MCP

`BudgetContext` is the budget a caller hands down and the cost a callee
reports back, in the shape of the
[A2A budget extension draft](https://github.com/a2aproject/A2A/issues/2121)
(`budget{cost, tokens, deadline}`, `incurred{cost, aggregatedChildCost,
tokens, delegations}`), plus a `reservation{id, amount, currency, expires}`
for the slice the caller reserved.

```python
from agent_budget import Budget, outbound, to_a2a_metadata, from_a2a_metadata, to_baggage

with Budget(1.00).child("remote", usd=0.30) as slice_:
    ctx = outbound(slice_)
    metadata = to_a2a_metadata(ctx)      # {"https://a2a-protocol.org/extensions/budget/v1": {...}}
    header = to_baggage(ctx)             # "llm.budget.cost.amount=0.3,llm.budget.cost.currency=USD,..."
```

- **A2A**: the whole context under the extension URI in request, message or
  task metadata.
- **W3C baggage**: `llm.budget.*` members, so anything that already
  propagates OpenTelemetry baggage carries the budget.
- **MCP**: MCP reserves `params._meta.baggage` for W3C baggage (SEP-414);
  `inject_mcp_meta` / `extract_mcp_meta` put the same members there.

Money is a decimal string on the wire, exact to the nano-dollar. Fields
nobody measured are left out, never written as zero.

## a2a-python plugins

```python
from a2a import plugins
from agent_budget.a2a_plugin import BudgetClientPlugin, BudgetExecutionPlugin

# Caller: requests made inside a scope carry its budget; replies are booked on it.
plugins.set_global(plugins.sequential(BudgetClientPlugin()))
with orchestrator.child("remote-researcher", usd=0.30):
    async for event in client.send_message(request):
        ...

# Callee: each execution runs inside a Budget of what the caller allowed,
# and every task update reports the cumulative cost back.
handler = DefaultRequestHandlerV2(..., plugins=plugins.sequential(BudgetExecutionPlugin(max_usd=5.00)))
# inside AgentExecutor.execute: current_budget().reserve(...).settle(...)
```

## Same format as the Rust crate

[llm-budget](https://crates.io/crates/llm-budget) 0.8 (Rust) has the same
scopes and the same contexts. Both test suites read one set of golden
fixtures (`tests/fixtures/budget_context/`), so a Rust orchestrator and a
Python agent agree to the last digit.

More: [docs/guide.md](docs/guide.md). Adding it to another project (dependencies, typing, stability, the five-line integration): [docs/integrating.md](docs/integrating.md).

MIT license.
