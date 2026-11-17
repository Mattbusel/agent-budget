# agent-budget guide

## Why child scopes

An orchestrator that starts several sub-agents at once and tells each one
"you have $X left" gives every sub-agent a figure that is valid when it reads
it and stale the moment a sibling spends. Each one stays under $X; together
they spend a multiple of it. Measured with `examples/fanout.py` (four
sub-agents, eight gpt-4o-sized calls each, $1.00 cap):

| Setup | Calls | Spent |
|---|---|---|
| remainder passed down | 32 | $1.664 (66% over the cap) |
| `parent.child(.., usd=0.25)` per sub-agent | 16 | $0.832 |

`tests/test_budget.py` has the two-thread version of the same overshoot
($1.60 on a $1.00 cap) and the same two delegates under child scopes (one
gets its slice, the other is refused before it spends anything).

## The rules

1. `Budget.reserve(estimate)` holds headroom before a call, atomically. If it
   does not fit, `BudgetExhausted` is raised and the call is never made.
2. `Reservation.settle(actual)` replaces the hold with the real cost, even
   past the cap (that money is already spent). `release()` drops it. Use
   `with budget.reserve(x) as hold:` around the call so a cancelled task or
   an exception releases the hold at once; a reservation garbage-collected
   unsettled is also released, but only when the collector gets to it.
3. `Budget.child(name, usd=... | fraction=...)` reserves the child's whole
   slice from the parent first. A `fraction` is of what the parent has left
   at that moment.
4. Everything a child books is booked on every ancestor at the same moment,
   out of the child's hold.
5. Closing a child (`with` exit, `close()`) stops new work in it and its
   subtree (`ScopeClosed`) and refunds the part of its hold no in-flight call
   needs.
6. `kill(reason)` stops a scope and its subtree (`BudgetKilled`).
7. Depth is capped (`max_depth`, default 8; `DepthExceeded`).

For every scope, under any interleaving: open holds plus everything booked
never exceed its limit through admission. The only way past is a settlement
above its reservation, which is booked in full and reported (`over_cap`,
`ScopeReceipt.over_hold`). `tests/test_budget.py` checks this with 64
threads and a monitor, an asyncio fan-out with a cancelled task, and a
Hypothesis property test over random scope trees.

## Contexts on the wire

```json
{
  "budget": {"cost": {"amount": "0.3", "currency": "USD"}, "tokens": {"limit": 5000}, "deadline": "2026-12-01T00:00:00Z"},
  "reservation": {"id": "rsv_a2dcda7352c5fe3c882a05e2b66a162a", "amount": "0.3", "currency": "USD"},
  "incurred": {
    "cost": {"amount": "0.04", "currency": "USD", "source": "local"},
    "aggregatedChildCost": {"amount": "0.02", "currency": "USD"},
    "tokens": {"input": 1200, "output": 340, "cacheRead": 1000},
    "delegations": 1
  }
}
```

- `budget` and `incurred` follow the A2A budget extension draft,
  [a2aproject/A2A#2121](https://github.com/a2aproject/A2A/issues/2121),
  under its proposed URI `https://a2a-protocol.org/extensions/budget/v1`.
  The draft is still open. Its author has said money will be decimal strings
  (done), that `aggregatedChildCost` is a provisional name, and that token
  counts gain cache fields. agent-budget follows the draft in minor releases
  and keeps reading what it wrote before.
- `reservation` is this package's addition: the slice the caller reserved,
  with an id unique to that reservation (never shared between siblings, so
  a callee cannot learn how many siblings it has).
- `incurred.cost` is the reporter's own spend. `aggregatedChildCost` is its
  delegates'. This split matches the OpenTelemetry GenAI cost proposal
  (open-telemetry/semantic-conventions-genai#443), where an operation's cost
  must not include nested operations.
- Absent is not zero: what was not measured is left out.
- Reports are cumulative; `IncurredBooker` books only their growth, so a
  resent final task is not booked twice.

### W3C baggage and MCP

The outbound half also travels as `llm.budget.*` baggage members:

```
llm.budget.cost.amount=0.3,llm.budget.cost.currency=USD,llm.budget.reservation.id=rsv_...,llm.budget.reservation.amount=0.3,llm.budget.reservation.currency=USD
```

MCP reserves `params._meta.baggage` for W3C baggage (SEP-414), so
`inject_mcp_meta(meta, ctx)` on the client and `extract_mcp_meta(meta)` on
the server carry a budget into any MCP tool call without an MCP change.
Readers reject malformed input with `ContextError` (and never anything else:
`tests/test_carriers.py` fuzzes them with Hypothesis).

## a2a-python

`agent_budget.a2a_plugin` (install the `a2a` extra; a2a-sdk 1.3+):

- `BudgetClientPlugin`: for `send_message` / `send_message_streaming`
  calls made while a scope is current (or with a fixed `budget=`), writes the
  outbound context into the request metadata, asks for the extension in
  `A2A-Extensions`, and books each reply's `incurred` on the scope.
- `BudgetExecutionPlugin(default_usd=None, max_usd=None, source="local")`:
  reads the budget from the request (request metadata first, then message
  metadata), opens a `Budget` of that amount (capped by `max_usd`; requests
  without one get `default_usd` or no budget), makes it `current_budget()`
  inside `AgentExecutor.execute`, and stamps the cumulative `incurred`
  report on every task, status update and message the agent emits.
- `BudgetClientInterceptor`: the client half for code that passes
  interceptors to `ClientFactory.create`.

`tests/test_a2a_plugin.py` runs all of this over real JSON-RPC and HTTP+JSON
transports (Starlette in-process): the budget arrives, the extension header
arrives, the callee spends inside it, reports are cumulative, the caller
books once, three concurrent delegations from one orchestrator add up.
`examples/a2a_delegation.py` is the same flow as a script.

## Interop with llm-budget (Rust)

[llm-budget](https://crates.io/crates/llm-budget) 0.8 has the same scopes
(`AgentBudget::child`, `Limit::Usd`, `Limit::Fraction`) and the same
contexts (`llm_budget::context`). Both libraries read the same golden
fixtures in `tests/fixtures/budget_context/`; CI checks that the two copies
are identical.
