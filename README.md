# agent-budget: sub-agents spend money their parent no longer has

[![PyPI](https://img.shields.io/pypi/v/llm-agent-budget.svg)](https://pypi.org/project/llm-agent-budget/)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-agent-budget.svg)](https://pypi.org/project/llm-agent-budget/)
[![CI](https://github.com/Mattbusel/agent-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/agent-budget/actions/workflows/ci.yml)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy%20--strict-blue.svg)](https://github.com/Mattbusel/agent-budget/blob/main/docs/integrating.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mattbusel/agent-budget/blob/main/LICENSE)

When an orchestrator hands work to several sub-agents at once, each one checks the same "remaining" figure, none of them sees the others, and together they spend a multiple of what was left. The protocols they talk over have no place to say how much a delegated task may spend. The open A2A proposal for one puts it plainly: "There's no way to tell a downstream agent 'you have $0.50 remaining.'" ([a2aproject/A2A #2121](https://github.com/a2aproject/A2A/issues/2121)). agent-budget reserves each sub-agent's slice from the parent **before it starts**, books every amount on the parent as it happens, and returns the unused part when it finishes.

```bash
pip install llm-agent-budget            # one dependency: llm-prices (no dependencies of its own)
pip install "llm-agent-budget[a2a]"     # plus the a2a-python plugins
```

## Fail closed, on purpose

A soft limit records the overspend after it has happened. Since 0.2 agent-budget refuses every call it cannot price exactly:

| What other trackers do | What agent-budget does | Loud opt-out (warns on every call) |
|---|---|---|
| Unknown model priced at $0 or a $1/$3 default | `UnknownModel`, call refused | `Budget(..., allow_unpriced=True)` |
| Cached tokens billed as plain input | `CachePriceMissing` unless the cache rate is listed | `allow_unpriced=True` |
| Price table pasted into code, never updated | `StalePrice` once a price is 90 days old | `Pricer(allow_stale=True)`, `max_price_age_days=` |
| Check spend, make the call, book it after | refused before the call is made | `Budget(..., unsafe_soft_limit=True)` |
| A budget with no cap | `Budget(None)` raises | none |

```python
from agent_budget import Budget

b = Budget(0.10)
hold = b.reserve_for("gpt-4o", {"prompt_tokens": 8_000, "completion_tokens": 1_000})  # $0.03 held
# ... the call, with any client ...
hold.settle_usage("gpt-4o", response.usage)   # OpenAI, Anthropic or Gemini usage, cache tokens included
b.reserve_for("gtp-4o", ...)                  # UnknownModel: refused, not booked at $0
```

Prices come from [llm-prices](https://pypi.org/project/llm-prices/): every row cites the provider's page and the date it was checked. The opt-outs issue an `UnsafeBudgetWarning` on every call they let through (never deduplicated), and log it to the `agent_budget` logger.

## How much would a soft limit have let through?

```bash
agent-budget audit calls.jsonl --cap 1      # or: python -m agent_budget audit ...
```

One JSON object per line: `model`, the provider's `usage` block, and optionally `booked_usd` (what your tracker booked). On twelve gpt-5.5 calls booked at a $1/$3 default (the [PraisonAI](https://github.com/MervinPraison/PraisonAI/pull/6433) pattern):

```text
Soft limit (checks spend before each call, books after, $0 for unpriced models):
  stopped the run at call 10, after letting 9 calls through
  tracker booked $1.0350; those calls really cost $5.8500
  MONEY PAST THE CAP: $4.8500 (5.85x the cap); real spend crossed $1.00 at call 2, then 7 more calls ran

Hard cap, fail closed (agent-budget default):
  admitted 1 calls, spent $0.6500 of $1.00
```

Exit status 2 when money got past the cap (a CI gate), `--json` for machines. Same report as the Rust crate's `llm-budget audit`.

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

## Strands Agents

```python
from strands import Agent
from agent_budget.strands import Price, SpendCeiling

agent = Agent(tools=[researcher], plugins=[SpendCeiling(usd=0.50, price=Price(input=3.00, output=15.00))])
```

A Strands plugin (`pip install "llm-agent-budget[strands]"`): each invocation can
spend at most `usd`, every model call reserves its worst case before it is
sent, and an agent used as a tool gets its slice from the caller's budget.
See [docs/strands.md](docs/strands.md).

## Same format as the Rust crate

[llm-budget](https://crates.io/crates/llm-budget) 0.8 (Rust) has the same
scopes and the same contexts. Both test suites read one set of golden
fixtures (`tests/fixtures/budget_context/`), so a Rust orchestrator and a
Python agent agree to the last digit.

More: [docs/guide.md](docs/guide.md). Adding it to another project (dependencies, typing, stability, the five-line integration): [docs/integrating.md](docs/integrating.md).

MIT license.
