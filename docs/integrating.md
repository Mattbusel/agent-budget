# Integrating agent-budget into another project

For maintainers deciding whether to depend on agent-budget, and for anyone
adding a hard USD cap to an agent framework, an orchestrator or an A2A / MCP
service. The snippets here run in CI (`examples/integrate.py`, also checked
with `mypy --strict`).

## At a glance

| | |
|---|---|
| License | MIT ([LICENSE](../LICENSE)) |
| Python | 3.9+ for the core, 3.10+ for the `a2a` extra (a2a-sdk's own minimum). Tested on 3.9 to 3.13 in CI. |
| Dependencies | One: [llm-prices](https://pypi.org/project/llm-prices/) 1.8+ (pure Python, no dependencies of its own), the dated price table. `a2a` extra: a2a-sdk 1.3+. |
| Typing | Ships `py.typed`; the package passes `mypy --strict` in CI. |
| Side effects | No network, no files, no threads, no global state beyond a `contextvars` variable for `current_budget()` and an "always" warnings filter for `UnsafeBudgetWarning` (so opt-outs stay loud). |
| Concurrency | One lock per budget tree; safe from threads and asyncio tasks. One budget can be entered by many tasks at once. |
| Money | Exact integer nano-dollars inside, never float accumulation. |
| Maintainer | Matthew Busel ([Tensorust LLC](https://tensorust.com/open-source/)). Issues and PRs on [GitHub](https://github.com/Mattbusel/agent-budget). |

## Add it

```bash
pip install llm-agent-budget            # depends on llm-prices only
pip install "llm-agent-budget[a2a]"     # plus the a2a-sdk plugins
```

As an optional extra of your own package:

```toml
[project.optional-dependencies]
budget = ["llm-agent-budget>=0.2,<0.3"]
```

and import it lazily where the cap is applied, so users who do not ask for
budgets never import it.

## Fail closed

Since 0.2, `Budget.reserve_for(model, usage)` / `Reservation.settle_usage` /
`Budget.record_usage` price calls from llm-prices and refuse what they cannot
price exactly: `UnknownModel`, `CachePriceMissing`, `StalePrice` (older than
90 days), `Unpriced`, all `PriceError(BudgetError)`. A budget needs a hard
cap. The opt-outs (`allow_unpriced=True`, `unsafe_soft_limit=True`,
`Pricer(allow_stale=True)`) issue an `UnsafeBudgetWarning` on every call they
let through. `agent-budget audit calls.jsonl --cap 5` shows what a soft
limit would have let past a cap on a log of real calls.

## The five-line integration

Works around any client. Reserve the worst case before the call, settle the
real cost after it:

```python
from agent_budget import Budget

budget = Budget(5.00, name="my-agent")             # $5 hard cap
with budget.reserve(0.01) as hold:                 # raises BudgetExhausted if it would not fit
    cost = call_model("...")                       # the real call, with any client
    hold.settle(cost)                              # books what it really cost
```

- `BudgetExhausted`, `BudgetKilled` and `ScopeClosed` all derive from
  `BudgetError`; catch that one to handle every refusal. The call is never
  made.
- Leaving the `with` block without settling (an exception, a cancelled task)
  releases the hold, so nothing is booked for a call that did not happen.
- `budget.charge_force(cost)` books a cost you only hear about afterwards,
  even past the cap.

## Pick the integration point

| Your code has | Use | See |
|---|---|---|
| Any client or SDK | `Budget.reserve` + `Reservation.settle` | above, `examples/integrate.py` |
| An orchestrator with sub-agents | `budget.child(name, usd=...)` or `fraction=...` | `examples/fanout.py`, [guide.md](guide.md) |
| Code that should find the active budget itself | `current_budget()` inside `with budget:` (per thread and asyncio task) | [guide.md](guide.md) |
| pydantic-ai | a `WrapperModel` that reserves and settles each request, delegates in child scopes | [pydantic-ai.md](pydantic-ai.md) |
| A2A (a2a-sdk 1.3+) | `BudgetClientPlugin` (caller) and `BudgetExecutionPlugin` (callee), or `BudgetClientInterceptor` | `examples/a2a_delegation.py` |
| OpenTelemetry baggage | `to_baggage` / `from_baggage` / `merge_baggage` (`llm.budget.*` members) | README |
| MCP tool servers | `inject_mcp_meta` / `extract_mcp_meta` on `params._meta` | README |
| A Rust service in the same call graph | [llm-budget](https://crates.io/crates/llm-budget) 0.8: same scopes, same wire format, same golden fixtures | |

## Stability

- The public API is what `agent_budget.__all__` and
  `agent_budget.a2a_plugin.__all__` list. Names starting with `_` are
  private.
- While the version is 0.x, a breaking change bumps the minor version (0.1
  to 0.2) and is listed in [CHANGELOG.md](../CHANGELOG.md). Patch releases do
  not break working code, so `>=0.2,<0.3` is a safe pin.
- The wire format (`BudgetContext` JSON, baggage keys, the A2A extension
  URI) follows the A2A budget extension draft (a2aproject/A2A#2121). A change
  to it is a minor release, made in step with llm-budget.
