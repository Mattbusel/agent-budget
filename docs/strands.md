# Strands Agents: a USD spend ceiling

Strands invocation limits count turns and tokens. `SpendCeiling` adds a cap in
dollars that is checked before each model call is sent, and that covers agents
used as tools.

```bash
pip install "llm-agent-budget[strands]"     # strands-agents 1.43 or later
```

## Usage

```python
from strands import Agent
from agent_budget.strands import Price, SpendCeiling

# Priced from llm-prices by the model's id; fail closed (unknown, stale or
# unpriced models cancel the call before it is sent).
agent = Agent(
    tools=[...],
    plugins=[SpendCeiling(usd=0.50)],
)
agent("Research the topic and write a summary.")

# Or your own rates, USD per million tokens. Cache prices are explicit: a call
# with cache tokens and no cache price raises CachePriceMissing.
sonnet = Price(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75)
SpendCeiling(usd=0.50, price=sonnet)
```

Every invocation of `agent` can spend at most $0.50. Before each model call the
plugin reserves the most that call can cost: the projected input tokens Strands
computes, plus the output-token cap (`max_tokens` from the model config, or
`max_output_tokens=`). If the reservation does not fit, the call is cancelled
before it is sent and the agent ends its turn with
`spend ceiling reached: budget '<agent name>' cannot hold $...: $... left`. After the
call, the reservation is replaced by what the response's usage says it cost.

`ceiling.last_spent_usd` is what the last invocation spent, and
`ceiling.spent_usd` the total over all invocations.

## Agents used as tools

```python
researcher = Agent(
    name="researcher",
    description="Researches one topic.",
    tools=[...],
    plugins=[SpendCeiling(usd=0.20, price=sonnet)],
)
orchestrator = Agent(
    tools=[researcher],                      # or researcher.as_tool()
    plugins=[SpendCeiling(usd=0.50, price=sonnet)],
)
```

When the orchestrator calls `researcher`, the researcher's $0.20 is reserved
from the orchestrator's budget before the researcher starts. What the
researcher spends is booked on the orchestrator as it happens, and the part it
did not use goes back when it finishes. If the orchestrator has less than $0.20
left, the researcher does not start and the tool result says
`spend ceiling: researcher not started, ...`.

This matters when the model calls several agents in one turn. Strands runs
those tool calls at the same time, and each one would otherwise see the same
remaining amount. With slices, they can never be admitted for more than the
orchestrator has left. Called on its own, `researcher` uses its $0.20 as a
normal per-invocation ceiling.

Measured in `tests/test_strands.py` with a scripted model ($0.0078 per call):
an orchestrator with a $0.05 ceiling asks four researchers (two calls each) to
start in the same turn.

| Setup | Model calls | Spent |
|---|---|---|
| no ceilings | 10 | $0.078 |
| `SpendCeiling(usd=0.05)` on the orchestrator, `usd=0.02` on each researcher | 6 | $0.0468 (two researchers refused before they spent anything) |

## One budget for many invocations

```python
from agent_budget import Budget

service = Budget(100.00, name="nightly-jobs")
ceiling = SpendCeiling(usd=0.50, price=sonnet, budget=service)
```

Each invocation still gets at most $0.50, and all invocations together stay
within $100. An invocation that cannot get its $0.50 does not start.

## Prices

`price=` takes a `Price`, a dict of model id to `Price`, or a function from
model id to `Price`. The model id comes from the model config (`model_id`). A
model call with no price is cancelled, so an unpriced model cannot spend past
the ceiling.

Strands providers report cache tokens two ways: Bedrock and Anthropic report
them on top of `inputTokens`, OpenAI folds them into it. `Price.cost` tells the
two apart the same way Strands' own metrics do
(`inputTokens + outputTokens == totalTokens` means they are folded in) and
prices each counter once.

## What it does not cover

- Model calls that do not fire `BeforeModelCallEvent`: `structured_output`
  calls and auxiliary calls (summarizing conversation managers, steering,
  classifiers). Their cost is not counted.
- Sub-agents built by the `subagent` tool. Only agents passed as tools
  (`tools=[agent]` or `agent.as_tool()`) get a slice of the caller's budget.
- The reservation for the first call of an invocation uses Strands' estimate
  of the prompt. If a provider bills more input than that estimate, the
  settled cost can exceed the reservation; the overrun is booked, and later
  calls see it.
- Agents in TypeScript. This package is Python only.
