# Recipe: a USD cap across parallel pydantic-ai delegates

pydantic-ai's `UsageLimits` checks requests, tokens and `cost_limit`, and a delegate that
gets `usage=ctx.usage` adds its usage to the parent's. When the parent's
model asks for several delegations in one turn, pydantic-ai runs those tool
calls at the same time, and every delegate checks the shared counters before
any of them has added to them. That is the overshoot in
[pydantic/pydantic-ai#10142](https://github.com/pydantic/pydantic-ai/issues/10142).

agent-budget fixes this at the delegation boundary with two small pieces.
The full script is [examples/pydantic_ai_delegation.py](../examples/pydantic_ai_delegation.py)
and runs offline.

## 1. Reserve and settle every model request

```python
from pydantic_ai.models.wrapper import WrapperModel
from agent_budget import current_budget


class BudgetedModel(WrapperModel):
    def __init__(self, wrapped, *, price_in, price_out, max_output_tokens=1024):
        super().__init__(wrapped)
        self.price_in, self.price_out, self.max_output_tokens = price_in, price_out, max_output_tokens

    def cost(self, input_tokens, output_tokens):
        return input_tokens * self.price_in + output_tokens * self.price_out

    async def request(self, messages, model_settings, model_request_parameters):
        budget = current_budget()
        if budget is None:
            return await super().request(messages, model_settings, model_request_parameters)
        max_out = (model_settings or {}).get("max_tokens", self.max_output_tokens)
        with budget.reserve(self.cost(estimate_prompt_tokens(messages), max_out)) as hold:
            response = await super().request(messages, model_settings, model_request_parameters)
            hold.settle(self.cost(response.usage.input_tokens, response.usage.output_tokens))
        return response
```

- The reservation is the most the request can cost (prompt estimate plus
  `max_tokens`). If it does not fit, `BudgetExhausted` is raised and the
  request is never sent.
- The settlement is what the response's usage says it cost. An exception
  or a cancelled task releases the hold.
- The example also wraps `request_stream` the same way, settling from the
  stream's usage when it ends.

Wrap the real model: `BudgetedModel(OpenAIChatModel("gpt-4o"), price_in=2.5e-6, price_out=10e-6)`.

## 2. Run each delegate in a child scope

```python
@orchestrator.tool
async def research(ctx: RunContext[None], topic: str) -> str:
    parent = current_budget()
    with parent.child(f"research:{topic}", usd=0.012):
        result = await researcher.run(f"Research {topic} power.", usage=ctx.usage)
    return result.output


with Budget(0.04, name="orchestrator"):
    result = await orchestrator.run("Compare renewable energy sources.")
```

`parent.child(...)` reserves the delegate's slice from the orchestrator
before the delegate starts, atomically, so four delegations started in the
same turn can never be admitted for more than the orchestrator holds. Inside
the `with`, `current_budget()` is the slice, so every request the delegate's
`BudgetedModel` makes is reserved against it. What the delegate spends is
booked on the orchestrator as it happens, and the unused part of the slice
returns when the `with` exits.

## What the example prints

The orchestrator asks for four topics in one turn on a $0.04 budget with
$0.012 slices:

```
  research:geothermal: not started (budget 'orchestrator' cannot hold $0.012000000: $0.003662500 left)
  research:solar: spent $0.003145 of its $0.012 slice, $0.008855 back
  research:wind: spent $0.003145 of its $0.012 slice, $0.008855 back
  research:hydro: spent $0.003145 of its $0.012 slice, $0.008855 back
answer: Summary of the findings.
orchestrator: spent $0.012435 of $0.04 (own $0.003000, delegates $0.009435), $0.000000 still held
```

Three slices fit. The fourth is refused before its delegate sends anything,
and the tool tells the orchestrator so. Nothing stays reserved at the end.

For delegates in other processes, pass the slice over A2A with
`agent_budget.a2a_plugin` (see the [guide](guide.md#a2a-python)).
