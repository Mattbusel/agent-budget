"""A USD cap across parallel pydantic-ai delegates, offline.

    pip install llm-agent-budget pydantic-ai-slim
    python examples/pydantic_ai_delegation.py

An orchestrator agent delegates research to a sub-agent through a tool, and
asks for four topics in one turn, so pydantic-ai runs the four tool calls at
the same time. Two pieces make the cap hold:

- ``BudgetedModel`` wraps any pydantic-ai model. Before each request it
  reserves the most the request can cost on ``current_budget()``, and after
  it settles what the response's usage says it cost. A request that does not
  fit raises ``BudgetExhausted`` and is never sent.
- The delegation tool runs the sub-agent inside ``parent.child(...)``: its
  slice is reserved from the orchestrator before it starts, its spend is
  booked on the orchestrator as it happens, and the unused part comes back
  when the tool returns.

The models here are ``FunctionModel`` stand-ins, so the script needs no API
key. Swap in a real model (``BudgetedModel(OpenAIChatModel("gpt-4o"), ...)``)
and the same code caps real spend.
"""

import asyncio
from contextlib import asynccontextmanager

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel

from agent_budget import Budget, BudgetExhausted, current_budget

# gpt-4o list prices, USD per token.
PRICE_IN, PRICE_OUT = 2.50e-6, 10.00e-6


def estimate_prompt_tokens(messages: list[ModelMessage]) -> int:
    """About 4 characters per token. The provider's count replaces it."""
    return sum(len(str(part)) for m in messages for part in m.parts) // 4 + 8


class BudgetedModel(WrapperModel):
    """Reserves before every request on ``current_budget()``, settles after."""

    def __init__(self, wrapped, *, price_in: float, price_out: float, max_output_tokens: int = 1024):
        super().__init__(wrapped)
        self.price_in, self.price_out, self.max_output_tokens = price_in, price_out, max_output_tokens

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.price_in + output_tokens * self.price_out

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        budget = current_budget()
        if budget is None:
            return await super().request(messages, model_settings, model_request_parameters)
        max_out = (model_settings or {}).get("max_tokens", self.max_output_tokens)
        # Raises BudgetExhausted before anything is sent; released on any exception.
        with budget.reserve(self.cost(estimate_prompt_tokens(messages), max_out)) as hold:
            response = await super().request(messages, model_settings, model_request_parameters)
            hold.settle(self.cost(response.usage.input_tokens, response.usage.output_tokens))
        return response

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
        budget = current_budget()
        if budget is None:
            async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as s:
                yield s
            return
        max_out = (model_settings or {}).get("max_tokens", self.max_output_tokens)
        with budget.reserve(self.cost(estimate_prompt_tokens(messages), max_out)) as hold:
            async with super().request_stream(messages, model_settings, model_request_parameters, run_context) as s:
                yield s
            usage = s.usage
            hold.settle(self.cost(usage.input_tokens, usage.output_tokens))


# --- offline stand-ins for the two models ------------------------------------

TOPICS = ["solar", "wind", "hydro", "geothermal"]


def orchestrator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:  # first turn: delegate all four topics at once
        return ModelResponse(parts=[ToolCallPart("research", {"topic": t}) for t in TOPICS])
    return ModelResponse(parts=[TextPart("Summary of the findings.")])


def researcher_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Findings: " + "lorem ipsum " * 150)])


# --- the agents --------------------------------------------------------------

orchestrator = Agent(BudgetedModel(FunctionModel(orchestrator_model), price_in=PRICE_IN, price_out=PRICE_OUT))
researcher = Agent(BudgetedModel(FunctionModel(researcher_model), price_in=PRICE_IN, price_out=PRICE_OUT))
SLICE_USD = 0.012


@orchestrator.tool
async def research(ctx: RunContext[None], topic: str) -> str:
    """Delegate one topic to the researcher, on its own slice of the budget."""
    parent = current_budget()
    try:
        with parent.child(f"research:{topic}", usd=SLICE_USD) as scope:
            result = await researcher.run(f"Research {topic} power.", usage=ctx.usage)
            receipt = scope.close()
    except BudgetExhausted as e:
        print(f"  research:{topic}: not started ({e})")
        return f"No budget left to research {topic}."
    print(f"  {receipt.scope}: spent ${receipt.spent_usd:.6f} of its ${receipt.held_usd:.3f} slice, ${receipt.released_usd:.6f} back")
    return result.output


async def main():
    # $0.04 covers three $0.012 slices and the orchestrator's own calls, not four slices.
    with Budget(0.04, name="orchestrator") as budget:
        result = await orchestrator.run("Compare renewable energy sources.")
    print(f"answer: {result.output}")
    print(
        f"orchestrator: spent ${budget.spent_usd:.6f} of ${budget.limit_usd:.2f} "
        f"(own ${budget.own_cost_usd:.6f}, delegates ${budget.child_cost_usd:.6f}), "
        f"${budget.reserved_usd:.6f} still held"
    )
    assert budget.spent_usd <= budget.limit_usd


if __name__ == "__main__":
    asyncio.run(main())
