"""SpendCeiling against real Strands agents, with a scripted model (offline)."""

import json

import pytest

pytest.importorskip("strands")

from strands import Agent, tool  # noqa: E402
from strands.models import Model  # noqa: E402

from agent_budget import Budget  # noqa: E402
from agent_budget.strands import Price, SpendCeiling  # noqa: E402

# 100 input + 500 output tokens at $3 / $15 per million: $0.0078 per call.
PRICE = Price(input=3.00, output=15.00)
USAGE = {"inputTokens": 100, "outputTokens": 500, "totalTokens": 600}
CALL_USD = 0.0078


class ScriptedModel(Model):
    """Answers from a script; each step is a list of tool calls, or text to end the turn."""

    def __init__(self, script, model_id="scripted"):
        self.script = list(script)
        self.calls = 0
        self.config = {"model_id": model_id, "max_tokens": 600}

    def get_config(self):
        return self.config

    def update_config(self, **config):
        self.config.update(config)

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, tool_choice=None, **kwargs):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(step, str):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step}}}
            yield {"contentBlockStop": {}}
            stop = "end_turn"
        else:
            for i, (name, args) in enumerate(step):
                use_id = f"t{self.calls}_{i}"
                yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": use_id}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
                yield {"contentBlockStop": {}}
            stop = "tool_use"
        yield {"messageStop": {"stopReason": stop}}
        yield {"metadata": {"usage": dict(USAGE), "metrics": {"latencyMs": 1}}}


@tool
def search(query: str) -> str:
    """Search for something."""
    return f"results for {query}"


def looping_model(tool_turns=10):
    return ScriptedModel([[("search", {"query": f"q{i}"})] for i in range(tool_turns)] + ["done"])


def test_unbounded_agent_spends_every_call():
    model = looping_model()
    Agent(model=model, tools=[search], callback_handler=None)("go")
    assert model.calls == 11  # 11 * $0.0078 = $0.0858


def test_ceiling_stops_the_agent_before_the_call_that_does_not_fit():
    model = looping_model()
    ceiling = SpendCeiling(usd=0.04, price=PRICE)
    result = Agent(model=model, tools=[search], plugins=[ceiling], callback_handler=None)("go")

    # Each call reserves at least 600 output tokens ($0.009), so the 5th does not fit after 4.
    assert model.calls == 4
    assert ceiling.last_spent_usd == pytest.approx(4 * CALL_USD)
    assert ceiling.last_spent_usd <= 0.04
    assert "spend ceiling reached" in str(result)


def test_each_invocation_gets_its_own_ceiling():
    model = looping_model(tool_turns=0)
    ceiling = SpendCeiling(usd=0.012, price=PRICE)
    agent = Agent(model=model, plugins=[ceiling], callback_handler=None)
    agent("one")
    agent("two")
    assert model.calls == 2
    assert ceiling.spent_usd == pytest.approx(2 * CALL_USD)


def test_shared_budget_caps_all_invocations_together():
    service = Budget(0.025, name="service")
    model = looping_model(tool_turns=0)
    ceiling = SpendCeiling(usd=0.012, price=PRICE, budget=service)
    agent = Agent(model=model, plugins=[ceiling], callback_handler=None)
    agent("one")
    agent("two")
    result = agent("three")  # $0.0094 left, a $0.012 slice does not fit

    assert model.calls == 2
    assert "not started" in str(result)
    assert service.spent_usd == pytest.approx(2 * CALL_USD)
    assert service.reserved_usd == 0


def test_unpriced_model_is_cancelled():
    model = looping_model(tool_turns=0)
    ceiling = SpendCeiling(usd=1.0, price={"other-model": PRICE})
    result = Agent(model=model, plugins=[ceiling], callback_handler=None)("go")
    assert model.calls == 0
    assert "no price for model 'scripted'" in str(result)


def _team(with_ceilings):
    """An orchestrator that starts four researchers in one turn, then answers."""
    researchers = []
    for i in range(4):
        plugins = [SpendCeiling(usd=0.02, price=PRICE)] if with_ceilings else []
        # Each researcher: one search, then an answer (2 calls, $0.0156).
        researchers.append(
            Agent(
                name=f"researcher_{i}",
                description="Researches a topic.",
                model=ScriptedModel([[("search", {"query": "x"})], "found it"]),
                tools=[search],
                plugins=plugins,
                callback_handler=None,
            )
        )
    fanout = [(f"researcher_{i}", {"input": f"topic {i}"}) for i in range(4)]
    orchestrator_model = ScriptedModel([fanout, "summary"])
    root = SpendCeiling(usd=0.05, price=PRICE) if with_ceilings else None
    orchestrator = Agent(
        model=orchestrator_model,
        tools=researchers,
        plugins=[root] if root else [],
        callback_handler=None,
    )
    return orchestrator, orchestrator_model, researchers, root


def test_parallel_sub_agents_without_ceilings():
    orchestrator, orch_model, researchers, _ = _team(with_ceilings=False)
    orchestrator("research")
    calls = orch_model.calls + sum(r.model.calls for r in researchers)
    assert calls == 10  # $0.078 for what was meant to be a $0.05 job


def test_parallel_sub_agents_draw_slices_from_the_caller():
    orchestrator, orch_model, researchers, root = _team(with_ceilings=True)
    orchestrator("research")

    started = [r for r in researchers if r.model.calls]
    # $0.05 - $0.0078 for the orchestrator's first call leaves $0.0422: room for two $0.02 slices.
    assert len(started) == 2
    refused = [m for m in orchestrator.messages if m["role"] == "user" for b in m["content"] if "toolResult" in b
               for c in b["toolResult"]["content"] if "not started" in c.get("text", "")]
    assert len(refused) == 2
    calls = orch_model.calls + sum(r.model.calls for r in researchers)
    assert root.last_spent_usd == pytest.approx(calls * CALL_USD)
    assert root.last_spent_usd <= 0.05


def test_sub_agent_alone_uses_its_own_ceiling():
    _, _, researchers, _ = _team(with_ceilings=True)
    researchers[0]("standalone")
    assert researchers[0].model.calls == 2


def test_price_cost_handles_both_cache_conventions():
    price = Price(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
    # Bedrock / Anthropic: cache tokens on top of inputTokens.
    separate = {"inputTokens": 100, "outputTokens": 10, "totalTokens": 1310,
                "cacheReadInputTokens": 1000, "cacheWriteInputTokens": 200}
    # OpenAI: cache tokens inside inputTokens.
    folded = {"inputTokens": 1300, "outputTokens": 10, "totalTokens": 1310,
              "cacheReadInputTokens": 1000, "cacheWriteInputTokens": 200}
    expected = (100 * 3.0 + 10 * 15.0 + 1000 * 0.3 + 200 * 3.75) / 1e6
    assert price.cost(separate) == pytest.approx(expected)
    assert price.cost(folded) == pytest.approx(expected)
    assert Price(input=3.0, output=15.0).cost(separate) == pytest.approx((1300 * 3.0 + 150.0) / 1e6)


def test_usd_must_be_positive():
    with pytest.raises(ValueError):
        SpendCeiling(usd=0, price=PRICE)
