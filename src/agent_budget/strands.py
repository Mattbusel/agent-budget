"""A USD spend ceiling for Strands Agents (``pip install "llm-agent-budget[strands]"``).

:class:`SpendCeiling` is a Strands plugin. Each invocation of the agent runs in
a budget of ``usd``; every model call reserves its worst case (projected input
plus the output-token cap) before it is sent and settles what the response's
usage says it cost. A call that does not fit is cancelled before it is sent,
and the agent stops with a message saying why::

    from strands import Agent
    from agent_budget.strands import Price, SpendCeiling

    sonnet = Price(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75)  # USD per 1M tokens
    researcher = Agent(name="researcher", plugins=[SpendCeiling(usd=0.25, price=sonnet)])
    orchestrator = Agent(tools=[researcher], plugins=[SpendCeiling(usd=1.00, price=sonnet)])

When an agent with a ``SpendCeiling`` runs as a tool of another agent with one
(``tools=[agent]`` or ``agent.as_tool()``), its ``usd`` is reserved from the
caller's budget before it starts, its spend is booked on the caller as it
happens, and the unused part goes back when it finishes. Sub-agents started in
the same turn can never be admitted for more than the caller has left.

Pass ``budget=`` (an :class:`agent_budget.Budget`) to draw every invocation from
one shared budget, for example one ceiling for a whole service.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.plugins import Plugin, hook

from .budget import Budget, BudgetError, Reservation, Scope

__all__ = ["Price", "SpendCeiling"]

DEFAULT_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class Price:
    """Model prices in USD per million tokens. Cache prices default to the input price."""

    input: float
    output: float
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None

    def cost(self, usage: Mapping[str, Any]) -> float:
        """USD cost of one Strands ``Usage``.

        Providers either fold cache tokens into ``inputTokens`` (OpenAI) or
        report them on top (Bedrock, Anthropic); as in Strands' own metrics,
        ``inputTokens + outputTokens == totalTokens`` means they are folded in.
        """
        inp = int(usage.get("inputTokens", 0))
        out = int(usage.get("outputTokens", 0))
        read = int(usage.get("cacheReadInputTokens", 0))
        write = int(usage.get("cacheWriteInputTokens", 0))
        if inp + out == int(usage.get("totalTokens", inp + out)):
            inp = max(0, inp - read - write)
        cr = self.input if self.cache_read is None else self.cache_read
        cw = self.input if self.cache_write is None else self.cache_write
        return (inp * self.input + out * self.output + read * cr + write * cw) / 1e6

    def worst_case(self, input_tokens: int, output_tokens: int) -> float:
        """The most a call with this prompt and output cap can cost."""
        cache_rates = [r for r in (self.cache_read, self.cache_write) if r is not None]
        per_input = max([self.input, *cache_rates])
        return (input_tokens * per_input + output_tokens * self.output) / 1e6


PriceSpec = Union[Price, Mapping[str, Price], Callable[[Optional[str]], Optional[Price]]]

# Sub-agent -> budgets of the callers running it as a tool right now.
_callers: "weakref.WeakKeyDictionary[Any, list[Budget]]" = weakref.WeakKeyDictionary()
_callers_lock = threading.Lock()


@dataclass
class _Run:
    budget: Budget
    hold: Optional[Reservation] = None


class SpendCeiling(Plugin):
    """Cap what each invocation of an agent spends, in USD, before the money is spent.

    Args:
        usd: The most one invocation may spend, its sub-agents included.
        price: A :class:`Price`, a mapping of model id to :class:`Price`, or a
            callable from model id to :class:`Price`. A model call with no price
            is cancelled, so an unpriced model cannot spend past the ceiling.
        budget: Draw each invocation's ``usd`` from this budget instead of
            starting a fresh one, so all invocations together stay within it.
            Ignored when the agent runs as a tool of an agent with a ``SpendCeiling``.
        max_output_tokens: Output tokens to reserve per call. Defaults to the
            model config's ``max_tokens``, else 4096.
    """

    name = "agent-budget"

    def __init__(
        self,
        usd: float,
        *,
        price: PriceSpec,
        budget: Optional[Budget] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not usd > 0:
            raise ValueError(f"usd must be > 0, got {usd!r}")
        self.usd = usd
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self._price = price
        self._runs: dict[int, _Run] = {}
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        """Total spent by finished invocations, sub-agents included."""
        self.last_spent_usd = 0.0
        """What the last finished invocation spent, sub-agents included."""

    # ── invocation: open and close the budget ────────────────────────────
    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _open(self, event: BeforeInvocationEvent) -> None:
        agent = event.agent
        label = getattr(agent, "name", None) or "agent"
        parent = _current_caller(agent) or self.budget
        try:
            budget = parent.child(label, usd=self.usd) if parent is not None else Budget(self.usd, name=label)
        except BudgetError as err:
            event.cancel = f"spend ceiling: {label} not started, {err}"
            return
        with self._lock:
            self._runs[id(agent)] = _Run(budget)

    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _close(self, event: AfterInvocationEvent) -> None:
        with self._lock:
            run = self._runs.pop(id(event.agent), None)
        if run is None:
            return
        if run.hold is not None:
            run.hold.release()
        spent = run.budget.close().spent_usd if isinstance(run.budget, Scope) else run.budget.spent_usd
        with self._lock:
            self.last_spent_usd = spent
            self.spent_usd += spent

    # ── model calls: reserve the worst case, settle the real cost ────────
    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _reserve(self, event: BeforeModelCallEvent) -> None:
        run = self._runs.get(id(event.agent))
        if run is None:
            return
        model_id, max_out = _model_limits(event.agent)
        price = self._price_for(model_id)
        if price is None:
            event.cancel = f"spend ceiling: no price for model {model_id!r}"
            return
        estimate = price.worst_case(event.projected_input_tokens or 0, self.max_output_tokens or max_out)
        try:
            run.hold = run.budget.reserve(estimate)
        except BudgetError as err:
            event.cancel = f"spend ceiling reached: {err}"

    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _settle(self, event: AfterModelCallEvent) -> None:
        run = self._runs.get(id(event.agent))
        if run is None or run.hold is None:
            return
        hold, run.hold = run.hold, None
        usage = _usage_of(event)
        price = self._price_for(_model_limits(event.agent)[0])
        if usage is None or price is None:
            hold.release()
            return
        hold.settle(price.cost(usage))

    # ── agents used as tools: their budget is a slice of the caller's ────
    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _enter_tool(self, event: BeforeToolCallEvent) -> None:
        sub, run = _sub_agent(event.selected_tool), self._runs.get(id(event.agent))
        if sub is None or run is None:
            return
        with _callers_lock:
            _callers.setdefault(sub, []).append(run.budget)

    @hook  # type: ignore[call-overload]  # strands types hook() for functions, not methods
    def _exit_tool(self, event: AfterToolCallEvent) -> None:
        sub, run = _sub_agent(event.selected_tool), self._runs.get(id(event.agent))
        if sub is None or run is None:
            return
        with _callers_lock:
            stack = _callers.get(sub, [])
            if run.budget in stack:
                stack.remove(run.budget)
            if not stack:
                _callers.pop(sub, None)

    def _price_for(self, model_id: Optional[str]) -> Optional[Price]:
        if isinstance(self._price, Price):
            return self._price
        if isinstance(self._price, Mapping):
            return self._price.get(model_id or "")
        return self._price(model_id)


def _current_caller(agent: Any) -> Optional[Budget]:
    with _callers_lock:
        stack = _callers.get(agent)
        return stack[-1] if stack else None


def _sub_agent(tool: Any) -> Any:
    """The agent behind an agent-as-tool, or None for any other tool."""
    sub = getattr(tool, "agent", None)
    return sub if sub is not None and hasattr(sub, "hooks") else None


def _model_limits(agent: Any) -> tuple[Optional[str], int]:
    try:
        config = agent.model.get_config() or {}
    except Exception:
        config = {}
    params = config.get("params") or {}
    max_out = config.get("max_tokens") or params.get("max_tokens") or params.get("max_completion_tokens")
    return config.get("model_id"), int(max_out or DEFAULT_OUTPUT_TOKENS)


def _usage_of(event: AfterModelCallEvent) -> Optional[Mapping[str, Any]]:
    if event.stop_response is None:
        return None
    return (event.stop_response.message.get("metadata") or {}).get("usage")
