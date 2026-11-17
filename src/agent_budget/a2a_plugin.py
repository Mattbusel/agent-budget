"""Budget propagation for a2a-python (``pip install "agent-budget[a2a]"``).

Two plugins for the a2a-sdk plugin system (a2a-sdk 1.3 and later):

* :class:`BudgetClientPlugin` (caller side). For every ``send_message`` /
  ``send_message_streaming`` call made while a budget scope is current
  (``with parent.child("remote", usd=0.30):``), it writes the scope's
  :func:`~agent_budget.context.outbound` context into the request metadata
  under the budget extension URI, asks for the extension in the
  ``A2A-Extensions`` service parameter, and books the callee's reported
  ``incurred`` cost on the scope (once, however many stream events repeat
  it).
* :class:`BudgetExecutionPlugin` (callee side). Before the agent runs it
  reads the budget from the request, opens a :class:`~agent_budget.Budget`
  capped at what the caller allowed, and makes it
  :func:`~agent_budget.current_budget` inside ``AgentExecutor.execute``.
  Every task, status update and message the agent emits carries the
  cumulative ``incurred`` report in its metadata.

For a2a-sdk versions with only the older interceptor API,
:class:`BudgetClientInterceptor` does the caller-side request half.

Register them like any other plugin::

    from a2a import plugins
    from agent_budget.a2a_plugin import BudgetClientPlugin, BudgetExecutionPlugin

    plugins.set_global(plugins.sequential(BudgetClientPlugin()))              # caller
    handler = DefaultRequestHandlerV2(..., plugins=plugins.sequential(
        BudgetExecutionPlugin(max_usd=5.00)))                                   # callee
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Iterable, Optional

from google.protobuf import json_format, struct_pb2

from a2a.client.interceptors import AfterArgs, BeforeArgs, ClientCallInterceptor
from a2a.client.service_parameters import with_a2a_extensions
from a2a.plugins import ClientPlugin, ExecutionPlugin

from ._money import nanos_to_usd
from .budget import Budget, _current, current_budget
from .context import EXTENSION_URI, BudgetContext, ContextError, IncurredBooker, incurred_of, outbound

logger = logging.getLogger(__name__)

SEND_METHODS = frozenset({"send_message", "send_message_streaming"})
_STATE_KEY = "agent_budget.call"


def _struct_get(meta: Any) -> Optional[dict]:
    """The extension payload in a protobuf ``Struct`` (or a dict), as a dict."""
    if meta is None:
        return None
    if isinstance(meta, struct_pb2.Struct):
        if EXTENSION_URI not in meta.fields:
            return None
        payload = json_format.MessageToDict(meta.fields[EXTENSION_URI])
        return payload if isinstance(payload, dict) else {"__invalid__": payload}
    if isinstance(meta, dict):
        return meta.get(EXTENSION_URI)
    return None


def _set_struct(meta: struct_pb2.Struct, ctx: BudgetContext) -> None:
    meta.update({EXTENSION_URI: ctx.to_dict()})


def _metadata_holders(payload: Any) -> Iterable[Any]:
    """Every ``metadata`` Struct in a response or stream event that may hold
    an ``incurred`` report: the task, the message, a status update."""
    if payload is None:
        return []
    out = []
    names = {f.name for f in payload.DESCRIPTOR.fields} if hasattr(payload, "DESCRIPTOR") else set()
    for field in ("task", "message", "status_update", "artifact_update"):
        if field in names and payload.HasField(field):
            out.append(getattr(payload, field).metadata)
    if "metadata" in names and not out:
        out.append(payload.metadata)  # a bare Task or Message
    return out


def read_context(meta: Any) -> Optional[BudgetContext]:
    """The budget context in a metadata ``Struct`` or dict, or ``None``.
    Malformed payloads are logged and read as ``None``."""
    payload = _struct_get(meta)
    if payload is None:
        return None
    try:
        return BudgetContext.from_dict(payload)
    except ContextError as e:
        logger.warning("ignoring malformed A2A budget metadata: %s", e)
        return None


class _CallState:
    __slots__ = ("scope", "booker")

    def __init__(self, scope: Budget):
        self.scope = scope
        self.booker = IncurredBooker()


def _book_reports(state: _CallState, payload: Any) -> None:
    """Book any ``incurred`` report in a response or stream event (only its
    growth since the last one: reports are cumulative)."""
    for meta in _metadata_holders(payload):
        ctx = read_context(meta)
        if ctx is not None and ctx.incurred is not None:
            out = state.booker.book(state.scope, ctx.incurred)
            if out is not None and out.over_cap:
                logger.warning(
                    "remote agent reported $%.6f in total, over the cap of scope %r",
                    nanos_to_usd(state.booker.booked_nanos),
                    state.scope.name,
                )


class BudgetClientPlugin(ClientPlugin):
    """Caller side: send the current scope's budget, book what comes back.

    ``budget`` fixes the budget to send from; by default the plugin uses
    :func:`~agent_budget.current_budget` at call time and does nothing when
    there is none. ``deadline`` and ``token_limit`` are added to every
    outbound context.
    """

    name = "agent-budget"

    def __init__(self, budget: Optional[Budget] = None, *, deadline: Optional[str] = None,
                 token_limit: Optional[int] = None):
        self._budget = budget
        self._deadline = deadline
        self._token_limit = token_limit

    async def before_client_request(self, request) -> None:
        if request.method not in SEND_METHODS:
            return
        scope = self._budget or current_budget()
        if scope is None:
            return
        ctx = outbound(scope, deadline=self._deadline, token_limit=self._token_limit)
        _set_struct(request.payload.metadata, ctx)
        context = request.context
        params = dict(context.service_parameters or {})
        with_a2a_extensions([EXTENSION_URI])(params)
        context.service_parameters = params
        context.state[_STATE_KEY] = _CallState(scope)

    async def after_client_request(self, response) -> None:
        state = response.context.state.get(_STATE_KEY) if response.context is not None else None
        if state is None:
            return
        _book_reports(state, response.payload)


class BudgetClientInterceptor(ClientCallInterceptor):
    """Caller side for the interceptor API
    (``ClientFactory(config).create(card, interceptors=[...])``): writes the
    current scope's budget into ``send_message`` requests and books the
    reported cost of every response and stream event on it. Same behavior
    as :class:`BudgetClientPlugin`, for code that wires interceptors rather
    than plugins."""

    def __init__(self, budget: Optional[Budget] = None):
        self._plugin = BudgetClientPlugin(budget)

    async def before(self, args: BeforeArgs) -> None:
        if args.method not in SEND_METHODS:
            return
        scope = self._plugin._budget or current_budget()
        if scope is None:
            return
        _set_struct(args.input.metadata, outbound(scope))
        if args.context is None:
            from a2a.client.client import ClientCallContext

            args.context = ClientCallContext()
        params = dict(args.context.service_parameters or {})
        with_a2a_extensions([EXTENSION_URI])(params)
        args.context.service_parameters = params
        args.context.state[_STATE_KEY] = _CallState(scope)

    async def after(self, args: AfterArgs) -> None:
        state = args.context.state.get(_STATE_KEY) if args.context is not None else None
        if state is None:
            return
        _book_reports(state, args.result)


class _Execution:
    __slots__ = ("budget", "token")

    def __init__(self, budget: Budget, token: contextvars.Token):
        self.budget = budget
        self.token = token


class BudgetExecutionPlugin(ExecutionPlugin):
    """Callee side: run each execution inside a budget of what the caller
    allowed, and report the cost back on every event.

    ``default_usd`` caps executions whose request carries no budget (without
    it they run with no budget). ``max_usd`` caps what any caller can grant.
    ``source`` is the ``incurred.cost.source`` to report: ``"local"`` when
    the agent books costs it computed from prices, ``"provider"`` when it
    books costs its provider reported.
    """

    name = "agent-budget"

    def __init__(self, *, default_usd: Optional[float] = None, max_usd: Optional[float] = None,
                 source: Optional[str] = "local"):
        self._default = default_usd
        self._max = max_usd
        self._source = source
        self._runs: dict[int, _Execution] = {}

    def _granted(self, request_context) -> Optional[float]:
        ctx = read_context(request_context.metadata)  # SendMessageRequest.metadata
        if ctx is None and request_context.message is not None:
            ctx = read_context(request_context.message.metadata)
        nanos = ctx.budget_nanos() if ctx is not None else None
        usd = nanos_to_usd(nanos) if nanos else self._default
        if usd is not None and self._max is not None:
            usd = min(usd, self._max)
        return usd

    async def before_execution(self, request_context) -> None:
        usd = self._granted(request_context)
        if usd is None or usd <= 0:
            return
        name = request_context.task_id or "a2a-execution"
        budget = Budget(usd, name=name)
        token = _current.set(budget)
        self._runs[id(request_context)] = _Execution(budget, token)

    async def on_execution_event(self, request_context, event) -> None:
        run = self._runs.get(id(request_context))
        if run is None or not hasattr(event, "metadata"):
            return
        report = BudgetContext(incurred=incurred_of(run.budget, source=self._source))
        _set_struct(event.metadata, report)

    async def after_execution(self, request_context, error) -> None:
        run = self._runs.pop(id(request_context), None)
        if run is None:
            return
        try:
            _current.reset(run.token)
        except ValueError:
            # Reset from another context: the execution task already ended.
            pass

    def budget_for(self, request_context) -> Optional[Budget]:
        """The budget of a running execution (also available as
        :func:`~agent_budget.current_budget` inside ``execute``)."""
        run = self._runs.get(id(request_context))
        return run.budget if run else None


__all__ = [
    "BudgetClientInterceptor",
    "BudgetClientPlugin",
    "BudgetExecutionPlugin",
    "read_context",
]
