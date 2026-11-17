"""The a2a-python plugins over real transports (JSON-RPC and HTTP+JSON,
served in-process by Starlette): the caller's scope travels in the request,
the callee runs inside a budget of exactly that, and the callee's reported
cost is booked on the caller's scope once."""

import json

import pytest

a2a = pytest.importorskip("a2a")

from a2a import plugins  # noqa: E402
from a2a.client import ClientConfig, ClientFactory  # noqa: E402
from a2a.helpers.proto_helpers import new_task_from_user_message  # noqa: E402
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandlerV2  # noqa: E402
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # noqa: E402
from a2a.types.a2a_pb2 import (  # noqa: E402
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from a2a.utils import TransportProtocol  # noqa: E402
from a2a.utils._httpx_compat import httpx_lib  # noqa: E402
from google.protobuf import json_format  # noqa: E402
from starlette.applications import Starlette  # noqa: E402

from agent_budget import EXTENSION_URI, Budget, BudgetExhausted, current_budget  # noqa: E402
from agent_budget.a2a_plugin import BudgetClientPlugin, BudgetExecutionPlugin, read_context  # noqa: E402

CALL_COST = 0.03  # what each "model call" of the remote agent costs


class SpendingAgent(AgentExecutor):
    """Makes up to five $0.03 model calls, each reserved first on the budget
    the execution plugin opened; then completes the task."""

    def __init__(self):
        self.seen_budget = []
        self.headers = []

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        budget = current_budget()
        self.seen_budget.append(None if budget is None else budget.limit_usd)
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        calls = 0
        for _ in range(5):
            if budget is None:
                break
            try:
                hold = budget.reserve(0.04)  # worst case for one call
            except BudgetExhausted:
                break
            hold.settle(CALL_COST)
            calls += 1
        await updater.add_artifact(parts=[Part(text=f"{calls} calls")], name="out")
        await updater.complete()

    async def cancel(self, context, event_queue):
        pass


def card():
    return AgentCard(
        name="Budget Agent",
        description="spends from the budget it is given",
        version="1.0",
        capabilities=AgentCapabilities(
            streaming=True, extensions=[AgentExtension(uri=EXTENSION_URI, description="budget/v1")]
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supported_interfaces=[
            AgentInterface(protocol_binding=TransportProtocol.HTTP_JSON, url="http://testserver"),
            AgentInterface(protocol_binding=TransportProtocol.JSONRPC, url="http://testserver"),
        ],
    )


class HeaderSpy(plugins.ServerPlugin if hasattr(plugins, "ServerPlugin") else object):
    name = "spy"

    def __init__(self):
        self.extensions = []

    async def before_handler_run(self, request):
        headers = request.context.state.get("headers", {})
        self.extensions.append(headers.get("a2a-extensions") or headers.get("A2A-Extensions"))

    async def after_handler_run(self, response):
        pass


@pytest.fixture(autouse=True)
def restore_global():
    previous = plugins.get_global()
    yield
    plugins.set_global(previous)


@pytest.fixture(params=[TransportProtocol.JSONRPC, TransportProtocol.HTTP_JSON], ids=["JSON-RPC", "HTTP+JSON"])
def transport(request):
    return request.param


def make(transport, client_plugin, execution_plugin, interceptors=None):
    agent = SpendingAgent()
    spy = HeaderSpy()
    c = card()
    handler = DefaultRequestHandlerV2(
        agent_executor=agent,
        task_store=InMemoryTaskStore(),
        agent_card=c,
        plugins=plugins.sequential(spy, execution_plugin),
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=c, card_url="/card"),
            *create_jsonrpc_routes(request_handler=handler, rpc_url="/"),
            *create_rest_routes(request_handler=handler),
        ]
    )
    plugins.set_global(plugins.sequential(*([client_plugin] if client_plugin else [])))
    config = ClientConfig(
        httpx_client=httpx_lib.AsyncClient(transport=httpx_lib.ASGITransport(app=app), base_url="http://testserver"),
        supported_protocol_bindings=[transport],
    )
    return ClientFactory(config).create(c, interceptors), agent, spy


def request(text="research X"):
    return SendMessageRequest(message=Message(role=Role.ROLE_USER, message_id="m-1", parts=[Part(text=text)]))


async def test_budget_goes_down_and_cost_comes_back(transport):
    client, agent, spy = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin(max_usd=5.0))
    orchestrator = Budget(1.00, name="orchestrator")
    with orchestrator.child("remote-researcher", usd=0.10) as slice_:
        events = [e async for e in client.send_message(request())]
        receipt = slice_.close()
    await client.close()

    # The callee ran inside a budget of exactly the slice.
    assert agent.seen_budget == [pytest.approx(0.10)]
    # The extension was requested in the A2A-Extensions header.
    assert spy.extensions and EXTENSION_URI in (spy.extensions[0] or "")
    # $0.10 covers two $0.04 holds settled at $0.03, then $0.04 fits once more: 3 calls.
    assert events[-1].status_update.status.state == TaskState.TASK_STATE_COMPLETED
    final = read_context(events[-1].status_update.metadata)
    assert final.incurred.cost.nanos == 90_000_000
    assert final.incurred.cost.source == "local"
    # Booked once on the caller's slice, though every event repeated it.
    assert receipt.spent_usd == pytest.approx(0.09)
    assert receipt.released_usd == pytest.approx(0.01)
    assert orchestrator.spent_usd == pytest.approx(0.09)
    assert orchestrator.reserved_usd == 0


async def test_streamed_reports_are_cumulative(transport):
    client, agent, _ = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin())
    with Budget(1.00).child("remote", usd=0.50) as slice_:
        events = [e async for e in client.send_message(request())]
    await client.close()
    amounts = []
    for e in events:
        for holder in (e.task, e.status_update, e.artifact_update):
            ctx = read_context(holder.metadata) if holder.ByteSize() else None
            if ctx and ctx.incurred:
                amounts.append(ctx.incurred.cost.nanos)
    assert amounts == sorted(amounts)  # never a delta, never shrinking
    assert amounts[-1] == 150_000_000  # five calls of $0.03
    assert slice_.spent_usd == pytest.approx(0.15)


async def test_no_scope_no_metadata_no_budget(transport):
    client, agent, spy = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin())
    events = [e async for e in client.send_message(request())]
    await client.close()
    assert agent.seen_budget == [None]
    assert not spy.extensions[0]
    assert read_context(events[-1].status_update.metadata) is None


async def test_callee_caps_what_a_caller_may_grant(transport):
    client, agent, _ = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin(max_usd=0.05))
    with Budget(10.00).child("generous", usd=5.00) as slice_:
        [e async for e in client.send_message(request())]
    await client.close()
    assert agent.seen_budget == [pytest.approx(0.05)]
    assert slice_.spent_usd == pytest.approx(0.03)


async def test_default_budget_for_requests_without_one(transport):
    client, agent, _ = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin(default_usd=0.07))
    [e async for e in client.send_message(request())]
    await client.close()
    assert agent.seen_budget == [pytest.approx(0.07)]


def test_request_metadata_is_the_wire_format():
    """What the client plugin writes is exactly the shared JSON shape."""
    from agent_budget import outbound
    from agent_budget.a2a_plugin import _set_struct

    with Budget(1.0).child("x", usd=0.25) as s:
        req = request()
        _set_struct(req.metadata, outbound(s))
    d = json_format.MessageToDict(req)["metadata"][EXTENSION_URI]
    assert d["budget"] == {"cost": {"amount": "0.25", "currency": "USD"}}
    assert d["reservation"]["amount"] == "0.25"
    assert json.loads(json.dumps(d)) == d


async def test_interceptor_api_does_the_same(transport):
    from agent_budget.a2a_plugin import BudgetClientInterceptor

    client, agent, spy = make(transport, None, BudgetExecutionPlugin(), interceptors=[BudgetClientInterceptor()])
    orchestrator = Budget(1.00)
    with orchestrator.child("remote", usd=0.10) as slice_:
        [e async for e in client.send_message(request())]
    await client.close()
    assert agent.seen_budget == [pytest.approx(0.10)]
    assert EXTENSION_URI in (spy.extensions[0] or "")
    assert orchestrator.spent_usd == pytest.approx(0.09)


async def test_concurrent_delegations_from_one_orchestrator(transport):
    """Three remote calls in parallel, each in its own slice: the callee of
    each gets its own slice, and the orchestrator's books add up."""
    import asyncio

    client, agent, _ = make(transport, BudgetClientPlugin(), BudgetExecutionPlugin())
    orchestrator = Budget(0.30, name="orchestrator")

    async def delegate(i):
        with orchestrator.child(f"remote-{i}", fraction=1 / (3 - i)) as s:
            [e async for e in client.send_message(request(f"task {i}"))]
            return s.close()

    receipts = await asyncio.gather(*(delegate(i) for i in range(3)))
    await client.close()
    assert sorted(agent.seen_budget) == [pytest.approx(0.10)] * 3
    assert sum(r.spent_usd for r in receipts) == pytest.approx(orchestrator.spent_usd)
    assert orchestrator.spent_usd == pytest.approx(0.27)
    assert orchestrator.reserved_usd == 0
