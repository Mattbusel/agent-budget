"""An orchestrator delegates to a remote A2A agent with a budget, offline.

    pip install "llm-agent-budget[a2a]" "a2a-sdk[http-server]"
    python examples/a2a_delegation.py

The remote agent is served in-process over JSON-RPC. The orchestrator
reserves a $0.10 slice; the client plugin sends it in the request metadata;
the execution plugin opens a Budget of exactly $0.10 around the agent; the
agent's model calls are reserved and settled on it; every task update
carries the cumulative cost back; the client plugin books it on the slice
once. The unused part of the slice returns to the orchestrator.
"""

import asyncio
import json

from a2a import plugins
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    SendMessageRequest,
)
from a2a.utils import TransportProtocol
from a2a.utils._httpx_compat import httpx_lib
from starlette.applications import Starlette

from agent_budget import EXTENSION_URI, Budget, BudgetExhausted, current_budget
from agent_budget.a2a_plugin import BudgetClientPlugin, BudgetExecutionPlugin, read_context


class Researcher(AgentExecutor):
    """Makes up to five model calls; each is reserved at $0.04 and costs $0.03."""

    async def execute(self, context, event_queue):
        budget = current_budget()  # opened by BudgetExecutionPlugin
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        calls = 0
        for _ in range(5):
            try:
                hold = budget.reserve(0.04)
            except BudgetExhausted:
                break
            hold.settle(0.03)  # the provider's bill for the call
            calls += 1
        print(f"  [researcher] budget ${budget.limit_usd:.2f}, made {calls} calls, spent ${budget.spent_usd:.2f}")
        await updater.add_artifact(parts=[Part(text=f"{calls} findings")], name="findings")
        await updater.complete()

    async def cancel(self, context, event_queue):
        pass


CARD = AgentCard(
    name="Researcher",
    description="researches within the budget it is given",
    version="1.0",
    capabilities=AgentCapabilities(streaming=True, extensions=[AgentExtension(uri=EXTENSION_URI)]),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    supported_interfaces=[
        AgentInterface(protocol_binding=TransportProtocol.JSONRPC, url="http://researcher", protocol_version="1.0")
    ],
    skills=[AgentSkill(id="research", name="research", description="researches a topic", tags=["research"])],
)


async def main():
    handler = DefaultRequestHandlerV2(
        agent_executor=Researcher(),
        task_store=InMemoryTaskStore(),
        agent_card=CARD,
        plugins=plugins.sequential(BudgetExecutionPlugin(max_usd=1.00)),
    )
    app = Starlette(routes=create_jsonrpc_routes(request_handler=handler, rpc_url="/"))
    plugins.set_global(plugins.sequential(BudgetClientPlugin()))
    http = httpx_lib.AsyncClient(transport=httpx_lib.ASGITransport(app=app), base_url="http://researcher")
    client = ClientFactory(ClientConfig(httpx_client=http, supported_protocol_bindings=[TransportProtocol.JSONRPC])).create(CARD)

    orchestrator = Budget(1.00, name="orchestrator")
    request = SendMessageRequest(message=Message(role=Role.ROLE_USER, message_id="m-1", parts=[Part(text="research X")]))
    with orchestrator.child("researcher", usd=0.10) as slice_:
        last = None
        async for event in client.send_message(request):
            last = event
        receipt = slice_.close()
    await client.close()

    report = read_context(last.status_update.metadata)
    print("final task update carries:", json.dumps(report.to_dict()))
    print(f"orchestrator: slice ${receipt.held_usd:.2f}, remote spent ${receipt.spent_usd:.2f}, "
          f"${receipt.released_usd:.2f} back; orchestrator spent ${orchestrator.spent_usd:.2f} of ${orchestrator.limit_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
