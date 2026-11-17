"""agent-budget: hard USD budgets for agents that delegate.

* :class:`Budget` with :meth:`~Budget.reserve` / :meth:`Reservation.settle`
  and child scopes (:meth:`Budget.child`) that reserve a slice for each
  sub-agent, so parallel children can never be admitted for more than the
  parent has left.
* :class:`BudgetContext`: the budget handed down and the cost reported back,
  in the shape of the A2A budget extension draft, with carriers for A2A
  metadata, W3C baggage and MCP ``_meta`` (:mod:`agent_budget.carriers`).
* ``agent_budget.a2a_plugin``: a2a-python client and execution plugins
  (install the ``a2a`` extra).

Same semantics and wire format as the Rust crate llm-budget (0.8+).
"""

from ._money import AmountError
from .budget import (
    DEFAULT_MAX_DEPTH,
    Budget,
    BudgetError,
    BudgetExhausted,
    BudgetKilled,
    DepthExceeded,
    Outcome,
    Reservation,
    Scope,
    ScopeClosed,
    ScopeReceipt,
    current_budget,
)
from .carriers import (
    extract_mcp_meta,
    from_a2a_metadata,
    from_baggage,
    inject_mcp_meta,
    merge_baggage,
    to_a2a_metadata,
    to_baggage,
)
from .context import (
    EXTENSION_URI,
    BudgetContext,
    BudgetLimit,
    ContextError,
    Cost,
    Incurred,
    IncurredBooker,
    Money,
    ReservationInfo,
    TokenCounts,
    TokenLimit,
    incurred_of,
    outbound,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "EXTENSION_URI",
    "AmountError",
    "Budget",
    "BudgetContext",
    "BudgetError",
    "BudgetExhausted",
    "BudgetKilled",
    "BudgetLimit",
    "ContextError",
    "Cost",
    "DepthExceeded",
    "Incurred",
    "IncurredBooker",
    "Money",
    "Outcome",
    "Reservation",
    "ReservationInfo",
    "Scope",
    "ScopeClosed",
    "ScopeReceipt",
    "TokenCounts",
    "TokenLimit",
    "current_budget",
    "extract_mcp_meta",
    "from_a2a_metadata",
    "from_baggage",
    "incurred_of",
    "inject_mcp_meta",
    "merge_baggage",
    "outbound",
    "to_a2a_metadata",
    "to_baggage",
]
