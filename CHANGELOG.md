# Changelog

## [0.1.0] - 2026-11-17

First release.

- `Budget` with `reserve` / `Reservation.settle` / `release` (exact nano-dollar accounting, thread-safe, asyncio-safe) and child scopes: `Budget.child(name, usd=... | fraction=...)` reserves a sub-agent's slice from its parent before it starts; everything the child books is booked on every ancestor at once; closing it refunds the unused part. `ScopeClosed`, `BudgetKilled` (kill switches stop a whole subtree), `DepthExceeded`, `BudgetExhausted`. `ScopeReceipt` with own and child cost. `current_budget()` follows `with` blocks per thread and asyncio task.
- `BudgetContext` in the shape of the A2A budget extension draft (a2aproject/A2A#2121): `budget{cost, tokens, deadline}`, `incurred{cost, aggregatedChildCost, tokens, delegations}`, plus `reservation{id, amount, currency, expires}`. Money as decimal strings, absent never written as zero. `outbound(scope)`, `incurred_of(budget)`, `IncurredBooker` (books cumulative reports once).
- Carriers: A2A metadata (`to_a2a_metadata`, `from_a2a_metadata`), W3C baggage (`to_baggage`, `from_baggage`, `merge_baggage`) and MCP `_meta.baggage` (`inject_mcp_meta`, `extract_mcp_meta`).
- `agent_budget.a2a_plugin` (extra `a2a`): `BudgetClientPlugin`, `BudgetExecutionPlugin` and `BudgetClientInterceptor` for a2a-sdk 1.3+.
- Golden fixtures shared byte for byte with the Rust crate llm-budget 0.8.
