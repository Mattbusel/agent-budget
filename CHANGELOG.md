# Changelog

## [0.1.1] - 2026-11-17

The first version meant for PyPI; 0.1.0 was never uploaded.

### Fixed
- Untrusted amounts could raise `OverflowError` instead of `ContextError`: a JSON number such as `1e300`, or an integer too large for a float, escaped `BudgetContext.from_dict` and the a2a plugin's `read_context`, so a peer could make `before_execution` or `after_client_request` raise. Amounts are now range-checked before they are scaled, `from_dict` turns anything that slips past a field check into `ContextError`, and `read_context` logs and ignores every malformed payload.
- JSON integer amounts are read exactly (`18446744073` stays `"18446744073"`; through float it came out as `18446744072.999999488`), as llm-budget reads them.
- `IncurredBooker.book` raised `AmountError` on a large but valid report (the largest amount, or `cost + aggregatedChildCost` above it), after the remote work was done. It now books in nano-units without a float round trip and saturates, like llm-budget.
- Entering one `Budget` with `with` from two asyncio tasks or threads at once raised `ValueError` (the token stack lived on the shared object). Entries are now kept per context.
- `BudgetExecutionPlugin` gave a caller that granted $0 (an exhausted caller) the `default_usd` budget, and did the same for a budget in another currency. A $0 grant is now a $0 budget that refuses every reservation; a non-USD budget is logged and treated as $0.
- A `Scope` nobody closed kept its slice reserved in the parent forever. It is now closed when garbage-collected, with a `ResourceWarning`, matching llm-budget's drop of a `ScopeGuard`.
- A token count given as both `cacheRead` and `cache_read` (or `cacheWrite` and `cache_write`) is rejected, as llm-budget rejects it.

### Added
- `BudgetClientPlugin(per_call_usd=..., per_call_fraction=...)` opens a child scope for each call, sends that slice and closes it when the call ends, so parallel calls from one root budget are no longer each told the same remainder.
- `cap_reports=True` on `BudgetClientPlugin` and `BudgetClientInterceptor`, and `IncurredBooker(cap_nanos=...)` / `IncurredBooker.capped_at(scope)` with `excess_nanos`: book at most what the callee was given and flag the rest.
- `BudgetClientInterceptor` takes `deadline` and `token_limit`, like the plugin.
- `tests/test_hardening.py` and new cases in `tests/test_a2a_plugin.py` for each fix; the shared fixtures gain oversized JSON-number amounts, counts given under both names and an exact large whole-number amount.

## [0.1.0] - 2026-11-17

First release.

- `Budget` with `reserve` / `Reservation.settle` / `release` (exact nano-dollar accounting, thread-safe, asyncio-safe) and child scopes: `Budget.child(name, usd=... | fraction=...)` reserves a sub-agent's slice from its parent before it starts; everything the child books is booked on every ancestor at once; closing it refunds the unused part. `ScopeClosed`, `BudgetKilled` (kill switches stop a whole subtree), `DepthExceeded`, `BudgetExhausted`. `ScopeReceipt` with own and child cost. `current_budget()` follows `with` blocks per thread and asyncio task.
- `BudgetContext` in the shape of the A2A budget extension draft (a2aproject/A2A#2121): `budget{cost, tokens, deadline}`, `incurred{cost, aggregatedChildCost, tokens, delegations}`, plus `reservation{id, amount, currency, expires}`. Money as decimal strings, absent never written as zero. `outbound(scope)`, `incurred_of(budget)`, `IncurredBooker` (books cumulative reports once).
- Carriers: A2A metadata (`to_a2a_metadata`, `from_a2a_metadata`), W3C baggage (`to_baggage`, `from_baggage`, `merge_baggage`) and MCP `_meta.baggage` (`inject_mcp_meta`, `extract_mcp_meta`).
- `agent_budget.a2a_plugin` (extra `a2a`): `BudgetClientPlugin`, `BudgetExecutionPlugin` and `BudgetClientInterceptor` for a2a-sdk 1.3+.
- Golden fixtures shared byte for byte with the Rust crate llm-budget 0.8.
