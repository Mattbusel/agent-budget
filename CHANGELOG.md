# Changelog

## [0.2.0] - 2026-11-19

**Fail closed by default.** A budget that guesses a price is not a cap. Cost trackers keep shipping the same guesses: an unknown model priced at $0 or a placeholder (in PraisonAI every newer model was billed at a $1/$3 default, under a fifth of a real gpt-5.5 call, so `max_budget=1.0` let about $3.42 through; PraisonAI #6433), cached tokens billed as plain input, and a price table nobody updates. 0.2 refuses each of these instead of guessing, and every way out is explicit and warns on every call it lets through. Same semantics as llm-budget 0.10 (Rust).

Breaking: `strands.Price.cost` raises on cache tokens without a cache price, `Budget(None)` and non-finite caps raise, and the package now depends on llm-prices.

### Added
- `agent_budget.pricing`: `Pricer` prices calls from [llm-prices](https://pypi.org/project/llm-prices/) and refuses every guess: `UnknownModel` (no source lists the model), `CachePriceMissing` (cache reads or writes on a price with no cache rate; llm-prices itself bills them as plain input), `StalePrice` (a price last checked more than `max_price_age_days`, default 90, ago; the message says `pip install -U llm-prices`), `Unpriced` (listed without a price the call needs). All subclass `PriceError(BudgetError)`. `usage_from` normalizes OpenAI (Chat Completions and Responses), Anthropic (5m and 1h cache writes) and Gemini usage blocks. `Pricer(prices={...})` for your own rates, `today=` for replays, `worst_case(model, input, output)`.
- `Budget.reserve_for(model, usage)`, `Reservation.settle_usage(model, usage)` and `Budget.record_usage(model, usage)`: model calls priced at the budget's `pricer`. A settlement that cannot be priced books the hold (trying the reserved model first), never $0.
- Loud opt-outs: `Budget(..., allow_unpriced=True)` (let unpriced calls through at $0) and `Budget(..., unsafe_soft_limit=True)` (admit reservations over the cap; kills and closed scopes still refuse, child scopes still need room), `Pricer(allow_stale=True)`. Each issues an `UnsafeBudgetWarning` on every call it lets through (the package registers an "always" filter for that category, so the default warnings setup does not deduplicate them) and logs it to the `agent_budget` logger. `Budget.unsafe_events` counts them per tree; child scopes share the settings.
- **`agent-budget audit`** (also `python -m agent_budget audit`): replays a JSON-lines log of calls under a soft limit and under a fail-closed hard cap, and reports the true cost of what the soft limit let through, the call at which real spend crossed the cap, the money past it, booked-versus-true drift per model, and every model with no usable price. Exit status 2 when money got past the cap, `--json`, `--max-price-age`, `--today`. Library API `agent_budget.audit.replay`. Same report as `llm-budget audit`.
- Tests: `tests/test_fail_closed.py`, `tests/test_audit.py`.

### Changed (breaking)
- `Budget(limit_usd)` requires a hard cap: `None`, booleans, NaN and infinity raise `AmountError` ("a budget needs a hard cap").
- `strands.Price` no longer bills cache tokens at the input price when its cache price is `None`: `cost` raises `CachePriceMissing`. `SpendCeiling(price=...)` is optional: by default each model call is priced from llm-prices by the model's id (`Price.from_llm_prices`), fail closed, so an unknown, stale or unpriced model cancels the call before it is sent.
- `SpendCeiling` books the full hold for a model call that returned no usage or could not be priced (it used to release the hold, booking $0 for a call that happened). A call that raised is still released.
- New dependency: `llm-prices>=1.8,<2` (no dependencies of its own).

## [0.1.3] - 2026-11-17

The first release on PyPI, under the distribution name **llm-agent-budget**
(`pip install llm-agent-budget`). PyPI refused the name `agent-budget` as too
similar to an existing project. The import name is unchanged:
`import agent_budget`. 0.1.2 was tagged on GitHub but never reached PyPI.

### Added
- `agent_budget.strands` (extra `strands`, strands-agents 1.43 or later): `SpendCeiling`, a Strands plugin that caps what each invocation spends in USD. It reserves each model call's worst case (projected input plus the output-token cap) in `BeforeModelCallEvent`, cancels the call if it does not fit, and settles the reported usage in `AfterModelCallEvent`. An agent with a `SpendCeiling` that runs as a tool of another one gets its `usd` as a child slice of the caller's budget, so sub-agents started in one turn cannot be admitted for more than the caller has left. `budget=` draws every invocation from one shared `Budget`. `Price` prices a Strands `Usage` once per counter whether the provider folds cache tokens into `inputTokens` or reports them on top.
- `docs/strands.md` and `tests/test_strands.py` (real Strands agents with a scripted model, offline). Four researchers started in one turn under a $0.05 orchestrator: $0.078 without ceilings, $0.0468 with them.

## [0.1.2] - 2026-11-17

For projects deciding whether to depend on agent-budget. No behavior changes.

### Added
- `docs/integrating.md`: dependencies (none for the core), supported Python versions, typing, side effects, concurrency, how to make agent-budget an optional extra of your own package, the five-line reserve/settle integration, which entry point fits which stack (any client, sub-agents, pydantic-ai, A2A, baggage, MCP, a Rust service), and the stability policy.
- `examples/integrate.py`: that integration, runnable; CI runs it on every Python version.
- CI job `typecheck`: `mypy --strict` over the package (with the `a2a` extra) and the integration example.

### Fixed
- The package ships `py.typed` but did not pass `mypy --strict` (36 errors): unannotated parameters on the a2a plugins' hooks, `Budget` internals and `outbound` / `incurred_of` / `IncurredBooker`, and bare `dict` / `tuple` / `Token` types. Every public function is now fully annotated, so a strict type checker in a project that uses agent-budget sees real types instead of `Any`.

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
- `docs/pydantic-ai.md` and `examples/pydantic_ai_delegation.py`: a USD cap across parallel pydantic-ai delegates (a `WrapperModel` that reserves and settles every request, and a delegation tool that runs the sub-agent in a child scope), run offline in CI.
- `tests/test_hardening.py` and new cases in `tests/test_a2a_plugin.py` for each fix; the shared fixtures gain oversized JSON-number amounts, counts given under both names and an exact large whole-number amount.

## [0.1.0] - 2026-11-17

First release.

- `Budget` with `reserve` / `Reservation.settle` / `release` (exact nano-dollar accounting, thread-safe, asyncio-safe) and child scopes: `Budget.child(name, usd=... | fraction=...)` reserves a sub-agent's slice from its parent before it starts; everything the child books is booked on every ancestor at once; closing it refunds the unused part. `ScopeClosed`, `BudgetKilled` (kill switches stop a whole subtree), `DepthExceeded`, `BudgetExhausted`. `ScopeReceipt` with own and child cost. `current_budget()` follows `with` blocks per thread and asyncio task.
- `BudgetContext` in the shape of the A2A budget extension draft (a2aproject/A2A#2121): `budget{cost, tokens, deadline}`, `incurred{cost, aggregatedChildCost, tokens, delegations}`, plus `reservation{id, amount, currency, expires}`. Money as decimal strings, absent never written as zero. `outbound(scope)`, `incurred_of(budget)`, `IncurredBooker` (books cumulative reports once).
- Carriers: A2A metadata (`to_a2a_metadata`, `from_a2a_metadata`), W3C baggage (`to_baggage`, `from_baggage`, `merge_baggage`) and MCP `_meta.baggage` (`inject_mcp_meta`, `extract_mcp_meta`).
- `agent_budget.a2a_plugin` (extra `a2a`): `BudgetClientPlugin`, `BudgetExecutionPlugin` and `BudgetClientInterceptor` for a2a-sdk 1.3+.
- Golden fixtures shared byte for byte with the Rust crate llm-budget 0.8.
