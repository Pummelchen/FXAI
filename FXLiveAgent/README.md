# FXLiveAgent

Future source root for live-account execution.

FXLiveAgent will apply approved FXAI parameters to live trading accounts across the
same broker and terminal families as FXDemoAgent, but with stronger approval,
monitoring, risk, and rollback gates.

## Boundary

- FXLiveAgent is an execution adapter, not a database owner.
- It must never connect to ClickHouse directly.
- All historical data, plugin evidence, and parameter provenance must come through
  FXDatabase and FXBacktest APIs.
- Live execution must stay isolated from demo execution, including credentials,
  account scopes, open-position state, and emergency controls.

## Implementation Notes

- Require explicit promotion evidence from FXBacktest before enabling a live run.
- Keep broker connectors behind a versioned execution API shared with FXDemoAgent
  where safe, with live-only safety gates layered above it.
- Include kill-switch, max exposure, max loss, and stale-data protection before any
  connector can route real orders.
- Persist immutable audit events for every decision, order request, broker response,
  and safety intervention.
