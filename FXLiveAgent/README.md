# FXLiveAgent

FXLiveAgent is the live-account execution boundary for FXAI.

The current package provides the versioned promoted-workload contract,
fail-closed safety validation, human-release planning runtime, and tests. Real
broker/terminal adapters must stay behind this boundary and must not bypass
promotion evidence, risk limits, or kill-switch controls.
Live execution changes are governed by the root
[FXAI Governance](../GOVERNANCE.md) contract.

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
- Keep human-release evidence, promotion lineage, account scope, risk validation,
  stale-data status, and kill-switch state linked before any live connector can
  route real orders.

## Verify

```bash
swift test --package-path FXLiveAgent
```
