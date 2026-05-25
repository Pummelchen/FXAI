# FXDemoAgent

FXDemoAgent is the demo-account execution boundary for FXAI.

The current package provides the versioned workload contract, fail-closed safety
validation, dry-run planning runtime, and tests. Broker/terminal adapters must
plug into this boundary instead of bypassing the shared execution contracts.

## Boundary

- FXDemoAgent is an execution adapter, not a research or database service.
- It must never connect to ClickHouse directly.
- Any historical data, model evidence, or result lookup must go through FXDatabase APIs.
- Demo execution must be separated from live execution and must not share credentials
  or mutable runtime state with FXLiveAgent.

## Implementation Notes

- Keep every broker or terminal connector behind a small execution API.
- Require explicit strategy, symbol, risk, and account-scoping inputs from FXBacktest.
- Prefer dry-run and paper-trade modes in new connectors before allowing order routing.
- Store audit events through FXDatabase or a future approved agent telemetry API.

## Verify

```bash
swift test --package-path FXDemoAgent
```
