# FXBacktestAgent

Future source root for distributed offline backtest workers.

FXBacktestAgent will let other Macs pull backtest batches from FXBacktest over TCP,
execute the assigned work locally, and report results back to the FXBacktest server.
It is the Swift/Metal replacement path for MT5-style remote backtest agents.

## Boundary

- FXBacktestAgent is a worker, not a database owner.
- It must never connect to ClickHouse directly.
- Market data and result persistence must flow through FXBacktest and FXDatabase APIs.
- Plugin execution must use the same FXDataEngine and FXPlugins contracts as local
  FXBacktest runs.

## Implementation Notes

- Keep the transport protocol explicit and versioned.
- Treat batches as immutable work units with deterministic inputs.
- Report worker capabilities, including CPU, Metal device, PyTorch MPS, TensorFlow,
  and NLP runtime availability.
- Include SineTest certification before accepting real backtest work.
