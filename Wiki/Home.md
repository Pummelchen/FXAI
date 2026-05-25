# FXAI Wiki

FXAI is a Swift, Metal, PyTorch, and TensorFlow stack organized as focused subprojects with one database authority and a flat plugin zoo.

## The Short Version

FXImporter gets data from external sources. FXDatabase validates and stores it in ClickHouse. FXDataEngine prepares plugin-ready feature and context payloads. FXPlugins owns the model zoo. FXBacktest runs offline backtests through those APIs and now sees the root plugin zoo through a FXDataEngine-backed bridge. FXBacktestAgent has a versioned fail-closed worker protocol, and demo/live execution safety contracts are shared in FXExecutionContracts.

```text
FXImporter -> FXDatabase -> FXDataEngine -> FXPlugins -> FXBacktest
                  |
                  +-> FXGUI
                  +-> FXBacktestAgent / future FXDemoAgent / FXLiveAgent
```

Only FXDatabase may touch ClickHouse directly.

## Start Here

- [Architecture](Architecture.md)
- [User Roles](User-Roles.md)
- [Installation](Installation.md)
- [Project Map](Project-Map.md)
- [Top 5 Implementation Roadmap](Top-5-Implementation-Roadmap.md)

## Who This Is For

- Backtest researchers who want native Swift/Metal offline testing instead of MT5 Strategy Tester.
- Plugin developers who need CPU reference implementations plus optional Metal, PyTorch, TensorFlow, or NLP accelerators.
- Data operators who need one controlled path into ClickHouse.
- Fleet operators who want remote Macs to act as future TCP backtest agents.
- Demo and live traders who need a clean separation between research, demo deployment, and live execution.
- Operators who want FXGUI to mirror terminal workflows through a command catalog and versioned API paths.

## Current Rule

The data contract is `M1 OHLCV`. M1 OHLC is required. Volume is used whenever a provider supplies positive volume. Spread is retired from the offline contract.
