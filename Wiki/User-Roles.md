# User Roles

FXAI serves several user types. The repo structure is designed so each user can work in the project that matches their responsibility without bypassing shared contracts.

## Backtest Researcher

You use FXBacktest to run offline backtests against FXDatabase data. The main benefit is speed, repeatability, and Apple Silicon acceleration without relying on MT5 Strategy Tester.

Use:

- `FXBacktest/`
- `FXPlugins/`
- `FXDataEngine/`
- `FXDatabase/`

## Plugin Developer

You work inside one plugin folder at a time. CPU behavior is the reference path. Add Metal, PyTorch, TensorFlow, or NLP only where the plugin benefits from it.

Use:

- `FXPlugins/plugin_name/CPU`
- `FXPlugins/plugin_name/Metal`
- `FXPlugins/plugin_name/PyTorch`
- `FXPlugins/plugin_name/TensorFlow`
- `FXPlugins/plugin_name/NLP`
- `FXPlugins/API`

## Data Operator

You manage external data ingestion and the ClickHouse-backed database. FXDatabase is your authority layer. You should not add direct ClickHouse access to FXBacktest, FXDataEngine, FXPlugins, or future agents.

Use:

- `FXImporter/`
- `FXDatabase/`

## Demo Trader

The future FXDemoAgent will take selected backtest parameters from FXBacktest and apply them to demo accounts across supported terminals and broker APIs. This keeps demo deployment separate from research and live trading.

Use later:

- `FXDemoAgent/`

## Live Trader

The future FXLiveAgent will apply approved parameters to live accounts. It should share broker abstractions with FXDemoAgent, but with stricter safety, approval, and monitoring gates.

Use later:

- `FXLiveAgent/`

## Fleet Operator

The future FXBacktestAgent lets other Macs pull backtest batches over TCP, run the assigned work, and report results back to FXBacktest acting as a backtest server.

Use later:

- `FXBacktestAgent/`

## System Architect

You care about boundaries. The important rule is that each project has one job and cannot bypass the project that owns the data or runtime contract.

Use:

- Root README.
- [Architecture](Architecture.md).
- [Project Map](Project-Map.md).
