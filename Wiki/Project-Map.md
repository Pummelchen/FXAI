# Project Map

## Active Projects

| Folder | Description |
| --- | --- |
| `FXImporter/` | Provider connectors and import adapters for MT5, Yahoo Finance, and future sources such as IBKR/TWS and TradingView. |
| `FXDatabase/` | The only ClickHouse authority. Owns database config, validation, ingestion, storage, deletion, persistent SineTest data, and public data APIs. |
| `FXDataEngine/` | Post-processing and feature layer. Converts M1 OHLCV into labels, context, audit artifacts, and plugin request payloads. |
| `FXPlugins/` | Flat plugin zoo. Each plugin owns its own CPU and optional accelerator implementations. |
| `FXBacktest/` | Offline Swift/Metal backtest framework. Uses FXDatabase APIs and calls plugins through FXDataEngine/plugin contracts. |
| `FXGUI/` | macOS SwiftUI GUI for operator-facing dashboards, reports, and workflow access. |

## Future Agent Projects

| Folder | Planned role |
| --- | --- |
| `FXBacktestAgent/` | Remote Mac backtest worker. It pulls batches over TCP and reports results to FXBacktest. |
| `FXDemoAgent/` | Demo-account execution agent for applying approved backtest parameters to demo terminals and broker APIs. |
| `FXLiveAgent/` | Live-account execution agent for approved parameters with stricter safety and approval gates. |

## Data Ownership

FXImporter can talk to outside data sources. FXDatabase can talk to ClickHouse. Everything else must use APIs.

## Runtime Ownership

FXBacktest owns backtest scheduling. FXPlugins owns plugin behavior. FXDataEngine owns feature preparation. Agents will own distributed or account-specific execution, not database access.
