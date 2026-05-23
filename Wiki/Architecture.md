# Architecture

FXAI is organized around explicit ownership boundaries.

## Data Flow

```text
External providers
  MT5, IBKR/TWS, Yahoo Finance, TradingView, broker files, future feeds
      |
      v
FXImporter
  Provider connectors and external-source normalization.
      |
      v
FXDatabase
  The only direct ClickHouse owner.
  Validates, stores, deletes, verifies, and serves data through APIs.
      |
      +--> FXDataEngine
      |      Builds features, contexts, labels, audits, and plugin payloads.
      |
      +--> FXBacktest
      |      Requests data and stores results through FXDatabase APIs.
      |      Calls plugins through the FXAI plugin contracts.
      |
      +--> FXDataEngineGUI
             Reads artifacts and gives operators a workflow surface.
```

## Project Responsibilities

| Project | Responsibility |
| --- | --- |
| `FXImporter` | External data source connectors. It should know how to talk to providers, not how to run backtests. |
| `FXDatabase` | ClickHouse gatekeeper, ingestion, validation, history access, deletion, SineTest, and FXBacktest data APIs. |
| `FXDataEngine` | Post-processing, feature engineering, label preparation, audit tools, and plugin payload contracts. |
| `FXPlugins` | Flat plugin zoo with plugin-local CPU, Metal, PyTorch, TensorFlow, and NLP implementations. |
| `FXBacktest` | Offline Swift/Metal backtesting and optimization. It does not own raw database access. |
| `FXDataEngineGUI` | macOS operator interface for reports, dashboards, promotion review, and workflow actions. |
| `FXBacktestAgent` | Future remote Mac worker that pulls backtest batches over TCP and reports results. |
| `FXDemoAgent` | Future demo-account execution agent for approved backtest parameters. |
| `FXLiveAgent` | Future live-account execution agent with stricter approval, safety, and broker controls. |

## Database Boundary

FXDatabase is the only project allowed to import or implement ClickHouse access. Other projects must use FXDatabase APIs for:

- Historical market data.
- Backtest result storage and deletion.
- SineTest data.
- Dataset validation and metadata.
- Future data access needed by agents.

This prevents hidden database paths, inconsistent schemas, and backtest runs that cannot be reproduced.

## Plugin Boundary

`FXPlugins` is flat by design. A plugin owns its implementation folder and all plugin-specific accelerator code.

```text
FXPlugins/plugin_name/
  CPU/
  Metal/
  PyTorch/
  TensorFlow/
  NLP/
```

The only shared code is the API and registry layer under `FXPlugins/API/`.

## Agent Boundary

Agents are future runtime projects. They are not data owners.

- `FXBacktestAgent` receives batch work from FXBacktest over TCP and returns results.
- `FXDemoAgent` applies selected parameters to demo accounts across MT5, IBKR, TradingView, and other terminals or broker APIs.
- `FXLiveAgent` applies approved parameters to live accounts with stronger operational gates.
