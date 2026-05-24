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
      +--> FXGUI
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
| `FXGUI` | macOS operator interface for reports, dashboards, promotion review, and workflow actions. |
| `FXBacktestAgent` | Future remote Mac worker that pulls backtest batches over TCP and reports results. |
| `FXDemoAgent` | Future demo-account execution agent for approved backtest parameters. |
| `FXLiveAgent` | Future live-account execution agent with stricter approval, safety, and broker controls. |

## Database Boundary

FXDatabase is the only project allowed to import or implement ClickHouse access. Other projects must use FXDatabase APIs for:

- Historical market data.
- Backtest result storage and deletion.
- SineTest data, including the persistent synthetic M1 OHLCV series from `2000-01-01` through runtime-now that is refreshed every 10 seconds by FXDatabase.
- Dataset validation and metadata.
- Future data access needed by agents.

This prevents hidden database paths, inconsistent schemas, and backtest runs that cannot be reproduced.

FXPlugins treats SineTest as a required certification fixture. The full registry gate checks every plugin on a broad holdout window, and the accelerator gate switches each plugin through every declared non-CPU backend so Metal, PyTorch MPS, TensorFlow Metal, and NLP runtime paths cannot bypass SineTest prediction safety. Every evaluated plugin and accelerator prediction must also report at least 85% confidence on this deterministic fixture.

## API Version Boundary

Every project-to-project API has one supported latest version. FXAI does not support older API versions in parallel. A caller using an older version is rejected at descriptor or request validation time.

| Boundary | Latest version |
| --- | --- |
| FXImporter connector API | `fximporter.connector.v1` |
| FXDatabase FXBacktest API | `fxdatabase.fxbacktest.v1` |
| FXBacktest plugin API | `fxbacktest.plugin-api.v1` |
| FXBacktest plugin acceleration API | `fxbacktest.plugin-acceleration.v1` |
| FXBacktest plugin IR | `fxbacktest.plugin-ir.v1` |
| FXDataEngine / FXPlugins API | `4` |
| FXDataEngine tokenizer contract | `fxai-tokenizer-v1` |

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
