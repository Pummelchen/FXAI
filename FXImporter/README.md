# FXImporter

FXImporter owns external market-data connectors for FXAI. Connectors translate provider-specific APIs into one importer contract for M1 OHLC data, source timestamps, optional UTC timestamps, and volume when the provider supplies it.

Current connector:

- `Connectors/MetaTrader5/EA/FXDatabase.mq5`: the MT5 Expert Advisor copied into the main MT5 Experts folder as `FXDatabase.mq5`.
- `Sources/MT5Bridge`: the Swift TCP protocol client used by FXDatabase to talk to the MT5 EA.

Shared API:

- `Sources/FXImporterAPI`: provider-neutral connector descriptors, capabilities, symbols, health, and M1 OHLC batch DTOs.
- `Sources/FXImporter`: umbrella module that re-exports the importer API and current MT5 bridge, plus `MT5ImporterConnector` for mapping MT5 bridge responses into the provider-neutral M1 batch contract.

Future connectors such as Interactive Brokers/TWS, Yahoo Finance History, TradingView, or broker-specific feeds should live under this project and expose their data through `FXImporterAPI`. FXDatabase remains responsible for canonical storage, UTC authority, validation, and ClickHouse integration.
