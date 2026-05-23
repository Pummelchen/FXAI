# FXImporter

FXImporter owns external market-data connectors for FXAI. Connectors translate provider-specific APIs into importer contracts for M1 OHLC data, D1 OHLC history, source timestamps, optional UTC timestamps, and volume when the provider supplies it.

Current connectors:

- `Connectors/MetaTrader5/EA/FXDatabase.mq5`: the MT5 Expert Advisor copied into the main MT5 Experts folder as `FXDatabase.mq5`.
- `Sources/MT5Bridge`: the Swift TCP protocol client used by FXDatabase to talk to the MT5 EA.
- `YahooFinanceHistoryConnector`: stateless Yahoo Finance daily-history connector. It uses the public web chart endpoint to fetch D1 OHLCV and adjusted-close data for caller-supplied Yahoo symbols. Yahoo does not provide a contracted official market-data API for this endpoint, so production use must treat provider availability and terms as external risk.

Shared API:

- `Sources/FXImporterAPI`: provider-neutral connector descriptors, capabilities, symbols, health, M1 OHLC batch DTOs, and D1 OHLC batch DTOs.
- `Sources/FXImporter`: umbrella module that re-exports the importer API and current MT5 bridge, plus `MT5ImporterConnector` for mapping MT5 bridge responses into the provider-neutral M1 batch contract and `YahooFinanceHistoryConnector` for D1 historical web data.

The connector API latest version is `fximporter.connector.v1`. Each connector descriptor must declare that version and pass `validateLatestAPI()` before health, symbol, or history calls are accepted. Older connector API versions are not supported.

Future connectors such as Interactive Brokers Client Portal/TWS, TradingView-compatible licensed feeds, or broker-specific feeds should live under this project and expose their data through `FXImporterAPI`. FXDatabase remains responsible for canonical storage, UTC authority, validation, and ClickHouse integration.
