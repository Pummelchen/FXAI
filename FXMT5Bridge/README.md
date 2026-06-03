# FXMT5Bridge

FXMT5Bridge owns the standalone Swift TCP protocol client used by FXAI to talk to the MetaTrader 5 Expert Advisor bridge.

It intentionally contains only framed transport, request DTOs, response DTOs, and bridge errors. It does not depend on FXImporter, FXDatabase, ClickHouse, or strategy/runtime packages.

Current product:

- `MT5Bridge`: socket transport, JSON framing, bridge commands, and MT5 response DTOs.

FXImporter uses this package to build importer connectors. FXDatabase uses this package to ingest and verify MT5-origin M1 data. Keeping the bridge separate prevents FXDatabase from depending on the broader importer package.
