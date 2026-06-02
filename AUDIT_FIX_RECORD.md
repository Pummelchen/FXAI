# FXAI Audit Fix Record

Source report: `/Users/andreborchert/.codex/attachments/c779ecf5-bc7c-40de-ae88-e5da6a48f625/pasted-text.txt`

## Fixed Findings

- H-1/H-2: Shared M1 write pipeline now centralizes raw insert, canonical rewrite, conflict recording, readback verification, and ingest-operation audit writes for backfill and live ingestion.
- H-1/H-2: M1 timestamp advancement now uses overflow-reporting `MT5ServerSecond.addingOneMinute()` and `UtcSecond.addingOneMinute()`.
- H-3: Quantile-to-normal math now lives in `NormalizationFitTools` and both runtime normalization and fit-state lookups call the shared implementation.
- M-1: `NormalizationWindowRuntimeState` stores a single `config` source of truth, migrates legacy-only payloads on decode, and exposes `legacy` as a compatibility mirror.
- M-3: Feature math now guards short windows before reading previous bars for return, slope, standard deviation, volume activity, and timeframe state helpers.
- M-4: Backtest profit factor is capped at finite `BacktestMetricLimits.maxProfitFactor` instead of returning infinity.
- M-5/M-6: Operational health validates ClickHouse database identifiers, distinguishes schema/query/transport failures, and rejects non-numeric scalar responses.
- M-7: Demo/live request validation now reuses the shared execution-contract non-empty validator.
- L-3: Price scaling now uses a bounded power-of-ten lookup table instead of an unchecked multiplier loop.
- L-4: Plugin symbol hash now uses the high 53 bits of the FNV-1a hash to preserve full `Double` mantissa precision.
- L-5: Operational health server polling timeout is configurable and defaults to 100 ms.
- L-7: Canonical readback row parsing now reports row, field index, field name, and raw value on parse failures.
- L-10: Session-bucket derivation reuses a static UTC calendar instead of allocating one per call.
- L-11: The audit classification and fix status are recorded here.

## Rejected Or Deferred Findings

- H-4: The broad "signed evidence registry" request is not a bug in the current certification entrypoint; `./fxai certify --all` already runs package tests. A wider evidence-signing redesign is deferred.
- M-2: Replacing the operational health raw socket server with a framework server is architecture work, not a localized bug fix. The actionable poll-timeout defect was fixed under L-5.
- L-1: Broad documentation cleanup is not a concrete defect. This record captures the applied audit changes.
- L-2: Changing public plugin `HyperParameters` integer-like `Double` fields to `Int` would break existing plugin API compatibility. Keep as-is unless a versioned plugin API migration is planned.
- L-6: Re-parsing framed protocol payloads through a dictionary would break checksum compatibility because checksums are computed over exact peer payload bytes. The existing raw-byte extraction remains intentional.
- L-8: The proposed FXGUI saved-view state refactor is a broad UI architecture change and was deferred.
- L-9: `sanitizeInputVector` already sanitizes plugin input at the contract boundary; a larger doc/assert sweep is deferred unless a failing path is identified.

## Verification Record

- Completed targeted Swift package tests for FXDatabase, FXDataEngine, FXBacktest, FXExecutionContracts, FXDemoAgent, and FXLiveAgent.
- Completed `./fxai certify --all` after package tests passed.
- Reviewed the final diff against this recorded fix list before commit and push.
