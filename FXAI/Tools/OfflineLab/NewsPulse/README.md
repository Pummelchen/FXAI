# NewsPulse

NewsPulse is FXAI's phase-1 news-risk subsystem for live FX trading.

It is not a new prediction model. It is shared infrastructure that combines:
- MT5 Economic Calendar state for scheduled macro events
- free GDELT DOC 2.0 polling for breaking-news bursts
- deterministic currency and pair risk-state aggregation
- runtime trade gating and operator visibility

Phase 1 is intentionally gating-first:
- no paid data feeds
- no calendar scraping
- no broad NLP stack
- no forced retraining of the existing model zoo
- no change to the canonical model-input contract by default

## Architecture

NewsPulse has four parts:

1. `Services/FXAI_NewsPulseCalendar.mq5`
   MT5 Service that exports Economic Calendar changes into `FILE_COMMON`.

2. `Tools/offline_lab/newspulse_*.py`
   Python collector, GDELT query layer, fusion logic, daemon loop, and service install helpers.

3. `Engine/Runtime/Trade/runtime_trade_newspulse.mqh`
   MT5 runtime adapter that reads the flat snapshot and applies pair-level gates.

4. `GUI/Sources/.../NewsPulse*.swift`
   macOS GUI surface for source health, currency heatmap, pair risk, and recent tape.

## Canonical Artifacts

Shared runtime artifacts are written under:

`FILE_COMMON/FXAI/Runtime/`

Main files:
- `news_snapshot.json`
- `news_snapshot_flat.tsv`
- `news_history.ndjson`
- `news_calendar_feed.tsv`
- `news_calendar_state.tsv`
- `news_calendar_history.ndjson`

Local operator mirrors and config live under:

`FXAI/Tools/OfflineLab/NewsPulse/`

Tracked operator-editable files:
- `newspulse_config.json`
- `newspulse_sources.json`

Generated local state:
- `newspulse_status.json`
- `news_history.ndjson`
- `State/newspulse_state.json`

## Time Semantics

MT5 Economic Calendar time uses trade-server semantics.

NewsPulse preserves that explicitly:
- raw trade-server timestamps are retained for audit and history
- UTC-derived timestamps are emitted for fusion, freshness, and GUI/runtime consumers
- stale or invalid calendar freshness is treated as unknown, not safe

## Setup

Install and compile the MT5 calendar service:

```bash
python3 FXAI/Tools/fxai_offline_lab.py newspulse-install-service
```

Then start the service from MT5:

`Navigator -> Services -> FXAI_NewsPulseCalendar`

The service is per terminal or machine, not per chart.

## Commands

Validate NewsPulse config and paths:

```bash
python3 FXAI/Tools/fxai_offline_lab.py newspulse-validate
```

Run a one-shot fusion cycle:

```bash
python3 FXAI/Tools/fxai_offline_lab.py newspulse-once
```

Run the daemon continuously:

```bash
python3 FXAI/Tools/fxai_offline_lab.py newspulse-daemon --interval-seconds 60
```

## Config

Main config:

`FXAI/Tools/OfflineLab/NewsPulse/newspulse_config.json`

Controls:
- poll interval
- calendar stale thresholds
- GDELT request timeout
- max query sets per cycle
- max per-cycle runtime budget
- rate-limit backoff
- calendar pre/post windows
- risk thresholds
- supported currencies and topic groups

Source whitelist:

`FXAI/Tools/OfflineLab/NewsPulse/newspulse_sources.json`

Controls:
- domain whitelist
- quality tier
- enabled or disabled state
- optional currency restrictions
- optional language and source-country restrictions

Only whitelisted sources enter scoring.

## Runtime Behavior

The phase-1 MT5 runtime integration is gating-first.

NewsPulse can:
- block entries when high-impact events are imminent or just printed
- block or caution pairs when source state is stale or unknown
- raise entry confidence floors during caution windows
- scale lot size down during caution windows
- emit machine-readable reasons for blocks and cautions

It does not alter model weights or the canonical feature vector by default.

## GUI Behavior

The macOS GUI NewsPulse surface shows:
- source health and staleness
- currency heatmap
- pair risk table
- upcoming event countdowns
- recent event tape
- active pair gates and reasons where runtime state is present

## Stale And Failure Rules

NewsPulse fails safe.

If the calendar collector is not running, GDELT is stale, or the merged snapshot is missing:
- state is treated as unknown
- runtime can block entries if `NewsPulseBlockOnUnknown` is enabled
- GUI surfaces show stale or failed source state instead of pretending conditions are safe

## History And Replay

`news_history.ndjson` is append-only.

It records:
- merged snapshot states
- newly seen GDELT items
- enough timestamp and source metadata to replay what NewsPulse knew at a given time

## Phase 1 vs Deferred

Phase 1 includes:
- scheduled macro awareness
- breaking-news burst awareness
- runtime trade gating
- GUI observability
- append-only audit history

Deferred:
- broad institutional feed expansion
- heavier NLP models
- directional news alpha as a primary trading signal
- universal model-vector expansion across the whole zoo
