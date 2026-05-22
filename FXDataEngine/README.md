# FXDataEngine

Repository-root Swift data-engine package for the FXAI migration away from the old MT5/MQL5 backtest path.

## Scope

- Swift tools 6.3, Swift language mode 6, macOS 26 deployment floor.
- Canonical market input is M1 OHLCV from FXDatabase.
- Spread, swap, commission, margin, bid/ask, ticks, and execution metadata are not part of this offline data-engine contract.
- Volume is first-class: when any loaded dataset has nonzero volume, the feature pipeline emits volume-derived features and plugin contexts set `dataHasVolume = true`.
- The migrated contract preserves the old 180-feature / 181-input shape for plugin compatibility while replacing legacy spread/cost-dependent slots with volume-aware OHLCV features.
- The old MQL5 `TensorCore` is intentionally not part of the Swift data-engine port. AI model training, inference, and tensor execution will move into FXPlugins and use PyTorch or TensorFlow per plugin.
- PyTorch and TensorFlow support is represented by explicit backend descriptors and payload DTOs; process runners will be wired when individual AI plugins are converted.
- Metal support starts with device probing and kernel descriptors so FXBacktest can adopt accelerated feature/model stages incrementally.
- Offline macro-event and calendar-cache support mirrors the legacy MQL5 `event_macro` and `runtime_calendar_cache` contracts: TSV parsing, leakage-safe dataset stats, event-window features, macro-state pressure features, 20-slot macro feature-vector overlays, news gate states, stale detection, and calendar reason payloads.
- Offline factor-context support ports the legacy trend/carry/policy/value/commodity scoring math from prepared daily closes, swap snapshots, and calendar states; MT5 symbol lookup is intentionally replaced by explicit Swift inputs.
- The MQL5-to-Swift port completion record is tracked in `Docs/MQL5PortPlan.md`.

## Build

```bash
swift test
swift build -c release
```

The legacy `FXAI/FXDataEngine` MQL5 tree has been removed after the non-tensor data-engine behavior was ported, tested, and promoted into this package. Legacy tensor work is no longer an FXDataEngine gate; learned model execution belongs in FXPlugins through PyTorch or TensorFlow backends.
