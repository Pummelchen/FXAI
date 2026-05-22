# FXDataEngine

Repository-root Swift data-engine package for the FXAI migration away from the old MT5/MQL5 backtest path.

## Scope

- Swift tools 6.3, Swift language mode 6, macOS 26 deployment floor.
- Canonical market input is M1 OHLCV from FXDatabase.
- Spread, swap, commission, margin, bid/ask, ticks, and execution metadata are not part of this offline data-engine contract.
- Volume is first-class: when any loaded dataset has nonzero volume, the feature pipeline emits volume-derived features and plugin contexts set `dataHasVolume = true`.
- The migrated contract preserves the old 180-feature / 181-input shape for plugin compatibility while replacing legacy spread/cost-dependent slots with volume-aware OHLCV features.
- PyTorch and TensorFlow support is represented by explicit backend descriptors and payload DTOs; process runners will be wired when individual AI plugins are converted.
- Metal support starts with device probing and kernel descriptors so FXBacktest can adopt accelerated feature/model stages incrementally.

## Build

```bash
swift test
swift build -c release
```

The current MT5/MQL5 reference implementation remains under `FXAI/FXDataEngine/` until each layer has been ported, tested, and promoted into this package.
