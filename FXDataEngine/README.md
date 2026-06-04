# FXDataEngine

Repository-root Swift data-engine package for FXAI feature construction, normalization, labels, runtime context, and plugin payload preparation.

## Scope

- Swift tools 6.3, Swift language mode 6, macOS 26 deployment floor.
- Canonical market input is M1 OHLCV from FXDatabase.
- Spread, swap, commission, margin, bid/ask, ticks, and execution metadata are not part of this offline data-engine contract.
- Volume is first-class: when any loaded dataset has nonzero volume, the feature pipeline emits volume-derived features and plugin contexts set `dataHasVolume = true`.
- The data-engine contract preserves the 180-feature / 181-input shape for plugin compatibility while using volume-aware OHLCV features.
- AI model training, inference, and tensor execution live in FXPlugins and use PyTorch or TensorFlow per plugin.
- PyTorch and TensorFlow support is represented by explicit backend descriptors and payload DTOs.
- Metal support starts with device probing and kernel descriptors so FXBacktest can adopt accelerated feature/model stages incrementally.
- Offline macro-event and calendar-cache support provides TSV parsing, leakage-safe dataset stats, event-window features, macro-state pressure features, 20-slot macro feature-vector overlays, news gate states, stale detection, and calendar reason payloads.
- Offline factor-context support uses prepared daily closes, carry snapshots, and calendar states; provider symbol lookup is intentionally replaced by explicit Swift inputs.
- The former FXAI Python toolchain now lives in `Tools/` inside this package so non-plugin framework utilities stay with FXDataEngine.

## Build

```bash
swift test
swift build -c release
```

Learned model execution belongs in FXPlugins through PyTorch or TensorFlow backends; FXDataEngine owns deterministic data preparation and runtime contracts.

## Walk-Forward Audit Controls

TestLab exposes time-series cross-validation controls for `market_walkforward`
audits:

```bash
python3.12 FXDataEngine/Tools/fxai_testlab.py run-audit \
  --wf-train-years 1 --wf-test-years 0.25 --wf-window-mode rolling

python3.12 FXDataEngine/Tools/fxai_testlab.py walkforward-analyze
```

`--wf-window-mode rolling` advances fixed-size training windows. `anchored`
keeps the first training bar fixed and expands the training span each fold. The
year/day options resolve to deterministic M1 bar counts before the Swift audit
is launched, and the manifest records the resolved bars, minimum required bars,
and generated window plan. Fold-level Swift evidence is emitted beside the TSV
as `*.walkforward.json`; `walkforward-analyze` automatically loads that sidecar
when present and falls back to the aggregate TSV row when it is absent.
