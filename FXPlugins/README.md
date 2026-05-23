# FXPlugins

Repository-root plugin zoo shared across FXAI runtimes.

AI plugins own model execution. When a converted plugin needs tensor training or inference, it should use a plugin-local PyTorch or TensorFlow backend rather than re-creating the old MQL5 `TensorCore` inside Swift FXDataEngine.

Converted plugins should consume the Swift FXDataEngine OHLCV contracts and use volume-derived features whenever the loaded dataset has nonzero volume.

## Layout

- `API/`: non-plugin package surface only: registry, tests, docs, and backend process hooks. Shared implementation primitives live in `FXDataEngine`; plugin folders should not depend on shared plugin-zoo helpers.
- `<plugin_id>/`: every plugin lives directly under `FXPlugins` in its own folder named after its manifest `aiName`, for example `lin_sgd/`, `ai_lstm/`, `tree_xgb_fast/`, `rule_m1sync/`, and `fxbacktest_moving_average_cross/`. All plugin-specific Swift, Metal, Python, model assets, and state adapters belong inside that plugin folder.
- `Package.swift`: SwiftPM boundary for the zoo. There is no longer a root `Sources/`, `Tests/`, or `Python/` staging layout.

## Current Swift Coverage

Root `FXPlugins` now exposes all 65 FXAI model IDs through Swift `FXAIPluginV4`
contracts:

- 4 hand-ported legacy rule plugins.
- 2 former FXBacktest demo adapters: `fxbacktest_moving_average_cross` and `fxbacktest_fxstupid`.
- 15 full plugin-owned native conversions: `lin_sgd`, `lin_ftrl`, `lin_enhash`, `lin_pa`,
  `lin_elastic_logit`, `lin_profit_logit`, `dist_quantile`, `mem_retrdiff`, `mix_loffm`,
  `mix_moe_conformal`, `tree_xgb_fast`, `tree_xgb`, `tree_catboost`, `tree_lgbm`, and
  `tree_rf`,
  with Swift CPU code under each plugin's `CPU/` folder and accelerator sources under plugin-owned
  `Metal/` or `PyTorch/` folders where suitable.
- 44 Swift reference adapters for the remaining legacy plugins, each in its own plugin folder with
  volume-aware online centroid learning, deterministic fallback prediction, and explicit Apple Silicon
  backend metadata for Swift SIMD, Accelerate, Metal, PyTorch MPS,
  TensorFlow Metal, or Core ML / Neural Engine candidates.
- Python-backed plugins can use `PythonMLBackendBridge` from `FXDataEngine`.
  The bridge sends OHLCV feature vectors, sequence windows, volume availability,
  horizon, min-move, and price-cost context to plugin-local PyTorch/TensorFlow
  code. The included backend has a pure-Python fallback for contract tests and
  selects PyTorch MPS or TensorFlow Metal acceleration when those frameworks are
  installed. Training calls persist lightweight online state under
  `FXAI_PLUGIN_STATE_DIR`, or `~/.fxai/plugins/state` when the environment
  variable is not set. The backend follows the FXDataEngine volume contract:
  volume-derived features are used only when `dataHasVolume` is true.

Run the local verification gate with:

```bash
swift test
```

The legacy MQL5 plugin reference files have been removed from the repository. The current source of truth is this plugin-owned Swift zoo plus the full conversion plan in `API/Docs/FULL_PLUGIN_CONVERSION_PLAN.md`.
