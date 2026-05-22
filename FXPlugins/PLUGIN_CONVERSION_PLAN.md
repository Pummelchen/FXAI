# FXPlugins Swift Conversion Plan

Date: 2026-05-23

This plan converts the legacy MQL5 plugins from `FXAI/FXPlugins` into the root
`FXPlugins` area, then ports them one plugin at a time to Swift contracts backed
by `FXDataEngine`. It also connects the two FXBacktest demo/reference plugins to
the same `FXDataEngine` plugin contract surface.

## Hardware And Framework Policy

Use the fastest deterministic backend that preserves plugin behavior:

- Swift scalar: control flow, rule plugins, small online learners, stateful
  bookkeeping, and parity reference code.
- Swift SIMD and Accelerate: dense vector math, reductions, rolling statistics,
  PCA, linear models, distance search, calibration, and CPU fallback paths.
- Metal compute or MPSGraph: large parameter sweeps, embarrassingly parallel
  tree scoring, batched feature/window transforms, convolution/attention/state
  space kernels, and high-throughput backtest evaluation.
- PyTorch with `mps`: sequence models, transformer-family training, RL, and
  research-grade model iteration where Python ecosystem support is materially
  better than hand-coded Swift.
- TensorFlow with `tensorflow-metal`: TensorFlow-native sequence models or
  plugin variants where Keras/TensorFlow tooling is the best fit.
- Core ML / Neural Engine: inference-only deployment path after a PyTorch or
  TensorFlow model is trained and converted. Use it only when the model shape is
  stable and a compiled model can be profiled. Keep CPU/GPU fallback because
  Neural Engine routing is controlled by Core ML, not by a public low-level ANE
  API.

References checked:

- Apple Core ML uses CPU, GPU, and Neural Engine for optimized on-device
  inference: https://developer.apple.com/documentation/CoreML
- Apple MPSGraph uses hardware compute blocks through Metal Performance Shaders:
  https://developer.apple.com/documentation/metalperformanceshadersgraph
- Apple Accelerate provides CPU vector, BLAS/LAPACK, vDSP, and BNNS primitives:
  https://developer.apple.com/accelerate/
- PyTorch MPS runs tensor operations on Apple GPU through Metal/MPSGraph:
  https://docs.pytorch.org/docs/stable/notes/mps.html
- Apple tensorflow-metal accelerates TensorFlow training on Mac GPUs:
  https://developer.apple.com/metal/tensorflow-plugin/

## Scope Count

Legacy MQL5 plugin inventory:

- Total MQL5 plugins: 63
- Distribution: 1
- Factor: 4
- Linear: 6
- Memory: 1
- Mixture: 2
- RL: 1
- Rule: 4
- Sequence: 24
- Stat: 10
- Tree: 5
- Trend: 3
- World: 2

FXBacktest demo/reference plugins to connect to `FXDataEngine`:

- `MovingAverageCross`
- `FXStupid`

`FX7` is already a serious FXBacktest conversion plugin and should be handled in
the later FXBacktest integration wave, not counted as one of the two demo
plugins in this request.

## Directory Plan

1. Move the legacy MQL5 plugin tree up from `FXAI/FXPlugins` to root
   `FXPlugins`, preserving family folders and plugin filenames.
2. Keep the old MQL5 files temporarily as legacy reference files during porting.
   Do not delete them until the Swift implementation for that plugin has a
   passing parity/contract test.
3. Add a SwiftPM package in root `FXPlugins`.
4. Put Swift sources under `FXPlugins/Sources/FXAIPlugins/<Family>/`, mirroring
   the family structure.
5. Add tests under `FXPlugins/Tests/FXAIPluginsTests/<Family>/`.
6. Keep plugin identifiers identical to legacy `AIName()` values.
7. Every Swift plugin implements or adapts to `FXAIPluginV4`.
8. Every plugin manifest sets `requiresVolumeWhenAvailable = true` unless the
   plugin is a pure constant rule that ignores all market inputs.
9. Any legacy use of market spread is replaced by one of:
   - `priceCostPoints` from `PluginContextV4` for execution-cost awareness.
   - OHLCV-derived volatility/liquidity features for market structure.
   - Pair residual/cointegration naming where "spread" meant statistical
     residual, not bid/ask spread.

## Shared Conversion Interfaces

Create these shared Swift pieces before high-complexity ports:

- `FXAIPluginRegistry`: exposes converted plugins as `[any FXAIPluginV4]`.
- `PluginAccelerationPlan`: local metadata for CPU, Swift SIMD, Metal, PyTorch,
  TensorFlow, and Core ML suitability.
- `FXBacktestPluginFXDataEngineAdapter`: bridges FXBacktest demo plugins to
  `FXAIPluginV4` so they can participate in lifecycle, contract, and audit
  checks.
- `PluginParityFixtures`: fixture helpers that build deterministic
  `PredictRequestV4` and `TrainRequestV4` values from `FXDataEngine`.

## Conversion Order

Port from low risk to high risk:

1. Rule plugins: deterministic, small, no tensor backend.
2. FXBacktest demo adapters: connect existing Swift plugins to FXDataEngine.
3. Lightweight framework-model stubs: convert manifest and delegate behavior to
   shared Swift helpers.
4. Linear and trend plugins: direct CPU/SIMD online learning.
5. Distribution, memory, mixture, world, tree plugins: native Swift plus SIMD or
   Metal where profitable.
6. Statistical plugins: Accelerate first, Python optional for complex fitting.
7. Sequence/RL plugins: Python PyTorch/TensorFlow training and optional Core ML
   inference after model stabilization.

Each plugin step:

1. Read MQL5 implementation and identify inputs, state, training, prediction,
   persistence, and any legacy spread references.
2. Write Swift plugin implementation and manifest.
3. Add a focused contract test and an acceleration-plan assertion.
4. Run `swift test` for `FXPlugins`.
5. Run `swift test` for `FXDataEngine` when contracts are touched.
6. Review logic and bug-check API/data-pipeline integration before the next
   plugin.

## Per-Plugin Conversion Matrix

| Plugin | Family | Source | Primary Swift contract | Acceleration plan | Neural Engine/Core ML plan |
|---|---|---|---|---|---|
| dist_quantile | Distribution | Distribution/dist_quantile.mqh | Native Swift distributional learner with replay and calibration | Accelerate for quantile stats, optional Metal for batched scoring | Not first wave; possible Core ML only if converted to neural quantile model |
| factor_carry | Factor | Factor/factor_carry.mqh | Framework-model factor plugin using carry/context features | Swift scalar plus Accelerate reductions | No |
| factor_cmv_panel | Factor | Factor/factor_cmv_panel.mqh | Panel factor plugin using OHLCV context | Accelerate matrix/vector ops | No |
| factor_pca_panel | Factor | Factor/factor_pca_panel.mqh | PCA factor plugin | Accelerate LAPACK/SVD, optional Metal for large panels | No |
| factor_ppp_value | Factor | Factor/factor_ppp_value.mqh | Value factor plugin using prepared context | Swift scalar/Accelerate | No |
| lin_elastic_logit | Linear | Linear/lin_elastic_logit.mqh | Framework-model elastic logistic classifier | Accelerate dot products and regularized updates | No |
| lin_enhash | Linear | Linear/lin_enhash.mqh | Native online hashed linear model | Swift SIMD/Accelerate sparse updates | No |
| lin_ftrl | Linear | Linear/lin_ftrl.mqh | Native FTRL logistic model with replay/calibration | Swift SIMD/Accelerate, Metal only for huge batch scoring | No |
| lin_pa | Linear | Linear/lin_pa.mqh + subfolder | Passive-aggressive online classifier | Swift SIMD/Accelerate | No |
| lin_profit_logit | Linear | Linear/lin_profit_logit.mqh | Profit-aware logistic framework plugin | Swift SIMD/Accelerate | No |
| lin_sgd | Linear | Linear/lin_sgd.mqh | SGD logistic model | Swift SIMD/Accelerate | No |
| mem_retrdiff | Memory | Memory/mem_retrdiff.mqh | Retrieval/difference memory plugin | Accelerate distance search; Metal top-k later | No |
| mix_loffm | Mixture | Mixture/mix_loffm.mqh | Latent online factorization mixture | Accelerate dense math; Metal for batched latent scoring | No initially |
| mix_moe_conformal | Mixture | Mixture/mix_moe_conformal.mqh | Mixture-of-experts conformal router | Swift scalar plus Accelerate gating | No |
| rl_ppo | RL | RL/rl_ppo.mqh | External backend policy plugin | PyTorch MPS preferred for PPO; TensorFlow optional | Core ML inference candidate after training |
| rule_buyonly | Rule | Rule/rule_buyonly.mqh | Constant buy rule plugin | Swift scalar only | No |
| rule_m1sync | Rule | Rule/rule_m1sync.mqh | M1 synchronization/rule plugin using FXDataEngine OHLCV only | Swift scalar; no raw MT5 data access | No |
| rule_random | Rule | Rule/rule_random.mqh | Deterministic seeded random/no-skip rule | Swift scalar | No |
| rule_sellonly | Rule | Rule/rule_sellonly.mqh | Constant sell rule plugin | Swift scalar only | No |
| ai_attn_cnn_bilstm | Sequence | Sequence/ai_attn_cnn_bilstm.mqh | External sequence model | PyTorch MPS preferred; TensorFlow if Keras implementation is cleaner | Core ML inference candidate |
| ai_autoformer | Sequence | Sequence/ai_autoformer.mqh + subfolder | External transformer forecaster | PyTorch MPS preferred | Core ML inference candidate |
| ai_bilstm | Sequence | Sequence/ai_bilstm.mqh | External recurrent model | TensorFlow-metal or PyTorch MPS; choose after op coverage check | Core ML inference candidate |
| ai_chronos | Sequence | Sequence/ai_chronos.mqh + subfolder | External Chronos-like transformer forecaster | PyTorch MPS preferred | Core ML inference candidate |
| ai_cnn_lstm | Sequence | Sequence/ai_cnn_lstm.mqh | External CNN/LSTM model | TensorFlow-metal or PyTorch MPS | Core ML inference candidate |
| ai_fewc | Sequence | Sequence/ai_fewc.mqh | Native/ML hybrid sequence plugin | Swift reference plus PyTorch MPS if model grows | Maybe after stabilization |
| ai_geodesic | Sequence | Sequence/ai_geodesic.mqh + subfolder | Attention/geodesic sequence plugin | PyTorch MPS; Metal kernels for custom attention later | Core ML inference candidate |
| ai_gha | Sequence | Sequence/ai_gha.mqh | Native geometric/attention heuristic | Swift SIMD/Accelerate first, PyTorch only if trained model needed | Maybe |
| ai_gru | Sequence | Sequence/ai_gru.mqh | External GRU model | TensorFlow-metal or PyTorch MPS | Core ML inference candidate |
| ai_lstm | Sequence | Sequence/ai_lstm.mqh + subfolder | External LSTM model | TensorFlow-metal or PyTorch MPS | Core ML inference candidate |
| ai_lstm_tcn | Sequence | Sequence/ai_lstm_tcn.mqh | External LSTM/TCN model | PyTorch MPS preferred | Core ML inference candidate |
| ai_lstmg | Sequence | Sequence/ai_lstmg.mqh + subfolder | LSTM-gated external model | PyTorch MPS preferred | Core ML inference candidate |
| ai_mlp | Sequence | Sequence/ai_mlp.mqh + subfolder | Native MLP or external small MLP | BNNS/Accelerate for CPU, MPSGraph/Metal for batch inference | Core ML inference candidate |
| ai_mythos_rdt | Sequence | Sequence/ai_mythos_rdt.mqh | OpenMythos-inspired recursive decision transformer plugin | PyTorch MPS for training; Swift adapter for FXDataEngine | Core ML inference candidate |
| ai_patchtst | Sequence | Sequence/ai_patchtst.mqh + subfolder | PatchTST external transformer | PyTorch MPS preferred | Core ML inference candidate |
| ai_qcew | Sequence | Sequence/ai_qcew.mqh | Quantile/cross-entropy window model | PyTorch MPS or Swift distributional reference | Maybe |
| ai_s4 | Sequence | Sequence/ai_s4.mqh + subfolder | State-space sequence model | PyTorch MPS preferred; custom Metal later if stable | Core ML uncertain, profile first |
| ai_stmn | Sequence | Sequence/ai_stmn.mqh + subfolder | Spatio-temporal memory network | PyTorch MPS preferred | Core ML inference candidate if ops convert |
| ai_tcn | Sequence | Sequence/ai_tcn.mqh + subfolder | Temporal convolution model | PyTorch MPS or TensorFlow-metal | Core ML inference candidate |
| ai_tesseract | Sequence | Sequence/ai_tesseract.mqh | Native tensorized heuristic/model | Swift SIMD/Metal first, PyTorch if learned tensor model needed | Maybe |
| ai_tft | Sequence | Sequence/ai_tft.mqh + subfolder | Temporal Fusion Transformer | PyTorch MPS preferred | Core ML inference candidate |
| ai_timesfm | Sequence | Sequence/ai_timesfm.mqh + subfolder | TimesFM-style external forecaster | PyTorch MPS preferred | Core ML inference candidate |
| ai_trr | Sequence | Sequence/ai_trr.mqh | Trend/reversal recurrent hybrid | Swift reference plus PyTorch if recurrent training retained | Maybe |
| ai_tst | Sequence | Sequence/ai_tst.mqh + subfolder | Time-series transformer | PyTorch MPS preferred | Core ML inference candidate |
| stat_arimax_garch | Stat | Stat/stat_arimax_garch.mqh | Statistical model plugin | Accelerate for AR terms; Python stats/PyTorch optional for GARCH fitting | No |
| stat_coint_vecm | Stat | Stat/stat_coint_vecm.mqh | Cointegration/VECM plugin | Accelerate LAPACK; replace spread wording with residual | No |
| stat_emd_hht | Stat | Stat/stat_emd_hht.mqh | EMD/HHT signal plugin | Swift/Accelerate; Metal for batched decompositions later | No |
| stat_hmm_regime | Stat | Stat/stat_hmm_regime.mqh | HMM regime plugin | Swift/Accelerate forward-backward; Metal batch scoring optional | No |
| stat_microflow_proxy | Stat | Stat/stat_microflow_proxy.mqh | OHLCV microflow proxy plugin | Swift scalar/Accelerate; must use volume when present | No |
| stat_msgarch | Stat | Stat/stat_msgarch.mqh | Markov-switching GARCH plugin | Accelerate plus optional Python fitting | No |
| stat_ou_spread | Stat | Stat/stat_ou_spread.mqh | OU residual plugin, not bid/ask spread | Accelerate; rename internals to residual | No |
| stat_tvp_kalman | Stat | Stat/stat_tvp_kalman.mqh | Time-varying Kalman plugin | Accelerate matrix ops | No |
| stat_vmd | Stat | Stat/stat_vmd.mqh | Variational mode decomposition plugin | Accelerate/Metal candidate for FFT-like loops | No |
| stat_xrate_consistency | Stat | Stat/stat_xrate_consistency.mqh | Cross-rate consistency plugin | Swift scalar/Accelerate graph checks | No |
| tree_catboost | Tree | Tree/tree_catboost.mqh + subfolder | Native tree ensemble | Swift reference, Metal batched scoring, CPU training first | No |
| tree_lgbm | Tree | Tree/tree_lgbm.mqh + subfolder | Native LightGBM-like ensemble | Swift reference, Metal batched scoring, CPU training first | No |
| tree_rf | Tree | Tree/tree_rf.mqh | Random forest framework plugin | Swift reference, Metal batched scoring if many trees | No |
| tree_xgb | Tree | Tree/tree_xgb.mqh | Native XGBoost-like ensemble | Swift reference, Metal batched scoring, CPU training first | No |
| tree_xgb_fast | Tree | Tree/tree_xgb_fast.mqh | Optimized XGBoost-like ensemble | Swift reference plus Metal batch inference | No |
| trend_tsmom_vol | Trend | Trend/trend_tsmom_vol.mqh | Time-series momentum/vol trend plugin | Swift SIMD/Accelerate rolling windows | No |
| trend_vol_breakout | Trend | Trend/trend_vol_breakout.mqh | Volatility breakout plugin | Swift SIMD/Accelerate; Metal sweep possible | No |
| trend_xsmom_rank | Trend | Trend/trend_xsmom_rank.mqh | Cross-sectional momentum rank plugin | Accelerate sorting/reductions, Metal not first | No |
| wm_cfx | World | World/wm_cfx.mqh | Currency factor world model | Accelerate graph/matrix ops; Metal later | Maybe after model stabilization |
| wm_graph | World | World/wm_graph.mqh | Graph world model | Accelerate graph ops; PyTorch Geometric alternative if needed | Maybe after model stabilization |
| MovingAverageCross | FXBacktest demo | FXPlugins/Sources/FXAIPlugins/FXBacktestDemo/MovingAverageCrossFXDataEnginePlugin.swift | FXDataEngine adapter now independent of FXBacktest-local plugin code | Metal candidate for future FXDataEngine batch/sweep work; Swift scalar adapter now active | No |
| FXStupid | FXBacktest demo | FXPlugins/Sources/FXAIPlugins/FXBacktestDemo/FXStupidFXDataEnginePlugin.swift | FXDataEngine adapter now independent of FXBacktest-local plugin code | Stateful scalar adapter; no Metal until its order-control flow is redesigned | No |

## Review Pass 1

Findings:

- The inventory count of 63 matches `FXDataEngineConstants.aiCount` and the
  `AIModelID` enum count.
- The existing root `FXPlugins` folder has only documentation, so moving the
  nested MQL5 tree up will not overwrite converted Swift code.
- Several plugin names contain "spread"; those are not bid/ask spread
  dependencies by default. During conversion, inspect each occurrence and map it
  either to statistical residual or to `priceCostPoints`.
- Sequence and RL plugins should not be hand-reimplemented as large Swift neural
  networks first. The maintainable path is a Swift contract adapter with
  PyTorch/TensorFlow backend declaration, then optional Core ML inference.
- Rule and simple linear plugins should be native Swift first because Python
  process overhead would dominate.

Revision from pass 1:

- Start with `rule_buyonly`, `rule_sellonly`, `rule_random`, then
  `rule_m1sync`.
- Add demo adapters after the first rule plugin compiles, because those adapters
  need the same registry and test helpers.

## Review Pass 2

Findings:

- The plan does not require changing `FXDataEngine` contracts for the first
  plugin wave. Use existing `FXAIPluginV4`, `PluginManifestV4`,
  `PredictRequestV4`, `TrainRequestV4`, and `PredictionV4`.
- A root `FXPlugins` SwiftPM package can depend on `../FXDataEngine` and
  `../FXBacktest`; it should not create a dependency from `FXDataEngine` back to
  `FXPlugins`.
- The first plugins should be tested only through `FXPlugins` tests and
  `FXDataEngine` contract tests. Full FXBacktest tests are needed when demo
  adapters are added.
- Apple Neural Engine should be treated as a Core ML inference deployment route,
  not as a direct plugin compute backend.

Revision from pass 2:

- Implement a local `PluginAccelerationPlan` in `FXPlugins` rather than changing
  `FXDataEngine` now.
- Keep converted plugins in a registry owned by `FXPlugins`; later FXAI agents
  can import that registry.
- Do not delete nested `FXAI/FXPlugins` until the move is committed and verified
  by `git status`, because current tests and docs may still reference it.

## Immediate Coding Plan

1. Copy the nested plugin tree into root `FXPlugins` preserving the family
   structure.
2. Add a root `FXPlugins/Package.swift` package.
3. Add shared Swift support:
   - `PluginAccelerationPlan`
   - `FXAIPluginRegistry`
   - test request builders
4. Convert `rule_buyonly`.
5. Run `swift test` in `FXPlugins`.
6. Review `rule_buyonly` for logic and API integration before converting the
   next plugin.

## Implemented Migration Wave

- Root `FXPlugins` is now a SwiftPM package depending on `FXDataEngine` and
  `FXBacktest`.
- The four legacy rule plugins are Swift-native and contract-tested.
- The two FXBacktest demo plugins are exposed through `FXDataEngine`
  adapters.
- The remaining 59 legacy plugin identifiers are exposed through generated
  Swift adapters with volume-aware online centroid learning and deterministic
  fallback prediction.
- Every Swift-era plugin has acceleration metadata covering Swift scalar/SIMD,
  Accelerate, Metal, PyTorch MPS, TensorFlow Metal, and Core ML / Neural Engine
  suitability where applicable.
- `PythonMLBackendBridge` now runs plugin-local Python backends through a JSON
  process contract and sends OHLCV feature vectors, sequence windows, volume
  availability, horizon, min-move, and price-cost context.
- `Python/fxai_plugin_backend.py` provides the generic PyTorch/TensorFlow
  backend entrypoint, including PyTorch MPS / TensorFlow Metal dispatch when the
  frameworks are installed, persisted online backend state, volume-availability
  gating, and a pure-Python fallback for local contract tests.
- Generated adapters now route through a shared Swift family runtime with
  online class heads, move heads, context calibration, sequence-window state,
  tree-style stumps, memory retrieval slots, and mixture experts. This gives all
  59 non-handwritten plugins executable family-specific behavior while
  per-plugin parity fixtures are built.

Remaining parity work after this wave is plugin-specific: replace generated
surrogate bodies with direct Swift, Metal, PyTorch, or TensorFlow implementations
and add per-plugin MQL5 parity fixtures before deleting the matching legacy
reference file.

## Implementation Status

Completed first Swift wave on 2026-05-23:

- Root `FXPlugins` now contains the copied legacy MQL5 plugin reference tree.
- Added a SwiftPM `FXAIPlugins` package wired to `FXDataEngine`.
- Added `PluginAccelerationPlan` metadata for backend suitability.
- Added `FXAIPluginRegistry` for converted plugin discovery.
- Converted and tested `rule_buyonly`, `rule_sellonly`, `rule_random`, and
  `rule_m1sync`.
- Added and tested FXDataEngine adapters for the two FXBacktest demo plugins:
  `MovingAverageCross` and `FXStupid`.
- Removed the now-duplicated FXBacktest-local demo plugin implementations after
  the adapters became independent; FXBacktest keeps only backtest-native FX7.
- Extended `AIModelID` and the FXDataEngine model count from 63 to 65 so the two
  demo adapters have stable model identifiers.

Completed full Swift adapter wave on 2026-05-23:

- Added generated Swift `FXAIPluginV4` adapters for the other 59 legacy MQL5
  plugins.
- `FXAIPluginRegistry` now exposes all 65 model IDs: 63 legacy FXAI plugins plus
  the 2 FXBacktest demo adapters.
- Every generated adapter has a validated manifest, volume-aware online
  centroid learning, deterministic fallback prediction, and explicit Apple
  Silicon backend metadata.
- Sequence and RL plugins now have Swift contract adapters and declared
  PyTorch MPS, TensorFlow Metal, and/or Core ML inference candidates.
- `PythonMLBackendBridge` and `Python/fxai_plugin_backend.py` provide the
  process-level PyTorch/TensorFlow execution path for backend-owned plugins.

Remaining work:

- Replace generated surrogate prediction with full per-plugin native logic or
  model-specific PyTorch/TensorFlow training/inference as each backend is
  implemented.
- Add parity fixtures for every non-generated algorithm body before removing
  its MQL5 reference file.
- Remove or retire MQL5 reference files only after each corresponding Swift or
  backend plugin passes contract, parity, and acceleration-plan tests.
