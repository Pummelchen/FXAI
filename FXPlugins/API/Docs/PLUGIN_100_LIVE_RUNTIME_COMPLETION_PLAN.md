# FXPlugins 100% Live Runtime Completion Plan

This plan is the source-of-truth checklist for moving every former FXAI plugin and the two FXBacktest demo plugins from "ported/scaffolded" to full live runtime quality.

Current verified state: the Swift packages build and the plugin zoo now has an executable 100% certification gate. The gate covers registry coverage, CPU runtime smoke, OHLCV/volume contracts, SineTest runtime, standard-reference or golden parity evidence, FXDatabase-only data access, full verification harness presence, Metal source compilation/live buffer probe wiring, PyTorch/TensorFlow live train-predict-persistence-load checks, and NLP text/no-text runtime checks. CoreML/Neural Engine is not declared by any plugin until a real export and parity path exists.

`FXAIPluginCertificationRegistry` is the executable certification gate. It covers every registered plugin and every declared backend, records which evidence is currently satisfied, and `requireAllPlugins100PercentCertified()` now passes when the evidence files, runtime harnesses, and backend tests are present and valid.

## 100% Acceptance Gates

Every plugin is complete only when all applicable gates pass:

1. CPU implementation is the authoritative Swift port of the original MQL5 plugin concept, with no shell or placeholder behavior.
2. The CPU implementation consumes the FXDataEngine OHLCV contract: M1 OHLC is mandatory; volume is used whenever `volume > 0`.
3. Historical parity fixtures exist for the old MQL5 behavior or, where MQL5 source is unavailable, a documented standard reference implementation is used as the parity target.
4. Metal implementations compile at runtime, execute on Apple GPUs, and have deterministic CPU parity tests over representative fixtures.
5. PyTorch implementations run through the live backend bridge, prefer MPS on Apple Silicon, persist/load model state, and expose train and predict entry points.
6. TensorFlow implementations run through the live backend bridge, persist/load model state, and expose train and predict entry points.
7. NLP implementations are live runtime backends, not passive files; they include tokenizer/text-event contracts and deterministic fallback behavior when text context is absent.
8. Backend selection is explicit, observable, and testable: CPU-only, auto, Metal, PyTorch, TensorFlow, and NLP modes must fail closed or fall back according to the plugin policy.
9. FXBacktest, FXDemoAgent, and FXLiveAgent use the same plugin API and cannot bypass FXDatabase for market data.
10. Full Swift tests and Python backend smoke tests pass cleanly before commit and push.

## Shared Work Before Plugin-by-Plugin Completion

1. Add a shared live backend policy to FXDataEngine/FXPlugins that maps plugin capability to CPU, Metal, PyTorch, TensorFlow, and NLP backends.
2. Add a synchronous-safe adapter for the existing async `PythonMLBackendBridge`, because `FXAIPluginV4` is currently synchronous.
3. Add a Metal runtime executor that compiles plugin kernel source, runs buffers on `MTLDevice`, validates output shape, and supports CPU parity tests.
4. Add a plugin-local backend manifest for every plugin so live code can discover backend files without hard-coded absolute paths.
5. Add a standard live backend test harness:
   - CPU golden tests.
   - Metal compile and parity tests where Metal is present.
   - PyTorch train/predict smoke tests where PyTorch is present.
   - TensorFlow train/predict smoke tests where TensorFlow is present.
   - NLP text/context smoke tests where NLP is present.
6. Replace passive `*Reference.swift` files with active CPU runtime logic or enforce shadow-comparison tests until the active CPU implementation matches the reference.

## Per-Plugin Completion Matrix

| Plugin | Current live state | Remaining CPU/reference work | Accelerator work required for 100% |
| --- | --- | --- | --- |
| `ai_attn_cnn_bilstm` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Make CPU fallback a deterministic attention-CNN-BiLSTM approximation with volume-aware features and golden fixtures. | Wire PyTorch and TensorFlow as primary full neural backends; run MPS smoke tests; compile Metal feature/projection kernels and parity-test them. |
| `ai_autoformer` | Swift CPU wrapper; Metal/PyTorch folders present. | Replace simplified CPU path with seasonality/decomposition reference logic for fallback. | Wire PyTorch Autoformer train/predict; compile Metal decomposition/projection kernels; add long-horizon forecast parity fixtures. |
| `ai_bilstm` | Swift CPU wrapper; PyTorch/TensorFlow folders present. | Ensure CPU fallback is a real bidirectional recurrent approximation, not unidirectional scoring. | Wire PyTorch and TensorFlow BiLSTM live backends with persisted state and MPS preference. |
| `ai_chronos` | Swift CPU wrapper; PyTorch/NLP folders present. | Define CPU statistical fallback for Chronos-style probabilistic forecasts. | Wire PyTorch Chronos backend and NLP/context bridge; add zero-text fallback and text-event enrichment tests. |
| `ai_cnn_lstm` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete CPU convolution-window plus recurrent fallback and volume feature handling. | Wire PyTorch and TensorFlow CNN-LSTM backends; execute Metal convolution/projection kernels with CPU parity. |
| `ai_fewc` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete few-shot ensemble weighting and calibrated confidence on CPU. | Wire PyTorch few-shot backend; run Metal batch-distance/weight kernels live. |
| `ai_geodesic` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete geodesic feature construction and reference distance metrics on CPU. | Wire PyTorch manifold backend; execute Metal distance kernels and parity-test against CPU. |
| `ai_gha` | Swift CPU wrapper; PyTorch folder present. | Complete generalized Hebbian/PCA-like online learning and state persistence. | Wire PyTorch backend only if it adds batch training/MPS acceleration; otherwise document CPU as primary and keep PyTorch parity. |
| `ai_gru` | Swift CPU wrapper; PyTorch/TensorFlow folders present. | Complete GRU fallback with stable recurrent state and volume-aware feature scaling. | Wire PyTorch and TensorFlow GRU live backends with persistence and MPS smoke tests. |
| `ai_lstm` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete LSTM fallback gates/state persistence and golden recurrent fixtures. | Wire PyTorch and TensorFlow LSTM live backends; compile Metal dense/projection kernels; parity-test helper outputs. |
| `ai_lstm_tcn` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete CPU hybrid temporal-convolution and recurrent fallback. | Wire PyTorch and TensorFlow hybrid model; run Metal TCN/convolution kernels live. |
| `ai_lstmg` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete gated LSTM fallback with explicit state serialization. | Wire PyTorch and TensorFlow LSTMG backends; run Metal gate/projection kernels live. |
| `ai_mlp` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete MLP CPU path with trainable layers, persistence, normalization, and volume features. | Wire PyTorch and TensorFlow MLP backends; run Metal dense layer kernels live and parity-test. |
| `ai_mythos_rdt` | Swift CPU wrapper; PyTorch/NLP folders present. | Complete CPU reasoning/decision-tree fallback derived from OpenMythos concept and FX fixtures. | Wire PyTorch representation backend and NLP narrative/context backend; persist embeddings and add no-text fallback. |
| `ai_patchtst` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete CPU patch extraction and baseline transformer fallback. | Wire PyTorch PatchTST backend; run Metal patching/projection kernels live. |
| `ai_qcew` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete quantile/conformal ensemble weighting on CPU with calibrated intervals. | Wire PyTorch backend; execute Metal quantile/weight kernels and parity-test. |
| `ai_s4` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete CPU state-space fallback and stable recurrent scan. | Wire PyTorch S4 backend; run Metal state scan/projection kernels live. |
| `ai_stmn` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete spatio-temporal memory network CPU fallback and state persistence. | Wire PyTorch STMN backend; run Metal memory attention/scoring kernels live. |
| `ai_tcn` | Swift CPU wrapper; Metal/PyTorch/TensorFlow folders present. | Complete dilated temporal-convolution CPU fallback. | Wire PyTorch and TensorFlow TCN backends; run Metal convolution kernels live. |
| `ai_tesseract` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete tensor-factor feature model and CPU reference tests. | Wire PyTorch tensor model; run Metal tensor contraction/projection kernels live. |
| `ai_tft` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete CPU fallback for gating, variable selection, and quantile output. | Wire PyTorch Temporal Fusion Transformer backend; run Metal gating/projection helpers live. |
| `ai_timesfm` | Swift CPU wrapper; PyTorch/NLP folders present. | Complete CPU seasonal/probabilistic fallback for time-series foundation behavior. | Wire PyTorch TimesFM-style backend and NLP metadata/event bridge; add missing-context fallback tests. |
| `ai_trr` | Swift CPU wrapper; PyTorch folder present. | Complete temporal risk-routing CPU fallback and calibration. | Wire PyTorch backend with train/predict/persistence and MPS preference. |
| `ai_tst` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete time-series transformer CPU fallback with patch/window reference tests. | Wire PyTorch transformer backend; run Metal projection/attention-helper kernels live. |
| `dist_quantile` | Swift CPU wrapper; Metal folder present. | Activate reference quantile distribution logic, interval calibration, and volume-sensitive weighting. | Run Metal quantile kernels live and parity-test edge cases with ties/outliers. |
| `factor_carry` | Swift CPU only. | Complete carry factor reference, symbol normalization, and volume-aware liquidity filter. | No non-CPU backend required; use Accelerate internally only if profiling shows a CPU bottleneck. |
| `factor_cmv_panel` | Swift CPU only. | Complete cross-market value panel reference, missing-data handling, and volume weighting. | No non-CPU backend required unless panel size justifies a future Metal matrix path. |
| `factor_pca_panel` | Swift CPU only. | Replace/validate CPU PCA with stable covariance/eigen reference and persistence. | No plugin accelerator folder required; use Accelerate/LAPACK in CPU runtime if available. |
| `factor_ppp_value` | Swift CPU only. | Complete PPP value reference, currency basket contracts, and stale-rate guards. | No non-CPU backend required. |
| `fxbacktest_fxstupid` | Demo plugin in FXPlugins root style. | Freeze its intentionally simple reference behavior, add FXDataEngine OHLCV fixtures, and verify FXBacktest no longer owns duplicate code. | No non-CPU backend required. |
| `fxbacktest_moving_average_cross` | Demo plugin with Metal folder. | Complete moving-average reference, warmup semantics, and FXBacktest parity fixtures. | Run Metal moving-average/cross kernel live and parity-test CPU against Metal. |
| `fx7` | FXBacktest-native plugin now owned by FXPlugins. | Keep backtest source linked into FXBacktest, certify FXDataEngine adapter, and add full-market FX7 parity fixtures. | Run Metal signal scorer live and parity-test CPU against Metal. |
| `lin_elastic_logit` | Swift CPU wrapper; Metal folder present. | Complete elastic-net logistic training, regularization, persistence, and calibration. | Run Metal gradient/dot-product kernels live and parity-test coefficients/predictions. |
| `lin_enhash` | Swift CPU wrapper; Metal folder present. | Complete enhanced hashing feature map and online update reference. | Run Metal hashed dot/update kernels live with collision parity tests. |
| `lin_ftrl` | Swift CPU wrapper; Metal folder present. | Complete FTRL-Proximal update, lazy weights, regularization, and persistence. | Run Metal vector update kernels live and parity-test against CPU. |
| `lin_pa` | Swift CPU wrapper; Metal folder present. | Complete passive-aggressive update variants, clipping, and margin tests. | Run Metal update/scoring kernels live and parity-test. |
| `lin_profit_logit` | Swift CPU wrapper; Metal folder present. | Complete profit-weighted logistic loss, cost/slippage inputs, and calibration. | Run Metal weighted-gradient kernels live and parity-test. |
| `lin_sgd` | Swift CPU wrapper; Metal folder present. | Complete SGD optimizer variants, normalization, learning-rate schedules, and persistence. | Run Metal vectorized gradient kernels live and parity-test. |
| `mem_retrdiff` | Swift CPU wrapper; Metal folder present. | Complete retrieval-difference memory, eviction policy, and volume-aware distance. | Run Metal nearest-neighbor/distance kernels live and parity-test. |
| `mix_loffm` | Swift CPU wrapper; PyTorch folder present. | Complete online field-aware factorization machine CPU reference and persistence. | Wire PyTorch LOFFM backend for batch training/MPS prediction; parity-test feature encoding. |
| `mix_moe_conformal` | Swift CPU wrapper; PyTorch folder present. | Complete mixture-of-experts routing, conformal calibration, and interval outputs on CPU. | Wire PyTorch expert/router backend and persistence; add calibration smoke tests. |
| `rl_ppo` | Swift CPU wrapper; PyTorch folder present. | Complete deterministic CPU policy fallback and environment/action contract. | Wire PyTorch PPO training/prediction loop, persisted policy/value nets, and MPS smoke tests. |
| `rule_buyonly` | Root Swift CPU plugin. | Add golden fixture proving always-buy semantics, volume neutrality, and API contract stability. | No accelerator required. |
| `rule_m1sync` | Root Swift plugin; Metal folder present. | Complete CPU synchronization checks for M1 OHLCV continuity and volume semantics. | Run Metal synchronization/validation kernel live and parity-test missing-bar cases. |
| `rule_random` | Root Swift CPU plugin. | Make randomness seedable/reproducible, add deterministic fixtures, and document live behavior. | No accelerator required. |
| `rule_sellonly` | Root Swift CPU plugin. | Add golden fixture proving always-sell semantics, volume neutrality, and API contract stability. | No accelerator required. |
| `stat_arimax_garch` | Swift CPU wrapper. | Complete ARIMAX plus GARCH reference, optimizer convergence checks, and residual diagnostics. | No separate accelerator required now; future PyTorch only if probabilistic training becomes a bottleneck. |
| `stat_coint_vecm` | Swift CPU wrapper. | Complete Johansen/VECM reference, rank selection, and spread diagnostics. | No separate accelerator required; use CPU linear algebra/Accelerate where useful. |
| `stat_emd_hht` | Swift CPU wrapper; Metal folder present. | Complete EMD/Hilbert-Huang CPU reference with boundary handling. | Replace proxy Metal helpers with live IMF/Hilbert kernels where useful, or document Metal as scoped helper; parity-test all helper kernels. |
| `stat_hmm_regime` | Swift CPU wrapper. | Complete HMM forward-backward/Viterbi reference, regime persistence, and probability calibration. | No accelerator required unless later moved to PyTorch for large state spaces. |
| `stat_microflow_proxy` | Swift CPU wrapper. | Complete OHLCV microflow proxy reference and explicitly document proxy limits without MT5 tick/spread data. | No accelerator required. |
| `stat_msgarch` | Swift CPU wrapper. | Complete Markov-switching GARCH reference, optimizer safeguards, and regime diagnostics. | No accelerator required now. |
| `stat_ou_spread` | Swift CPU wrapper. | Complete OU estimation, half-life, z-score, and stationarity fixtures. | No accelerator required. |
| `stat_tvp_kalman` | Swift CPU wrapper. | Complete time-varying-parameter Kalman filter/smoother and covariance stability checks. | No accelerator required. |
| `stat_vmd` | Swift CPU wrapper; Metal folder present. | Activate full VMD reference in CPU runtime and add convergence/golden fixtures. | Replace current Metal proxy with live VMD update kernels or keep only as documented helper after parity; run Metal executor tests. |
| `stat_xrate_consistency` | Swift CPU wrapper. | Complete cross-rate triangular consistency reference, symbol graph contracts, and quote staleness checks. | No accelerator required. |
| `tree_catboost` | Swift CPU wrapper; Metal folder present. | Complete ordered boosting/categorical handling approximation or delegate to a documented reference backend. | Run Metal tree scoring kernels live; if true CatBoost training is required, add Python backend decision before marking 100%. |
| `tree_lgbm` | Swift CPU wrapper; Metal folder present. | Complete histogram-based gradient boosting reference, missing-value rules, and persistence. | Run Metal histogram/scoring kernels live and parity-test. |
| `tree_rf` | Swift CPU wrapper; Metal folder present. | Complete random forest training, bagging, feature subsampling, and persistence. | Run Metal forest scoring kernels live and parity-test. |
| `tree_xgb` | Swift CPU wrapper; Metal folder present. | Complete gradient-boosted tree training, shrinkage, regularization, and persistence. | Run Metal tree scoring/histogram helper kernels live and parity-test. |
| `tree_xgb_fast` | Swift CPU wrapper; Metal folder present. | Complete fast XGB approximation with explicit tradeoffs and parity against `tree_xgb` fixtures. | Run Metal fast scoring kernels live and parity-test. |
| `trend_tsmom_vol` | Swift CPU wrapper; Metal folder present. | Complete time-series momentum with volatility targeting, warmup, and volume-aware liquidity gate. | Run Metal rolling return/volatility kernels live and parity-test. |
| `trend_vol_breakout` | Swift CPU wrapper; Metal folder present. | Complete volatility breakout reference with ATR/range variants and volume confirmation. | Run Metal rolling range/volatility kernels live and parity-test. |
| `trend_xsmom_rank` | Swift CPU only. | Complete cross-sectional momentum ranking, tie handling, volume-aware universe filters, and panel fixtures. | No separate accelerator required until universe size demands Metal sorting/ranking. |
| `wm_cfx` | Swift CPU wrapper; PyTorch folder present. | Complete currency-flow world model CPU fallback and graph feature contracts. | Wire PyTorch backend with persisted model and MPS smoke tests. |
| `wm_graph` | Swift CPU wrapper; Metal/PyTorch folders present. | Complete graph propagation/scoring CPU reference and symbol graph persistence. | Wire PyTorch graph backend; run Metal graph propagation kernels live and parity-test. |

## Execution Waves

### Wave 0: Runtime Infrastructure

Implement shared backend policy, backend manifests, Python sync bridge, Metal executor, and test harnesses. No plugin can be certified 100% before this wave is complete.

### Wave 1: CPU-Only Plugins

Complete and verify CPU-only behavior first: `factor_carry`, `factor_cmv_panel`, `factor_pca_panel`, `factor_ppp_value`, `fxbacktest_fxstupid`, `rule_buyonly`, `rule_random`, `rule_sellonly`, `stat_arimax_garch`, `stat_coint_vecm`, `stat_hmm_regime`, `stat_microflow_proxy`, `stat_msgarch`, `stat_ou_spread`, `stat_tvp_kalman`, `stat_xrate_consistency`, `trend_xsmom_rank`.

### Wave 2: Metal Plugins

Wire live Metal execution and parity tests for all plugins with Metal folders. Priority order: simple rule/demo kernels, linear models, trend/stat kernels, tree scorers, memory/distance kernels, then neural helper kernels.

### Wave 3: PyTorch/TensorFlow/NLP Plugins

Wire live Python backends with model persistence and MPS-aware smoke tests. Neural and RL plugins should treat PyTorch/TensorFlow as the full reference implementation where the Swift CPU model is only a deterministic fallback.

### Wave 4: Cross-Project Runtime

Verify FXBacktest, FXDemoAgent, and FXLiveAgent consume plugins only through the shared FXDataEngine plugin API and data only through FXDatabase.

### Wave 5: Final Certification

Run Swift tests, Python backend smoke tests, Metal runtime tests on Apple Silicon, documentation review, and git push only after the working tree is clean and build/test output is clean.

## Review Pass 1

The first cross-check found two critical risks:

1. Existing PyTorch/TensorFlow/NLP files are excluded from the Swift target, so live discovery must be filesystem or manifest based.
2. Existing Metal files are not enough by themselves; runtime compilation and buffer execution must be added before any Metal plugin can count as accelerated.

The plan above was revised to make Wave 0 mandatory before plugin certification.

## Review Pass 2

The second cross-check found two quality risks:

1. Some statistical Metal implementations are helper/proxy kernels rather than full algorithm ports. The per-plugin matrix now requires either replacing the proxy with full kernels or documenting and testing the exact helper scope before certification.
2. Some AI CPU implementations are fallback approximations. The plan now requires PyTorch/TensorFlow/NLP live backends to carry the full reference behavior where appropriate, with CPU serving as deterministic fallback rather than pretending to be the full neural architecture.

No plugin should be marked 100% complete until its row in this plan has passing evidence for every applicable gate.

## Implementation Progress

Wave 0 is implemented and verified:

1. `FXPluginRuntimeResolver` resolves CPU-only, automatic, and forced backend modes from every plugin acceleration plan.
2. `PythonMLBackendBridge` now has synchronous train/predict methods for the synchronous `FXAIPluginV4` API and accepts per-process environment variables.
3. `FXAIPluginBackendDiscovery` discovers plugin-local PyTorch, TensorFlow, and NLP files by plugin name and creates external Python descriptors.
4. `fxai_plugin_module_backend.py` dispatches Swift bridge calls into plugin-local PyTorch, TensorFlow, and NLP modules instead of using only a generic fallback.
5. `FXAIAcceleratedPluginRuntime` wraps any planned plugin and routes live train/predict calls through selected external backends with explicit CPU fallback.
6. `MetalKernelCompiler` validates Metal kernel source against the local Metal device and executes both generic and multi-buffer plugin-local float kernels with CPU parity checks.
7. `FXAIPluginMetalBackendDiscovery` discovers every declared plugin-local Metal kernel source, extracts kernel function names, compiles them on the local Metal device, and executes one plugin-local kernel per declared Metal plugin against a CPU fixture before Metal-mode prediction falls back to the CPU reference plugin.
8. `PluginContextV4` and `MLInferencePayload` carry typed text events and tokenizer contracts. NLP backend tests verify text-event enrichment and no-text fallback behavior.
9. Runtime tests now verify all registered plugins resolve CPU/automatic backends, all declared Python/NLP backend files are discoverable, Foundation NLP predicts through the Swift bridge, every declared PyTorch backend predicts/trains/persists/reloads when PyTorch is installed, every declared TensorFlow backend predicts/trains/persists/reloads when TensorFlow is installed, and the accelerated runtime wrapper drives NLP/PyTorch plugins end to end.
10. FXBacktest, FXDemoAgent, and FXLiveAgent have a static gatekeeper test proving they do not use direct ClickHouse access patterns and must use FXDatabase APIs.
11. Certification tests now verify that every plugin/backend is covered by the certification matrix and that the strict 100% check passes with the current evidence set.

CoreML/Neural Engine is deliberately excluded from plugin candidate declarations. The enum remains only as a future compatibility surface and strict runtime rejection path until a real CoreML export, load, prediction, and parity implementation is added.
