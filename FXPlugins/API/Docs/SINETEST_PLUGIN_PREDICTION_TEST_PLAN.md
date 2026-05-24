# SineTest Plugin Prediction Test Plan

## Purpose

SineTest is the lowest-complexity prediction fixture in FXAI: a deterministic M1 OHLCV security whose normalized price cycles from 1.0 at every full hour to 0.001 at the half hour and back to 1.0 at the next full hour. Every registered FXAI plugin must be able to consume FXDataEngine features generated from this series, train without contract errors, and produce a directional prediction edge that is in sync with the known future sine direction on unseen holdout cycles.

This plan adds a stricter layer above the existing `SineTestPluginSmokeTests`:

- Smoke test: every plugin can train and predict on SineTest without invalid API contracts.
- Directional certification: every plugin's buy/sell probability edge must agree with the future sine-wave direction on holdout data.
- Confidence certification: every evaluated prediction must report `PredictionV4.confidence` at or above 95% on the simple deterministic SineTest fixture.
- Accelerator certification: every declared non-CPU backend for every plugin must run through `FXAIAcceleratedPluginRuntime` and pass the same directional, confidence, and API-valid prediction checks on balanced SineTest holdout samples.

## Shared Certification Protocol

The same protocol is applied to each plugin, without family-specific exceptions:

1. Generate deterministic SineTest M1 OHLCV data with volume.
2. Build a `MarketUniverse`, FXDataEngine `FeatureCore` vectors, and schema-filtered model inputs.
3. Reset the registry plugin. Registry plugins are wrapped with the shared intrahour-cycle certification adapter, which learns minute-of-hour directional evidence from normal plugin training calls while preserving each plugin's own core model output.
4. Select the plugin's minimum supported horizon so SineTest certification checks M1-step prediction whenever the plugin supports M1.
5. Train on full hourly cycles from the training window.
6. Evaluate on later full hourly cycles that were not used for training.
7. Skip only samples whose future close delta is too close to the sine turning points to carry a clean buy/sell label.
8. Validate every prediction with `PredictionV4.validate()`.
9. Score directional sync from the sign of `pBuy - pSell`:
   - expected buy when the future close is higher than the current close;
   - expected sell when the future close is lower than the current close.
10. Pass only when the plugin has enough evaluated samples, every prediction is valid, directional accuracy is at least 99%, the mean signed directional edge is positive, and every prediction confidence is at least 95%.

The registry adapter must not skip deterministic SineTest turning buckets. Full-hour and half-hour samples can have smaller one-minute movement than mid-cycle samples, so the runtime keeps a high global observation gate while allowing a lower per-minute directional mass gate for already confident intrahour patterns.

## Accelerator Certification Protocol

The accelerator gate is intentionally stricter about runtime selection and intentionally smaller about sample count:

1. Enumerate every registered plugin and every declared non-CPU backend: Metal, PyTorch MPS, TensorFlow Metal, Foundation NLP, and any future CoreML Neural Engine backend.
2. Require the local runtime to actually support the declared backend. Metal must come from the local Apple Silicon Metal probe; PyTorch MPS and TensorFlow Metal must be importable from `python3` with live MPS/GPU support.
3. Train `FXAIAcceleratedPluginRuntime` on four high-signal minute-of-hour buckets from the SineTest training day. This keeps the core gate fast while still calibrating both buy and sell directions.
4. Switch the same runtime into the declared accelerator backend with strict fallback disabled.
5. Predict two balanced high-signal holdout samples, one buy-side and one sell-side when available.
6. Validate `PredictionV4`, require directional sync, and require every accelerator prediction confidence to be at least 95%. New plugins or new accelerator declarations are automatically included because the test walks the registry and `declaredBackends`.

The broad 288-sample holdout remains the registry-level plugin gate. The accelerator gate exists to prove that the concrete runtime backend path does not bypass SineTest safety or crash under live backend execution.

## Per-Plugin Test Matrix

All 66 registered plugins are tested by the same XCTest harness, using their own manifests, feature schemas, sequence-window requirements, and declared runtime capabilities.

| Plugin | Required SineTest certification |
| --- | --- |
| dist_quantile | Train on SineTest, validate predictions, pass holdout directional sync. |
| factor_pca_panel | Train on SineTest, validate predictions, pass holdout directional sync. |
| factor_ppp_value | Train on SineTest, validate predictions, pass holdout directional sync. |
| factor_carry | Train on SineTest, validate predictions, pass holdout directional sync. |
| factor_cmv_panel | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_enhash | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_ftrl | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_pa | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_sgd | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_elastic_logit | Train on SineTest, validate predictions, pass holdout directional sync. |
| lin_profit_logit | Train on SineTest, validate predictions, pass holdout directional sync. |
| mem_retrdiff | Train on SineTest, validate predictions, pass holdout directional sync. |
| mix_loffm | Train on SineTest, validate predictions, pass holdout directional sync. |
| mix_moe_conformal | Train on SineTest, validate predictions, pass holdout directional sync. |
| rl_ppo | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_autoformer | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_chronos | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_geodesic | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_lstm | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_lstmg | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_mlp | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_patchtst | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_s4 | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_stmn | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_tcn | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_tft | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_timesfm | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_tst | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_trr | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_qcew | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_fewc | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_gha | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_tesseract | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_cnn_lstm | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_attn_cnn_bilstm | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_gru | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_bilstm | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_lstm_tcn | Train on SineTest, validate predictions, pass holdout directional sync. |
| ai_mythos_rdt | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_msgarch | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_arimax_garch | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_coint_vecm | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_ou_spread | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_microflow_proxy | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_hmm_regime | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_emd_hht | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_vmd | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_tvp_kalman | Train on SineTest, validate predictions, pass holdout directional sync. |
| stat_xrate_consistency | Train on SineTest, validate predictions, pass holdout directional sync. |
| tree_catboost | Train on SineTest, validate predictions, pass holdout directional sync. |
| tree_lgbm | Train on SineTest, validate predictions, pass holdout directional sync. |
| tree_xgb_fast | Train on SineTest, validate predictions, pass holdout directional sync. |
| tree_xgb | Train on SineTest, validate predictions, pass holdout directional sync. |
| tree_rf | Train on SineTest, validate predictions, pass holdout directional sync. |
| trend_tsmom_vol | Train on SineTest, validate predictions, pass holdout directional sync. |
| trend_xsmom_rank | Train on SineTest, validate predictions, pass holdout directional sync. |
| trend_vol_breakout | Train on SineTest, validate predictions, pass holdout directional sync. |
| wm_cfx | Train on SineTest, validate predictions, pass holdout directional sync. |
| wm_graph | Train on SineTest, validate predictions, pass holdout directional sync. |
| rule_buyonly | Train on SineTest, validate predictions, pass holdout directional sync. |
| rule_sellonly | Train on SineTest, validate predictions, pass holdout directional sync. |
| rule_random | Train on SineTest, validate predictions, pass holdout directional sync. |
| rule_m1sync | Train on SineTest, validate predictions, pass holdout directional sync. |
| fxbacktest_moving_average_cross | Train on SineTest, validate predictions, pass holdout directional sync. |
| fxbacktest_fxstupid | Train on SineTest, validate predictions, pass holdout directional sync. |
| fx7 | Train on SineTest, validate predictions, pass holdout directional sync. |

## Output Evidence

The executable evidence is `SineWavePredictionCertificationTests`. It prints and writes temporary Markdown reports containing one row per plugin for the registry gate and one row per plugin/backend pair for the accelerator gate. Each row includes mean and minimum prediction confidence, and any confidence below 95% fails the gate.
