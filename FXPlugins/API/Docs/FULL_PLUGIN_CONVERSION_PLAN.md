# Full MQL5 Plugin Zoo Conversion Plan

Date: 2026-05-23
Branch: `main`
Legacy source scanned from git commit: `f6ba0e7^`
Legacy scan root: `/tmp/fxai_legacy_mql5_scan/FXAI/FXPlugins`
Current Swift plugin root: `FXPlugins`

## Goal

Replace every current Swift wrapper/shell with a real plugin-local implementation.
Each plugin folder owns its own CPU implementation and every suitable accelerator
implementation. `FXPlugins/API` remains the only shared plugin access surface:
registry, docs, tests, backend process hooks, and contract tests.

Required target shape:

```text
FXPlugins/<plugin_id>/
  <PluginName>Plugin.swift          # public FXAIPluginV4 adapter and CPU selection
  CPU/                              # mandatory full Swift CPU implementation
  Metal/                            # only when Metal is required or useful
  PyTorch/                          # only when PyTorch MPS is required or useful
  TensorFlow/                       # only when TensorFlow Metal is required or useful
  NLP/                              # only when text/event/language features are required
  Tests/                            # plugin-local fixtures/parity notes when needed
```

Every plugin must expose:

- CPU-only deterministic implementation.
- Manifest and acceleration plan.
- Volume-aware feature handling when `dataHasVolume == true`.
- M1 OHLCV data contract compatibility; no spread/tick dependency.
- Parity tests against recovered MQL5 behavior where deterministic comparison is
  possible.
- Accelerator tests that at least validate request/response contracts; hardware
  tests can skip when the runtime is unavailable.

## Scan Result

Recovered legacy files:

- 115 legacy `.mqh` files under `FXAI/FXPlugins`.
- 64 top-level legacy plugin/common files:
  - 63 old MQL5 plugins.
  - 1 shared old `Common/fxai_framework_model.mqh`.
- 2 Swift-era FXBacktest demo plugins are part of the current 65-plugin package.

Current state:

- 65 root plugin folders exist under `FXPlugins`.
- 59 plugins now have plugin-local `CPU/` implementations plus accelerator
  folders where the matrix requires them.
- 4 rule plugins and the 2 FXBacktest demos remain scalar Swift adapters with
  plugin-specific CPU logic.
- 0 plugins delegate to `FXAIReferencePluginRuntime`.
- The former Swift wrapper layer has been removed from the plugin zoo.

## Execution Status

Completed on 2026-05-23:

- All 26 remaining sequence, world-model, and RL wrappers were converted one by
  one, with focused Swift tests passing before each commit.
- Plugin-local accelerator folders now cover Metal, PyTorch/MPS,
  TensorFlow/Metal, and NLP where required by the matrix.
- `rg "FXAIReferencePluginRuntime|FXAIPluginImplementationDescriptor" FXPlugins`
  returns no matches.

## Conversion Rules

1. No plugin may remain a wrapper around `FXAIReferencePluginRuntime`.
2. A plugin may import `FXDataEngine` contracts and shared API types.
3. A plugin may not depend on another plugin folder.
4. If a legacy plugin inherited `CFXAIFrameworkModelPlugin`, port the relevant
   framework branch into that plugin's own `CPU/` implementation. Do not keep one
   shared runtime switch as the plugin body.
5. Metal implementations are for deterministic batched math, rolling windows,
   tree scoring, histogram scans, top-k distance search, and large parameter
   sweeps.
6. PyTorch implementations are for research-grade neural training/inference,
   RL, transformers, graph/world models, and sequence models where MPS gives a
   strong path.
7. TensorFlow implementations are required for LSTM/GRU/CNN-LSTM variants where
   TensorFlow/Keras parity is the most direct model implementation.
8. NLP implementations are only required when plugin logic consumes macro/event
   text, calendar/news payloads, or semantic context. They must be optional and
   fall back to CPU numeric features.

## Accelerator Folder Semantics

- `CPU/`: always required. Pure Swift, no Python dependency.
- `Metal/`: Swift/Metal shaders or `.metal` kernels plus Swift dispatcher.
- `PyTorch/`: Python module using PyTorch with MPS when available, CPU fallback.
- `TensorFlow/`: Python module using TensorFlow with `tensorflow-metal` when
  available, CPU fallback.
- `NLP/`: local feature extraction for event/news/context text. May use Python
  NLP libraries behind the API hook, but numeric CPU fallback remains mandatory.

## Implementation Waves

### Wave 0: Shared Variant Contract

Before converting individual plugins, add a small local API that lets a plugin
select a technical variant without exposing implementation internals:

- `PluginVariantKind`: `cpu`, `metal`, `pyTorch`, `tensorFlow`, `nlp`.
- `PluginVariantDescriptor`: folder, availability, training support, inference
  support, required runtime.
- `PluginVariantRouter`: deterministic default selection with CPU fallback.
- Tests: every plugin reports CPU; accelerator descriptors match folders.

### Wave 1: Low-Risk Rule And Demo Plugins

Finish proper plugin-local structure for:

- `rule_buyonly`
- `rule_sellonly`
- `rule_random`
- `rule_m1sync`
- `fxbacktest_moving_average_cross`
- `fxbacktest_fxstupid`

These require CPU only except:

- `rule_m1sync`: implemented Metal batched chain scanner.
- `fxbacktest_moving_average_cross`: implemented Metal batched parameter sweep.

### Wave 2: Native Linear And Distribution Plugins

Port the full MQL5 state/update/calibration code into plugin-local Swift:

- `lin_sgd`
- `lin_ftrl`
- `lin_enhash`
- `lin_pa`
- `lin_elastic_logit`
- `lin_profit_logit`
- `dist_quantile`

Required accelerators:

- Swift SIMD/Accelerate in CPU implementation for vector math.
- Metal for batch inference or large interaction scoring where profitable.

### Wave 3: Trees, Memory, Mixtures, Factors, Trend, Stats, World

Port full MQL5 algorithms into plugin-local CPU code, then add accelerators:

- Tree plugins: Metal histogram/split scans and batched scoring.
- Memory plugin: Metal or Accelerate top-k distance search.
- Mixture plugins: Accelerate CPU, optional PyTorch gating for MoE.
- Factor/trend/stat plugins: Accelerate first; Metal where batched reductions
  dominate.
- World models: PyTorch/Metal for graph/world state where useful.

### Wave 4: Sequence, Transformer, RL, And NLP

Port MQL5 sequence state machines into CPU Swift first. Add ML accelerators:

- PyTorch MPS for transformer/state-space/RL/world models.
- TensorFlow Metal for LSTM/GRU/CNN-LSTM family.
- Metal for convolution/state-space kernels and batched inference.
- NLP only for plugins using macro/event/context text.

## Per-Plugin Conversion Matrix

Legend:

- CPU: mandatory full Swift plugin-local implementation.
- Metal: required accelerator implementation.
- PyTorch: required Python/PyTorch MPS implementation.
- TensorFlow: required Python/TensorFlow Metal implementation.
- NLP: required semantic/text/event feature implementation.
- Optional means create descriptor and folder only after the CPU code proves a
  useful batch path; do not block CPU parity.

| Plugin | Legacy source | Legacy algorithm scanned | Required implementations |
|---|---|---|---|
| `rule_buyonly` | `Rule/rule_buyonly.mqh` | Constant buy baseline. | CPU. |
| `rule_sellonly` | `Rule/rule_sellonly.mqh` | Constant sell baseline. | CPU. |
| `rule_random` | `Rule/rule_random.mqh` | Deterministic seeded no-skip random. | CPU. |
| `rule_m1sync` | `Rule/rule_m1sync.mqh` | M1 OHLC chain sync with volume confirmation. | CPU, Metal batch scanner. |
| `fxbacktest_moving_average_cross` | Swift demo | Moving-average cross signal adapter. | CPU, Metal parameter sweep. |
| `fxbacktest_fxstupid` | Swift demo | Stateful demo direction adapter. | CPU. |
| `lin_sgd` | `Linear/lin_sgd.mqh` | SGD multiclass model, hashed interactions, Adam moments, calibration, move head. | CPU, Metal batch interactions. |
| `lin_ftrl` | `Linear/lin_ftrl.mqh` | Native 3-class FTRL-prox, calibration, move head. | CPU, Metal optional batch scoring. |
| `lin_enhash` | `Linear/lin_enhash.mqh` | ENHash/FTRL hashed online learner, calibration, quality heads. | CPU, Metal hash scoring. |
| `lin_pa` | `Linear/lin_pa.mqh` plus `lin_pa/*` | Crammer-Singer passive-aggressive multiclass learner. | CPU, Metal optional batch scoring. |
| `lin_elastic_logit` | `Linear/lin_elastic_logit.mqh`, old framework kind 9 | Elastic-net logistic branch of framework runtime. | CPU, Accelerate, Metal optional. |
| `lin_profit_logit` | `Linear/lin_profit_logit.mqh`, old framework kind 10 | Profit-aware logistic branch of framework runtime. | CPU, Accelerate, Metal optional. |
| `dist_quantile` | `Distribution/dist_quantile.mqh` | Native quantile heads, distribution prediction, calibration, snapshot persistence. | CPU, Accelerate, Metal quantile batches. |
| `mem_retrdiff` | `Memory/mem_retrdiff.mqh` | Retrieval/difference memory bank, distance vote, native persistence. | CPU, Metal top-k search, Accelerate. |
| `mix_loffm` | `Mixture/mix_loffm.mqh` | Latent online factorization/mixing model. | CPU, Accelerate, PyTorch optional gating. |
| `mix_moe_conformal` | `Mixture/mix_moe_conformal.mqh` | Mixture-of-experts conformal router. | CPU, Accelerate, PyTorch MoE optional. |
| `tree_xgb_fast` | `Tree/tree_xgb_fast.mqh` | Native XGB fast trees, ring buffer, split search, calibration. | CPU, Metal split/scoring. |
| `tree_xgb` | `Tree/tree_xgb.mqh` | Native XGBoost-style trees. | CPU, Metal split/scoring. |
| `tree_catboost` | `Tree/tree_catboost.mqh` plus submodules | Ordered CTR/CatBoost-style trees, calibration, training modules. | CPU, Metal scoring/split scans. |
| `tree_lgbm` | `Tree/tree_lgbm.mqh` plus submodules | LightGBM-style histograms/splits/metrics. | CPU, Metal histogram/split scans. |
| `tree_rf` | `Tree/tree_rf.mqh`, old framework kind 3 | Random forest branch of framework runtime. | CPU, Metal batched tree scoring. |
| `factor_carry` | `Factor/factor_carry.mqh`, old framework kind 18 | Carry factor branch of framework runtime. | CPU, Accelerate. |
| `factor_cmv_panel` | `Factor/factor_cmv_panel.mqh`, old framework kind 19 | CMV panel factor branch. | CPU, Accelerate, Metal optional panel batches. |
| `factor_pca_panel` | `Factor/factor_pca_panel.mqh`, old framework kind 16 | PCA panel branch. | CPU, Accelerate SVD/eigens, Metal optional. |
| `factor_ppp_value` | `Factor/factor_ppp_value.mqh`, old framework kind 17 | PPP/value factor branch. | CPU, Accelerate. |
| `trend_tsmom_vol` | `Trend/trend_tsmom_vol.mqh`, old framework kind 20 | Time-series momentum/volatility branch. | CPU, Accelerate, Metal batched window scans. |
| `trend_xsmom_rank` | `Trend/trend_xsmom_rank.mqh`, old framework kind 21 | Cross-sectional momentum/rank branch. | CPU, Accelerate. |
| `trend_vol_breakout` | `Trend/trend_vol_breakout.mqh`, old framework kind 22 | Volatility breakout branch. | CPU, Accelerate, Metal batched breakout scans. |
| `stat_msgarch` | `Stat/stat_msgarch.mqh`, old framework kind 1 | Markov-switching GARCH branch. | CPU, Accelerate, PyTorch optional fitting. |
| `stat_arimax_garch` | `Stat/stat_arimax_garch.mqh`, old framework kind 2 | ARIMAX/GARCH branch. | CPU, Accelerate, PyTorch optional fitting. |
| `stat_coint_vecm` | `Stat/stat_coint_vecm.mqh`, old framework kind 4 | Cointegration/VECM branch; spread means residual. | CPU, Accelerate. |
| `stat_ou_spread` | `Stat/stat_ou_spread.mqh`, old framework kind 5 | OU residual branch; no bid/ask spread. | CPU, Accelerate. |
| `stat_microflow_proxy` | `Stat/stat_microflow_proxy.mqh`, old framework kind 7 | OHLCV microflow proxy branch. | CPU, Accelerate, NLP optional only if news context added. |
| `stat_hmm_regime` | `Stat/stat_hmm_regime.mqh`, old framework kind 8 | HMM regime branch. | CPU, Accelerate. |
| `stat_emd_hht` | `Stat/stat_emd_hht.mqh`, old framework kind 13 | EMD/HHT state-space branch. | CPU, Accelerate, Metal decomposition batches. |
| `stat_vmd` | `Stat/stat_vmd.mqh`, old framework kind 14 | VMD state-space branch. | CPU, Accelerate, Metal decomposition batches. |
| `stat_tvp_kalman` | `Stat/stat_tvp_kalman.mqh`, old framework kind 15 | Time-varying parameter Kalman branch. | CPU, Accelerate. |
| `stat_xrate_consistency` | `Stat/stat_xrate_consistency.mqh`, old framework kind 23 | Cross-rate consistency graph branch. | CPU, Accelerate. |
| `wm_cfx` | `World/wm_cfx.mqh` | Currency-factor world model. | CPU, Accelerate, PyTorch world-model optional. |
| `wm_graph` | `World/wm_graph.mqh` | Graph world model. | CPU, Accelerate, PyTorch graph model, Metal optional graph batches. |
| `rl_ppo` | `RL/rl_ppo.mqh`, old framework kind 6 | PPO policy/value branch. | CPU, PyTorch MPS, Core ML candidate after training export. |
| `ai_lstm` | `Sequence/ai_lstm.mqh` plus `ai_lstm/*` | Native LSTM sequence state, TBPTT batch, calibration, quality heads. | CPU, TensorFlow, PyTorch, Metal optional inference. |
| `ai_lstmg` | `Sequence/ai_lstmg.mqh` plus `ai_lstmg/*` | LSTM-gated model, batch training, calibration. | CPU, PyTorch, TensorFlow, Metal optional. |
| `ai_gru` | `Sequence/ai_gru.mqh`, old framework kind 24 | GRU branch of framework runtime. | CPU, TensorFlow, PyTorch. |
| `ai_bilstm` | `Sequence/ai_bilstm.mqh`, old framework kind 25 | BiLSTM branch of framework runtime. | CPU, TensorFlow, PyTorch. |
| `ai_lstm_tcn` | `Sequence/ai_lstm_tcn.mqh`, old framework kind 26 | LSTM/TCN branch of framework runtime. | CPU, PyTorch, TensorFlow, Metal convolution. |
| `ai_cnn_lstm` | `Sequence/ai_cnn_lstm.mqh`, old framework kind 11 | CNN/LSTM branch of framework runtime. | CPU, TensorFlow, PyTorch, Metal convolution. |
| `ai_attn_cnn_bilstm` | `Sequence/ai_attn_cnn_bilstm.mqh`, old framework kind 12 | Attention CNN BiLSTM branch. | CPU, TensorFlow, PyTorch, Metal attention/convolution. |
| `ai_mlp` | `Sequence/ai_mlp.mqh` plus `ai_mlp/*` | Native MLP state/head/calibration. | CPU, Accelerate, Metal, Core ML candidate. |
| `ai_tcn` | `Sequence/ai_tcn.mqh` plus `ai_tcn/*` | Temporal convolution network. | CPU, PyTorch, TensorFlow, Metal convolution. |
| `ai_s4` | `Sequence/ai_s4.mqh` plus `ai_s4/*` | State-space S4 model and batch training. | CPU, PyTorch, Metal state-space kernels. |
| `ai_stmn` | `Sequence/ai_stmn.mqh` plus `ai_stmn/*` | Spatio-temporal memory network and TBPTT. | CPU, PyTorch, Metal optional. |
| `ai_tst` | `Sequence/ai_tst.mqh` plus `ai_tst/*` | Time-series transformer. | CPU, PyTorch, Metal attention, Core ML candidate. |
| `ai_tft` | `Sequence/ai_tft.mqh` plus `ai_tft/*` | Temporal Fusion Transformer with heads and TBPTT. | CPU, PyTorch, Metal attention, Core ML candidate. |
| `ai_autoformer` | `Sequence/ai_autoformer.mqh` plus `ai_autoformer/*` | Autoformer decomposition/attention model. | CPU, PyTorch, Metal decomposition/attention, Core ML candidate. |
| `ai_patchtst` | `Sequence/ai_patchtst.mqh` plus `ai_patchtst/*` | PatchTST patch transformer. | CPU, PyTorch, Metal patch/attention, Core ML candidate. |
| `ai_chronos` | `Sequence/ai_chronos.mqh` plus `ai_chronos/*` | Chronos-like causal token forecaster. | CPU, PyTorch, NLP optional for token semantics, Core ML candidate. |
| `ai_timesfm` | `Sequence/ai_timesfm.mqh` plus `ai_timesfm/*` | TimesFM-style forecaster. | CPU, PyTorch, NLP optional for event tokens, Core ML candidate. |
| `ai_geodesic` | `Sequence/ai_geodesic.mqh` plus `ai_geodesic/*` | Geodesic attention model with replay/TBPTT. | CPU, PyTorch, Metal attention, Core ML candidate. |
| `ai_fewc` | `Sequence/ai_fewc.mqh` | FEWC native sequence/calibration model. | CPU, PyTorch optional, Metal optional. |
| `ai_gha` | `Sequence/ai_gha.mqh` | GHA native sequence/geometric model. | CPU, Accelerate, PyTorch optional. |
| `ai_qcew` | `Sequence/ai_qcew.mqh` | Quantile cross-entropy window model. | CPU, PyTorch optional, Metal quantile batches. |
| `ai_tesseract` | `Sequence/ai_tesseract.mqh` | Tensorized heuristic/model. | CPU, Metal tensor kernels, PyTorch optional. |
| `ai_trr` | `Sequence/ai_trr.mqh` | Trend/reversal recurrent hybrid. | CPU, PyTorch optional, Accelerate. |
| `ai_mythos_rdt` | `Sequence/ai_mythos_rdt.mqh` | Mythos recursive decision transformer. | CPU, PyTorch, NLP optional for recursive prompt/context text, Core ML candidate. |

## Review Pass 1

Findings:

- The current Swift package exposes all plugin IDs, but this is contract
  coverage, not algorithmic parity.
- The old shared framework model is a real algorithm source for 26 plugins; it
  must be decomposed into plugin-local CPU implementations or a generated local
  copy per plugin, not kept as one shared runtime switch.
- `spread` references in old statistical plugins are residual/statistical spread
  unless explicitly bid/ask. New code must use `priceCostPoints` only for
  execution cost.
- Sequence plugins with old submodules require CPU Swift state ports first,
  because accelerator parity depends on knowing the exact state layout.

Revision after pass 1:

- Add Wave 0 variant contract before bulk plugin edits.
- Convert `lin_sgd` first after rules/demos because it exercises dense weights,
  hashed interactions, calibration, move head, and volume-aware features.
- Convert `tree_xgb_fast` before other tree plugins because it is the richest
  tree implementation and can define the local Metal pattern.

## Review Pass 2

Findings:

- Full conversion is too risky if all plugin files are edited in a single large
  commit without parity gates.
- Each plugin folder can still have multiple technical implementations without
  changing the public registry: the public adapter should default to CPU and
  expose accelerator descriptors through the acceleration plan.
- Python accelerator code must be plugin-local, while the process protocol can
  remain in `FXPlugins/API/Backends`.
- NLP should not be created for every sequence model by default; only Chronos,
  TimesFM, Mythos RDT, and any future news/event-aware plugin qualify now.

Revision after pass 2:

- First coding commit creates the variant folder contract and converts
  `lin_sgd` as the template for full CPU plus Metal skeleton.
- Subsequent commits proceed one plugin at a time in the matrix order, with
  tests passing before moving on.
- A plugin is not marked done until its current shell no longer references
  `FXAIReferencePluginRuntime`.

## Definition Of Done Per Plugin

For every plugin:

1. `rg "FXAIReferencePluginRuntime|FXAIPluginImplementationDescriptor" FXPlugins/<plugin_id>` returns no matches.
2. `FXPlugins/<plugin_id>/CPU` exists and contains the full Swift CPU algorithm.
3. Accelerator subfolders exist exactly as required by the matrix.
4. Public adapter uses plugin-local CPU implementation as the deterministic
   fallback.
5. Manifest validates and `requiresVolumeWhenAvailable == true`.
6. Unit tests cover predict, train, reset, self-test, volume/no-volume behavior,
   and accelerator descriptor presence.
7. `swift test --package-path FXPlugins` passes.
8. `swift test --package-path FXDataEngine` passes if any contract/helper changed.
