# FXPlugins Reference Implementation Audit

Date: 2026-05-23

This audit checks whether each plugin is a full implementation of its core
concept compared with the normal reference implementation for that model family,
adapted to FXDataEngine contracts. Passing tests and having CPU/Metal/Python
folders is not enough for a passing grade here. A plugin must implement the
actual algorithmic core, not only a generic signal adapter with the right name.

## Verdict Key

- `A`: Reference-grade or close enough for the plugin's stated simple concept.
- `B`: Functional and useful FXAI implementation, but missing material pieces of
  the standard reference algorithm.
- `C`: Not reference-grade. It is currently a heuristic, proxy, or generic
  architecture-flavored adapter and needs a real implementation pass.

## High-Level Findings

- The plugin zoo is not made of empty shells: every registered plugin has a
  callable Swift path and the current tests exercise predict/train behavior.
- The linear, quantile, memory, mixture, and several tree plugins are the
  strongest implementations. They contain real online state, calibration, replay,
  split/search, or distribution logic.
- The weakest area is the AI sequence/world/RL group. Most of those Swift CPU
  models share the same 580-line generic architecture-switch template, and the
  PyTorch/TensorFlow variants are small dense encoders. They do not contain
  standard layers such as LSTM/GRU cells, temporal convolutions, transformer
  blocks, attention stacks, S4 state-space kernels, TFT variable selection, PPO
  clipped policy optimization, or pretrained Chronos/TimesFM style model
  wrappers.
- Several statistical plugins are explicitly proxy-level implementations:
  `stat_emd_hht`, `stat_vmd`, `stat_coint_vecm`, and parts of
  `stat_arimax_garch`/`stat_msgarch` need full reference math before they should
  be called production-grade statistical implementations.
- The newest Metal additions are useful batch kernels, but they are not a
  substitute for full reference algorithms in plugins whose CPU implementation is
  still proxy-level.

## Per-Plugin Audit

| Plugin | Verdict | Current implementation | Missing for reference-grade |
|---|---:|---|---|
| `rule_buyonly` | A | Constant buy baseline with validation and native prediction distribution. | None beyond keeping it deliberately simple. |
| `rule_sellonly` | A | Constant sell baseline with validation and native prediction distribution. | None beyond keeping it deliberately simple. |
| `rule_random` | A | Deterministic seeded random baseline; suitable as a rule baseline. | None unless true stochastic simulation modes are required. |
| `rule_m1sync` | A- | Real M1 OHLCV chain-sync rule with volume confirmation and Metal batch scanner. | Add direct FXDatabase streaming integration tests and multi-symbol batch parity tests. |
| `fxbacktest_moving_average_cross` | A- | Real MA-cross signal adapter plus Metal parameter-sweep kernel. | Add full backtest-grid execution wrapper and parity tests against FXBacktest strategy outputs. |
| `fxbacktest_fxstupid` | A | Demo/stateful baseline adapter. It is simple by design. | None unless it is promoted from demo to production strategy. |
| `fx7` | A- | Backtest-native FX7 source now lives in `FXPlugins/fx7`, with a volume-aware FXDataEngine adapter and Metal signal scorer. | Add broader CPU/Metal parity fixtures over full market-universe backtest batches. |
| `lin_sgd` | A- | Online multiclass SGD/logit model with Adam-style moments, hashed interactions, move head, drift guard, and calibration. | Add golden parity tests against a canonical online logistic implementation and larger hash-collision stress tests. |
| `lin_ftrl` | A- | FTRL-proximal multiclass learner with dual hashed interactions, adaptive parameters, calibration, and move head. | Add sparse-feature parity tests against a known FTRL implementation and stronger L1/L2 regularization coverage. |
| `lin_enhash` | B+ | Dual field/hash online learner with FTRL-like behavior and Metal pair-product support. | Formalize ENHash reference spec, verify collision handling, and add golden replay fixtures from legacy MQL5. |
| `lin_pa` | A- | Crammer-Singer passive-aggressive learner with replay, averaged weights, hashed interactions, and calibration. | Add PA-I/PA-II mode tests and canonical multiclass PA parity fixtures. |
| `lin_elastic_logit` | B | Functional elastic-logit style online classifier. | Add true elastic-net objective with explicit L1 proximal shrinkage, coefficient path diagnostics, and reference parity. |
| `lin_profit_logit` | B | Profit-aware logistic branch with online heads. | Add explicit profit-weighted loss, asymmetric cost model, and tests proving profit objective differs from plain logit. |
| `dist_quantile` | A- | Strong native quantile-head model with monotone quantiles, class head, calibration, PIT diagnostics, and Metal scoring kernels. | Add pinball-loss parity tests and stronger quantile crossing/coverage evaluation on held-out replay data. |
| `mem_retrdiff` | B+ | Real memory-bank retrieval/difference model with distance voting and Metal distance kernels. | Add approximate top-k/index acceleration, memory eviction policy tests, and deterministic nearest-neighbor parity fixtures. |
| `mix_loffm` | B | Functional latent online factorization/gating model with PyTorch batch helper. | Needs a formal LOFFM reference spec, factor update derivation, persistence parity, and expert/gating ablation tests. |
| `mix_moe_conformal` | B | Functional MoE router with conformal score rings and calibration buckets. | Needs full conformal prediction contract: coverage guarantees, split/calibration set handling, expert specialization tests, and interval validity metrics. |
| `rl_ppo` | C | Generic dense policy/value adapter using the shared AI architecture template; PyTorch file is supervised cross-entropy plus move loss. | Implement actual PPO: rollout buffer, advantage estimation, clipped policy loss, entropy/value losses, action log-probs, episode accounting, and offline market environment integration. |
| `tree_xgb_fast` | B+ | Real gradient/hessian tree machinery, multiclass trees, refresh, move stats, and Metal inference kernels. | Add canonical XGBoost gain/regularization parity, column subsampling, missing-value routing tests, and objective fixtures. |
| `tree_xgb` | B | Binary XGBoost-style online trees with leaf mass blending and Metal inference. | Upgrade to proper multiclass gradient boosting, shrinkage schedules, column/row sampling, missing-value learning, and reference gain tests. |
| `tree_lgbm` | B+ | Histogram/GOSS-style tree model with multiclass leaves, DART-compatible concepts, quantiles, and Metal scoring. | Add LightGBM parity for histogram binning, leaf-wise growth constraints, categorical handling, DART dropout behavior, and validation-based early stopping. |
| `tree_catboost` | B+ | Symmetric trees with ordered CTR-style features, multiclass leaf Newton updates, and Metal scoring. | Add strict ordered boosting permutation logic, categorical/CTR leakage tests, CatBoost-style priors, and reference fixtures. |
| `tree_rf` | B- | Fixed-size seeded random-forest-like leaf mass model with Metal scoring. | Add true bagging, bootstrap sampling, random feature subsets, split optimization, OOB metrics, and forest rebuild policy. |
| `factor_carry` | B | Carry-style factor formula with online move/quality state. | Needs actual interest-rate/forward-point carry inputs from FXDatabase and cross-currency carry normalization. |
| `factor_cmv_panel` | B | Functional panel factor heuristic. | Define CMV reference formula, add real panel data contract, cross-sectional normalization, and factor exposure tests. |
| `factor_pca_panel` | C | Online PCA-like state, but loadings are not a full orthonormal/eigen/SVD PCA implementation. | Implement covariance accumulation, eigensolver/SVD via Accelerate, component orthogonalization, explained variance, and panel parity tests. |
| `factor_ppp_value` | B | PPP/value-style factor formula with online quality state. | Needs real PPP/fair-value data inputs, valuation z-score normalization, and cross-currency panel tests. |
| `trend_tsmom_vol` | B | Time-series momentum/volatility window scanner with volume and Metal batch scoring. | Add canonical TSMOM volatility targeting, lookback/horizon grid, risk scaling, transaction-cost-aware signals, and portfolio-level tests. |
| `trend_xsmom_rank` | C | Functional trend/rank adapter, but not a full cross-sectional ranking implementation. | Needs multi-symbol panel input, rank normalization, ties/missing handling, neutralization, and cross-sectional portfolio construction tests. |
| `trend_vol_breakout` | B | Volatility breakout window scanner with range expansion and Metal batch scoring. | Add canonical breakout bands, ATR/realized-vol filters, stop/target semantics, parameter-grid backtest parity, and regime filters. |
| `stat_arimax_garch` | C | ARIMAX/GARCH-inspired online recursion. | Needs actual ARIMAX design matrix, AR/MA terms, exogenous regressors, GARCH likelihood/estimation, residual diagnostics, and forecast intervals. |
| `stat_coint_vecm` | C | Fixed residual formula plus OU-style state. | Needs Johansen cointegration rank test, estimated beta vectors, VECM alpha/gamma matrices, lag selection, residual diagnostics, and multi-series input contract. |
| `stat_emd_hht` | C | Explicit EMD/HHT proxy using recent deltas and mean shift; Metal kernel is also a proxy. | Implement empirical mode decomposition, sifting, Hilbert transform/instantaneous frequency, IMF selection, and reference signal tests. |
| `stat_hmm_regime` | B | Real small HMM forward filter with online transition/emission updates. | Add Baum-Welch/EM fitting, log-space forward/backward, multi-feature Gaussian emissions, state decoding, and reference likelihood tests. |
| `stat_microflow_proxy` | B | OHLCV microflow proxy with volume-aware features. | If meant as true microstructure, add order-flow/tick imbalance inputs; otherwise keep name as proxy and add contract tests for OHLCV-only semantics. |
| `stat_msgarch` | C+ | HMM/GARCH-inspired state-space model. | Needs actual Markov-switching GARCH likelihood, regime-dependent variance recursion, filtered/smoothed probabilities, and statistical diagnostics. |
| `stat_ou_spread` | B | OU residual model with online mean-reversion state. | Add maximum-likelihood OU parameter estimation, half-life diagnostics, residual stationarity tests, and multi-symbol residual source contract. |
| `stat_tvp_kalman` | B- | Diagonal recursive Kalman-style update with OU residual state. | Implement full state-space matrices, covariance propagation/update, measurement noise estimation, missing data handling, and Kalman parity fixtures. |
| `stat_vmd` | C | VMD proxy based on EMA mode components; Metal kernel is a batch proxy. | Implement variational mode decomposition with ADMM iterations, mode bandwidth constraints, convergence checks, and reference decomposition tests. |
| `stat_xrate_consistency` | B | Cross-rate consistency graph-style checks over FX features. | Add explicit triangular arbitrage graph, symbol graph inputs, normalization across quote currencies, and cycle-consistency tests. |
| `ai_mlp` | B+ | Real small MLP-style online model and PyTorch helper. | Add deeper configurable MLP, normalization, optimizer parity, checkpointing, and PyTorch/Swift output parity tests. |
| `ai_lstm` | C | Generic recurrent-flavored Swift model; PyTorch/TF variants are dense encoders, not LSTM cells. | Implement actual LSTM layers/cell state, sequence batching, truncated BPTT, checkpointing, and parity tests. |
| `ai_lstmg` | C | Generic gated-recurrent-flavored model. | Implement the actual LSTM-gated architecture, gate equations, sequence state, training loop, and PyTorch/TF parity. |
| `ai_gru` | C | Generic GRU-flavored model without GRU cells. | Implement GRU reset/update gates, recurrent state over sequence windows, training, checkpointing, and parity tests. |
| `ai_bilstm` | C | Generic bidirectional signal adapter, not a bidirectional LSTM. | Implement forward/backward LSTM passes, sequence packing, training, and output merge semantics. |
| `ai_lstm_tcn` | C | Generic LSTM/TCN-flavored adapter. | Implement real LSTM encoder plus dilated causal TCN stack, receptive-field tests, and framework parity. |
| `ai_cnn_lstm` | C | Generic convolutional/recurrent adapter. | Implement Conv1D feature extractor plus LSTM sequence model, pooling, training, and TensorFlow/PyTorch parity. |
| `ai_attn_cnn_bilstm` | C | Generic attention/CNN/BiLSTM-flavored adapter. | Implement convolutional front end, bidirectional LSTM, attention pooling, and true sequence training. |
| `ai_tcn` | C | Generic TCN-flavored adapter. | Implement dilated causal residual TCN blocks, receptive-field validation, and framework parity. |
| `ai_tst` | C | Generic transformer-flavored adapter. | Implement time-series transformer encoder with positional encoding, multi-head attention, feed-forward blocks, and masking. |
| `ai_tft` | C | Generic TFT-flavored adapter. | Implement TFT variable selection networks, gated residual networks, static covariate encoders, LSTM encoder/decoder, interpretable attention, and quantile heads. |
| `ai_autoformer` | C | Generic transformer-flavored adapter. | Implement decomposition blocks, auto-correlation attention, seasonal/trend paths, and forecasting heads. |
| `ai_patchtst` | C | Generic patch-transformer adapter. | Implement patch extraction, channel-independent transformer encoder, patch positional encoding, and horizon head. |
| `ai_s4` | C | Generic state-space-flavored adapter. | Implement S4/S4D state-space kernel, convolution/recurrent forms, parameterization, and stability tests. |
| `ai_stmn` | C | Generic memory-network-flavored adapter. | Define and implement actual spatio-temporal memory network: memory slots, read/write/update policy, and sequence training. |
| `ai_chronos` | C | Small numeric/NLP helper; no pretrained Chronos-style tokenizer/model wrapper. | Integrate real Chronos-style time-series tokenization/model inference or rename the plugin to a non-Chronos concept. |
| `ai_timesfm` | C | Small generic foundation-forecaster adapter and NLP helper. | Integrate real TimesFM-style model wrapper or implement equivalent patch/horizon foundation model with checkpoint support. |
| `ai_fewc` | C | Generic FEWC-flavored architecture switch. | Define the FEWC reference algorithm, implement its distinctive training/inference path, and add golden fixtures. |
| `ai_geodesic` | C | Generic geodesic-attention signal adapter. | Implement real geodesic distance/attention over feature manifolds or graph paths, with metric tests. |
| `ai_gha` | C | Generic GHA-flavored adapter. | Define whether GHA means generalized Hebbian algorithm or another legacy model; implement the actual update equations. |
| `ai_qcew` | C | Generic QCEW-flavored adapter. | Implement actual quantile/cross-entropy window objective, distribution head, and calibration/coverage tests. |
| `ai_tesseract` | C | Generic tensor/tesseract-flavored adapter. | Implement true tensor factorization or tensor-network model, contraction kernels, rank controls, and parity tests. |
| `ai_trr` | C | Generic trend/reversal recurrent adapter. | Implement explicit trend/reversal regime model or recurrent architecture with transition logic and labelled replay tests. |
| `ai_mythos_rdt` | C | OpenMythos-inspired adapter, but not a real recursive decision transformer. | Implement recursive decision transformer blocks, trajectory/return conditioning, memory recursion, PyTorch training loop, and checkpointing. |
| `wm_cfx` | C | Generic currencyFactorWorld architecture switch with dense PyTorch helper. | Implement real currency-factor world model: currency graph/factors, exposure matrix, cross-rate consistency, and factor evolution. |
| `wm_graph` | C | Generic graphWorld adapter plus simple dense PyTorch helper and Metal projection. | Implement real graph model: nodes/edges, message passing or graph neural net, adjacency from FX universe, and graph consistency losses. |

## Required Upgrade Policy

1. Do not mark a plugin as complete just because it has a CPU model and tests.
   Completion must mean the algorithmic core matches the plugin name.
2. For every `C` plugin, either implement the reference model or rename the
   plugin to honestly describe the current proxy behavior. Keeping the current
   name with proxy code is misleading.
3. For every PyTorch/TensorFlow accelerator, add at least one test or script that
   proves the framework backend contains the expected reference-layer family.
   Examples: LSTM uses LSTM cells, GRU uses GRU cells, TFT has attention and
   variable selection, PPO has clipped policy loss.
4. For every Metal accelerator, define whether it is a full implementation or
   only a batch scoring helper. Batch scoring helpers are valuable, but should not
   be counted as reference implementations for decomposition, transformer, or RL
   plugins.
5. Add golden fixtures per family. The minimum useful fixture is a tiny deterministic
   dataset with expected intermediate states, not only final class probabilities.

## Recommended Implementation Waves

### Wave 1: Rename Or Replace Proxy Statistical Plugins

Target: `stat_emd_hht`, `stat_vmd`, `stat_coint_vecm`,
`stat_arimax_garch`, `stat_msgarch`, `factor_pca_panel`,
`trend_xsmom_rank`.

Deliverable: true decomposition/statistical implementations with reference
fixtures, or explicit proxy names if the current lightweight semantics are kept.

### Wave 2: Real Neural Sequence Backends

Target: `ai_lstm`, `ai_gru`, `ai_bilstm`, `ai_lstm_tcn`,
`ai_cnn_lstm`, `ai_attn_cnn_bilstm`, `ai_tcn`, `ai_tst`, `ai_tft`,
`ai_autoformer`, `ai_patchtst`, `ai_s4`.

Deliverable: PyTorch-first reference implementations with MPS support, checkpoint
state, training/inference scripts, and Swift bridge contracts. Swift CPU can stay
as deterministic fallback, but it must be labelled fallback, not the reference
model.

### Wave 3: Foundation And Novel AI Plugins

Target: `ai_chronos`, `ai_timesfm`, `ai_mythos_rdt`, `ai_fewc`,
`ai_geodesic`, `ai_gha`, `ai_qcew`, `ai_tesseract`, `ai_trr`,
`ai_stmn`.

Deliverable: either integrate the real published model family or write a local
reference spec first, then implement exactly that spec.

### Wave 4: Tree/Linear Hardening

Target: all `lin_*`, `tree_*`, `dist_quantile`, `mem_retrdiff`,
`mix_*`.

Deliverable: keep current code, add golden reference fixtures, stress tests, and
parity tests. These are closest to production and mostly need validation depth,
not a full rewrite.
