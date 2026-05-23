# FXPlugins 99 Percent Reference Implementation Plan

Date: 2026-05-23

Goal: move every plugin from "callable Swift conversion" to approximately 99%
of the standard reference implementation for its model family, while preserving
the FXDataEngine V4 contracts, M1 OHLCV-only data contract, volume gating, and
plugin-local code ownership.

This plan supersedes completion claims that only mean "the plugin runs." A
plugin is complete only when its algorithmic core matches the model name, its
accelerator folders contain real implementation code, and its tests prove the
important intermediate states, not only final probabilities.

## Common Acceptance Gate

Every plugin must satisfy these shared gates:

1. CPU path stays deterministic and valid for offline backtests.
2. Volume-derived features are used only when `dataHasVolume == true`.
3. Any PyTorch backend uses MPS when available and CPU otherwise.
4. Any TensorFlow backend uses Keras layers and is compatible with
   `tensorflow-metal` when installed.
5. Any Metal backend declares whether it is a full algorithm kernel or a batch
   scoring/parameter-sweep helper.
6. Tests must cover reference math, volume gating, state reset, persistence
   where relevant, and at least one golden fixture per model family.

## Per-Plugin Work Plan

| Plugin | 99% reference code enhancement | Accelerator implementation | Validation needed |
|---|---|---|---|
| `rule_buyonly` | Keep as exact constant buy reference baseline. | CPU only; no useful Metal/PyTorch/TF/NLP path. | Golden distribution and no-op train contract. |
| `rule_sellonly` | Keep as exact constant sell reference baseline. | CPU only; no useful accelerator path. | Golden distribution and no-op train contract. |
| `rule_random` | Keep deterministic seeded random baseline; document seed/hash semantics. | CPU only; optional SIMD batch simulator only if backtest batching needs it. | Seed stability and reproducible sequence tests. |
| `rule_m1sync` | Add FXDatabase M1 stream fixture, broken-chain handling, multi-symbol chain checks. | Metal batch scanner remains valid; add CPU/Metal parity fixture. | Multi-symbol chain-sync, volume confirmation, broken-chain skip. |
| `fxbacktest_moving_average_cross` | Add full MA parameter-grid backtest wrapper and parity with FXBacktest strategy output. | Metal parameter sweep is suitable; add batch parity and edge-case kernel tests. | Fast/slow window parity, flat-market skip, grid result match. |
| `fxbacktest_fxstupid` | Keep as demo baseline unless promoted; if promoted, specify real strategy objective first. | CPU only. | Demo strategy golden signals and reset behavior. |
| `fx7` | Keep the full backtest-native FX7 source under plugin ownership and add the FXDataEngine adapter for momentum, regime, volatility, panic, volume, and cost gates. | Metal signal scorer is required; Swift SIMD/Accelerate remain CPU candidates. | CPU/Metal signal-score parity, FXBacktest source linkage, trend/skip fixtures. |
| `lin_sgd` | Add canonical online multinomial logistic SGD objective, optimizer modes, coefficient diagnostics. | Swift SIMD/Accelerate for dot products; Metal only for batch scoring. | Golden parity against canonical online logistic replay. |
| `lin_ftrl` | Add strict FTRL-Proximal equations, L1/L2 path diagnostics, sparse feature fixtures. | Swift SIMD/Accelerate; Metal batch scoring if sparse gather becomes bottleneck. | FTRL z/n update parity and L1 shrinkage tests. |
| `lin_enhash` | Formalize ENHash spec, collision policy, dual-field interaction math. | Existing Metal pair-product helper; add collision stress batch kernel tests. | Collision determinism, legacy replay, volume-gated hashed features. |
| `lin_pa` | Add PA-I/PA-II/C modes and exact Crammer-Singer multiclass update tests. | Swift SIMD/Accelerate; Metal optional batch margin scoring. | Canonical multiclass PA parity and averaged-weight replay. |
| `lin_elastic_logit` | Implement true elastic-net logistic objective with proximal L1 and L2 shrinkage. | Swift SIMD/Accelerate; Metal batch scoring only. | Coefficient path, shrinkage, and logistic parity fixtures. |
| `lin_profit_logit` | Implement profit-weighted/cost-sensitive logistic loss with asymmetric trade costs. | Swift SIMD/Accelerate; Metal batch scoring only. | Prove loss diverges from plain logit under asymmetric costs. |
| `dist_quantile` | Add full pinball-loss training checks, monotonic quantile constraints, PIT/coverage diagnostics. | Metal scoring kernels remain useful; add quantile parity kernels only if training moves to GPU. | Pinball-loss, crossing, coverage, PIT golden tests. |
| `mem_retrdiff` | Add exact kNN retrieval reference, eviction policies, approximate top-k option. | Metal distance/top-k kernels; CPU exact fallback. | Deterministic nearest-neighbor parity and memory eviction fixtures. |
| `mix_loffm` | Write formal LOFFM spec, factor/expert update derivation, state persistence. | PyTorch MoE/factor backend with `nn.Module`; optional Metal gate scoring. | Expert specialization, gate ablation, persistence parity. |
| `mix_moe_conformal` | Add split conformal prediction contract, calibration set management, coverage metrics. | PyTorch MoE router with conformal calibration tensors; CPU fallback. | Coverage guarantee, expert routing, interval validity tests. |
| `rl_ppo` | Replace supervised dense adapter with actor-critic PPO: rollout buffer, GAE, clipped policy loss, entropy/value losses, log-probs, offline FX environment. | PyTorch/MPS reference trainer; CPU Swift remains inference fallback; no TF unless needed. | PPO loss terms, GAE, clipping, rollout accounting, offline reward tests. |
| `tree_xgb_fast` | Add canonical XGBoost gain/regularization, missing-value routing, column/row sampling. | Metal inference stays; future Metal histogram/gain search if backtest grid needs it. | Gain parity, missing-value default direction, objective fixtures. |
| `tree_xgb` | Upgrade to multiclass gradient boosting with shrinkage, sampling, learned missing routing. | Metal inference and batch scoring; CPU builder is reference. | Multiclass tree parity and leaf-weight tests. |
| `tree_lgbm` | Add full histogram binning, leaf-wise growth constraints, categorical handling, DART dropout. | Metal histogram/scoring kernels where useful. | LightGBM-style bin, split, DART, and early-stop fixtures. |
| `tree_catboost` | Add strict ordered boosting permutation, ordered CTR priors, leakage guards. | Metal symmetric-tree scoring; CPU ordered training reference. | CTR leakage, ordered permutation, symmetric split parity. |
| `tree_rf` | Implement true bagging, bootstrap sampling, random feature subsets, split search, OOB metrics. | Metal forest scoring; CPU training reference. | Bootstrap/OOB and split-search golden tests. |
| `factor_carry` | Wire FXDatabase rates/forward-point inputs, cross-currency carry normalization. | CPU/Accelerate vector normalization; no ML accelerator needed. | Rates input contract and carry ranking fixtures. |
| `factor_cmv_panel` | Define CMV formula, panel input contract, cross-sectional exposure normalization. | CPU/Accelerate panel vector ops. | Panel normalization, missing symbol, volume gating tests. |
| `factor_pca_panel` | Implement covariance accumulation, Accelerate eigensolver/SVD, orthonormal loadings, explained variance. | Accelerate is primary; Metal only for large panel covariance batches. | Eigenvalue/loadings parity and orthogonality tests. |
| `factor_ppp_value` | Add PPP/fair-value data contract, z-score normalization, valuation half-life. | CPU/Accelerate cross-sectional scoring. | PPP input and valuation rank fixtures. |
| `trend_tsmom_vol` | Add canonical TSMOM volatility targeting, lookback/horizon grids, cost-aware risk scaling. | Metal batch scanner and parameter-grid scoring. | Vol-target parity, costs, and portfolio-level tests. |
| `trend_xsmom_rank` | Replace single-symbol adapter with true cross-sectional rank engine, tie/missing handling, neutralization. | Accelerate for rank/z-score; Metal optional for large universe scans. | Multi-symbol rank, neutralization, tie/missing fixtures. |
| `trend_vol_breakout` | Add ATR/realized-vol breakout bands, stops/targets, regime filters, grid backtest parity. | Metal batch scanner remains useful. | Breakout band, stop/target, and regime filter tests. |
| `stat_arimax_garch` | Implement ARIMAX design matrix, AR/MA residual recursion, exogenous regressors, GARCH likelihood and intervals. | CPU/Accelerate linear algebra; PyTorch optional only for likelihood optimization experiments. | ARIMA/GARCH synthetic series and residual diagnostics. |
| `stat_coint_vecm` | Implement Johansen rank test, beta vectors, VECM alpha/gamma matrices, lag selection. | CPU/Accelerate eigensolver; Metal not useful. | Cointegration rank, residual stationarity, VECM forecast tests. |
| `stat_emd_hht` | Replace proxy with EMD sifting, IMF extraction, Hilbert transform and instantaneous frequency. | Metal can accelerate batch extrema/envelope/Hilbert scoring after CPU reference. | IMF decomposition and Hilbert frequency fixtures. |
| `stat_hmm_regime` | Add Baum-Welch EM, log-space forward/backward, multi-feature Gaussian emissions, Viterbi decode. | CPU/Accelerate log-likelihood; PyTorch optional only for batch EM research. | Likelihood, posterior, and Viterbi parity tests. |
| `stat_microflow_proxy` | Keep honest OHLCV proxy name; add explicit contract that no tick/order book inputs are used. | CPU only unless batch feature extraction needs Metal. | OHLCV proxy semantics and volume gating tests. |
| `stat_msgarch` | Implement Markov-switching GARCH likelihood, regime variance recursion, filtered/smoothed probabilities. | CPU/Accelerate optimizer; no GPU until likelihood batch needs it. | MSGARCH synthetic regime and variance fixtures. |
| `stat_ou_spread` | Add MLE OU parameter estimation, half-life diagnostics, stationarity checks, residual source contract. | CPU/Accelerate. | OU parameter recovery and half-life tests. |
| `stat_tvp_kalman` | Implement full state-space matrices, covariance propagation/update, Q/R estimation, missing-data update. | CPU/Accelerate matrices. | Kalman predict/update parity and missing data fixtures. |
| `stat_vmd` | Replace EMA proxy with VMD ADMM iterations, bandwidth constraints, convergence checks. | Metal can accelerate mode update batches after CPU reference. | VMD decomposition, convergence, mode bandwidth tests. |
| `stat_xrate_consistency` | Add explicit triangular arbitrage graph, quote normalization, cycle consistency scoring. | CPU/Accelerate graph math; Metal optional for large cycle batches. | Triangle/cycle parity and symbol graph fixtures. |
| `ai_mlp` | Upgrade PyTorch/TF to real configurable deep MLP with normalization, dropout, checkpointable heads. | PyTorch MPS and TensorFlow Metal reference backends; CPU Swift fallback. | Layer presence, train step, checkpoint, Swift/Python parity smoke. |
| `ai_lstm` | Implement true LSTM sequence model, hidden/cell state, truncated BPTT, checkpointing. | PyTorch `nn.LSTM`; TensorFlow `LSTM`; optional CoreML export later. | LSTM layer tests, sequence batching, hidden-state reset. |
| `ai_lstmg` | Implement LSTM plus explicit gating/GLU residual gate over market features. | PyTorch and TensorFlow gated LSTM backends. | Gate activation, sequence state, training parity. |
| `ai_gru` | Implement true GRU reset/update gates and recurrent sequence training. | PyTorch `nn.GRU`; TensorFlow `GRU`. | GRU layer tests, recurrent state, reset. |
| `ai_bilstm` | Implement bidirectional LSTM with forward/backward merge and sequence packing. | PyTorch/TensorFlow BiLSTM. | Bidirectional output shape and merge tests. |
| `ai_lstm_tcn` | Implement LSTM encoder plus dilated causal TCN residual stack. | PyTorch/TensorFlow Conv1D + LSTM. | Receptive field and sequence parity tests. |
| `ai_cnn_lstm` | Implement Conv1D feature extractor, pooling, LSTM decoder/head. | PyTorch/TensorFlow Conv1D + LSTM. | Conv/LSTM layer presence and output shape tests. |
| `ai_attn_cnn_bilstm` | Implement Conv1D front end, BiLSTM, multi-head attention pooling. | PyTorch/TensorFlow attention backends. | Attention weight shape and BiLSTM tests. |
| `ai_tcn` | Implement dilated causal residual TCN blocks with receptive-field controls. | PyTorch/TensorFlow Conv1D backends. | Causal padding and receptive-field tests. |
| `ai_tst` | Implement time-series transformer encoder with positional encoding, masking, FFN blocks. | PyTorch `TransformerEncoder`; no TF unless required later. | Attention/mask/position tests. |
| `ai_tft` | Implement Temporal Fusion Transformer: variable selection, GRN, LSTM encoder/decoder, interpretable attention, quantile heads. | PyTorch MPS primary; TensorFlow optional later if parity is needed. | Variable selection, GRN, attention, quantile tests. |
| `ai_autoformer` | Implement decomposition blocks, auto-correlation attention, seasonal/trend heads. | PyTorch MPS. | Decomposition and autocorrelation fixtures. |
| `ai_patchtst` | Implement patch extraction, channel-independent transformer, patch positional encoding, horizon head. | PyTorch MPS. | Patch shape, transformer, horizon tests. |
| `ai_s4` | Implement S4/S4D diagonal state-space kernel with stable parameterization. | PyTorch MPS where supported; CPU fallback for unsupported ops. | Kernel stability and sequence impulse tests. |
| `ai_stmn` | Implement spatio-temporal memory network with memory slots, attention read/write, update policy. | PyTorch MPS. | Memory read/write and slot update tests. |
| `ai_chronos` | Implement Chronos-style time-series tokenization, causal token transformer, optional pretrained checkpoint hook; NLP event features stay auxiliary. | PyTorch MPS plus NLP feature merger. | Tokenization, causal mask, checkpoint hook tests. |
| `ai_timesfm` | Implement TimesFM-style patch/horizon foundation forecaster with quantile horizon heads; NLP remains auxiliary. | PyTorch MPS plus NLP feature merger. | Patch horizon, quantile, and checkpoint tests. |
| `ai_fewc` | Define FEWC as feature-ensemble with elastic weight consolidation; implement Fisher penalty. | PyTorch MPS. | Fisher update and EWC penalty tests. |
| `ai_geodesic` | Implement geodesic/RBF attention over feature manifold distances. | PyTorch MPS. | Distance kernel and attention normalization tests. |
| `ai_gha` | Implement Generalized Hebbian Algorithm/Oja projection with orthonormal components. | PyTorch MPS for batch projection; CPU Swift fallback. | Orthogonality and principal component recovery tests. |
| `ai_qcew` | Implement quantile cross-entropy window objective with quantile and class heads. | PyTorch MPS. | Quantile CE loss and coverage tests. |
| `ai_tesseract` | Implement tensor factorization/tensor-network contraction with rank controls. | PyTorch MPS; Metal only if contractions become hot. | Tensor rank, contraction shape, and parity tests. |
| `ai_trr` | Implement trend/reversal recurrent regime model with transition logic and separate heads. | PyTorch MPS. | Regime transition and trend/reversal labels. |
| `ai_mythos_rdt` | Implement recursive decision transformer: trajectory tokens, return conditioning, recursive memory, causal transformer. | PyTorch MPS plus NLP event merger. | Causal mask, return conditioning, recursion tests. |
| `wm_cfx` | Implement currency-factor world model: currency graph, exposure matrix, latent factor evolution, cross-rate loss. | PyTorch graph/factor backend; CPU fallback. | Exposure, factor evolution, cross-rate consistency. |
| `wm_graph` | Implement graph neural network: nodes, edges, message passing, adjacency from FX universe, graph consistency loss. | PyTorch message-passing backend; Metal projection helper remains batch support. | Message passing, adjacency, graph loss tests. |

## Revision Pass 1: Contract And Data Crosscheck

Applied checks:

- No plugin may reintroduce spread inputs; all plans use M1 OHLCV and context
  fields available through FXDataEngine/FXDatabase.
- Volume is explicitly gated by `dataHasVolume`.
- Plugin-local implementation ownership remains intact: accelerators live under
  each plugin folder; shared code is limited to FXDataEngine/API contracts.
- CPU fallback remains deterministic for offline Swift/Metal backtesting.

Revision result: statistical and panel models must first receive richer
FXDatabase contracts before they can reach 99%; AI framework backends can be
upgraded immediately because their plugin-local Python contracts already exist.

## Revision Pass 2: Accelerator Realism Crosscheck

Applied checks:

- AI plugins must contain real framework layers, not dense placeholder encoders.
- Metal is accepted as a batch scorer only where the full training algorithm is
  not a good GPU-kernel fit.
- TensorFlow is required only where a TensorFlow folder already exists or where
  Keras parity adds material value; PyTorch is primary for most advanced AI and
  RL plugins because MPS support is broad and implementation velocity is higher.
- Neural Engine use should come through later CoreML export artifacts, not
  hand-written plugin logic.

Revision result: the first implementation wave is framework backend realism for
AI/RL/world plugins, because this closes the largest shell/proxy gap without
breaking Swift CPU behavior. The second wave is statistical reference math and
the third wave is tree/linear/factor validation depth.

## Implementation Progress

### 2026-05-23 Wave 1 Completed: AI/RL/World Framework Backends

Implemented:

- Replaced generic dense PyTorch adapters for AI/RL/world plugins with
  plugin-local reference backends using real layer families:
  LSTM, gated LSTM, GRU, BiLSTM, Conv1D, TCN, transformer encoder, Temporal
  Fusion Transformer components, Autoformer-style decomposition/autocorrelation,
  PatchTST patch encoder, S4D state-space layer, spatio-temporal memory slots,
  Chronos-style causal tokenization, TimesFM-style patch horizon heads, FEWC
  Fisher penalty, geodesic attention, GHA projection, QCEW quantile objective,
  tensor contractions, trend/reversal recurrence, recursive decision transformer,
  currency-factor world modeling, FX graph message passing, and PPO
  actor-critic/GAE/clipped loss.
- Replaced existing TensorFlow AI backends with Keras `tf.keras.Model` code that
  contains the expected LSTM/GRU/BiLSTM/Conv1D/TCN/attention layer families.
- Upgraded foundation-model NLP helpers with n-gram, policy, volatility,
  currency-focus, and novelty features.
- Fixed PyTorch accelerator data-shape handling so sequence windows are accepted
  by PPO and mixture backends.
- Added `ReferenceBackendConformanceTests` so backend files cannot regress to
  placeholder dense adapters without failing Swift tests.

Verified:

- `python3 -m py_compile` over all plugin-local PyTorch, TensorFlow, and NLP files.
- Runtime PyTorch smoke over every plugin-local PyTorch backend, including
  sequence-window predict/train checks.
- `swift test --package-path FXPlugins --scratch-path /tmp/fxai-swiftpm/FXPlugins`
  with 429 tests passing.

### 2026-05-23 Wave 2 Completed: Statistical Reference Math

Implemented plugin-local Swift reference math for:

- `stat_arimax_garch`: ARIMAX design-matrix fitting and GARCH likelihood grid
  evaluation.
- `stat_coint_vecm`: pair cointegration OLS, residual ADF diagnostic,
  error-correction speeds, spread z-scores, and VECM next-step deltas.
- `stat_emd_hht`: EMD sifting with IMF extraction, envelopes, analytic-signal
  Hilbert summary, instantaneous amplitude, and instantaneous frequency.
- `stat_hmm_regime`: log-space Baum-Welch EM, posterior normalization, and
  Viterbi decoding.
- `stat_msgarch`: Markov-switching GARCH filtered probabilities, variance
  recursion, and likelihood.
- `stat_ou_spread`: OU MLE-style mean-reversion estimate, half-life, and
  z-score.
- `stat_tvp_kalman`: full matrix predict/update Kalman reference with
  covariance propagation and missing-observation handling.
- `stat_vmd`: deterministic spectral VMD-style mode updates, center-frequency
  tracking, reconstruction, residuals, and convergence norms.
- `stat_xrate_consistency`: normalized quote graph, triangle/cycle scoring,
  implied-vs-quoted rates, and basis-point imbalance.

Verified by `Wave2StatisticalReferenceTests`, covering synthetic recovery,
positive variance/likelihood invariants, log-space HMM posterior sums, Kalman
missing-data behavior, VECM beta/residual diagnostics, EMD/HHT and VMD
decomposition, and triangular FX consistency.

### 2026-05-23 Wave 3 Completed: Factor And Trend Panel Contracts

Implemented plugin-local reference contracts for:

- `factor_carry`: cross-currency rate/forward carry normalization with explicit
  liquidity weighting only when `dataHasVolume` is true.
- `factor_cmv_panel`: cross-sectional momentum, value, and volume exposure
  normalization with volume fully gated by dataset volume availability.
- `factor_pca_panel`: covariance accumulation, power-iteration first principal
  component, orthonormal loadings, scores, and explained variance ratio.
- `factor_ppp_value`: PPP/fair-value misvaluation z-scores with stale-data
  half-life decay.
- `trend_tsmom_vol`: time-series momentum, realized volatility targeting,
  liquidity confidence, and final leverage cap after volume adjustment.
- `trend_xsmom_rank`: cross-sectional momentum ranking, neutralization,
  average tie ranks, and balanced long/short weights.
- `trend_vol_breakout`: ATR, prior-window breakout bands, stops, targets, and
  direction.

Verified by `Wave3FactorTrendReferenceTests`, including volume-gating checks,
PCA orthogonality, PPP staleness, final leverage cap, rank balance, and ATR
breakout behavior.

### 2026-05-23 Wave 4 Completed: Linear, Tree, Distribution, And Memory Parity

Implemented plugin-local reference fixtures for:

- `lin_sgd`: canonical multinomial logistic softmax SGD update.
- `lin_ftrl`: FTRL-Proximal z/n state, L1 shrinkage, and binary logistic
  update.
- `lin_enhash`: deterministic field interaction hashing and collision
  diagnostics.
- `lin_pa`: Crammer-Singer PA, PA-I, and PA-II margin updates.
- `lin_elastic_logit`: proximal elastic-net logistic step.
- `lin_profit_logit`: profit-weighted logistic loss and asymmetric gradients.
- `dist_quantile`: pinball loss, coverage, and monotonic quantile projection.
- `mem_retrdiff`: exact Euclidean top-k retrieval and deterministic recency
  eviction.
- `tree_xgb_fast`: XGBoost split gain, leaf weight, and missing-direction
  comparison.
- `tree_xgb`: multiclass gradient/hessian reference and missing-value routing.
- `tree_lgbm`: histogram binning, best-split scan, and deterministic DART mask.
- `tree_catboost`: ordered CTRs without target leakage and symmetric-tree leaf
  indexing.
- `tree_rf`: deterministic bootstrap/OOB sample and Gini split scan.

Verified by `Wave4ReferenceParityTests`, covering canonical update equations,
collision determinism, quantile monotonicity, exact nearest-neighbor retrieval,
boosting gain math, ordered CTR leakage prevention, histogram split gain, DART
determinism, and random-forest OOB/split invariants.

Remaining waves: none. The 99 percent plan now has implementation coverage
across framework backends, statistical reference math, factor/trend panel
contracts, and linear/tree/distribution/memory parity fixtures.
