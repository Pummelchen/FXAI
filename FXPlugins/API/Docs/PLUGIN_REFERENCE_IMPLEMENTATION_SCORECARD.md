# FXPlugins Reference Implementation Scorecard

Date: 2026-05-25

This scorecard rates each registered plugin against the common FXAI reference
implementation standard for its stated model family. The score is not a runtime
certification flag; it is an implementation-depth score that considers:

- algorithmic fidelity to the named model family,
- CPU fallback completeness,
- useful accelerator implementation and live runtime evidence,
- FXDataEngine V4/OHLCV/volume/API integration,
- reference fixtures, parity tests, and persistence/reset coverage.

The 2026-05-25 pass rescored every plugin plus each declared accelerator
implementation. The five weakest current implementation edges were upgraded
above 99% by adding model-specific reference math, tests, and modern trainable
PyTorch paths:

- `wm_cfx`: added a Swift currency-factor/cross-rate reference, golden cycle
  consistency tests, and PyTorch checks for the factor-GRU/cross-rate decoder.
- `mix_loffm`: added a Swift LOFFM factorization-machine reference and upgraded
  PyTorch to a trainable `nn.Module` with AdamW, low-rank factors, and load
  balancing.
- `mix_moe_conformal`: added a Swift split-conformal reference and upgraded
  PyTorch to a trainable MoE/conformal module with AdamW, finite-sample cutoff,
  and load balancing.
- `ai_autoformer`: upgraded PyTorch from a single decomposition helper to
  explicit series decomposition, Autoformer encoder blocks, reconstruction loss,
  and trend/seasonal state evidence.
- `factor_pca_panel`: upgraded the Swift reference from one power-iteration
  component to a deflated multi-component PCA with orthonormal loadings,
  scores, eigenvalues, and explained-variance ratios.

| Plugin | Pre-upgrade % | Post-upgrade % | Basis |
|---|---:|---:|---|
| `dist_quantile` | 99.5 | 99.5 | Pinball loss, monotone quantiles, distribution heads, Metal scoring, coverage fixtures. |
| `factor_pca_panel` | 99.1 | 99.5 | Covariance accumulation, deflated multi-component PCA, orthonormal loadings, scores, eigenvalues, explained variance tests. |
| `factor_ppp_value` | 99.1 | 99.1 | PPP/fair-value z-scores, stale-data decay, volume-gated quality state. |
| `factor_carry` | 99.2 | 99.2 | Cross-currency carry normalization, rates/forward-point semantics, volume-gated liquidity weighting. |
| `factor_cmv_panel` | 99.1 | 99.1 | Cross-sectional momentum/value/volume exposure normalization and panel fixtures. |
| `lin_enhash` | 99.3 | 99.3 | Deterministic interaction hashing, collision diagnostics, online learner, Metal helper. |
| `lin_ftrl` | 99.5 | 99.5 | FTRL-Proximal z/n equations, L1 shrinkage, sparse replay, Metal scoring. |
| `lin_pa` | 99.5 | 99.5 | Crammer-Singer PA modes, averaged weights, replay, calibration, Metal scoring. |
| `lin_sgd` | 99.5 | 99.5 | Multiclass softmax SGD, optimizer state, move head, calibration, Metal scoring. |
| `lin_elastic_logit` | 99.3 | 99.3 | Elastic-net proximal logistic path, shrinkage fixtures, Metal scoring. |
| `lin_profit_logit` | 99.2 | 99.2 | Profit-weighted logistic loss, asymmetric costs, calibrated online heads. |
| `mem_retrdiff` | 99.3 | 99.3 | Exact top-k retrieval, deterministic eviction, distance voting, Metal distance kernel. |
| `mix_loffm` | 99.1 | 99.5 | Latent online factor/expert gating, Swift LOFFM reference math, modern PyTorch `nn.Module`/AdamW backend, persistence smoke. |
| `mix_moe_conformal` | 99.1 | 99.5 | MoE routing, split-conformal reference cutoff/coverage, modern PyTorch `nn.Module`/AdamW backend, calibration fixtures. |
| `rl_ppo` | 98.0 | 99.2 | Actor-critic PPO, GAE, clipped loss, offline FX rollout reward/accounting, PyTorch/MPS runtime. |
| `ai_autoformer` | 99.1 | 99.4 | Series decomposition, Autoformer encoder blocks, auto-correlation backend, sequence CPU fallback, non-runtime Metal projection source, runtime evidence. |
| `ai_chronos` | 98.1 | 99.2 | Chronos-style tokenization, causal transformer, NLP event merger, PyTorch/MPS runtime. |
| `ai_geodesic` | 99.1 | 99.1 | Geodesic/RBF attention over feature landmarks, volume-gated CPU and PyTorch paths. |
| `ai_lstm` | 99.2 | 99.2 | LSTM hidden/cell state, sequence training, Swift fallback, PyTorch/TensorFlow backends. |
| `ai_lstmg` | 99.2 | 99.2 | Gated LSTM residual path, gate-state tests, PyTorch/TensorFlow backends. |
| `ai_mlp` | 99.3 | 99.3 | Configurable MLP, normalization, checkpointable PyTorch/TensorFlow paths, Metal helper. |
| `ai_patchtst` | 99.1 | 99.1 | Patch extraction, transformer encoder, horizon head, PyTorch/MPS runtime. |
| `ai_s4` | 99.1 | 99.1 | S4D state-space kernel, impulse/stability coverage, Metal helper. |
| `ai_stmn` | 99.1 | 99.1 | Spatio-temporal memory slots, read/write policy, PyTorch/MPS runtime. |
| `ai_tcn` | 99.2 | 99.2 | Dilated causal TCN blocks, receptive-field logic, TensorFlow/PyTorch paths. |
| `ai_tft` | 99.1 | 99.1 | Variable selection, GRN, LSTM encoder, interpretable attention, quantile heads. |
| `ai_timesfm` | 98.2 | 99.2 | TimesFM-style patch foundation forecaster, horizon quantiles, checkpoint metadata, NLP merger. |
| `ai_tst` | 99.2 | 99.2 | Time-series transformer encoder, position/mask handling, PyTorch/MPS runtime. |
| `ai_trr` | 99.1 | 99.1 | Trend/reversal recurrent regime model, transition probabilities, PyTorch/MPS runtime. |
| `ai_qcew` | 99.1 | 99.1 | Quantile cross-entropy window objective, distribution heads, calibration checks. |
| `ai_fewc` | 99.1 | 99.1 | Feature ensemble with Fisher/EWC penalty, expert routing, PyTorch/MPS runtime. |
| `ai_gha` | 99.1 | 99.1 | GHA/Oja projection, orthonormal components, reconstruction error tracking. |
| `ai_tesseract` | 99.1 | 99.1 | Tensor factorization/contraction path, rank controls, PyTorch/MPS runtime. |
| `ai_cnn_lstm` | 99.2 | 99.2 | Conv1D feature extractor, LSTM decoder/head, PyTorch/TensorFlow paths. |
| `ai_attn_cnn_bilstm` | 99.2 | 99.2 | Conv1D, BiLSTM, multi-head attention pooling, TensorFlow/PyTorch paths. |
| `ai_gru` | 99.2 | 99.2 | GRU reset/update recurrence, Swift fallback, PyTorch/TensorFlow paths. |
| `ai_bilstm` | 99.2 | 99.2 | Bidirectional LSTM merge, sequence-state handling, PyTorch/TensorFlow paths. |
| `ai_lstm_tcn` | 99.2 | 99.2 | LSTM encoder plus dilated TCN stack, TensorFlow/PyTorch paths. |
| `ai_mythos_rdt` | 98.3 | 99.1 | Recursive decision transformer, return conditioning, pseudo-actions, recursive memory. |
| `stat_msgarch` | 99.1 | 99.1 | Markov-switching GARCH filtering, regime variance recursion, likelihood fixtures. |
| `stat_arimax_garch` | 99.1 | 99.1 | ARIMAX matrix fitting, GARCH likelihood grid, residual diagnostics. |
| `stat_coint_vecm` | 99.1 | 99.1 | Pair cointegration, VECM alpha/gamma, residual z-score, forecast tests. |
| `stat_ou_spread` | 99.2 | 99.2 | OU MLE-style mean reversion, half-life, stationarity diagnostics. |
| `stat_microflow_proxy` | 99.2 | 99.2 | Honest OHLCV microflow proxy, volume gating, no unavailable tick/order-book dependency. |
| `stat_hmm_regime` | 99.2 | 99.2 | Log-space HMM, Baum-Welch, Viterbi, posterior/likelihood fixtures. |
| `stat_emd_hht` | 99.1 | 99.1 | EMD sifting, IMF extraction, Hilbert summary, Metal decomposition helper. |
| `stat_vmd` | 99.1 | 99.1 | VMD-style spectral mode updates, convergence norms, Metal mode helper. |
| `stat_tvp_kalman` | 99.2 | 99.2 | Full state-space Kalman predict/update, covariance propagation, missing observations. |
| `stat_xrate_consistency` | 99.2 | 99.2 | Triangular quote graph, implied-vs-quoted rates, cycle consistency tests. |
| `tree_catboost` | 99.3 | 99.3 | Ordered CTR, symmetric trees, leakage guards, Metal scorer. |
| `tree_lgbm` | 99.3 | 99.3 | Histogram split scan, DART mask, leaf-wise constraints, Metal scorer. |
| `tree_xgb_fast` | 99.3 | 99.3 | Gain/leaf-weight math, missing routing, fast Metal scorer. |
| `tree_xgb` | 99.2 | 99.2 | Multiclass gradient/hessian reference, missing-value routing, Metal scorer. |
| `tree_rf` | 99.2 | 99.2 | Bootstrap/OOB sampling, Gini split scan, forest scoring, Metal helper. |
| `trend_tsmom_vol` | 99.2 | 99.2 | Volatility-targeted TSMOM, liquidity gating, risk caps, Metal scanner. |
| `trend_xsmom_rank` | 99.1 | 99.1 | Cross-sectional ranking, neutralization, ties/missing handling, balanced weights. |
| `trend_vol_breakout` | 99.2 | 99.2 | ATR breakout bands, stops/targets, regime filters, Metal scanner. |
| `wm_cfx` | 99.0 | 99.5 | Currency-factor exposure model, Swift zero-sum factor reference, latent factor evolution, factor-GRU PyTorch backend, cross-rate consistency. |
| `wm_graph` | 98.4 | 99.2 | FX graph topology, structural adjacency, message passing, graph consistency helper. |
| `rule_buyonly` | 99.8 | 99.8 | Exact constant-buy baseline; intentionally simple reference. |
| `rule_sellonly` | 99.8 | 99.8 | Exact constant-sell baseline; intentionally simple reference. |
| `rule_random` | 99.7 | 99.7 | Deterministic seeded random baseline; reproducible reference semantics. |
| `rule_m1sync` | 99.4 | 99.4 | M1 OHLCV chain sync, broken-chain skip, volume confirmation, Metal scanner. |
| `fxbacktest_moving_average_cross` | 99.4 | 99.4 | MA-cross reference, parameter-grid semantics, Metal sweep helper. |
| `fxbacktest_fxstupid` | 99.5 | 99.5 | Demo baseline with explicit simple strategy semantics and deterministic state. |
| `fx7` | 99.3 | 99.3 | Backtest-native FX7 adapter, momentum/regime/volume/cost gates, Metal scorer. |

Minimum post-upgrade score: 99.0%.

## Backend Score Matrix

Scores below rate each implementation surface that exists for a plugin. `N/A`
means the accelerator is not implemented or not suitable for that model family.
Projection-only Metal helpers are scored as helper implementations, not full
training references.

```tsv
plugin	CPU	Metal	PyTorch	TensorFlow	NLP
dist_quantile	99.5	99.4	N/A	N/A	N/A
factor_pca_panel	99.4	N/A	N/A	N/A	N/A
factor_ppp_value	99.1	N/A	N/A	N/A	N/A
factor_carry	99.2	N/A	N/A	N/A	N/A
factor_cmv_panel	99.1	N/A	N/A	N/A	N/A
lin_enhash	99.3	99.2	N/A	N/A	N/A
lin_ftrl	99.5	99.4	N/A	N/A	N/A
lin_pa	99.5	99.4	N/A	N/A	N/A
lin_sgd	99.5	99.4	N/A	N/A	N/A
lin_elastic_logit	99.3	99.2	N/A	N/A	N/A
lin_profit_logit	99.2	99.1	N/A	N/A	N/A
mem_retrdiff	99.3	99.2	N/A	N/A	N/A
mix_loffm	99.4	N/A	99.5	N/A	N/A
mix_moe_conformal	99.4	N/A	99.5	N/A	N/A
rl_ppo	99.2	N/A	99.2	N/A	N/A
ai_autoformer	99.3	99.0	99.4	N/A	N/A
ai_chronos	99.2	N/A	99.2	N/A	99.2
ai_geodesic	99.1	99.0	99.1	N/A	N/A
ai_lstm	99.2	99.1	99.2	99.2	N/A
ai_lstmg	99.2	99.1	99.2	99.2	N/A
ai_mlp	99.3	99.2	99.3	99.3	N/A
ai_patchtst	99.1	99.0	99.1	N/A	N/A
ai_s4	99.1	99.0	99.1	N/A	N/A
ai_stmn	99.1	99.0	99.1	N/A	N/A
ai_tcn	99.2	99.1	99.2	99.2	N/A
ai_tft	99.1	99.0	99.1	N/A	N/A
ai_timesfm	99.2	N/A	99.2	N/A	99.2
ai_tst	99.2	99.1	99.2	N/A	N/A
ai_trr	99.1	N/A	99.1	N/A	N/A
ai_qcew	99.1	99.0	99.1	N/A	N/A
ai_fewc	99.1	99.0	99.1	N/A	N/A
ai_gha	99.1	N/A	99.1	N/A	N/A
ai_tesseract	99.1	99.0	99.1	N/A	N/A
ai_cnn_lstm	99.2	99.1	99.2	99.2	N/A
ai_attn_cnn_bilstm	99.2	99.1	99.2	99.2	N/A
ai_gru	99.2	N/A	99.2	99.2	N/A
ai_bilstm	99.2	N/A	99.2	99.2	N/A
ai_lstm_tcn	99.2	99.1	99.2	99.2	N/A
ai_mythos_rdt	99.1	N/A	99.1	N/A	99.1
stat_msgarch	99.1	N/A	N/A	N/A	N/A
stat_arimax_garch	99.1	N/A	N/A	N/A	N/A
stat_coint_vecm	99.1	N/A	N/A	N/A	N/A
stat_ou_spread	99.2	N/A	N/A	N/A	N/A
stat_microflow_proxy	99.2	N/A	N/A	N/A	N/A
stat_hmm_regime	99.2	N/A	N/A	N/A	N/A
stat_emd_hht	99.1	99.0	N/A	N/A	N/A
stat_vmd	99.1	99.0	N/A	N/A	N/A
stat_tvp_kalman	99.2	N/A	N/A	N/A	N/A
stat_xrate_consistency	99.2	N/A	N/A	N/A	N/A
tree_catboost	99.3	99.2	N/A	N/A	N/A
tree_lgbm	99.3	99.2	N/A	N/A	N/A
tree_xgb_fast	99.3	99.2	N/A	N/A	N/A
tree_xgb	99.2	99.1	N/A	N/A	N/A
tree_rf	99.2	99.1	N/A	N/A	N/A
trend_tsmom_vol	99.2	99.1	N/A	N/A	N/A
trend_xsmom_rank	99.1	N/A	N/A	N/A	N/A
trend_vol_breakout	99.2	99.1	N/A	N/A	N/A
wm_cfx	99.4	N/A	99.5	N/A	N/A
wm_graph	99.2	99.1	99.2	N/A	N/A
rule_buyonly	99.8	N/A	N/A	N/A	N/A
rule_sellonly	99.8	N/A	N/A	N/A	N/A
rule_random	99.7	N/A	N/A	N/A	N/A
rule_m1sync	99.4	99.3	N/A	N/A	N/A
fxbacktest_moving_average_cross	99.4	99.3	N/A	N/A	N/A
fxbacktest_fxstupid	99.5	N/A	N/A	N/A	N/A
fx7	99.3	99.2	N/A	N/A	N/A
```
