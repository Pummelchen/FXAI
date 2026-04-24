# Model Zoo

This page explains the FXAI model zoo from an operator and researcher point of view. The live runtime stays MT5-native: plugins consume the FXAI plugin payload, return calibrated class probabilities plus skip/abstain behavior, and do not call external live inference services.

## User Matrix

| User | Main Goal | Primary FXAI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Why This Page Matters

- Live Trader: model names are not trade permission. Use them to understand which family contributed to the final posture.
- Demo Trader: the zoo helps explain why two market states with similar direction scores can produce different actions.
- Backtester: compare families by scenario, not only by headline score.
- EA Researcher: use the zoo to choose the right family for the weakness shown in audit results.
- System Architect: verify that every plugin follows the same native API and persistence contract.

## Expanded Native Families

The expanded zoo adds these MT5-native plugin modules:

| Area | Plugins | Primary Use |
|---|---|---|
| Regime and volatility | `stat_msgarch`, `stat_hmm_regime`, `stat_tvp_kalman` | regime probability, variance state, transition caution |
| Econometric mean/variance | `stat_arimax_garch`, `stat_coint_vecm`, `stat_ou_spread` | mean forecast, residual risk, equilibrium dislocation |
| Trees and linear | `tree_rf`, `lin_elastic_logit`, `lin_profit_logit` | bagged voting, sparse classification, cost-aware direction |
| Sequence hybrids | `ai_gru`, `ai_bilstm`, `ai_lstm_tcn`, `ai_cnn_lstm`, `ai_attn_cnn_bilstm` | historical-window sequence patterns without future leakage |
| Decomposition | `stat_emd_hht`, `stat_vmd` | trend/cycle/noise separation and unstable-decomposition abstention |
| Factor and value | `factor_pca_panel`, `factor_ppp_value`, `factor_carry`, `factor_cmv_panel` | panel factors, value, carry, cross-sectional ranking |
| Trend and breakout | `trend_tsmom_vol`, `trend_xsmom_rank`, `trend_vol_breakout` | volatility-scaled trend, relative strength, range expansion |
| Cross-pair and policy | `stat_xrate_consistency`, `rl_ppo`, `stat_microflow_proxy` | cross-rate pressure, policy action scoring, broker-visible microstructure proxy gating |

## How To Use It

1. Run the normal FXAI verification path after changing plugin breadth: `python3 Tools/fxai_testlab.py verify-all`.
2. Use Audit Lab to compare families on scenarios such as `random_walk`, `drift_up`, `regime_shift`, `vol_cluster`, and `mean_revert`.
3. Use plugin oracle findings to decide whether a family is behaving as intended.
4. Let Adaptive Router and Dynamic Ensemble decide live weights; do not manually force all plugins to vote equally.
5. Treat `stat_microflow_proxy` as a broker-visible proxy layer only. It is not centralized FX order flow or a true institutional order book.

## Example Case Scenarios

### Scenario: Random walk overtrading

If a family is active too often on random-walk tests, inspect its oracle entry and skip ratio. Prefer models whose confidence falls when edge after cost is weak.

### Scenario: Volatility cluster

Check `stat_msgarch`, `stat_hmm_regime`, `stat_tvp_kalman`, `trend_vol_breakout`, and execution-quality output together. A healthy result is often caution or abstention, not more trades.

### Scenario: Mean reversion basket idea

Use `stat_coint_vecm` and `stat_ou_spread` as the first inspection points, then check Pair Network so the final trade does not duplicate or contradict existing currency exposure.

### Scenario: Sequence model upgrade

Compare `ai_lstm`, `ai_tcn`, `ai_gru`, `ai_bilstm`, `ai_lstm_tcn`, and `ai_cnn_lstm` on the same audit window. Directional improvement is not enough; reset stability, skip behavior, and post-cost quality must also hold.

## Engineering Rules

- Plugins consume FXAI payloads and context only; they do not pull raw MT5 bars directly.
- Live inference remains MT5-native and DLL-free.
- The canonical market-training contract remains `M1 OHLC + spread`.
- Microstructure features remain MT5-visible broker-side proxies and must fail safe when stale or missing.
- All plugin outputs must remain probability, confidence, and abstention compatible.

## Next Pages

- [Adaptive Router](Adaptive%20Router.md)
- [Dynamic Ensemble](Dynamic%20Ensemble.md)
- [Audit Lab](Audit%20Lab.md)
- [Data Policy](Data%20Policy.md)
