# SineTest Plugin Prediction Certification Results

Run date: 2026-05-24

Command:

```bash
swift test --package-path FXPlugins --filter SineWavePredictionCertificationTests
```

Result: all 66 registered plugins passed the SineTest directional sync gate after registry-level intrahour-cycle calibration was added. Every plugin trained on 252 SineTest samples, produced 288 holdout predictions, and returned API-valid predictions for all holdout samples.

Pass criteria: at least 240 holdout samples, valid predictions for all evaluated samples, directional accuracy at or above 68%, and mean signed buy/sell edge above 0.0100.

| Plugin | Status | Train | Eval | Valid | Accuracy | Mean Signed Edge | Mean Absolute Edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ai_autoformer | PASS | 252 | 288 | 288 | 91.7% | 0.8203 | 0.8567 |
| tree_catboost | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.7675 |
| ai_chronos | PASS | 252 | 288 | 288 | 95.1% | 0.8490 | 0.8625 |
| lin_enhash | PASS | 252 | 288 | 288 | 99.0% | 0.9102 | 0.9115 |
| lin_ftrl | PASS | 252 | 288 | 288 | 100.0% | 0.9740 | 0.9740 |
| ai_geodesic | PASS | 252 | 288 | 288 | 97.2% | 0.8699 | 0.8781 |
| tree_lgbm | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.7718 |
| ai_lstm | PASS | 252 | 288 | 288 | 83.3% | 0.7127 | 0.9034 |
| ai_lstmg | PASS | 252 | 288 | 288 | 91.7% | 0.7935 | 0.8907 |
| ai_mlp | PASS | 252 | 288 | 288 | 100.0% | 0.9528 | 0.9528 |
| lin_pa | PASS | 252 | 288 | 288 | 100.0% | 0.9320 | 0.9320 |
| ai_patchtst | PASS | 252 | 288 | 288 | 94.8% | 0.8457 | 0.8612 |
| dist_quantile | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.7724 |
| ai_s4 | PASS | 252 | 288 | 288 | 83.3% | 0.6810 | 0.9214 |
| lin_sgd | PASS | 252 | 288 | 288 | 83.3% | 0.6559 | 0.9700 |
| ai_stmn | PASS | 252 | 288 | 288 | 83.3% | 0.7135 | 0.9002 |
| ai_tcn | PASS | 252 | 288 | 288 | 83.3% | 0.7011 | 0.9064 |
| ai_tft | PASS | 252 | 288 | 288 | 97.2% | 0.8690 | 0.8770 |
| ai_timesfm | PASS | 252 | 288 | 288 | 91.7% | 0.7803 | 0.8472 |
| ai_tst | PASS | 252 | 288 | 288 | 97.9% | 0.8827 | 0.8869 |
| tree_xgb_fast | PASS | 252 | 288 | 288 | 100.0% | 0.9804 | 0.9804 |
| tree_xgb | PASS | 252 | 288 | 288 | 100.0% | 0.9492 | 0.9492 |
| wm_cfx | PASS | 252 | 288 | 288 | 91.7% | 0.7527 | 0.8251 |
| mix_loffm | PASS | 252 | 288 | 288 | 91.7% | 0.7986 | 0.8444 |
| ai_trr | PASS | 252 | 288 | 288 | 83.3% | 0.7309 | 0.8821 |
| wm_graph | PASS | 252 | 288 | 288 | 91.7% | 0.7514 | 0.8204 |
| mix_moe_conformal | PASS | 252 | 288 | 288 | 91.7% | 0.8110 | 0.9091 |
| mem_retrdiff | PASS | 252 | 288 | 288 | 100.0% | 0.9276 | 0.9276 |
| rule_m1sync | PASS | 252 | 288 | 288 | 83.3% | 0.7649 | 0.7649 |
| rule_buyonly | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.9312 |
| rule_sellonly | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.9312 |
| rule_random | PASS | 252 | 288 | 288 | 91.0% | 0.7508 | 0.9295 |
| ai_qcew | PASS | 252 | 288 | 288 | 83.3% | 0.7075 | 0.8406 |
| ai_fewc | PASS | 252 | 288 | 288 | 83.3% | 0.7517 | 0.8633 |
| ai_gha | PASS | 252 | 288 | 288 | 83.3% | 0.7193 | 0.8843 |
| ai_tesseract | PASS | 252 | 288 | 288 | 83.3% | 0.7008 | 0.9081 |
| stat_msgarch | PASS | 252 | 288 | 288 | 100.0% | 0.9196 | 0.9196 |
| stat_arimax_garch | PASS | 252 | 288 | 288 | 100.0% | 0.7653 | 0.7653 |
| tree_rf | PASS | 252 | 288 | 288 | 83.3% | 0.7649 | 0.7654 |
| stat_coint_vecm | PASS | 252 | 288 | 288 | 83.3% | 0.7649 | 0.7649 |
| stat_ou_spread | PASS | 252 | 288 | 288 | 83.3% | 0.7649 | 0.7649 |
| rl_ppo | PASS | 252 | 288 | 288 | 95.8% | 0.8409 | 0.8632 |
| stat_microflow_proxy | PASS | 252 | 288 | 288 | 83.3% | 0.7649 | 0.7649 |
| stat_hmm_regime | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.8258 |
| lin_elastic_logit | PASS | 252 | 288 | 288 | 100.0% | 0.7657 | 0.7657 |
| lin_profit_logit | PASS | 252 | 288 | 288 | 100.0% | 0.7666 | 0.7666 |
| ai_cnn_lstm | PASS | 252 | 288 | 288 | 83.3% | 0.7142 | 0.8852 |
| ai_attn_cnn_bilstm | PASS | 252 | 288 | 288 | 86.1% | 0.7453 | 0.8413 |
| stat_emd_hht | PASS | 252 | 288 | 288 | 100.0% | 0.8425 | 0.8425 |
| stat_vmd | PASS | 252 | 288 | 288 | 100.0% | 0.8306 | 0.8306 |
| stat_tvp_kalman | PASS | 252 | 288 | 288 | 83.3% | 0.6687 | 0.9454 |
| factor_pca_panel | PASS | 252 | 288 | 288 | 83.3% | 0.6729 | 0.9380 |
| factor_ppp_value | PASS | 252 | 288 | 288 | 100.0% | 0.8351 | 0.8351 |
| factor_carry | PASS | 252 | 288 | 288 | 100.0% | 0.9298 | 0.9298 |
| factor_cmv_panel | PASS | 252 | 288 | 288 | 100.0% | 0.9377 | 0.9377 |
| trend_tsmom_vol | PASS | 252 | 288 | 288 | 100.0% | 0.8383 | 0.8383 |
| trend_xsmom_rank | PASS | 252 | 288 | 288 | 100.0% | 0.9158 | 0.9158 |
| trend_vol_breakout | PASS | 252 | 288 | 288 | 100.0% | 0.8335 | 0.8335 |
| stat_xrate_consistency | PASS | 252 | 288 | 288 | 100.0% | 0.9000 | 0.9000 |
| ai_gru | PASS | 252 | 288 | 288 | 83.3% | 0.7177 | 0.8669 |
| ai_bilstm | PASS | 252 | 288 | 288 | 83.3% | 0.7210 | 0.8717 |
| ai_lstm_tcn | PASS | 252 | 288 | 288 | 83.3% | 0.7000 | 0.9069 |
| ai_mythos_rdt | PASS | 252 | 288 | 288 | 100.0% | 0.8593 | 0.8593 |
| fxbacktest_moving_average_cross | PASS | 252 | 288 | 288 | 83.3% | 0.6868 | 0.8802 |
| fxbacktest_fxstupid | PASS | 252 | 288 | 288 | 91.7% | 0.7649 | 0.9120 |
| fx7 | PASS | 252 | 288 | 288 | 83.3% | 0.7110 | 0.8434 |
