   bool   m_initialized;
   bool   m_shadow_ready;
   bool   m_printed_reco;
   int    m_step;
   int    m_adam_t;

   double m_beta1_pow;
   double m_beta2_pow;

   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq_state[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
   double m_seq_season[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
   double m_seq_trend[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];

   // Robust + RevIN-aware normalization state.
   bool   m_norm_ready;
   int    m_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];
   double m_x_loc[FXAI_AI_WEIGHTS];
   double m_x_scale[FXAI_AI_WEIGHTS];

   // Input embedding.
   double m_w_in[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_b_in[FXAI_AI_MLP_HIDDEN];

   // Multi-kernel decomposition + projections.
   double m_w_mix[FXAI_AI_MLP_HIDDEN][FXAI_AF_MA_KERNELS];
   double m_b_mix[FXAI_AI_MLP_HIDDEN];

   double m_w_season[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_b_season[FXAI_AI_MLP_HIDDEN];
   double m_w_trend[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_b_trend[FXAI_AI_MLP_HIDDEN];

   // Stacked auto-correlation blocks.
   double m_wq[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wk[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wv[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wo[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_AF_D_HEAD];

   double m_w_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_b_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_w_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_b_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_w_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_b_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_ln_pre_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_ln_pre_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_ln_post1_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_ln_post1_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_ln_post2_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_ln_post2_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   // Multi-horizon gate + heads.
   double m_w_hgate[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_hgate[FXAI_AF_HORIZONS];

   double m_w_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_b_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT];

   double m_w_mu_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_mu_h[FXAI_AF_HORIZONS];
   double m_w_logv_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_logv_h[FXAI_AF_HORIZONS];
   double m_w_q25_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_q25_h[FXAI_AF_HORIZONS];
   double m_w_q75_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_q75_h[FXAI_AF_HORIZONS];

   // Plugin-native multiclass calibration by session.
   double m_cal_temp[FXAI_AF_SESSIONS];
   double m_cal_bias[FXAI_AF_SESSIONS][FXAI_AF_CLASS_COUNT];
   double m_cal_iso_pos[FXAI_AF_SESSIONS][FXAI_AF_CLASS_COUNT][FXAI_AF_CAL_BINS];
   double m_cal_iso_cnt[FXAI_AF_SESSIONS][FXAI_AF_CLASS_COUNT][FXAI_AF_CAL_BINS];

   // AdamW moments (per-parameter) for critical trainable sets.
   double m_m_w_in[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_v_w_in[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_m_b_in[FXAI_AI_MLP_HIDDEN];
   double m_v_b_in[FXAI_AI_MLP_HIDDEN];

   double m_m_w_mix[FXAI_AI_MLP_HIDDEN][FXAI_AF_MA_KERNELS];
   double m_v_w_mix[FXAI_AI_MLP_HIDDEN][FXAI_AF_MA_KERNELS];
   double m_m_b_mix[FXAI_AI_MLP_HIDDEN];
   double m_v_b_mix[FXAI_AI_MLP_HIDDEN];

   double m_m_w_season[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_v_w_season[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_b_season[FXAI_AI_MLP_HIDDEN];
   double m_v_b_season[FXAI_AI_MLP_HIDDEN];
   double m_m_w_trend[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_v_w_trend[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_b_trend[FXAI_AI_MLP_HIDDEN];
   double m_v_b_trend[FXAI_AI_MLP_HIDDEN];

   double m_m_wq[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_v_wq[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wk[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_v_wk[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wv[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_v_wv[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wo[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_AF_D_HEAD];
   double m_v_wo[FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_AF_D_HEAD];

   double m_m_w_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_v_w_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_b_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_b_gate[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_m_w_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_v_w_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_b_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_b_ff1[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_w_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_v_w_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_b_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_b_ff2[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_m_ln_pre_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_pre_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_ln_pre_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_pre_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_ln_post1_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_post1_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_ln_post1_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_post1_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_ln_post2_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_post2_g[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_m_ln_post2_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_v_ln_post2_b[FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_m_w_hgate[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_hgate[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_hgate[FXAI_AF_HORIZONS];
   double m_v_b_hgate[FXAI_AF_HORIZONS];

   double m_m_w_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_v_w_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_m_b_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT];
   double m_v_b_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT];

   double m_m_w_mu_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_mu_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_mu_h[FXAI_AF_HORIZONS];
   double m_v_b_mu_h[FXAI_AF_HORIZONS];

   double m_m_w_logv_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_logv_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_logv_h[FXAI_AF_HORIZONS];
   double m_v_b_logv_h[FXAI_AF_HORIZONS];

   double m_m_w_q25_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_q25_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_q25_h[FXAI_AF_HORIZONS];
   double m_v_b_q25_h[FXAI_AF_HORIZONS];

   double m_m_w_q75_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_q75_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_q75_h[FXAI_AF_HORIZONS];
   double m_v_b_q75_h[FXAI_AF_HORIZONS];

   // EMA shadow weights for inference stabilization.
   double m_sh_w_hgate[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_hgate[FXAI_AF_HORIZONS];
   double m_sh_w_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_cls_h[FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT];
   double m_sh_w_mu_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_mu_h[FXAI_AF_HORIZONS];
   double m_sh_w_logv_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_logv_h[FXAI_AF_HORIZONS];
   double m_sh_w_q25_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_q25_h[FXAI_AF_HORIZONS];
   double m_sh_w_q75_h[FXAI_AF_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_sh_b_q75_h[FXAI_AF_HORIZONS];

   // TBPTT buffers.
   int    m_train_len;
   double m_train_x[FXAI_AF_TBPTT][FXAI_AI_WEIGHTS];
   int    m_train_cls[FXAI_AF_TBPTT];
   double m_train_move[FXAI_AF_TBPTT];
   double m_train_cost[FXAI_AF_TBPTT];
   double m_train_w[FXAI_AF_TBPTT];

   // Forward caches.
   double m_cache_xn[FXAI_AF_TBPTT][FXAI_AI_WEIGHTS];
   double m_cache_embed[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_trend_raw[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_season_raw[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_trend[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_season[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_ma[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN][FXAI_AF_MA_KERNELS];
   double m_cache_mix_alpha[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN][FXAI_AF_MA_KERNELS];

   double m_cache_blk_in[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_cache_blk_pre[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_cache_blk_attn[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_cache_blk_res1[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_cache_blk_ff1[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];
   double m_cache_blk_out[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AI_MLP_HIDDEN];

   double m_cache_head_ctx[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_D_HEAD];
   int    m_cache_lag_idx[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_TOPK_LAGS];
   double m_cache_lag_w[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_TOPK_LAGS];
   double m_cache_lag_src[FXAI_AF_TBPTT][FXAI_AF_BLOCKS][FXAI_AF_HEADS][FXAI_AF_TOPK_LAGS * FXAI_AI_MLP_HIDDEN];

   double m_cache_final[FXAI_AF_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_probs_raw[FXAI_AF_TBPTT][FXAI_AF_CLASS_COUNT];
   double m_cache_h_alpha[FXAI_AF_TBPTT][FXAI_AF_HORIZONS];
   double m_cache_h_probs[FXAI_AF_TBPTT][FXAI_AF_HORIZONS][FXAI_AF_CLASS_COUNT];
   double m_cache_h_mu[FXAI_AF_TBPTT][FXAI_AF_HORIZONS];
   double m_cache_h_logv[FXAI_AF_TBPTT][FXAI_AF_HORIZONS];
   double m_cache_h_q25[FXAI_AF_TBPTT][FXAI_AF_HORIZONS];
   double m_cache_h_q75[FXAI_AF_TBPTT][FXAI_AF_HORIZONS];

