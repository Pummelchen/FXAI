   bool   m_initialized;
   int    m_step;
   int    m_adam_t;

   datetime m_last_m1_train_bar;

   // Rolling multivariate sequence state.
   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq[FXAI_CHR_SEQ][FXAI_AI_FEATURES];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // Class balance EMA.
   double m_cls_ema[FXAI_CHR_CLASS_COUNT];

   // Tokenization statistics and foundation vocabulary.
   double m_feat_mean[FXAI_AI_FEATURES];
   double m_feat_var[FXAI_AI_FEATURES];
   bool   m_feat_stats_ready;
   int    m_feat_stats_steps;

   double m_codebook[FXAI_CHR_CODEBOOK][FXAI_CHR_D_MODEL];
   double m_codebook_usage[FXAI_CHR_CODEBOOK];
   double m_codebook_gate[FXAI_AI_FEATURES];

   // Patch embedding + channel gating.
   double m_w_patch[FXAI_CHR_D_MODEL][FXAI_AI_FEATURES][FXAI_CHR_PATCH_LEN];
   double m_b_patch[FXAI_CHR_D_MODEL];
   double m_ch_gate[FXAI_AI_FEATURES];

   // Positional embedding per patch token.
   double m_pos[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];

   // Encoder stack.
   double m_wq[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_D_HEAD][FXAI_CHR_D_MODEL];
   double m_wk[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_D_HEAD][FXAI_CHR_D_MODEL];
   double m_wv[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_D_HEAD][FXAI_CHR_D_MODEL];
   double m_wo[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL][FXAI_CHR_D_MODEL];

   double m_wff1[FXAI_CHR_LAYERS][FXAI_CHR_D_FF][FXAI_CHR_D_MODEL];
   double m_bff1[FXAI_CHR_LAYERS][FXAI_CHR_D_FF];
   double m_wff2[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL][FXAI_CHR_D_FF];
   double m_bff2[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];

   double m_ln1_g[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
   double m_ln1_b[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
   double m_ln2_g[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];
   double m_ln2_b[FXAI_CHR_LAYERS][FXAI_CHR_D_MODEL];

   // Output heads.
   double m_w_cls[FXAI_CHR_CLASS_COUNT][FXAI_CHR_D_MODEL];
   double m_b_cls[FXAI_CHR_CLASS_COUNT];

   double m_w_mu[FXAI_CHR_D_MODEL];
   double m_b_mu;
   double m_w_logv[FXAI_CHR_D_MODEL];
   double m_b_logv;
   double m_w_q[FXAI_CHR_QUANTILES][FXAI_CHR_D_MODEL];
   double m_b_q[FXAI_CHR_QUANTILES];
   double m_w_mu_h[FXAI_CHR_HORIZONS][FXAI_CHR_D_MODEL];
   double m_b_mu_h[FXAI_CHR_HORIZONS];

   // Token-level language modeling head (Chronos-style discrete forecasting objective).
   double m_w_tok[FXAI_CHR_CODEBOOK][FXAI_CHR_D_MODEL];
   double m_b_tok[FXAI_CHR_CODEBOOK];

   // Retrieval memory bank to emulate foundation priors.
   double m_mem_k[FXAI_CHR_MEMORY][FXAI_CHR_D_MODEL];
   double m_mem_v[FXAI_CHR_MEMORY][FXAI_CHR_D_MODEL];
   double m_mem_usage[FXAI_CHR_MEMORY];
   int    m_mem_ptr;
   double m_w_mem_q[FXAI_CHR_D_MODEL][FXAI_CHR_D_MODEL];
   double m_w_mem_gate[FXAI_CHR_D_MODEL];
   double m_b_mem_gate;

   // Native 3-class calibration (vector scaling + session/regime context).
   double m_cal_vs_w[FXAI_CHR_CLASS_COUNT][FXAI_CHR_CLASS_COUNT];
   double m_cal_vs_b[FXAI_CHR_CLASS_COUNT];
   double m_cal_session_b[4][FXAI_CHR_CLASS_COUNT];
   double m_cal_regime_b[2][FXAI_CHR_CLASS_COUNT];
   double m_cal_iso_pos[FXAI_CHR_CLASS_COUNT][FXAI_CHR_CAL_BINS];
   double m_cal_iso_cnt[FXAI_CHR_CLASS_COUNT][FXAI_CHR_CAL_BINS];
   int    m_cal3_steps;

   // Stability: replay + teacher distillation.
   int    m_chr_replay_head;
   int    m_chr_replay_size;
   int    m_replay_pos[FXAI_CHR_REPLAY];
   double m_chr_replay_x[FXAI_CHR_REPLAY][FXAI_AI_WEIGHTS];
   int    m_replay_cls[FXAI_CHR_REPLAY];
   double m_chr_replay_move[FXAI_CHR_REPLAY];
   double m_chr_replay_cost[FXAI_CHR_REPLAY];
   double m_replay_w[FXAI_CHR_REPLAY];
   datetime m_chr_replay_time[FXAI_CHR_REPLAY];

   double m_t_w_cls[FXAI_CHR_CLASS_COUNT][FXAI_CHR_D_MODEL];
   double m_t_b_cls[FXAI_CHR_CLASS_COUNT];

   // Lightweight adaptive optimizer moments.
   double m_opt_m[16];
   double m_opt_v[16];

   // Training caches for token-level transformer backprop.
   int    m_cache_token_count;
   int    m_cache_token_target;
   double m_cache_x0[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
   double m_cache_layer_in[FXAI_CHR_LAYERS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
   double m_cache_q[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];
   double m_cache_k[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];
   double m_cache_v[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];
   double m_cache_att[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_MAX_PATCHES];
   double m_cache_ctx[FXAI_CHR_LAYERS][FXAI_CHR_HEADS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_HEAD];
   double m_cache_u[FXAI_CHR_LAYERS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
   double m_cache_ff1[FXAI_CHR_LAYERS][FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_FF];
   double m_cache_x_out[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
   double m_cache_patch_stat[FXAI_AI_FEATURES][FXAI_CHR_PATCH_LEN];
   int    m_cache_patch_start[FXAI_CHR_MAX_PATCHES];
   double m_cache_patch_code[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];
   double m_cache_patch_z[FXAI_CHR_MAX_PATCHES][FXAI_CHR_D_MODEL];

   // Validation gate metrics for reference-quality readiness.
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_nll_fast;
   double m_val_nll_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_val_ev_after_cost_fast;
   double m_val_ev_after_cost_slow;
   double m_ece_mass[FXAI_CHR_ECE_BINS];
   double m_ece_acc[FXAI_CHR_ECE_BINS];
   double m_ece_conf[FXAI_CHR_ECE_BINS];
   bool   m_reference_ready;

