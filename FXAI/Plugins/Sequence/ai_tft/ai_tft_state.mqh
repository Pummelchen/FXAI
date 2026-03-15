   bool   m_initialized;
   bool   m_shadow_ready;
   int    m_step;
   int    m_seen;
   int    m_adam_t;

   // Symbol/session-aware runtime state.
   uint   m_symbol_hash;

   // Input normalization.
   bool   m_norm_ready;
   int    m_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // History for inference context.
   int    m_hist_len;
   int    m_hist_ptr;
   double m_hist_x[FXAI_TFT_SEQ][FXAI_AI_WEIGHTS];

   // Train sequence for TBPTT.
   int    m_train_len;
   double m_train_x[FXAI_TFT_TBPTT][FXAI_AI_WEIGHTS];
   int    m_train_cls[FXAI_TFT_TBPTT];
   double m_train_move[FXAI_TFT_TBPTT];
   double m_train_cost[FXAI_TFT_TBPTT];
   double m_train_w[FXAI_TFT_TBPTT];

   // Per-feature variable selection network.
   double m_vsn_gate_w[FXAI_AI_WEIGHTS][FXAI_AI_WEIGHTS];
   double m_vsn_gate_b[FXAI_AI_WEIGHTS];
   double m_vsn_proj_w[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
   double m_vsn_proj_b[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];

   // Static context and encoder/decoder initialization transforms.
   double m_static_mask[FXAI_AI_WEIGHTS];
   double m_static_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_static_b[FXAI_AI_MLP_HIDDEN];

   double m_enc_h0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_enc_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_enc_c0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_enc_c0_b[FXAI_AI_MLP_HIDDEN];

   double m_dec_h0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_dec_h0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_dec_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_dec_c0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_dec_c0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_dec_c0_b[FXAI_AI_MLP_HIDDEN];

   // Encoder LSTM.
   double m_e_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_e_bi[FXAI_AI_MLP_HIDDEN];
   double m_e_bf[FXAI_AI_MLP_HIDDEN];
   double m_e_bo[FXAI_AI_MLP_HIDDEN];
   double m_e_bg[FXAI_AI_MLP_HIDDEN];

   // Decoder LSTM.
   double m_d_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_d_bi[FXAI_AI_MLP_HIDDEN];
   double m_d_bf[FXAI_AI_MLP_HIDDEN];
   double m_d_bo[FXAI_AI_MLP_HIDDEN];
   double m_d_bg[FXAI_AI_MLP_HIDDEN];

   // Multi-head temporal attention + relative bias.
   double m_wq[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wk[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wv[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wo[FXAI_TFT_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_TFT_D_HEAD];
   double m_rel_bias[FXAI_TFT_SEQ];

   // Feed-forward and residual branch.
   double m_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ff1_b[FXAI_AI_MLP_HIDDEN];
   double m_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ff2_b[FXAI_AI_MLP_HIDDEN];

   // Heads: 3-class + distributional move.
   double m_w_cls[FXAI_TFT_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_b_cls[FXAI_TFT_CLASS_COUNT];

   double m_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_b_q75;
   CFXAINativeQualityHeads m_quality_heads;

   // AdamW moments.
   double m_m_vsn_gate_w[FXAI_AI_WEIGHTS][FXAI_AI_WEIGHTS], m_v_vsn_gate_w[FXAI_AI_WEIGHTS][FXAI_AI_WEIGHTS];
   double m_m_vsn_gate_b[FXAI_AI_WEIGHTS], m_v_vsn_gate_b[FXAI_AI_WEIGHTS];
   double m_m_vsn_proj_w[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN], m_v_vsn_proj_w[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
   double m_m_vsn_proj_b[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN], m_v_vsn_proj_b[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];

   double m_m_static_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS], m_v_static_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_m_static_b[FXAI_AI_MLP_HIDDEN], m_v_static_b[FXAI_AI_MLP_HIDDEN];

   double m_m_enc_h0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_enc_h0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_enc_h0_b[FXAI_AI_MLP_HIDDEN], m_v_enc_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_m_enc_c0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_enc_c0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_enc_c0_b[FXAI_AI_MLP_HIDDEN], m_v_enc_c0_b[FXAI_AI_MLP_HIDDEN];

   double m_m_dec_h0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_dec_h0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_dec_h0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_dec_h0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_dec_h0_b[FXAI_AI_MLP_HIDDEN], m_v_dec_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_m_dec_c0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_dec_c0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_dec_c0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_dec_c0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_dec_c0_b[FXAI_AI_MLP_HIDDEN], m_v_dec_c0_b[FXAI_AI_MLP_HIDDEN];

   double m_m_e_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_e_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_e_bi[FXAI_AI_MLP_HIDDEN], m_v_e_bi[FXAI_AI_MLP_HIDDEN];
   double m_m_e_bf[FXAI_AI_MLP_HIDDEN], m_v_e_bf[FXAI_AI_MLP_HIDDEN];
   double m_m_e_bo[FXAI_AI_MLP_HIDDEN], m_v_e_bo[FXAI_AI_MLP_HIDDEN];
   double m_m_e_bg[FXAI_AI_MLP_HIDDEN], m_v_e_bg[FXAI_AI_MLP_HIDDEN];

   double m_m_d_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_d_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_d_bi[FXAI_AI_MLP_HIDDEN], m_v_d_bi[FXAI_AI_MLP_HIDDEN];
   double m_m_d_bf[FXAI_AI_MLP_HIDDEN], m_v_d_bf[FXAI_AI_MLP_HIDDEN];
   double m_m_d_bo[FXAI_AI_MLP_HIDDEN], m_v_d_bo[FXAI_AI_MLP_HIDDEN];
   double m_m_d_bg[FXAI_AI_MLP_HIDDEN], m_v_d_bg[FXAI_AI_MLP_HIDDEN];

   double m_m_wq[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN], m_v_wq[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wk[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN], m_v_wk[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wv[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN], m_v_wv[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_m_wo[FXAI_TFT_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_TFT_D_HEAD], m_v_wo[FXAI_TFT_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_TFT_D_HEAD];
   double m_m_rel_bias[FXAI_TFT_SEQ], m_v_rel_bias[FXAI_TFT_SEQ];

   double m_m_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_ff1_b[FXAI_AI_MLP_HIDDEN], m_v_ff1_b[FXAI_AI_MLP_HIDDEN];
   double m_m_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN], m_v_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_m_ff2_b[FXAI_AI_MLP_HIDDEN], m_v_ff2_b[FXAI_AI_MLP_HIDDEN];

   double m_m_w_cls[FXAI_TFT_CLASS_COUNT][FXAI_AI_MLP_HIDDEN], m_v_w_cls[FXAI_TFT_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_m_b_cls[FXAI_TFT_CLASS_COUNT], m_v_b_cls[FXAI_TFT_CLASS_COUNT];
   double m_m_w_mu[FXAI_AI_MLP_HIDDEN], m_v_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_m_b_mu, m_v_b_mu;
   double m_m_w_logv[FXAI_AI_MLP_HIDDEN], m_v_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_m_b_logv, m_v_b_logv;
   double m_m_w_q25[FXAI_AI_MLP_HIDDEN], m_v_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_m_b_q25, m_v_b_q25;
   double m_m_w_q75[FXAI_AI_MLP_HIDDEN], m_v_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_m_b_q75, m_v_b_q75;

   // EMA shadow parameters (for stable inference).
   double m_s_vsn_gate_w[FXAI_AI_WEIGHTS][FXAI_AI_WEIGHTS];
   double m_s_vsn_gate_b[FXAI_AI_WEIGHTS];
   double m_s_vsn_proj_w[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
   double m_s_vsn_proj_b[FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];

   double m_s_static_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_s_static_b[FXAI_AI_MLP_HIDDEN];

   double m_s_enc_h0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_enc_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_s_enc_c0_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_enc_c0_b[FXAI_AI_MLP_HIDDEN];

   double m_s_dec_h0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_dec_h0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_dec_h0_b[FXAI_AI_MLP_HIDDEN];
   double m_s_dec_c0_s_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_dec_c0_e_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_dec_c0_b[FXAI_AI_MLP_HIDDEN];

   double m_s_e_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_e_bi[FXAI_AI_MLP_HIDDEN];
   double m_s_e_bf[FXAI_AI_MLP_HIDDEN];
   double m_s_e_bo[FXAI_AI_MLP_HIDDEN];
   double m_s_e_bg[FXAI_AI_MLP_HIDDEN];

   double m_s_d_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_d_bi[FXAI_AI_MLP_HIDDEN];
   double m_s_d_bf[FXAI_AI_MLP_HIDDEN];
   double m_s_d_bo[FXAI_AI_MLP_HIDDEN];
   double m_s_d_bg[FXAI_AI_MLP_HIDDEN];

   double m_s_wq[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_s_wk[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_s_wv[FXAI_TFT_HEADS][FXAI_TFT_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_s_wo[FXAI_TFT_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_TFT_D_HEAD];
   double m_s_rel_bias[FXAI_TFT_SEQ];

   double m_s_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_ff1_b[FXAI_AI_MLP_HIDDEN];
   double m_s_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_s_ff2_b[FXAI_AI_MLP_HIDDEN];

   double m_s_w_cls[FXAI_TFT_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_s_b_cls[FXAI_TFT_CLASS_COUNT];
   double m_s_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_s_b_mu;
   double m_s_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_s_b_logv;
   double m_s_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_s_b_q25;
   double m_s_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_s_b_q75;

   // Session calibration and walk-forward threshold optimization.
   double m_sess_a[FXAI_TFT_SESSIONS];
   double m_sess_b[FXAI_TFT_SESSIONS];
   int    m_sess_steps[FXAI_TFT_SESSIONS];
   bool   m_sess_ready[FXAI_TFT_SESSIONS];

   double m_thr_buy;
   double m_thr_sell;
   double m_thr_skip;

   int    m_wf_len;
   int    m_wf_ptr;
   double m_wf_pup[FXAI_TFT_WF];
   double m_wf_pskip[FXAI_TFT_WF];
   double m_wf_move[FXAI_TFT_WF];
   double m_wf_cost[FXAI_TFT_WF];
   int    m_wf_cls[FXAI_TFT_WF];
   int    m_wf_sess[FXAI_TFT_WF];

   // Training caches.
   double c_xn[FXAI_TFT_TBPTT][FXAI_AI_WEIGHTS];
   double c_alpha[FXAI_TFT_TBPTT][FXAI_AI_WEIGHTS];
   double c_feat[FXAI_TFT_TBPTT][FXAI_AI_WEIGHTS][FXAI_AI_MLP_HIDDEN];
   double c_emb[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];

   double c_static[FXAI_AI_MLP_HIDDEN];
   double c_enc_h0[FXAI_AI_MLP_HIDDEN];
   double c_enc_c0[FXAI_AI_MLP_HIDDEN];
   double c_dec_h0[FXAI_AI_MLP_HIDDEN];
   double c_dec_c0[FXAI_AI_MLP_HIDDEN];

   double c_e_h_prev[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_c_prev[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_i[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_f[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_o[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_g[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_h[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_e_c[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];

   double c_d_h_prev[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_c_prev[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_i[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_f[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_o[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_g[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_h[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_d_c[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];

   double c_attn_q[FXAI_TFT_TBPTT][FXAI_TFT_HEADS][FXAI_TFT_D_HEAD];
   double c_attn_k[FXAI_TFT_TBPTT][FXAI_TFT_HEADS][FXAI_TFT_TBPTT][FXAI_TFT_D_HEAD];
   double c_attn_v[FXAI_TFT_TBPTT][FXAI_TFT_HEADS][FXAI_TFT_TBPTT][FXAI_TFT_D_HEAD];
   double c_attn_w[FXAI_TFT_TBPTT][FXAI_TFT_HEADS][FXAI_TFT_TBPTT];
   double c_attn_ctx[FXAI_TFT_TBPTT][FXAI_TFT_HEADS][FXAI_TFT_D_HEAD];
   double c_attn_out[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];

   double c_pre[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_ff1_raw[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_ff1[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_ff1_mask[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_ff2[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];
   double c_stoch_scale[FXAI_TFT_TBPTT];
   double c_final[FXAI_TFT_TBPTT][FXAI_AI_MLP_HIDDEN];

   double c_logits[FXAI_TFT_TBPTT][FXAI_TFT_CLASS_COUNT];
   double c_probs[FXAI_TFT_TBPTT][FXAI_TFT_CLASS_COUNT];
   double c_mu[FXAI_TFT_TBPTT];
   double c_logv[FXAI_TFT_TBPTT];
   double c_q25[FXAI_TFT_TBPTT];
   double c_q75[FXAI_TFT_TBPTT];

