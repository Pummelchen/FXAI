#ifndef __FX6_AI_TFT_MQH__
#define __FX6_AI_TFT_MQH__

#include "..\plugin_base.mqh"

#define FX6_TFT_SEQ 48
#define FX6_TFT_TBPTT 16
#define FX6_TFT_HEADS 2
#define FX6_TFT_D_HEAD (FX6_AI_MLP_HIDDEN / FX6_TFT_HEADS)
#define FX6_TFT_CLASS_COUNT 3
#define FX6_TFT_WF 256
#define FX6_TFT_SESSIONS 4

#define FX6_TFT_SELL 0
#define FX6_TFT_BUY  1
#define FX6_TFT_SKIP 2

class CFX6AITFT : public CFX6AIPlugin
{
private:
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
   double m_x_mean[FX6_AI_WEIGHTS];
   double m_x_var[FX6_AI_WEIGHTS];

   // History for inference context.
   int    m_hist_len;
   int    m_hist_ptr;
   double m_hist_x[FX6_TFT_SEQ][FX6_AI_WEIGHTS];

   // Train sequence for TBPTT.
   int    m_train_len;
   double m_train_x[FX6_TFT_TBPTT][FX6_AI_WEIGHTS];
   int    m_train_cls[FX6_TFT_TBPTT];
   double m_train_move[FX6_TFT_TBPTT];
   double m_train_cost[FX6_TFT_TBPTT];
   double m_train_w[FX6_TFT_TBPTT];

   // Per-feature variable selection network.
   double m_vsn_gate_w[FX6_AI_WEIGHTS][FX6_AI_WEIGHTS];
   double m_vsn_gate_b[FX6_AI_WEIGHTS];
   double m_vsn_proj_w[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];
   double m_vsn_proj_b[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];

   // Static context and encoder/decoder initialization transforms.
   double m_static_mask[FX6_AI_WEIGHTS];
   double m_static_w[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_static_b[FX6_AI_MLP_HIDDEN];

   double m_enc_h0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_enc_h0_b[FX6_AI_MLP_HIDDEN];
   double m_enc_c0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_enc_c0_b[FX6_AI_MLP_HIDDEN];

   double m_dec_h0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_dec_h0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_dec_h0_b[FX6_AI_MLP_HIDDEN];
   double m_dec_c0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_dec_c0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_dec_c0_b[FX6_AI_MLP_HIDDEN];

   // Encoder LSTM.
   double m_e_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_e_bi[FX6_AI_MLP_HIDDEN];
   double m_e_bf[FX6_AI_MLP_HIDDEN];
   double m_e_bo[FX6_AI_MLP_HIDDEN];
   double m_e_bg[FX6_AI_MLP_HIDDEN];

   // Decoder LSTM.
   double m_d_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_d_bi[FX6_AI_MLP_HIDDEN];
   double m_d_bf[FX6_AI_MLP_HIDDEN];
   double m_d_bo[FX6_AI_MLP_HIDDEN];
   double m_d_bg[FX6_AI_MLP_HIDDEN];

   // Multi-head temporal attention + relative bias.
   double m_wq[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wk[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wv[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wo[FX6_TFT_HEADS][FX6_AI_MLP_HIDDEN][FX6_TFT_D_HEAD];
   double m_rel_bias[FX6_TFT_SEQ];

   // Feed-forward and residual branch.
   double m_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ff1_b[FX6_AI_MLP_HIDDEN];
   double m_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ff2_b[FX6_AI_MLP_HIDDEN];

   // Heads: 3-class + distributional move.
   double m_w_cls[FX6_TFT_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_TFT_CLASS_COUNT];

   double m_w_mu[FX6_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN];
   double m_b_q75;

   // AdamW moments.
   double m_m_vsn_gate_w[FX6_AI_WEIGHTS][FX6_AI_WEIGHTS], m_v_vsn_gate_w[FX6_AI_WEIGHTS][FX6_AI_WEIGHTS];
   double m_m_vsn_gate_b[FX6_AI_WEIGHTS], m_v_vsn_gate_b[FX6_AI_WEIGHTS];
   double m_m_vsn_proj_w[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN], m_v_vsn_proj_w[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];
   double m_m_vsn_proj_b[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN], m_v_vsn_proj_b[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];

   double m_m_static_w[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS], m_v_static_w[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_m_static_b[FX6_AI_MLP_HIDDEN], m_v_static_b[FX6_AI_MLP_HIDDEN];

   double m_m_enc_h0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_enc_h0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_enc_h0_b[FX6_AI_MLP_HIDDEN], m_v_enc_h0_b[FX6_AI_MLP_HIDDEN];
   double m_m_enc_c0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_enc_c0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_enc_c0_b[FX6_AI_MLP_HIDDEN], m_v_enc_c0_b[FX6_AI_MLP_HIDDEN];

   double m_m_dec_h0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_dec_h0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_dec_h0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_dec_h0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_dec_h0_b[FX6_AI_MLP_HIDDEN], m_v_dec_h0_b[FX6_AI_MLP_HIDDEN];
   double m_m_dec_c0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_dec_c0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_dec_c0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_dec_c0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_dec_c0_b[FX6_AI_MLP_HIDDEN], m_v_dec_c0_b[FX6_AI_MLP_HIDDEN];

   double m_m_e_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_e_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_e_bi[FX6_AI_MLP_HIDDEN], m_v_e_bi[FX6_AI_MLP_HIDDEN];
   double m_m_e_bf[FX6_AI_MLP_HIDDEN], m_v_e_bf[FX6_AI_MLP_HIDDEN];
   double m_m_e_bo[FX6_AI_MLP_HIDDEN], m_v_e_bo[FX6_AI_MLP_HIDDEN];
   double m_m_e_bg[FX6_AI_MLP_HIDDEN], m_v_e_bg[FX6_AI_MLP_HIDDEN];

   double m_m_d_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_d_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_d_bi[FX6_AI_MLP_HIDDEN], m_v_d_bi[FX6_AI_MLP_HIDDEN];
   double m_m_d_bf[FX6_AI_MLP_HIDDEN], m_v_d_bf[FX6_AI_MLP_HIDDEN];
   double m_m_d_bo[FX6_AI_MLP_HIDDEN], m_v_d_bo[FX6_AI_MLP_HIDDEN];
   double m_m_d_bg[FX6_AI_MLP_HIDDEN], m_v_d_bg[FX6_AI_MLP_HIDDEN];

   double m_m_wq[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN], m_v_wq[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_m_wk[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN], m_v_wk[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_m_wv[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN], m_v_wv[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_m_wo[FX6_TFT_HEADS][FX6_AI_MLP_HIDDEN][FX6_TFT_D_HEAD], m_v_wo[FX6_TFT_HEADS][FX6_AI_MLP_HIDDEN][FX6_TFT_D_HEAD];
   double m_m_rel_bias[FX6_TFT_SEQ], m_v_rel_bias[FX6_TFT_SEQ];

   double m_m_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_ff1_b[FX6_AI_MLP_HIDDEN], m_v_ff1_b[FX6_AI_MLP_HIDDEN];
   double m_m_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], m_v_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_ff2_b[FX6_AI_MLP_HIDDEN], m_v_ff2_b[FX6_AI_MLP_HIDDEN];

   double m_m_w_cls[FX6_TFT_CLASS_COUNT][FX6_AI_MLP_HIDDEN], m_v_w_cls[FX6_TFT_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_m_b_cls[FX6_TFT_CLASS_COUNT], m_v_b_cls[FX6_TFT_CLASS_COUNT];
   double m_m_w_mu[FX6_AI_MLP_HIDDEN], m_v_w_mu[FX6_AI_MLP_HIDDEN];
   double m_m_b_mu, m_v_b_mu;
   double m_m_w_logv[FX6_AI_MLP_HIDDEN], m_v_w_logv[FX6_AI_MLP_HIDDEN];
   double m_m_b_logv, m_v_b_logv;
   double m_m_w_q25[FX6_AI_MLP_HIDDEN], m_v_w_q25[FX6_AI_MLP_HIDDEN];
   double m_m_b_q25, m_v_b_q25;
   double m_m_w_q75[FX6_AI_MLP_HIDDEN], m_v_w_q75[FX6_AI_MLP_HIDDEN];
   double m_m_b_q75, m_v_b_q75;

   // EMA shadow parameters (for stable inference).
   double m_s_vsn_gate_w[FX6_AI_WEIGHTS][FX6_AI_WEIGHTS];
   double m_s_vsn_gate_b[FX6_AI_WEIGHTS];
   double m_s_vsn_proj_w[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];
   double m_s_vsn_proj_b[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];

   double m_s_static_w[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_s_static_b[FX6_AI_MLP_HIDDEN];

   double m_s_enc_h0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_enc_h0_b[FX6_AI_MLP_HIDDEN];
   double m_s_enc_c0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_enc_c0_b[FX6_AI_MLP_HIDDEN];

   double m_s_dec_h0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_dec_h0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_dec_h0_b[FX6_AI_MLP_HIDDEN];
   double m_s_dec_c0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_dec_c0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_dec_c0_b[FX6_AI_MLP_HIDDEN];

   double m_s_e_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_e_bi[FX6_AI_MLP_HIDDEN];
   double m_s_e_bf[FX6_AI_MLP_HIDDEN];
   double m_s_e_bo[FX6_AI_MLP_HIDDEN];
   double m_s_e_bg[FX6_AI_MLP_HIDDEN];

   double m_s_d_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_d_bi[FX6_AI_MLP_HIDDEN];
   double m_s_d_bf[FX6_AI_MLP_HIDDEN];
   double m_s_d_bo[FX6_AI_MLP_HIDDEN];
   double m_s_d_bg[FX6_AI_MLP_HIDDEN];

   double m_s_wq[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_s_wk[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_s_wv[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_s_wo[FX6_TFT_HEADS][FX6_AI_MLP_HIDDEN][FX6_TFT_D_HEAD];
   double m_s_rel_bias[FX6_TFT_SEQ];

   double m_s_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_ff1_b[FX6_AI_MLP_HIDDEN];
   double m_s_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_s_ff2_b[FX6_AI_MLP_HIDDEN];

   double m_s_w_cls[FX6_TFT_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_s_b_cls[FX6_TFT_CLASS_COUNT];
   double m_s_w_mu[FX6_AI_MLP_HIDDEN];
   double m_s_b_mu;
   double m_s_w_logv[FX6_AI_MLP_HIDDEN];
   double m_s_b_logv;
   double m_s_w_q25[FX6_AI_MLP_HIDDEN];
   double m_s_b_q25;
   double m_s_w_q75[FX6_AI_MLP_HIDDEN];
   double m_s_b_q75;

   // Session calibration and walk-forward threshold optimization.
   double m_sess_a[FX6_TFT_SESSIONS];
   double m_sess_b[FX6_TFT_SESSIONS];
   int    m_sess_steps[FX6_TFT_SESSIONS];
   bool   m_sess_ready[FX6_TFT_SESSIONS];

   double m_thr_buy;
   double m_thr_sell;
   double m_thr_skip;

   int    m_wf_len;
   int    m_wf_ptr;
   double m_wf_pup[FX6_TFT_WF];
   double m_wf_pskip[FX6_TFT_WF];
   double m_wf_move[FX6_TFT_WF];
   double m_wf_cost[FX6_TFT_WF];
   int    m_wf_cls[FX6_TFT_WF];
   int    m_wf_sess[FX6_TFT_WF];

   // Training caches.
   double c_xn[FX6_TFT_TBPTT][FX6_AI_WEIGHTS];
   double c_alpha[FX6_TFT_TBPTT][FX6_AI_WEIGHTS];
   double c_feat[FX6_TFT_TBPTT][FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];
   double c_emb[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];

   double c_static[FX6_AI_MLP_HIDDEN];
   double c_enc_h0[FX6_AI_MLP_HIDDEN];
   double c_enc_c0[FX6_AI_MLP_HIDDEN];
   double c_dec_h0[FX6_AI_MLP_HIDDEN];
   double c_dec_c0[FX6_AI_MLP_HIDDEN];

   double c_e_h_prev[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_c_prev[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_i[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_f[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_o[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_g[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_h[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_e_c[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];

   double c_d_h_prev[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_c_prev[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_i[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_f[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_o[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_g[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_h[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_d_c[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];

   double c_attn_q[FX6_TFT_TBPTT][FX6_TFT_HEADS][FX6_TFT_D_HEAD];
   double c_attn_k[FX6_TFT_TBPTT][FX6_TFT_HEADS][FX6_TFT_TBPTT][FX6_TFT_D_HEAD];
   double c_attn_v[FX6_TFT_TBPTT][FX6_TFT_HEADS][FX6_TFT_TBPTT][FX6_TFT_D_HEAD];
   double c_attn_w[FX6_TFT_TBPTT][FX6_TFT_HEADS][FX6_TFT_TBPTT];
   double c_attn_ctx[FX6_TFT_TBPTT][FX6_TFT_HEADS][FX6_TFT_D_HEAD];
   double c_attn_out[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];

   double c_pre[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_ff1_raw[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_ff1[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_ff1_mask[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_ff2[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
   double c_stoch_scale[FX6_TFT_TBPTT];
   double c_final[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];

   double c_logits[FX6_TFT_TBPTT][FX6_TFT_CLASS_COUNT];
   double c_probs[FX6_TFT_TBPTT][FX6_TFT_CLASS_COUNT];
   double c_mu[FX6_TFT_TBPTT];
   double c_logv[FX6_TFT_TBPTT];
   double c_q25[FX6_TFT_TBPTT];
   double c_q75[FX6_TFT_TBPTT];

   int SessionBucket(const datetime ts) const
   {
      MqlDateTime md;
      TimeToStruct(ts, md);
      int h = md.hour;
      if(h >= 6 && h <= 12) return 1;  // Europe
      if(h >= 13 && h <= 20) return 2; // US
      if(h >= 21 || h <= 2) return 0;  // Asia
      return 3;                        // transition/off
   }

   uint SymbolHash(const string s) const
   {
      uint h = 2166136261U;
      int n = StringLen(s);
      for(int i=0; i<n; i++)
      {
         uint c = (uint)StringGetCharacter(s, i);
         h ^= c;
         h *= 16777619U;
      }
      return h;
   }

   double DropMask(const int idx,
                   const int salt,
                   const double rate,
                   const bool training) const
   {
      if(!training) return 1.0;
      if(rate <= 1e-9) return 1.0;

      uint h = (uint)(m_step * 2654435761U);
      h ^= (uint)((idx + 3) * 2246822519U);
      h ^= (uint)((salt + 11) * 3266489917U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < rate) return 0.0;
      return 1.0 / (1.0 - rate);
   }

   double StochScale(const int idx,
                     const bool training) const
   {
      if(!training) return 1.0;
      const double drop = 0.12;
      uint h = (uint)((m_step + 17) * 40503U);
      h ^= (uint)((idx + 19) * 2654435761U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < drop) return 0.0;
      return 1.0 / (1.0 - drop);
   }

   double ScheduledLR(const FX6AIHyperParams &hp) const
   {
      double base = FX6_Clamp(hp.lr, 0.00002, 0.25000);
      double st = (double)MathMax(m_step, 1);

      double warm = FX6_Clamp(st / 128.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0012 * MathMax(0.0, st - 128.0));

      double period = 2048.0;
      double phase = MathMod(st, period) / period;
      double cosine = 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
      double floor = 0.20 + 0.80 * cosine;

      double lr = base * warm * invsqrt * floor;
      return FX6_Clamp(lr, 0.00001, 0.05000);
   }

   void AdamWStep(double &p,
                  double &m,
                  double &v,
                  const double g,
                  const double lr,
                  const double wd)
   {
      const double b1 = 0.90;
      const double b2 = 0.999;
      const double eps = 1e-8;

      double grad = FX6_ClipSym(g, 10.0);
      m = b1 * m + (1.0 - b1) * grad;
      v = b2 * v + (1.0 - b2) * grad * grad;

      double t = (double)MathMax(m_adam_t, 1);
      double mh = m / (1.0 - MathPow(b1, t));
      double vh = v / (1.0 - MathPow(b2, t));

      p -= lr * (mh / (MathSqrt(vh) + eps));
      if(wd > 0.0)
         p -= lr * wd * p;
   }

   double HuberGrad(const double err,
                    const double delta) const
   {
      double d = (delta > 0.0 ? delta : 1.0);
      if(err > d) return d;
      if(err < -d) return -d;
      return err;
   }

   double PinballHuberGrad(const double target,
                           const double q,
                           const double tau,
                           const double kappa) const
   {
      double k = (kappa > 1e-6 ? kappa : 1.0);
      double u = target - q;
      if(u >= k) return -tau;
      if(u <= -k) return (1.0 - tau);

      double g = (0.5 - tau) - (u / (2.0 * k));
      return FX6_Clamp(g, -tau, 1.0 - tau);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost_points,
                      const double sample_w) const
   {
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);

      if(cls == FX6_TFT_SKIP)
      {
         if(edge <= 0.0) return FX6_Clamp(sw * 1.40, 0.25, 6.00);
         return FX6_Clamp(sw * 0.75, 0.25, 6.00);
      }

      if(edge <= 0.0) return FX6_Clamp(sw * 0.65, 0.25, 6.00);
      return FX6_Clamp(sw * (1.0 + 0.05 * MathMin(edge, 20.0)), 0.25, 6.00);
   }

   double MoveWeight(const double move_points,
                     const double cost_points,
                     const double sample_w) const
   {
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double ew = FX6_MoveEdgeWeight(move_points, cost_points);
      double edge = MathMax(0.0, MathAbs(move_points) - MathMax(cost_points, 0.0));
      return FX6_Clamp(sw * ew * (1.0 + 0.05 * MathMin(edge, 20.0)), 0.25, 8.00);
   }

   int MapClass(const int y,
                const double &x[],
                const double move_points) const
   {
      if(y == FX6_TFT_SELL || y == FX6_TFT_BUY || y == FX6_TFT_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FX6_TFT_SKIP;

      if(y > 0) return FX6_TFT_BUY;
      if(y == 0) return FX6_TFT_SELL;
      return (move_points >= 0.0 ? FX6_TFT_BUY : FX6_TFT_SELL);
   }

   void SoftmaxN(const double &logits[],
                 const int n,
                 double &probs[]) const
   {
      double m = logits[0];
      for(int i=1; i<n; i++)
         if(logits[i] > m) m = logits[i];

      double den = 0.0;
      for(int i=0; i<n; i++)
      {
         probs[i] = MathExp(FX6_ClipSym(logits[i] - m, 30.0));
         den += probs[i];
      }
      if(den <= 0.0) den = 1.0;
      for(int i=0; i<n; i++)
         probs[i] /= den;
   }

   void ResetNorm(void)
   {
      m_norm_ready = false;
      m_norm_steps = 0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void ResetHistory(void)
   {
      m_hist_len = 0;
      m_hist_ptr = 0;
      for(int t=0; t<FX6_TFT_SEQ; t++)
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_hist_x[t][i] = 0.0;
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FX6_TFT_TBPTT; t++)
      {
         m_train_cls[t] = FX6_TFT_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_train_x[t][i] = 0.0;
      }
   }

   void ResetWalkForward(void)
   {
      m_wf_len = 0;
      m_wf_ptr = 0;
      for(int i=0; i<FX6_TFT_WF; i++)
      {
         m_wf_pup[i] = 0.5;
         m_wf_pskip[i] = 0.5;
         m_wf_move[i] = 0.0;
         m_wf_cost[i] = 0.0;
         m_wf_cls[i] = FX6_TFT_SKIP;
         m_wf_sess[i] = 3;
      }
   }

   void ResetSessionCalibration(void)
   {
      for(int s=0; s<FX6_TFT_SESSIONS; s++)
      {
         m_sess_a[s] = 1.0;
         m_sess_b[s] = 0.0;
         m_sess_steps[s] = 0;
         m_sess_ready[s] = false;
      }
   }

   void InitStaticMask(void)
   {
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_static_mask[i] = 0.0;

      m_static_mask[0] = 1.0;
      // feature index = i-1; mark more stable/context features.
      m_static_mask[5] = 1.0;
      m_static_mask[7] = 1.0;
      m_static_mask[11] = 1.0;
      m_static_mask[12] = 1.0;
      m_static_mask[13] = 1.0;
      m_static_mask[14] = 1.0;
      m_static_mask[15] = 1.0;
   }

   void ZeroMoments(void)
   {
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_m_vsn_gate_b[i] = 0.0;
         m_v_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
         {
            m_m_vsn_gate_w[i][j] = 0.0;
            m_v_vsn_gate_w[i][j] = 0.0;
         }
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_m_vsn_proj_w[i][h] = 0.0;
            m_v_vsn_proj_w[i][h] = 0.0;
            m_m_vsn_proj_b[i][h] = 0.0;
            m_v_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_m_static_b[h] = 0.0;
         m_v_static_b[h] = 0.0;

         m_m_enc_h0_b[h] = 0.0; m_v_enc_h0_b[h] = 0.0;
         m_m_enc_c0_b[h] = 0.0; m_v_enc_c0_b[h] = 0.0;
         m_m_dec_h0_b[h] = 0.0; m_v_dec_h0_b[h] = 0.0;
         m_m_dec_c0_b[h] = 0.0; m_v_dec_c0_b[h] = 0.0;

         m_m_e_bi[h] = 0.0; m_v_e_bi[h] = 0.0;
         m_m_e_bf[h] = 0.0; m_v_e_bf[h] = 0.0;
         m_m_e_bo[h] = 0.0; m_v_e_bo[h] = 0.0;
         m_m_e_bg[h] = 0.0; m_v_e_bg[h] = 0.0;

         m_m_d_bi[h] = 0.0; m_v_d_bi[h] = 0.0;
         m_m_d_bf[h] = 0.0; m_v_d_bf[h] = 0.0;
         m_m_d_bo[h] = 0.0; m_v_d_bo[h] = 0.0;
         m_m_d_bg[h] = 0.0; m_v_d_bg[h] = 0.0;

         m_m_ff1_b[h] = 0.0; m_v_ff1_b[h] = 0.0;
         m_m_ff2_b[h] = 0.0; m_v_ff2_b[h] = 0.0;

         m_m_w_mu[h] = 0.0; m_v_w_mu[h] = 0.0;
         m_m_w_logv[h] = 0.0; m_v_w_logv[h] = 0.0;
         m_m_w_q25[h] = 0.0; m_v_w_q25[h] = 0.0;
         m_m_w_q75[h] = 0.0; m_v_w_q75[h] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_m_static_w[h][i] = 0.0;
            m_v_static_w[h][i] = 0.0;
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_m_enc_h0_w[h][j] = 0.0; m_v_enc_h0_w[h][j] = 0.0;
            m_m_enc_c0_w[h][j] = 0.0; m_v_enc_c0_w[h][j] = 0.0;
            m_m_dec_h0_s_w[h][j] = 0.0; m_v_dec_h0_s_w[h][j] = 0.0;
            m_m_dec_h0_e_w[h][j] = 0.0; m_v_dec_h0_e_w[h][j] = 0.0;
            m_m_dec_c0_s_w[h][j] = 0.0; m_v_dec_c0_s_w[h][j] = 0.0;
            m_m_dec_c0_e_w[h][j] = 0.0; m_v_dec_c0_e_w[h][j] = 0.0;

            m_m_e_wi_x[h][j] = 0.0; m_v_e_wi_x[h][j] = 0.0;
            m_m_e_wf_x[h][j] = 0.0; m_v_e_wf_x[h][j] = 0.0;
            m_m_e_wo_x[h][j] = 0.0; m_v_e_wo_x[h][j] = 0.0;
            m_m_e_wg_x[h][j] = 0.0; m_v_e_wg_x[h][j] = 0.0;
            m_m_e_wi_h[h][j] = 0.0; m_v_e_wi_h[h][j] = 0.0;
            m_m_e_wf_h[h][j] = 0.0; m_v_e_wf_h[h][j] = 0.0;
            m_m_e_wo_h[h][j] = 0.0; m_v_e_wo_h[h][j] = 0.0;
            m_m_e_wg_h[h][j] = 0.0; m_v_e_wg_h[h][j] = 0.0;

            m_m_d_wi_x[h][j] = 0.0; m_v_d_wi_x[h][j] = 0.0;
            m_m_d_wf_x[h][j] = 0.0; m_v_d_wf_x[h][j] = 0.0;
            m_m_d_wo_x[h][j] = 0.0; m_v_d_wo_x[h][j] = 0.0;
            m_m_d_wg_x[h][j] = 0.0; m_v_d_wg_x[h][j] = 0.0;
            m_m_d_wi_h[h][j] = 0.0; m_v_d_wi_h[h][j] = 0.0;
            m_m_d_wf_h[h][j] = 0.0; m_v_d_wf_h[h][j] = 0.0;
            m_m_d_wo_h[h][j] = 0.0; m_v_d_wo_h[h][j] = 0.0;
            m_m_d_wg_h[h][j] = 0.0; m_v_d_wg_h[h][j] = 0.0;

            m_m_ff1_w[h][j] = 0.0; m_v_ff1_w[h][j] = 0.0;
            m_m_ff2_w[h][j] = 0.0; m_v_ff2_w[h][j] = 0.0;
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         {
            m_m_w_cls[c][h] = 0.0;
            m_v_w_cls[c][h] = 0.0;
         }
      }

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
      {
         m_m_b_cls[c] = 0.0;
         m_v_b_cls[c] = 0.0;
      }

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               m_m_wq[hd][d][h] = 0.0; m_v_wq[hd][d][h] = 0.0;
               m_m_wk[hd][d][h] = 0.0; m_v_wk[hd][d][h] = 0.0;
               m_m_wv[hd][d][h] = 0.0; m_v_wv[hd][d][h] = 0.0;
            }
            for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
            {
               m_m_wo[hd][o][d] = 0.0;
               m_v_wo[hd][o][d] = 0.0;
            }
         }
      }

      for(int i=0; i<FX6_TFT_SEQ; i++)
      {
         m_m_rel_bias[i] = 0.0;
         m_v_rel_bias[i] = 0.0;
      }

      m_m_b_mu = 0.0; m_v_b_mu = 0.0;
      m_m_b_logv = 0.0; m_v_b_logv = 0.0;
      m_m_b_q25 = 0.0; m_v_b_q25 = 0.0;
      m_m_b_q75 = 0.0; m_v_b_q75 = 0.0;
   }

   void InitParams(void)
   {
      InitStaticMask();
      ResetNorm();
      ResetHistory();
      ResetTrainBuffer();
      ResetWalkForward();
      ResetSessionCalibration();
      m_step = 0;
      m_seen = 0;
      m_adam_t = 0;

      m_thr_buy = 0.62;
      m_thr_sell = 0.38;
      m_thr_skip = 0.58;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
         {
            double s = (double)((i + 1) * (j + 3));
            m_vsn_gate_w[i][j] = 0.05 * MathSin(0.63 * s);
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double s2 = (double)((i + 2) * (h + 1));
            m_vsn_proj_w[i][h] = 0.06 * MathCos(0.71 * s2);
            m_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_static_b[h] = 0.0;
         m_enc_h0_b[h] = 0.0;
         m_enc_c0_b[h] = 0.0;
         m_dec_h0_b[h] = 0.0;
         m_dec_c0_b[h] = 0.0;

         m_e_bi[h] = 0.0;
         m_e_bf[h] = 0.40;
         m_e_bo[h] = 0.0;
         m_e_bg[h] = 0.0;

         m_d_bi[h] = 0.0;
         m_d_bf[h] = 0.35;
         m_d_bo[h] = 0.0;
         m_d_bg[h] = 0.0;

         m_ff1_b[h] = 0.0;
         m_ff2_b[h] = 0.0;

         m_w_mu[h] = 0.04 * MathSin((double)(h + 1) * 0.91);
         m_w_logv[h] = 0.03 * MathCos((double)(h + 1) * 0.99);
         m_w_q25[h] = 0.03 * MathSin((double)(h + 1) * 1.07);
         m_w_q75[h] = 0.03 * MathCos((double)(h + 1) * 1.13);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double s = (double)((h + 1) * (i + 2));
            m_static_w[h][i] = 0.04 * MathSin(0.67 * s);
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            double s2 = (double)((h + 2) * (j + 3));
            m_enc_h0_w[h][j] = 0.05 * MathSin(0.59 * s2);
            m_enc_c0_w[h][j] = 0.05 * MathCos(0.61 * s2);
            m_dec_h0_s_w[h][j] = 0.05 * MathCos(0.63 * s2);
            m_dec_h0_e_w[h][j] = 0.05 * MathSin(0.65 * s2);
            m_dec_c0_s_w[h][j] = 0.05 * MathSin(0.69 * s2);
            m_dec_c0_e_w[h][j] = 0.05 * MathCos(0.73 * s2);

            m_e_wi_x[h][j] = 0.05 * MathSin(0.77 * s2);
            m_e_wf_x[h][j] = 0.05 * MathCos(0.79 * s2);
            m_e_wo_x[h][j] = 0.05 * MathSin(0.81 * s2);
            m_e_wg_x[h][j] = 0.05 * MathCos(0.83 * s2);

            m_e_wi_h[h][j] = 0.04 * MathSin(0.85 * s2);
            m_e_wf_h[h][j] = 0.04 * MathCos(0.87 * s2);
            m_e_wo_h[h][j] = 0.04 * MathSin(0.89 * s2);
            m_e_wg_h[h][j] = 0.04 * MathCos(0.91 * s2);

            m_d_wi_x[h][j] = 0.05 * MathCos(0.93 * s2);
            m_d_wf_x[h][j] = 0.05 * MathSin(0.95 * s2);
            m_d_wo_x[h][j] = 0.05 * MathCos(0.97 * s2);
            m_d_wg_x[h][j] = 0.05 * MathSin(0.99 * s2);

            m_d_wi_h[h][j] = 0.04 * MathCos(1.01 * s2);
            m_d_wf_h[h][j] = 0.04 * MathSin(1.03 * s2);
            m_d_wo_h[h][j] = 0.04 * MathCos(1.07 * s2);
            m_d_wg_h[h][j] = 0.04 * MathSin(1.09 * s2);

            m_ff1_w[h][j] = 0.05 * MathCos(1.11 * s2);
            m_ff2_w[h][j] = 0.05 * MathSin(1.13 * s2);
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         {
            double s3 = (double)((c + 2) * (h + 1));
            m_w_cls[c][h] = 0.05 * MathSin(0.74 * s3);
         }
      }

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double s = (double)((hd + 1) * (d + 2) * (h + 3));
               m_wq[hd][d][h] = 0.05 * MathSin(0.67 * s);
               m_wk[hd][d][h] = 0.05 * MathCos(0.71 * s);
               m_wv[hd][d][h] = 0.05 * MathSin(0.73 * s);
               m_wo[hd][h][d] = 0.05 * MathCos(0.79 * s);
            }
         }
      }

      for(int i=0; i<FX6_TFT_SEQ; i++)
         m_rel_bias[i] = 0.0;

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;
      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.5;

      ZeroMoments();
      SyncShadow(true);
      m_shadow_ready = false;
      m_initialized = true;
   }

   void SyncShadow(const bool hard)
   {
      double a = (hard ? 0.0 : 0.995);
      double b = (hard ? 1.0 : (1.0 - a));

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_s_vsn_gate_b[i] = (hard ? m_vsn_gate_b[i] : a * m_s_vsn_gate_b[i] + b * m_vsn_gate_b[i]);
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            m_s_vsn_gate_w[i][j] = (hard ? m_vsn_gate_w[i][j] : a * m_s_vsn_gate_w[i][j] + b * m_vsn_gate_w[i][j]);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_s_vsn_proj_w[i][h] = (hard ? m_vsn_proj_w[i][h] : a * m_s_vsn_proj_w[i][h] + b * m_vsn_proj_w[i][h]);
            m_s_vsn_proj_b[i][h] = (hard ? m_vsn_proj_b[i][h] : a * m_s_vsn_proj_b[i][h] + b * m_vsn_proj_b[i][h]);
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_s_static_b[h] = (hard ? m_static_b[h] : a * m_s_static_b[h] + b * m_static_b[h]);

         m_s_enc_h0_b[h] = (hard ? m_enc_h0_b[h] : a * m_s_enc_h0_b[h] + b * m_enc_h0_b[h]);
         m_s_enc_c0_b[h] = (hard ? m_enc_c0_b[h] : a * m_s_enc_c0_b[h] + b * m_enc_c0_b[h]);
         m_s_dec_h0_b[h] = (hard ? m_dec_h0_b[h] : a * m_s_dec_h0_b[h] + b * m_dec_h0_b[h]);
         m_s_dec_c0_b[h] = (hard ? m_dec_c0_b[h] : a * m_s_dec_c0_b[h] + b * m_dec_c0_b[h]);

         m_s_e_bi[h] = (hard ? m_e_bi[h] : a * m_s_e_bi[h] + b * m_e_bi[h]);
         m_s_e_bf[h] = (hard ? m_e_bf[h] : a * m_s_e_bf[h] + b * m_e_bf[h]);
         m_s_e_bo[h] = (hard ? m_e_bo[h] : a * m_s_e_bo[h] + b * m_e_bo[h]);
         m_s_e_bg[h] = (hard ? m_e_bg[h] : a * m_s_e_bg[h] + b * m_e_bg[h]);

         m_s_d_bi[h] = (hard ? m_d_bi[h] : a * m_s_d_bi[h] + b * m_d_bi[h]);
         m_s_d_bf[h] = (hard ? m_d_bf[h] : a * m_s_d_bf[h] + b * m_d_bf[h]);
         m_s_d_bo[h] = (hard ? m_d_bo[h] : a * m_s_d_bo[h] + b * m_d_bo[h]);
         m_s_d_bg[h] = (hard ? m_d_bg[h] : a * m_s_d_bg[h] + b * m_d_bg[h]);

         m_s_ff1_b[h] = (hard ? m_ff1_b[h] : a * m_s_ff1_b[h] + b * m_ff1_b[h]);
         m_s_ff2_b[h] = (hard ? m_ff2_b[h] : a * m_s_ff2_b[h] + b * m_ff2_b[h]);

         m_s_w_mu[h] = (hard ? m_w_mu[h] : a * m_s_w_mu[h] + b * m_w_mu[h]);
         m_s_w_logv[h] = (hard ? m_w_logv[h] : a * m_s_w_logv[h] + b * m_w_logv[h]);
         m_s_w_q25[h] = (hard ? m_w_q25[h] : a * m_s_w_q25[h] + b * m_w_q25[h]);
         m_s_w_q75[h] = (hard ? m_w_q75[h] : a * m_s_w_q75[h] + b * m_w_q75[h]);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_s_static_w[h][i] = (hard ? m_static_w[h][i] : a * m_s_static_w[h][i] + b * m_static_w[h][i]);

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_s_enc_h0_w[h][j] = (hard ? m_enc_h0_w[h][j] : a * m_s_enc_h0_w[h][j] + b * m_enc_h0_w[h][j]);
            m_s_enc_c0_w[h][j] = (hard ? m_enc_c0_w[h][j] : a * m_s_enc_c0_w[h][j] + b * m_enc_c0_w[h][j]);
            m_s_dec_h0_s_w[h][j] = (hard ? m_dec_h0_s_w[h][j] : a * m_s_dec_h0_s_w[h][j] + b * m_dec_h0_s_w[h][j]);
            m_s_dec_h0_e_w[h][j] = (hard ? m_dec_h0_e_w[h][j] : a * m_s_dec_h0_e_w[h][j] + b * m_dec_h0_e_w[h][j]);
            m_s_dec_c0_s_w[h][j] = (hard ? m_dec_c0_s_w[h][j] : a * m_s_dec_c0_s_w[h][j] + b * m_dec_c0_s_w[h][j]);
            m_s_dec_c0_e_w[h][j] = (hard ? m_dec_c0_e_w[h][j] : a * m_s_dec_c0_e_w[h][j] + b * m_dec_c0_e_w[h][j]);

            m_s_e_wi_x[h][j] = (hard ? m_e_wi_x[h][j] : a * m_s_e_wi_x[h][j] + b * m_e_wi_x[h][j]);
            m_s_e_wf_x[h][j] = (hard ? m_e_wf_x[h][j] : a * m_s_e_wf_x[h][j] + b * m_e_wf_x[h][j]);
            m_s_e_wo_x[h][j] = (hard ? m_e_wo_x[h][j] : a * m_s_e_wo_x[h][j] + b * m_e_wo_x[h][j]);
            m_s_e_wg_x[h][j] = (hard ? m_e_wg_x[h][j] : a * m_s_e_wg_x[h][j] + b * m_e_wg_x[h][j]);
            m_s_e_wi_h[h][j] = (hard ? m_e_wi_h[h][j] : a * m_s_e_wi_h[h][j] + b * m_e_wi_h[h][j]);
            m_s_e_wf_h[h][j] = (hard ? m_e_wf_h[h][j] : a * m_s_e_wf_h[h][j] + b * m_e_wf_h[h][j]);
            m_s_e_wo_h[h][j] = (hard ? m_e_wo_h[h][j] : a * m_s_e_wo_h[h][j] + b * m_e_wo_h[h][j]);
            m_s_e_wg_h[h][j] = (hard ? m_e_wg_h[h][j] : a * m_s_e_wg_h[h][j] + b * m_e_wg_h[h][j]);

            m_s_d_wi_x[h][j] = (hard ? m_d_wi_x[h][j] : a * m_s_d_wi_x[h][j] + b * m_d_wi_x[h][j]);
            m_s_d_wf_x[h][j] = (hard ? m_d_wf_x[h][j] : a * m_s_d_wf_x[h][j] + b * m_d_wf_x[h][j]);
            m_s_d_wo_x[h][j] = (hard ? m_d_wo_x[h][j] : a * m_s_d_wo_x[h][j] + b * m_d_wo_x[h][j]);
            m_s_d_wg_x[h][j] = (hard ? m_d_wg_x[h][j] : a * m_s_d_wg_x[h][j] + b * m_d_wg_x[h][j]);
            m_s_d_wi_h[h][j] = (hard ? m_d_wi_h[h][j] : a * m_s_d_wi_h[h][j] + b * m_d_wi_h[h][j]);
            m_s_d_wf_h[h][j] = (hard ? m_d_wf_h[h][j] : a * m_s_d_wf_h[h][j] + b * m_d_wf_h[h][j]);
            m_s_d_wo_h[h][j] = (hard ? m_d_wo_h[h][j] : a * m_s_d_wo_h[h][j] + b * m_d_wo_h[h][j]);
            m_s_d_wg_h[h][j] = (hard ? m_d_wg_h[h][j] : a * m_s_d_wg_h[h][j] + b * m_d_wg_h[h][j]);

            m_s_ff1_w[h][j] = (hard ? m_ff1_w[h][j] : a * m_s_ff1_w[h][j] + b * m_ff1_w[h][j]);
            m_s_ff2_w[h][j] = (hard ? m_ff2_w[h][j] : a * m_s_ff2_w[h][j] + b * m_ff2_w[h][j]);
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
            m_s_w_cls[c][h] = (hard ? m_w_cls[c][h] : a * m_s_w_cls[c][h] + b * m_w_cls[c][h]);
      }

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               m_s_wq[hd][d][h] = (hard ? m_wq[hd][d][h] : a * m_s_wq[hd][d][h] + b * m_wq[hd][d][h]);
               m_s_wk[hd][d][h] = (hard ? m_wk[hd][d][h] : a * m_s_wk[hd][d][h] + b * m_wk[hd][d][h]);
               m_s_wv[hd][d][h] = (hard ? m_wv[hd][d][h] : a * m_s_wv[hd][d][h] + b * m_wv[hd][d][h]);
               m_s_wo[hd][h][d] = (hard ? m_wo[hd][h][d] : a * m_s_wo[hd][h][d] + b * m_wo[hd][h][d]);
            }
         }
      }

      for(int i=0; i<FX6_TFT_SEQ; i++)
         m_s_rel_bias[i] = (hard ? m_rel_bias[i] : a * m_s_rel_bias[i] + b * m_rel_bias[i]);

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         m_s_b_cls[c] = (hard ? m_b_cls[c] : a * m_s_b_cls[c] + b * m_b_cls[c]);
      m_s_b_mu = (hard ? m_b_mu : a * m_s_b_mu + b * m_b_mu);
      m_s_b_logv = (hard ? m_b_logv : a * m_s_b_logv + b * m_b_logv);
      m_s_b_q25 = (hard ? m_b_q25 : a * m_s_b_q25 + b * m_b_q25);
      m_s_b_q75 = (hard ? m_b_q75 : a * m_s_b_q75 + b * m_b_q75);
   }

   void UpdateNorm(const double &x[])
   {
      double a = (m_norm_steps < 128 ? 0.05 : 0.015);
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_norm_steps++;
      if(m_norm_steps >= 32) m_norm_ready = true;
   }

   void Normalize(const double &x[],
                  double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         if(!m_norm_ready)
         {
            xn[i] = FX6_ClipSym(x[i], 8.0);
            continue;
         }
         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FX6_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void PushHistory(const double &x[])
   {
      int p = m_hist_ptr;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_hist_x[p][i] = x[i];

      m_hist_ptr++;
      if(m_hist_ptr >= FX6_TFT_SEQ) m_hist_ptr = 0;
      if(m_hist_len < FX6_TFT_SEQ) m_hist_len++;
   }

   int HistIndexBack(const int back) const
   {
      if(m_hist_len <= 0) return -1;
      int b = back;
      if(b < 0) b = 0;
      if(b >= m_hist_len) b = m_hist_len - 1;
      int idx = m_hist_ptr - 1 - b;
      while(idx < 0) idx += FX6_TFT_SEQ;
      while(idx >= FX6_TFT_SEQ) idx -= FX6_TFT_SEQ;
      return idx;
   }

   void BuildInferenceSequence(const double &x_current[],
                               double &seq[][FX6_AI_WEIGHTS],
                               int &n) const
   {
      int keep = m_hist_len;
      if(keep > FX6_TFT_SEQ - 1) keep = FX6_TFT_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = HistIndexBack(back);
         if(idx < 0) continue;
         for(int k=0; k<FX6_AI_WEIGHTS; k++)
            seq[i][k] = m_hist_x[idx][k];
      }

      for(int k=0; k<FX6_AI_WEIGHTS; k++)
         seq[keep][k] = x_current[k];
      n = keep + 1;
      if(n < 1) n = 1;
   }

   void VSNForward(const double &xn[],
                   const bool use_shadow,
                   const bool training,
                   const int cache_t,
                   double &emb[])
   {
      double logits[FX6_AI_WEIGHTS];
      double alpha[FX6_AI_WEIGHTS];
      double feat[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double s = (use_shadow ? m_s_vsn_gate_b[i] : m_vsn_gate_b[i]);
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            s += (use_shadow ? m_s_vsn_gate_w[i][j] : m_vsn_gate_w[i][j]) * xn[j];
         logits[i] = FX6_ClipSym(s, 20.0);
      }
      SoftmaxN(logits, FX6_AI_WEIGHTS, alpha);

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double z = (use_shadow ? m_s_vsn_proj_b[i][h] : m_vsn_proj_b[i][h]);
            z += (use_shadow ? m_s_vsn_proj_w[i][h] : m_vsn_proj_w[i][h]) * xn[i];
            feat[i][h] = FX6_Tanh(FX6_ClipSym(z, 8.0));
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double s = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            s += alpha[i] * feat[i][h];
         emb[h] = s;
      }

      if(training)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            emb[h] *= DropMask(h, 111 + cache_t, 0.05, true);
      }

      if(cache_t >= 0 && cache_t < FX6_TFT_TBPTT)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            c_alpha[cache_t][i] = alpha[i];
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               c_feat[cache_t][i][h] = feat[i][h];
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            c_emb[cache_t][h] = emb[h];
      }
   }

   void StaticContext(const double &xn_last[],
                      const bool use_shadow,
                      double &s[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_static_b[h] : m_static_b[h]);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            z += (use_shadow ? m_s_static_w[h][i] : m_static_w[h][i]) * (xn_last[i] * m_static_mask[i]);
         s[h] = FX6_Tanh(FX6_ClipSym(z, 8.0));
      }
   }

   void InitEncoderState(const double &s[],
                         const bool use_shadow,
                         double &h0[],
                         double &c0[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double zh = (use_shadow ? m_s_enc_h0_b[h] : m_enc_h0_b[h]);
         double zc = (use_shadow ? m_s_enc_c0_b[h] : m_enc_c0_b[h]);
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            zh += (use_shadow ? m_s_enc_h0_w[h][j] : m_enc_h0_w[h][j]) * s[j];
            zc += (use_shadow ? m_s_enc_c0_w[h][j] : m_enc_c0_w[h][j]) * s[j];
         }
         h0[h] = FX6_Tanh(FX6_ClipSym(zh, 8.0));
         c0[h] = FX6_Tanh(FX6_ClipSym(zc, 8.0));
      }
   }

   void InitDecoderState(const double &s[],
                         const double &enc_last[],
                         const bool use_shadow,
                         double &h0[],
                         double &c0[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double zh = (use_shadow ? m_s_dec_h0_b[h] : m_dec_h0_b[h]);
         double zc = (use_shadow ? m_s_dec_c0_b[h] : m_dec_c0_b[h]);
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            zh += (use_shadow ? m_s_dec_h0_s_w[h][j] : m_dec_h0_s_w[h][j]) * s[j];
            zh += (use_shadow ? m_s_dec_h0_e_w[h][j] : m_dec_h0_e_w[h][j]) * enc_last[j];

            zc += (use_shadow ? m_s_dec_c0_s_w[h][j] : m_dec_c0_s_w[h][j]) * s[j];
            zc += (use_shadow ? m_s_dec_c0_e_w[h][j] : m_dec_c0_e_w[h][j]) * enc_last[j];
         }
         h0[h] = FX6_Tanh(FX6_ClipSym(zh, 8.0));
         c0[h] = FX6_Tanh(FX6_ClipSym(zc, 8.0));
      }
   }

   void EncoderStep(const double &x_in[],
                    const double &h_prev[],
                    const double &c_prev[],
                    const bool use_shadow,
                    double &ig[],
                    double &fg[],
                    double &og[],
                    double &gg[],
                    double &c_new[],
                    double &h_new[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double zi = (use_shadow ? m_s_e_bi[h] : m_e_bi[h]);
         double zf = (use_shadow ? m_s_e_bf[h] : m_e_bf[h]);
         double zo = (use_shadow ? m_s_e_bo[h] : m_e_bo[h]);
         double zg = (use_shadow ? m_s_e_bg[h] : m_e_bg[h]);

         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            zi += (use_shadow ? m_s_e_wi_x[h][i] : m_e_wi_x[h][i]) * x_in[i];
            zf += (use_shadow ? m_s_e_wf_x[h][i] : m_e_wf_x[h][i]) * x_in[i];
            zo += (use_shadow ? m_s_e_wo_x[h][i] : m_e_wo_x[h][i]) * x_in[i];
            zg += (use_shadow ? m_s_e_wg_x[h][i] : m_e_wg_x[h][i]) * x_in[i];

            zi += (use_shadow ? m_s_e_wi_h[h][i] : m_e_wi_h[h][i]) * h_prev[i];
            zf += (use_shadow ? m_s_e_wf_h[h][i] : m_e_wf_h[h][i]) * h_prev[i];
            zo += (use_shadow ? m_s_e_wo_h[h][i] : m_e_wo_h[h][i]) * h_prev[i];
            zg += (use_shadow ? m_s_e_wg_h[h][i] : m_e_wg_h[h][i]) * h_prev[i];
         }

         ig[h] = FX6_Sigmoid(FX6_ClipSym(zi, 20.0));
         fg[h] = FX6_Sigmoid(FX6_ClipSym(zf, 20.0));
         og[h] = FX6_Sigmoid(FX6_ClipSym(zo, 20.0));
         gg[h] = FX6_Tanh(FX6_ClipSym(zg, 8.0));

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         h_new[h] = og[h] * FX6_Tanh(c_new[h]);
      }
   }

   void DecoderStep(const double &x_in[],
                    const double &h_prev[],
                    const double &c_prev[],
                    const bool use_shadow,
                    double &ig[],
                    double &fg[],
                    double &og[],
                    double &gg[],
                    double &c_new[],
                    double &h_new[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double zi = (use_shadow ? m_s_d_bi[h] : m_d_bi[h]);
         double zf = (use_shadow ? m_s_d_bf[h] : m_d_bf[h]);
         double zo = (use_shadow ? m_s_d_bo[h] : m_d_bo[h]);
         double zg = (use_shadow ? m_s_d_bg[h] : m_d_bg[h]);

         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            zi += (use_shadow ? m_s_d_wi_x[h][i] : m_d_wi_x[h][i]) * x_in[i];
            zf += (use_shadow ? m_s_d_wf_x[h][i] : m_d_wf_x[h][i]) * x_in[i];
            zo += (use_shadow ? m_s_d_wo_x[h][i] : m_d_wo_x[h][i]) * x_in[i];
            zg += (use_shadow ? m_s_d_wg_x[h][i] : m_d_wg_x[h][i]) * x_in[i];

            zi += (use_shadow ? m_s_d_wi_h[h][i] : m_d_wi_h[h][i]) * h_prev[i];
            zf += (use_shadow ? m_s_d_wf_h[h][i] : m_d_wf_h[h][i]) * h_prev[i];
            zo += (use_shadow ? m_s_d_wo_h[h][i] : m_d_wo_h[h][i]) * h_prev[i];
            zg += (use_shadow ? m_s_d_wg_h[h][i] : m_d_wg_h[h][i]) * h_prev[i];
         }

         ig[h] = FX6_Sigmoid(FX6_ClipSym(zi, 20.0));
         fg[h] = FX6_Sigmoid(FX6_ClipSym(zf, 20.0));
         og[h] = FX6_Sigmoid(FX6_ClipSym(zo, 20.0));
         gg[h] = FX6_Tanh(FX6_ClipSym(zg, 8.0));

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         h_new[h] = og[h] * FX6_Tanh(c_new[h]);
      }
   }

   void AttentionStep(const int t,
                      const int n,
                      const bool use_shadow,
                      const double &dec_h[],
                      const double &enc_h[][FX6_AI_MLP_HIDDEN],
                      const bool cache,
                      double &attn_out[])
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         attn_out[h] = 0.0;

      double inv_scale = 1.0 / MathSqrt((double)FX6_TFT_D_HEAD);

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         double q[FX6_TFT_D_HEAD];
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            double s = 0.0;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               s += (use_shadow ? m_s_wq[hd][d][h] : m_wq[hd][d][h]) * dec_h[h];
            q[d] = s;
            if(cache) c_attn_q[t][hd][d] = s;
         }

         double logits[FX6_TFT_TBPTT];
         double den = 0.0;
         double maxv = -1e100;

         for(int j=0; j<=t && j<n; j++)
         {
            double score = 0.0;
            for(int d=0; d<FX6_TFT_D_HEAD; d++)
            {
               double k = 0.0;
               double v = 0.0;
               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               {
                  k += (use_shadow ? m_s_wk[hd][d][h] : m_wk[hd][d][h]) * enc_h[j][h];
                  v += (use_shadow ? m_s_wv[hd][d][h] : m_wv[hd][d][h]) * enc_h[j][h];
               }
               if(cache)
               {
                  c_attn_k[t][hd][j][d] = k;
                  c_attn_v[t][hd][j][d] = v;
               }
               score += q[d] * k;
            }

            int lag = t - j;
            if(lag < 0) lag = 0;
            if(lag >= FX6_TFT_SEQ) lag = FX6_TFT_SEQ - 1;
            score = score * inv_scale + (use_shadow ? m_s_rel_bias[lag] : m_rel_bias[lag]);
            logits[j] = score;
            if(score > maxv) maxv = score;
         }

         for(int j=0; j<=t && j<n; j++)
         {
            logits[j] = MathExp(FX6_ClipSym(logits[j] - maxv, 30.0));
            den += logits[j];
         }
         if(den <= 0.0) den = 1.0;

         double ctx[FX6_TFT_D_HEAD];
         for(int d=0; d<FX6_TFT_D_HEAD; d++) ctx[d] = 0.0;

         for(int j=0; j<=t && j<n; j++)
         {
            double a = logits[j] / den;
            if(cache) c_attn_w[t][hd][j] = a;
            for(int d=0; d<FX6_TFT_D_HEAD; d++)
               ctx[d] += a * (cache ? c_attn_v[t][hd][j][d] : 0.0);
         }

      if(!cache)
      {
         // If no cache requested, recompute context values directly.
         for(int d=0; d<FX6_TFT_D_HEAD; d++) ctx[d] = 0.0;
         for(int j=0; j<=t && j<n; j++)
         {
            double a = logits[j] / den;
            for(int d=0; d<FX6_TFT_D_HEAD; d++)
            {
               double vv = 0.0;
               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
                  vv += (use_shadow ? m_s_wv[hd][d][h] : m_wv[hd][d][h]) * enc_h[j][h];
               ctx[d] += a * vv;
            }
         }
      }

         for(int d=0; d<FX6_TFT_D_HEAD; d++)
            if(cache) c_attn_ctx[t][hd][d] = ctx[d];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double s = 0.0;
            for(int d=0; d<FX6_TFT_D_HEAD; d++)
               s += (use_shadow ? m_s_wo[hd][h][d] : m_wo[hd][h][d]) * ctx[d];
            attn_out[h] += s;
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         attn_out[h] = FX6_ClipSym(attn_out[h], 8.0);

      if(cache)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            c_attn_out[t][h] = attn_out[h];
      }
   }

   void FFNStep(const int t,
                const bool use_shadow,
                const bool training,
                const double &pre[],
                double &out[])
   {
      double ff1_raw[FX6_AI_MLP_HIDDEN];
      double ff1[FX6_AI_MLP_HIDDEN];

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_ff1_b[h] : m_ff1_b[h]);
         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            z += (use_shadow ? m_s_ff1_w[h][i] : m_ff1_w[h][i]) * pre[i];

         ff1_raw[h] = FX6_Tanh(FX6_ClipSym(z, 8.0));
         double m = DropMask(h, 190 + t, 0.10, training);
         ff1[h] = ff1_raw[h] * m;
      }

      double ff2[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double z = (use_shadow ? m_s_ff2_b[h] : m_ff2_b[h]);
         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            z += (use_shadow ? m_s_ff2_w[h][i] : m_ff2_w[h][i]) * ff1[i];
         ff2[h] = FX6_ClipSym(z, 8.0);
      }

      double sc = StochScale(t, training);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         out[h] = FX6_ClipSym(pre[h] + sc * ff2[h], 8.0);

      if(t >= 0 && t < FX6_TFT_TBPTT)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            c_pre[t][h] = pre[h];
            c_ff1_raw[t][h] = ff1_raw[h];
            c_ff1[t][h] = ff1[h];
            c_ff1_mask[t][h] = (MathAbs(ff1_raw[h]) > 1e-9 ? ff1[h] / ff1_raw[h] : 0.0);
            c_ff2[t][h] = ff2[h];
            c_final[t][h] = out[h];
         }
         c_stoch_scale[t] = sc;
      }
   }

   void HeadsStep(const int t,
                  const bool use_shadow,
                  const double &state[],
                  double &probs,
                  double &mu,
                  double &logv,
                  double &q25,
                  double &q75,
                  double &p_sell,
                  double &p_buy,
                  double &p_skip)
   {
      double logits[FX6_TFT_CLASS_COUNT];
      double probs3[FX6_TFT_CLASS_COUNT];

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
      {
         double z = (use_shadow ? m_s_b_cls[c] : m_b_cls[c]);
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            z += (use_shadow ? m_s_w_cls[c][h] : m_w_cls[c][h]) * state[h];
         logits[c] = FX6_ClipSym(z, 20.0);
      }

      SoftmaxN(logits, FX6_TFT_CLASS_COUNT, probs3);

      mu = (use_shadow ? m_s_b_mu : m_b_mu);
      logv = (use_shadow ? m_s_b_logv : m_b_logv);
      q25 = (use_shadow ? m_s_b_q25 : m_b_q25);
      q75 = (use_shadow ? m_s_b_q75 : m_b_q75);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         mu += (use_shadow ? m_s_w_mu[h] : m_w_mu[h]) * state[h];
         logv += (use_shadow ? m_s_w_logv[h] : m_w_logv[h]) * state[h];
         q25 += (use_shadow ? m_s_w_q25[h] : m_w_q25[h]) * state[h];
         q75 += (use_shadow ? m_s_w_q75[h] : m_w_q75[h]) * state[h];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double tmp = q25;
         q25 = q75;
         q75 = tmp;
      }

      p_sell = probs3[FX6_TFT_SELL];
      p_buy = probs3[FX6_TFT_BUY];
      p_skip = probs3[FX6_TFT_SKIP];
      probs = p_buy;

      if(t >= 0 && t < FX6_TFT_TBPTT)
      {
         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         {
            c_logits[t][c] = logits[c];
            c_probs[t][c] = probs3[c];
         }
         c_mu[t] = mu;
         c_logv[t] = logv;
         c_q25[t] = q25;
         c_q75[t] = q75;
      }
   }

   void AppendTrainSample(const int cls,
                          const double &x[],
                          const double move_points,
                          const double cost_points,
                          const double sample_w)
   {
      if(m_train_len < FX6_TFT_TBPTT)
      {
         int p = m_train_len;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_train_x[p][i] = x[i];
         m_train_cls[p] = cls;
         m_train_move[p] = move_points;
         m_train_cost[p] = cost_points;
         m_train_w[p] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FX6_TFT_TBPTT; t++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_train_x[FX6_TFT_TBPTT - 1][i] = x[i];
      m_train_cls[FX6_TFT_TBPTT - 1] = cls;
      m_train_move[FX6_TFT_TBPTT - 1] = move_points;
      m_train_cost[FX6_TFT_TBPTT - 1] = cost_points;
      m_train_w[FX6_TFT_TBPTT - 1] = sample_w;
   }

   void ForwardTrainSequence(const int n)
   {
      double sctx[FX6_AI_MLP_HIDDEN];
      double eh[FX6_AI_MLP_HIDDEN], ec[FX6_AI_MLP_HIDDEN];
      double dh[FX6_AI_MLP_HIDDEN], dc[FX6_AI_MLP_HIDDEN];

      for(int t=0; t<n; t++)
      {
         double xin[FX6_AI_WEIGHTS];
         double xout[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xin[i] = m_train_x[t][i];
         Normalize(xin, xout);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            c_xn[t][i] = xout[i];
      }

      double xn_last[FX6_AI_WEIGHTS];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         xn_last[i] = c_xn[n - 1][i];
      StaticContext(xn_last, false, sctx);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         c_static[h] = sctx[h];

      InitEncoderState(sctx, false, eh, ec);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         c_enc_h0[h] = eh[h];
         c_enc_c0[h] = ec[h];
      }

      // Encoder unroll.
      for(int t=0; t<n; t++)
      {
         double emb[FX6_AI_MLP_HIDDEN];
         double ig[FX6_AI_MLP_HIDDEN], fg[FX6_AI_MLP_HIDDEN], og[FX6_AI_MLP_HIDDEN], gg[FX6_AI_MLP_HIDDEN];
         double cnew[FX6_AI_MLP_HIDDEN], hnew[FX6_AI_MLP_HIDDEN];

         double xn_t[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xn_t[i] = c_xn[t][i];
         VSNForward(xn_t, false, true, t, emb);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            c_e_h_prev[t][h] = eh[h];
            c_e_c_prev[t][h] = ec[h];
         }

         EncoderStep(emb, eh, ec, false, ig, fg, og, gg, cnew, hnew);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            c_e_i[t][h] = ig[h];
            c_e_f[t][h] = fg[h];
            c_e_o[t][h] = og[h];
            c_e_g[t][h] = gg[h];
            c_e_c[t][h] = cnew[h];
            c_e_h[t][h] = hnew[h];
            eh[h] = hnew[h];
            ec[h] = cnew[h];
         }
      }

      double enc_last[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         enc_last[h] = c_e_h[n - 1][h];
      InitDecoderState(sctx, enc_last, false, dh, dc);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         c_dec_h0[h] = dh[h];
         c_dec_c0[h] = dc[h];
      }

      // Decoder + attention + FFN + heads.
      for(int t=0; t<n; t++)
      {
         double ig[FX6_AI_MLP_HIDDEN], fg[FX6_AI_MLP_HIDDEN], og[FX6_AI_MLP_HIDDEN], gg[FX6_AI_MLP_HIDDEN];
         double cnew[FX6_AI_MLP_HIDDEN], hnew[FX6_AI_MLP_HIDDEN];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            c_d_h_prev[t][h] = dh[h];
            c_d_c_prev[t][h] = dc[h];
         }

         double emb_t[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            emb_t[h] = c_emb[t][h];
         DecoderStep(emb_t, dh, dc, false, ig, fg, og, gg, cnew, hnew);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            c_d_i[t][h] = ig[h];
            c_d_f[t][h] = fg[h];
            c_d_o[t][h] = og[h];
            c_d_g[t][h] = gg[h];
            c_d_c[t][h] = cnew[h];
            c_d_h[t][h] = hnew[h];
            dh[h] = hnew[h];
            dc[h] = cnew[h];
         }

         double attn[FX6_AI_MLP_HIDDEN];
         double dec_h_t[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            dec_h_t[h] = c_d_h[t][h];
         AttentionStep(t, n, false, dec_h_t, c_e_h, true, attn);

         double pre[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            pre[h] = FX6_ClipSym(c_d_h[t][h] + attn[h], 8.0);

         double finalv[FX6_AI_MLP_HIDDEN];
         FFNStep(t, false, true, pre, finalv);

         double dump, mu, logv, q25, q75, ps, pb, pk;
         HeadsStep(t, false, finalv, dump, mu, logv, q25, q75, ps, pb, pk);
      }
   }

   void UpdateSessionCalibration(const int sess,
                                 const double p_raw,
                                 const int y,
                                 const double sample_w)
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s >= FX6_TFT_SESSIONS) s = FX6_TFT_SESSIONS - 1;

      double pr = FX6_Clamp(p_raw, 0.001, 0.999);
      double z = FX6_Logit(pr);
      double py = FX6_Sigmoid(m_sess_a[s] * z + m_sess_b[s]);
      double e = (double)y - py;

      double lr = FX6_Clamp(0.010 * sample_w, 0.001, 0.030);
      m_sess_a[s] += lr * (e * z - 0.001 * (m_sess_a[s] - 1.0));
      m_sess_b[s] += lr * e;
      m_sess_a[s] = FX6_Clamp(m_sess_a[s], 0.20, 5.00);
      m_sess_b[s] = FX6_Clamp(m_sess_b[s], -4.0, 4.0);

      m_sess_steps[s]++;
      if(m_sess_steps[s] >= 20)
         m_sess_ready[s] = true;
   }

   double ApplySessionCalibration(const int sess,
                                  const double p_raw) const
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s >= FX6_TFT_SESSIONS) s = FX6_TFT_SESSIONS - 1;

      double p = FX6_Clamp(p_raw, 0.001, 0.999);
      double z = FX6_Logit(p);
      double pc = FX6_Sigmoid(m_sess_a[s] * z + m_sess_b[s]);
      return FX6_Clamp(pc, 0.001, 0.999);
   }

   void AddWalkForwardSample(const double p_up,
                             const double p_skip,
                             const int cls,
                             const double move_points,
                             const double cost_points,
                             const int sess)
   {
      int p = m_wf_ptr;
      m_wf_pup[p] = FX6_Clamp(p_up, 0.001, 0.999);
      m_wf_pskip[p] = FX6_Clamp(p_skip, 0.001, 0.999);
      m_wf_cls[p] = cls;
      m_wf_move[p] = move_points;
      m_wf_cost[p] = cost_points;
      m_wf_sess[p] = sess;

      m_wf_ptr++;
      if(m_wf_ptr >= FX6_TFT_WF) m_wf_ptr = 0;
      if(m_wf_len < FX6_TFT_WF) m_wf_len++;
   }

   double EvalDecisionUtility(const int pred,
                              const int cls,
                              const double move_points,
                              const double cost_points) const
   {
      double cost = MathMax(cost_points, 0.0);
      double edge = MathMax(0.0, MathAbs(move_points) - cost);

      if(pred == FX6_TFT_SKIP)
      {
         if(cls == FX6_TFT_SKIP) return 0.10;
         return 0.00;
      }

      if(pred == cls)
         return edge;

      if(cls == FX6_TFT_SKIP)
         return -0.35 * cost;

      return -(edge + 0.5 * cost);
   }

   void OptimizeThresholds(void)
   {
      if(m_wf_len < 48) return;

      double best_score = -1e100;
      double best_buy = m_thr_buy;
      double best_sell = m_thr_sell;
      double best_skip = m_thr_skip;

      for(double th_buy=0.52; th_buy<=0.80; th_buy+=0.04)
      {
         for(double th_sell=0.20; th_sell<=0.48; th_sell+=0.04)
         {
            if(th_sell >= th_buy) continue;
            for(double th_skip=0.45; th_skip<=0.75; th_skip+=0.05)
            {
               double score = 0.0;
               for(int k=0; k<m_wf_len; k++)
               {
                  int idx = m_wf_ptr - 1 - k;
                  while(idx < 0) idx += FX6_TFT_WF;
                  while(idx >= FX6_TFT_WF) idx -= FX6_TFT_WF;

                  int pred = FX6_TFT_SKIP;
                  if(m_wf_pskip[idx] < th_skip)
                  {
                     if(m_wf_pup[idx] >= th_buy) pred = FX6_TFT_BUY;
                     else if(m_wf_pup[idx] <= th_sell) pred = FX6_TFT_SELL;
                  }

                  score += EvalDecisionUtility(pred,
                                               m_wf_cls[idx],
                                               m_wf_move[idx],
                                               m_wf_cost[idx]);
               }

               if(score > best_score)
               {
                  best_score = score;
                  best_buy = th_buy;
                  best_sell = th_sell;
                  best_skip = th_skip;
               }
            }
         }
      }

      m_thr_buy = FX6_Clamp(best_buy, 0.50, 0.90);
      m_thr_sell = FX6_Clamp(best_sell, 0.10, 0.50);
      m_thr_skip = FX6_Clamp(best_skip, 0.35, 0.90);
   }

   void TrainTBPTT(const FX6AIHyperParams &hp)
   {
      int n = m_train_len;
      if(n <= 0) return;
      if(n > FX6_TFT_TBPTT) n = FX6_TFT_TBPTT;
      if(n < 4) return;

      ForwardTrainSequence(n);

      // Gradients.
      double g_vsn_gate_w[FX6_AI_WEIGHTS][FX6_AI_WEIGHTS];
      double g_vsn_gate_b[FX6_AI_WEIGHTS];
      double g_vsn_proj_w[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];
      double g_vsn_proj_b[FX6_AI_WEIGHTS][FX6_AI_MLP_HIDDEN];

      double g_static_w[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_static_b[FX6_AI_MLP_HIDDEN];

      double g_enc_h0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_enc_h0_b[FX6_AI_MLP_HIDDEN];
      double g_enc_c0_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_enc_c0_b[FX6_AI_MLP_HIDDEN];
      double g_dec_h0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_dec_h0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_dec_h0_b[FX6_AI_MLP_HIDDEN];
      double g_dec_c0_s_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_dec_c0_e_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_dec_c0_b[FX6_AI_MLP_HIDDEN];

      double g_e_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_e_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_e_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_e_bi[FX6_AI_MLP_HIDDEN], g_e_bf[FX6_AI_MLP_HIDDEN], g_e_bo[FX6_AI_MLP_HIDDEN], g_e_bg[FX6_AI_MLP_HIDDEN];

      double g_d_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_d_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_d_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_d_bi[FX6_AI_MLP_HIDDEN], g_d_bf[FX6_AI_MLP_HIDDEN], g_d_bo[FX6_AI_MLP_HIDDEN], g_d_bg[FX6_AI_MLP_HIDDEN];

      double g_wq[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN], g_wk[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN], g_wv[FX6_TFT_HEADS][FX6_TFT_D_HEAD][FX6_AI_MLP_HIDDEN];
      double g_wo[FX6_TFT_HEADS][FX6_AI_MLP_HIDDEN][FX6_TFT_D_HEAD];
      double g_rel_bias[FX6_TFT_SEQ];

      double g_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_ff1_b[FX6_AI_MLP_HIDDEN];
      double g_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN], g_ff2_b[FX6_AI_MLP_HIDDEN];

      double g_w_cls[FX6_TFT_CLASS_COUNT][FX6_AI_MLP_HIDDEN], g_b_cls[FX6_TFT_CLASS_COUNT];
      double g_w_mu[FX6_AI_MLP_HIDDEN], g_b_mu;
      double g_w_logv[FX6_AI_MLP_HIDDEN], g_b_logv;
      double g_w_q25[FX6_AI_MLP_HIDDEN], g_b_q25;
      double g_w_q75[FX6_AI_MLP_HIDDEN], g_b_q75;

      // Zero gradients.
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         g_vsn_gate_b[i] = 0.0;
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            g_vsn_gate_w[i][j] = 0.0;
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            g_vsn_proj_w[i][h] = 0.0;
            g_vsn_proj_b[i][h] = 0.0;
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         g_static_b[h] = 0.0;
         g_enc_h0_b[h] = 0.0;
         g_enc_c0_b[h] = 0.0;
         g_dec_h0_b[h] = 0.0;
         g_dec_c0_b[h] = 0.0;

         g_e_bi[h] = 0.0; g_e_bf[h] = 0.0; g_e_bo[h] = 0.0; g_e_bg[h] = 0.0;
         g_d_bi[h] = 0.0; g_d_bf[h] = 0.0; g_d_bo[h] = 0.0; g_d_bg[h] = 0.0;

         g_ff1_b[h] = 0.0;
         g_ff2_b[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g_static_w[h][i] = 0.0;

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g_enc_h0_w[h][j] = 0.0; g_enc_c0_w[h][j] = 0.0;
            g_dec_h0_s_w[h][j] = 0.0; g_dec_h0_e_w[h][j] = 0.0;
            g_dec_c0_s_w[h][j] = 0.0; g_dec_c0_e_w[h][j] = 0.0;

            g_e_wi_x[h][j] = 0.0; g_e_wf_x[h][j] = 0.0; g_e_wo_x[h][j] = 0.0; g_e_wg_x[h][j] = 0.0;
            g_e_wi_h[h][j] = 0.0; g_e_wf_h[h][j] = 0.0; g_e_wo_h[h][j] = 0.0; g_e_wg_h[h][j] = 0.0;

            g_d_wi_x[h][j] = 0.0; g_d_wf_x[h][j] = 0.0; g_d_wo_x[h][j] = 0.0; g_d_wg_x[h][j] = 0.0;
            g_d_wi_h[h][j] = 0.0; g_d_wf_h[h][j] = 0.0; g_d_wo_h[h][j] = 0.0; g_d_wg_h[h][j] = 0.0;

            g_ff1_w[h][j] = 0.0;
            g_ff2_w[h][j] = 0.0;
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
            g_w_cls[c][h] = 0.0;
      }
      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         g_b_cls[c] = 0.0;
      g_b_mu = 0.0;
      g_b_logv = 0.0;
      g_b_q25 = 0.0;
      g_b_q75 = 0.0;

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               g_wq[hd][d][h] = 0.0;
               g_wk[hd][d][h] = 0.0;
               g_wv[hd][d][h] = 0.0;
               g_wo[hd][h][d] = 0.0;
            }
         }
      }
      for(int i=0; i<FX6_TFT_SEQ; i++)
         g_rel_bias[i] = 0.0;

      // Backprop buffers.
      double d_emb_total[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
      double d_enc_attn[FX6_TFT_TBPTT][FX6_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            d_emb_total[t][h] = 0.0;
            d_enc_attn[t][h] = 0.0;
         }
      }

      double dh_dec_next[FX6_AI_MLP_HIDDEN], dc_dec_next[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         dh_dec_next[h] = 0.0;
         dc_dec_next[h] = 0.0;
      }

      // Decoder-time reverse pass.
      for(int t=n - 1; t>=0; t--)
      {
         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FX6_Clamp(m_train_w[t], 0.25, 4.00);

         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);

         double pt = FX6_Clamp(c_probs[t][cls], 0.001, 0.999);
         double focal = MathPow(FX6_Clamp(1.0 - pt, 0.02, 1.0), 1.50);

         double dlogit[FX6_TFT_CLASS_COUNT];
         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         {
            double yk = (c == cls ? 1.0 : 0.0);
            dlogit[c] = FX6_ClipSym((c_probs[t][c] - yk) * w_cls * focal, 4.0);
         }

         double target = MathAbs(mv);
         double mu = c_mu[t];
         double logv = c_logv[t];
         double q25 = c_q25[t];
         double q75 = c_q75[t];
         double var = FX6_Clamp(MathExp(logv), 0.05, 100.0);

         double gmu = FX6_ClipSym((HuberGrad(mu - target, 6.0) / MathMax(var, 0.25)) * w_mv, 5.0);
         double glv = FX6_ClipSym(0.5 * w_mv * (1.0 - ((mu - target) * (mu - target)) / MathMax(var, 0.25)), 4.0);
         double gq25 = FX6_ClipSym(PinballHuberGrad(target, q25, 0.25, 1.5) * w_mv, 3.0);
         double gq75 = FX6_ClipSym(PinballHuberGrad(target, q75, 0.75, 1.5) * w_mv, 3.0);
         if(q25 > q75)
         {
            double pen = 0.20 * FX6_ClipSym(q25 - q75, 4.0);
            gq25 += pen;
            gq75 -= pen;
         }

         double d_final[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double d = dh_dec_next[h];
            for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
               d += dlogit[c] * m_w_cls[c][h];
            d += gmu * m_w_mu[h] + glv * m_w_logv[h] + gq25 * m_w_q25[h] + gq75 * m_w_q75[h];
            d_final[h] = d;

            g_w_mu[h] += gmu * c_final[t][h];
            g_w_logv[h] += glv * c_final[t][h];
            g_w_q25[h] += gq25 * c_final[t][h];
            g_w_q75[h] += gq75 * c_final[t][h];

            for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
               g_w_cls[c][h] += dlogit[c] * c_final[t][h];
         }
         g_b_mu += gmu;
         g_b_logv += glv;
         g_b_q25 += gq25;
         g_b_q75 += gq75;
         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
            g_b_cls[c] += dlogit[c];

         // FFN backward: final = pre + stoch * ff2.
         double d_pre[FX6_AI_MLP_HIDDEN];
         double d_ff2[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            d_pre[h] = d_final[h];
            d_ff2[h] = d_final[h] * c_stoch_scale[t];
            g_ff2_b[h] += d_ff2[h];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
               g_ff2_w[h][i] += d_ff2[h] * c_ff1[t][i];
         }

         double d_ff1[FX6_AI_MLP_HIDDEN];
         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            double s = 0.0;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               s += m_ff2_w[h][i] * d_ff2[h];
            d_ff1[i] = s;
         }

         double d_z1[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            d_z1[h] = d_ff1[h] * c_ff1_mask[t][h] * (1.0 - c_ff1_raw[t][h] * c_ff1_raw[t][h]);
            g_ff1_b[h] += d_z1[h];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
               g_ff1_w[h][i] += d_z1[h] * c_pre[t][i];
         }

         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            double s = 0.0;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               s += m_ff1_w[h][i] * d_z1[h];
            d_pre[i] += s;
         }

         // Split pre = dec_h + attn_out.
         double d_dec[FX6_AI_MLP_HIDDEN];
         double d_attn_out[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            d_dec[h] = d_pre[h];
            d_attn_out[h] = d_pre[h];
         }

         // Attention backward.
         double inv_scale = 1.0 / MathSqrt((double)FX6_TFT_D_HEAD);
         for(int hd=0; hd<FX6_TFT_HEADS; hd++)
         {
            double d_ctx[FX6_TFT_D_HEAD];
            for(int d=0; d<FX6_TFT_D_HEAD; d++) d_ctx[d] = 0.0;

            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               for(int d=0; d<FX6_TFT_D_HEAD; d++)
               {
                  g_wo[hd][h][d] += d_attn_out[h] * c_attn_ctx[t][hd][d];
                  d_ctx[d] += d_attn_out[h] * m_wo[hd][h][d];
               }
            }

            double d_a[FX6_TFT_TBPTT];
            for(int j=0; j<n; j++) d_a[j] = 0.0;
            double d_v[FX6_TFT_TBPTT][FX6_TFT_D_HEAD];
            for(int j=0; j<n; j++)
               for(int d=0; d<FX6_TFT_D_HEAD; d++)
                  d_v[j][d] = 0.0;

            for(int j=0; j<=t && j<n; j++)
            {
               for(int d=0; d<FX6_TFT_D_HEAD; d++)
               {
                  d_a[j] += d_ctx[d] * c_attn_v[t][hd][j][d];
                  d_v[j][d] += d_ctx[d] * c_attn_w[t][hd][j];
               }
            }

            double sum_ad = 0.0;
            for(int j=0; j<=t && j<n; j++)
               sum_ad += c_attn_w[t][hd][j] * d_a[j];

            double d_s[FX6_TFT_TBPTT];
            for(int j=0; j<n; j++) d_s[j] = 0.0;
            for(int j=0; j<=t && j<n; j++)
               d_s[j] = c_attn_w[t][hd][j] * (d_a[j] - sum_ad);

            double d_q[FX6_TFT_D_HEAD];
            for(int d=0; d<FX6_TFT_D_HEAD; d++) d_q[d] = 0.0;

            for(int j=0; j<=t && j<n; j++)
            {
               int lag = t - j;
               if(lag < 0) lag = 0;
               if(lag >= FX6_TFT_SEQ) lag = FX6_TFT_SEQ - 1;
               g_rel_bias[lag] += d_s[j];

               for(int d=0; d<FX6_TFT_D_HEAD; d++)
               {
                  d_q[d] += d_s[j] * inv_scale * c_attn_k[t][hd][j][d];

                  double dk = d_s[j] * inv_scale * c_attn_q[t][hd][d];
                  g_wk[hd][d][0] += 0.0; // keep compiler from optimizing away dimensions in strict builds.
                  for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
                  {
                     g_wk[hd][d][h] += dk * c_e_h[j][h];
                     d_enc_attn[j][h] += m_wk[hd][d][h] * dk;
                  }

                  for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
                  {
                     g_wv[hd][d][h] += d_v[j][d] * c_e_h[j][h];
                     d_enc_attn[j][h] += m_wv[hd][d][h] * d_v[j][d];
                  }
               }
            }

            for(int d=0; d<FX6_TFT_D_HEAD; d++)
            {
               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               {
                  g_wq[hd][d][h] += d_q[d] * c_d_h[t][h];
                  d_dec[h] += m_wq[hd][d][h] * d_q[d];
               }
            }
         }

         // Decoder LSTM backward.
         double dh_prev[FX6_AI_MLP_HIDDEN];
         double dc_prev[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh_prev[h] = 0.0;
            dc_prev[h] = 0.0;
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double tanh_c = FX6_Tanh(c_d_c[t][h]);
            double dh = d_dec[h];
            double dc = dc_dec_next[h] + dh * c_d_o[t][h] * (1.0 - tanh_c * tanh_c);

            double doo = (dh * tanh_c) * c_d_o[t][h] * (1.0 - c_d_o[t][h]);
            double di = (dc * c_d_g[t][h]) * c_d_i[t][h] * (1.0 - c_d_i[t][h]);
            double df = (dc * c_d_c_prev[t][h]) * c_d_f[t][h] * (1.0 - c_d_f[t][h]);
            double dg = (dc * c_d_i[t][h]) * (1.0 - c_d_g[t][h] * c_d_g[t][h]);

            g_d_bi[h] += di;
            g_d_bf[h] += df;
            g_d_bo[h] += doo;
            g_d_bg[h] += dg;

            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               g_d_wi_x[h][i] += di * c_emb[t][i];
               g_d_wf_x[h][i] += df * c_emb[t][i];
               g_d_wo_x[h][i] += doo * c_emb[t][i];
               g_d_wg_x[h][i] += dg * c_emb[t][i];

               d_emb_total[t][i] += m_d_wi_x[h][i] * di +
                                    m_d_wf_x[h][i] * df +
                                    m_d_wo_x[h][i] * doo +
                                    m_d_wg_x[h][i] * dg;

               g_d_wi_h[h][i] += di * c_d_h_prev[t][i];
               g_d_wf_h[h][i] += df * c_d_h_prev[t][i];
               g_d_wo_h[h][i] += doo * c_d_h_prev[t][i];
               g_d_wg_h[h][i] += dg * c_d_h_prev[t][i];

               dh_prev[i] += m_d_wi_h[h][i] * di +
                             m_d_wf_h[h][i] * df +
                             m_d_wo_h[h][i] * doo +
                             m_d_wg_h[h][i] * dg;
            }

            dc_prev[h] = dc * c_d_f[t][h];
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh_dec_next[h] = dh_prev[h];
            dc_dec_next[h] = dc_prev[h];
         }

         // Calibration + walk-forward updates with pre-calibration raw outputs.
         double den = c_probs[t][FX6_TFT_BUY] + c_probs[t][FX6_TFT_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir_raw = c_probs[t][FX6_TFT_BUY] / den;

         int ydir = (cls == FX6_TFT_BUY ? 1 : 0);
         if(cls == FX6_TFT_SKIP)
            ydir = (mv >= 0.0 ? 1 : 0);

         int sess = SessionBucket(ResolveContextTime());
         UpdateSessionCalibration(sess, p_dir_raw, ydir, sw);

         if(cls == FX6_TFT_SKIP)
            UpdateCalibration(p_dir_raw, ydir, 0.25 * sw);
         else
            UpdateCalibration(p_dir_raw, ydir, sw);

         FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         double x_train_t[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            x_train_t[i] = m_train_x[t][i];
         UpdateMoveHead(x_train_t, mv, hp, sw);

         AddWalkForwardSample(p_dir_raw,
                              c_probs[t][FX6_TFT_SKIP],
                              cls,
                              mv,
                              cost,
                              sess);
      }

      // Decoder init gradients.
      double d_static[FX6_AI_MLP_HIDDEN];
      double d_enc_last_from_dec_init[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         d_static[h] = 0.0;
         d_enc_last_from_dec_init[h] = 0.0;
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double dz_h = dh_dec_next[h] * (1.0 - c_dec_h0[h] * c_dec_h0[h]);
         double dz_c = dc_dec_next[h] * (1.0 - c_dec_c0[h] * c_dec_c0[h]);

         g_dec_h0_b[h] += dz_h;
         g_dec_c0_b[h] += dz_c;

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g_dec_h0_s_w[h][j] += dz_h * c_static[j];
            g_dec_h0_e_w[h][j] += dz_h * c_e_h[n - 1][j];
            g_dec_c0_s_w[h][j] += dz_c * c_static[j];
            g_dec_c0_e_w[h][j] += dz_c * c_e_h[n - 1][j];

            d_static[j] += m_dec_h0_s_w[h][j] * dz_h + m_dec_c0_s_w[h][j] * dz_c;
            d_enc_last_from_dec_init[j] += m_dec_h0_e_w[h][j] * dz_h + m_dec_c0_e_w[h][j] * dz_c;
         }
      }

      // Encoder reverse pass.
      double dh_enc_next[FX6_AI_MLP_HIDDEN], dc_enc_next[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         dh_enc_next[h] = d_enc_last_from_dec_init[h];
         dc_enc_next[h] = 0.0;
      }

      for(int t=n - 1; t>=0; t--)
      {
         double dh_prev[FX6_AI_MLP_HIDDEN];
         double dc_prev[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh_prev[h] = 0.0;
            dc_prev[h] = 0.0;
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double dh = dh_enc_next[h] + d_enc_attn[t][h];
            double tanh_c = FX6_Tanh(c_e_c[t][h]);
            double dc = dc_enc_next[h] + dh * c_e_o[t][h] * (1.0 - tanh_c * tanh_c);

            double doo = (dh * tanh_c) * c_e_o[t][h] * (1.0 - c_e_o[t][h]);
            double di = (dc * c_e_g[t][h]) * c_e_i[t][h] * (1.0 - c_e_i[t][h]);
            double df = (dc * c_e_c_prev[t][h]) * c_e_f[t][h] * (1.0 - c_e_f[t][h]);
            double dg = (dc * c_e_i[t][h]) * (1.0 - c_e_g[t][h] * c_e_g[t][h]);

            g_e_bi[h] += di;
            g_e_bf[h] += df;
            g_e_bo[h] += doo;
            g_e_bg[h] += dg;

            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               g_e_wi_x[h][i] += di * c_emb[t][i];
               g_e_wf_x[h][i] += df * c_emb[t][i];
               g_e_wo_x[h][i] += doo * c_emb[t][i];
               g_e_wg_x[h][i] += dg * c_emb[t][i];

               d_emb_total[t][i] += m_e_wi_x[h][i] * di +
                                    m_e_wf_x[h][i] * df +
                                    m_e_wo_x[h][i] * doo +
                                    m_e_wg_x[h][i] * dg;

               g_e_wi_h[h][i] += di * c_e_h_prev[t][i];
               g_e_wf_h[h][i] += df * c_e_h_prev[t][i];
               g_e_wo_h[h][i] += doo * c_e_h_prev[t][i];
               g_e_wg_h[h][i] += dg * c_e_h_prev[t][i];

               dh_prev[i] += m_e_wi_h[h][i] * di +
                             m_e_wf_h[h][i] * df +
                             m_e_wo_h[h][i] * doo +
                             m_e_wg_h[h][i] * dg;
            }

            dc_prev[h] = dc * c_e_f[t][h];
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh_enc_next[h] = dh_prev[h];
            dc_enc_next[h] = dc_prev[h];
         }
      }

      // Encoder init gradients.
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double dz_h = dh_enc_next[h] * (1.0 - c_enc_h0[h] * c_enc_h0[h]);
         double dz_c = dc_enc_next[h] * (1.0 - c_enc_c0[h] * c_enc_c0[h]);

         g_enc_h0_b[h] += dz_h;
         g_enc_c0_b[h] += dz_c;

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g_enc_h0_w[h][j] += dz_h * c_static[j];
            g_enc_c0_w[h][j] += dz_c * c_static[j];
            d_static[j] += m_enc_h0_w[h][j] * dz_h + m_enc_c0_w[h][j] * dz_c;
         }
      }

      // Static context gradients to static projection.
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double dz = d_static[h] * (1.0 - c_static[h] * c_static[h]);
         g_static_b[h] += dz;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g_static_w[h][i] += dz * (c_xn[n - 1][i] * m_static_mask[i]);
      }

      // VSN backward (full softmax + per-feature projection gradients).
      for(int t=n - 1; t>=0; t--)
      {
         double d_alpha[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) d_alpha[i] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double d_feat = d_emb_total[t][h] * c_alpha[t][i];
               double zgrad = d_feat * (1.0 - c_feat[t][i][h] * c_feat[t][i][h]);
               g_vsn_proj_w[i][h] += zgrad * c_xn[t][i];
               g_vsn_proj_b[i][h] += zgrad;
               d_alpha[i] += d_emb_total[t][h] * c_feat[t][i][h];
            }
         }

         double sumad = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            sumad += c_alpha[t][i] * d_alpha[i];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double dlogit = c_alpha[t][i] * (d_alpha[i] - sumad);
            g_vsn_gate_b[i] += dlogit;
            for(int j=0; j<FX6_AI_WEIGHTS; j++)
               g_vsn_gate_w[i][j] += dlogit * c_xn[t][j];
         }
      }

      // Global gradient norm clip.
      double g2 = 0.0;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         g2 += g_vsn_gate_b[i] * g_vsn_gate_b[i];
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            g2 += g_vsn_gate_w[i][j] * g_vsn_gate_w[i][j];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            g2 += g_vsn_proj_w[i][h] * g_vsn_proj_w[i][h];
            g2 += g_vsn_proj_b[i][h] * g_vsn_proj_b[i][h];
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         g2 += g_static_b[h] * g_static_b[h];
         g2 += g_enc_h0_b[h] * g_enc_h0_b[h] + g_enc_c0_b[h] * g_enc_c0_b[h] + g_dec_h0_b[h] * g_dec_h0_b[h] + g_dec_c0_b[h] * g_dec_c0_b[h];
         g2 += g_e_bi[h] * g_e_bi[h] + g_e_bf[h] * g_e_bf[h] + g_e_bo[h] * g_e_bo[h] + g_e_bg[h] * g_e_bg[h];
         g2 += g_d_bi[h] * g_d_bi[h] + g_d_bf[h] * g_d_bf[h] + g_d_bo[h] * g_d_bo[h] + g_d_bg[h] * g_d_bg[h];
         g2 += g_ff1_b[h] * g_ff1_b[h] + g_ff2_b[h] * g_ff2_b[h];
         g2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g2 += g_static_w[h][i] * g_static_w[h][i];

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g2 += g_enc_h0_w[h][j] * g_enc_h0_w[h][j] + g_enc_c0_w[h][j] * g_enc_c0_w[h][j];
            g2 += g_dec_h0_s_w[h][j] * g_dec_h0_s_w[h][j] + g_dec_h0_e_w[h][j] * g_dec_h0_e_w[h][j];
            g2 += g_dec_c0_s_w[h][j] * g_dec_c0_s_w[h][j] + g_dec_c0_e_w[h][j] * g_dec_c0_e_w[h][j];

            g2 += g_e_wi_x[h][j] * g_e_wi_x[h][j] + g_e_wf_x[h][j] * g_e_wf_x[h][j] + g_e_wo_x[h][j] * g_e_wo_x[h][j] + g_e_wg_x[h][j] * g_e_wg_x[h][j];
            g2 += g_e_wi_h[h][j] * g_e_wi_h[h][j] + g_e_wf_h[h][j] * g_e_wf_h[h][j] + g_e_wo_h[h][j] * g_e_wo_h[h][j] + g_e_wg_h[h][j] * g_e_wg_h[h][j];

            g2 += g_d_wi_x[h][j] * g_d_wi_x[h][j] + g_d_wf_x[h][j] * g_d_wf_x[h][j] + g_d_wo_x[h][j] * g_d_wo_x[h][j] + g_d_wg_x[h][j] * g_d_wg_x[h][j];
            g2 += g_d_wi_h[h][j] * g_d_wi_h[h][j] + g_d_wf_h[h][j] * g_d_wf_h[h][j] + g_d_wo_h[h][j] * g_d_wo_h[h][j] + g_d_wg_h[h][j] * g_d_wg_h[h][j];

            g2 += g_ff1_w[h][j] * g_ff1_w[h][j] + g_ff2_w[h][j] * g_ff2_w[h][j];
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
            g2 += g_w_cls[c][h] * g_w_cls[c][h];
      }

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         g2 += g_b_cls[c] * g_b_cls[c];
      g2 += g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               g2 += g_wq[hd][d][h] * g_wq[hd][d][h];
               g2 += g_wk[hd][d][h] * g_wk[hd][d][h];
               g2 += g_wv[hd][d][h] * g_wv[hd][d][h];
               g2 += g_wo[hd][h][d] * g_wo[hd][h][d];
            }
         }
      }
      for(int i=0; i<FX6_TFT_SEQ; i++)
         g2 += g_rel_bias[i] * g_rel_bias[i];

      double gnorm = MathSqrt(g2 + 1e-12);
      double gscale = (gnorm > 5.0 ? (5.0 / gnorm) : 1.0);

      // AdamW update.
      double lr = ScheduledLR(hp);
      double wd = FX6_Clamp(0.50 * hp.l2, 0.0, 0.05);
      m_adam_t++;
      m_step += n;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         AdamWStep(m_vsn_gate_b[i], m_m_vsn_gate_b[i], m_v_vsn_gate_b[i], gscale * g_vsn_gate_b[i], lr, 0.0);
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            AdamWStep(m_vsn_gate_w[i][j], m_m_vsn_gate_w[i][j], m_v_vsn_gate_w[i][j], gscale * g_vsn_gate_w[i][j], lr, wd);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            AdamWStep(m_vsn_proj_w[i][h], m_m_vsn_proj_w[i][h], m_v_vsn_proj_w[i][h], gscale * g_vsn_proj_w[i][h], lr, wd);
            AdamWStep(m_vsn_proj_b[i][h], m_m_vsn_proj_b[i][h], m_v_vsn_proj_b[i][h], gscale * g_vsn_proj_b[i][h], lr, 0.0);
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         AdamWStep(m_static_b[h], m_m_static_b[h], m_v_static_b[h], gscale * g_static_b[h], lr, 0.0);

         AdamWStep(m_enc_h0_b[h], m_m_enc_h0_b[h], m_v_enc_h0_b[h], gscale * g_enc_h0_b[h], lr, 0.0);
         AdamWStep(m_enc_c0_b[h], m_m_enc_c0_b[h], m_v_enc_c0_b[h], gscale * g_enc_c0_b[h], lr, 0.0);
         AdamWStep(m_dec_h0_b[h], m_m_dec_h0_b[h], m_v_dec_h0_b[h], gscale * g_dec_h0_b[h], lr, 0.0);
         AdamWStep(m_dec_c0_b[h], m_m_dec_c0_b[h], m_v_dec_c0_b[h], gscale * g_dec_c0_b[h], lr, 0.0);

         AdamWStep(m_e_bi[h], m_m_e_bi[h], m_v_e_bi[h], gscale * g_e_bi[h], lr, 0.0);
         AdamWStep(m_e_bf[h], m_m_e_bf[h], m_v_e_bf[h], gscale * g_e_bf[h], lr, 0.0);
         AdamWStep(m_e_bo[h], m_m_e_bo[h], m_v_e_bo[h], gscale * g_e_bo[h], lr, 0.0);
         AdamWStep(m_e_bg[h], m_m_e_bg[h], m_v_e_bg[h], gscale * g_e_bg[h], lr, 0.0);

         AdamWStep(m_d_bi[h], m_m_d_bi[h], m_v_d_bi[h], gscale * g_d_bi[h], lr, 0.0);
         AdamWStep(m_d_bf[h], m_m_d_bf[h], m_v_d_bf[h], gscale * g_d_bf[h], lr, 0.0);
         AdamWStep(m_d_bo[h], m_m_d_bo[h], m_v_d_bo[h], gscale * g_d_bo[h], lr, 0.0);
         AdamWStep(m_d_bg[h], m_m_d_bg[h], m_v_d_bg[h], gscale * g_d_bg[h], lr, 0.0);

         AdamWStep(m_ff1_b[h], m_m_ff1_b[h], m_v_ff1_b[h], gscale * g_ff1_b[h], lr, 0.0);
         AdamWStep(m_ff2_b[h], m_m_ff2_b[h], m_v_ff2_b[h], gscale * g_ff2_b[h], lr, 0.0);

         AdamWStep(m_w_mu[h], m_m_w_mu[h], m_v_w_mu[h], gscale * g_w_mu[h], lr, wd);
         AdamWStep(m_w_logv[h], m_m_w_logv[h], m_v_w_logv[h], gscale * g_w_logv[h], lr, wd);
         AdamWStep(m_w_q25[h], m_m_w_q25[h], m_v_w_q25[h], gscale * g_w_q25[h], lr, wd);
         AdamWStep(m_w_q75[h], m_m_w_q75[h], m_v_w_q75[h], gscale * g_w_q75[h], lr, wd);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            AdamWStep(m_static_w[h][i], m_m_static_w[h][i], m_v_static_w[h][i], gscale * g_static_w[h][i], lr, wd);

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            AdamWStep(m_enc_h0_w[h][j], m_m_enc_h0_w[h][j], m_v_enc_h0_w[h][j], gscale * g_enc_h0_w[h][j], lr, wd);
            AdamWStep(m_enc_c0_w[h][j], m_m_enc_c0_w[h][j], m_v_enc_c0_w[h][j], gscale * g_enc_c0_w[h][j], lr, wd);
            AdamWStep(m_dec_h0_s_w[h][j], m_m_dec_h0_s_w[h][j], m_v_dec_h0_s_w[h][j], gscale * g_dec_h0_s_w[h][j], lr, wd);
            AdamWStep(m_dec_h0_e_w[h][j], m_m_dec_h0_e_w[h][j], m_v_dec_h0_e_w[h][j], gscale * g_dec_h0_e_w[h][j], lr, wd);
            AdamWStep(m_dec_c0_s_w[h][j], m_m_dec_c0_s_w[h][j], m_v_dec_c0_s_w[h][j], gscale * g_dec_c0_s_w[h][j], lr, wd);
            AdamWStep(m_dec_c0_e_w[h][j], m_m_dec_c0_e_w[h][j], m_v_dec_c0_e_w[h][j], gscale * g_dec_c0_e_w[h][j], lr, wd);

            AdamWStep(m_e_wi_x[h][j], m_m_e_wi_x[h][j], m_v_e_wi_x[h][j], gscale * g_e_wi_x[h][j], lr, wd);
            AdamWStep(m_e_wf_x[h][j], m_m_e_wf_x[h][j], m_v_e_wf_x[h][j], gscale * g_e_wf_x[h][j], lr, wd);
            AdamWStep(m_e_wo_x[h][j], m_m_e_wo_x[h][j], m_v_e_wo_x[h][j], gscale * g_e_wo_x[h][j], lr, wd);
            AdamWStep(m_e_wg_x[h][j], m_m_e_wg_x[h][j], m_v_e_wg_x[h][j], gscale * g_e_wg_x[h][j], lr, wd);
            AdamWStep(m_e_wi_h[h][j], m_m_e_wi_h[h][j], m_v_e_wi_h[h][j], gscale * g_e_wi_h[h][j], lr, wd);
            AdamWStep(m_e_wf_h[h][j], m_m_e_wf_h[h][j], m_v_e_wf_h[h][j], gscale * g_e_wf_h[h][j], lr, wd);
            AdamWStep(m_e_wo_h[h][j], m_m_e_wo_h[h][j], m_v_e_wo_h[h][j], gscale * g_e_wo_h[h][j], lr, wd);
            AdamWStep(m_e_wg_h[h][j], m_m_e_wg_h[h][j], m_v_e_wg_h[h][j], gscale * g_e_wg_h[h][j], lr, wd);

            AdamWStep(m_d_wi_x[h][j], m_m_d_wi_x[h][j], m_v_d_wi_x[h][j], gscale * g_d_wi_x[h][j], lr, wd);
            AdamWStep(m_d_wf_x[h][j], m_m_d_wf_x[h][j], m_v_d_wf_x[h][j], gscale * g_d_wf_x[h][j], lr, wd);
            AdamWStep(m_d_wo_x[h][j], m_m_d_wo_x[h][j], m_v_d_wo_x[h][j], gscale * g_d_wo_x[h][j], lr, wd);
            AdamWStep(m_d_wg_x[h][j], m_m_d_wg_x[h][j], m_v_d_wg_x[h][j], gscale * g_d_wg_x[h][j], lr, wd);
            AdamWStep(m_d_wi_h[h][j], m_m_d_wi_h[h][j], m_v_d_wi_h[h][j], gscale * g_d_wi_h[h][j], lr, wd);
            AdamWStep(m_d_wf_h[h][j], m_m_d_wf_h[h][j], m_v_d_wf_h[h][j], gscale * g_d_wf_h[h][j], lr, wd);
            AdamWStep(m_d_wo_h[h][j], m_m_d_wo_h[h][j], m_v_d_wo_h[h][j], gscale * g_d_wo_h[h][j], lr, wd);
            AdamWStep(m_d_wg_h[h][j], m_m_d_wg_h[h][j], m_v_d_wg_h[h][j], gscale * g_d_wg_h[h][j], lr, wd);

            AdamWStep(m_ff1_w[h][j], m_m_ff1_w[h][j], m_v_ff1_w[h][j], gscale * g_ff1_w[h][j], lr, wd);
            AdamWStep(m_ff2_w[h][j], m_m_ff2_w[h][j], m_v_ff2_w[h][j], gscale * g_ff2_w[h][j], lr, wd);
         }

         for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
            AdamWStep(m_w_cls[c][h], m_m_w_cls[c][h], m_v_w_cls[c][h], gscale * g_w_cls[c][h], lr, wd);
      }

      for(int c=0; c<FX6_TFT_CLASS_COUNT; c++)
         AdamWStep(m_b_cls[c], m_m_b_cls[c], m_v_b_cls[c], gscale * g_b_cls[c], lr, 0.0);
      AdamWStep(m_b_mu, m_m_b_mu, m_v_b_mu, gscale * g_b_mu, lr, 0.0);
      AdamWStep(m_b_logv, m_m_b_logv, m_v_b_logv, gscale * g_b_logv, lr, 0.0);
      AdamWStep(m_b_q25, m_m_b_q25, m_v_b_q25, gscale * g_b_q25, lr, 0.0);
      AdamWStep(m_b_q75, m_m_b_q75, m_v_b_q75, gscale * g_b_q75, lr, 0.0);

      for(int hd=0; hd<FX6_TFT_HEADS; hd++)
      {
         for(int d=0; d<FX6_TFT_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               AdamWStep(m_wq[hd][d][h], m_m_wq[hd][d][h], m_v_wq[hd][d][h], gscale * g_wq[hd][d][h], lr, wd);
               AdamWStep(m_wk[hd][d][h], m_m_wk[hd][d][h], m_v_wk[hd][d][h], gscale * g_wk[hd][d][h], lr, wd);
               AdamWStep(m_wv[hd][d][h], m_m_wv[hd][d][h], m_v_wv[hd][d][h], gscale * g_wv[hd][d][h], lr, wd);
               AdamWStep(m_wo[hd][h][d], m_m_wo[hd][h][d], m_v_wo[hd][h][d], gscale * g_wo[hd][h][d], lr, wd);
            }
         }
      }
      for(int i=0; i<FX6_TFT_SEQ; i++)
         AdamWStep(m_rel_bias[i], m_m_rel_bias[i], m_v_rel_bias[i], gscale * g_rel_bias[i], lr, 0.0);

      m_b_logv = FX6_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_w_mu[h] = FX6_ClipSym(m_w_mu[h], 8.0);
         m_w_logv[h] = FX6_ClipSym(m_w_logv[h], 8.0);
         m_w_q25[h] = FX6_ClipSym(m_w_q25[h], 8.0);
         m_w_q75[h] = FX6_ClipSym(m_w_q75[h], 8.0);
      }

      SyncShadow(false);
      if(!m_shadow_ready && m_adam_t >= 64)
         m_shadow_ready = true;

      if((m_step % 32) == 0)
         OptimizeThresholds();
   }

   void ForwardInference(const double &x[],
                         const bool use_shadow,
                         double &p_sell,
                         double &p_buy,
                         double &p_skip,
                         double &mu,
                         double &logv,
                         double &q25,
                         double &q75)
   {
      double seq[FX6_TFT_SEQ][FX6_AI_WEIGHTS];
      int n = 0;
      BuildInferenceSequence(x, seq, n);
      if(n < 1) n = 1;

      double xn[FX6_TFT_SEQ][FX6_AI_WEIGHTS];
      double emb[FX6_TFT_SEQ][FX6_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         double seq_t[FX6_AI_WEIGHTS];
         double xn_t[FX6_AI_WEIGHTS];
         double emb_t[FX6_AI_MLP_HIDDEN];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            seq_t[i] = seq[t][i];

         Normalize(seq_t, xn_t);
         VSNForward(xn_t, use_shadow, false, -1, emb_t);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xn[t][i] = xn_t[i];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            emb[t][h] = emb_t[h];
      }

      double sctx[FX6_AI_MLP_HIDDEN];
      double xn_last[FX6_AI_WEIGHTS];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         xn_last[i] = xn[n - 1][i];
      StaticContext(xn_last, use_shadow, sctx);

      double eh[FX6_AI_MLP_HIDDEN], ec[FX6_AI_MLP_HIDDEN];
      InitEncoderState(sctx, use_shadow, eh, ec);

      double enc_h[FX6_TFT_SEQ][FX6_AI_MLP_HIDDEN];
      for(int t=0; t<n; t++)
      {
         double ig[FX6_AI_MLP_HIDDEN], fg[FX6_AI_MLP_HIDDEN], og[FX6_AI_MLP_HIDDEN], gg[FX6_AI_MLP_HIDDEN];
         double cnew[FX6_AI_MLP_HIDDEN], hnew[FX6_AI_MLP_HIDDEN];
         double emb_t[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            emb_t[h] = emb[t][h];
         EncoderStep(emb_t, eh, ec, use_shadow, ig, fg, og, gg, cnew, hnew);
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            eh[h] = hnew[h];
            ec[h] = cnew[h];
            enc_h[t][h] = hnew[h];
         }
      }

      double dh[FX6_AI_MLP_HIDDEN], dc[FX6_AI_MLP_HIDDEN];
      double enc_last[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         enc_last[h] = enc_h[n - 1][h];
      InitDecoderState(sctx, enc_last, use_shadow, dh, dc);

      for(int t=0; t<n; t++)
      {
         double ig[FX6_AI_MLP_HIDDEN], fg[FX6_AI_MLP_HIDDEN], og[FX6_AI_MLP_HIDDEN], gg[FX6_AI_MLP_HIDDEN];
         double cnew[FX6_AI_MLP_HIDDEN], hnew[FX6_AI_MLP_HIDDEN];
         double emb_t[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            emb_t[h] = emb[t][h];
         DecoderStep(emb_t, dh, dc, use_shadow, ig, fg, og, gg, cnew, hnew);
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh[h] = hnew[h];
            dc[h] = cnew[h];
         }

         double attn[FX6_AI_MLP_HIDDEN];
         AttentionStep(t, n, use_shadow, dh, enc_h, false, attn);

         double pre[FX6_AI_MLP_HIDDEN], fin[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            pre[h] = FX6_ClipSym(dh[h] + attn[h], 8.0);
         FFNStep(-1, use_shadow, false, pre, fin);

         double dump;
         HeadsStep(-1, use_shadow, fin, dump, mu, logv, q25, q75, p_sell, p_buy, p_skip);
      }
   }

public:
   CFX6AITFT(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_TFT; }
   virtual string AIName(void) const { return "tft"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_shadow_ready = false;
      m_step = 0;
      m_seen = 0;
      m_adam_t = 0;
      m_symbol_hash = SymbolHash(_Symbol);

      ResetNorm();
      ResetHistory();
      ResetTrainBuffer();
      ResetWalkForward();
      ResetSessionCalibration();

      m_thr_buy = 0.62;
      m_thr_sell = 0.38;
      m_thr_skip = 0.58;
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
         InitParams();
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FX6AIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen++;

      // Per-symbol lifecycle guard.
      uint sh = SymbolHash(_Symbol);
      if(sh != m_symbol_hash)
      {
         m_symbol_hash = sh;
         ResetHistory();
         ResetTrainBuffer();
         ResetWalkForward();
         ResetSessionCalibration();
      }

      // Controlled reset policy for state-bleed / regime shocks.
      if((m_seen % 1024) == 0 || MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
      {
         ResetHistory();
         ResetTrainBuffer();
      }

      UpdateNorm(x);

      int cls = MapClass(y, x, move_points);
      double sw = FX6_Clamp(MoveSampleWeight(x, move_points), 0.25, 4.00);
      double cost = InputCostProxyPoints(x);

      AppendTrainSample(cls, x, move_points, cost, sw);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      TrainTBPTT(h);

      PushHistory(x);
   }

   virtual double PredictProb(const double &x[],
                              const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double ps = 0.3333, pb = 0.3333, pk = 0.3333;
      double mu = 0.0, logv = MathLog(1.0), q25 = 0.0, q75 = 0.0;

      ForwardInference(x, m_shadow_ready, ps, pb, pk, mu, logv, q25, q75);

      double den = pb + ps;
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = pb / den;

      double p_cal = CalibrateProb(p_dir_raw);
      int sess = SessionBucket(ResolveContextTime());
      p_cal = ApplySessionCalibration(sess, p_cal);

      double active = FX6_Clamp(1.0 - pk, 0.0, 1.0);
      if(pk > m_thr_skip) active *= 0.25;
      else if(p_cal < m_thr_buy && p_cal > m_thr_sell) active *= 0.35;

      double p_up = p_cal * active;
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[],
                                            const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double ps = 0.3333, pb = 0.3333, pk = 0.3333;
      double mu = 0.0, logv = MathLog(1.0), q25 = 0.0, q75 = 0.0;

      ForwardInference(x, m_shadow_ready, ps, pb, pk, mu, logv, q25, q75);

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);

      double active = FX6_Clamp(1.0 - pk, 0.0, 1.0);
      double ev = MathMax(0.0, (0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr) * active);

      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      if(ev > 0.0 && base_ev > 0.0) return 0.65 * ev + 0.35 * base_ev;
      if(ev > 0.0) return ev;
      return base_ev;
   }
};

#endif // __FX6_AI_TFT_MQH__
