#ifndef __FXAI_AI_AUTOFORMER_MQH__
#define __FXAI_AI_AUTOFORMER_MQH__

#include "..\API\plugin_base.mqh"

#define FXAI_AF_SEQ 128
#define FXAI_AF_TBPTT 32
#define FXAI_AF_HEADS 4
#define FXAI_AF_D_HEAD (FXAI_AI_MLP_HIDDEN / FXAI_AF_HEADS)
#define FXAI_AF_TOPK_LAGS 8
#define FXAI_AF_CLASS_COUNT 3
#define FXAI_AF_BLOCKS 2
#define FXAI_AF_MA_KERNELS 4
#define FXAI_AF_HORIZONS 3
#define FXAI_AF_CAL_BINS 12
#define FXAI_AF_SESSIONS 4
#define FXAI_AF_COS_CYCLE 4096
#define FXAI_AF_PI 3.14159265358979323846

#define FXAI_AF_SELL 0
#define FXAI_AF_BUY  1
#define FXAI_AF_SKIP 2

class CFXAIAIAutoformer : public CFXAIAIPlugin
{
private:
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

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   int LagSrcFlat(const int k, const int h) const
   {
      return k * FXAI_AI_MLP_HIDDEN + h;
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct((t > 0 ? t : TimeCurrent()), dt);
      int h = dt.hour;
      // 0: Asia, 1: Europe, 2: US, 3: Off-hours overlap/illiquid bucket.
      if(h >= 0 && h <= 6) return 0;
      if(h >= 7 && h <= 12) return 1;
      if(h >= 13 && h <= 20) return 2;
      return 3;
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -40.0, 40.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -40.0, 40.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -40.0, 40.0));
      double s = e0 + e1 + e2;
      if(s <= 0.0)
      {
         probs[0] = 0.3333333;
         probs[1] = 0.3333333;
         probs[2] = 0.3333333;
         return;
      }
      probs[0] = e0 / s;
      probs[1] = e1 / s;
      probs[2] = e2 / s;
   }

   void SoftmaxN(const double &logits[], const int n, double &probs[]) const
   {
      if(n <= 0) return;
      double m = logits[0];
      for(int i=1; i<n; i++) if(logits[i] > m) m = logits[i];
      double s = 0.0;
      for(int i=0; i<n; i++)
      {
         probs[i] = MathExp(FXAI_Clamp(logits[i] - m, -40.0, 40.0));
         s += probs[i];
      }
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<n; i++) probs[i] /= s;
   }

   void LayerNormAffine(const double &in[],
                        const double &g[],
                        const double &b[],
                        double &out[]) const
   {
      for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         out[i] = in[i];
      FXAI_ModuleLayerNormAffine(out, FXAI_AI_MLP_HIDDEN, g, b);
   }

   void ResetNorm(void)
   {
      m_norm_ready = false;
      m_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
         m_x_loc[i] = 0.0;
         m_x_scale[i] = 1.0;
      }
   }

   void UpdateNormStats(const double &x[])
   {
      double a = (m_norm_steps < 128 ? 0.05 : 0.0125);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double xi = x[i];

         double d = xi - m_x_mean[i];
         m_x_mean[i] += a * d;
         double d2 = xi - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * d2 * d2;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;

         // Robust running location/scale (quantile-like SGD).
         double sgn = 0.0;
         if(xi > m_x_loc[i]) sgn = 1.0;
         else if(xi < m_x_loc[i]) sgn = -1.0;
         m_x_loc[i] += a * 0.35 * sgn;

         double absdev = MathAbs(xi - m_x_loc[i]);
         m_x_scale[i] = (1.0 - a) * m_x_scale[i] + a * absdev;
         if(m_x_scale[i] < 1e-3) m_x_scale[i] = 1e-3;
      }

      m_norm_steps++;
      if(m_norm_steps >= 32) m_norm_ready = true;
   }

   void NormalizeInput(const double &x[], double &xn[]) const
   {
      xn[0] = 1.0;

      double inst_mean = 0.0;
      int nfeat = FXAI_AI_WEIGHTS - 1;
      if(nfeat < 1) nfeat = 1;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++) inst_mean += x[i];
      inst_mean /= (double)nfeat;

      double inst_var = 0.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - inst_mean;
         inst_var += d * d;
      }
      inst_var /= (double)nfeat;
      double inst_std = MathSqrt(inst_var + 1e-6);

      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double xi = x[i];
         if(!m_norm_ready)
         {
            xn[i] = FXAI_ClipSym(xi, 8.0);
            continue;
         }

         double z_std = (xi - m_x_mean[i]) / MathSqrt(m_x_var[i] + 1e-6);
         double z_rob = (xi - m_x_loc[i]) / MathMax(m_x_scale[i], 1e-3);
         double z_rev = (xi - inst_mean) / MathMax(inst_std, 1e-3);

         // Strict train/infer-consistent blend: standard + robust + RevIN.
         xn[i] = FXAI_ClipSym(0.45 * z_std + 0.35 * z_rob + 0.20 * z_rev, 8.0);
      }
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }

      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;

      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean6 = CurrentWindowFeatureMean(5);
      double first1 = CurrentWindowValue(0, 1);
      double last1  = CurrentWindowValue(win_n - 1, 1);
      double first2 = CurrentWindowValue(0, 2);
      double last2  = CurrentWindowValue(win_n - 1, 2);
      double vol1 = 0.0;
      for(int i=0; i<win_n; i++)
      {
         double d = CurrentWindowValue(i, 1) - mean1;
         vol1 += d * d;
      }
      vol1 = MathSqrt(vol1 / (double)win_n);

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      BuildPackedSequenceTensorCapped(x, SequenceContextSpan(), seq, seq_len, seq_mask, seq_pos_bias, true);
      double attn[];
      double conv_fast[];
      double conv_slow[];
      ArrayResize(attn, FXAI_AI_WEIGHTS);
      ArrayResize(conv_fast, FXAI_AI_WEIGHTS);
      ArrayResize(conv_slow, FXAI_AI_WEIGHTS);
      ArrayInitialize(attn, 0.0);
      ArrayInitialize(conv_fast, 0.0);
      ArrayInitialize(conv_slow, 0.0);
      if(seq_len > 1)
      {
         double query[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            query[k] = seq[seq_len - 1][k];
         double k_fast[3] = {0.58, 0.27, 0.15};
         double k_slow[5] = {0.34, 0.24, 0.18, 0.14, 0.10};
         FXAI_ModuleMultiHeadAttentionSummary(seq, seq_len, query, seq_mask, seq_pos_bias, 2, attn);
         FXAI_ModuleConv1DSummary(seq, seq_len, k_fast, 3, conv_fast);
         FXAI_ModuleConv1DSummary(seq, seq_len, k_slow, 5, conv_slow);
      }

      xa[1] = FXAI_ClipSym(0.50 * xa[1] + 0.20 * mean1 + 0.10 * (first1 - last1) + 0.12 * attn[1] + 0.08 * conv_fast[1], 8.0);
      xa[2] = FXAI_ClipSym(0.50 * xa[2] + 0.20 * mean2 + 0.10 * (first2 - last2) + 0.12 * attn[2] + 0.08 * conv_fast[2], 8.0);
      xa[6] = FXAI_ClipSym(0.55 * xa[6] + 0.25 * vol1 + 0.10 * MathAbs(attn[6]) + 0.10 * MathAbs(conv_slow[6]), 8.0);
      xa[7] = FXAI_ClipSym(0.50 * xa[7] + 0.20 * mean6 + 0.15 * attn[7] + 0.15 * conv_slow[7], 8.0);
      xa[10] = FXAI_ClipSym(0.75 * xa[10] + 0.15 * attn[10] + 0.10 * conv_fast[10], 8.0);
      xa[11] = FXAI_ClipSym(0.75 * xa[11] + 0.15 * attn[11] + 0.10 * conv_slow[11], 8.0);
   }

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FXAI_AF_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_seq_state[t][h] = 0.0;
            m_seq_season[t][h] = 0.0;
            m_seq_trend[t][h] = 0.0;
         }
      }
   }

   int SeqIndexBack(const int back) const
   {
      if(m_seq_len <= 0 || m_seq_ptr < 0) return 0;
      int idx = m_seq_ptr - back;
      while(idx < 0) idx += FXAI_AF_SEQ;
      while(idx >= FXAI_AF_SEQ) idx -= FXAI_AF_SEQ;
      return idx;
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FXAI_AF_TBPTT; t++)
      {
         m_train_cls[t] = FXAI_AF_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_train_x[t][i] = 0.0;
            m_cache_xn[t][i] = 0.0;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[t][h] = 0.0;
            m_cache_trend_raw[t][h] = 0.0;
            m_cache_season_raw[t][h] = 0.0;
            m_cache_trend[t][h] = 0.0;
            m_cache_season[t][h] = 0.0;
            m_cache_final[t][h] = 0.0;
            for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
            {
               m_cache_ma[t][h][k] = 0.0;
               m_cache_mix_alpha[t][h][k] = 0.25;
            }
         }

         for(int b=0; b<FXAI_AF_BLOCKS; b++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               m_cache_blk_in[t][b][h] = 0.0;
               m_cache_blk_pre[t][b][h] = 0.0;
               m_cache_blk_attn[t][b][h] = 0.0;
               m_cache_blk_res1[t][b][h] = 0.0;
               m_cache_blk_ff1[t][b][h] = 0.0;
               m_cache_blk_out[t][b][h] = 0.0;
            }

            for(int hd=0; hd<FXAI_AF_HEADS; hd++)
            {
               for(int d=0; d<FXAI_AF_D_HEAD; d++)
                  m_cache_head_ctx[t][b][hd][d] = 0.0;

               for(int k=0; k<FXAI_AF_TOPK_LAGS; k++)
               {
                  m_cache_lag_idx[t][b][hd][k] = -1;
                  m_cache_lag_w[t][b][hd][k] = 0.0;
                  for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
                     m_cache_lag_src[t][b][hd][LagSrcFlat(k, h)] = 0.0;
               }
            }
         }

         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
            m_cache_probs_raw[t][c] = 1.0 / 3.0;

         for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
         {
            m_cache_h_alpha[t][hz] = 1.0 / (double)FXAI_AF_HORIZONS;
            m_cache_h_mu[t][hz] = 0.0;
            m_cache_h_logv[t][hz] = MathLog(1.0);
            m_cache_h_q25[t][hz] = 0.0;
            m_cache_h_q75[t][hz] = 0.0;
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
               m_cache_h_probs[t][hz][c] = 1.0 / 3.0;
         }
      }
   }

   void ResetCalibrator3(void)
   {
      for(int s=0; s<FXAI_AF_SESSIONS; s++)
      {
         m_cal_temp[s] = 1.0;
         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
         {
            m_cal_bias[s][c] = 0.0;
            for(int b=0; b<FXAI_AF_CAL_BINS; b++)
            {
               m_cal_iso_pos[s][c][b] = 0.0;
               m_cal_iso_cnt[s][c][b] = 0.0;
            }
         }
      }
   }

   void Calibrate3(const int sess,
                   const double &p_raw[],
                   double &p_out[]) const
   {
      int s = ClampI(sess, 0, FXAI_AF_SESSIONS - 1);

      double logits[FXAI_AF_CLASS_COUNT];
      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
      {
         double p = FXAI_Clamp(p_raw[c], 0.001, 0.999);
         double t = FXAI_Clamp(m_cal_temp[s], 0.50, 2.50);
         logits[c] = FXAI_Clamp((MathLog(p) + m_cal_bias[s][c]) / t, -30.0, 30.0);
      }
      double p_temp[FXAI_AF_CLASS_COUNT];
      Softmax3(logits, p_temp);

      double p_iso[FXAI_AF_CLASS_COUNT];
      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_AF_CAL_BINS; b++) total += m_cal_iso_cnt[s][c][b];

         if(total < 25.0)
         {
            p_iso[c] = p_temp[c];
            continue;
         }

         double mono[FXAI_AF_CAL_BINS];
         double prev = 0.5;
         for(int b=0; b<FXAI_AF_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[s][c][b] > 1e-9)
               r = m_cal_iso_pos[s][c][b] / m_cal_iso_cnt[s][c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(b > 0 && r < mono[b - 1]) r = mono[b - 1];
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_temp[c] * (double)FXAI_AF_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_AF_CAL_BINS) bi = FXAI_AF_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      double sumv = 0.0;
      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
      {
         p_out[c] = FXAI_Clamp(0.70 * p_temp[c] + 0.30 * p_iso[c], 0.0005, 0.9990);
         sumv += p_out[c];
      }
      if(sumv <= 0.0) sumv = 1.0;
      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) p_out[c] /= sumv;
   }

   void UpdateCalibrator3(const int sess,
                          const double &p_raw[],
                          const int cls,
                          const double sample_w,
                          const double lr)
   {
      int s = ClampI(sess, 0, FXAI_AF_SESSIONS - 1);
      if(cls < FXAI_AF_SELL || cls > FXAI_AF_SKIP) return;

      double p_cal[FXAI_AF_CLASS_COUNT];
      Calibrate3(s, p_raw, p_cal);

      double w = FXAI_Clamp(sample_w, 0.25, 4.0);
      double ll = FXAI_Clamp(lr, 0.0005, 0.05) * w;

      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double err = target - p_cal[c];
         m_cal_bias[s][c] += ll * err;
         m_cal_bias[s][c] = FXAI_Clamp(m_cal_bias[s][c], -4.0, 4.0);

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_AF_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_AF_CAL_BINS) bi = FXAI_AF_CAL_BINS - 1;
         m_cal_iso_cnt[s][c][bi] += w;
         m_cal_iso_pos[s][c][bi] += w * target;
      }

      double p_true = FXAI_Clamp(p_cal[cls], 0.001, 0.999);
      double delta_t = FXAI_Clamp(-MathLog(p_true) - 0.7, -2.0, 2.0);
      m_cal_temp[s] = FXAI_Clamp(m_cal_temp[s] + 0.01 * ll * delta_t, 0.50, 2.50);
   }

   void ZeroMoments(void)
   {
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         m_m_b_in[o] = 0.0; m_v_b_in[o] = 0.0;
         m_m_b_mix[o] = 0.0; m_v_b_mix[o] = 0.0;
         m_m_b_season[o] = 0.0; m_v_b_season[o] = 0.0;
         m_m_b_trend[o] = 0.0; m_v_b_trend[o] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_m_w_in[o][i] = 0.0;
            m_v_w_in[o][i] = 0.0;
         }

         for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
         {
            m_m_w_mix[o][k] = 0.0;
            m_v_w_mix[o][k] = 0.0;
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            m_m_w_season[o][i] = 0.0; m_v_w_season[o][i] = 0.0;
            m_m_w_trend[o][i] = 0.0; m_v_w_trend[o][i] = 0.0;
         }
      }

      for(int b=0; b<FXAI_AF_BLOCKS; b++)
      {
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            m_m_b_gate[b][o] = 0.0; m_v_b_gate[b][o] = 0.0;
            m_m_b_ff1[b][o] = 0.0; m_v_b_ff1[b][o] = 0.0;
            m_m_b_ff2[b][o] = 0.0; m_v_b_ff2[b][o] = 0.0;

            m_m_ln_pre_g[b][o] = 0.0; m_v_ln_pre_g[b][o] = 0.0;
            m_m_ln_pre_b[b][o] = 0.0; m_v_ln_pre_b[b][o] = 0.0;
            m_m_ln_post1_g[b][o] = 0.0; m_v_ln_post1_g[b][o] = 0.0;
            m_m_ln_post1_b[b][o] = 0.0; m_v_ln_post1_b[b][o] = 0.0;
            m_m_ln_post2_g[b][o] = 0.0; m_v_ln_post2_g[b][o] = 0.0;
            m_m_ln_post2_b[b][o] = 0.0; m_v_ln_post2_b[b][o] = 0.0;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               m_m_w_gate[b][o][i] = 0.0; m_v_w_gate[b][o][i] = 0.0;
               m_m_w_ff1[b][o][i] = 0.0; m_v_w_ff1[b][o][i] = 0.0;
               m_m_w_ff2[b][o][i] = 0.0; m_v_w_ff2[b][o][i] = 0.0;
            }
         }

         for(int hd=0; hd<FXAI_AF_HEADS; hd++)
         {
            for(int d=0; d<FXAI_AF_D_HEAD; d++)
            {
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  m_m_wq[b][hd][d][h] = 0.0; m_v_wq[b][hd][d][h] = 0.0;
                  m_m_wk[b][hd][d][h] = 0.0; m_v_wk[b][hd][d][h] = 0.0;
                  m_m_wv[b][hd][d][h] = 0.0; m_v_wv[b][hd][d][h] = 0.0;
                  m_m_wo[b][hd][h][d] = 0.0; m_v_wo[b][hd][h][d] = 0.0;
               }
            }
         }
      }

      for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
      {
         m_m_b_hgate[hz] = 0.0; m_v_b_hgate[hz] = 0.0;
         m_m_b_mu_h[hz] = 0.0; m_v_b_mu_h[hz] = 0.0;
         m_m_b_logv_h[hz] = 0.0; m_v_b_logv_h[hz] = 0.0;
         m_m_b_q25_h[hz] = 0.0; m_v_b_q25_h[hz] = 0.0;
         m_m_b_q75_h[hz] = 0.0; m_v_b_q75_h[hz] = 0.0;

         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
         {
            m_m_b_cls_h[hz][c] = 0.0;
            m_v_b_cls_h[hz][c] = 0.0;
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            m_m_w_hgate[hz][i] = 0.0; m_v_w_hgate[hz][i] = 0.0;
            m_m_w_mu_h[hz][i] = 0.0; m_v_w_mu_h[hz][i] = 0.0;
            m_m_w_logv_h[hz][i] = 0.0; m_v_w_logv_h[hz][i] = 0.0;
            m_m_w_q25_h[hz][i] = 0.0; m_v_w_q25_h[hz][i] = 0.0;
            m_m_w_q75_h[hz][i] = 0.0; m_v_w_q75_h[hz][i] = 0.0;

            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
            {
               m_m_w_cls_h[hz][c][i] = 0.0;
               m_v_w_cls_h[hz][c][i] = 0.0;
            }
         }
      }

      m_beta1_pow = 1.0;
      m_beta2_pow = 1.0;
      m_adam_t = 0;
   }

   void AdamStepInit(void)
   {
      m_adam_t++;
      m_beta1_pow *= 0.9;
      m_beta2_pow *= 0.999;
      if(m_beta1_pow < 1e-12) m_beta1_pow = 1e-12;
      if(m_beta2_pow < 1e-12) m_beta2_pow = 1e-12;
   }

   void AdamWUpdate(double &w,
                    double &m,
                    double &v,
                    const double grad,
                    const double lr,
                    const double wd)
   {
      FXAI_OptAdamWStep(w, m, v, grad, lr, 0.90, 0.999, wd, MathMax(m_adam_t, 1));
   }

   void UpdateShadowHeads(void)
   {
      double decay = 0.995;
      if(m_adam_t < 64) decay = 0.98;

      for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
      {
         if(!m_shadow_ready)
         {
            m_sh_b_hgate[hz] = m_b_hgate[hz];
            m_sh_b_mu_h[hz] = m_b_mu_h[hz];
            m_sh_b_logv_h[hz] = m_b_logv_h[hz];
            m_sh_b_q25_h[hz] = m_b_q25_h[hz];
            m_sh_b_q75_h[hz] = m_b_q75_h[hz];
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
               m_sh_b_cls_h[hz][c] = m_b_cls_h[hz][c];
         }
         else
         {
            m_sh_b_hgate[hz] = decay * m_sh_b_hgate[hz] + (1.0 - decay) * m_b_hgate[hz];
            m_sh_b_mu_h[hz] = decay * m_sh_b_mu_h[hz] + (1.0 - decay) * m_b_mu_h[hz];
            m_sh_b_logv_h[hz] = decay * m_sh_b_logv_h[hz] + (1.0 - decay) * m_b_logv_h[hz];
            m_sh_b_q25_h[hz] = decay * m_sh_b_q25_h[hz] + (1.0 - decay) * m_b_q25_h[hz];
            m_sh_b_q75_h[hz] = decay * m_sh_b_q75_h[hz] + (1.0 - decay) * m_b_q75_h[hz];
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
               m_sh_b_cls_h[hz][c] = decay * m_sh_b_cls_h[hz][c] + (1.0 - decay) * m_b_cls_h[hz][c];
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            if(!m_shadow_ready)
            {
               m_sh_w_hgate[hz][i] = m_w_hgate[hz][i];
               m_sh_w_mu_h[hz][i] = m_w_mu_h[hz][i];
               m_sh_w_logv_h[hz][i] = m_w_logv_h[hz][i];
               m_sh_w_q25_h[hz][i] = m_w_q25_h[hz][i];
               m_sh_w_q75_h[hz][i] = m_w_q75_h[hz][i];
               for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
                  m_sh_w_cls_h[hz][c][i] = m_w_cls_h[hz][c][i];
            }
            else
            {
               m_sh_w_hgate[hz][i] = decay * m_sh_w_hgate[hz][i] + (1.0 - decay) * m_w_hgate[hz][i];
               m_sh_w_mu_h[hz][i] = decay * m_sh_w_mu_h[hz][i] + (1.0 - decay) * m_w_mu_h[hz][i];
               m_sh_w_logv_h[hz][i] = decay * m_sh_w_logv_h[hz][i] + (1.0 - decay) * m_w_logv_h[hz][i];
               m_sh_w_q25_h[hz][i] = decay * m_sh_w_q25_h[hz][i] + (1.0 - decay) * m_w_q25_h[hz][i];
               m_sh_w_q75_h[hz][i] = decay * m_sh_w_q75_h[hz][i] + (1.0 - decay) * m_w_q75_h[hz][i];
               for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
                  m_sh_w_cls_h[hz][c][i] = decay * m_sh_w_cls_h[hz][c][i] + (1.0 - decay) * m_w_cls_h[hz][c][i];
            }
         }
      }

      if(m_adam_t >= 32) m_shadow_ready = true;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetTrainBuffer();
      ResetNorm();
      ResetCalibrator3();
      ZeroMoments();

      m_step = 0;
      m_printed_reco = false;
      m_shadow_ready = false;

      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         m_b_in[o] = 0.0;
         m_b_mix[o] = 0.0;
         m_b_season[o] = 0.0;
         m_b_trend[o] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s = (double)((o + 1) * (i + 2));
            m_w_in[o][i] = 0.03 * MathSin(0.57 * s);
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            double a = (double)((o + 1) * (i + 3));
            m_w_season[o][i] = 0.03 * MathCos(0.71 * a);
            m_w_trend[o][i] = 0.03 * MathSin(0.73 * a);
         }

         for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
            m_w_mix[o][k] = 0.1 * MathSin((double)((o + 1) * (k + 1)) * 0.39);
      }

      for(int b=0; b<FXAI_AF_BLOCKS; b++)
      {
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            m_b_gate[b][o] = 0.0;
            m_b_ff1[b][o] = 0.0;
            m_b_ff2[b][o] = 0.0;

            m_ln_pre_g[b][o] = 1.0; m_ln_pre_b[b][o] = 0.0;
            m_ln_post1_g[b][o] = 1.0; m_ln_post1_b[b][o] = 0.0;
            m_ln_post2_g[b][o] = 1.0; m_ln_post2_b[b][o] = 0.0;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               double a = (double)((b + 1) * (o + 1) * (i + 2));
               m_w_gate[b][o][i] = 0.03 * MathSin(0.55 * a);
               m_w_ff1[b][o][i] = 0.04 * MathCos(0.59 * a);
               m_w_ff2[b][o][i] = 0.04 * MathSin(0.61 * a);
            }
         }

         for(int hd=0; hd<FXAI_AF_HEADS; hd++)
         {
            for(int d=0; d<FXAI_AF_D_HEAD; d++)
            {
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  double t = (double)((b + 1) * (hd + 2) * (d + 3) * (h + 4));
                  m_wq[b][hd][d][h] = 0.03 * MathSin(0.67 * t);
                  m_wk[b][hd][d][h] = 0.03 * MathCos(0.71 * t);
                  m_wv[b][hd][d][h] = 0.03 * MathSin(0.73 * t);
                  m_wo[b][hd][h][d] = 0.03 * MathCos(0.79 * t);
               }
            }
         }
      }

      for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
      {
         m_b_hgate[hz] = 0.0;
         m_b_mu_h[hz] = 0.0;
         m_b_logv_h[hz] = MathLog(1.0);
         m_b_q25_h[hz] = 0.0;
         m_b_q75_h[hz] = 0.0;

         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
            m_b_cls_h[hz][c] = 0.0;

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            double a = (double)((hz + 1) * (i + 2));
            m_w_hgate[hz][i] = 0.04 * MathSin(0.43 * a);
            m_w_mu_h[hz][i] = 0.03 * MathSin(0.47 * a);
            m_w_logv_h[hz][i] = 0.03 * MathCos(0.53 * a);
            m_w_q25_h[hz][i] = 0.03 * MathSin(0.59 * a);
            m_w_q75_h[hz][i] = 0.03 * MathCos(0.61 * a);

            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
               m_w_cls_h[hz][c][i] = 0.03 * MathSin((double)((hz + 1) * (c + 2) * (i + 1)) * 0.69);
         }
      }

      m_initialized = true;
   }

   int NextPow2(const int n) const
   {
      int p = 1;
      while(p < n) p <<= 1;
      if(p < 2) p = 2;
      return p;
   }

   void FFT(double &re[], double &im[], const int n, const bool inverse) const
   {
      int j = 0;
      for(int i=1; i<n; i++)
      {
         int bit = n >> 1;
         while((j & bit) != 0)
         {
            j ^= bit;
            bit >>= 1;
         }
         j ^= bit;
         if(i < j)
         {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
         }
      }

      for(int len=2; len<=n; len<<=1)
      {
         double ang = 2.0 * FXAI_AF_PI / (double)len;
         if(!inverse) ang = -ang;
         double wlen_re = MathCos(ang);
         double wlen_im = MathSin(ang);

         for(int i=0; i<n; i+=len)
         {
            double w_re = 1.0;
            double w_im = 0.0;
            int half = len >> 1;
            for(int k=0; k<half; k++)
            {
               int u = i + k;
               int v = i + k + half;

               double vr = re[v] * w_re - im[v] * w_im;
               double vi = re[v] * w_im + im[v] * w_re;

               re[v] = re[u] - vr;
               im[v] = im[u] - vi;
               re[u] += vr;
               im[u] += vi;

               double nwr = w_re * wlen_re - w_im * wlen_im;
               double nwi = w_re * wlen_im + w_im * wlen_re;
               w_re = nwr;
               w_im = nwi;
            }
         }
      }

      if(inverse)
      {
         for(int i=0; i<n; i++)
         {
            re[i] /= (double)n;
            im[i] /= (double)n;
         }
      }
   }

   void AutoCorrFFT(const double &seq[], const int n, double &ac[]) const
   {
      if(n <= 1)
      {
         for(int i=0; i<n; i++) ac[i] = 0.0;
         return;
      }

      int nfft = NextPow2(2 * n);
      double re[];
      double im[];
      ArrayResize(re, nfft);
      ArrayResize(im, nfft);
      for(int i=0; i<nfft; i++) { re[i] = 0.0; im[i] = 0.0; }
      for(int i=0; i<n; i++) re[i] = seq[i];

      FFT(re, im, nfft, false);
      for(int i=0; i<nfft; i++)
      {
         double a = re[i];
         double b = im[i];
         re[i] = a * a + b * b;
         im[i] = 0.0;
      }
      FFT(re, im, nfft, true);

      for(int i=0; i<n; i++)
         ac[i] = re[i];
   }

   int ResolveClass(const int y, const double &x[], const double move_points) const
   {
      if(y >= FXAI_AF_SELL && y <= FXAI_AF_SKIP) return y;
      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      if(edge <= 0.10 + 0.25 * MathMax(cost, 0.0)) return FXAI_AF_SKIP;
      if(y > 0) return FXAI_AF_BUY;
      if(y == 0) return FXAI_AF_SELL;
      return (move_points >= 0.0 ? FXAI_AF_BUY : FXAI_AF_SELL);
   }

   double ClassWeight(const int cls, const double move_points, const double cost, const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double w = FXAI_Clamp(sample_w, 0.25, 5.0);
      if(cls == FXAI_AF_SKIP)
      {
         if(edge <= 0.0) return FXAI_Clamp(w * 1.5, 0.25, 8.0);
         return FXAI_Clamp(w * 0.70, 0.25, 8.0);
      }
      if(edge <= 0.0) return FXAI_Clamp(w * 0.55, 0.25, 8.0);
      return FXAI_Clamp(w * (1.0 + 0.07 * MathMin(edge, 20.0)), 0.25, 8.0);
   }

   double MoveWeight(const double move_points, const double cost, const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double e = FXAI_Clamp(0.5 + edge / denom, 0.25, 4.0);
      return FXAI_Clamp(sample_w * e, 0.25, 8.0);
   }

   void BuildEmbedding(const double &xn[], double &embed[]) const
   {
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double z = m_b_in[o];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) z += m_w_in[o][i] * xn[i];
         embed[o] = FXAI_Tanh(z);
      }
   }

   void Decompose(const double &embed[],
                  const bool cache_on,
                  const int cache_t,
                  double &trend_raw[],
                  double &season_raw[],
                  double &trend[],
                  double &season[])
   {
      int wins[FXAI_AF_MA_KERNELS] = {3, 5, 9, 17};

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double ma[FXAI_AF_MA_KERNELS];
         for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
         {
            int w = wins[k];
            if(w < 2) w = 2;
            if(w > FXAI_AF_SEQ) w = FXAI_AF_SEQ;

            double sum = embed[h];
            int cnt = 1;
            for(int b=0; b<w - 1 && b<m_seq_len; b++)
            {
               int idx = SeqIndexBack(b);
               sum += m_seq_state[idx][h];
               cnt++;
            }
            ma[k] = sum / (double)MathMax(cnt, 1);
            if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
               m_cache_ma[cache_t][h][k] = ma[k];
         }

         double logits_k[FXAI_AF_MA_KERNELS];
         double alpha_k[FXAI_AF_MA_KERNELS];
         for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
            logits_k[k] = m_b_mix[h] + m_w_mix[h][k];
         SoftmaxN(logits_k, FXAI_AF_MA_KERNELS, alpha_k);

         double tr = 0.0;
         for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
         {
            tr += alpha_k[k] * ma[k];
            if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
               m_cache_mix_alpha[cache_t][h][k] = alpha_k[k];
         }

         trend_raw[h] = tr;
         season_raw[h] = embed[h] - tr;
      }

      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double zt = m_b_trend[o];
         double zs = m_b_season[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            zt += m_w_trend[o][i] * trend_raw[i];
            zs += m_w_season[o][i] * season_raw[i];
         }
         trend[o] = FXAI_Tanh(zt);
         season[o] = FXAI_Tanh(zs);
      }
   }

   void BuildTokens(const double &cur[],
                    double &tokens[][FXAI_AI_MLP_HIDDEN],
                    int &count) const
   {
      int keep = m_seq_len;
      if(keep > FXAI_AF_SEQ - 1) keep = FXAI_AF_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = SeqIndexBack(back);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            tokens[i][h] = m_seq_state[idx][h];
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         tokens[keep][h] = cur[h];

      count = keep + 1;
   }

   void AutoCorrelationAttentionFFT(const int block_id,
                                    const double &x_pre[],
                                    const double &tokens[][FXAI_AI_MLP_HIDDEN],
                                    const int token_count,
                                    const bool cache_on,
                                    const int cache_t,
                                    double &attn_out[])
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) attn_out[h] = 0.0;
      if(token_count <= 1) return;

      double vproj[FXAI_AF_SEQ][FXAI_AF_D_HEAD];
      double sc_seq[FXAI_AF_SEQ];
      double ac[FXAI_AF_SEQ];

      for(int hd=0; hd<FXAI_AF_HEADS; hd++)
      {
         double q[FXAI_AF_D_HEAD];
         for(int d=0; d<FXAI_AF_D_HEAD; d++)
         {
            double s = 0.0;
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               s += m_wq[block_id][hd][d][j] * x_pre[j];
            q[d] = s;
         }

         for(int t=0; t<token_count; t++)
         {
            double sc = 0.0;
            for(int d=0; d<FXAI_AF_D_HEAD; d++)
            {
               double kv = 0.0;
               double vv = 0.0;
               for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               {
                  kv += m_wk[block_id][hd][d][j] * tokens[t][j];
                  vv += m_wv[block_id][hd][d][j] * tokens[t][j];
               }
               sc += q[d] * kv;
               vproj[t][d] = vv;
            }
            sc_seq[t] = sc / MathSqrt((double)FXAI_AF_D_HEAD);
         }

         AutoCorrFFT(sc_seq, token_count, ac);

         // Period discovery by top-k positive lags.
         int lag_idx[FXAI_AF_TOPK_LAGS];
         double lag_sc[FXAI_AF_TOPK_LAGS];
         for(int k=0; k<FXAI_AF_TOPK_LAGS; k++) { lag_idx[k] = -1; lag_sc[k] = -1e100; }

         for(int lag=1; lag<token_count; lag++)
         {
            double s = ac[lag];
            int worst = 0;
            for(int k=1; k<FXAI_AF_TOPK_LAGS; k++)
               if(lag_sc[k] < lag_sc[worst]) worst = k;
            if(s > lag_sc[worst])
            {
               lag_sc[worst] = s;
               lag_idx[worst] = lag;
            }
         }

         double mx = -1e100;
         for(int k=0; k<FXAI_AF_TOPK_LAGS; k++)
            if(lag_idx[k] >= 0 && lag_sc[k] > mx) mx = lag_sc[k];
         if(mx < -1e50) mx = 0.0;

         double alpha[FXAI_AF_TOPK_LAGS];
         double den = 0.0;
         for(int k=0; k<FXAI_AF_TOPK_LAGS; k++)
         {
            if(lag_idx[k] < 0) { alpha[k] = 0.0; continue; }
            alpha[k] = MathExp(FXAI_ClipSym(lag_sc[k] - mx, 30.0));
            den += alpha[k];
         }
         if(den <= 0.0) den = 1.0;
         for(int k=0; k<FXAI_AF_TOPK_LAGS; k++) alpha[k] /= den;

         double ctx[FXAI_AF_D_HEAD];
         for(int d=0; d<FXAI_AF_D_HEAD; d++) ctx[d] = 0.0;

         for(int k=0; k<FXAI_AF_TOPK_LAGS; k++)
         {
            int lag = lag_idx[k];
            if(lag <= 0) continue;
            int src = token_count - 1 - lag;
            while(src < 0) src += token_count;
            while(src >= token_count) src -= token_count;

            double a = alpha[k];
            for(int d=0; d<FXAI_AF_D_HEAD; d++)
               ctx[d] += a * vproj[src][d];

            if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
            {
               m_cache_lag_idx[cache_t][block_id][hd][k] = lag;
               m_cache_lag_w[cache_t][block_id][hd][k] = a;
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
                  m_cache_lag_src[cache_t][block_id][hd][LagSrcFlat(k, h)] = tokens[src][h];
            }
         }

         if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
            for(int d=0; d<FXAI_AF_D_HEAD; d++) m_cache_head_ctx[cache_t][block_id][hd][d] = ctx[d];

         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double s = 0.0;
            for(int d=0; d<FXAI_AF_D_HEAD; d++)
               s += m_wo[block_id][hd][o][d] * ctx[d];
            attn_out[o] += s;
         }
      }
   }

   void ComputeHeads(const double &final_state[],
                     const bool use_shadow,
                     const bool cache_on,
                     const int cache_t,
                     double &probs_raw[],
                     double &mu_out,
                     double &logv_out,
                     double &q25_out,
                     double &q75_out)
   {
      double gate_logits[FXAI_AF_HORIZONS];
      double alpha[FXAI_AF_HORIZONS];
      for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
      {
         double z = (use_shadow ? m_sh_b_hgate[hz] : m_b_hgate[hz]);
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z += (use_shadow ? m_sh_w_hgate[hz][i] : m_w_hgate[hz][i]) * final_state[i];
         gate_logits[hz] = z;
      }
      SoftmaxN(gate_logits, FXAI_AF_HORIZONS, alpha);

      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) probs_raw[c] = 0.0;
      mu_out = 0.0;
      logv_out = 0.0;
      q25_out = 0.0;
      q75_out = 0.0;

      for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
      {
         double logits_h[FXAI_AF_CLASS_COUNT];
         double probs_h[FXAI_AF_CLASS_COUNT];

         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
         {
            double z = (use_shadow ? m_sh_b_cls_h[hz][c] : m_b_cls_h[hz][c]);
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               z += (use_shadow ? m_sh_w_cls_h[hz][c][i] : m_w_cls_h[hz][c][i]) * final_state[i];
            logits_h[c] = FXAI_ClipSym(z, 20.0);
         }
         Softmax3(logits_h, probs_h);

         double mu = (use_shadow ? m_sh_b_mu_h[hz] : m_b_mu_h[hz]);
         double lv = (use_shadow ? m_sh_b_logv_h[hz] : m_b_logv_h[hz]);
         double q25 = (use_shadow ? m_sh_b_q25_h[hz] : m_b_q25_h[hz]);
         double q75 = (use_shadow ? m_sh_b_q75_h[hz] : m_b_q75_h[hz]);

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            mu += (use_shadow ? m_sh_w_mu_h[hz][i] : m_w_mu_h[hz][i]) * final_state[i];
            lv += (use_shadow ? m_sh_w_logv_h[hz][i] : m_w_logv_h[hz][i]) * final_state[i];
            q25 += (use_shadow ? m_sh_w_q25_h[hz][i] : m_w_q25_h[hz][i]) * final_state[i];
            q75 += (use_shadow ? m_sh_w_q75_h[hz][i] : m_w_q75_h[hz][i]) * final_state[i];
         }

         if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
         {
            m_cache_h_alpha[cache_t][hz] = alpha[hz];
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
               m_cache_h_probs[cache_t][hz][c] = probs_h[c];
            m_cache_h_mu[cache_t][hz] = mu;
            m_cache_h_logv[cache_t][hz] = FXAI_Clamp(lv, -4.0, 4.0);
            if(q25 <= q75)
            {
               m_cache_h_q25[cache_t][hz] = q25;
               m_cache_h_q75[cache_t][hz] = q75;
            }
            else
            {
               m_cache_h_q25[cache_t][hz] = q75;
               m_cache_h_q75[cache_t][hz] = q25;
            }
         }

         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
            probs_raw[c] += alpha[hz] * probs_h[c];
         mu_out += alpha[hz] * mu;
         logv_out += alpha[hz] * FXAI_Clamp(lv, -4.0, 4.0);
         q25_out += alpha[hz] * q25;
         q75_out += alpha[hz] * q75;
      }

      if(q25_out > q75_out)
      {
         double t = q25_out;
         q25_out = q75_out;
         q75_out = t;
      }

      double s = probs_raw[0] + probs_raw[1] + probs_raw[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) probs_raw[c] /= s;
   }

   void ForwardStep(const double &x[],
                    const bool commit,
                    const bool cache_on,
                    const int cache_t,
                    double &final_out[],
                    double &probs_raw[],
                    double &mu,
                    double &logv,
                    double &q25,
                    double &q75)
   {
      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double embed[FXAI_AI_MLP_HIDDEN];
      BuildEmbedding(xn, embed);

      double trend_raw[FXAI_AI_MLP_HIDDEN];
      double season_raw[FXAI_AI_MLP_HIDDEN];
      double trend[FXAI_AI_MLP_HIDDEN];
      double season[FXAI_AI_MLP_HIDDEN];
      Decompose(embed, cache_on, cache_t, trend_raw, season_raw, trend, season);

      double state[FXAI_AI_MLP_HIDDEN];
      for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         state[i] = season[i] + 0.25 * trend[i];

      if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_cache_xn[cache_t][i] = xn[i];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[cache_t][h] = embed[h];
            m_cache_trend_raw[cache_t][h] = trend_raw[h];
            m_cache_season_raw[cache_t][h] = season_raw[h];
            m_cache_trend[cache_t][h] = trend[h];
            m_cache_season[cache_t][h] = season[h];
         }
      }

      double tokens[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
      int token_count = 0;

      for(int b=0; b<FXAI_AF_BLOCKS; b++)
      {
         double blk_in[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) blk_in[i] = state[i];

         double pre[FXAI_AI_MLP_HIDDEN];
         double ln_pre_g[FXAI_AI_MLP_HIDDEN];
         double ln_pre_b[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            ln_pre_g[i] = m_ln_pre_g[b][i];
            ln_pre_b[i] = m_ln_pre_b[b][i];
         }
         LayerNormAffine(blk_in, ln_pre_g, ln_pre_b, pre);

         BuildTokens(pre, tokens, token_count);

         double attn[FXAI_AI_MLP_HIDDEN];
         AutoCorrelationAttentionFFT(b, pre, tokens, token_count, cache_on, cache_t, attn);

         double gate[FXAI_AI_MLP_HIDDEN];
         double fused[FXAI_AI_MLP_HIDDEN];
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double zg = m_b_gate[b][o];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               zg += m_w_gate[b][o][i] * pre[i];
            gate[o] = FXAI_Sigmoid(zg);
            fused[o] = gate[o] * attn[o] + (1.0 - gate[o]) * pre[o];
         }

         double res1[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            res1[i] = blk_in[i] + fused[i];

         double n1[FXAI_AI_MLP_HIDDEN];
         double ln_post1_g[FXAI_AI_MLP_HIDDEN];
         double ln_post1_b[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            ln_post1_g[i] = m_ln_post1_g[b][i];
            ln_post1_b[i] = m_ln_post1_b[b][i];
         }
         LayerNormAffine(res1, ln_post1_g, ln_post1_b, n1);

         double ff1[FXAI_AI_MLP_HIDDEN];
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double z = m_b_ff1[b][o];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) z += m_w_ff1[b][o][i] * n1[i];
            ff1[o] = FXAI_Tanh(z);
         }

         double ff2[FXAI_AI_MLP_HIDDEN];
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double z = m_b_ff2[b][o];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) z += m_w_ff2[b][o][i] * ff1[i];
            ff2[o] = z;
         }

         double out_pre[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            out_pre[i] = res1[i] + 0.30 * FXAI_ClipSym(ff2[i], 8.0);

         double ln_post2_g[FXAI_AI_MLP_HIDDEN];
         double ln_post2_b[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            ln_post2_g[i] = m_ln_post2_g[b][i];
            ln_post2_b[i] = m_ln_post2_b[b][i];
         }
         LayerNormAffine(out_pre, ln_post2_g, ln_post2_b, state);

         if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               m_cache_blk_in[cache_t][b][h] = blk_in[h];
               m_cache_blk_pre[cache_t][b][h] = pre[h];
               m_cache_blk_attn[cache_t][b][h] = attn[h];
               m_cache_blk_res1[cache_t][b][h] = res1[h];
               m_cache_blk_ff1[cache_t][b][h] = ff1[h];
               m_cache_blk_out[cache_t][b][h] = state[h];
            }
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         final_out[h] = state[h] + 0.30 * trend[h];
         if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
            m_cache_final[cache_t][h] = final_out[h];
      }

      ComputeHeads(final_out,
                   (m_shadow_ready && !cache_on),
                   cache_on,
                   cache_t,
                   probs_raw,
                   mu,
                   logv,
                   q25,
                   q75);

      if(cache_on && cache_t >= 0 && cache_t < FXAI_AF_TBPTT)
         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) m_cache_probs_raw[cache_t][c] = probs_raw[c];

      if(commit)
      {
         m_seq_ptr++;
         if(m_seq_ptr >= FXAI_AF_SEQ) m_seq_ptr = 0;

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_seq_state[m_seq_ptr][h] = final_out[h];
            m_seq_season[m_seq_ptr][h] = season[h];
            m_seq_trend[m_seq_ptr][h] = trend[h];
         }

         if(m_seq_len < FXAI_AF_SEQ) m_seq_len++;
      }
   }

   void AppendTrainSample(const int cls,
                          const double &x[],
                          const double move_points,
                          const double cost_points,
                          const double sample_w)
   {
      if(m_train_len < FXAI_AF_TBPTT)
      {
         int p = m_train_len;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[p][i] = x[i];
         m_train_cls[p] = cls;
         m_train_move[p] = move_points;
         m_train_cost[p] = cost_points;
         m_train_w[p] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FXAI_AF_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[FXAI_AF_TBPTT - 1][i] = x[i];
      m_train_cls[FXAI_AF_TBPTT - 1] = cls;
      m_train_move[FXAI_AF_TBPTT - 1] = move_points;
      m_train_cost[FXAI_AF_TBPTT - 1] = cost_points;
      m_train_w[FXAI_AF_TBPTT - 1] = sample_w;
   }

   double LearningRateCosine(const double lr0) const
   {
      double p = (double)(m_step % FXAI_AF_COS_CYCLE) / (double)FXAI_AF_COS_CYCLE;
      double c = 0.5 * (1.0 + MathCos(2.0 * FXAI_AF_PI * p));
      return FXAI_Clamp(lr0 * (0.12 + 0.88 * c), 0.00005, 0.50);
   }

   void TrainTBPTT(const FXAIAIHyperParams &hp)
   {
      if(m_train_len <= 0) return;

      int n = m_train_len;
      if(n > FXAI_AF_TBPTT) n = FXAI_AF_TBPTT;
      if(n <= 0) return;

      // Save live sequence and isolate TBPTT replay to avoid state bleed.
      int saved_ptr = m_seq_ptr;
      int saved_len = m_seq_len;
      double saved_state[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
      double saved_season[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
      double saved_trend[FXAI_AF_SEQ][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<FXAI_AF_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            saved_state[t][h] = m_seq_state[t][h];
            saved_season[t][h] = m_seq_season[t][h];
            saved_trend[t][h] = m_seq_trend[t][h];
         }
      }
      ResetSequence();

      // Forward over TBPTT window.
      for(int t=0; t<n; t++)
      {
         double final_state[FXAI_AI_MLP_HIDDEN];
         double probs_raw[FXAI_AF_CLASS_COUNT];
         double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
         double xrow[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xrow[i] = m_train_x[t][i];
         ForwardStep(xrow, true, true, t, final_state, probs_raw, mu, lv, q25, q75);
      }

      // Backward with full-path parameter updates over all trainable groups.
      double next_delta[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) next_delta[h] = 0.0;

      double hz_loss_w[FXAI_AF_HORIZONS] = {0.45, 0.35, 0.20};
      double hz_move_sc[FXAI_AF_HORIZONS] = {0.75, 1.00, 1.25};

      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.20);
      double lr0 = FXAI_Clamp(hp.lr, 0.0001, 1.0);

      for(int t=n - 1; t>=0; t--)
      {
         m_step++;
         AdamStepInit();

         double lr = LearningRateCosine(lr0);

         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FXAI_Clamp(m_train_w[t], 0.25, 6.0);
         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);

         // Aggregate delta from all horizon heads (joint objective).
         double delta_final[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) delta_final[i] = 0.30 * next_delta[i];

         double head_loss[FXAI_AF_HORIZONS];
         double head_loss_mean = 0.0;

         for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
         {
            double alpha = FXAI_Clamp(m_cache_h_alpha[t][hz], 0.001, 0.999);
            double wl = w_cls * hz_loss_w[hz] * alpha;
            double wm = w_mv * hz_loss_w[hz] * alpha;

            double probs_h[FXAI_AF_CLASS_COUNT];
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) probs_h[c] = m_cache_h_probs[t][hz][c];

            double ce = 0.0;
            for(int c=0; c<FXAI_AF_CLASS_COUNT; c++)
            {
               double target = (c == cls ? 1.0 : 0.0);
               double p = FXAI_Clamp(probs_h[c], 1e-6, 1.0);
               if(target > 0.5) ce = -MathLog(p);

               double g = (probs_h[c] - target) * wl;
               g = FXAI_ClipSym(g, 4.0);

               AdamWUpdate(m_b_cls_h[hz][c], m_m_b_cls_h[hz][c], m_v_b_cls_h[hz][c], g, lr, l2);
               for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               {
                  double gx = g * m_cache_final[t][i];
                  AdamWUpdate(m_w_cls_h[hz][c][i], m_m_w_cls_h[hz][c][i], m_v_w_cls_h[hz][c][i], gx, lr, l2);
                  delta_final[i] += g * m_w_cls_h[hz][c][i];
               }
            }

            double mu = m_cache_h_mu[t][hz];
            double lv = m_cache_h_logv[t][hz];
            double q25 = m_cache_h_q25[t][hz];
            double q75 = m_cache_h_q75[t][hz];
            double target_mv = mv * hz_move_sc[hz];

            double s2 = MathExp(FXAI_Clamp(lv, -4.0, 4.0));
            s2 = FXAI_Clamp(s2, 0.05, 100.0);
            double err = FXAI_ClipSym(mu - target_mv, 30.0);

            double g_mu = FXAI_ClipSym((err / s2) * wm, 4.0);
            double g_lv = FXAI_ClipSym(0.5 * (1.0 - (err * err) / s2) * wm, 4.0);
            double e25 = target_mv - q25;
            double e75 = target_mv - q75;
            double g_q25 = FXAI_ClipSym(((e25 < 0.0 ? -0.75 : 0.25)) * wm, 3.5);
            double g_q75 = FXAI_ClipSym(((e75 < 0.0 ? -0.25 : 0.75)) * wm, 3.5);

            AdamWUpdate(m_b_mu_h[hz], m_m_b_mu_h[hz], m_v_b_mu_h[hz], g_mu, lr, l2);
            AdamWUpdate(m_b_logv_h[hz], m_m_b_logv_h[hz], m_v_b_logv_h[hz], g_lv, lr, l2);
            AdamWUpdate(m_b_q25_h[hz], m_m_b_q25_h[hz], m_v_b_q25_h[hz], g_q25, lr, l2);
            AdamWUpdate(m_b_q75_h[hz], m_m_b_q75_h[hz], m_v_b_q75_h[hz], g_q75, lr, l2);

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               double xi = m_cache_final[t][i];
               AdamWUpdate(m_w_mu_h[hz][i], m_m_w_mu_h[hz][i], m_v_w_mu_h[hz][i], g_mu * xi, lr, l2);
               AdamWUpdate(m_w_logv_h[hz][i], m_m_w_logv_h[hz][i], m_v_w_logv_h[hz][i], g_lv * xi, lr, l2);
               AdamWUpdate(m_w_q25_h[hz][i], m_m_w_q25_h[hz][i], m_v_w_q25_h[hz][i], g_q25 * xi, lr, l2);
               AdamWUpdate(m_w_q75_h[hz][i], m_m_w_q75_h[hz][i], m_v_w_q75_h[hz][i], g_q75 * xi, lr, l2);

               delta_final[i] += g_mu * m_w_mu_h[hz][i] + g_lv * m_w_logv_h[hz][i] +
                                 g_q25 * m_w_q25_h[hz][i] + g_q75 * m_w_q75_h[hz][i];
            }

            head_loss[hz] = ce + 0.05 * MathAbs(err);
            head_loss_mean += alpha * head_loss[hz];
         }

         // Horizon gate gradient.
         for(int hz=0; hz<FXAI_AF_HORIZONS; hz++)
         {
            double a = FXAI_Clamp(m_cache_h_alpha[t][hz], 0.001, 0.999);
            double g_gate = FXAI_ClipSym((head_loss[hz] - head_loss_mean) * a * (1.0 - a) * w_cls, 3.0);
            AdamWUpdate(m_b_hgate[hz], m_m_b_hgate[hz], m_v_b_hgate[hz], g_gate, lr, l2);
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               double gi = g_gate * m_cache_final[t][i];
               AdamWUpdate(m_w_hgate[hz][i], m_m_w_hgate[hz][i], m_v_w_hgate[hz][i], gi, lr, l2);
               delta_final[i] += g_gate * m_w_hgate[hz][i];
            }
         }

         // Global gradient norm clipping before lower-layer updates.
         double gn2 = 0.0;
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) gn2 += delta_final[i] * delta_final[i];
         double gn = MathSqrt(gn2);
         double gscale = 1.0;
         if(gn > 5.0) gscale = 5.0 / gn;
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) delta_final[i] *= gscale;

         // Backprop through stacked blocks.
         double delta_cur[FXAI_AI_MLP_HIDDEN];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) delta_cur[i] = delta_final[i];

         for(int b=FXAI_AF_BLOCKS - 1; b>=0; b--)
         {
            // Approximate layernorm derivative via affine scale.
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               delta_cur[i] *= m_ln_post2_g[b][i];

            double d_res1[FXAI_AI_MLP_HIDDEN];
            double d_ff1[FXAI_AI_MLP_HIDDEN];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) { d_res1[i] = delta_cur[i]; d_ff1[i] = 0.0; }

            // FF2 path.
            for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
            {
               double g_b = 0.30 * delta_cur[o];
               AdamWUpdate(m_b_ff2[b][o], m_m_b_ff2[b][o], m_v_b_ff2[b][o], g_b, lr, l2);

               for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               {
                  double g_w = g_b * m_cache_blk_ff1[t][b][i];
                  AdamWUpdate(m_w_ff2[b][o][i], m_m_w_ff2[b][o][i], m_v_w_ff2[b][o][i], g_w, lr, l2);
                  d_ff1[i] += g_b * m_w_ff2[b][o][i];
               }
            }

            // FF1 path.
            for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
            {
               double a1 = m_cache_blk_ff1[t][b][o];
               double dpre = d_ff1[o] * (1.0 - a1 * a1);
               AdamWUpdate(m_b_ff1[b][o], m_m_b_ff1[b][o], m_v_b_ff1[b][o], dpre, lr, l2);
               for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               {
                  double g_w = dpre * m_cache_blk_res1[t][b][i];
                  AdamWUpdate(m_w_ff1[b][o][i], m_m_w_ff1[b][o][i], m_v_w_ff1[b][o][i], g_w, lr, l2);
                  d_res1[i] += dpre * m_w_ff1[b][o][i];
               }
            }

            // Residual split to input and gated attention.
            double d_fused[FXAI_AI_MLP_HIDDEN];
            double d_pre[FXAI_AI_MLP_HIDDEN];
            double d_attn[FXAI_AI_MLP_HIDDEN];
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               d_fused[i] = d_res1[i] * m_ln_post1_g[b][i];
               double pre_i = m_cache_blk_pre[t][b][i];
               double attn_i = m_cache_blk_attn[t][b][i];
               double fused_i = m_cache_blk_res1[t][b][i] - m_cache_blk_in[t][b][i];

               double gate_i = 0.5;
               if(MathAbs(attn_i - pre_i) > 1e-6)
                  gate_i = FXAI_Clamp((fused_i - pre_i) / (attn_i - pre_i), 0.01, 0.99);

               d_attn[i] = d_fused[i] * gate_i;
               d_pre[i] = d_fused[i] * (1.0 - gate_i);

               double dg = d_fused[i] * (attn_i - pre_i) * gate_i * (1.0 - gate_i);
               AdamWUpdate(m_b_gate[b][i], m_m_b_gate[b][i], m_v_b_gate[b][i], dg, lr, l2);
               for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               {
                  double g_w = dg * m_cache_blk_pre[t][b][j];
                  AdamWUpdate(m_w_gate[b][i][j], m_m_w_gate[b][i][j], m_v_w_gate[b][i][j], g_w, lr, l2);
                  d_pre[j] += dg * m_w_gate[b][i][j];
               }
            }

            // Attention projection updates.
            double d_head[FXAI_AF_HEADS][FXAI_AF_D_HEAD];
            for(int hd=0; hd<FXAI_AF_HEADS; hd++)
            {
               for(int d=0; d<FXAI_AF_D_HEAD; d++)
               {
                  double s = 0.0;
                  for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
                  {
                     double g_wo = d_attn[o] * m_cache_head_ctx[t][b][hd][d];
                     AdamWUpdate(m_wo[b][hd][o][d], m_m_wo[b][hd][o][d], m_v_wo[b][hd][o][d], g_wo, lr, l2);
                     s += d_attn[o] * m_wo[b][hd][o][d];
                  }
                  d_head[hd][d] = FXAI_ClipSym(s, 3.0);
               }
            }

            for(int hd=0; hd<FXAI_AF_HEADS; hd++)
            {
               for(int d=0; d<FXAI_AF_D_HEAD; d++)
               {
                  double dh = d_head[hd][d];
                  for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
                  {
                     // q from pre-state.
                     double gq = dh * m_cache_blk_pre[t][b][j] * 0.10;
                     AdamWUpdate(m_wq[b][hd][d][j], m_m_wq[b][hd][d][j], m_v_wq[b][hd][d][j], gq, lr, l2);

                     for(int k=0; k<FXAI_AF_TOPK_LAGS; k++)
                     {
                        double a = m_cache_lag_w[t][b][hd][k];
                        if(a <= 0.0) continue;
                        double src = m_cache_lag_src[t][b][hd][LagSrcFlat(k, j)];
                        double gkv = dh * a * src * 0.10;
                        AdamWUpdate(m_wk[b][hd][d][j], m_m_wk[b][hd][d][j], m_v_wk[b][hd][d][j], gkv, lr, l2);
                        AdamWUpdate(m_wv[b][hd][d][j], m_m_wv[b][hd][d][j], m_v_wv[b][hd][d][j], gkv, lr, l2);
                     }
                  }
               }
            }

            // Pre-norm params + propagate to previous block.
            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               double gp = d_pre[i] * m_cache_blk_in[t][b][i];
               AdamWUpdate(m_ln_pre_g[b][i], m_m_ln_pre_g[b][i], m_v_ln_pre_g[b][i], 0.02 * gp, lr, l2);
               AdamWUpdate(m_ln_pre_b[b][i], m_m_ln_pre_b[b][i], m_v_ln_pre_b[b][i], 0.02 * d_pre[i], lr, l2);

               AdamWUpdate(m_ln_post1_g[b][i], m_m_ln_post1_g[b][i], m_v_ln_post1_g[b][i], 0.02 * d_res1[i], lr, l2);
               AdamWUpdate(m_ln_post1_b[b][i], m_m_ln_post1_b[b][i], m_v_ln_post1_b[b][i], 0.02 * d_res1[i], lr, l2);
               AdamWUpdate(m_ln_post2_g[b][i], m_m_ln_post2_g[b][i], m_v_ln_post2_g[b][i], 0.02 * delta_cur[i], lr, l2);
               AdamWUpdate(m_ln_post2_b[b][i], m_m_ln_post2_b[b][i], m_v_ln_post2_b[b][i], 0.02 * delta_cur[i], lr, l2);
            }

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
               delta_cur[i] = FXAI_ClipSym(0.75 * d_pre[i] + 0.25 * d_res1[i], 6.0);
         }

         // Decomposition + embedding gradients.
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double d = delta_cur[o];
            AdamWUpdate(m_b_trend[o], m_m_b_trend[o], m_v_b_trend[o], 0.10 * d, lr, l2);
            AdamWUpdate(m_b_season[o], m_m_b_season[o], m_v_b_season[o], 0.10 * d, lr, l2);
            AdamWUpdate(m_b_in[o], m_m_b_in[o], m_v_b_in[o], 0.06 * d, lr, l2);
            AdamWUpdate(m_b_mix[o], m_m_b_mix[o], m_v_b_mix[o], 0.04 * d, lr, l2);

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               AdamWUpdate(m_w_trend[o][i], m_m_w_trend[o][i], m_v_w_trend[o][i], 0.10 * d * m_cache_trend_raw[t][i], lr, l2);
               AdamWUpdate(m_w_season[o][i], m_m_w_season[o][i], m_v_w_season[o][i], 0.10 * d * m_cache_season_raw[t][i], lr, l2);
            }

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
               AdamWUpdate(m_w_in[o][i], m_m_w_in[o][i], m_v_w_in[o][i], 0.06 * d * m_cache_xn[t][i], lr, l2);

            for(int k=0; k<FXAI_AF_MA_KERNELS; k++)
            {
               double gk = 0.04 * d * m_cache_ma[t][o][k];
               AdamWUpdate(m_w_mix[o][k], m_m_w_mix[o][k], m_v_w_mix[o][k], gk, lr, l2);
            }
         }

         // Plugin-native calibrator update.
         int sess = SessionBucket(ResolveContextTime());
         double p_raw_row[FXAI_AF_CLASS_COUNT];
         for(int c=0; c<FXAI_AF_CLASS_COUNT; c++) p_raw_row[c] = m_cache_probs_raw[t][c];
         UpdateCalibrator3(sess, p_raw_row, cls, w_cls, 0.02 * lr0);

         // Keep binary calibrator aligned.
         double den = m_cache_probs_raw[t][FXAI_AF_BUY] + m_cache_probs_raw[t][FXAI_AF_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir_raw = m_cache_probs_raw[t][FXAI_AF_BUY] / den;
         if(cls == FXAI_AF_BUY) UpdateCalibration(p_dir_raw, 1, w_cls);
         else if(cls == FXAI_AF_SELL) UpdateCalibration(p_dir_raw, 0, w_cls);
         else UpdateCalibration(p_dir_raw, (mv >= 0.0 ? 1 : 0), 0.25 * w_cls);

         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         double xrow2[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xrow2[i] = m_train_x[t][i];
         UpdateMoveHead(xrow2, mv, hp, sw);

         UpdateShadowHeads();

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++) next_delta[i] = delta_cur[i];
      }

      // Restore live sequence.
      m_seq_ptr = saved_ptr;
      m_seq_len = saved_len;
      for(int t=0; t<FXAI_AF_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_seq_state[t][h] = saved_state[t][h];
            m_seq_season[t][h] = saved_season[t][h];
            m_seq_trend[t][h] = saved_trend[t][h];
         }
      }
   }

   void PrintRecommendationOnce(void)
   {
      if(m_printed_reco) return;
      m_printed_reco = true;

      Print("FXAI Autoformer recommendation (98%+ parity target): d_model=32..64, heads=4, seq=128..256, TBPTT=32, topk_lags=8. ",
            "Current runtime uses d_model=", IntegerToString(FXAI_AI_MLP_HIDDEN),
            ", heads=", IntegerToString(FXAI_AF_HEADS),
            ", seq=", IntegerToString(FXAI_AF_SEQ),
            ", TBPTT=", IntegerToString(FXAI_AF_TBPTT),
            ", topk=", IntegerToString(FXAI_AF_TOPK_LAGS),
            ". Increase FXAI_AI_MLP_HIDDEN in shared.mqh for full-capacity preset.");
   }

public:
   CFXAIAIAutoformer(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_AUTOFORMER; }
   virtual string AIName(void) const { return "ai_autoformer"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 256);

   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, false, false, -1, fin, p_raw, mu, lv, q25, q75);

      int sess = SessionBucket(ResolveContextTime());
      Calibrate3(sess, p_raw, class_probs);

      double sigma = MathExp(0.5 * FXAI_Clamp(lv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double active = FXAI_Clamp(1.0 - class_probs[FXAI_AF_SKIP], 0.0, 1.0);

      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * iqr) * active;
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         expected_move_points = 0.65 * ev + 0.35 * m_move_ema_abs;
      else if(ev > 0.0)
         expected_move_points = ev;
      else
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, false, false, -1, fin, p_raw, mu, lv, q25, q75);

      int sess = SessionBucket(ResolveContextTime());
      Calibrate3(sess, p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);

      double sigma = FXAI_Clamp(MathExp(0.5 * FXAI_Clamp(lv, -4.0, 4.0)), 0.05, 30.0);
      double active = FXAI_Clamp(1.0 - out.class_probs[FXAI_AF_SKIP], 0.0, 1.0);
      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * MathAbs(q75 - q25)) * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;

      double ql = MathAbs(q25) * active;
      double qh = MathAbs(q75) * active;
      if(ql > qh) { double t = ql; ql = qh; qh = t; }

      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, ql);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, qh);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[FXAI_AF_BUY], out.class_probs[FXAI_AF_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_shadow_ready = false;
      m_printed_reco = false;
      m_step = 0;
      m_adam_t = 0;

      ResetNorm();
      ResetSequence();
      ResetTrainBuffer();
      ResetCalibrator3();
      ZeroMoments();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      PrintRecommendationOnce();
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      if((m_step % 1024) == 0 || MathAbs(x[1]) > 9.0 || MathAbs(x[2]) > 9.0)
         ResetSequence();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      UpdateNormStats(xa);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      int cls = ResolveClass(y, xa, move_points);
      double sw = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.25, 6.0);
      double cost = InputCostProxyPoints(xa);

      AppendTrainSample(cls, xa, move_points, cost, sw);
      TrainTBPTT(h);

      // Commit the newest sample to live state.
      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, true, false, -1, fin, p_raw, mu, lv, q25, q75);
      UpdateNativeQualityHeads(xa, sw, h.lr, h.l2);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_AF_CLASS_COUNT];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return 0.5;

      double den = probs[FXAI_AF_BUY] + probs[FXAI_AF_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_AF_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double p = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_AF_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_AF_CLASS_COUNT];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0)
         return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};

#endif // __FXAI_AI_AUTOFORMER_MQH__
