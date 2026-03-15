#ifndef __FXAI_AI_MLP_TINY_MQH__
#define __FXAI_AI_MLP_TINY_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_MLP_H1 16
#define FXAI_MLP_H2 16
#define FXAI_MLP_CTX 6
#define FXAI_MLP_CLASSES 3
#define FXAI_MLP_HIST 8
#define FXAI_MLP_REPLAY 384
#define FXAI_MLP_CAL_BINS 12
#define FXAI_MLP_ECE_BINS 12

class CFXAIAIMLPTiny : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_shadow_ready;
   int    m_step;
   int    m_adam_step;
   int    m_hist_len;
   int    m_hist_ptr;
   double m_symbol_hash;

   double m_shadow_decay;

   // Core network.
   double m_w1[FXAI_MLP_H1][FXAI_AI_WEIGHTS];
   double m_w1c[FXAI_MLP_H1][FXAI_MLP_CTX];
   double m_b1[FXAI_MLP_H1];

   double m_w2[FXAI_MLP_H2][FXAI_MLP_H1];
   double m_b2[FXAI_MLP_H2];

   // Residual gated block (3rd block).
   double m_w3[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_w3g[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_b3[FXAI_MLP_H2];
   double m_b3g[FXAI_MLP_H2];

   // 3-class head.
   double m_w_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2];
   double m_b_cls[FXAI_MLP_CLASSES];

   // Distributional move heads (absolute move points).
   double m_w_mu[FXAI_MLP_H2];
   double m_b_mu;
   double m_w_logv[FXAI_MLP_H2];
   double m_b_logv;
   double m_w_q25[FXAI_MLP_H2];
   double m_b_q25;
   double m_w_q75[FXAI_MLP_H2];
   double m_b_q75;
   CFXAINativeQualityHeads m_quality_heads;

   // AdamW moments.
   double m_m_w1[FXAI_MLP_H1][FXAI_AI_WEIGHTS], m_v_w1[FXAI_MLP_H1][FXAI_AI_WEIGHTS];
   double m_m_w1c[FXAI_MLP_H1][FXAI_MLP_CTX],   m_v_w1c[FXAI_MLP_H1][FXAI_MLP_CTX];
   double m_m_b1[FXAI_MLP_H1], m_v_b1[FXAI_MLP_H1];

   double m_m_w2[FXAI_MLP_H2][FXAI_MLP_H1], m_v_w2[FXAI_MLP_H2][FXAI_MLP_H1];
   double m_m_b2[FXAI_MLP_H2], m_v_b2[FXAI_MLP_H2];
   double m_m_w3[FXAI_MLP_H2][FXAI_MLP_H2], m_v_w3[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_m_w3g[FXAI_MLP_H2][FXAI_MLP_H2], m_v_w3g[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_m_b3[FXAI_MLP_H2], m_v_b3[FXAI_MLP_H2];
   double m_m_b3g[FXAI_MLP_H2], m_v_b3g[FXAI_MLP_H2];

   double m_m_w_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2], m_v_w_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2];
   double m_m_b_cls[FXAI_MLP_CLASSES], m_v_b_cls[FXAI_MLP_CLASSES];

   double m_m_w_mu[FXAI_MLP_H2], m_v_w_mu[FXAI_MLP_H2];
   double m_m_b_mu, m_v_b_mu;
   double m_m_w_logv[FXAI_MLP_H2], m_v_w_logv[FXAI_MLP_H2];
   double m_m_b_logv, m_v_b_logv;
   double m_m_w_q25[FXAI_MLP_H2], m_v_w_q25[FXAI_MLP_H2];
   double m_m_b_q25, m_v_b_q25;
   double m_m_w_q75[FXAI_MLP_H2], m_v_w_q75[FXAI_MLP_H2];
   double m_m_b_q75, m_v_b_q75;

   // EMA shadow params for stable inference.
   double m_sw1[FXAI_MLP_H1][FXAI_AI_WEIGHTS];
   double m_sw1c[FXAI_MLP_H1][FXAI_MLP_CTX];
   double m_sb1[FXAI_MLP_H1];

   double m_sw2[FXAI_MLP_H2][FXAI_MLP_H1];
   double m_sb2[FXAI_MLP_H2];
   double m_sw3[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_sw3g[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_sb3[FXAI_MLP_H2];
   double m_sb3g[FXAI_MLP_H2];

   double m_sw_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2];
   double m_sb_cls[FXAI_MLP_CLASSES];

   double m_sw_mu[FXAI_MLP_H2];
   double m_sb_mu;
   double m_sw_logv[FXAI_MLP_H2];
   double m_sb_logv;
   double m_sw_q25[FXAI_MLP_H2];
   double m_sb_q25;
   double m_sw_q75[FXAI_MLP_H2];
   double m_sb_q75;

   // Temporal context ring buffer.
   double m_hist_x[FXAI_MLP_HIST][FXAI_AI_WEIGHTS];

   // Adaptive feature normalization state (RevIN/DAIN-lite style).
   double m_norm_mean[FXAI_AI_WEIGHTS];
   double m_norm_var[FXAI_AI_WEIGHTS];
   double m_norm_median[FXAI_AI_WEIGHTS];
   double m_norm_iqr[FXAI_AI_WEIGHTS];
   double m_norm_gain[FXAI_AI_WEIGHTS];
   double m_norm_bias[FXAI_AI_WEIGHTS];
   int    m_norm_steps;

   // Context adapter projection.
   double m_ctx_w[FXAI_MLP_CTX][FXAI_MLP_CTX];
   double m_ctx_b[FXAI_MLP_CTX];

   // Replay buffer.
   double   m_mlp_replay_x[FXAI_MLP_REPLAY][FXAI_AI_WEIGHTS];
   int      m_replay_y[FXAI_MLP_REPLAY];
   double   m_mlp_replay_move[FXAI_MLP_REPLAY];
   double   m_mlp_replay_cost[FXAI_MLP_REPLAY];
   double   m_mlp_replay_min_move[FXAI_MLP_REPLAY];
   double   m_replay_w[FXAI_MLP_REPLAY];
   datetime m_mlp_replay_time[FXAI_MLP_REPLAY];
   int      m_mlp_replay_regime[FXAI_MLP_REPLAY];
   int      m_mlp_replay_horizon[FXAI_MLP_REPLAY];
   int      m_replay_session[FXAI_MLP_REPLAY];
   int      m_mlp_replay_feature_schema[FXAI_MLP_REPLAY];
   int      m_mlp_replay_norm_method[FXAI_MLP_REPLAY];
   int      m_mlp_replay_sequence_bars[FXAI_MLP_REPLAY];
   double   m_mlp_replay_point_value[FXAI_MLP_REPLAY];
   int      m_mlp_replay_window_size[FXAI_MLP_REPLAY];
   double   m_mlp_replay_window[FXAI_MLP_REPLAY][FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int      m_mlp_replay_head;
   int      m_mlp_replay_size;

   // SWA params (used with EMA in inference).
   bool   m_swa_ready;
   int    m_swa_count;
   double m_swa_w1[FXAI_MLP_H1][FXAI_AI_WEIGHTS];
   double m_swa_w1c[FXAI_MLP_H1][FXAI_MLP_CTX];
   double m_swa_b1[FXAI_MLP_H1];
   double m_swa_w2[FXAI_MLP_H2][FXAI_MLP_H1];
   double m_swa_b2[FXAI_MLP_H2];
   double m_swa_w3[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_swa_w3g[FXAI_MLP_H2][FXAI_MLP_H2];
   double m_swa_b3[FXAI_MLP_H2];
   double m_swa_b3g[FXAI_MLP_H2];
   double m_swa_w_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2];
   double m_swa_b_cls[FXAI_MLP_CLASSES];
   double m_swa_w_mu[FXAI_MLP_H2], m_swa_b_mu;
   double m_swa_w_logv[FXAI_MLP_H2], m_swa_b_logv;
   double m_swa_w_q25[FXAI_MLP_H2], m_swa_b_q25;
   double m_swa_w_q75[FXAI_MLP_H2], m_swa_b_q75;

   // Class-balance EMA.
   double m_class_ema[FXAI_MLP_CLASSES];

   // Native 3-class calibration.
   double m_cal_vs_w[FXAI_MLP_CLASSES][FXAI_MLP_CLASSES];
   double m_cal_vs_b[FXAI_MLP_CLASSES];
   double m_cal_iso_pos[FXAI_MLP_CLASSES][FXAI_MLP_CAL_BINS];
   double m_cal_iso_cnt[FXAI_MLP_CLASSES][FXAI_MLP_CAL_BINS];
   int    m_cal3_steps;

   // Validation / quality gate.
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_nll_fast, m_val_nll_slow;
   double m_val_brier_fast, m_val_brier_slow;
   double m_val_ece_fast, m_val_ece_slow;
   double m_val_ev_fast, m_val_ev_slow;
   double m_ece_mass[FXAI_MLP_ECE_BINS];
   double m_ece_acc[FXAI_MLP_ECE_BINS];
   double m_ece_conf[FXAI_MLP_ECE_BINS];
   bool   m_quality_degraded;

   int HistIndex(const int back) const
   {
      if(m_hist_len <= 0) return -1;
      int b = back;
      if(b < 0) b = 0;
      int max_back = m_hist_len - 1;
      if(max_back < 0) max_back = 0;
      if(b > max_back) b = max_back;

      int idx = m_hist_ptr - 1 - b;
      while(idx < 0) idx += FXAI_MLP_HIST;
      while(idx >= FXAI_MLP_HIST) idx -= FXAI_MLP_HIST;
      return idx;
   }

   double BlendShadowSWA(const double base,
                         const double shadow,
                         const double swa,
                         const bool use_shadow) const
   {
      if(!use_shadow) return base;
      if(m_swa_ready) return 0.60 * shadow + 0.40 * swa;
      return shadow;
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime md;
      TimeToStruct(t, md);
      int h = md.hour;
      if(h >= 6 && h <= 12) return 1;   // EU
      if(h >= 13 && h <= 20) return 2;  // US
      if(h >= 21 || h <= 2) return 0;   // Asia
      return 3;                         // transition
   }

   double SymbolHashNorm(void) const
   {
      string s = _Symbol;
      uint h = 2166136261U;
      int n = StringLen(s);
      for(int i=0; i<n; i++)
      {
         uint ch = (uint)StringGetCharacter(s, i);
         h ^= ch;
         h *= 16777619U;
      }
      return (double)(h % 100000U) / 100000.0;
   }

   void ResetHistory(void)
   {
      m_hist_len = 0;
      m_hist_ptr = 0;
      for(int t=0; t<FXAI_MLP_HIST; t++)
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_hist_x[t][i] = 0.0;
   }

   void PushHistory(const double &x[])
   {
      int p = m_hist_ptr;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_hist_x[p][i] = x[i];

      m_hist_ptr++;
      if(m_hist_ptr >= FXAI_MLP_HIST) m_hist_ptr = 0;
      if(m_hist_len < FXAI_MLP_HIST) m_hist_len++;
   }

   void NormalizeInput(const double &x[],
                       double &xn[]) const
   {
      double inst_mean = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) inst_mean += x[i];
      inst_mean /= (double)FXAI_AI_WEIGHTS;

      double inst_var = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - inst_mean;
         inst_var += d * d;
      }
      inst_var /= (double)FXAI_AI_WEIGHTS;
      double inst_std = MathSqrt(inst_var + 1e-9);

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double stdv = MathSqrt(MathMax(1e-8, m_norm_var[i]));
         double z = (x[i] - m_norm_mean[i]) / stdv;

         double iqr = MathMax(1e-5, m_norm_iqr[i]);
         double rz = (x[i] - m_norm_median[i]) / iqr;

         double v = 0.70 * z + 0.30 * rz;
         v = (v - inst_mean) / MathMax(inst_std, 1e-5); // RevIN style

         // DAIN-lite adaptive affine.
         v = (m_norm_gain[i] * v) + m_norm_bias[i];
         xn[i] = FXAI_ClipSym(v, 8.0);
      }
   }

   void UpdateNormStats(const double &x[])
   {
      m_norm_steps++;
      double a = (m_norm_steps < 256 ? 0.01 : 0.003);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = x[i];
         double dm = v - m_norm_mean[i];
         m_norm_mean[i] += a * dm;
         m_norm_var[i] = (1.0 - a) * m_norm_var[i] + a * dm * dm;
         if(m_norm_var[i] < 1e-8) m_norm_var[i] = 1e-8;

         m_norm_median[i] += a * FXAI_ClipSym(v - m_norm_median[i], 5.0);
         double ad = MathAbs(v - m_norm_median[i]);
         m_norm_iqr[i] += a * (ad - m_norm_iqr[i]);
         if(m_norm_iqr[i] < 1e-5) m_norm_iqr[i] = 1e-5;

         // Keep gains stable and adaptive.
         double stdv = MathSqrt(m_norm_var[i]);
         double target_gain = FXAI_Clamp(1.0 / MathMax(0.10, stdv), 0.20, 4.00);
         m_norm_gain[i] = 0.995 * m_norm_gain[i] + 0.005 * target_gain;
         m_norm_bias[i] = 0.995 * m_norm_bias[i] - 0.005 * m_norm_mean[i] * target_gain;
         m_norm_gain[i] = FXAI_Clamp(m_norm_gain[i], 0.20, 5.00);
         m_norm_bias[i] = FXAI_ClipSym(m_norm_bias[i], 5.0);
      }
   }

   void BuildTemporalContext(const double &x[],
                             double &ctx_raw[]) const
   {
      for(int i=0; i<FXAI_MLP_CTX; i++) ctx_raw[i] = 0.0;
      if(m_hist_len <= 0)
      {
         ctx_raw[4] = FXAI_Clamp(MathAbs(x[7]), 0.0, 8.0);
         return;
      }

      int i1 = HistIndex(0);
      int i2 = HistIndex(1);
      int i4 = HistIndex(3);
      if(i1 < 0) return;
      if(i2 < 0) i2 = i1;
      if(i4 < 0) i4 = i2;

      double x1_prev = m_hist_x[i1][1];
      double x2_prev = m_hist_x[i1][2];
      double x3_prev = m_hist_x[i1][3];

      ctx_raw[0] = x[1] - x1_prev;
      ctx_raw[1] = x[1] - m_hist_x[i2][1];
      ctx_raw[2] = x[1] - m_hist_x[i4][1];

      double vol = 0.0;
      int cnt = 0;
      for(int b=0; b<4; b++)
      {
         int ia = HistIndex(b);
         int ib = HistIndex(b + 1);
         if(ia < 0 || ib < 0) continue;
         vol += MathAbs(m_hist_x[ia][1] - m_hist_x[ib][1]);
         cnt++;
      }
      if(cnt > 0) vol /= (double)cnt;
      ctx_raw[3] = vol;

      ctx_raw[4] = MathAbs(x[7]); // spread/cost-aware feature channel.

      double regime = MathAbs(x[2] - x2_prev) + 0.5 * MathAbs(x[3] - x3_prev);
      regime += 0.35 * MathAbs(x[1] - x1_prev);
      ctx_raw[5] = regime;

      datetime t = ResolveContextTime();
      if(t <= 0) t = TimeCurrent();
      int sess = SessionBucket(t);
      double sess_shift = ((double)sess - 1.5) / 1.5;
      ctx_raw[4] = FXAI_ClipSym(ctx_raw[4] + 0.35 * sess_shift, 8.0);
      ctx_raw[5] = FXAI_ClipSym(ctx_raw[5] + 0.20 * (m_symbol_hash - 0.5), 8.0);

      for(int i=0; i<FXAI_MLP_CTX; i++)
         ctx_raw[i] = FXAI_ClipSym(ctx_raw[i], 8.0);
   }

    void BuildWindowAwareInput(const double &x[],
                               double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = FXAI_ClipSym(v, 8.0);
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
      xa[1] = FXAI_ClipSym(0.60 * xa[1] + 0.25 * mean1 + 0.15 * (first1 - last1), 8.0);
      xa[2] = FXAI_ClipSym(0.60 * xa[2] + 0.25 * mean2 + 0.15 * (first2 - last2), 8.0);
      xa[6] = FXAI_ClipSym(0.65 * xa[6] + 0.35 * vol1, 8.0);
      xa[7] = FXAI_ClipSym(0.65 * xa[7] + 0.35 * mean6, 8.0);
   }

   void AdaptContext(const double &ctx_raw[],
                     double &ctx[],
                     double &ctx_act[]) const
   {
      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         double z = m_ctx_b[k];
         for(int j=0; j<FXAI_MLP_CTX; j++)
            z += m_ctx_w[k][j] * ctx_raw[j];
         double a = FXAI_Tanh(FXAI_ClipSym(z, 8.0));
         ctx_act[k] = a;
         ctx[k] = FXAI_ClipSym(a + 0.25 * ctx_raw[k], 8.0);
      }
   }

   int ResolveClass(const int y,
                    const double &x[],
                    const double move_points) const
   {
      if(y == (int)FXAI_LABEL_SELL || y == (int)FXAI_LABEL_BUY || y == (int)FXAI_LABEL_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return (int)FXAI_LABEL_SKIP;

      if(y > 0) return (int)FXAI_LABEL_BUY;
      if(y == 0) return (int)FXAI_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -30.0, 30.0));
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

   void LayerNormVec(const double &v_in[],
                     double &v_norm[],
                     double &mean,
                     double &inv_std) const
   {
      mean = 0.0;
      for(int i=0; i<FXAI_MLP_H2; i++) mean += v_in[i];
      mean /= (double)FXAI_MLP_H2;

      double var = 0.0;
      for(int i=0; i<FXAI_MLP_H2; i++)
      {
         double d = v_in[i] - mean;
         var += d * d;
      }
      var /= (double)FXAI_MLP_H2;
      inv_std = 1.0 / MathSqrt(var + 1e-6);

      for(int i=0; i<FXAI_MLP_H2; i++)
         v_norm[i] = (v_in[i] - mean) * inv_std;
   }

   void LayerNormBackward(const double &dy[],
                          const double &x_norm[],
                          const double inv_std,
                          double &dx[]) const
   {
      double sum1 = 0.0, sum2 = 0.0;
      for(int i=0; i<FXAI_MLP_H2; i++)
      {
         sum1 += dy[i];
         sum2 += dy[i] * x_norm[i];
      }
      double n = (double)FXAI_MLP_H2;
      double k = inv_std / n;
      for(int i=0; i<FXAI_MLP_H2; i++)
      {
         double v = (n * dy[i]) - sum1 - x_norm[i] * sum2;
         dx[i] = k * v;
      }
   }

   double HuberGrad(const double err,
                    const double delta) const
   {
      double d = (delta > 0.0 ? delta : 1.0);
      if(err > d) return d;
      if(err < -d) return -d;
      return err;
   }

   double PinballGrad(const double y,
                      const double q,
                      const double tau) const
   {
      if(y >= q) return -tau;
      return (1.0 - tau);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost_points,
                      const double sample_w) const
   {
      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);

      double tot = m_class_ema[0] + m_class_ema[1] + m_class_ema[2];
      if(tot <= 0.0) tot = 3.0;
      double cnt = m_class_ema[cls];
      if(cnt < 1e-6) cnt = 1e-6;
      double balance = tot / (3.0 * cnt);
      balance = FXAI_Clamp(balance, 0.60, 2.60);

      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);
      double edge_scale = (edge > 0.0 ? 1.0 + 0.04 * MathMin(edge, 20.0) : 0.75);

      if(cls == (int)FXAI_LABEL_SKIP)
      {
         if(edge <= 0.0) edge_scale *= 1.20;
         else            edge_scale *= 0.80;
      }

      return FXAI_Clamp(sw * balance * edge_scale, 0.20, 6.00);
   }

   double MoveWeight(const double move_points,
                     const double cost_points,
                     const double sample_w) const
   {
      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);
      double edge_w = FXAI_MoveEdgeWeight(move_points, cost_points);
      double edge = MathMax(0.0, MathAbs(move_points) - MathMax(cost_points, 0.0));
      double scale = 1.0 + 0.05 * MathMin(edge, 20.0);
      return FXAI_Clamp(sw * edge_w * scale, 0.20, 8.00);
   }

   void BuildTargetDist(const int cls,
                        const double move_points,
                        const double cost_points,
                        double &target[]) const
   {
      for(int c=0; c<FXAI_MLP_CLASSES; c++) target[c] = 0.0;
      int y = cls;
      if(y < 0) y = 0;
      if(y >= FXAI_MLP_CLASSES) y = (int)FXAI_LABEL_SKIP;

      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);
      if(y == (int)FXAI_LABEL_SKIP)
      {
         target[(int)FXAI_LABEL_SKIP] = 1.0;
         return;
      }

      int dir = (y == (int)FXAI_LABEL_BUY ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      if(edge <= 0.0)
      {
         target[dir] = 0.35;
         target[(int)FXAI_LABEL_SKIP] = 0.65;
         return;
      }

      double pdir = FXAI_Clamp(0.74 + 0.10 * edge / MathMax(cost_points, 1.0), 0.74, 0.95);
      target[dir] = pdir;
      target[(int)FXAI_LABEL_SKIP] = 1.0 - pdir;
   }

   int ReplayPos(const int logical_idx) const
   {
      if(m_mlp_replay_size <= 0) return 0;
      int start = m_mlp_replay_head - m_mlp_replay_size;
      while(start < 0) start += FXAI_MLP_REPLAY;
      int p = start + logical_idx;
      while(p >= FXAI_MLP_REPLAY) p -= FXAI_MLP_REPLAY;
      return p;
   }

   void PushReplay(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double min_move_points,
                   const double sample_w,
                   const datetime t_sample,
                   const int sess,
                   const int regime_id,
                   const int horizon_minutes,
                   const int feature_schema_id,
                   const int norm_method_id,
                   const int sequence_bars,
                   const double point_value,
                   const int window_size,
                   const double &x_window[][FXAI_AI_WEIGHTS])
   {
      int p = m_mlp_replay_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_mlp_replay_x[p][i] = x[i];
      m_replay_y[p] = cls;
      m_mlp_replay_move[p] = move_points;
      m_mlp_replay_cost[p] = cost_points;
      m_mlp_replay_min_move[p] = min_move_points;
      m_replay_w[p] = sample_w;
      m_mlp_replay_time[p] = t_sample;
      m_mlp_replay_regime[p] = regime_id;
      m_mlp_replay_horizon[p] = horizon_minutes;
      m_replay_session[p] = sess;
      m_mlp_replay_feature_schema[p] = feature_schema_id;
      m_mlp_replay_norm_method[p] = norm_method_id;
      m_mlp_replay_sequence_bars[p] = MathMax(1, MathMin(sequence_bars, FXAI_MAX_SEQUENCE_BARS));
      m_mlp_replay_point_value[p] = (point_value > 0.0 ? point_value : (_Point > 0.0 ? _Point : 1.0));
      m_mlp_replay_window_size[p] = window_size;
      if(m_mlp_replay_window_size[p] < 0) m_mlp_replay_window_size[p] = 0;
      if(m_mlp_replay_window_size[p] > FXAI_MAX_SEQUENCE_BARS) m_mlp_replay_window_size[p] = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_mlp_replay_window[p][b][k] = (b < m_mlp_replay_window_size[p] ? x_window[b][k] : 0.0);

      m_mlp_replay_head++;
      if(m_mlp_replay_head >= FXAI_MLP_REPLAY) m_mlp_replay_head = 0;
      if(m_mlp_replay_size < FXAI_MLP_REPLAY) m_mlp_replay_size++;
   }

   double ReplayAgeWeight(const datetime t_sample,
                          const datetime t_now) const
   {
      if(t_sample <= 0 || t_now <= 0) return 1.0;
      double age_min = (double)MathMax(0, (int)(t_now - t_sample)) / 60.0;
      const double half_life = 24.0 * 60.0;
      return MathExp(-0.69314718056 * age_min / half_life);
   }

   void BuildCalLogits(const double &p_raw[],
                       double &logits[]) const
   {
      double lraw[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double z = m_cal_vs_b[c];
         for(int j=0; j<FXAI_MLP_CLASSES; j++) z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double logits[FXAI_MLP_CLASSES];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_MLP_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_MLP_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_MLP_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9) r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_MLP_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_MLP_CAL_BINS) bi = FXAI_MLP_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         p_cal[c] = FXAI_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

      double s = p_cal[0] + p_cal[1] + p_cal[2];
      if(s <= 0.0) s = 1.0;
      p_cal[0] /= s;
      p_cal[1] /= s;
      p_cal[2] /= s;
   }

   void UpdateCalibrator3(const double &p_raw[],
                          const int cls,
                          const double sample_w,
                          const double lr)
   {
      double logits[FXAI_MLP_CLASSES];
      BuildCalLogits(p_raw, logits);

      double p_cal[FXAI_MLP_CLASSES];
      Softmax3(logits, p_cal);

      double lraw[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.25 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<FXAI_MLP_CLASSES; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_cal_vs_w[c][j] - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(m_cal_vs_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_MLP_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_MLP_CAL_BINS) bi = FXAI_MLP_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }
      m_cal3_steps++;
   }

   void UpdateValidationMetrics(const int cls,
                                const double &p_cal[],
                                const double ev_after_cost)
   {
      int y = cls;
      if(y < 0) y = 0;
      if(y >= FXAI_MLP_CLASSES) y = (int)FXAI_LABEL_SKIP;

      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= (double)FXAI_MLP_CLASSES;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_MLP_CLASSES; c++) if(p_cal[c] > conf) { conf = p_cal[c]; pred = c; }
      double acc = (pred == y ? 1.0 : 0.0);

      int bi = (int)MathFloor(conf * (double)FXAI_MLP_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_MLP_ECE_BINS) bi = FXAI_MLP_ECE_BINS - 1;

      for(int b=0; b<FXAI_MLP_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      m_ece_mass[bi] += 1.0;
      m_ece_acc[bi] += acc;
      m_ece_conf[bi] += conf;

      double ece_num = 0.0, ece_den = 0.0;
      for(int b=0; b<FXAI_MLP_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);

      if(!m_val_ready)
      {
         m_val_nll_fast = m_val_nll_slow = ce;
         m_val_brier_fast = m_val_brier_slow = brier;
         m_val_ece_fast = m_val_ece_slow = ece;
         m_val_ev_fast = m_val_ev_slow = ev_after_cost;
         m_val_ready = true;
      }
      else
      {
         m_val_nll_fast = 0.92 * m_val_nll_fast + 0.08 * ce;
         m_val_nll_slow = 0.995 * m_val_nll_slow + 0.005 * ce;
         m_val_brier_fast = 0.92 * m_val_brier_fast + 0.08 * brier;
         m_val_brier_slow = 0.995 * m_val_brier_slow + 0.005 * brier;
         m_val_ece_fast = 0.92 * m_val_ece_fast + 0.08 * ece;
         m_val_ece_slow = 0.995 * m_val_ece_slow + 0.005 * ece;
         m_val_ev_fast = 0.92 * m_val_ev_fast + 0.08 * ev_after_cost;
         m_val_ev_slow = 0.995 * m_val_ev_slow + 0.005 * ev_after_cost;
      }

      m_val_steps++;
      m_quality_degraded = false;
      if(m_val_steps > 128)
      {
         if(m_val_nll_fast > 1.15 * MathMax(0.05, m_val_nll_slow)) m_quality_degraded = true;
         if(m_val_brier_fast > 1.20 * MathMax(0.03, m_val_brier_slow)) m_quality_degraded = true;
         if(m_val_ece_fast > 1.25 * MathMax(0.02, m_val_ece_slow)) m_quality_degraded = true;
         if(m_val_ev_fast < 0.85 * m_val_ev_slow) m_quality_degraded = true;
      }
   }

   double DropMask(const int idx,
                   const int salt,
                   const double drop_rate) const
   {
      if(drop_rate <= 1e-9) return 1.0;

      uint h = (uint)(m_step * 2654435761U);
      h ^= (uint)((idx + 3) * 2246822519U);
      h ^= (uint)((salt + 11) * 3266489917U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < drop_rate) return 0.0;
      return 1.0 / (1.0 - drop_rate);
   }

   double ScheduledLR(const FXAIAIHyperParams &hp,
                      const double sample_w) const
   {
      double base = FXAI_Clamp(hp.mlp_lr, 0.00005, 1.00000);
      double st = (double)MathMax(m_step, 1);

      double warm = FXAI_Clamp(st / 128.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0010 * MathMax(0.0, st - 128.0));

      double period = 2048.0;
      double phase = MathMod(st, period) / period;
      double cosine = 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
      double cosine_floor = 0.20 + 0.80 * cosine;

      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);
      double sw_scale = FXAI_Clamp(0.80 + 0.20 * MathSqrt(sw), 0.70, 1.60);

      double lr = base * warm * invsqrt * cosine_floor * sw_scale;
      return FXAI_Clamp(lr, 0.00001, 0.08000);
   }

   void AdamWStep(double &p,
                  double &m,
                  double &v,
                  const double g,
                  const double lr,
                  const double wd)
   {
      FXAI_OptAdamWStep(p, m, v, g, lr, 0.90, 0.999, wd, MathMax(m_adam_step, 1));
   }

   void InitMoments(void)
   {
      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         m_m_b1[h] = 0.0;
         m_v_b1[h] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_m_w1[h][i] = 0.0;
            m_v_w1[h][i] = 0.0;
         }
         for(int c=0; c<FXAI_MLP_CTX; c++)
         {
            m_m_w1c[h][c] = 0.0;
            m_v_w1c[h][c] = 0.0;
         }
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_m_b2[h] = 0.0;
         m_v_b2[h] = 0.0;
         m_m_b3[h] = 0.0;
         m_v_b3[h] = 0.0;
         m_m_b3g[h] = 0.0;
         m_v_b3g[h] = 0.0;

         m_m_w_mu[h] = 0.0;
         m_v_w_mu[h] = 0.0;
         m_m_w_logv[h] = 0.0;
         m_v_w_logv[h] = 0.0;
         m_m_w_q25[h] = 0.0;
         m_v_w_q25[h] = 0.0;
         m_m_w_q75[h] = 0.0;
         m_v_w_q75[h] = 0.0;

         for(int j=0; j<FXAI_MLP_H1; j++)
         {
            m_m_w2[h][j] = 0.0;
            m_v_w2[h][j] = 0.0;
         }
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_m_w3[h][j] = 0.0;
            m_v_w3[h][j] = 0.0;
            m_m_w3g[h][j] = 0.0;
            m_v_w3g[h][j] = 0.0;
         }
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_m_b_cls[c] = 0.0;
         m_v_b_cls[c] = 0.0;
         for(int h=0; h<FXAI_MLP_H2; h++)
         {
            m_m_w_cls[c][h] = 0.0;
            m_v_w_cls[c][h] = 0.0;
         }
      }

      m_m_b_mu = 0.0; m_v_b_mu = 0.0;
      m_m_b_logv = 0.0; m_v_b_logv = 0.0;
      m_m_b_q25 = 0.0; m_v_b_q25 = 0.0;
      m_m_b_q75 = 0.0; m_v_b_q75 = 0.0;
   }

   void InitWeights(const double init_scale)
   {
      double s = FXAI_Clamp(init_scale, 0.01, 0.60);

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double a = (double)((h + 1) * (i + 3));
            m_w1[h][i] = 0.04 * s * MathSin(0.87 * a);
         }
         for(int c=0; c<FXAI_MLP_CTX; c++)
         {
            double b = (double)((h + 2) * (c + 5));
            m_w1c[h][c] = 0.05 * s * MathCos(0.79 * b);
         }
         m_b1[h] = 0.0;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         for(int j=0; j<FXAI_MLP_H1; j++)
         {
            double a = (double)((h + 3) * (j + 2));
            m_w2[h][j] = 0.035 * s * MathCos(0.93 * a);
         }
         m_b2[h] = 0.0;

         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            double a3 = (double)((h + 5) * (j + 1));
            m_w3[h][j] = 0.030 * s * MathSin(0.71 * a3);
            m_w3g[h][j] = 0.020 * s * MathCos(0.67 * a3);
         }
         m_b3[h] = 0.0;
         m_b3g[h] = 0.1;
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_b_cls[c] = 0.0;
         for(int h=0; h<FXAI_MLP_H2; h++)
         {
            double a = (double)((c + 2) * (h + 1));
            m_w_cls[c][h] = 0.04 * s * MathSin(1.03 * a);
         }
      }

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.5;
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_w_mu[h] = 0.03 * s * MathCos((double)(h + 1) * 1.11);
         m_w_logv[h] = 0.02 * s * MathSin((double)(h + 1) * 1.07);
         m_w_q25[h] = 0.02 * s * MathSin((double)(h + 1) * 1.17);
         m_w_q75[h] = 0.02 * s * MathCos((double)(h + 1) * 1.19);
      }

      InitMoments();

      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         m_ctx_b[k] = 0.0;
         for(int j=0; j<FXAI_MLP_CTX; j++)
         {
            if(k == j) m_ctx_w[k][j] = 1.0;
            else m_ctx_w[k][j] = 0.02 * MathSin((double)((k + 1) * (j + 2)));
         }
      }

      m_symbol_hash = SymbolHashNorm();

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_norm_mean[i] = 0.0;
         m_norm_var[i] = 1.0;
         m_norm_median[i] = 0.0;
         m_norm_iqr[i] = 1.0;
         m_norm_gain[i] = 1.0;
         m_norm_bias[i] = 0.0;
      }
      m_norm_steps = 0;

      m_mlp_replay_head = 0;
      m_mlp_replay_size = 0;
      for(int i=0; i<FXAI_MLP_REPLAY; i++)
      {
         m_replay_y[i] = (int)FXAI_LABEL_SKIP;
         m_mlp_replay_move[i] = 0.0;
         m_mlp_replay_cost[i] = 0.0;
         m_mlp_replay_min_move[i] = 0.0;
         m_replay_w[i] = 1.0;
         m_mlp_replay_time[i] = 0;
         m_mlp_replay_regime[i] = 0;
         m_mlp_replay_horizon[i] = 1;
         m_replay_session[i] = -1;
         m_mlp_replay_feature_schema[i] = FXAI_SCHEMA_FULL;
         m_mlp_replay_norm_method[i] = FXAI_NORM_EXISTING;
         m_mlp_replay_sequence_bars[i] = 1;
         m_mlp_replay_point_value[i] = (_Point > 0.0 ? _Point : 1.0);
         m_mlp_replay_window_size[i] = 0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_mlp_replay_x[i][k] = 0.0;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_mlp_replay_window[i][b][k] = 0.0;
      }

      m_cal3_steps = 0;
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_MLP_CLASSES; j++) m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_MLP_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }

      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      for(int b=0; b<FXAI_MLP_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_quality_degraded = false;

      m_shadow_ready = false;
      m_shadow_decay = 0.995;
      SyncShadow(true);
      m_swa_ready = false;
      m_swa_count = 0;

      m_initialized = true;
   }

   void SyncShadow(const bool hard_copy)
   {
      double a = (hard_copy ? 0.0 : FXAI_Clamp(m_shadow_decay, 0.90, 0.9999));
      double b = (hard_copy ? 1.0 : (1.0 - a));

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         m_sb1[h] = (hard_copy ? m_b1[h] : a * m_sb1[h] + b * m_b1[h]);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_sw1[h][i] = (hard_copy ? m_w1[h][i] : a * m_sw1[h][i] + b * m_w1[h][i]);
         for(int c=0; c<FXAI_MLP_CTX; c++)
            m_sw1c[h][c] = (hard_copy ? m_w1c[h][c] : a * m_sw1c[h][c] + b * m_w1c[h][c]);
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_sb2[h] = (hard_copy ? m_b2[h] : a * m_sb2[h] + b * m_b2[h]);
         m_sb3[h] = (hard_copy ? m_b3[h] : a * m_sb3[h] + b * m_b3[h]);
         m_sb3g[h] = (hard_copy ? m_b3g[h] : a * m_sb3g[h] + b * m_b3g[h]);
         m_sw_mu[h] = (hard_copy ? m_w_mu[h] : a * m_sw_mu[h] + b * m_w_mu[h]);
         m_sw_logv[h] = (hard_copy ? m_w_logv[h] : a * m_sw_logv[h] + b * m_w_logv[h]);
         m_sw_q25[h] = (hard_copy ? m_w_q25[h] : a * m_sw_q25[h] + b * m_w_q25[h]);
         m_sw_q75[h] = (hard_copy ? m_w_q75[h] : a * m_sw_q75[h] + b * m_w_q75[h]);

         for(int j=0; j<FXAI_MLP_H1; j++)
            m_sw2[h][j] = (hard_copy ? m_w2[h][j] : a * m_sw2[h][j] + b * m_w2[h][j]);
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_sw3[h][j] = (hard_copy ? m_w3[h][j] : a * m_sw3[h][j] + b * m_w3[h][j]);
            m_sw3g[h][j] = (hard_copy ? m_w3g[h][j] : a * m_sw3g[h][j] + b * m_w3g[h][j]);
         }
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_sb_cls[c] = (hard_copy ? m_b_cls[c] : a * m_sb_cls[c] + b * m_b_cls[c]);
         for(int h=0; h<FXAI_MLP_H2; h++)
            m_sw_cls[c][h] = (hard_copy ? m_w_cls[c][h] : a * m_sw_cls[c][h] + b * m_w_cls[c][h]);
      }

      m_sb_mu = (hard_copy ? m_b_mu : a * m_sb_mu + b * m_b_mu);
      m_sb_logv = (hard_copy ? m_b_logv : a * m_sb_logv + b * m_b_logv);
      m_sb_q25 = (hard_copy ? m_b_q25 : a * m_sb_q25 + b * m_b_q25);
      m_sb_q75 = (hard_copy ? m_b_q75 : a * m_sb_q75 + b * m_b_q75);
   }

   void UpdateSWA(void)
   {
      if((m_step % 64) != 0) return;

      m_swa_count++;
      double rho = 1.0 / (double)MathMax(m_swa_count, 1);
      double k = 1.0 - rho;

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         m_swa_b1[h] = k * m_swa_b1[h] + rho * m_b1[h];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_swa_w1[h][i] = k * m_swa_w1[h][i] + rho * m_w1[h][i];
         for(int c=0; c<FXAI_MLP_CTX; c++)
            m_swa_w1c[h][c] = k * m_swa_w1c[h][c] + rho * m_w1c[h][c];
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_swa_b2[h] = k * m_swa_b2[h] + rho * m_b2[h];
         m_swa_b3[h] = k * m_swa_b3[h] + rho * m_b3[h];
         m_swa_b3g[h] = k * m_swa_b3g[h] + rho * m_b3g[h];
         m_swa_w_mu[h] = k * m_swa_w_mu[h] + rho * m_w_mu[h];
         m_swa_w_logv[h] = k * m_swa_w_logv[h] + rho * m_w_logv[h];
         m_swa_w_q25[h] = k * m_swa_w_q25[h] + rho * m_w_q25[h];
         m_swa_w_q75[h] = k * m_swa_w_q75[h] + rho * m_w_q75[h];
         for(int j=0; j<FXAI_MLP_H1; j++)
            m_swa_w2[h][j] = k * m_swa_w2[h][j] + rho * m_w2[h][j];
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_swa_w3[h][j] = k * m_swa_w3[h][j] + rho * m_w3[h][j];
            m_swa_w3g[h][j] = k * m_swa_w3g[h][j] + rho * m_w3g[h][j];
         }
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_swa_b_cls[c] = k * m_swa_b_cls[c] + rho * m_b_cls[c];
         for(int h=0; h<FXAI_MLP_H2; h++)
            m_swa_w_cls[c][h] = k * m_swa_w_cls[c][h] + rho * m_w_cls[c][h];
      }

      m_swa_b_mu = k * m_swa_b_mu + rho * m_b_mu;
      m_swa_b_logv = k * m_swa_b_logv + rho * m_b_logv;
      m_swa_b_q25 = k * m_swa_b_q25 + rho * m_b_q25;
      m_swa_b_q75 = k * m_swa_b_q75 + rho * m_b_q75;
      if(m_swa_count >= 4) m_swa_ready = true;
   }

   void Forward(const double &x[],
                const double &ctx[],
                const bool use_shadow,
                const bool training,
                double &h1_raw[],
                double &h1_out[],
                double &h2_out[],
                double &h2_norm[],
                double &h3_tanh[],
                double &h3_gate[],
                double &h3_out[],
                double &ln_inv_std,
                double &drop_mask[],
                double &logits[],
                double &probs[],
                double &mu,
                double &logv,
                double &q25,
      double &q75) const
   {
      double drop_rate = (training ? 0.10 : 0.0);

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         double s = BlendShadowSWA(m_b1[h], m_sb1[h], m_swa_b1[h], use_shadow);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            s += BlendShadowSWA(m_w1[h][i], m_sw1[h][i], m_swa_w1[h][i], use_shadow) * x[i];
         for(int c=0; c<FXAI_MLP_CTX; c++)
            s += BlendShadowSWA(m_w1c[h][c], m_sw1c[h][c], m_swa_w1c[h][c], use_shadow) * ctx[c];

         double a = FXAI_Tanh(FXAI_ClipSym(s, 8.0));
         double m = (training ? DropMask(h, 17, drop_rate) : 1.0);

         h1_raw[h] = a;
         drop_mask[h] = m;
         h1_out[h] = a * m;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         double s = BlendShadowSWA(m_b2[h], m_sb2[h], m_swa_b2[h], use_shadow);
         for(int j=0; j<FXAI_MLP_H1; j++)
            s += BlendShadowSWA(m_w2[h][j], m_sw2[h][j], m_swa_w2[h][j], use_shadow) * h1_out[j];
         h2_out[h] = FXAI_Tanh(FXAI_ClipSym(s, 8.0));
      }

      double ln_mean = 0.0;
      LayerNormVec(h2_out, h2_norm, ln_mean, ln_inv_std);

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         double z = BlendShadowSWA(m_b3[h], m_sb3[h], m_swa_b3[h], use_shadow);
         double zg = BlendShadowSWA(m_b3g[h], m_sb3g[h], m_swa_b3g[h], use_shadow);
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            z += BlendShadowSWA(m_w3[h][j], m_sw3[h][j], m_swa_w3[h][j], use_shadow) * h2_norm[j];
            zg += BlendShadowSWA(m_w3g[h][j], m_sw3g[h][j], m_swa_w3g[h][j], use_shadow) * h2_norm[j];
         }
         double t3 = FXAI_Tanh(FXAI_ClipSym(z, 8.0));
         double g3 = FXAI_Sigmoid(FXAI_ClipSym(zg, 8.0));
         h3_tanh[h] = t3;
         h3_gate[h] = g3;
         h3_out[h] = FXAI_ClipSym(t3 * g3 + 0.50 * h2_out[h], 8.0);
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double z = BlendShadowSWA(m_b_cls[c], m_sb_cls[c], m_swa_b_cls[c], use_shadow);
         for(int h=0; h<FXAI_MLP_H2; h++)
            z += BlendShadowSWA(m_w_cls[c][h], m_sw_cls[c][h], m_swa_w_cls[c][h], use_shadow) * h3_out[h];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = BlendShadowSWA(m_b_mu, m_sb_mu, m_swa_b_mu, use_shadow);
      logv = BlendShadowSWA(m_b_logv, m_sb_logv, m_swa_b_logv, use_shadow);
      q25 = BlendShadowSWA(m_b_q25, m_sb_q25, m_swa_b_q25, use_shadow);
      q75 = BlendShadowSWA(m_b_q75, m_sb_q75, m_swa_b_q75, use_shadow);
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         mu += BlendShadowSWA(m_w_mu[h], m_sw_mu[h], m_swa_w_mu[h], use_shadow) * h3_out[h];
         logv += BlendShadowSWA(m_w_logv[h], m_sw_logv[h], m_swa_w_logv[h], use_shadow) * h3_out[h];
         q25 += BlendShadowSWA(m_w_q25[h], m_sw_q25[h], m_swa_w_q25[h], use_shadow) * h3_out[h];
         q75 += BlendShadowSWA(m_w_q75[h], m_sw_q75[h], m_swa_w_q75[h], use_shadow) * h3_out[h];
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double tmp = q25;
         q25 = q75;
         q75 = tmp;
      }
   }

   void UpdateWeighted(const int y,
                       const double &x[],
                       const FXAIAIHyperParams &hp,
                       const double sample_w,
                       const double move_points,
                       const bool from_replay)
   {
      EnsureInitialized(hp);
      m_step++;
      m_adam_step++;

      int cls = ResolveClass(y, x, move_points);
      if(cls < 0) cls = 0;
      if(cls >= FXAI_MLP_CLASSES) cls = FXAI_MLP_CLASSES - 1;

      for(int k=0; k<FXAI_MLP_CLASSES; k++)
         m_class_ema[k] = 0.995 * m_class_ema[k] + (k == cls ? 0.005 : 0.0);

      double cost = InputCostProxyPoints(x);
      double sw = FXAI_Clamp(sample_w, 0.25, 4.00);
      if(from_replay) sw *= 0.85;
      double lr = ScheduledLR(hp, sw);
      if(m_quality_degraded) lr *= 0.70;
      double wd = FXAI_Clamp(hp.mlp_l2, 0.0, 0.0500);

      double x_norm[FXAI_AI_WEIGHTS];
      NormalizeInput(x, x_norm);

      double ctx_raw[FXAI_MLP_CTX], ctx_act[FXAI_MLP_CTX], ctx[FXAI_MLP_CTX];
      BuildTemporalContext(x, ctx_raw);
      AdaptContext(ctx_raw, ctx, ctx_act);

      double h1_raw[FXAI_MLP_H1], h1[FXAI_MLP_H1], h2[FXAI_MLP_H2], h2_norm[FXAI_MLP_H2];
      double h3_tanh[FXAI_MLP_H2], h3_gate[FXAI_MLP_H2], h3[FXAI_MLP_H2];
      double ln_inv_std = 1.0;
      double drop1[FXAI_MLP_H1];
      double logits[FXAI_MLP_CLASSES], probs[FXAI_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x_norm, ctx, false, true, h1_raw, h1, h2, h2_norm, h3_tanh, h3_gate, h3, ln_inv_std, drop1, logits, probs, mu, logv, q25, q75);

      double cls_w = ClassWeight(cls, move_points, cost, sw);
      double mv_w = MoveWeight(move_points, cost, sw);
      if(from_replay) mv_w *= 0.90;

      double pt = FXAI_Clamp(probs[cls], 0.001, 0.999);
      double focal = MathPow(FXAI_Clamp(1.0 - pt, 0.02, 1.0), 1.50);

      double target_dist[FXAI_MLP_CLASSES];
      BuildTargetDist(cls, move_points, cost, target_dist);

      double dlogits[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double yk = target_dist[c];
         dlogits[c] = (probs[c] - yk) * cls_w * focal;
      }

      double target = MathAbs(move_points);
      double err = mu - target;
      double var = MathExp(logv);
      var = FXAI_Clamp(var, 0.05, 100.0);

      double dmu = FXAI_ClipSym((HuberGrad(err, 6.0) / MathMax(var, 0.25)) * mv_w, 5.0);
      double dlogv = FXAI_ClipSym(0.5 * mv_w * (1.0 - (err * err) / MathMax(var, 0.25)), 4.0);
      double dq25 = FXAI_ClipSym(PinballGrad(target, q25, 0.25) * 0.25 * mv_w, 3.0);
      double dq75 = FXAI_ClipSym(PinballGrad(target, q75, 0.75) * 0.25 * mv_w, 3.0);
      if(q25 > q75)
      {
         double pen = 0.25 * FXAI_ClipSym(q25 - q75, 4.0);
         dq25 += pen;
         dq75 -= pen;
      }

      // Gradient accumulators.
      double g_w1[FXAI_MLP_H1][FXAI_AI_WEIGHTS];
      double g_w1c[FXAI_MLP_H1][FXAI_MLP_CTX];
      double g_b1[FXAI_MLP_H1];

      double g_w2[FXAI_MLP_H2][FXAI_MLP_H1];
      double g_b2[FXAI_MLP_H2];
      double g_w3[FXAI_MLP_H2][FXAI_MLP_H2];
      double g_w3g[FXAI_MLP_H2][FXAI_MLP_H2];
      double g_b3[FXAI_MLP_H2];
      double g_b3g[FXAI_MLP_H2];

      double g_w_cls[FXAI_MLP_CLASSES][FXAI_MLP_H2];
      double g_b_cls[FXAI_MLP_CLASSES];

      double g_w_mu[FXAI_MLP_H2], g_w_logv[FXAI_MLP_H2], g_w_q25[FXAI_MLP_H2], g_w_q75[FXAI_MLP_H2];
      double g_b_mu = dmu, g_b_logv = dlogv, g_b_q25 = dq25, g_b_q75 = dq75;
      double g_ctx_w[FXAI_MLP_CTX][FXAI_MLP_CTX];
      double g_ctx_b[FXAI_MLP_CTX];

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         g_b1[h] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_w1[h][i] = 0.0;
         for(int c=0; c<FXAI_MLP_CTX; c++) g_w1c[h][c] = 0.0;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         g_b2[h] = 0.0;
         g_b3[h] = 0.0;
         g_b3g[h] = 0.0;
         g_w_mu[h] = dmu * h3[h];
         g_w_logv[h] = dlogv * h3[h];
         g_w_q25[h] = dq25 * h3[h];
         g_w_q75[h] = dq75 * h3[h];

         for(int j=0; j<FXAI_MLP_H1; j++) g_w2[h][j] = 0.0;
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            g_w3[h][j] = 0.0;
            g_w3g[h][j] = 0.0;
         }
      }
      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         g_ctx_b[k] = 0.0;
         for(int j=0; j<FXAI_MLP_CTX; j++)
            g_ctx_w[k][j] = 0.0;
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         g_b_cls[c] = dlogits[c];
         for(int h=0; h<FXAI_MLP_H2; h++)
            g_w_cls[c][h] = dlogits[c] * h3[h];
      }

      double dh3[FXAI_MLP_H2];
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         double d = dmu * m_w_mu[h] + dlogv * m_w_logv[h] + dq25 * m_w_q25[h] + dq75 * m_w_q75[h];
         for(int c=0; c<FXAI_MLP_CLASSES; c++)
            d += dlogits[c] * m_w_cls[c][h];
         dh3[h] = d;
      }

      double dh2[FXAI_MLP_H2];
      double dh2_norm[FXAI_MLP_H2];
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         dh2[h] = 0.5 * dh3[h];
         dh2_norm[h] = 0.0;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         double dtanh = dh3[h] * h3_gate[h];
         double dgate = dh3[h] * h3_tanh[h];
         double dz3 = dtanh * (1.0 - h3_tanh[h] * h3_tanh[h]);
         double dz3g = dgate * h3_gate[h] * (1.0 - h3_gate[h]);

         g_b3[h] += dz3;
         g_b3g[h] += dz3g;
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            g_w3[h][j] += dz3 * h2_norm[j];
            g_w3g[h][j] += dz3g * h2_norm[j];
            dh2_norm[j] += dz3 * m_w3[h][j] + dz3g * m_w3g[h][j];
         }
      }

      double dln_in[FXAI_MLP_H2];
      LayerNormBackward(dh2_norm, h2_norm, ln_inv_std, dln_in);
      for(int h=0; h<FXAI_MLP_H2; h++)
         dh2[h] += dln_in[h];

      double dz2[FXAI_MLP_H2];
      double dh1[FXAI_MLP_H1];
      double dctx[FXAI_MLP_CTX];
      for(int h=0; h<FXAI_MLP_H1; h++) dh1[h] = 0.0;
      for(int c=0; c<FXAI_MLP_CTX; c++) dctx[c] = 0.0;

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         dz2[h] = dh2[h] * (1.0 - h2[h] * h2[h]);
         g_b2[h] += dz2[h];

         for(int j=0; j<FXAI_MLP_H1; j++)
         {
            g_w2[h][j] += dz2[h] * h1[j];
            dh1[j] += dz2[h] * m_w2[h][j];
         }
      }

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         dh1[h] *= drop1[h];
         double dz1 = dh1[h] * (1.0 - h1_raw[h] * h1_raw[h]);
         g_b1[h] += dz1;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            g_w1[h][i] += dz1 * x_norm[i];
         for(int c=0; c<FXAI_MLP_CTX; c++)
         {
            g_w1c[h][c] += dz1 * ctx[c];
            dctx[c] += dz1 * m_w1c[h][c];
         }
      }

      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         double dz = dctx[k] * (1.0 - ctx_act[k] * ctx_act[k]);
         g_ctx_b[k] += dz;
         for(int j=0; j<FXAI_MLP_CTX; j++)
            g_ctx_w[k][j] += dz * ctx_raw[j];
      }

      // Gradient centralization for matrix parameters.
      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         double m1 = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m1 += g_w1[h][i];
         m1 /= (double)FXAI_AI_WEIGHTS;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_w1[h][i] -= m1;

         double m1c = 0.0;
         for(int c=0; c<FXAI_MLP_CTX; c++) m1c += g_w1c[h][c];
         m1c /= (double)FXAI_MLP_CTX;
         for(int c=0; c<FXAI_MLP_CTX; c++) g_w1c[h][c] -= m1c;
      }
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         double m2 = 0.0;
         for(int j=0; j<FXAI_MLP_H1; j++) m2 += g_w2[h][j];
         m2 /= (double)FXAI_MLP_H1;
         for(int j=0; j<FXAI_MLP_H1; j++) g_w2[h][j] -= m2;

         double m3 = 0.0, m3g = 0.0;
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m3 += g_w3[h][j];
            m3g += g_w3g[h][j];
         }
         m3 /= (double)FXAI_MLP_H2;
         m3g /= (double)FXAI_MLP_H2;
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            g_w3[h][j] -= m3;
            g_w3g[h][j] -= m3g;
         }
      }
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         double mc = 0.0;
         for(int h=0; h<FXAI_MLP_H2; h++) mc += g_w_cls[c][h];
         mc /= (double)FXAI_MLP_H2;
         for(int h=0; h<FXAI_MLP_H2; h++) g_w_cls[c][h] -= mc;
      }
      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         double mctx = 0.0;
         for(int j=0; j<FXAI_MLP_CTX; j++) mctx += g_ctx_w[k][j];
         mctx /= (double)FXAI_MLP_CTX;
         for(int j=0; j<FXAI_MLP_CTX; j++) g_ctx_w[k][j] -= mctx;
      }

      // Per-layer clipping.
      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         g_b1[h] = FXAI_ClipSym(g_b1[h], 3.0);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_w1[h][i] = FXAI_ClipSym(g_w1[h][i], 3.0);
         for(int c=0; c<FXAI_MLP_CTX; c++) g_w1c[h][c] = FXAI_ClipSym(g_w1c[h][c], 3.0);
      }
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         g_b2[h] = FXAI_ClipSym(g_b2[h], 3.5);
         g_b3[h] = FXAI_ClipSym(g_b3[h], 3.5);
         g_b3g[h] = FXAI_ClipSym(g_b3g[h], 3.5);
         g_w_mu[h] = FXAI_ClipSym(g_w_mu[h], 4.0);
         g_w_logv[h] = FXAI_ClipSym(g_w_logv[h], 4.0);
         g_w_q25[h] = FXAI_ClipSym(g_w_q25[h], 4.0);
         g_w_q75[h] = FXAI_ClipSym(g_w_q75[h], 4.0);
         for(int j=0; j<FXAI_MLP_H1; j++) g_w2[h][j] = FXAI_ClipSym(g_w2[h][j], 3.5);
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            g_w3[h][j] = FXAI_ClipSym(g_w3[h][j], 3.5);
            g_w3g[h][j] = FXAI_ClipSym(g_w3g[h][j], 3.5);
         }
      }
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         g_b_cls[c] = FXAI_ClipSym(g_b_cls[c], 4.0);
         for(int h=0; h<FXAI_MLP_H2; h++) g_w_cls[c][h] = FXAI_ClipSym(g_w_cls[c][h], 4.0);
      }
      for(int c=0; c<FXAI_MLP_CTX; c++)
      {
         g_ctx_b[c] = FXAI_ClipSym(g_ctx_b[c], 2.5);
         for(int j=0; j<FXAI_MLP_CTX; j++) g_ctx_w[c][j] = FXAI_ClipSym(g_ctx_w[c][j], 2.5);
      }
      g_b_mu = FXAI_ClipSym(g_b_mu, 4.0);
      g_b_logv = FXAI_ClipSym(g_b_logv, 4.0);
      g_b_q25 = FXAI_ClipSym(g_b_q25, 4.0);
      g_b_q75 = FXAI_ClipSym(g_b_q75, 4.0);

      // Global norm clip.
      double g2 = g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         g2 += g_b1[h] * g_b1[h];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) g2 += g_w1[h][i] * g_w1[h][i];
         for(int c=0; c<FXAI_MLP_CTX; c++) g2 += g_w1c[h][c] * g_w1c[h][c];
      }
      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         g2 += g_b2[h] * g_b2[h] + g_b3[h] * g_b3[h] + g_b3g[h] * g_b3g[h];
         g2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];
         for(int j=0; j<FXAI_MLP_H1; j++) g2 += g_w2[h][j] * g_w2[h][j];
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            g2 += g_w3[h][j] * g_w3[h][j];
            g2 += g_w3g[h][j] * g_w3g[h][j];
         }
      }
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         g2 += g_b_cls[c] * g_b_cls[c];
         for(int h=0; h<FXAI_MLP_H2; h++) g2 += g_w_cls[c][h] * g_w_cls[c][h];
      }
      for(int c=0; c<FXAI_MLP_CTX; c++)
      {
         g2 += g_ctx_b[c] * g_ctx_b[c];
         for(int j=0; j<FXAI_MLP_CTX; j++) g2 += g_ctx_w[c][j] * g_ctx_w[c][j];
      }

      double gnorm = MathSqrt(g2 + 1e-12);
      double gscale = (gnorm > 3.20 ? (3.20 / gnorm) : 1.0);

      // AdamW updates.
      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         AdamWStep(m_b1[h], m_m_b1[h], m_v_b1[h], gscale * g_b1[h], lr, 0.0);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            AdamWStep(m_w1[h][i], m_m_w1[h][i], m_v_w1[h][i], gscale * g_w1[h][i], lr, wd);
         for(int c=0; c<FXAI_MLP_CTX; c++)
            AdamWStep(m_w1c[h][c], m_m_w1c[h][c], m_v_w1c[h][c], gscale * g_w1c[h][c], lr, wd);
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         AdamWStep(m_b2[h], m_m_b2[h], m_v_b2[h], gscale * g_b2[h], lr, 0.0);
         AdamWStep(m_b3[h], m_m_b3[h], m_v_b3[h], gscale * g_b3[h], lr, 0.0);
         AdamWStep(m_b3g[h], m_m_b3g[h], m_v_b3g[h], gscale * g_b3g[h], lr, 0.0);
         for(int j=0; j<FXAI_MLP_H1; j++)
            AdamWStep(m_w2[h][j], m_m_w2[h][j], m_v_w2[h][j], gscale * g_w2[h][j], lr, wd);
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            AdamWStep(m_w3[h][j], m_m_w3[h][j], m_v_w3[h][j], gscale * g_w3[h][j], lr, wd);
            AdamWStep(m_w3g[h][j], m_m_w3g[h][j], m_v_w3g[h][j], gscale * g_w3g[h][j], lr, wd);
         }

         AdamWStep(m_w_mu[h],   m_m_w_mu[h],   m_v_w_mu[h],   gscale * g_w_mu[h],   lr, wd);
         AdamWStep(m_w_logv[h], m_m_w_logv[h], m_v_w_logv[h], gscale * g_w_logv[h], lr, wd);
         AdamWStep(m_w_q25[h],  m_m_w_q25[h],  m_v_w_q25[h],  gscale * g_w_q25[h],  lr, wd);
         AdamWStep(m_w_q75[h],  m_m_w_q75[h],  m_v_w_q75[h],  gscale * g_w_q75[h],  lr, wd);
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         AdamWStep(m_b_cls[c], m_m_b_cls[c], m_v_b_cls[c], gscale * g_b_cls[c], lr, 0.0);
         for(int h=0; h<FXAI_MLP_H2; h++)
            AdamWStep(m_w_cls[c][h], m_m_w_cls[c][h], m_v_w_cls[c][h], gscale * g_w_cls[c][h], lr, wd);
      }

      AdamWStep(m_b_mu,   m_m_b_mu,   m_v_b_mu,   gscale * g_b_mu,   lr, 0.0);
      AdamWStep(m_b_logv, m_m_b_logv, m_v_b_logv, gscale * g_b_logv, lr, 0.0);
      AdamWStep(m_b_q25,  m_m_b_q25,  m_v_b_q25,  gscale * g_b_q25,  lr, 0.0);
      AdamWStep(m_b_q75,  m_m_b_q75,  m_v_b_q75,  gscale * g_b_q75,  lr, 0.0);

      // Lightweight SGD update for context adapter (small block, no moments).
      for(int k=0; k<FXAI_MLP_CTX; k++)
      {
         m_ctx_b[k] -= 0.35 * lr * gscale * g_ctx_b[k];
         m_ctx_b[k] = FXAI_ClipSym(m_ctx_b[k], 4.0);
         for(int j=0; j<FXAI_MLP_CTX; j++)
         {
            m_ctx_w[k][j] -= 0.35 * lr * (gscale * g_ctx_w[k][j] + wd * m_ctx_w[k][j]);
            m_ctx_w[k][j] = FXAI_ClipSym(m_ctx_w[k][j], 4.0);
         }
      }

      m_b_logv = FXAI_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_w_mu[h] = FXAI_ClipSym(m_w_mu[h], 6.0);
         m_w_logv[h] = FXAI_ClipSym(m_w_logv[h], 6.0);
         m_w_q25[h] = FXAI_ClipSym(m_w_q25[h], 6.0);
         m_w_q75[h] = FXAI_ClipSym(m_w_q75[h], 6.0);
         m_b3[h] = FXAI_ClipSym(m_b3[h], 4.0);
         m_b3g[h] = FXAI_ClipSym(m_b3g[h], 4.0);
         for(int c=0; c<FXAI_MLP_CLASSES; c++)
            m_w_cls[c][h] = FXAI_ClipSym(m_w_cls[c][h], 6.0);
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_w3[h][j] = FXAI_ClipSym(m_w3[h][j], 6.0);
            m_w3g[h][j] = FXAI_ClipSym(m_w3g[h][j], 6.0);
         }
      }

      SyncShadow(false);
      UpdateSWA();
      if(!m_shadow_ready && m_step >= 64)
         m_shadow_ready = true;

      double p_raw[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         p_raw[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      UpdateCalibrator3(p_raw, cls, cls_w, lr);

      double p_cal[FXAI_MLP_CLASSES];
      Calibrate3(p_raw, p_cal);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr);
      double active = FXAI_Clamp(1.0 - p_cal[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double ev_after_cost = amp * active - MathMax(0.0, cost);
      UpdateValidationMetrics(cls, p_cal, ev_after_cost);

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hp, sw);

      if(!from_replay)
      {
         datetime t_sample = ResolveContextTime();
         if(t_sample <= 0) t_sample = TimeCurrent();
         int sess = SessionBucket(t_sample);
         PushReplay(cls,
                    x,
                    move_points,
                    cost,
                    ResolveMinMovePoints(),
                    sw,
                    t_sample,
                    sess,
                    m_ctx_regime_id,
                    m_ctx_horizon_minutes,
                    m_ctx_feature_schema_id,
                    m_ctx_normalization_method_id,
                    m_ctx_sequence_bars,
                    m_ctx_point_value,
                    m_ctx_window_size,
                    m_ctx_window);
         UpdateNormStats(x);
         PushHistory(x);
      }
   }

public:
   CFXAIAIMLPTiny(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_MLP_TINY; }
   virtual string AIName(void) const { return "ai_mlp"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_OTHER, caps, 1, 1);

   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double x_norm[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, x_norm);

      double ctx_raw[FXAI_MLP_CTX], ctx_act[FXAI_MLP_CTX], ctx[FXAI_MLP_CTX];
      BuildTemporalContext(xa, ctx_raw);
      AdaptContext(ctx_raw, ctx, ctx_act);

      double h1_raw[FXAI_MLP_H1], h1[FXAI_MLP_H1], h2[FXAI_MLP_H2], h2_norm[FXAI_MLP_H2];
      double h3_tanh[FXAI_MLP_H2], h3_gate[FXAI_MLP_H2], h3[FXAI_MLP_H2];
      double ln_inv_std = 1.0, drop1[FXAI_MLP_H1];
      double logits[FXAI_MLP_CLASSES], probs[FXAI_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x_norm, ctx, m_shadow_ready, false, h1_raw, h1, h2, h2_norm, h3_tanh, h3_gate, h3, ln_inv_std, drop1, logits, probs, mu, logv, q25, q75);

      double p_raw[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         p_raw[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      Calibrate3(p_raw, class_probs);

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr);
      double active = FXAI_Clamp(1.0 - class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double ev = amp * active;

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.65 * ev + 0.35 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);
      if(expected_move_points < 0.0)
         expected_move_points = 0.0;
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
      double x_norm[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, x_norm);
      double ctx_raw[FXAI_MLP_CTX], ctx_act[FXAI_MLP_CTX], ctx[FXAI_MLP_CTX];
      BuildTemporalContext(xa, ctx_raw);
      AdaptContext(ctx_raw, ctx, ctx_act);
      double h1_raw[FXAI_MLP_H1], h1[FXAI_MLP_H1], h2[FXAI_MLP_H2], h2_norm[FXAI_MLP_H2];
      double h3_tanh[FXAI_MLP_H2], h3_gate[FXAI_MLP_H2], h3[FXAI_MLP_H2];
      double ln_inv_std = 1.0, drop1[FXAI_MLP_H1];
      double logits[FXAI_MLP_CLASSES], probs[FXAI_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x_norm, ctx, m_shadow_ready, false, h1_raw, h1, h2, h2_norm, h3_tanh, h3_gate, h3, ln_inv_std, drop1, logits, probs, mu, logv, q25, q75);
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         out.class_probs[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      Calibrate3(out.class_probs, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * MathAbs(q75 - q25));
      double active = FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double ev = amp * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * active);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * active);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(xa,
                              out.move_mean_points,
                              FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                              out.reliability,
                              out.confidence,
                              bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust,
                              out);
      return true;
   }


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_shadow_ready = false;
      m_step = 0;
      m_adam_step = 0;
      m_shadow_decay = 0.995;
      m_symbol_hash = 0.0;
      m_norm_steps = 0;
      m_mlp_replay_head = 0;
      m_mlp_replay_size = 0;
      m_swa_ready = false;
      m_swa_count = 0;
      m_cal3_steps = 0;
      m_val_ready = false;
      m_val_steps = 0;
      m_quality_degraded = false;
      m_quality_heads.Reset();

      ResetHistory();

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         m_class_ema[c] = 1.0;

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         m_b1[h] = 0.0;
         m_sb1[h] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w1[h][i] = 0.0;
            m_sw1[h][i] = 0.0;
         }
         for(int c=0; c<FXAI_MLP_CTX; c++)
         {
            m_w1c[h][c] = 0.0;
            m_sw1c[h][c] = 0.0;
            m_swa_w1c[h][c] = 0.0;
         }
         m_swa_b1[h] = 0.0;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_b2[h] = 0.0;
         m_sb2[h] = 0.0;
         m_b3[h] = 0.0;
         m_sb3[h] = 0.0;
         m_b3g[h] = 0.0;
         m_sb3g[h] = 0.0;
         m_swa_b2[h] = 0.0;
         m_swa_b3[h] = 0.0;
         m_swa_b3g[h] = 0.0;

         m_w_mu[h] = 0.0; m_sw_mu[h] = 0.0;
         m_w_logv[h] = 0.0; m_sw_logv[h] = 0.0;
         m_w_q25[h] = 0.0; m_sw_q25[h] = 0.0;
         m_w_q75[h] = 0.0; m_sw_q75[h] = 0.0;
         m_swa_w_mu[h] = 0.0;
         m_swa_w_logv[h] = 0.0;
         m_swa_w_q25[h] = 0.0;
         m_swa_w_q75[h] = 0.0;

         for(int j=0; j<FXAI_MLP_H1; j++)
         {
            m_w2[h][j] = 0.0;
            m_sw2[h][j] = 0.0;
            m_swa_w2[h][j] = 0.0;
         }
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_w3[h][j] = 0.0;
            m_sw3[h][j] = 0.0;
            m_swa_w3[h][j] = 0.0;
            m_w3g[h][j] = 0.0;
            m_sw3g[h][j] = 0.0;
            m_swa_w3g[h][j] = 0.0;
         }
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_b_cls[c] = 0.0;
         m_sb_cls[c] = 0.0;
         m_swa_b_cls[c] = 0.0;
         for(int h=0; h<FXAI_MLP_H2; h++)
         {
            m_w_cls[c][h] = 0.0;
            m_sw_cls[c][h] = 0.0;
            m_swa_w_cls[c][h] = 0.0;
         }
      }

      m_b_mu = 0.0; m_sb_mu = 0.0;
      m_b_logv = 0.0; m_sb_logv = 0.0;
      m_b_q25 = 0.0; m_sb_q25 = 0.0;
      m_b_q75 = 0.0; m_sb_q75 = 0.0;
      m_swa_b_mu = 0.0;
      m_swa_b_logv = 0.0;
      m_swa_b_q25 = 0.0;
      m_swa_b_q75 = 0.0;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_norm_mean[i] = 0.0;
         m_norm_var[i] = 1.0;
         m_norm_median[i] = 0.0;
         m_norm_iqr[i] = 1.0;
         m_norm_gain[i] = 1.0;
         m_norm_bias[i] = 0.0;
      }
      for(int c=0; c<FXAI_MLP_CTX; c++)
      {
         m_ctx_b[c] = 0.0;
         for(int j=0; j<FXAI_MLP_CTX; j++)
            m_ctx_w[c][j] = (c == j ? 1.0 : 0.0);
      }
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_MLP_CLASSES; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_MLP_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      for(int b=0; b<FXAI_MLP_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;

      for(int r=0; r<FXAI_MLP_REPLAY; r++)
      {
         m_replay_y[r] = (int)FXAI_LABEL_SKIP;
         m_mlp_replay_move[r] = 0.0;
         m_mlp_replay_cost[r] = 0.0;
         m_mlp_replay_min_move[r] = 0.0;
         m_replay_w[r] = 1.0;
         m_mlp_replay_time[r] = 0;
         m_mlp_replay_regime[r] = 0;
         m_mlp_replay_horizon[r] = 1;
         m_replay_session[r] = -1;
         m_mlp_replay_feature_schema[r] = FXAI_SCHEMA_FULL;
         m_mlp_replay_norm_method[r] = FXAI_NORM_EXISTING;
         m_mlp_replay_sequence_bars[r] = 1;
         m_mlp_replay_point_value[r] = (_Point > 0.0 ? _Point : 1.0);
         m_mlp_replay_window_size[r] = 0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_mlp_replay_x[r][i] = 0.0;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_mlp_replay_window[r][b][k] = 0.0;
      }

      InitMoments();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights(hp.mlp_init);
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(xa, move_points);
      m_quality_heads.Update(xa,
                             w,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             h.lr,
                             h.l2);
      datetime cur_t = ResolveContextTime();
      if(cur_t <= 0) cur_t = TimeCurrent();
      double cur_cost = ResolveCostPoints(xa);
      double cur_min = ResolveMinMovePoints();
      int cur_regime = m_ctx_regime_id;
      int cur_session = m_ctx_session_bucket;
      int cur_horizon = m_ctx_horizon_minutes;
      int cur_feature_schema = m_ctx_feature_schema_id;
      int cur_norm_method = m_ctx_normalization_method_id;
      int cur_sequence_bars = m_ctx_sequence_bars;
      double cur_point_value = m_ctx_point_value;
      int cur_window_size = m_ctx_window_size;
      double cur_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            cur_window[b][k] = m_ctx_window[b][k];
      int cur_sess = SessionBucket(cur_t);

      UpdateWeighted(y, xa, h, w, move_points, false);

      int replay_n = 0;
      if(m_mlp_replay_size >= 96) replay_n = 3;
      else if(m_mlp_replay_size >= 32) replay_n = 2;
      else if(m_mlp_replay_size >= 12) replay_n = 1;

      for(int r=0; r<replay_n; r++)
      {
         if(m_mlp_replay_size <= 0) break;
         int li = PluginRandIndex(m_mlp_replay_size);
         int p = ReplayPos(li);
         double rw = FXAI_Clamp(m_replay_w[p], 0.20, 4.00);
         rw *= ReplayAgeWeight(m_mlp_replay_time[p], cur_t);
         if(m_replay_session[p] >= 0 && m_replay_session[p] != cur_sess) rw *= 0.85;
         FXAIAIContextV4 replay_ctx;
         FXAI_ClearContextV4(replay_ctx);
         replay_ctx.sample_time = m_mlp_replay_time[p];
         replay_ctx.cost_points = m_mlp_replay_cost[p];
         replay_ctx.min_move_points = m_mlp_replay_min_move[p];
         replay_ctx.regime_id = m_mlp_replay_regime[p];
         replay_ctx.session_bucket = (m_replay_session[p] >= 0 ? m_replay_session[p] : cur_sess);
         replay_ctx.horizon_minutes = m_mlp_replay_horizon[p];
         replay_ctx.feature_schema_id = m_mlp_replay_feature_schema[p];
         replay_ctx.normalization_method_id = m_mlp_replay_norm_method[p];
         replay_ctx.sequence_bars = m_mlp_replay_sequence_bars[p];
         replay_ctx.point_value = m_mlp_replay_point_value[p];
         SetContext(replay_ctx);
         double replay_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               replay_window[b][k] = m_mlp_replay_window[p][b][k];
         SetWindowPayload(m_mlp_replay_window_size[p], replay_window);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            replay_x[i] = m_mlp_replay_x[p][i];
         UpdateWeighted(m_replay_y[p], replay_x, h, rw, m_mlp_replay_move[p], true);
      }

      FXAIAIContextV4 cur_ctx;
      FXAI_ClearContextV4(cur_ctx);
      cur_ctx.sample_time = cur_t;
      cur_ctx.cost_points = cur_cost;
      cur_ctx.min_move_points = cur_min;
      cur_ctx.regime_id = cur_regime;
      cur_ctx.session_bucket = cur_session;
      cur_ctx.horizon_minutes = cur_horizon;
      cur_ctx.feature_schema_id = cur_feature_schema;
      cur_ctx.normalization_method_id = cur_norm_method;
      cur_ctx.sequence_bars = cur_sequence_bars;
      cur_ctx.point_value = cur_point_value;
      SetContext(cur_ctx);
      m_ctx_window_size = cur_window_size;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = cur_window[b][k];
   }

   virtual double PredictProb(const double &x[],
                              const FXAIAIHyperParams &hp)
   {
      double cp[FXAI_MLP_CLASSES];
      double em = 0.0;
      PredictModelCore(x, hp, cp, em);
      double den = cp[(int)FXAI_LABEL_BUY] + cp[(int)FXAI_LABEL_SELL];
      if(den <= 1e-9) return 0.5;
      return FXAI_Clamp(cp[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[],
                                            const FXAIAIHyperParams &hp)
   {
      double cp[FXAI_MLP_CLASSES];
      double em = 0.0;
      PredictModelCore(x, hp, cp, em);
      return em;
   }
};

#endif // __FXAI_AI_MLP_TINY_MQH__
