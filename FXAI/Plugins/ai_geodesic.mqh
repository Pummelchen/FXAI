#ifndef __FXAI_AI_GEODESICATTENTION_MQH__
#define __FXAI_AI_GEODESICATTENTION_MQH__

#include "..\API\plugin_base.mqh"

// Reference-grade geodesic attention plugin.
#define FXAI_GA_SEQ 96
#define FXAI_GA_TBPTT 24
#define FXAI_GA_HEADS 4
#define FXAI_GA_D_MODEL 32
#define FXAI_GA_D_HEAD (FXAI_GA_D_MODEL / FXAI_GA_HEADS)
#define FXAI_GA_CLASS_COUNT 3
#define FXAI_GA_BLOCKS 2
#define FXAI_GA_QUANTILES 7
#define FXAI_GA_CAL_BINS 16
#define FXAI_GA_ECE_BINS 12
#define FXAI_GA_REPLAY 256
#define FXAI_GA_SELL 0
#define FXAI_GA_BUY  1
#define FXAI_GA_SKIP 2

class CFXAIAIGeodesicAttention : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_step;

   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq_h[FXAI_GA_SEQ][FXAI_GA_D_MODEL];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // Variable-selection groups: observed / known / static.
   double m_group_mask[3][FXAI_AI_WEIGHTS];
   double m_v_gate_w[3][FXAI_AI_WEIGHTS];
   double m_v_gate_b[3];

   double m_w_obs[FXAI_GA_D_MODEL][FXAI_AI_WEIGHTS];
   double m_w_known[FXAI_GA_D_MODEL][FXAI_AI_WEIGHTS];
   double m_w_static[FXAI_GA_D_MODEL][FXAI_AI_WEIGHTS];
   double m_b_embed[FXAI_GA_D_MODEL];

   // GRN blocks.
   double m_grn1_w1[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn1_w2[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn1_wg[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn1_b1[FXAI_GA_D_MODEL];
   double m_grn1_b2[FXAI_GA_D_MODEL];
   double m_grn1_bg[FXAI_GA_D_MODEL];

   double m_grn2_w1[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn2_w2[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn2_wg[FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_grn2_b1[FXAI_GA_D_MODEL];
   double m_grn2_b2[FXAI_GA_D_MODEL];
   double m_grn2_bg[FXAI_GA_D_MODEL];

   // Two-block geodesic temporal attention stack.
   double m_wq[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_GA_D_MODEL];
   double m_wk[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_GA_D_MODEL];
   double m_wv[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_GA_D_MODEL];
   double m_wo[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_MODEL][FXAI_GA_D_HEAD];

   double m_geo_kappa[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
   double m_geo_beta[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
   double m_geo_time_beta[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
   double m_geo_mix[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
   double m_geo_bias[FXAI_GA_BLOCKS][FXAI_GA_HEADS];

   double m_attn_wg[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_attn_bg[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];

   // Per-block feed-forward blocks.
   double m_ff1_w[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_ff1_b[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
   double m_ff2_w[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL][FXAI_GA_D_MODEL];
   double m_ff2_b[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];

   // Normalization affine params.
   double m_ln1_g[FXAI_GA_D_MODEL];
   double m_ln1_b[FXAI_GA_D_MODEL];
   double m_ln_attn_g[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
   double m_ln_attn_b[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
   double m_ln2_g[FXAI_GA_D_MODEL];
   double m_ln2_b[FXAI_GA_D_MODEL];
   double m_ln_out_g[FXAI_GA_D_MODEL];
   double m_ln_out_b[FXAI_GA_D_MODEL];

   // Output heads: true 3-class + distributional move (full quantile grid).
   double m_w_cls[FXAI_GA_CLASS_COUNT][FXAI_GA_D_MODEL];
   double m_b_cls[FXAI_GA_CLASS_COUNT];

   double m_w_mu[FXAI_GA_D_MODEL];
   double m_b_mu;
   double m_w_logv[FXAI_GA_D_MODEL];
   double m_b_logv;
   double m_w_q[FXAI_GA_QUANTILES][FXAI_GA_D_MODEL];
   double m_b_q[FXAI_GA_QUANTILES];

   // TBPTT buffers.
   int    m_train_len;
   double m_train_x[FXAI_GA_TBPTT][FXAI_AI_WEIGHTS];
   int    m_train_cls[FXAI_GA_TBPTT];
   double m_train_move[FXAI_GA_TBPTT];
   double m_train_cost[FXAI_GA_TBPTT];
   double m_train_w[FXAI_GA_TBPTT];

   // Forward caches for TBPTT updates.
   double m_cache_embed[FXAI_GA_TBPTT][FXAI_GA_D_MODEL];
   double m_cache_local[FXAI_GA_TBPTT][FXAI_GA_D_MODEL];
   double m_cache_block_in[FXAI_GA_TBPTT][FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
   double m_cache_block_out[FXAI_GA_TBPTT][FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
   double m_cache_final[FXAI_GA_TBPTT][FXAI_GA_D_MODEL];
   double m_cache_group[FXAI_GA_TBPTT][3];
   double m_cache_ctx[FXAI_GA_TBPTT][FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
   double m_cache_geo_cons[FXAI_GA_TBPTT][FXAI_GA_BLOCKS][FXAI_GA_HEADS];

   double m_cache_logits[FXAI_GA_TBPTT][FXAI_GA_CLASS_COUNT];
   double m_cache_probs[FXAI_GA_TBPTT][FXAI_GA_CLASS_COUNT];
   double m_cache_mu[FXAI_GA_TBPTT];
   double m_cache_logv[FXAI_GA_TBPTT];
   double m_cache_q[FXAI_GA_TBPTT][FXAI_GA_QUANTILES];

   // Native 3-class calibration: vector scaling + isotonic bins.
   double m_cal_vs_w[FXAI_GA_CLASS_COUNT][FXAI_GA_CLASS_COUNT];
   double m_cal_vs_b[FXAI_GA_CLASS_COUNT];
   double m_cal_iso_pos[FXAI_GA_CLASS_COUNT][FXAI_GA_CAL_BINS];
   double m_cal_iso_cnt[FXAI_GA_CLASS_COUNT][FXAI_GA_CAL_BINS];
   int    m_cal3_steps;

   // Validation and calibration quality tracking.
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_nll_fast;
   double m_val_nll_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_val_ev_fast;
   double m_val_ev_slow;
   double m_ece_mass[FXAI_GA_ECE_BINS];
   double m_ece_acc[FXAI_GA_ECE_BINS];
   double m_ece_conf[FXAI_GA_ECE_BINS];
   bool   m_quality_degraded;

   // Replay and teacher distillation.
   int    m_ga_replay_head;
   int    m_ga_replay_size;
   double m_ga_replay_x[FXAI_GA_REPLAY][FXAI_AI_WEIGHTS];
   int    m_replay_cls[FXAI_GA_REPLAY];
   double m_ga_replay_move[FXAI_GA_REPLAY];
   double m_ga_replay_cost[FXAI_GA_REPLAY];
   double m_replay_w[FXAI_GA_REPLAY];

   double m_t_w_cls[FXAI_GA_CLASS_COUNT][FXAI_GA_D_MODEL];
   double m_t_b_cls[FXAI_GA_CLASS_COUNT];

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   double QuantileLevel(const int qi) const
   {
      static const double qv[FXAI_GA_QUANTILES] = {0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95};
      int q = ClampI(qi, 0, FXAI_GA_QUANTILES - 1);
      return qv[q];
   }

   double ScheduledLR(const FXAIAIHyperParams &hp) const
   {
      double base = FXAI_Clamp(hp.lr, 0.0002, 0.0500);
      double st = (double)MathMax(m_step, 1);
      double warm = FXAI_Clamp(st / 160.0, 0.05, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0012 * MathMax(0.0, st - 160.0));
      return FXAI_Clamp(base * warm * invsqrt, 0.00002, 0.05000);
   }

   void LayerNormAffine(double &v[], const double &g[], const double &b[]) const
   {
      double mean = 0.0;
      for(int i=0; i<FXAI_GA_D_MODEL; i++)
         mean += v[i];
      mean /= (double)FXAI_GA_D_MODEL;

      double var = 0.0;
      for(int i=0; i<FXAI_GA_D_MODEL; i++)
      {
         double d = v[i] - mean;
         var += d * d;
      }

      double inv = 1.0 / MathSqrt(var / (double)FXAI_GA_D_MODEL + 1e-6);
      for(int i=0; i<FXAI_GA_D_MODEL; i++)
      {
         double n = (v[i] - mean) * inv;
         v[i] = FXAI_ClipSym(g[i] * n + b[i], 8.0);
      }
   }

   void LayerNormAffineBlock(const int block, double &v[]) const
   {
      double g[FXAI_GA_D_MODEL];
      double b[FXAI_GA_D_MODEL];
      int bi = ClampI(block, 0, FXAI_GA_BLOCKS - 1);
      for(int i=0; i<FXAI_GA_D_MODEL; i++)
      {
         g[i] = m_ln_attn_g[bi][i];
         b[i] = m_ln_attn_b[bi][i];
      }
      LayerNormAffine(v, g, b);
   }

   void Softmax3(const double &logits[], double &probs[]) const
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

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FXAI_GA_SEQ; t++)
      {
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
            m_seq_h[t][h] = 0.0;
      }
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FXAI_GA_TBPTT; t++)
      {
         m_train_cls[t] = FXAI_GA_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;

         m_cache_mu[t] = 0.0;
         m_cache_logv[t] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[t][i] = 0.0;
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
         {
            m_cache_embed[t][h] = 0.0;
            m_cache_local[t][h] = 0.0;
            m_cache_final[t][h] = 0.0;
            for(int b=0; b<FXAI_GA_BLOCKS; b++)
            {
               m_cache_block_in[t][b][h] = 0.0;
               m_cache_block_out[t][b][h] = 0.0;
            }
         }

         for(int g=0; g<3; g++) m_cache_group[t][g] = 0.0;
         for(int b=0; b<FXAI_GA_BLOCKS; b++)
         {
            for(int hd=0; hd<FXAI_GA_HEADS; hd++)
            {
               m_cache_geo_cons[t][b][hd] = 0.0;
               for(int d=0; d<FXAI_GA_D_HEAD; d++)
                  m_cache_ctx[t][b][hd][d] = 0.0;
            }
         }

         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = 0.0;
            m_cache_probs[t][c] = 1.0 / 3.0;
         }
         for(int q=0; q<FXAI_GA_QUANTILES; q++)
            m_cache_q[t][q] = 0.0;
      }
   }

   void ResetInputNorm(void)
   {
      m_x_norm_ready = false;
      m_x_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 192 ? 0.040 : 0.012);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_x_norm_steps++;
      if(m_x_norm_steps >= 32) m_x_norm_ready = true;
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn ? x[i] : 0.0);
         xa[i] = (MathIsValidNumber(v) ? FXAI_ClipSym(v, 8.0) : 0.0);
      }

      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;

      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean4 = CurrentWindowFeatureMean(3);
      double first1 = CurrentWindowValue(0, 1);
      double last1 = CurrentWindowValue(win_n - 1, 1);
      double slope1 = (last1 - first1) / (double)MathMax(win_n - 1, 1);
      double curv = 0.0;
      for(int i=2; i<win_n; i++)
      {
         double a = CurrentWindowValue(i - 2, 1);
         double b = CurrentWindowValue(i - 1, 1);
         double c = CurrentWindowValue(i, 1);
         curv += MathAbs(c - 2.0 * b + a);
      }
      curv /= (double)MathMax(win_n - 2, 1);

      xa[1] = FXAI_ClipSym(0.55 * xa[1] + 0.25 * mean1 + 0.20 * slope1, 8.0);
      xa[2] = FXAI_ClipSym(0.60 * xa[2] + 0.25 * mean2 + 0.15 * curv, 8.0);
      xa[4] = FXAI_ClipSym(0.70 * xa[4] + 0.30 * mean4, 8.0);
      xa[5] = FXAI_ClipSym(0.70 * xa[5] + 0.30 * curv, 8.0);
   }

   void NormalizeInput(const double &x[], double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         if(!m_x_norm_ready)
         {
            xn[i] = FXAI_ClipSym(x[i], 8.0);
            continue;
         }

         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   int SeqIndexBack(const int back) const
   {
      if(m_seq_len <= 0 || m_seq_ptr < 0) return 0;
      int idx = m_seq_ptr - back;
      while(idx < 0) idx += FXAI_GA_SEQ;
      while(idx >= FXAI_GA_SEQ) idx -= FXAI_GA_SEQ;
      return idx;
   }

   int MapClass(const int y,
                const double &x[],
                const double move_points) const
   {
      if(y == FXAI_GA_SELL || y == FXAI_GA_BUY || y == FXAI_GA_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FXAI_GA_SKIP;

      if(y > 0) return FXAI_GA_BUY;
      if(y == 0) return FXAI_GA_SELL;
      return (move_points >= 0.0 ? FXAI_GA_BUY : FXAI_GA_SELL);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost,
                      const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double base = FXAI_Clamp(sample_w, 0.25, 4.00);

      if(cls == FXAI_GA_SKIP)
      {
         if(edge <= 0.0) return FXAI_Clamp(base * 1.8, 0.25, 8.0);
         return FXAI_Clamp(base * 0.80, 0.25, 8.0);
      }

      if(edge <= 0.0) return FXAI_Clamp(base * 0.60, 0.25, 8.0);
      return FXAI_Clamp(base * (1.0 + 0.08 * MathMin(edge, 25.0)), 0.25, 8.0);
   }

   double MoveWeight(const double move_points,
                     const double cost,
                     const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double ew = FXAI_Clamp(0.5 + edge / denom, 0.20, 4.0);
      return FXAI_Clamp(sample_w * ew, 0.20, 10.0);
   }

   void InitGroupMasks(void)
   {
      for(int g=0; g<3; g++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_group_mask[g][i] = 0.0;
      }

      for(int g=0; g<3; g++) m_group_mask[g][0] = 1.0;

      // feature index = i - 1 (x[0] is bias)
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         int fi = i - 1;

         bool obs = (fi <= 5 || fi == 7 || fi == 8 || fi == 9 || fi == 13 || fi == 14);
         bool known = (fi == 6 || fi == 10 || fi == 11 || fi == 12);
         bool stat = (fi == 4 || fi == 6 || fi == 10 || fi == 11 || fi == 12 || fi == 13 || fi == 14);

         m_group_mask[0][i] = (obs ? 1.0 : 0.0);
         m_group_mask[1][i] = (known ? 1.0 : 0.0);
         m_group_mask[2][i] = (stat ? 1.0 : 0.0);
      }
   }

   void BuildCalLogits(const double &p_raw[], double &logits[]) const
   {
      double lraw[FXAI_GA_CLASS_COUNT];
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double z = m_cal_vs_b[c];
         for(int j=0; j<FXAI_GA_CLASS_COUNT; j++)
            z += m_cal_vs_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void Calibrate3(const double &p_raw[], double &p_cal[]) const
   {
      double logits[FXAI_GA_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 24) return;

      double p_iso[FXAI_GA_CLASS_COUNT];
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_GA_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_GA_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_GA_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9)
               r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_GA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_GA_CAL_BINS) bi = FXAI_GA_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
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
      double logits[FXAI_GA_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);

      double p_cal[FXAI_GA_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_GA_CLASS_COUNT];
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.25 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_vs_b[c] = FXAI_ClipSym(m_cal_vs_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<FXAI_GA_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_cal_vs_w[c][j] - target_w);
            m_cal_vs_w[c][j] = FXAI_ClipSym(m_cal_vs_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_GA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_GA_CAL_BINS) bi = FXAI_GA_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }

      m_cal3_steps++;
   }

   void UpdateValidationMetrics(const int cls,
                                const double &p_cal[],
                                const double expected_move_points,
                                const double cost_points)
   {
      int y = ClampI(cls, 0, FXAI_GA_CLASS_COUNT - 1);
      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= 3.0;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_GA_CLASS_COUNT; c++)
      {
         if(p_cal[c] > conf) { conf = p_cal[c]; pred = c; }
      }
      double acc = (pred == y ? 1.0 : 0.0);

      int bi = (int)MathFloor(conf * (double)FXAI_GA_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_GA_ECE_BINS) bi = FXAI_GA_ECE_BINS - 1;
      for(int b=0; b<FXAI_GA_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      m_ece_mass[bi] += 1.0;
      m_ece_acc[bi] += acc;
      m_ece_conf[bi] += conf;

      double ece_num = 0.0, ece_den = 0.0;
      for(int b=0; b<FXAI_GA_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);

      double ev_after_cost = expected_move_points - MathMax(cost_points, 0.0);
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

   void ReplayPush(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double sample_w)
   {
      int p = m_ga_replay_head;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_ga_replay_x[p][k] = x[k];
      m_replay_cls[p] = cls;
      m_ga_replay_move[p] = move_points;
      m_ga_replay_cost[p] = cost_points;
      m_replay_w[p] = sample_w;
      m_ga_replay_head++;
      if(m_ga_replay_head >= FXAI_GA_REPLAY) m_ga_replay_head = 0;
      if(m_ga_replay_size < FXAI_GA_REPLAY) m_ga_replay_size++;
   }

   int ReplaySampleSlot(void)
   {
      if(m_ga_replay_size <= 0) return -1;
      double u = PluginRand01();
      int age = (int)MathFloor(u * (double)m_ga_replay_size);
      if(age < 0) age = 0;
      if(age >= m_ga_replay_size) age = m_ga_replay_size - 1;

      int slot = m_ga_replay_head - 1 - age;
      while(slot < 0) slot += FXAI_GA_REPLAY;
      while(slot >= FXAI_GA_REPLAY) slot -= FXAI_GA_REPLAY;
      return slot;
   }

   void UpdateTeacherHeads(void)
   {
      const double a = 0.995;
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         m_t_b_cls[c] = a * m_t_b_cls[c] + (1.0 - a) * m_b_cls[c];
         for(int d=0; d<FXAI_GA_D_MODEL; d++)
            m_t_w_cls[c][d] = a * m_t_w_cls[c][d] + (1.0 - a) * m_w_cls[c][d];
      }
   }

   double ExpectedMoveFromHeads(const double mu,
                                const double logv,
                                const double &q_all[],
                                const double skip_prob) const
   {
      double sigma = MathExp(0.5 * FXAI_Clamp(logv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q_all[4] - q_all[2]);
      double tail = MathAbs(q_all[6] - q_all[0]);
      double ev = (0.58 * MathAbs(mu) + 0.22 * sigma + 0.14 * iqr + 0.06 * tail) * FXAI_Clamp(1.0 - skip_prob, 0.0, 1.0);
      return ev;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
      InitGroupMasks();
      m_step = 0;

      // Calibrator init.
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_GA_CLASS_COUNT; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_GA_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal3_steps = 0;

      // Validation init.
      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      m_quality_degraded = false;
      for(int b=0; b<FXAI_GA_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }

      // Replay init.
      m_ga_replay_head = 0;
      m_ga_replay_size = 0;
      for(int r=0; r<FXAI_GA_REPLAY; r++)
      {
         m_replay_cls[r] = FXAI_GA_SKIP;
         m_ga_replay_move[r] = 0.0;
         m_ga_replay_cost[r] = 0.0;
         m_replay_w[r] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_ga_replay_x[r][k] = 0.0;
      }

      for(int g=0; g<3; g++)
      {
         m_v_gate_b[g] = (g == 0 ? 0.25 : 0.0);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s = (double)((g + 1) * (i + 2));
            m_v_gate_w[g][i] = 0.03 * MathSin(0.71 * s);
         }
      }

      for(int o=0; o<FXAI_GA_D_MODEL; o++)
      {
         m_b_embed[o] = 0.0;

         m_grn1_b1[o] = 0.0;
         m_grn1_b2[o] = 0.0;
         m_grn1_bg[o] = 0.0;

         m_grn2_b1[o] = 0.0;
         m_grn2_b2[o] = 0.0;
         m_grn2_bg[o] = 0.0;

         m_ln1_g[o] = 1.0;
         m_ln1_b[o] = 0.0;
         m_ln2_g[o] = 1.0;
         m_ln2_b[o] = 0.0;
         m_ln_out_g[o] = 1.0;
         m_ln_out_b[o] = 0.0;

         m_w_mu[o] = 0.03 * MathSin((double)(o + 1) * 0.93);
         m_w_logv[o] = 0.03 * MathCos((double)(o + 2) * 1.01);
         for(int q=0; q<FXAI_GA_QUANTILES; q++)
            m_w_q[q][o] = 0.025 * MathSin((double)((q + 2) * (o + 3)) * 0.67);

         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_w_cls[c][o] = 0.03 * MathSin((double)((c + 2) * (o + 1)) * 0.87);
            m_t_w_cls[c][o] = m_w_cls[c][o];
         }

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s0 = (double)((o + 1) * (i + 3));
            double s1 = (double)((o + 2) * (i + 5));
            double s2 = (double)((o + 3) * (i + 7));
            m_w_obs[o][i] = 0.03 * MathSin(0.67 * s0);
            m_w_known[o][i] = 0.03 * MathCos(0.73 * s1);
            m_w_static[o][i] = 0.03 * MathSin(0.79 * s2);
         }

         for(int i=0; i<FXAI_GA_D_MODEL; i++)
         {
            double a = (double)((o + 1) * (i + 2));

            m_grn1_w1[o][i] = 0.04 * MathSin(0.61 * a);
            m_grn1_w2[o][i] = 0.04 * MathCos(0.67 * a);
            m_grn1_wg[o][i] = 0.03 * MathSin(0.71 * a);

            m_grn2_w1[o][i] = 0.04 * MathCos(0.73 * a);
            m_grn2_w2[o][i] = 0.04 * MathSin(0.77 * a);
            m_grn2_wg[o][i] = 0.03 * MathCos(0.81 * a);
         }
      }

      for(int b=0; b<FXAI_GA_BLOCKS; b++)
      {
         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            m_attn_bg[b][o] = 0.0;
            m_ff1_b[b][o] = 0.0;
            m_ff2_b[b][o] = 0.0;
            m_ln_attn_g[b][o] = 1.0;
            m_ln_attn_b[b][o] = 0.0;

            for(int i=0; i<FXAI_GA_D_MODEL; i++)
            {
               double a = (double)((b + 1) * (o + 2) * (i + 3));
               m_attn_wg[b][o][i] = 0.03 * MathSin(0.83 * a);
               m_ff1_w[b][o][i] = 0.04 * MathCos(0.57 * a);
               m_ff2_w[b][o][i] = 0.04 * MathSin(0.59 * a);
            }
         }

         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            m_geo_kappa[b][hd] = 1.00 + 0.10 * (double)b + 0.04 * (double)hd;
            m_geo_beta[b][hd] = 1.00 + 0.12 * (double)hd;
            m_geo_time_beta[b][hd] = 0.35 + 0.07 * (double)hd;
            m_geo_mix[b][hd] = -0.20 + 0.15 * (double)hd;
            m_geo_bias[b][hd] = 0.0;

            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               for(int h=0; h<FXAI_GA_D_MODEL; h++)
               {
                  double t = (double)((b + 1) * (hd + 2) * (d + 3) * (h + 1));
                  m_wq[b][hd][d][h] = 0.03 * MathSin(0.69 * t);
                  m_wk[b][hd][d][h] = 0.03 * MathCos(0.73 * t);
                  m_wv[b][hd][d][h] = 0.03 * MathSin(0.79 * t);
                  m_wo[b][hd][h][d] = 0.03 * MathCos(0.83 * t);
               }
            }
         }
      }

      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         m_b_cls[c] = 0.0;
         m_t_b_cls[c] = 0.0;
      }

      m_b_mu = 0.0;
      m_b_logv = 0.0;
      for(int q=0; q<FXAI_GA_QUANTILES; q++) m_b_q[q] = 0.0;

      // Slight skip prior before calibration settles.
      m_b_cls[FXAI_GA_SKIP] = 0.18;

      m_initialized = true;
   }

   void ApplyGRN(const double &in[],
                 const double &w1[][FXAI_GA_D_MODEL],
                 const double &w2[][FXAI_GA_D_MODEL],
                 const double &wg[][FXAI_GA_D_MODEL],
                 const double &b1[],
                 const double &b2[],
                 const double &bg[],
                 const double &ln_g[],
                 const double &ln_b[],
                 double &out[]) const
   {
      double h1[FXAI_GA_D_MODEL];
      for(int o=0; o<FXAI_GA_D_MODEL; o++)
      {
         double z1 = b1[o];
         double zg = bg[o];
         for(int i=0; i<FXAI_GA_D_MODEL; i++)
         {
            z1 += w1[o][i] * in[i];
            zg += wg[o][i] * in[i];
         }
         h1[o] = FXAI_Tanh(z1);

         double z2 = b2[o];
         for(int i=0; i<FXAI_GA_D_MODEL; i++) z2 += w2[o][i] * h1[i];

         double g = FXAI_Sigmoid(zg);
         out[o] = g * z2 + (1.0 - g) * in[o];
      }

      LayerNormAffine(out, ln_g, ln_b);
   }

   void BuildVariableSelection(const double &xn[],
                               double &embed[],
                               double &group_gate[]) const
   {
      double gl[3];
      double maxs = -1e100;
      for(int g=0; g<3; g++)
      {
         double s = m_v_gate_b[g];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            s += m_v_gate_w[g][i] * (xn[i] * m_group_mask[g][i]);
         gl[g] = s;
         if(s > maxs) maxs = s;
      }

      double den = 0.0;
      for(int g=0; g<3; g++)
      {
         group_gate[g] = MathExp(FXAI_ClipSym(gl[g] - maxs, 30.0));
         den += group_gate[g];
      }
      if(den <= 0.0) den = 1.0;
      for(int g=0; g<3; g++) group_gate[g] /= den;

      for(int h=0; h<FXAI_GA_D_MODEL; h++)
      {
         double so = 0.0;
         double sk = 0.0;
         double ss = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            so += m_w_obs[h][i] * xn[i] * m_group_mask[0][i];
            sk += m_w_known[h][i] * xn[i] * m_group_mask[1][i];
            ss += m_w_static[h][i] * xn[i] * m_group_mask[2][i];
         }

         double z = m_b_embed[h] + group_gate[0] * so + group_gate[1] * sk + group_gate[2] * ss;
         embed[h] = FXAI_Tanh(z);
      }
   }

   void BuildTokens(const double &current[],
                    double &tokens[][FXAI_GA_D_MODEL],
                    int &count) const
   {
      int keep = m_seq_len;
      if(keep > FXAI_GA_SEQ - 1) keep = FXAI_GA_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = SeqIndexBack(back);
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
            tokens[i][h] = m_seq_h[idx][h];
      }

      for(int h=0; h<FXAI_GA_D_MODEL; h++) tokens[keep][h] = current[h];
      count = keep + 1;
   }

   void GeodesicAttentionBlock(const int block,
                               const double &query_src[],
                               const double &tokens[][FXAI_GA_D_MODEL],
                               const int token_count,
                               double &attn_out[],
                               double &head_ctx[][FXAI_GA_D_HEAD],
                               double &head_geo_cons[]) const
   {
      for(int h=0; h<FXAI_GA_D_MODEL; h++) attn_out[h] = 0.0;
      for(int hd=0; hd<FXAI_GA_HEADS; hd++)
      {
         head_geo_cons[hd] = 0.0;
         for(int d=0; d<FXAI_GA_D_HEAD; d++) head_ctx[hd][d] = 0.0;
      }
      if(token_count <= 0) return;

      double q[FXAI_GA_D_HEAD];
      double qn[FXAI_GA_D_HEAD];
      double scores[FXAI_GA_SEQ];
      double k_unit[FXAI_GA_SEQ][FXAI_GA_D_HEAD];
      double v_cache[FXAI_GA_SEQ][FXAI_GA_D_HEAD];
      const double PI = 3.141592653589793;
      double inv_scale = 1.0 / MathSqrt((double)FXAI_GA_D_HEAD);

      for(int hd=0; hd<FXAI_GA_HEADS; hd++)
      {
         // Query projection and normalization.
         double qn2 = 0.0;
         for(int d=0; d<FXAI_GA_D_HEAD; d++)
         {
            double s = 0.0;
            for(int j=0; j<FXAI_GA_D_MODEL; j++) s += m_wq[block][hd][d][j] * query_src[j];
            q[d] = s;
            qn2 += s * s;
         }
         double q_inv = 1.0 / MathSqrt(qn2 + 1e-9);
         for(int d=0; d<FXAI_GA_D_HEAD; d++) qn[d] = q[d] * q_inv;

         double mix = FXAI_Sigmoid(m_geo_mix[block][hd]);
         double beta = FXAI_Clamp(m_geo_beta[block][hd], 0.05, 8.0);
         double time_beta = FXAI_Clamp(m_geo_time_beta[block][hd], 0.0, 4.0);
         double kappa = MathAbs(m_geo_kappa[block][hd]);
         if(kappa < 1e-4) kappa = 1e-4;
         double curv = MathSqrt(kappa);

         double max_sc = -1e100;
         for(int t=0; t<token_count; t++)
         {
            double kn2 = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               double kv = 0.0;
               double vv = 0.0;
               for(int j=0; j<FXAI_GA_D_MODEL; j++)
               {
                  kv += m_wk[block][hd][d][j] * tokens[t][j];
                  vv += m_wv[block][hd][d][j] * tokens[t][j];
               }
               k_unit[t][d] = kv;
               v_cache[t][d] = vv;
               kn2 += kv * kv;
            }

            double k_inv = 1.0 / MathSqrt(kn2 + 1e-9);
            double dot_uv = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               k_unit[t][d] *= k_inv;
               dot_uv += qn[d] * k_unit[t][d];
            }

            dot_uv = FXAI_Clamp(dot_uv, -0.999999, 0.999999);
            double theta = MathArccos(dot_uv);
            double geod_dist = curv * theta;

            double lag_frac = 0.0;
            if(token_count > 1)
               lag_frac = (double)(token_count - 1 - t) / (double)(token_count - 1);
            lag_frac = FXAI_Clamp(lag_frac, 0.0, 1.0);
            double time_geo = PI * lag_frac;

            double content = (dot_uv * inv_scale) + m_geo_bias[block][hd];
            double geo = -(beta * geod_dist) - (time_beta * time_geo);
            double sc = mix * content + (1.0 - mix) * geo;
            scores[t] = sc;
            if(sc > max_sc) max_sc = sc;
         }

         double den = 0.0;
         for(int t=0; t<token_count; t++)
         {
            scores[t] = MathExp(FXAI_ClipSym(scores[t] - max_sc, 30.0));
            den += scores[t];
         }
         if(den <= 0.0) den = 1.0;

         // Weighted context in value space.
         for(int d=0; d<FXAI_GA_D_HEAD; d++)
         {
            double c = 0.0;
            for(int t=0; t<token_count; t++) c += (scores[t] / den) * v_cache[t][d];
            head_ctx[hd][d] = c;
         }

         for(int j=0; j<FXAI_GA_D_MODEL; j++)
         {
            double s = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++) s += m_wo[block][hd][j][d] * head_ctx[hd][d];
            attn_out[j] += s;
         }

         // Log-map / exp-map consistency in tangent space (manifold regularizer signal).
         double tan_mean[FXAI_GA_D_HEAD];
         for(int d=0; d<FXAI_GA_D_HEAD; d++) tan_mean[d] = 0.0;

         for(int t=0; t<token_count; t++)
         {
            double a = scores[t] / den;
            double dot_uv = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++) dot_uv += qn[d] * k_unit[t][d];
            dot_uv = FXAI_Clamp(dot_uv, -0.999999, 0.999999);

            double theta = MathArccos(dot_uv);
            double sin_t = MathSqrt(MathMax(1e-9, 1.0 - dot_uv * dot_uv));
            double fac = theta / sin_t;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               double lv = fac * (k_unit[t][d] - dot_uv * qn[d]);
               tan_mean[d] += a * lv;
            }
         }

         double tv2 = 0.0;
         for(int d=0; d<FXAI_GA_D_HEAD; d++) tv2 += tan_mean[d] * tan_mean[d];
         double tv = MathSqrt(tv2);

         double expq[FXAI_GA_D_HEAD];
         if(tv <= 1e-9)
         {
            for(int d=0; d<FXAI_GA_D_HEAD; d++) expq[d] = qn[d];
         }
         else
         {
            double c = MathCos(tv);
            double s = MathSin(tv) / tv;
            for(int d=0; d<FXAI_GA_D_HEAD; d++) expq[d] = c * qn[d] + s * tan_mean[d];
         }

         double cons = 0.0;
         for(int t=0; t<token_count; t++)
         {
            double a = scores[t] / den;
            double dk = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++) dk += expq[d] * k_unit[t][d];
            dk = FXAI_Clamp(dk, -1.0, 1.0);
            cons += a * (1.0 - dk);
         }
         head_geo_cons[hd] = FXAI_Clamp(cons, 0.0, 2.0);
      }
   }

   void ForwardStep(const double &x[],
                    const bool commit,
                    double &state_embed[],
                    double &state_local[],
                    double &state_final[],
                    double &group_gate[],
                    double &head_ctx[][FXAI_GA_HEADS][FXAI_GA_D_HEAD],
                    double &geo_cons[][FXAI_GA_HEADS],
                    double &block_in[][FXAI_GA_D_MODEL],
                    double &block_out[][FXAI_GA_D_MODEL])
   {
      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);

      BuildVariableSelection(xn, state_embed, group_gate);
      ApplyGRN(state_embed,
               m_grn1_w1,
               m_grn1_w2,
               m_grn1_wg,
               m_grn1_b1,
               m_grn1_b2,
               m_grn1_bg,
               m_ln1_g,
               m_ln1_b,
               state_local);

      double cur[FXAI_GA_D_MODEL];
      for(int o=0; o<FXAI_GA_D_MODEL; o++) cur[o] = state_local[o];

      for(int b=0; b<FXAI_GA_BLOCKS; b++)
      {
         for(int o=0; o<FXAI_GA_D_MODEL; o++) block_in[b][o] = cur[o];

         double tokens[FXAI_GA_SEQ][FXAI_GA_D_MODEL];
         int token_count = 0;
         BuildTokens(cur, tokens, token_count);

         double attn_mix[FXAI_GA_D_MODEL];
         double head_ctx_b[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
         double geo_cons_b[FXAI_GA_HEADS];
         GeodesicAttentionBlock(b, cur, tokens, token_count, attn_mix, head_ctx_b, geo_cons_b);

         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            geo_cons[b][hd] = geo_cons_b[hd];
            for(int d=0; d<FXAI_GA_D_HEAD; d++) head_ctx[b][hd][d] = head_ctx_b[hd][d];
         }

         double attn_gated[FXAI_GA_D_MODEL];
         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            double zg = m_attn_bg[b][o];
            for(int i=0; i<FXAI_GA_D_MODEL; i++) zg += m_attn_wg[b][o][i] * cur[i];
            double g = FXAI_Sigmoid(zg);
            attn_gated[o] = g * attn_mix[o] + (1.0 - g) * cur[o];
         }
         LayerNormAffineBlock(b, attn_gated);

         double ff1[FXAI_GA_D_MODEL];
         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            double z1 = m_ff1_b[b][o];
            for(int i=0; i<FXAI_GA_D_MODEL; i++) z1 += m_ff1_w[b][o][i] * attn_gated[i];
            ff1[o] = FXAI_Tanh(z1);
         }

         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            double z2 = m_ff2_b[b][o];
            for(int i=0; i<FXAI_GA_D_MODEL; i++) z2 += m_ff2_w[b][o][i] * ff1[i];
            cur[o] = attn_gated[o] + 0.25 * FXAI_ClipSym(z2, 8.0);
         }
         LayerNormAffineBlock(b, cur);

         for(int o=0; o<FXAI_GA_D_MODEL; o++) block_out[b][o] = cur[o];
      }

      ApplyGRN(cur,
               m_grn2_w1,
               m_grn2_w2,
               m_grn2_wg,
               m_grn2_b1,
               m_grn2_b2,
               m_grn2_bg,
               m_ln2_g,
               m_ln2_b,
               state_final);

      LayerNormAffine(state_final, m_ln_out_g, m_ln_out_b);

      if(commit)
      {
         m_seq_ptr++;
         if(m_seq_ptr >= FXAI_GA_SEQ) m_seq_ptr = 0;
         for(int h=0; h<FXAI_GA_D_MODEL; h++) m_seq_h[m_seq_ptr][h] = state_final[h];
         if(m_seq_len < FXAI_GA_SEQ) m_seq_len++;
      }
   }

   void ComputeHeads(const double &state_final[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q_all[]) const
   {
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int h=0; h<FXAI_GA_D_MODEL; h++) z += m_w_cls[c][h] * state_final[h];
         logits[c] = FXAI_ClipSym(z, 20.0);
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      for(int h=0; h<FXAI_GA_D_MODEL; h++)
      {
         mu += m_w_mu[h] * state_final[h];
         logv += m_w_logv[h] * state_final[h];
      }
      logv = FXAI_Clamp(logv, -4.0, 4.0);

      for(int q=0; q<FXAI_GA_QUANTILES; q++)
      {
         double z = m_b_q[q];
         for(int h=0; h<FXAI_GA_D_MODEL; h++) z += m_w_q[q][h] * state_final[h];
         q_all[q] = z;
      }

      for(int q=1; q<FXAI_GA_QUANTILES; q++)
      {
         if(q_all[q] < q_all[q - 1]) q_all[q] = q_all[q - 1];
      }
   }

   void AppendTrainSample(const int cls,
                          const double &x[],
                          const double move_points,
                          const double cost_points,
                          const double sample_w)
   {
      if(m_train_len < FXAI_GA_TBPTT)
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

      for(int t=1; t<FXAI_GA_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[FXAI_GA_TBPTT - 1][i] = x[i];
      m_train_cls[FXAI_GA_TBPTT - 1] = cls;
      m_train_move[FXAI_GA_TBPTT - 1] = move_points;
      m_train_cost[FXAI_GA_TBPTT - 1] = cost_points;
      m_train_w[FXAI_GA_TBPTT - 1] = sample_w;
   }

   void TrainReplayMiniBatch(const FXAIAIHyperParams &hp)
   {
      if(m_ga_replay_size < 32) return;

      int steps = 1;
      if(m_ga_replay_size >= 192) steps = 2;
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.1000);
      double lr = FXAI_Clamp(0.20 * ScheduledLR(hp), 0.00002, 0.01000);

      for(int s=0; s<steps; s++)
      {
         int slot = ReplaySampleSlot();
         if(slot < 0) break;
         double xr[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xr[i] = m_ga_replay_x[slot][i];

         double emb[FXAI_GA_D_MODEL];
         double loc[FXAI_GA_D_MODEL];
         double fin[FXAI_GA_D_MODEL];
         double grp[3];
         double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
         double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
         double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
         double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];

         ForwardStep(xr, false, emb, loc, fin, grp, ctx, gcons, bin, bout);

         double logits[FXAI_GA_CLASS_COUNT];
         double probs[FXAI_GA_CLASS_COUNT];
         double mu = 0.0, logv = 0.0;
         double q_all[FXAI_GA_QUANTILES];
         ComputeHeads(fin, logits, probs, mu, logv, q_all);

         int cls = ClampI(m_replay_cls[slot], 0, FXAI_GA_CLASS_COUNT - 1);
         double sw = FXAI_Clamp(0.35 * m_replay_w[slot], 0.05, 2.00);

         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            double target = (c == cls ? 1.0 : 0.0);
            double err = (probs[c] - target) * sw;
            m_b_cls[c] -= lr * err;
            for(int h=0; h<FXAI_GA_D_MODEL; h++)
               m_w_cls[c][h] -= lr * (err * fin[h] + l2 * 0.10 * m_w_cls[c][h]);
         }

         UpdateCalibrator3(probs, cls, sw, lr);
      }
   }

   void TrainTBPTT(const FXAIAIHyperParams &hp)
   {
      if(m_train_len <= 0) return;

      int n = m_train_len;
      if(n > FXAI_GA_TBPTT) n = FXAI_GA_TBPTT;
      if(n <= 0) return;

      double l2 = FXAI_Clamp(hp.l2, 0.0000, 0.1000);

      // Save live state then isolate sequence for TBPTT to avoid hidden-state bleed.
      int saved_ptr = m_seq_ptr;
      int saved_len = m_seq_len;
      double saved_seq[FXAI_GA_SEQ][FXAI_GA_D_MODEL];
      for(int t=0; t<FXAI_GA_SEQ; t++)
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
            saved_seq[t][h] = m_seq_h[t][h];
      ResetSequence();

      // Forward pass over truncated sequence.
      for(int t=0; t<n; t++)
      {
         double xt[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xt[i] = m_train_x[t][i];

         double emb[FXAI_GA_D_MODEL];
         double loc[FXAI_GA_D_MODEL];
         double fin[FXAI_GA_D_MODEL];
         double grp[3];
         double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
         double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
         double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
         double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];

         ForwardStep(xt, true, emb, loc, fin, grp, ctx, gcons, bin, bout);

         for(int h=0; h<FXAI_GA_D_MODEL; h++)
         {
            m_cache_embed[t][h] = emb[h];
            m_cache_local[t][h] = loc[h];
            m_cache_final[t][h] = fin[h];
            for(int b=0; b<FXAI_GA_BLOCKS; b++)
            {
               m_cache_block_in[t][b][h] = bin[b][h];
               m_cache_block_out[t][b][h] = bout[b][h];
            }
         }

         for(int g=0; g<3; g++) m_cache_group[t][g] = grp[g];
         for(int b=0; b<FXAI_GA_BLOCKS; b++)
         {
            for(int hd=0; hd<FXAI_GA_HEADS; hd++)
            {
               m_cache_geo_cons[t][b][hd] = gcons[b][hd];
               for(int d=0; d<FXAI_GA_D_HEAD; d++) m_cache_ctx[t][b][hd][d] = ctx[b][hd][d];
            }
         }

         double logits[FXAI_GA_CLASS_COUNT];
         double probs[FXAI_GA_CLASS_COUNT];
         double qtmp[FXAI_GA_QUANTILES];
         ComputeHeads(fin, logits, probs, m_cache_mu[t], m_cache_logv[t], qtmp);
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = logits[c];
            m_cache_probs[t][c] = probs[c];
         }
         for(int q=0; q<FXAI_GA_QUANTILES; q++) m_cache_q[t][q] = qtmp[q];
      }

      // Reverse-time updates with truncated temporal error flow.
      double next_delta[FXAI_GA_D_MODEL];
      for(int h=0; h<FXAI_GA_D_MODEL; h++) next_delta[h] = 0.0;

      for(int t=n - 1; t>=0; t--)
      {
         m_step++;
         double lr = ScheduledLR(hp);
         if(m_quality_degraded) lr *= 0.80;

         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FXAI_Clamp(m_train_w[t], 0.20, 8.00);

         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);

         // EV-aware class targets.
         double edge = MathAbs(mv) - cost;
         double target_cls[FXAI_GA_CLASS_COUNT];
         target_cls[0] = 0.0; target_cls[1] = 0.0; target_cls[2] = 0.0;
         if(cls == FXAI_GA_SKIP)
         {
            target_cls[FXAI_GA_SKIP] = 1.0;
         }
         else
         {
            int dir = (cls == FXAI_GA_BUY ? FXAI_GA_BUY : FXAI_GA_SELL);
            if(edge <= 0.0)
            {
               target_cls[dir] = 0.35;
               target_cls[FXAI_GA_SKIP] = 0.65;
            }
            else
            {
               double str = FXAI_Clamp(edge / MathMax(cost, 1.0), 0.0, 2.0);
               double pdir = FXAI_Clamp(0.75 + 0.10 * str, 0.75, 0.95);
               target_cls[dir] = pdir;
               target_cls[FXAI_GA_SKIP] = 1.0 - pdir;
            }
         }

         double t_logits[FXAI_GA_CLASS_COUNT];
         double p_teacher[FXAI_GA_CLASS_COUNT];
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            double z = m_t_b_cls[c];
            for(int h=0; h<FXAI_GA_D_MODEL; h++) z += m_t_w_cls[c][h] * m_cache_final[t][h];
            t_logits[c] = z;
         }
         Softmax3(t_logits, p_teacher);

         // Class loss with EV weighting + teacher KL term.
         double ev_mult = FXAI_Clamp(0.65 + MathMax(edge, 0.0) / MathMax(cost, 1.0), 0.40, 3.00);
         double g_logits[FXAI_GA_CLASS_COUNT];
         double kd = 0.15;
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            double gl = (m_cache_probs[t][c] - target_cls[c]) + kd * (m_cache_probs[t][c] - p_teacher[c]);
            g_logits[c] = FXAI_ClipSym(gl * w_cls * ev_mult, 6.0);
         }

         // Distributional move targets.
         double target_move = mv;
         double mu = m_cache_mu[t];
         double logv = m_cache_logv[t];
         double sigma2 = MathExp(logv);
         sigma2 = FXAI_Clamp(sigma2, 0.05, 100.0);

         double err_mu = FXAI_ClipSym(mu - target_move, 30.0);
         double g_mu = FXAI_ClipSym((err_mu / sigma2) * w_mv, 6.0);

         double ratio = (err_mu * err_mu) / sigma2;
         double g_logv = FXAI_ClipSym(0.5 * (1.0 - ratio) * w_mv, 6.0);

         // Full quantile pinball-Huber + monotonicity penalty.
         double g_q[FXAI_GA_QUANTILES];
         const double huber_delta = 1.0;
         for(int q=0; q<FXAI_GA_QUANTILES; q++)
         {
            double qq = m_cache_q[t][q];
            double tau = QuantileLevel(q);
            double e = target_move - qq;
            double ae = MathAbs(e);
            double hub = 0.0;
            if(ae <= huber_delta) hub = e / huber_delta;
            else hub = FXAI_Sign(e);
            double pin = (e >= 0.0 ? tau : (tau - 1.0));
            g_q[q] = FXAI_ClipSym(-(pin * hub) * w_mv, 6.0);
         }

         const double mono_lambda = 0.08;
         for(int q=1; q<FXAI_GA_QUANTILES; q++)
         {
            double gap = m_cache_q[t][q - 1] - m_cache_q[t][q];
            if(gap > 0.0)
            {
               g_q[q - 1] += mono_lambda * gap;
               g_q[q] -= mono_lambda * gap;
            }
         }

         // Representation delta with truncated temporal flow.
         double delta[FXAI_GA_D_MODEL];
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
         {
            double d = 0.35 * next_delta[h];
            for(int c=0; c<FXAI_GA_CLASS_COUNT; c++) d += g_logits[c] * m_w_cls[c][h];
            d += g_mu * m_w_mu[h];
            d += g_logv * m_w_logv[h];
            for(int q=0; q<FXAI_GA_QUANTILES; q++) d += 0.20 * g_q[q] * m_w_q[q][h];
            delta[h] = d;
         }

         // Full gradient-norm clipping.
         double gnorm2 = g_mu * g_mu + g_logv * g_logv;
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++) gnorm2 += g_logits[c] * g_logits[c];
         for(int q=0; q<FXAI_GA_QUANTILES; q++) gnorm2 += g_q[q] * g_q[q];
         for(int h=0; h<FXAI_GA_D_MODEL; h++) gnorm2 += delta[h] * delta[h];
         double gnorm = MathSqrt(gnorm2);
         if(gnorm > 2.5)
         {
            double sc = 2.5 / MathMax(gnorm, 1e-9);
            g_mu *= sc;
            g_logv *= sc;
            for(int c=0; c<FXAI_GA_CLASS_COUNT; c++) g_logits[c] *= sc;
            for(int q=0; q<FXAI_GA_QUANTILES; q++) g_q[q] *= sc;
            for(int h=0; h<FXAI_GA_D_MODEL; h++) delta[h] *= sc;
         }

         // Head updates.
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_b_cls[c] -= lr * g_logits[c];
            m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 8.0);
            for(int h=0; h<FXAI_GA_D_MODEL; h++)
            {
               double g = g_logits[c] * m_cache_final[t][h];
               m_w_cls[c][h] -= lr * (FXAI_ClipSym(g, 8.0) + l2 * 0.15 * m_w_cls[c][h]);
            }
         }

         m_b_mu -= lr * g_mu;
         m_b_logv -= lr * g_logv;
         m_b_mu = FXAI_ClipSym(m_b_mu, 20.0);
         m_b_logv = FXAI_Clamp(m_b_logv, -4.0, 4.0);

         for(int q=0; q<FXAI_GA_QUANTILES; q++)
         {
            m_b_q[q] -= lr * g_q[q];
            m_b_q[q] = FXAI_ClipSym(m_b_q[q], 20.0);
         }

         for(int h=0; h<FXAI_GA_D_MODEL; h++)
         {
            m_w_mu[h] -= lr * (FXAI_ClipSym(g_mu * m_cache_final[t][h], 8.0) + l2 * 0.10 * m_w_mu[h]);
            m_w_logv[h] -= lr * (FXAI_ClipSym(g_logv * m_cache_final[t][h], 8.0) + l2 * 0.10 * m_w_logv[h]);
            for(int q=0; q<FXAI_GA_QUANTILES; q++)
               m_w_q[q][h] -= lr * (FXAI_ClipSym(g_q[q] * m_cache_final[t][h], 8.0) + l2 * 0.08 * m_w_q[q][h]);
         }

         // Block + FF updates.
         for(int b=0; b<FXAI_GA_BLOCKS; b++)
         {
            for(int o=0; o<FXAI_GA_D_MODEL; o++)
            {
               double d = delta[o];
               m_attn_bg[b][o] -= lr * 0.030 * d;
               m_ff2_b[b][o] -= lr * 0.045 * d;
               m_ff1_b[b][o] -= lr * 0.025 * d;

               m_ln_attn_b[b][o] -= lr * 0.006 * d;
               m_ln_attn_g[b][o] -= lr * (0.006 * d * m_cache_block_out[t][b][o] + l2 * 0.03 * m_ln_attn_g[b][o]);

               for(int i=0; i<FXAI_GA_D_MODEL; i++)
               {
                  m_attn_wg[b][o][i] -= lr * (0.030 * FXAI_ClipSym(d * m_cache_block_in[t][b][i], 6.0) + l2 * 0.05 * m_attn_wg[b][o][i]);
                  m_ff2_w[b][o][i] -= lr * (0.045 * FXAI_ClipSym(d * m_cache_block_out[t][b][i], 6.0) + l2 * 0.05 * m_ff2_w[b][o][i]);
                  m_ff1_w[b][o][i] -= lr * (0.025 * FXAI_ClipSym(d * m_cache_block_in[t][b][i], 6.0) + l2 * 0.05 * m_ff1_w[b][o][i]);
               }
            }

            // Attention projections + geodesic parameters.
            double head_delta[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
            for(int hd=0; hd<FXAI_GA_HEADS; hd++)
            {
               for(int d=0; d<FXAI_GA_D_HEAD; d++)
               {
                  double s = 0.0;
                  for(int o=0; o<FXAI_GA_D_MODEL; o++) s += delta[o] * m_wo[b][hd][o][d];
                  head_delta[hd][d] = FXAI_ClipSym(s, 4.0);
               }
            }

            for(int hd=0; hd<FXAI_GA_HEADS; hd++)
            {
               for(int d=0; d<FXAI_GA_D_HEAD; d++)
               {
                  double dh = head_delta[hd][d];
                  for(int h=0; h<FXAI_GA_D_MODEL; h++)
                  {
                     m_wo[b][hd][h][d] -= lr * (0.05 * FXAI_ClipSym(delta[h] * m_cache_ctx[t][b][hd][d], 6.0) + l2 * 0.04 * m_wo[b][hd][h][d]);
                     m_wq[b][hd][d][h] -= lr * (0.03 * FXAI_ClipSym(dh * m_cache_block_in[t][b][h], 6.0) + l2 * 0.04 * m_wq[b][hd][d][h]);
                     m_wk[b][hd][d][h] -= lr * (0.03 * FXAI_ClipSym(dh * m_cache_block_out[t][b][h], 6.0) + l2 * 0.04 * m_wk[b][hd][d][h]);
                     m_wv[b][hd][d][h] -= lr * (0.03 * FXAI_ClipSym(dh * m_cache_block_out[t][b][h], 6.0) + l2 * 0.04 * m_wv[b][hd][d][h]);
                  }
               }

               double geo_err = m_cache_geo_cons[t][b][hd];
               double dir = FXAI_Sign(m_cache_logits[t][FXAI_GA_BUY] - m_cache_logits[t][FXAI_GA_SELL]);
               if(dir == 0.0) dir = 1.0;
               double gaux = FXAI_ClipSym((0.45 * geo_err + 0.20 * MathAbs(dir)) * dir, 4.0);

               m_geo_bias[b][hd] -= lr * 0.007 * gaux;
               m_geo_beta[b][hd] -= lr * (0.006 * gaux + l2 * 0.03 * m_geo_beta[b][hd]);
               m_geo_time_beta[b][hd] -= lr * (0.005 * gaux + l2 * 0.03 * m_geo_time_beta[b][hd]);
               m_geo_mix[b][hd] -= lr * (0.007 * gaux + l2 * 0.03 * m_geo_mix[b][hd]);
               m_geo_kappa[b][hd] -= lr * (0.004 * gaux + l2 * 0.02 * m_geo_kappa[b][hd]);

               m_geo_bias[b][hd] = FXAI_Clamp(m_geo_bias[b][hd], -4.0, 4.0);
               m_geo_beta[b][hd] = FXAI_Clamp(m_geo_beta[b][hd], 0.05, 8.0);
               m_geo_time_beta[b][hd] = FXAI_Clamp(m_geo_time_beta[b][hd], 0.0, 4.0);
               m_geo_mix[b][hd] = FXAI_Clamp(m_geo_mix[b][hd], -6.0, 6.0);
               m_geo_kappa[b][hd] = FXAI_Clamp(m_geo_kappa[b][hd], 0.05, 10.0);
            }
         }

         // GRN + variable-selection updates.
         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            double d = delta[o];

            m_grn2_b2[o] -= lr * 0.07 * d;
            m_grn2_b1[o] -= lr * 0.05 * d;
            m_grn2_bg[o] -= lr * 0.03 * d;

            m_grn1_b2[o] -= lr * 0.05 * d;
            m_grn1_b1[o] -= lr * 0.03 * d;
            m_grn1_bg[o] -= lr * 0.02 * d;

            m_b_embed[o] -= lr * 0.015 * d;

            m_ln_out_b[o] -= lr * 0.004 * d;
            m_ln_out_g[o] -= lr * (0.004 * d * m_cache_final[t][o] + l2 * 0.03 * m_ln_out_g[o]);
            m_ln2_b[o] -= lr * 0.004 * d;
            m_ln2_g[o] -= lr * (0.004 * d * m_cache_local[t][o] + l2 * 0.03 * m_ln2_g[o]);
            m_ln1_b[o] -= lr * 0.003 * d;
            m_ln1_g[o] -= lr * (0.003 * d * m_cache_local[t][o] + l2 * 0.03 * m_ln1_g[o]);

            for(int i=0; i<FXAI_GA_D_MODEL; i++)
            {
               m_grn2_w2[o][i] -= lr * (0.07 * FXAI_ClipSym(d * m_cache_final[t][i], 6.0) + l2 * 0.05 * m_grn2_w2[o][i]);
               m_grn2_w1[o][i] -= lr * (0.05 * FXAI_ClipSym(d * m_cache_local[t][i], 6.0) + l2 * 0.05 * m_grn2_w1[o][i]);
               m_grn2_wg[o][i] -= lr * (0.03 * FXAI_ClipSym(d * m_cache_final[t][i], 6.0) + l2 * 0.05 * m_grn2_wg[o][i]);

               m_grn1_w2[o][i] -= lr * (0.05 * FXAI_ClipSym(d * m_cache_local[t][i], 6.0) + l2 * 0.05 * m_grn1_w2[o][i]);
               m_grn1_w1[o][i] -= lr * (0.03 * FXAI_ClipSym(d * m_cache_embed[t][i], 6.0) + l2 * 0.05 * m_grn1_w1[o][i]);
               m_grn1_wg[o][i] -= lr * (0.02 * FXAI_ClipSym(d * m_cache_local[t][i], 6.0) + l2 * 0.05 * m_grn1_wg[o][i]);
            }
         }

         double xn[FXAI_AI_WEIGHTS];
         double xrow_norm[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xrow_norm[i] = m_train_x[t][i];
         NormalizeInput(xrow_norm, xn);
         for(int g=0; g<3; g++)
         {
            int hs = (g + t) % FXAI_GA_D_MODEL;
            double gs = delta[hs] * m_cache_local[t][hs];
            m_v_gate_b[g] -= lr * 0.003 * FXAI_ClipSym(gs, 3.0);
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xg = xn[i] * m_group_mask[g][i];
               m_v_gate_w[g][i] -= lr * (0.002 * FXAI_ClipSym(gs * xg, 3.0) + l2 * 0.05 * m_v_gate_w[g][i]);
            }
         }

         for(int o=0; o<FXAI_GA_D_MODEL; o++)
         {
            double d = delta[o];
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xi = xn[i];
               double a0 = m_cache_group[t][0] * xi * m_group_mask[0][i];
               double a1 = m_cache_group[t][1] * xi * m_group_mask[1][i];
               double a2 = m_cache_group[t][2] * xi * m_group_mask[2][i];
               m_w_obs[o][i] -= lr * (0.015 * FXAI_ClipSym(d * a0, 3.0) + l2 * 0.04 * m_w_obs[o][i]);
               m_w_known[o][i] -= lr * (0.015 * FXAI_ClipSym(d * a1, 3.0) + l2 * 0.04 * m_w_known[o][i]);
               m_w_static[o][i] -= lr * (0.015 * FXAI_ClipSym(d * a2, 3.0) + l2 * 0.04 * m_w_static[o][i]);
            }
         }

         // Plugin-level 3-class calibrator and validation gating.
         double p_raw_t[FXAI_GA_CLASS_COUNT];
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++) p_raw_t[c] = m_cache_probs[t][c];
         double qtmp2[FXAI_GA_QUANTILES];
         for(int q=0; q<FXAI_GA_QUANTILES; q++) qtmp2[q] = m_cache_q[t][q];
         double p_cal[FXAI_GA_CLASS_COUNT];
         Calibrate3(p_raw_t, p_cal);
         double ev = ExpectedMoveFromHeads(m_cache_mu[t], m_cache_logv[t], qtmp2, p_cal[FXAI_GA_SKIP]);
         UpdateValidationMetrics(cls, p_cal, ev, cost);
         UpdateCalibrator3(p_raw_t, cls, w_cls, lr);

         // Keep base move head in sync.
         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         double xrow_move[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xrow_move[i] = m_train_x[t][i];
         UpdateMoveHead(xrow_move, mv, hp, sw);

         for(int h=0; h<FXAI_GA_D_MODEL; h++) next_delta[h] = delta[h];
      }

      UpdateTeacherHeads();

      // Restore live sequence state.
      m_seq_ptr = saved_ptr;
      m_seq_len = saved_len;
      for(int t=0; t<FXAI_GA_SEQ; t++)
         for(int h=0; h<FXAI_GA_D_MODEL; h++)
            m_seq_h[t][h] = saved_seq[t][h];
   }

public:
   CFXAIAIGeodesicAttention(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_GEODESICATTENTION; }
   virtual string AIName(void) const { return "ai_geodesic"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 128);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, false, emb, loc, fin, grp, ctx, gcons, bin, bout);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0;
      double q_all[FXAI_GA_QUANTILES];
      ComputeHeads(fin, logits, probs, mu, logv, q_all);

      Calibrate3(probs, class_probs);

      double geo_mean = 0.0;
      int geo_n = 0;
      for(int b=0; b<FXAI_GA_BLOCKS; b++)
         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            geo_mean += gcons[b][hd];
            geo_n++;
         }
      if(geo_n <= 0) geo_n = 1;
      geo_mean /= (double)geo_n;
      double ev = ExpectedMoveFromHeads(mu, logv, q_all, class_probs[FXAI_GA_SKIP]);
      ev *= (0.82 + 0.18 * (1.0 - FXAI_Clamp(geo_mean / 2.0, 0.0, 1.0)));
      expected_move_points = (ev > 0.0 ? ev : (m_move_ready ? m_move_ema_abs : 0.0));
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, false, emb, loc, fin, grp, ctx, gcons, bin, bout);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0;
      double q_all[FXAI_GA_QUANTILES];
      ComputeHeads(fin, logits, probs, mu, logv, q_all);
      Calibrate3(probs, out.class_probs);

      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 50.0);
      double geo_mean = 0.0;
      int geo_n = 0;
      for(int b=0; b<FXAI_GA_BLOCKS; b++)
         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            geo_mean += gcons[b][hd];
            geo_n++;
         }
      if(geo_n <= 0) geo_n = 1;
      geo_mean /= (double)geo_n;
      double geo_cons = 1.0 - FXAI_Clamp(geo_mean / 2.0, 0.0, 1.0);
      double ev = ExpectedMoveFromHeads(mu, logv, q_all, out.class_probs[FXAI_GA_SKIP]);
      out.move_mean_points = (ev > 0.0 ? ev * (0.82 + 0.18 * geo_cons) : (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, q_all[1]);
      out.move_q50_points = MathMax(out.move_q25_points, q_all[3]);
      out.move_q75_points = MathMax(out.move_q50_points, q_all[5]);
      if(out.move_q75_points <= 0.0)
      {
         out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
         out.move_q50_points = out.move_mean_points;
         out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      }

      double dir_conf = MathMax(out.class_probs[FXAI_GA_BUY], out.class_probs[FXAI_GA_SELL]);
      out.confidence = FXAI_Clamp(0.50 * dir_conf + 0.15 * (1.0 - out.class_probs[FXAI_GA_SKIP]) + 0.20 * (1.0 - FXAI_Clamp(m_val_ece_slow, 0.0, 1.0)) + 0.15 * geo_cons, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.18 * (1.0 - FXAI_Clamp(m_val_ece_fast, 0.0, 1.0)) + 0.12 * (1.0 - FXAI_Clamp(m_val_nll_fast / 2.5, 0.0, 1.0)) + 0.10 * (m_quality_degraded ? -0.5 : 0.5) + 0.15 * geo_cons, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PopulatePathQualityHeads(out, x, FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), out.reliability, out.confidence);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? FXAI_GA_BUY : FXAI_GA_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      // Controlled reset policy to avoid state bleed in non-stationary jumps.
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      if((m_step % 768) == 0) ResetSequence();
      if(MathAbs(xa[1]) > 8.0 || MathAbs(xa[2]) > 8.0) ResetSequence();

      UpdateInputStats(xa);

      int cls = MapClass(y, xa, move_points);
      double sw = MoveSampleWeight(xa, move_points);
      sw = FXAI_Clamp(sw, 0.20, 8.00);
      double cost = InputCostProxyPoints(x);

      AppendTrainSample(cls, xa, move_points, cost, sw);
      ReplayPush(cls, xa, move_points, cost, sw);

      TrainTBPTT(h);
      TrainReplayMiniBatch(h);

      // Update live state with most recent bar after isolated TBPTT step.
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, true, emb, loc, fin, grp, ctx, gcons, bin, bout);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double expected_move = 0.0;
      if(!PredictModelCore(x, hp, probs, expected_move)) return 0.5;
      double den = probs[FXAI_GA_BUY] + probs[FXAI_GA_SELL];
      if(den < 1e-9) den = 1e-9;
      return FXAI_Clamp(probs[FXAI_GA_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0) return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
};

#endif // __FXAI_AI_GEODESICATTENTION_MQH__
