#ifndef __FX6_AI_TST_MQH__
#define __FX6_AI_TST_MQH__

#include "..\plugin_base.mqh"

#define FX6_TST_SEQ 48
#define FX6_TST_TBPTT 16
#define FX6_TST_HEADS 2
#define FX6_TST_D_HEAD (FX6_AI_MLP_HIDDEN / FX6_TST_HEADS)
#define FX6_TST_CLASS_COUNT 3
#define FX6_TST_MAX_STACK 4
#define FX6_TST_HORIZONS 3
#define FX6_TST_REPLAY 128
#define FX6_TST_SESSIONS 4

#define FX6_TST_SELL 0
#define FX6_TST_BUY  1
#define FX6_TST_SKIP 2

class CFX6AITST : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   bool   m_cfg_ready;
   int    m_step;
   int    m_adam_t;
   int    m_stack_depth;
   double m_drop_rate;
   double m_stoch_rate;

   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq_h[FX6_TST_SEQ][FX6_AI_MLP_HIDDEN];

   // Session-aware conditioning + calibration.
   double m_session_bias[FX6_TST_SESSIONS][FX6_AI_MLP_HIDDEN];
   double m_sess_cal_a[FX6_TST_SESSIONS];
   double m_sess_cal_b[FX6_TST_SESSIONS];
   int    m_sess_cal_steps[FX6_TST_SESSIONS];
   bool   m_sess_cal_ready[FX6_TST_SESSIONS];

   // Replay memory for hard-sample and class-balanced training.
   int    m_replay_len;
   int    m_replay_ptr;
   double m_replay_x[FX6_TST_REPLAY][FX6_AI_WEIGHTS];
   int    m_replay_cls[FX6_TST_REPLAY];
   double m_replay_move[FX6_TST_REPLAY];
   double m_replay_cost[FX6_TST_REPLAY];
   double m_replay_w[FX6_TST_REPLAY];
   double m_replay_hard[FX6_TST_REPLAY];

   // Light AdamW moments on grouped gradient magnitudes.
   double m_opt_m[8];
   double m_opt_v[8];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FX6_AI_WEIGHTS];
   double m_x_var[FX6_AI_WEIGHTS];

   // Variable-selection groups: observed / known / static.
   double m_group_mask[3][FX6_AI_WEIGHTS];
   double m_v_gate_w[3][FX6_AI_WEIGHTS];
   double m_v_gate_b[3];

   double m_w_obs[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_w_known[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_w_static[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_b_embed[FX6_AI_MLP_HIDDEN];

   // GRN blocks.
   double m_grn1_w1[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn1_w2[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn1_wg[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn1_b1[FX6_AI_MLP_HIDDEN];
   double m_grn1_b2[FX6_AI_MLP_HIDDEN];
   double m_grn1_bg[FX6_AI_MLP_HIDDEN];

   double m_grn2_w1[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn2_w2[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn2_wg[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_grn2_b1[FX6_AI_MLP_HIDDEN];
   double m_grn2_b2[FX6_AI_MLP_HIDDEN];
   double m_grn2_bg[FX6_AI_MLP_HIDDEN];

   // Multi-head temporal attention.
   double m_wq[FX6_TST_HEADS][FX6_TST_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wk[FX6_TST_HEADS][FX6_TST_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wv[FX6_TST_HEADS][FX6_TST_D_HEAD][FX6_AI_MLP_HIDDEN];
   double m_wo[FX6_TST_HEADS][FX6_AI_MLP_HIDDEN][FX6_TST_D_HEAD];
   double m_rel_decay[FX6_TST_HEADS];

   double m_attn_wg[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_attn_bg[FX6_AI_MLP_HIDDEN];

   // Position-wise feed-forward block.
   double m_ff1_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ff1_b[FX6_AI_MLP_HIDDEN];
   double m_ff2_w[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ff2_b[FX6_AI_MLP_HIDDEN];

   // Normalization affine params.
   double m_ln1_g[FX6_AI_MLP_HIDDEN];
   double m_ln1_b[FX6_AI_MLP_HIDDEN];
   double m_ln_attn_g[FX6_AI_MLP_HIDDEN];
   double m_ln_attn_b[FX6_AI_MLP_HIDDEN];
   double m_ln2_g[FX6_AI_MLP_HIDDEN];
   double m_ln2_b[FX6_AI_MLP_HIDDEN];
   double m_ln_out_g[FX6_AI_MLP_HIDDEN];
   double m_ln_out_b[FX6_AI_MLP_HIDDEN];

   // Output heads: 3-class + distributional move.
   double m_w_cls[FX6_TST_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_TST_CLASS_COUNT];

   double m_w_mu[FX6_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN];
   double m_b_q75;
   double m_w_mu_h[FX6_TST_HORIZONS][FX6_AI_MLP_HIDDEN];
   double m_b_mu_h[FX6_TST_HORIZONS];

   // TBPTT buffers.
   int    m_train_len;
   double m_train_x[FX6_TST_TBPTT][FX6_AI_WEIGHTS];
   int    m_train_cls[FX6_TST_TBPTT];
   double m_train_move[FX6_TST_TBPTT];
   double m_train_cost[FX6_TST_TBPTT];
   double m_train_w[FX6_TST_TBPTT];

   // Forward caches for TBPTT updates.
   double m_cache_embed[FX6_TST_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_local[FX6_TST_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_attn[FX6_TST_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_final[FX6_TST_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_group[FX6_TST_TBPTT][3];
   double m_cache_ctx[FX6_TST_TBPTT][FX6_TST_HEADS][FX6_TST_D_HEAD];

   double m_cache_logits[FX6_TST_TBPTT][FX6_TST_CLASS_COUNT];
   double m_cache_probs[FX6_TST_TBPTT][FX6_TST_CLASS_COUNT];
   double m_cache_mu[FX6_TST_TBPTT];
   double m_cache_logv[FX6_TST_TBPTT];
   double m_cache_q25[FX6_TST_TBPTT];
   double m_cache_q75[FX6_TST_TBPTT];
   double m_cache_mu_h[FX6_TST_TBPTT][FX6_TST_HORIZONS];

   // Attention caches used by gradient-correct updates.
   int    m_cache_tok_count[FX6_TST_TBPTT];
   double m_cache_tok[FX6_TST_TBPTT][FX6_TST_SEQ][FX6_AI_MLP_HIDDEN];
   double m_cache_qv[FX6_TST_TBPTT][FX6_TST_HEADS][FX6_TST_D_HEAD];
   double m_cache_kv[FX6_TST_TBPTT][FX6_TST_HEADS][FX6_TST_SEQ][FX6_TST_D_HEAD];
   double m_cache_vv[FX6_TST_TBPTT][FX6_TST_HEADS][FX6_TST_SEQ][FX6_TST_D_HEAD];
   double m_cache_aw[FX6_TST_TBPTT][FX6_TST_HEADS][FX6_TST_SEQ];

   void LayerNormAffine(double &v[], const double &g[], const double &b[]) const
   {
      double mean = 0.0;
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         mean += v[i];
      mean /= (double)FX6_AI_MLP_HIDDEN;

      double var = 0.0;
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
      {
         double d = v[i] - mean;
         var += d * d;
      }

      double inv = 1.0 / MathSqrt(var / (double)FX6_AI_MLP_HIDDEN + 1e-6);
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
      {
         double n = (v[i] - mean) * inv;
         v[i] = FX6_ClipSym(g[i] * n + b[i], 8.0);
      }
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FX6_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FX6_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FX6_Clamp(logits[2] - m, -30.0, 30.0));

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

   int SessionBucketNow(void) const
   {
      MqlDateTime md;
      TimeToStruct(ResolveContextTime(), md);
      int h = md.hour;
      if(h >= 6 && h <= 12) return 1;   // EU
      if(h >= 13 && h <= 20) return 2;  // US
      if(h >= 21 || h <= 2) return 0;   // Asia
      return 3;                         // transition
   }

   double ScheduledLR(const FX6AIHyperParams &hp) const
   {
      double base = FX6_Clamp(hp.lr, 0.00002, 0.25000);
      double st = (double)MathMax(m_step, 1);

      double warm = FX6_Clamp(st / 192.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0011 * MathMax(0.0, st - 192.0));

      double cyc = 3072.0;
      double ph = MathMod(st, cyc) / cyc;
      double cs = 0.5 * (1.0 + MathCos(3.141592653589793 * ph));
      double floor = 0.25 + 0.75 * cs;

      double lr = base * warm * invsqrt * floor;
      return FX6_Clamp(lr, 0.00001, 0.05000);
   }

   double AdamGroupLR(const int group_idx,
                      const double grad_mag,
                      const double base_lr)
   {
      int g = group_idx;
      if(g < 0) g = 0;
      if(g > 7) g = 7;

      const double b1 = 0.90;
      const double b2 = 0.999;
      const double eps = 1e-8;

      double gm = MathAbs(grad_mag);
      m_opt_m[g] = b1 * m_opt_m[g] + (1.0 - b1) * gm;
      m_opt_v[g] = b2 * m_opt_v[g] + (1.0 - b2) * gm * gm;

      double t = (double)MathMax(m_adam_t, 1);
      double mh = m_opt_m[g] / (1.0 - MathPow(b1, t));
      double vh = m_opt_v[g] / (1.0 - MathPow(b2, t));
      double scale = mh / (MathSqrt(vh) + eps);
      return FX6_Clamp(base_lr * (0.60 + 0.40 * scale), 0.000002, 0.100000);
   }

   double DropMask(const int step_salt,
                   const int layer,
                   const int channel,
                   const bool training) const
   {
      if(!training) return 1.0;
      if(m_drop_rate <= 1e-9) return 1.0;

      uint h = (uint)(m_step * 2654435761U);
      h ^= (uint)((step_salt + 5) * 2246822519U);
      h ^= (uint)((layer + 11) * 3266489917U);
      h ^= (uint)((channel + 17) * 668265263U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < m_drop_rate) return 0.0;
      return 1.0 / (1.0 - m_drop_rate);
   }

   double StochScale(const int step_salt,
                     const int layer,
                     const bool training) const
   {
      if(!training) return 1.0;
      if(m_stoch_rate <= 1e-9) return 1.0;

      uint h = (uint)((m_step + 13) * 40503U);
      h ^= (uint)((step_salt + 29) * 2654435761U);
      h ^= (uint)((layer + 37) * 2246822519U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < m_stoch_rate) return 0.0;
      return 1.0 / (1.0 - m_stoch_rate);
   }

   void ResetSessionCal(void)
   {
      for(int s=0; s<FX6_TST_SESSIONS; s++)
      {
         m_sess_cal_a[s] = 1.0;
         m_sess_cal_b[s] = 0.0;
         m_sess_cal_steps[s] = 0;
         m_sess_cal_ready[s] = false;
      }
   }

   void ResetReplay(void)
   {
      m_replay_len = 0;
      m_replay_ptr = 0;
      for(int i=0; i<FX6_TST_REPLAY; i++)
      {
         m_replay_cls[i] = FX6_TST_SKIP;
         m_replay_move[i] = 0.0;
         m_replay_cost[i] = 0.0;
         m_replay_w[i] = 1.0;
         m_replay_hard[i] = 0.0;
         for(int j=0; j<FX6_AI_WEIGHTS; j++)
            m_replay_x[i][j] = 0.0;
      }
   }

   void PushReplay(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double sample_w,
                   const double hardness)
   {
      int p = m_replay_ptr;
      if(p < 0 || p >= FX6_TST_REPLAY) p = 0;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_replay_x[p][i] = x[i];
      m_replay_cls[p] = cls;
      m_replay_move[p] = move_points;
      m_replay_cost[p] = cost_points;
      m_replay_w[p] = sample_w;
      m_replay_hard[p] = FX6_Clamp(hardness, 0.0, 10.0);

      m_replay_ptr++;
      if(m_replay_ptr >= FX6_TST_REPLAY) m_replay_ptr = 0;
      if(m_replay_len < FX6_TST_REPLAY) m_replay_len++;
   }

   int PickReplayIndex(const int prefer_cls) const
   {
      if(m_replay_len <= 0) return -1;

      int best = -1;
      double best_h = -1e100;
      for(int i=0; i<m_replay_len; i++)
      {
         if(prefer_cls >= 0 && m_replay_cls[i] != prefer_cls) continue;
         double h = m_replay_hard[i];
         if(h > best_h)
         {
            best_h = h;
            best = i;
         }
      }

      if(best >= 0) return best;

      // fallback: hardest overall
      best = 0;
      best_h = m_replay_hard[0];
      for(int i=1; i<m_replay_len; i++)
      {
         if(m_replay_hard[i] > best_h)
         {
            best_h = m_replay_hard[i];
            best = i;
         }
      }
      return best;
   }

   void AppendReplayToTrain(const int idx)
   {
      if(idx < 0 || idx >= m_replay_len) return;
      double xr[FX6_AI_WEIGHTS];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         xr[i] = m_replay_x[idx][i];
      AppendTrainSample(m_replay_cls[idx], xr, m_replay_move[idx], m_replay_cost[idx], m_replay_w[idx]);
   }

   void UpdateSessionCalibration(const int sess,
                                 const double p_raw,
                                 const int y,
                                 const double sample_w)
   {
      int s = sess;
      if(s < 0 || s >= FX6_TST_SESSIONS) s = FX6_TST_SESSIONS - 1;

      double pr = FX6_Clamp(p_raw, 0.001, 0.999);
      double z = FX6_Logit(pr);
      double py = FX6_Sigmoid(m_sess_cal_a[s] * z + m_sess_cal_b[s]);
      double e = ((double)y - py);

      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double lr = 0.012 * w;
      double reg = 0.0005;

      m_sess_cal_a[s] += lr * (e * z - reg * (m_sess_cal_a[s] - 1.0));
      m_sess_cal_b[s] += lr * e;

      m_sess_cal_a[s] = FX6_Clamp(m_sess_cal_a[s], 0.20, 5.00);
      m_sess_cal_b[s] = FX6_Clamp(m_sess_cal_b[s], -4.0, 4.0);

      m_sess_cal_steps[s]++;
      if(m_sess_cal_steps[s] >= 20) m_sess_cal_ready[s] = true;
   }

   double CalibrateSessionProb(const int sess,
                               const double p_raw) const
   {
      int s = sess;
      if(s < 0 || s >= FX6_TST_SESSIONS) s = FX6_TST_SESSIONS - 1;

      double p0 = CalibrateProb(p_raw);
      if(!m_sess_cal_ready[s]) return p0;

      double ps = FX6_Sigmoid(m_sess_cal_a[s] * FX6_Logit(FX6_Clamp(p0, 0.001, 0.999)) + m_sess_cal_b[s]);
      return FX6_Clamp(0.65 * p0 + 0.35 * ps, 0.001, 0.999);
   }

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FX6_TST_SEQ; t++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            m_seq_h[t][h] = 0.0;
      }
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FX6_TST_TBPTT; t++)
      {
         m_train_cls[t] = FX6_TST_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;

         m_cache_mu[t] = 0.0;
         m_cache_logv[t] = MathLog(1.0);
         m_cache_q25[t] = 0.0;
         m_cache_q75[t] = 0.0;
         m_cache_tok_count[t] = 0;
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            m_cache_mu_h[t][hz] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_train_x[t][i] = 0.0;

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[t][h] = 0.0;
            m_cache_local[t][h] = 0.0;
            m_cache_attn[t][h] = 0.0;
            m_cache_final[t][h] = 0.0;
         }

         for(int g=0; g<3; g++)
            m_cache_group[t][g] = 0.0;

         for(int hd=0; hd<FX6_TST_HEADS; hd++)
         {
            for(int d=0; d<FX6_TST_D_HEAD; d++)
            {
               m_cache_ctx[t][hd][d] = 0.0;
               m_cache_qv[t][hd][d] = 0.0;
            }
         }

         for(int tok=0; tok<FX6_TST_SEQ; tok++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               m_cache_tok[t][tok][h] = 0.0;
            for(int hd=0; hd<FX6_TST_HEADS; hd++)
            {
               m_cache_aw[t][hd][tok] = 0.0;
               for(int d=0; d<FX6_TST_D_HEAD; d++)
               {
                  m_cache_kv[t][hd][tok][d] = 0.0;
                  m_cache_vv[t][hd][tok][d] = 0.0;
               }
            }
         }

         for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = 0.0;
            m_cache_probs[t][c] = 1.0 / 3.0;
         }
      }
   }

   int SeqIndexBack(const int back) const
   {
      if(m_seq_len <= 0 || m_seq_ptr < 0) return 0;
      int idx = m_seq_ptr - back;
      while(idx < 0) idx += FX6_TST_SEQ;
      while(idx >= FX6_TST_SEQ) idx -= FX6_TST_SEQ;
      return idx;
   }

   int MapClass(const int y,
                const double &x[],
                const double move_points) const
   {
      if(y == FX6_TST_SELL || y == FX6_TST_BUY || y == FX6_TST_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FX6_TST_SKIP;

      if(y > 0) return FX6_TST_BUY;
      if(y == 0) return FX6_TST_SELL;
      return (move_points >= 0.0 ? FX6_TST_BUY : FX6_TST_SELL);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost,
                      const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double base = FX6_Clamp(sample_w, 0.25, 4.00);

      if(cls == FX6_TST_SKIP)
      {
         if(edge <= 0.0) return FX6_Clamp(base * 1.6, 0.25, 6.0);
         return FX6_Clamp(base * 0.75, 0.25, 6.0);
      }

      if(edge <= 0.0) return FX6_Clamp(base * 0.55, 0.25, 6.0);
      return FX6_Clamp(base * (1.0 + 0.06 * MathMin(edge, 20.0)), 0.25, 6.0);
   }

   double MoveWeight(const double move_points,
                     const double cost,
                     const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double ew = FX6_Clamp(0.5 + edge / denom, 0.25, 4.0);
      return FX6_Clamp(sample_w * ew, 0.25, 8.0);
   }

   void InitGroupMasks(void)
   {
      // feature index = i - 1 (x[0] is bias)
      for(int g=0; g<3; g++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_group_mask[g][i] = 0.0;
      }

      for(int g=0; g<3; g++)
         m_group_mask[g][0] = 1.0;

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
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

   void ResetInputNorm(void)
   {
      m_x_norm_ready = false;
      m_x_norm_steps = 0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 128 ? 0.05 : 0.015);
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
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

   void NormalizeInput(const double &x[], double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         if(!m_x_norm_ready)
         {
            xn[i] = FX6_ClipSym(x[i], 8.0);
            continue;
         }

         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FX6_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void ConfigureFromHP(const FX6AIHyperParams &hp)
   {
      int d = (int)MathRound(hp.tcn_layers);
      if(d < 2) d = 2;
      if(d > FX6_TST_MAX_STACK) d = FX6_TST_MAX_STACK;
      m_stack_depth = d;

      double k = FX6_Clamp(hp.tcn_kernel, 2.0, 7.0);
      m_drop_rate = FX6_Clamp(0.02 * k, 0.05, 0.22);
      m_stoch_rate = FX6_Clamp(0.01 * k, 0.03, 0.18);
      m_cfg_ready = true;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
      ResetReplay();
      ResetSessionCal();
      InitGroupMasks();
      m_step = 0;
      m_adam_t = 0;

      m_cfg_ready = false;
      m_stack_depth = 2;
      m_drop_rate = 0.10;
      m_stoch_rate = 0.08;
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      for(int g=0; g<3; g++)
      {
         m_v_gate_b[g] = (g == 0 ? 0.25 : 0.0);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double s = (double)((g + 1) * (i + 2));
            m_v_gate_w[g][i] = 0.03 * MathSin(0.71 * s);
         }
      }

      for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
      {
         m_b_embed[o] = 0.0;

         m_grn1_b1[o] = 0.0;
         m_grn1_b2[o] = 0.0;
         m_grn1_bg[o] = 0.0;

         m_grn2_b1[o] = 0.0;
         m_grn2_b2[o] = 0.0;
         m_grn2_bg[o] = 0.0;

         m_attn_bg[o] = 0.0;

         m_ff1_b[o] = 0.0;
         m_ff2_b[o] = 0.0;

         m_ln1_g[o] = 1.0;
         m_ln1_b[o] = 0.0;
         m_ln_attn_g[o] = 1.0;
         m_ln_attn_b[o] = 0.0;
         m_ln2_g[o] = 1.0;
         m_ln2_b[o] = 0.0;
         m_ln_out_g[o] = 1.0;
         m_ln_out_b[o] = 0.0;

         m_w_mu[o] = 0.04 * MathSin((double)(o + 1) * 0.93);
         m_w_logv[o] = 0.03 * MathCos((double)(o + 2) * 1.01);
         m_w_q25[o] = 0.03 * MathSin((double)(o + 3) * 1.07);
         m_w_q75[o] = 0.03 * MathCos((double)(o + 4) * 1.11);
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            m_w_mu_h[hz][o] = 0.03 * MathSin((double)((hz + 3) * (o + 1)) * 0.69);

         for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
            m_w_cls[c][o] = 0.03 * MathSin((double)((c + 2) * (o + 1)) * 0.87);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double s0 = (double)((o + 1) * (i + 3));
            double s1 = (double)((o + 2) * (i + 5));
            double s2 = (double)((o + 3) * (i + 7));
            m_w_obs[o][i] = 0.04 * MathSin(0.67 * s0);
            m_w_known[o][i] = 0.04 * MathCos(0.73 * s1);
            m_w_static[o][i] = 0.04 * MathSin(0.79 * s2);
         }

         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            double a = (double)((o + 1) * (i + 2));

            m_grn1_w1[o][i] = 0.05 * MathSin(0.61 * a);
            m_grn1_w2[o][i] = 0.05 * MathCos(0.67 * a);
            m_grn1_wg[o][i] = 0.04 * MathSin(0.71 * a);

            m_grn2_w1[o][i] = 0.05 * MathCos(0.73 * a);
            m_grn2_w2[o][i] = 0.05 * MathSin(0.77 * a);
            m_grn2_wg[o][i] = 0.04 * MathCos(0.81 * a);

            m_attn_wg[o][i] = 0.04 * MathSin(0.85 * a);

            m_ff1_w[o][i] = 0.05 * MathCos(0.57 * a);
            m_ff2_w[o][i] = 0.05 * MathSin(0.59 * a);
         }

         for(int s=0; s<FX6_TST_SESSIONS; s++)
            m_session_bias[s][o] = 0.03 * MathSin((double)((s + 2) * (o + 1)) * 0.57);
      }

      for(int hd=0; hd<FX6_TST_HEADS; hd++)
      {
         m_rel_decay[hd] = 0.02 * (double)(hd + 1);
         for(int d=0; d<FX6_TST_D_HEAD; d++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double t = (double)((hd + 1) * (d + 2) * (h + 3));
               m_wq[hd][d][h] = 0.04 * MathSin(0.69 * t);
               m_wk[hd][d][h] = 0.04 * MathCos(0.73 * t);
               m_wv[hd][d][h] = 0.04 * MathSin(0.79 * t);
               m_wo[hd][h][d] = 0.04 * MathCos(0.83 * t);
            }
         }
      }

      for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;
      for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
         m_b_mu_h[hz] = 0.0;

      m_initialized = true;
   }

   void ApplyGRN(const double &in[],
                 const double &w1[][FX6_AI_MLP_HIDDEN],
                 const double &w2[][FX6_AI_MLP_HIDDEN],
                 const double &wg[][FX6_AI_MLP_HIDDEN],
                 const double &b1[],
                 const double &b2[],
                 const double &bg[],
                 const double &ln_g[],
                 const double &ln_b[],
                 double &out[]) const
   {
      double h1[FX6_AI_MLP_HIDDEN];
      for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
      {
         double z1 = b1[o];
         double zg = bg[o];
         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
         {
            z1 += w1[o][i] * in[i];
            zg += wg[o][i] * in[i];
         }
         h1[o] = FX6_Tanh(z1);

         double z2 = b2[o];
         for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            z2 += w2[o][i] * h1[i];

         double g = FX6_Sigmoid(zg);
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
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            s += m_v_gate_w[g][i] * (xn[i] * m_group_mask[g][i]);
         gl[g] = s;
         if(s > maxs) maxs = s;
      }

      double den = 0.0;
      for(int g=0; g<3; g++)
      {
         group_gate[g] = MathExp(FX6_ClipSym(gl[g] - maxs, 30.0));
         den += group_gate[g];
      }
      if(den <= 0.0) den = 1.0;
      for(int g=0; g<3; g++)
         group_gate[g] /= den;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double so = 0.0;
         double sk = 0.0;
         double ss = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            so += m_w_obs[h][i] * xn[i] * m_group_mask[0][i];
            sk += m_w_known[h][i] * xn[i] * m_group_mask[1][i];
            ss += m_w_static[h][i] * xn[i] * m_group_mask[2][i];
         }

         double z = m_b_embed[h] + group_gate[0] * so + group_gate[1] * sk + group_gate[2] * ss;
         embed[h] = FX6_Tanh(z);
      }
   }

   void BuildTokens(const double &current[],
                    double &tokens[][FX6_AI_MLP_HIDDEN],
                    int &count) const
   {
      int keep = m_seq_len;
      if(keep > FX6_TST_SEQ - 1) keep = FX6_TST_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = SeqIndexBack(back);
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            tokens[i][h] = m_seq_h[idx][h];
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         tokens[keep][h] = current[h];

      // Relative-time positional encoding (oldest->newest).
      for(int i=0; i<=keep; i++)
      {
         int rel = keep - i;
         double r = (double)rel;
         double e0 = 0.10 * MathSin(0.35 * r);
         double e1 = 0.10 * MathCos(0.21 * r);
         double e2 = 0.06 * MathSin(0.13 * r);
         double e3 = 0.06 * MathCos(0.09 * r);
         tokens[i][0] = FX6_ClipSym(tokens[i][0] + e0, 8.0);
         if(FX6_AI_MLP_HIDDEN > 1) tokens[i][1] = FX6_ClipSym(tokens[i][1] + e1, 8.0);
         if(FX6_AI_MLP_HIDDEN > 2) tokens[i][2] = FX6_ClipSym(tokens[i][2] + e2, 8.0);
         if(FX6_AI_MLP_HIDDEN > 3) tokens[i][3] = FX6_ClipSym(tokens[i][3] + e3, 8.0);
      }

      count = keep + 1;
   }

   void MultiHeadAttention(const double &query_src[],
                           const double &tokens[][FX6_AI_MLP_HIDDEN],
                           const int token_count,
                           double &attn_out[],
                           double &head_ctx[][FX6_TST_D_HEAD],
                           const bool cache_train,
                           const int cache_slot)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) attn_out[h] = 0.0;
      for(int hd=0; hd<FX6_TST_HEADS; hd++)
         for(int d=0; d<FX6_TST_D_HEAD; d++)
            head_ctx[hd][d] = 0.0;

      if(token_count <= 0) return;

      double inv_scale = 1.0 / MathSqrt((double)FX6_TST_D_HEAD);
      double q[FX6_TST_D_HEAD];
      double scores[FX6_TST_SEQ];
      double v_cache[FX6_TST_SEQ][FX6_TST_D_HEAD];

      for(int hd=0; hd<FX6_TST_HEADS; hd++)
      {
         for(int d=0; d<FX6_TST_D_HEAD; d++)
         {
            double s = 0.0;
            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
               s += m_wq[hd][d][j] * query_src[j];
            q[d] = s;
         }

         double max_sc = -1e100;
         for(int t=0; t<token_count; t++)
         {
            int rel = (token_count - 1 - t);
            double sc = 0.0;
            for(int d=0; d<FX6_TST_D_HEAD; d++)
            {
               double kv = 0.0;
               double vv = 0.0;
               for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
               {
                  kv += m_wk[hd][d][j] * tokens[t][j];
                  vv += m_wv[hd][d][j] * tokens[t][j];
               }
               sc += q[d] * kv;
               v_cache[t][d] = vv;
               if(cache_train && cache_slot >= 0 && cache_slot < FX6_TST_TBPTT)
               {
                  m_cache_kv[cache_slot][hd][t][d] = kv;
                  m_cache_vv[cache_slot][hd][t][d] = vv;
               }
            }
            sc -= m_rel_decay[hd] * (double)rel;
            sc *= inv_scale;
            scores[t] = sc;
            if(sc > max_sc) max_sc = sc;
         }

         double den = 0.0;
         for(int t=0; t<token_count; t++)
         {
            scores[t] = MathExp(FX6_ClipSym(scores[t] - max_sc, 30.0));
            den += scores[t];
         }
         if(den <= 0.0) den = 1.0;

         for(int d=0; d<FX6_TST_D_HEAD; d++)
         {
            double c = 0.0;
            for(int t=0; t<token_count; t++)
               c += (scores[t] / den) * v_cache[t][d];
            head_ctx[hd][d] = c;
         }

         if(cache_train && cache_slot >= 0 && cache_slot < FX6_TST_TBPTT)
         {
            for(int d=0; d<FX6_TST_D_HEAD; d++)
               m_cache_qv[cache_slot][hd][d] = q[d];
            for(int t=0; t<token_count; t++)
               m_cache_aw[cache_slot][hd][t] = scores[t] / den;
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            double s = 0.0;
            for(int d=0; d<FX6_TST_D_HEAD; d++)
               s += m_wo[hd][j][d] * head_ctx[hd][d];
            attn_out[j] += s;
         }
      }
   }

   void ForwardStep(const double &x[],
                    const bool commit,
                    const bool training,
                    const int cache_slot,
                    double &state_embed[],
                    double &state_local[],
                    double &state_attn[],
                    double &state_final[],
                    double &group_gate[],
                    double &head_ctx[][FX6_TST_D_HEAD])
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, xn);

      BuildVariableSelection(xn, state_embed, group_gate);
      int sess = SessionBucketNow();
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         state_embed[h] = FX6_ClipSym(state_embed[h] + 0.08 * m_session_bias[sess][h], 8.0);

      double layer_in[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) layer_in[h] = state_embed[h];

      for(int layer=0; layer<m_stack_depth; layer++)
      {
         ApplyGRN(layer_in,
                  m_grn1_w1,
                  m_grn1_w2,
                  m_grn1_wg,
                  m_grn1_b1,
                  m_grn1_b2,
                  m_grn1_bg,
                  m_ln1_g,
                  m_ln1_b,
                  state_local);

         double tokens[FX6_TST_SEQ][FX6_AI_MLP_HIDDEN];
         int token_count = 0;
         BuildTokens(state_local, tokens, token_count);
         if(training && cache_slot >= 0 && cache_slot < FX6_TST_TBPTT)
         {
            m_cache_tok_count[cache_slot] = token_count;
            for(int t=0; t<token_count; t++)
            {
               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
                  m_cache_tok[cache_slot][t][h] = tokens[t][h];
            }
         }

         double attn_mix[FX6_AI_MLP_HIDDEN];
         MultiHeadAttention(state_local,
                            tokens,
                            token_count,
                            attn_mix,
                            head_ctx,
                            (training && cache_slot >= 0),
                            cache_slot);

         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double zg = m_attn_bg[o];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
               zg += m_attn_wg[o][i] * state_local[i];

            double g = FX6_Sigmoid(zg);
            state_attn[o] = g * attn_mix[o] + (1.0 - g) * state_local[o];
         }
         LayerNormAffine(state_attn, m_ln_attn_g, m_ln_attn_b);

         ApplyGRN(state_attn,
                  m_grn2_w1,
                  m_grn2_w2,
                  m_grn2_wg,
                  m_grn2_b1,
                  m_grn2_b2,
                  m_grn2_bg,
                  m_ln2_g,
                  m_ln2_b,
                  state_final);

         double ff1[FX6_AI_MLP_HIDDEN];
         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double z1 = m_ff1_b[o];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
               z1 += m_ff1_w[o][i] * state_final[i];
            ff1[o] = FX6_Tanh(z1) * DropMask(cache_slot + 1, layer, o, training);
         }

         double stoch = StochScale(cache_slot + 1, layer, training);
         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double z2 = m_ff2_b[o];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
               z2 += m_ff2_w[o][i] * ff1[i];
            state_final[o] = state_final[o] + stoch * (0.25 * FX6_ClipSym(z2, 8.0));
         }
         LayerNormAffine(state_final, m_ln_out_g, m_ln_out_b);

         // Residual bridge for configurable deep stack.
         if(layer < m_stack_depth - 1)
         {
            double blend = 0.70 + 0.08 * (double)layer;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               layer_in[h] = FX6_ClipSym(blend * state_final[h] + (1.0 - blend) * layer_in[h], 8.0);
         }
      }

      if(commit)
      {
         m_seq_ptr++;
         if(m_seq_ptr >= FX6_TST_SEQ) m_seq_ptr = 0;
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            m_seq_h[m_seq_ptr][h] = state_final[h];
         if(m_seq_len < FX6_TST_SEQ) m_seq_len++;
      }
   }

   void ComputeHeads(const double &state_final[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75,
                     double &mu_h0,
                     double &mu_h1,
                     double &mu_h2) const
   {
      for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            z += m_w_cls[c][h] * state_final[h];
         logits[c] = FX6_ClipSym(z, 20.0);
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      double r25 = m_b_q25;
      double r75 = m_b_q75;
      double mh0 = m_b_mu_h[0];
      double mh1 = m_b_mu_h[1];
      double mh2 = m_b_mu_h[2];

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         mu += m_w_mu[h] * state_final[h];
         logv += m_w_logv[h] * state_final[h];
         r25 += m_w_q25[h] * state_final[h];
         r75 += m_w_q75[h] * state_final[h];
         mh0 += m_w_mu_h[0][h] * state_final[h];
         mh1 += m_w_mu_h[1][h] * state_final[h];
         mh2 += m_w_mu_h[2][h] * state_final[h];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(r25 <= r75)
      {
         q25 = r25;
         q75 = r75;
      }
      else
      {
         q25 = r75;
         q75 = r25;
      }

      // Keep horizons monotonic in magnitude.
      if(MathAbs(mh1) < MathAbs(mh0))
         mh1 = FX6_Sign(mh1 == 0.0 ? mh0 : mh1) * MathAbs(mh0);
      if(MathAbs(mh2) < MathAbs(mh1))
         mh2 = FX6_Sign(mh2 == 0.0 ? mh1 : mh2) * MathAbs(mh1);

      mu_h0 = mh0;
      mu_h1 = mh1;
      mu_h2 = mh2;
   }

   void AppendTrainSample(const int cls,
                          const double &x[],
                          const double move_points,
                          const double cost_points,
                          const double sample_w)
   {
      if(m_train_len < FX6_TST_TBPTT)
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

      for(int t=1; t<FX6_TST_TBPTT; t++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_train_x[FX6_TST_TBPTT - 1][i] = x[i];
      m_train_cls[FX6_TST_TBPTT - 1] = cls;
      m_train_move[FX6_TST_TBPTT - 1] = move_points;
      m_train_cost[FX6_TST_TBPTT - 1] = cost_points;
      m_train_w[FX6_TST_TBPTT - 1] = sample_w;
   }

   void TrainTBPTT(const FX6AIHyperParams &hp)
   {
      if(m_train_len <= 0) return;

      int n = m_train_len;
      if(n > FX6_TST_TBPTT) n = FX6_TST_TBPTT;
      if(n <= 0) return;

      double lr_base = ScheduledLR(hp);
      double l2 = FX6_Clamp(hp.l2, 0.0000, 1.0000);
      double wd = FX6_Clamp(0.25 * l2, 0.0, 0.10);

      int cls_count[FX6_TST_CLASS_COUNT];
      for(int c=0; c<FX6_TST_CLASS_COUNT; c++) cls_count[c] = 0;
      for(int t=0; t<n; t++)
      {
         int cc = m_train_cls[t];
         if(cc < 0 || cc >= FX6_TST_CLASS_COUNT) cc = FX6_TST_SKIP;
         cls_count[cc]++;
      }

      // Save live state then isolate sequence for TBPTT to avoid hidden-state bleed.
      int saved_ptr = m_seq_ptr;
      int saved_len = m_seq_len;
      double saved_seq[FX6_TST_SEQ][FX6_AI_MLP_HIDDEN];
      for(int t=0; t<FX6_TST_SEQ; t++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            saved_seq[t][h] = m_seq_h[t][h];
      }

      ResetSequence();

      // Forward pass over truncated sequence.
      for(int t=0; t<n; t++)
      {
         double xrow[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xrow[i] = m_train_x[t][i];

         double emb[FX6_AI_MLP_HIDDEN];
         double loc[FX6_AI_MLP_HIDDEN];
         double att[FX6_AI_MLP_HIDDEN];
         double fin[FX6_AI_MLP_HIDDEN];
         double grp[3];
         double ctx[FX6_TST_HEADS][FX6_TST_D_HEAD];

         ForwardStep(xrow, true, true, t, emb, loc, att, fin, grp, ctx);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[t][h] = emb[h];
            m_cache_local[t][h] = loc[h];
            m_cache_attn[t][h] = att[h];
            m_cache_final[t][h] = fin[h];
         }

         for(int g=0; g<3; g++)
            m_cache_group[t][g] = grp[g];

         for(int hd=0; hd<FX6_TST_HEADS; hd++)
            for(int d=0; d<FX6_TST_D_HEAD; d++)
               m_cache_ctx[t][hd][d] = ctx[hd][d];

         double logits[FX6_TST_CLASS_COUNT];
         double probs[FX6_TST_CLASS_COUNT];
         double mh0 = 0.0, mh1 = 0.0, mh2 = 0.0;
         ComputeHeads(fin,
                      logits,
                      probs,
                      m_cache_mu[t],
                      m_cache_logv[t],
                      m_cache_q25[t],
                      m_cache_q75[t],
                      mh0,
                      mh1,
                      mh2);
         m_cache_mu_h[t][0] = mh0;
         m_cache_mu_h[t][1] = mh1;
         m_cache_mu_h[t][2] = mh2;
         for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = logits[c];
            m_cache_probs[t][c] = probs[c];
         }
      }

      // Reverse-time updates with truncated temporal error flow.
      double next_delta[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) next_delta[h] = 0.0;

      for(int t=n - 1; t>=0; t--)
      {
         m_step++;
         m_adam_t++;
         double lr = lr_base;

         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FX6_Clamp(m_train_w[t], 0.25, 4.00);

         double edge = MathAbs(mv) - MathMax(cost, 0.0);
         double ev_scale = FX6_Clamp(0.60 + edge / MathMax(cost, 1.0), 0.25, 4.00);

         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);
         w_cls = FX6_Clamp(w_cls * ev_scale, 0.25, 8.00);
         w_mv = FX6_Clamp(w_mv * ev_scale, 0.25, 10.00);

         int cc = cls;
         if(cc < 0 || cc >= FX6_TST_CLASS_COUNT) cc = FX6_TST_SKIP;
         double invf = (cls_count[cc] > 0 ? ((double)n / ((double)FX6_TST_CLASS_COUNT * (double)cls_count[cc])) : 1.0);
         w_cls = FX6_Clamp(w_cls * FX6_Clamp(invf, 0.50, 2.50), 0.25, 10.0);

         double g_logits[FX6_TST_CLASS_COUNT];
         for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
         {
            double target = (c == cls ? 1.0 : 0.0);
            g_logits[c] = FX6_ClipSym((m_cache_probs[t][c] - target) * w_cls, 4.0);
         }

         double target_move = mv;
         double mu = m_cache_mu[t];
         double logv = m_cache_logv[t];
         double q25 = m_cache_q25[t];
         double q75 = m_cache_q75[t];
         double target_h[FX6_TST_HORIZONS];
         target_h[0] = target_move;
         target_h[1] = FX6_ClipSym(1.40 * target_move, 80.0);
         target_h[2] = FX6_ClipSym(1.90 * target_move, 120.0);
         double g_mu_h[FX6_TST_HORIZONS];
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            g_mu_h[hz] = 0.0;

         double sigma2 = MathExp(logv);
         sigma2 = FX6_Clamp(sigma2, 0.05, 100.0);

         double err_mu = FX6_ClipSym(mu - target_move, 30.0);
         double g_mu = FX6_ClipSym((err_mu / sigma2) * w_mv, 5.0);

         double ratio = (err_mu * err_mu) / sigma2;
         double g_logv = FX6_ClipSym(0.5 * (1.0 - ratio) * w_mv, 4.0);

         double e25 = target_move - q25;
         double e75 = target_move - q75;
         double g_q25 = FX6_ClipSym(((e25 < 0.0 ? -0.75 : 0.25)) * w_mv, 3.0);
         double g_q75 = FX6_ClipSym(((e75 < 0.0 ? -0.25 : 0.75)) * w_mv, 3.0);

         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
         {
            double err_h = FX6_ClipSym(m_cache_mu_h[t][hz] - target_h[hz], 30.0);
            double hw = (hz == 0 ? 0.60 : (hz == 1 ? 0.28 : 0.12));
            g_mu_h[hz] = FX6_ClipSym(hw * w_mv * err_h / MathMax(MathSqrt(sigma2), 0.25), 4.0);
         }

         double delta[FX6_AI_MLP_HIDDEN];
         double gn = 0.0;
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double d = 0.35 * next_delta[h];
            for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
               d += g_logits[c] * m_w_cls[c][h];
            d += g_mu * m_w_mu[h];
            d += g_logv * m_w_logv[h];
            d += g_q25 * m_w_q25[h];
            d += g_q75 * m_w_q75[h];
            for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
               d += g_mu_h[hz] * m_w_mu_h[hz][h];
            delta[h] = d;
            gn += d * d;
         }

         for(int c=0; c<FX6_TST_CLASS_COUNT; c++) gn += g_logits[c] * g_logits[c];
         gn += g_mu * g_mu + g_logv * g_logv + g_q25 * g_q25 + g_q75 * g_q75;
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++) gn += g_mu_h[hz] * g_mu_h[hz];

         gn = MathSqrt(gn);
         if(gn > 5.0)
         {
            double sc = 5.0 / gn;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               delta[h] *= sc;
            for(int c=0; c<FX6_TST_CLASS_COUNT; c++) g_logits[c] *= sc;
            g_mu *= sc;
            g_logv *= sc;
            g_q25 *= sc;
            g_q75 *= sc;
            for(int hz=0; hz<FX6_TST_HORIZONS; hz++) g_mu_h[hz] *= sc;
         }

         // Head updates.
         double gmag_head = MathAbs(g_mu) + MathAbs(g_logv) + MathAbs(g_q25) + MathAbs(g_q75);
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++) gmag_head += MathAbs(g_mu_h[hz]);
         for(int c=0; c<FX6_TST_CLASS_COUNT; c++) gmag_head += MathAbs(g_logits[c]);
         double lr_head = AdamGroupLR(0, gmag_head, lr);

         for(int c=0; c<FX6_TST_CLASS_COUNT; c++)
         {
            m_b_cls[c] -= lr_head * g_logits[c];
            m_b_cls[c] = FX6_ClipSym(m_b_cls[c], 8.0);
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double g = g_logits[c] * m_cache_final[t][h];
               m_w_cls[c][h] -= lr_head * FX6_ClipSym(g, 6.0);
               m_w_cls[c][h] -= lr_head * wd * m_w_cls[c][h];
            }
         }

         m_b_mu -= lr_head * g_mu;
         m_b_logv -= lr_head * g_logv;
         m_b_q25 -= lr_head * g_q25;
         m_b_q75 -= lr_head * g_q75;
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            m_b_mu_h[hz] -= lr_head * g_mu_h[hz];

         m_b_mu = FX6_ClipSym(m_b_mu, 20.0);
         m_b_logv = FX6_Clamp(m_b_logv, -4.0, 4.0);
         m_b_q25 = FX6_ClipSym(m_b_q25, 20.0);
         m_b_q75 = FX6_ClipSym(m_b_q75, 20.0);
         for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            m_b_mu_h[hz] = FX6_ClipSym(m_b_mu_h[hz], 30.0);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_w_mu[h] -= lr_head * FX6_ClipSym(g_mu * m_cache_final[t][h], 6.0);
            m_w_logv[h] -= lr_head * FX6_ClipSym(g_logv * m_cache_final[t][h], 6.0);
            m_w_q25[h] -= lr_head * FX6_ClipSym(g_q25 * m_cache_final[t][h], 6.0);
            m_w_q75[h] -= lr_head * FX6_ClipSym(g_q75 * m_cache_final[t][h], 6.0);
            m_w_mu[h] -= lr_head * wd * m_w_mu[h];
            m_w_logv[h] -= lr_head * wd * m_w_logv[h];
            m_w_q25[h] -= lr_head * wd * m_w_q25[h];
            m_w_q75[h] -= lr_head * wd * m_w_q75[h];
            for(int hz=0; hz<FX6_TST_HORIZONS; hz++)
            {
               m_w_mu_h[hz][h] -= lr_head * FX6_ClipSym(g_mu_h[hz] * m_cache_final[t][h], 5.0);
               m_w_mu_h[hz][h] -= lr_head * wd * m_w_mu_h[hz][h];
            }
         }

         // FFN + normalization affine updates.
         double lr_ffn = AdamGroupLR(1, gn, lr);
         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_ff2_b[o] -= lr_ffn * 0.08 * d;
            m_ff1_b[o] -= lr_ffn * 0.04 * d;

            m_ln_out_b[o] -= lr_ffn * 0.01 * d;
            m_ln_out_g[o] -= lr_ffn * 0.01 * d * m_cache_final[t][o];
            m_ln2_b[o] -= lr_ffn * 0.008 * d;
            m_ln2_g[o] -= lr_ffn * 0.008 * d * m_cache_attn[t][o];
            m_ln_attn_b[o] -= lr_ffn * 0.008 * d;
            m_ln_attn_g[o] -= lr_ffn * 0.008 * d * m_cache_attn[t][o];
            m_ln1_b[o] -= lr_ffn * 0.006 * d;
            m_ln1_g[o] -= lr_ffn * 0.006 * d * m_cache_local[t][o];
            m_ln_out_g[o] -= lr_ffn * wd * m_ln_out_g[o];
            m_ln2_g[o] -= lr_ffn * wd * m_ln2_g[o];
            m_ln_attn_g[o] -= lr_ffn * wd * m_ln_attn_g[o];
            m_ln1_g[o] -= lr_ffn * wd * m_ln1_g[o];

            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               m_ff2_w[o][i] -= lr_ffn * (0.08 * FX6_ClipSym(d * m_cache_final[t][i], 4.0));
               m_ff1_w[o][i] -= lr_ffn * (0.04 * FX6_ClipSym(d * m_cache_local[t][i], 4.0));
               m_ff2_w[o][i] -= lr_ffn * wd * m_ff2_w[o][i];
               m_ff1_w[o][i] -= lr_ffn * wd * m_ff1_w[o][i];
            }
         }

         // GRN2 + attention gate.
         double lr_core = AdamGroupLR(2, gn, lr);
         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_grn2_b2[o] -= lr_core * 0.18 * d;
            m_grn2_b1[o] -= lr_core * 0.10 * d;
            m_grn2_bg[o] -= lr_core * 0.06 * d;
            m_attn_bg[o] -= lr_core * 0.05 * d;

            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               m_grn2_w2[o][i] -= lr_core * (0.18 * FX6_ClipSym(d * m_cache_attn[t][i], 4.0));
               m_grn2_w1[o][i] -= lr_core * (0.10 * FX6_ClipSym(d * m_cache_local[t][i], 4.0));
               m_grn2_wg[o][i] -= lr_core * (0.06 * FX6_ClipSym(d * m_cache_attn[t][i], 4.0));
               m_attn_wg[o][i] -= lr_core * (0.05 * FX6_ClipSym(d * m_cache_local[t][i], 4.0));
               m_grn2_w2[o][i] -= lr_core * wd * m_grn2_w2[o][i];
               m_grn2_w1[o][i] -= lr_core * wd * m_grn2_w1[o][i];
               m_grn2_wg[o][i] -= lr_core * wd * m_grn2_wg[o][i];
               m_attn_wg[o][i] -= lr_core * wd * m_attn_wg[o][i];
            }
         }

         // Attention projections.
         double head_delta[FX6_TST_HEADS][FX6_TST_D_HEAD];
         int token_count = m_cache_tok_count[t];
         if(token_count < 1) token_count = 1;
         if(token_count > FX6_TST_SEQ) token_count = FX6_TST_SEQ;
         double inv_scale = 1.0 / MathSqrt((double)FX6_TST_D_HEAD);
         double lr_attn = AdamGroupLR(3, gn, lr);

         for(int hd=0; hd<FX6_TST_HEADS; hd++)
         {
            for(int d=0; d<FX6_TST_D_HEAD; d++)
            {
               double s = 0.0;
               for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
                  s += delta[o] * m_wo[hd][o][d];
               head_delta[hd][d] = FX6_ClipSym(s, 3.0);
            }
         }

         for(int hd=0; hd<FX6_TST_HEADS; hd++)
         {
            double ds[FX6_TST_SEQ];
            double dot_ctx = 0.0;
            for(int d=0; d<FX6_TST_D_HEAD; d++)
               dot_ctx += head_delta[hd][d] * m_cache_ctx[t][hd][d];

            for(int tok=0; tok<token_count; tok++)
            {
               double dot_v = 0.0;
               for(int d=0; d<FX6_TST_D_HEAD; d++)
                  dot_v += head_delta[hd][d] * m_cache_vv[t][hd][tok][d];
               ds[tok] = m_cache_aw[t][hd][tok] * (dot_v - dot_ctx);
            }

            double rel_grad = 0.0;
            for(int d=0; d<FX6_TST_D_HEAD; d++)
            {
               double dh = head_delta[hd][d];
               double dq = 0.0;
               for(int tok=0; tok<token_count; tok++)
                  dq += ds[tok] * m_cache_kv[t][hd][tok][d] * inv_scale;

               for(int tok=0; tok<token_count; tok++)
               {
                  int rel = token_count - 1 - tok;
                  double dk = ds[tok] * m_cache_qv[t][hd][d] * inv_scale;
                  rel_grad += ds[tok] * (-(double)rel) * inv_scale;

                  for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
                  {
                     double tokh = m_cache_tok[t][tok][h];
                     m_wk[hd][d][h] -= lr_attn * FX6_ClipSym(dk * tokh, 3.0);
                     m_wv[hd][d][h] -= lr_attn * FX6_ClipSym((m_cache_aw[t][hd][tok] * dh) * tokh, 3.0);
                     m_wk[hd][d][h] -= lr_attn * wd * m_wk[hd][d][h];
                     m_wv[hd][d][h] -= lr_attn * wd * m_wv[hd][d][h];
                  }
               }

               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               {
                  m_wo[hd][h][d] -= lr_attn * 0.08 * FX6_ClipSym(delta[h] * m_cache_ctx[t][hd][d], 4.0);
                  m_wq[hd][d][h] -= lr_attn * 0.04 * FX6_ClipSym(dq * m_cache_local[t][h], 4.0);
                  m_wo[hd][h][d] -= lr_attn * wd * m_wo[hd][h][d];
                  m_wq[hd][d][h] -= lr_attn * wd * m_wq[hd][d][h];
               }
            }

            m_rel_decay[hd] -= lr_attn * FX6_ClipSym(rel_grad, 3.0);
            m_rel_decay[hd] = FX6_Clamp(m_rel_decay[hd], -0.5, 0.5);
         }

         // GRN1 + variable-selection updates.
         double xn[FX6_AI_WEIGHTS];
         double xrow2[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xrow2[i] = m_train_x[t][i];
         NormalizeInput(xrow2, xn);

         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_grn1_b2[o] -= lr_core * 0.08 * d;
            m_grn1_b1[o] -= lr_core * 0.05 * d;
            m_grn1_bg[o] -= lr_core * 0.03 * d;

            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               m_grn1_w2[o][i] -= lr_core * (0.08 * FX6_ClipSym(d * m_cache_local[t][i], 4.0));
               m_grn1_w1[o][i] -= lr_core * (0.05 * FX6_ClipSym(d * m_cache_embed[t][i], 4.0));
               m_grn1_wg[o][i] -= lr_core * (0.03 * FX6_ClipSym(d * m_cache_local[t][i], 4.0));
               m_grn1_w2[o][i] -= lr_core * wd * m_grn1_w2[o][i];
               m_grn1_w1[o][i] -= lr_core * wd * m_grn1_w1[o][i];
               m_grn1_wg[o][i] -= lr_core * wd * m_grn1_wg[o][i];
            }
         }

         for(int g=0; g<3; g++)
         {
            double gs = 0.0;
            int hs = (g + t) % FX6_AI_MLP_HIDDEN;
            gs = delta[hs] * m_cache_local[t][hs];
            m_v_gate_b[g] -= lr_core * 0.004 * FX6_ClipSym(gs, 3.0);

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               double xg = xn[i] * m_group_mask[g][i];
               m_v_gate_w[g][i] -= lr_core * (0.002 * FX6_ClipSym(gs * xg, 3.0));
               m_v_gate_w[g][i] -= lr_core * wd * m_v_gate_w[g][i];
            }
         }

         for(int o=0; o<FX6_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_b_embed[o] -= lr_core * 0.02 * d;

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               double xi = xn[i];
               double a0 = m_cache_group[t][0] * xi * m_group_mask[0][i];
               double a1 = m_cache_group[t][1] * xi * m_group_mask[1][i];
               double a2 = m_cache_group[t][2] * xi * m_group_mask[2][i];

               m_w_obs[o][i] -= lr_core * (0.02 * FX6_ClipSym(d * a0, 3.0));
               m_w_known[o][i] -= lr_core * (0.02 * FX6_ClipSym(d * a1, 3.0));
               m_w_static[o][i] -= lr_core * (0.02 * FX6_ClipSym(d * a2, 3.0));
               m_w_obs[o][i] -= lr_core * wd * m_w_obs[o][i];
               m_w_known[o][i] -= lr_core * wd * m_w_known[o][i];
               m_w_static[o][i] -= lr_core * wd * m_w_static[o][i];
            }
         }

         // Plugin-level calibration + move-head support.
         double den = m_cache_probs[t][FX6_TST_BUY] + m_cache_probs[t][FX6_TST_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir_raw = m_cache_probs[t][FX6_TST_BUY] / den;
         int sess = SessionBucketNow();
         if(cls == FX6_TST_BUY)
         {
            UpdateCalibration(p_dir_raw, 1, w_cls);
            UpdateSessionCalibration(sess, p_dir_raw, 1, w_cls);
         }
         else if(cls == FX6_TST_SELL)
         {
            UpdateCalibration(p_dir_raw, 0, w_cls);
            UpdateSessionCalibration(sess, p_dir_raw, 0, w_cls);
         }
         else
         {
            UpdateCalibration(p_dir_raw, (mv >= 0.0 ? 1 : 0), 0.25 * w_cls);
            UpdateSessionCalibration(sess, p_dir_raw, (mv >= 0.0 ? 1 : 0), 0.25 * w_cls);
         }

         FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         UpdateMoveHead(xrow2, mv, hp, sw);

         // Regime/session embedding adaptation.
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_session_bias[sess][h] -= lr_core * 0.005 * FX6_ClipSym(delta[h], 3.0);
            m_session_bias[sess][h] = FX6_ClipSym(m_session_bias[sess][h], 4.0);
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            next_delta[h] = delta[h];
      }

      // Restore live sequence state.
      m_seq_ptr = saved_ptr;
      m_seq_len = saved_len;
      for(int t=0; t<FX6_TST_SEQ; t++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            m_seq_h[t][h] = saved_seq[t][h];
      }
   }

public:
   CFX6AITST(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_TST; }
   virtual string AIName(void) const { return "tst"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_cfg_ready = false;
      m_step = 0;
      m_adam_t = 0;
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
      ResetReplay();
      ResetSessionCal();
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      if(!m_cfg_ready) ConfigureFromHP(hp);
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
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
      ConfigureFromHP(hp);
      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      // Controlled reset policy to avoid state bleed in non-stationary jumps.
      if((m_step % 512) == 0)
         ResetSequence();
      if(MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
         ResetSequence();

      UpdateInputStats(x);

      int cls = MapClass(y, x, move_points);
      double sw = MoveSampleWeight(x, move_points);
      sw = FX6_Clamp(sw, 0.25, 4.00);
      double cost = InputCostProxyPoints(x);

      // Hard-example replay and class-balance sampler.
      double p_before = PredictProb(x, h);
      double y01 = (cls == FX6_TST_BUY ? 1.0 : (cls == FX6_TST_SELL ? 0.0 : 0.5));
      double hardness = MathAbs(y01 - p_before) + 0.20 * (cls == FX6_TST_SKIP ? 1.0 : 0.0);
      hardness += 0.03 * MathAbs(MathAbs(move_points) - cost);
      PushReplay(cls, x, move_points, cost, sw, hardness);

      AppendTrainSample(cls, x, move_points, cost, sw);

      int local_cnt[FX6_TST_CLASS_COUNT];
      for(int c=0; c<FX6_TST_CLASS_COUNT; c++) local_cnt[c] = 0;
      for(int i=0; i<m_train_len; i++)
      {
         int cc = m_train_cls[i];
         if(cc < 0 || cc >= FX6_TST_CLASS_COUNT) cc = FX6_TST_SKIP;
         local_cnt[cc]++;
      }
      int minority = 0;
      for(int c=1; c<FX6_TST_CLASS_COUNT; c++)
      {
         if(local_cnt[c] < local_cnt[minority]) minority = c;
      }

      int ridx = PickReplayIndex(minority);
      if(ridx >= 0)
         AppendReplayToTrain(ridx);
      if((m_step % 3) == 0)
      {
         int ridx2 = PickReplayIndex(-1);
         if(ridx2 >= 0)
            AppendReplayToTrain(ridx2);
      }

      TrainTBPTT(h);

      // Update live state with most recent bar after isolated TBPTT step.
      double emb[FX6_AI_MLP_HIDDEN];
      double loc[FX6_AI_MLP_HIDDEN];
      double att[FX6_AI_MLP_HIDDEN];
      double fin[FX6_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FX6_TST_HEADS][FX6_TST_D_HEAD];
      ForwardStep(x, true, false, -1, emb, loc, att, fin, grp, ctx);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FX6_AI_MLP_HIDDEN];
      double loc[FX6_AI_MLP_HIDDEN];
      double att[FX6_AI_MLP_HIDDEN];
      double fin[FX6_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FX6_TST_HEADS][FX6_TST_D_HEAD];
      ForwardStep(x, false, false, -1, emb, loc, att, fin, grp, ctx);

      double logits[FX6_TST_CLASS_COUNT];
      double probs[FX6_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FX6_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);

      double den = probs[FX6_TST_BUY] + probs[FX6_TST_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FX6_TST_BUY] / den;
      double p_dir_cal = CalibrateSessionProb(SessionBucketNow(), p_dir_raw);

      double p_up = p_dir_cal * FX6_Clamp(1.0 - probs[FX6_TST_SKIP], 0.0, 1.0);
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FX6_AI_MLP_HIDDEN];
      double loc[FX6_AI_MLP_HIDDEN];
      double att[FX6_AI_MLP_HIDDEN];
      double fin[FX6_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FX6_TST_HEADS][FX6_TST_D_HEAD];
      ForwardStep(x, false, false, -1, emb, loc, att, fin, grp, ctx);

      double logits[FX6_TST_CLASS_COUNT];
      double probs[FX6_TST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FX6_TST_HORIZONS];
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75, mu_h[0], mu_h[1], mu_h[2]);

      double sigma = MathExp(0.5 * logv);
      sigma = FX6_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double ev_h = 0.50 * MathAbs(mu_h[0]) + 0.30 * MathAbs(mu_h[1]) + 0.20 * MathAbs(mu_h[2]);
      double ev = (0.55 * MathAbs(mu) + 0.45 * ev_h + 0.22 * sigma + 0.10 * iqr) * FX6_Clamp(1.0 - probs[FX6_TST_SKIP], 0.0, 1.0);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FX6_AI_TST_MQH__
