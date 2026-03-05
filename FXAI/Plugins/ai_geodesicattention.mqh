// FXAI v1
#ifndef __FXAI_AI_GEODESICATTENTION_MQH__
#define __FXAI_AI_GEODESICATTENTION_MQH__

#include "..\plugin_base.mqh"

#define FXAI_GA_SEQ 48
#define FXAI_GA_TBPTT 16
#define FXAI_GA_HEADS 2
#define FXAI_GA_D_HEAD (FXAI_AI_MLP_HIDDEN / FXAI_GA_HEADS)
#define FXAI_GA_CLASS_COUNT 3

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
   double m_seq_h[FXAI_GA_SEQ][FXAI_AI_MLP_HIDDEN];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // Variable-selection groups: observed / known / static.
   double m_group_mask[3][FXAI_AI_WEIGHTS];
   double m_v_gate_w[3][FXAI_AI_WEIGHTS];
   double m_v_gate_b[3];

   double m_w_obs[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_w_known[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_w_static[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_b_embed[FXAI_AI_MLP_HIDDEN];

   // GRN blocks.
   double m_grn1_w1[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn1_w2[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn1_wg[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn1_b1[FXAI_AI_MLP_HIDDEN];
   double m_grn1_b2[FXAI_AI_MLP_HIDDEN];
   double m_grn1_bg[FXAI_AI_MLP_HIDDEN];

   double m_grn2_w1[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn2_w2[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn2_wg[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_grn2_b1[FXAI_AI_MLP_HIDDEN];
   double m_grn2_b2[FXAI_AI_MLP_HIDDEN];
   double m_grn2_bg[FXAI_AI_MLP_HIDDEN];

   // Multi-head temporal attention.
   double m_wq[FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wk[FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wv[FXAI_GA_HEADS][FXAI_GA_D_HEAD][FXAI_AI_MLP_HIDDEN];
   double m_wo[FXAI_GA_HEADS][FXAI_AI_MLP_HIDDEN][FXAI_GA_D_HEAD];
   double m_geo_beta[FXAI_GA_HEADS];
   double m_geo_time_beta[FXAI_GA_HEADS];
   double m_geo_mix[FXAI_GA_HEADS];
   double m_geo_bias[FXAI_GA_HEADS];

   double m_attn_wg[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_attn_bg[FXAI_AI_MLP_HIDDEN];

   // Position-wise feed-forward block.
   double m_ff1_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ff1_b[FXAI_AI_MLP_HIDDEN];
   double m_ff2_w[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ff2_b[FXAI_AI_MLP_HIDDEN];

   // Normalization affine params.
   double m_ln1_g[FXAI_AI_MLP_HIDDEN];
   double m_ln1_b[FXAI_AI_MLP_HIDDEN];
   double m_ln_attn_g[FXAI_AI_MLP_HIDDEN];
   double m_ln_attn_b[FXAI_AI_MLP_HIDDEN];
   double m_ln2_g[FXAI_AI_MLP_HIDDEN];
   double m_ln2_b[FXAI_AI_MLP_HIDDEN];
   double m_ln_out_g[FXAI_AI_MLP_HIDDEN];
   double m_ln_out_b[FXAI_AI_MLP_HIDDEN];

   // Output heads: 3-class + distributional move.
   double m_w_cls[FXAI_GA_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_b_cls[FXAI_GA_CLASS_COUNT];

   double m_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_b_q75;

   // TBPTT buffers.
   int    m_train_len;
   double m_train_x[FXAI_GA_TBPTT][FXAI_AI_WEIGHTS];
   int    m_train_cls[FXAI_GA_TBPTT];
   double m_train_move[FXAI_GA_TBPTT];
   double m_train_cost[FXAI_GA_TBPTT];
   double m_train_w[FXAI_GA_TBPTT];

   // Forward caches for TBPTT updates.
   double m_cache_embed[FXAI_GA_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_local[FXAI_GA_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_attn[FXAI_GA_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_final[FXAI_GA_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_group[FXAI_GA_TBPTT][3];
   double m_cache_ctx[FXAI_GA_TBPTT][FXAI_GA_HEADS][FXAI_GA_D_HEAD];

   double m_cache_logits[FXAI_GA_TBPTT][FXAI_GA_CLASS_COUNT];
   double m_cache_probs[FXAI_GA_TBPTT][FXAI_GA_CLASS_COUNT];
   double m_cache_mu[FXAI_GA_TBPTT];
   double m_cache_logv[FXAI_GA_TBPTT];
   double m_cache_q25[FXAI_GA_TBPTT];
   double m_cache_q75[FXAI_GA_TBPTT];

   void LayerNormAffine(double &v[], const double &g[], const double &b[]) const
   {
      double mean = 0.0;
      for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         mean += v[i];
      mean /= (double)FXAI_AI_MLP_HIDDEN;

      double var = 0.0;
      for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
      {
         double d = v[i] - mean;
         var += d * d;
      }

      double inv = 1.0 / MathSqrt(var / (double)FXAI_AI_MLP_HIDDEN + 1e-6);
      for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
      {
         double n = (v[i] - mean) * inv;
         v[i] = FXAI_ClipSym(g[i] * n + b[i], 8.0);
      }
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
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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
         m_cache_logv[t] = MathLog(1.0);
         m_cache_q25[t] = 0.0;
         m_cache_q75[t] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[t][i] = 0.0;

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[t][h] = 0.0;
            m_cache_local[t][h] = 0.0;
            m_cache_attn[t][h] = 0.0;
            m_cache_final[t][h] = 0.0;
         }

         for(int g=0; g<3; g++)
            m_cache_group[t][g] = 0.0;

         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
               m_cache_ctx[t][hd][d] = 0.0;
         }

         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
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
         if(edge <= 0.0) return FXAI_Clamp(base * 1.6, 0.25, 6.0);
         return FXAI_Clamp(base * 0.75, 0.25, 6.0);
      }

      if(edge <= 0.0) return FXAI_Clamp(base * 0.55, 0.25, 6.0);
      return FXAI_Clamp(base * (1.0 + 0.06 * MathMin(edge, 20.0)), 0.25, 6.0);
   }

   double MoveWeight(const double move_points,
                     const double cost,
                     const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double ew = FXAI_Clamp(0.5 + edge / denom, 0.25, 4.0);
      return FXAI_Clamp(sample_w * ew, 0.25, 8.0);
   }

   void InitGroupMasks(void)
   {
      // feature index = i - 1 (x[0] is bias)
      for(int g=0; g<3; g++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_group_mask[g][i] = 0.0;
      }

      for(int g=0; g<3; g++)
         m_group_mask[g][0] = 1.0;

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
      double a = (m_x_norm_steps < 128 ? 0.05 : 0.015);
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

   void InitWeights(void)
   {
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
      InitGroupMasks();
      m_step = 0;

      for(int g=0; g<3; g++)
      {
         m_v_gate_b[g] = (g == 0 ? 0.25 : 0.0);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s = (double)((g + 1) * (i + 2));
            m_v_gate_w[g][i] = 0.03 * MathSin(0.71 * s);
         }
      }

      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
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

         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
            m_w_cls[c][o] = 0.03 * MathSin((double)((c + 2) * (o + 1)) * 0.87);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s0 = (double)((o + 1) * (i + 3));
            double s1 = (double)((o + 2) * (i + 5));
            double s2 = (double)((o + 3) * (i + 7));
            m_w_obs[o][i] = 0.04 * MathSin(0.67 * s0);
            m_w_known[o][i] = 0.04 * MathCos(0.73 * s1);
            m_w_static[o][i] = 0.04 * MathSin(0.79 * s2);
         }

         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
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
      }

      for(int hd=0; hd<FXAI_GA_HEADS; hd++)
      {
         m_geo_beta[hd] = 0.90 + 0.15 * (double)hd;
         m_geo_time_beta[hd] = 0.45 + 0.10 * (double)hd;
         m_geo_mix[hd] = -0.25 + 0.20 * (double)hd;
         m_geo_bias[hd] = 0.0;

         for(int d=0; d<FXAI_GA_D_HEAD; d++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double t = (double)((hd + 1) * (d + 2) * (h + 3));
               m_wq[hd][d][h] = 0.04 * MathSin(0.69 * t);
               m_wk[hd][d][h] = 0.04 * MathCos(0.73 * t);
               m_wv[hd][d][h] = 0.04 * MathSin(0.79 * t);
               m_wo[hd][h][d] = 0.04 * MathCos(0.83 * t);
            }
         }
      }

      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;

      m_initialized = true;
   }

   void ApplyGRN(const double &in[],
                 const double &w1[][FXAI_AI_MLP_HIDDEN],
                 const double &w2[][FXAI_AI_MLP_HIDDEN],
                 const double &wg[][FXAI_AI_MLP_HIDDEN],
                 const double &b1[],
                 const double &b2[],
                 const double &bg[],
                 const double &ln_g[],
                 const double &ln_b[],
                 double &out[]) const
   {
      double h1[FXAI_AI_MLP_HIDDEN];
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double z1 = b1[o];
         double zg = bg[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
         {
            z1 += w1[o][i] * in[i];
            zg += wg[o][i] * in[i];
         }
         h1[o] = FXAI_Tanh(z1);

         double z2 = b2[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z2 += w2[o][i] * h1[i];

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
      for(int g=0; g<3; g++)
         group_gate[g] /= den;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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
                    double &tokens[][FXAI_AI_MLP_HIDDEN],
                    int &count) const
   {
      int keep = m_seq_len;
      if(keep > FXAI_GA_SEQ - 1) keep = FXAI_GA_SEQ - 1;

      for(int i=0; i<keep; i++)
      {
         int back = keep - 1 - i;
         int idx = SeqIndexBack(back);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            tokens[i][h] = m_seq_h[idx][h];
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         tokens[keep][h] = current[h];

      count = keep + 1;
   }

   void MultiHeadAttention(const double &query_src[],
                           const double &tokens[][FXAI_AI_MLP_HIDDEN],
                           const int token_count,
                           double &attn_out[],
                           double &head_ctx[][FXAI_GA_D_HEAD]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) attn_out[h] = 0.0;
      for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         for(int d=0; d<FXAI_GA_D_HEAD; d++)
            head_ctx[hd][d] = 0.0;

      if(token_count <= 0) return;

      double inv_scale = 1.0 / MathSqrt((double)FXAI_GA_D_HEAD);
      double q[FXAI_GA_D_HEAD];
      double qn[FXAI_GA_D_HEAD];
      double scores[FXAI_GA_SEQ];
      double k_norm_cache[FXAI_GA_SEQ][FXAI_GA_D_HEAD];
      double v_cache[FXAI_GA_SEQ][FXAI_GA_D_HEAD];
      const double PI = 3.141592653589793;

      for(int hd=0; hd<FXAI_GA_HEADS; hd++)
      {
         double qn2 = 0.0;
         for(int d=0; d<FXAI_GA_D_HEAD; d++)
         {
            double s = 0.0;
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               s += m_wq[hd][d][j] * query_src[j];
            q[d] = s;
            qn2 += s * s;
         }
         double q_inv = 1.0 / MathSqrt(qn2 + 1e-9);
         for(int d=0; d<FXAI_GA_D_HEAD; d++)
            qn[d] = q[d] * q_inv;

         double mix = FXAI_Sigmoid(m_geo_mix[hd]);
         double geo_beta = FXAI_Clamp(m_geo_beta[hd], 0.05, 8.0);
         double time_beta = FXAI_Clamp(m_geo_time_beta[hd], 0.0, 4.0);

         double max_sc = -1e100;
         for(int t=0; t<token_count; t++)
         {
            double dot_uv = 0.0;
            double kn2 = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               double kv = 0.0;
               double vv = 0.0;
               for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               {
                  kv += m_wk[hd][d][j] * tokens[t][j];
                  vv += m_wv[hd][d][j] * tokens[t][j];
               }
               k_norm_cache[t][d] = kv;
               kn2 += kv * kv;
               v_cache[t][d] = vv;
            }

            double k_inv = 1.0 / MathSqrt(kn2 + 1e-9);
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               k_norm_cache[t][d] *= k_inv;
               dot_uv += qn[d] * k_norm_cache[t][d];
            }

            dot_uv = FXAI_Clamp(dot_uv, -0.999999, 0.999999);
            double theta = MathArccos(dot_uv);

            double lag_frac = 0.0;
            if(token_count > 1)
               lag_frac = (double)(token_count - 1 - t) / (double)(token_count - 1);
            lag_frac = FXAI_Clamp(lag_frac, 0.0, 1.0);
            double time_geo = PI * lag_frac;

            double content = (dot_uv * inv_scale) + m_geo_bias[hd];
            double geo = -(geo_beta * theta) - (time_beta * time_geo);
            double sc = (mix * content) + ((1.0 - mix) * geo);

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

         for(int d=0; d<FXAI_GA_D_HEAD; d++)
         {
            double c = 0.0;
            for(int t=0; t<token_count; t++)
               c += (scores[t] / den) * v_cache[t][d];
            head_ctx[hd][d] = c;
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            double s = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
               s += m_wo[hd][j][d] * head_ctx[hd][d];
            attn_out[j] += s;
         }
      }
   }

   void ForwardStep(const double &x[],
                    const bool commit,
                    double &state_embed[],
                    double &state_local[],
                    double &state_attn[],
                    double &state_final[],
                    double &group_gate[],
                    double &head_ctx[][FXAI_GA_D_HEAD])
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

      double tokens[FXAI_GA_SEQ][FXAI_AI_MLP_HIDDEN];
      int token_count = 0;
      BuildTokens(state_local, tokens, token_count);

      double attn_mix[FXAI_AI_MLP_HIDDEN];
      MultiHeadAttention(state_local, tokens, token_count, attn_mix, head_ctx);

      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double zg = m_attn_bg[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            zg += m_attn_wg[o][i] * state_local[i];

         double g = FXAI_Sigmoid(zg);
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

      double ff1[FXAI_AI_MLP_HIDDEN];
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double z1 = m_ff1_b[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z1 += m_ff1_w[o][i] * state_final[i];
         ff1[o] = FXAI_Tanh(z1);
      }

      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
      {
         double z2 = m_ff2_b[o];
         for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            z2 += m_ff2_w[o][i] * ff1[i];
         state_final[o] = state_final[o] + 0.25 * FXAI_ClipSym(z2, 8.0);
      }
      LayerNormAffine(state_final, m_ln_out_g, m_ln_out_b);

      if(commit)
      {
         m_seq_ptr++;
         if(m_seq_ptr >= FXAI_GA_SEQ) m_seq_ptr = 0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            m_seq_h[m_seq_ptr][h] = state_final[h];
         if(m_seq_len < FXAI_GA_SEQ) m_seq_len++;
      }
   }

   void ComputeHeads(const double &state_final[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75) const
   {
      for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            z += m_w_cls[c][h] * state_final[h];
         logits[c] = FXAI_ClipSym(z, 20.0);
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      double r25 = m_b_q25;
      double r75 = m_b_q75;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         mu += m_w_mu[h] * state_final[h];
         logv += m_w_logv[h] * state_final[h];
         r25 += m_w_q25[h] * state_final[h];
         r75 += m_w_q75[h] * state_final[h];
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
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
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[p][i] = x[i];
         m_train_cls[p] = cls;
         m_train_move[p] = move_points;
         m_train_cost[p] = cost_points;
         m_train_w[p] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FXAI_GA_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_train_x[t - 1][i] = m_train_x[t][i];
         m_train_cls[t - 1] = m_train_cls[t];
         m_train_move[t - 1] = m_train_move[t];
         m_train_cost[t - 1] = m_train_cost[t];
         m_train_w[t - 1] = m_train_w[t];
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_train_x[FXAI_GA_TBPTT - 1][i] = x[i];
      m_train_cls[FXAI_GA_TBPTT - 1] = cls;
      m_train_move[FXAI_GA_TBPTT - 1] = move_points;
      m_train_cost[FXAI_GA_TBPTT - 1] = cost_points;
      m_train_w[FXAI_GA_TBPTT - 1] = sample_w;
   }

   void TrainTBPTT(const FXAIAIHyperParams &hp)
   {
      if(m_train_len <= 0) return;

      int n = m_train_len;
      if(n > FXAI_GA_TBPTT) n = FXAI_GA_TBPTT;
      if(n <= 0) return;

      double lr0 = FXAI_Clamp(hp.lr, 0.0001, 1.0000);
      double l2 = FXAI_Clamp(hp.l2, 0.0000, 1.0000);

      // Save live state then isolate sequence for TBPTT to avoid hidden-state bleed.
      int saved_ptr = m_seq_ptr;
      int saved_len = m_seq_len;
      double saved_seq[FXAI_GA_SEQ][FXAI_AI_MLP_HIDDEN];
      for(int t=0; t<FXAI_GA_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            saved_seq[t][h] = m_seq_h[t][h];
      }

      ResetSequence();

      // Forward pass over truncated sequence.
      for(int t=0; t<n; t++)
      {
         double xrow[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xrow[i] = m_train_x[t][i];

         double emb[FXAI_AI_MLP_HIDDEN];
         double loc[FXAI_AI_MLP_HIDDEN];
         double att[FXAI_AI_MLP_HIDDEN];
         double fin[FXAI_AI_MLP_HIDDEN];
         double grp[3];
         double ctx[FXAI_GA_HEADS][FXAI_GA_D_HEAD];

         ForwardStep(xrow, true, emb, loc, att, fin, grp, ctx);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_cache_embed[t][h] = emb[h];
            m_cache_local[t][h] = loc[h];
            m_cache_attn[t][h] = att[h];
            m_cache_final[t][h] = fin[h];
         }

         for(int g=0; g<3; g++)
            m_cache_group[t][g] = grp[g];

         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
               m_cache_ctx[t][hd][d] = ctx[hd][d];

         double logits[FXAI_GA_CLASS_COUNT];
         double probs[FXAI_GA_CLASS_COUNT];
         ComputeHeads(fin,
                      logits,
                      probs,
                      m_cache_mu[t],
                      m_cache_logv[t],
                      m_cache_q25[t],
                      m_cache_q75[t]);
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = logits[c];
            m_cache_probs[t][c] = probs[c];
         }
      }

      // Reverse-time updates with truncated temporal error flow.
      double next_delta[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) next_delta[h] = 0.0;

      for(int t=n - 1; t>=0; t--)
      {
         m_step++;
         double lr = lr0 / MathSqrt(1.0 + 0.0012 * (double)m_step);

         int cls = m_train_cls[t];
         double mv = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = FXAI_Clamp(m_train_w[t], 0.25, 4.00);

         double w_cls = ClassWeight(cls, mv, cost, sw);
         double w_mv = MoveWeight(mv, cost, sw);

         double g_logits[FXAI_GA_CLASS_COUNT];
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            double target = (c == cls ? 1.0 : 0.0);
            g_logits[c] = FXAI_ClipSym((m_cache_probs[t][c] - target) * w_cls, 4.0);
         }

         double target_move = mv;
         double mu = m_cache_mu[t];
         double logv = m_cache_logv[t];
         double q25 = m_cache_q25[t];
         double q75 = m_cache_q75[t];

         double sigma2 = MathExp(logv);
         sigma2 = FXAI_Clamp(sigma2, 0.05, 100.0);

         double err_mu = FXAI_ClipSym(mu - target_move, 30.0);
         double g_mu = FXAI_ClipSym((err_mu / sigma2) * w_mv, 5.0);

         double ratio = (err_mu * err_mu) / sigma2;
         double g_logv = FXAI_ClipSym(0.5 * (1.0 - ratio) * w_mv, 4.0);

         double e25 = target_move - q25;
         double e75 = target_move - q75;
         double g_q25 = FXAI_ClipSym(((e25 < 0.0 ? -0.75 : 0.25)) * w_mv, 3.0);
         double g_q75 = FXAI_ClipSym(((e75 < 0.0 ? -0.25 : 0.75)) * w_mv, 3.0);

         double delta[FXAI_AI_MLP_HIDDEN];
         double gn = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double d = 0.35 * next_delta[h];
            for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
               d += g_logits[c] * m_w_cls[c][h];
            d += g_mu * m_w_mu[h];
            d += g_logv * m_w_logv[h];
            d += g_q25 * m_w_q25[h];
            d += g_q75 * m_w_q75[h];
            delta[h] = d;
            gn += d * d;
         }

         gn = MathSqrt(gn);
         if(gn > 4.0)
         {
            double sc = 4.0 / gn;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               delta[h] *= sc;
         }

         // Head updates.
         for(int c=0; c<FXAI_GA_CLASS_COUNT; c++)
         {
            m_b_cls[c] -= lr * g_logits[c];
            m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 8.0);
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double g = g_logits[c] * m_cache_final[t][h];
               m_w_cls[c][h] -= lr * (FXAI_ClipSym(g, 6.0) + l2 * m_w_cls[c][h]);
            }
         }

         m_b_mu -= lr * g_mu;
         m_b_logv -= lr * g_logv;
         m_b_q25 -= lr * g_q25;
         m_b_q75 -= lr * g_q75;

         m_b_mu = FXAI_ClipSym(m_b_mu, 20.0);
         m_b_logv = FXAI_Clamp(m_b_logv, -4.0, 4.0);
         m_b_q25 = FXAI_ClipSym(m_b_q25, 20.0);
         m_b_q75 = FXAI_ClipSym(m_b_q75, 20.0);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_w_mu[h] -= lr * (FXAI_ClipSym(g_mu * m_cache_final[t][h], 6.0) + l2 * m_w_mu[h]);
            m_w_logv[h] -= lr * (FXAI_ClipSym(g_logv * m_cache_final[t][h], 6.0) + l2 * m_w_logv[h]);
            m_w_q25[h] -= lr * (FXAI_ClipSym(g_q25 * m_cache_final[t][h], 6.0) + l2 * m_w_q25[h]);
            m_w_q75[h] -= lr * (FXAI_ClipSym(g_q75 * m_cache_final[t][h], 6.0) + l2 * m_w_q75[h]);
         }

         // FFN + normalization affine updates.
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_ff2_b[o] -= lr * 0.08 * d;
            m_ff1_b[o] -= lr * 0.04 * d;

            m_ln_out_b[o] -= lr * 0.01 * d;
            m_ln_out_g[o] -= lr * (0.01 * d * m_cache_final[t][o] + l2 * m_ln_out_g[o]);
            m_ln2_b[o] -= lr * 0.008 * d;
            m_ln2_g[o] -= lr * (0.008 * d * m_cache_attn[t][o] + l2 * m_ln2_g[o]);
            m_ln_attn_b[o] -= lr * 0.008 * d;
            m_ln_attn_g[o] -= lr * (0.008 * d * m_cache_attn[t][o] + l2 * m_ln_attn_g[o]);
            m_ln1_b[o] -= lr * 0.006 * d;
            m_ln1_g[o] -= lr * (0.006 * d * m_cache_local[t][o] + l2 * m_ln1_g[o]);

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               m_ff2_w[o][i] -= lr * (0.08 * FXAI_ClipSym(d * m_cache_final[t][i], 4.0) + l2 * m_ff2_w[o][i]);
               m_ff1_w[o][i] -= lr * (0.04 * FXAI_ClipSym(d * m_cache_local[t][i], 4.0) + l2 * m_ff1_w[o][i]);
            }
         }

         // GRN2 + attention gate.
         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_grn2_b2[o] -= lr * 0.18 * d;
            m_grn2_b1[o] -= lr * 0.10 * d;
            m_grn2_bg[o] -= lr * 0.06 * d;
            m_attn_bg[o] -= lr * 0.05 * d;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               m_grn2_w2[o][i] -= lr * (0.18 * FXAI_ClipSym(d * m_cache_attn[t][i], 4.0) + l2 * m_grn2_w2[o][i]);
               m_grn2_w1[o][i] -= lr * (0.10 * FXAI_ClipSym(d * m_cache_local[t][i], 4.0) + l2 * m_grn2_w1[o][i]);
               m_grn2_wg[o][i] -= lr * (0.06 * FXAI_ClipSym(d * m_cache_attn[t][i], 4.0) + l2 * m_grn2_wg[o][i]);
               m_attn_wg[o][i] -= lr * (0.05 * FXAI_ClipSym(d * m_cache_local[t][i], 4.0) + l2 * m_attn_wg[o][i]);
            }
         }

         // Attention projections.
         double head_delta[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               double s = 0.0;
               for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
                  s += delta[o] * m_wo[hd][o][d];
               head_delta[hd][d] = FXAI_ClipSym(s, 3.0);
            }
         }

         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               double dh = head_delta[hd][d];
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  m_wo[hd][h][d] -= lr * (0.08 * FXAI_ClipSym(delta[h] * m_cache_ctx[t][hd][d], 4.0) + l2 * m_wo[hd][h][d]);
                  m_wq[hd][d][h] -= lr * (0.04 * FXAI_ClipSym(dh * m_cache_local[t][h], 4.0) + l2 * m_wq[hd][d][h]);
                  m_wk[hd][d][h] -= lr * (0.03 * FXAI_ClipSym(dh * m_cache_attn[t][h], 4.0) + l2 * m_wk[hd][d][h]);
                  m_wv[hd][d][h] -= lr * (0.03 * FXAI_ClipSym(dh * m_cache_attn[t][h], 4.0) + l2 * m_wv[hd][d][h]);
               }
            }

            double dh_abs = 0.0;
            double ctx_abs = 0.0;
            for(int d=0; d<FXAI_GA_D_HEAD; d++)
            {
               dh_abs += MathAbs(head_delta[hd][d]);
               ctx_abs += MathAbs(m_cache_ctx[t][hd][d]);
            }
            dh_abs /= (double)FXAI_GA_D_HEAD;
            ctx_abs /= (double)FXAI_GA_D_HEAD;

            double gaux = FXAI_ClipSym(0.60 * dh_abs + 0.40 * ctx_abs, 3.0);
            double dir = FXAI_Sign(m_cache_logits[t][FXAI_GA_BUY] - m_cache_logits[t][FXAI_GA_SELL]);
            if(dir == 0.0) dir = 1.0;
            gaux *= dir;

            m_geo_bias[hd] -= lr * 0.010 * gaux;
            m_geo_beta[hd] -= lr * (0.006 * gaux + l2 * m_geo_beta[hd]);
            m_geo_time_beta[hd] -= lr * (0.005 * gaux + l2 * m_geo_time_beta[hd]);
            m_geo_mix[hd] -= lr * (0.008 * gaux + l2 * m_geo_mix[hd]);

            m_geo_bias[hd] = FXAI_Clamp(m_geo_bias[hd], -4.0, 4.0);
            m_geo_beta[hd] = FXAI_Clamp(m_geo_beta[hd], 0.05, 8.0);
            m_geo_time_beta[hd] = FXAI_Clamp(m_geo_time_beta[hd], 0.0, 4.0);
            m_geo_mix[hd] = FXAI_Clamp(m_geo_mix[hd], -6.0, 6.0);
         }

         // GRN1 + variable-selection updates.
         double xn[FXAI_AI_WEIGHTS];
         double xrow2[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xrow2[i] = m_train_x[t][i];
         NormalizeInput(xrow2, xn);

         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_grn1_b2[o] -= lr * 0.08 * d;
            m_grn1_b1[o] -= lr * 0.05 * d;
            m_grn1_bg[o] -= lr * 0.03 * d;

            for(int i=0; i<FXAI_AI_MLP_HIDDEN; i++)
            {
               m_grn1_w2[o][i] -= lr * (0.08 * FXAI_ClipSym(d * m_cache_local[t][i], 4.0) + l2 * m_grn1_w2[o][i]);
               m_grn1_w1[o][i] -= lr * (0.05 * FXAI_ClipSym(d * m_cache_embed[t][i], 4.0) + l2 * m_grn1_w1[o][i]);
               m_grn1_wg[o][i] -= lr * (0.03 * FXAI_ClipSym(d * m_cache_local[t][i], 4.0) + l2 * m_grn1_wg[o][i]);
            }
         }

         for(int g=0; g<3; g++)
         {
            double gs = 0.0;
            int hs = (g + t) % FXAI_AI_MLP_HIDDEN;
            gs = delta[hs] * m_cache_local[t][hs];
            m_v_gate_b[g] -= lr * 0.004 * FXAI_ClipSym(gs, 3.0);

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xg = xn[i] * m_group_mask[g][i];
               m_v_gate_w[g][i] -= lr * (0.002 * FXAI_ClipSym(gs * xg, 3.0) + l2 * m_v_gate_w[g][i]);
            }
         }

         for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         {
            double d = delta[o];
            m_b_embed[o] -= lr * 0.02 * d;

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xi = xn[i];
               double a0 = m_cache_group[t][0] * xi * m_group_mask[0][i];
               double a1 = m_cache_group[t][1] * xi * m_group_mask[1][i];
               double a2 = m_cache_group[t][2] * xi * m_group_mask[2][i];

               m_w_obs[o][i] -= lr * (0.02 * FXAI_ClipSym(d * a0, 3.0) + l2 * m_w_obs[o][i]);
               m_w_known[o][i] -= lr * (0.02 * FXAI_ClipSym(d * a1, 3.0) + l2 * m_w_known[o][i]);
               m_w_static[o][i] -= lr * (0.02 * FXAI_ClipSym(d * a2, 3.0) + l2 * m_w_static[o][i]);
            }
         }

         // Plugin-level calibration + move-head support.
         double den = m_cache_probs[t][FXAI_GA_BUY] + m_cache_probs[t][FXAI_GA_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir_raw = m_cache_probs[t][FXAI_GA_BUY] / den;
         if(cls == FXAI_GA_BUY)
            UpdateCalibration(p_dir_raw, 1, w_cls);
         else if(cls == FXAI_GA_SELL)
            UpdateCalibration(p_dir_raw, 0, w_cls);
         else
            UpdateCalibration(p_dir_raw, (mv >= 0.0 ? 1 : 0), 0.25 * w_cls);

         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, mv, 0.05);
         UpdateMoveHead(xrow2, mv, hp, sw);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            next_delta[h] = delta[h];
      }

      // Restore live sequence state.
      m_seq_ptr = saved_ptr;
      m_seq_len = saved_len;
      for(int t=0; t<FXAI_GA_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            m_seq_h[t][h] = saved_seq[t][h];
      }
   }

public:
   CFXAIAIGeodesicAttention(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_GEODESICATTENTION; }
   virtual string AIName(void) const { return "geodesicattention"; }

   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      return BuildNativeFromDirectional(x, hp, class_probs, expected_move_points);
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
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      // Controlled reset policy to avoid state bleed in non-stationary jumps.
      if((m_step % 512) == 0)
         ResetSequence();
      if(MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
         ResetSequence();

      UpdateInputStats(x);

      int cls = MapClass(y, x, move_points);
      double sw = MoveSampleWeight(x, move_points);
      sw = FXAI_Clamp(sw, 0.25, 4.00);
      double cost = InputCostProxyPoints(x);

      AppendTrainSample(cls, x, move_points, cost, sw);
      TrainTBPTT(h);

      // Update live state with most recent bar after isolated TBPTT step.
      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      ForwardStep(x, true, emb, loc, att, fin, grp, ctx);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      ForwardStep(x, false, emb, loc, att, fin, grp, ctx);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75);

      double den = probs[FXAI_GA_BUY] + probs[FXAI_GA_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_GA_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);

      double p_up = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_GA_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double emb[FXAI_AI_MLP_HIDDEN];
      double loc[FXAI_AI_MLP_HIDDEN];
      double att[FXAI_AI_MLP_HIDDEN];
      double fin[FXAI_AI_MLP_HIDDEN];
      double grp[3];
      double ctx[FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      ForwardStep(x, false, emb, loc, att, fin, grp, ctx);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(fin, logits, probs, mu, logv, q25, q75);

      double sigma = MathExp(0.5 * logv);
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);

      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * iqr) * FXAI_Clamp(1.0 - probs[FXAI_GA_SKIP], 0.0, 1.0);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return CFXAIAIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FXAI_AI_GEODESICATTENTION_MQH__
