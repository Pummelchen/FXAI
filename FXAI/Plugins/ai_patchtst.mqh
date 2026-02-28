// FXAI v1
#ifndef __FX6_AI_PATCHTST_MQH__
#define __FX6_AI_PATCHTST_MQH__

#include "..\plugin_base.mqh"

// Multivariate PatchTST reference plugin for FX6.
// Design: patch embedding -> positional encoding -> transformer encoder stack
// -> 3-class probabilities + move-distribution heads (mu/logvar/q25/q75).
#define FX6_PTST_CLASS_COUNT 3
#define FX6_PTST_SEQ 96
#define FX6_PTST_PATCH_LEN 8
#define FX6_PTST_STRIDE 4
#define FX6_PTST_MAX_PATCHES 24
#define FX6_PTST_LAYERS 2
#define FX6_PTST_HEADS 2
#define FX6_PTST_D_MODEL FX6_AI_MLP_HIDDEN
#define FX6_PTST_D_HEAD (FX6_PTST_D_MODEL / FX6_PTST_HEADS)
#define FX6_PTST_D_FF 16
#define FX6_PTST_CAL_BINS 12

class CFX6AIPatchTST : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   int    m_adam_t;

   // Rolling multivariate sequence state.
   int    m_seq_ptr;
   int    m_seq_len;
   double m_seq[FX6_PTST_SEQ][FX6_AI_FEATURES];

   // Input normalization.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FX6_AI_WEIGHTS];
   double m_x_var[FX6_AI_WEIGHTS];

   // Class balance EMA.
   double m_cls_ema[FX6_PTST_CLASS_COUNT];

   // Patch embedding + channel gating.
   double m_w_patch[FX6_PTST_D_MODEL][FX6_AI_FEATURES][FX6_PTST_PATCH_LEN];
   double m_b_patch[FX6_PTST_D_MODEL];
   double m_ch_gate[FX6_AI_FEATURES];

   // Positional embedding per patch token.
   double m_pos[FX6_PTST_MAX_PATCHES][FX6_PTST_D_MODEL];

   // Encoder stack.
   double m_wq[FX6_PTST_LAYERS][FX6_PTST_HEADS][FX6_PTST_D_HEAD][FX6_PTST_D_MODEL];
   double m_wk[FX6_PTST_LAYERS][FX6_PTST_HEADS][FX6_PTST_D_HEAD][FX6_PTST_D_MODEL];
   double m_wv[FX6_PTST_LAYERS][FX6_PTST_HEADS][FX6_PTST_D_HEAD][FX6_PTST_D_MODEL];
   double m_wo[FX6_PTST_LAYERS][FX6_PTST_D_MODEL][FX6_PTST_D_MODEL];

   double m_wff1[FX6_PTST_LAYERS][FX6_PTST_D_FF][FX6_PTST_D_MODEL];
   double m_bff1[FX6_PTST_LAYERS][FX6_PTST_D_FF];
   double m_wff2[FX6_PTST_LAYERS][FX6_PTST_D_MODEL][FX6_PTST_D_FF];
   double m_bff2[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];

   double m_ln1_g[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
   double m_ln1_b[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
   double m_ln2_g[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
   double m_ln2_b[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];

   // Output heads.
   double m_w_cls[FX6_PTST_CLASS_COUNT][FX6_PTST_D_MODEL];
   double m_b_cls[FX6_PTST_CLASS_COUNT];

   double m_w_mu[FX6_PTST_D_MODEL];
   double m_b_mu;
   double m_w_logv[FX6_PTST_D_MODEL];
   double m_b_logv;
   double m_w_q25[FX6_PTST_D_MODEL];
   double m_b_q25;
   double m_w_q75[FX6_PTST_D_MODEL];
   double m_b_q75;

   // Native 3-class calibration.
   double m_cal_temp;
   double m_cal_bias[FX6_PTST_CLASS_COUNT];
   double m_cal_iso_pos[FX6_PTST_CLASS_COUNT][FX6_PTST_CAL_BINS];
   double m_cal_iso_cnt[FX6_PTST_CLASS_COUNT][FX6_PTST_CAL_BINS];
   int    m_cal3_steps;

   // Lightweight adaptive optimizer moments.
   double m_opt_m[8];
   double m_opt_v[8];

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   double GELU(const double x) const
   {
      double x3 = x * x * x;
      double t = 0.7978845608 * (x + 0.044715 * x3);
      return 0.5 * x * (1.0 + FX6_Tanh(t));
   }

   double GELUDerivApprox(const double x) const
   {
      // Lightweight smooth derivative approximation.
      double s = FX6_Sigmoid(1.702 * x);
      return FX6_Clamp(s * (1.0 + 1.702 * x * (1.0 - s)), 0.02, 1.20);
   }

   void LayerNormAffine(double &v[],
                        const double &g[],
                        const double &b[]) const
   {
      double mean = 0.0;
      for(int i=0; i<FX6_PTST_D_MODEL; i++)
         mean += v[i];
      mean /= (double)FX6_PTST_D_MODEL;

      double var = 0.0;
      for(int i=0; i<FX6_PTST_D_MODEL; i++)
      {
         double d = v[i] - mean;
         var += d * d;
      }

      double inv = 1.0 / MathSqrt(var / (double)FX6_PTST_D_MODEL + 1e-6);
      for(int i=0; i<FX6_PTST_D_MODEL; i++)
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

   double ScheduledLR(const FX6AIHyperParams &hp) const
   {
      double base = FX6_Clamp(hp.lr, 0.0002, 0.0800);
      double st = (double)MathMax(m_step, 1);
      double warm = FX6_Clamp(st / 160.0, 0.08, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0011 * MathMax(0.0, st - 160.0));
      double lr = base * warm * invsqrt;
      return FX6_Clamp(lr, 0.00005, 0.05000);
   }

   double AdamGroupLR(const int group_idx,
                      const double grad_mag,
                      const double base_lr)
   {
      int g = ClampI(group_idx, 0, 7);
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
      return FX6_Clamp(base_lr * (0.60 + 0.40 * scale), 0.000003, 0.100000);
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
      double a = (m_x_norm_steps < 160 ? 0.04 : 0.012);
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

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FX6_PTST_SEQ; t++)
      {
         for(int f=0; f<FX6_AI_FEATURES; f++)
            m_seq[t][f] = 0.0;
      }
   }

   void PushSequence(const double &xn[])
   {
      int next = m_seq_ptr + 1;
      if(next >= FX6_PTST_SEQ) next = 0;

      for(int f=0; f<FX6_AI_FEATURES; f++)
         m_seq[next][f] = xn[f + 1];

      m_seq_ptr = next;
      if(m_seq_len < FX6_PTST_SEQ) m_seq_len++;
   }

   void BuildTemporalMatrix(const double &xn[],
                            double &seq_out[][FX6_AI_FEATURES],
                            int &out_len) const
   {
      out_len = 0;
      if(m_seq_len > 0 && m_seq_ptr >= 0)
      {
         int start = m_seq_ptr - (m_seq_len - 1);
         while(start < 0) start += FX6_PTST_SEQ;

         for(int i=0; i<m_seq_len; i++)
         {
            int idx = start + i;
            while(idx >= FX6_PTST_SEQ) idx -= FX6_PTST_SEQ;
            for(int f=0; f<FX6_AI_FEATURES; f++)
               seq_out[out_len][f] = m_seq[idx][f];
            out_len++;
         }
      }

      // Append current normalized observation for prediction/training step.
      if(out_len < FX6_PTST_SEQ)
      {
         for(int f=0; f<FX6_AI_FEATURES; f++)
            seq_out[out_len][f] = xn[f + 1];
         out_len++;
      }
      else
      {
         // Overwrite the newest slot if full.
         for(int i=0; i<FX6_PTST_SEQ - 1; i++)
         {
            for(int f=0; f<FX6_AI_FEATURES; f++)
               seq_out[i][f] = seq_out[i + 1][f];
         }
         for(int f=0; f<FX6_AI_FEATURES; f++)
            seq_out[FX6_PTST_SEQ - 1][f] = xn[f + 1];
         out_len = FX6_PTST_SEQ;
      }
   }

   void BuildPatchTokens(const double &seq_mat[][FX6_AI_FEATURES],
                         const int seq_len,
                         double &tokens[][FX6_PTST_D_MODEL],
                         int &token_count,
                         double &patch_stat[][FX6_PTST_PATCH_LEN]) const
   {
      for(int f=0; f<FX6_AI_FEATURES; f++)
      {
         for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
            patch_stat[f][t] = 0.0;
      }

      int count = 1;
      if(seq_len >= FX6_PTST_PATCH_LEN)
         count = 1 + (seq_len - FX6_PTST_PATCH_LEN) / FX6_PTST_STRIDE;
      if(count < 1) count = 1;
      if(count > FX6_PTST_MAX_PATCHES) count = FX6_PTST_MAX_PATCHES;
      token_count = count;

      for(int p=0; p<count; p++)
      {
         int start;
         if(seq_len >= FX6_PTST_PATCH_LEN)
         {
            int base_start = seq_len - FX6_PTST_PATCH_LEN - (count - 1 - p) * FX6_PTST_STRIDE;
            start = ClampI(base_start, 0, MathMax(seq_len - FX6_PTST_PATCH_LEN, 0));
         }
         else
         {
            start = 0;
         }

         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            double z = m_b_patch[d];
            for(int f=0; f<FX6_AI_FEATURES; f++)
            {
               double g = FX6_Clamp(m_ch_gate[f], 0.10, 4.00);
               for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
               {
                  int idx = start + t;
                  if(idx >= seq_len) idx = seq_len - 1;
                  if(idx < 0) idx = 0;
                  double xv = seq_mat[idx][f];
                  z += m_w_patch[d][f][t] * (g * xv);
                  patch_stat[f][t] += xv;
               }
            }

            tokens[p][d] = FX6_ClipSym(FX6_Tanh(z) + m_pos[p][d], 8.0);
         }
      }

      double inv = 1.0 / (double)MathMax(count, 1);
      for(int f=0; f<FX6_AI_FEATURES; f++)
      {
         for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
            patch_stat[f][t] *= inv;
      }
   }

   void AttentionLayer(const int layer,
                       const double &in_tokens[][FX6_PTST_D_MODEL],
                       const int n_tokens,
                       double &out_tokens[][FX6_PTST_D_MODEL],
                       double &mean_in[],
                       double &mean_ctx[],
                       double &mean_ff[])
   {
      double ln1g[FX6_PTST_D_MODEL];
      double ln1b[FX6_PTST_D_MODEL];
      double ln2g[FX6_PTST_D_MODEL];
      double ln2b[FX6_PTST_D_MODEL];
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         ln1g[d] = m_ln1_g[layer][d];
         ln1b[d] = m_ln1_b[layer][d];
         ln2g[d] = m_ln2_g[layer][d];
         ln2b[d] = m_ln2_b[layer][d];
      }

      double inv_tok = 1.0 / (double)MathMax(n_tokens, 1);
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         mean_in[d] = 0.0;
         mean_ctx[d] = 0.0;
      }
      for(int r=0; r<FX6_PTST_D_FF; r++)
         mean_ff[r] = 0.0;

      for(int i=0; i<n_tokens; i++)
      {
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            mean_in[d] += in_tokens[i][d] * inv_tok;
      }

      double q[FX6_PTST_D_HEAD];
      double k[FX6_PTST_D_HEAD];
      double v[FX6_PTST_D_HEAD];
      double score[FX6_PTST_MAX_PATCHES];
      double ctx_head[FX6_PTST_HEADS][FX6_PTST_D_HEAD];

      double inv_scale = 1.0 / MathSqrt((double)FX6_PTST_D_HEAD);

      for(int i=0; i<n_tokens; i++)
      {
         for(int h=0; h<FX6_PTST_HEADS; h++)
         {
            for(int d=0; d<FX6_PTST_D_HEAD; d++)
            {
               double s = 0.0;
               for(int j=0; j<FX6_PTST_D_MODEL; j++)
                  s += m_wq[layer][h][d][j] * in_tokens[i][j];
               q[d] = s;
            }

            double mx = -1e100;
            for(int t=0; t<n_tokens; t++)
            {
               double sc = 0.0;
               for(int d=0; d<FX6_PTST_D_HEAD; d++)
               {
                  double ks = 0.0;
                  for(int j=0; j<FX6_PTST_D_MODEL; j++)
                     ks += m_wk[layer][h][d][j] * in_tokens[t][j];
                  k[d] = ks;
                  sc += q[d] * k[d];
               }
               score[t] = sc * inv_scale;
               if(score[t] > mx) mx = score[t];
            }

            double den = 0.0;
            for(int t=0; t<n_tokens; t++)
            {
               score[t] = MathExp(FX6_Clamp(score[t] - mx, -30.0, 30.0));
               den += score[t];
            }
            if(den <= 0.0) den = 1.0;

            for(int d=0; d<FX6_PTST_D_HEAD; d++)
               ctx_head[h][d] = 0.0;

            for(int t=0; t<n_tokens; t++)
            {
               double a = score[t] / den;
               for(int d=0; d<FX6_PTST_D_HEAD; d++)
               {
                  double vs = 0.0;
                  for(int j=0; j<FX6_PTST_D_MODEL; j++)
                     vs += m_wv[layer][h][d][j] * in_tokens[t][j];
                  v[d] = vs;
                  ctx_head[h][d] += a * v[d];
               }
            }
         }

         double att[FX6_PTST_D_MODEL];
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            double s = 0.0;
            for(int h=0; h<FX6_PTST_HEADS; h++)
            {
               for(int hd=0; hd<FX6_PTST_D_HEAD; hd++)
               {
                  int od = h * FX6_PTST_D_HEAD + hd;
                  s += m_wo[layer][d][od] * ctx_head[h][hd];
               }
            }
            att[d] = s;
         }

         double u[FX6_PTST_D_MODEL];
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            u[d] = in_tokens[i][d] + att[d];
         LayerNormAffine(u, ln1g, ln1b);

         double ff1[FX6_PTST_D_FF];
         for(int r=0; r<FX6_PTST_D_FF; r++)
         {
            double z = m_bff1[layer][r];
            for(int d=0; d<FX6_PTST_D_MODEL; d++)
               z += m_wff1[layer][r][d] * u[d];
            ff1[r] = GELU(z);
            mean_ff[r] += ff1[r] * inv_tok;
         }

         double v2[FX6_PTST_D_MODEL];
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            double z = m_bff2[layer][d];
            for(int r=0; r<FX6_PTST_D_FF; r++)
               z += m_wff2[layer][d][r] * ff1[r];
            v2[d] = u[d] + z;
         }
         LayerNormAffine(v2, ln2g, ln2b);

         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            out_tokens[i][d] = v2[d];
            mean_ctx[d] += att[d] * inv_tok;
         }
      }
   }

   void PoolRepresentation(const double &tokens[][FX6_PTST_D_MODEL],
                           const int n_tokens,
                           double &rep[]) const
   {
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
         rep[d] = 0.0;

      if(n_tokens <= 0)
         return;

      for(int t=0; t<n_tokens; t++)
      {
         double w = 1.0;
         if(t == n_tokens - 1) w = 1.75;
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            rep[d] += w * tokens[t][d];
      }

      double den = (double)n_tokens + 0.75;
      if(den <= 0.0) den = 1.0;
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
         rep[d] = FX6_ClipSym(rep[d] / den, 8.0);
   }

   void ComputeHeads(const double &rep[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75) const
   {
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            z += m_w_cls[c][d] * rep[d];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         mu += m_w_mu[d] * rep[d];
         logv += m_w_logv[d] * rep[d];
         q25 += m_w_q25[d] * rep[d];
         q75 += m_w_q75[d] * rep[d];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double t = q25;
         q25 = q75;
         q75 = t;
      }
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal_temp, 0.50, 3.00);
      double logits[FX6_PTST_CLASS_COUNT];
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_PTST_CLASS_COUNT];
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_PTST_CAL_BINS; b++) total += m_cal_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_PTST_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_PTST_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal_iso_cnt[c][b] > 1e-9)
               r = m_cal_iso_pos[c][b] / m_cal_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_PTST_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_PTST_CAL_BINS) bi = FX6_PTST_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
         p_cal[c] = FX6_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

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
      double inv_temp = 1.0 / FX6_Clamp(m_cal_temp, 0.50, 3.00);
      double logits[FX6_PTST_CLASS_COUNT];
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }

      double p_cal[FX6_PTST_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_bias[c] = FX6_ClipSym(m_cal_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_PTST_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_PTST_CAL_BINS) bi = FX6_PTST_CAL_BINS - 1;
         m_cal_iso_cnt[c][bi] += w;
         m_cal_iso_pos[c][bi] += w * target;
      }

      m_cal_temp = FX6_Clamp(m_cal_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void InitWeights(void)
   {
      ResetSequence();
      ResetInputNorm();
      m_step = 0;
      m_adam_t = 0;

      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_cal_bias[c] = 0.0;
         m_b_cls[c] = 0.0;
         for(int b=0; b<FX6_PTST_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal_temp = 1.0;
      m_cal3_steps = 0;

      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      for(int f=0; f<FX6_AI_FEATURES; f++)
         m_ch_gate[f] = 1.0;

      for(int p=0; p<FX6_PTST_MAX_PATCHES; p++)
      {
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            m_pos[p][d] = 0.015 * MathSin((double)((p + 1) * (d + 2)) * 0.43);
         }
      }

      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         m_b_patch[d] = 0.0;
         m_w_mu[d] = 0.03 * MathSin((double)(d + 2) * 0.83);
         m_w_logv[d] = 0.03 * MathCos((double)(d + 3) * 0.89);
         m_w_q25[d] = 0.03 * MathSin((double)(d + 4) * 0.97);
         m_w_q75[d] = 0.03 * MathCos((double)(d + 5) * 1.03);

         for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
            m_w_cls[c][d] = 0.03 * MathSin((double)((c + 2) * (d + 1)) * 0.79);

         for(int f=0; f<FX6_AI_FEATURES; f++)
         {
            for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
            {
               double s = (double)((d + 1) * (f + 2) * (t + 3));
               m_w_patch[d][f][t] = 0.025 * MathSin(0.61 * s);
            }
         }
      }

      for(int l=0; l<FX6_PTST_LAYERS; l++)
      {
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            m_bff2[l][d] = 0.0;
            m_ln1_g[l][d] = 1.0;
            m_ln1_b[l][d] = 0.0;
            m_ln2_g[l][d] = 1.0;
            m_ln2_b[l][d] = 0.0;

            for(int od=0; od<FX6_PTST_D_MODEL; od++)
            {
               double s = (double)((l + 1) * (d + 2) * (od + 3));
               m_wo[l][d][od] = 0.02 * MathCos(0.53 * s);
            }

            for(int r=0; r<FX6_PTST_D_FF; r++)
            {
               double s2 = (double)((l + 1) * (d + 1) * (r + 2));
               m_wff2[l][d][r] = 0.02 * MathSin(0.57 * s2);
            }
         }

         for(int r=0; r<FX6_PTST_D_FF; r++)
         {
            m_bff1[l][r] = 0.0;
            for(int d=0; d<FX6_PTST_D_MODEL; d++)
            {
               double s = (double)((l + 2) * (r + 1) * (d + 3));
               m_wff1[l][r][d] = 0.02 * MathCos(0.59 * s);
            }
         }

         for(int h=0; h<FX6_PTST_HEADS; h++)
         {
            for(int dh=0; dh<FX6_PTST_D_HEAD; dh++)
            {
               for(int d=0; d<FX6_PTST_D_MODEL; d++)
               {
                  double s = (double)((l + 1) * (h + 2) * (dh + 3) * (d + 1));
                  m_wq[l][h][dh][d] = 0.02 * MathSin(0.47 * s);
                  m_wk[l][h][dh][d] = 0.02 * MathCos(0.49 * s);
                  m_wv[l][h][dh][d] = 0.02 * MathSin(0.51 * s);
               }
            }
         }
      }

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;

      // Slight skip prior before calibration settles.
      m_b_cls[(int)FX6_LABEL_SKIP] = 0.20;

      m_initialized = true;
   }

   void ForwardPass(const double &x[],
                    const bool commit,
                    double &rep[],
                    double &p_raw[],
                    double &mu,
                    double &logv,
                    double &q25,
                    double &q75,
                    double &patch_stat[][FX6_PTST_PATCH_LEN],
                    double &layer_in_mean[][FX6_PTST_D_MODEL],
                    double &layer_ctx_mean[][FX6_PTST_D_MODEL],
                    double &layer_ff_mean[][FX6_PTST_D_FF],
                    int &token_count)
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double seq_mat[FX6_PTST_SEQ][FX6_AI_FEATURES];
      int seq_len = 0;
      BuildTemporalMatrix(xn, seq_mat, seq_len);

      double tokens_a[FX6_PTST_MAX_PATCHES][FX6_PTST_D_MODEL];
      double tokens_b[FX6_PTST_MAX_PATCHES][FX6_PTST_D_MODEL];
      BuildPatchTokens(seq_mat, seq_len, tokens_a, token_count, patch_stat);

      if(token_count < 1) token_count = 1;
      if(token_count > FX6_PTST_MAX_PATCHES) token_count = FX6_PTST_MAX_PATCHES;

      for(int l=0; l<FX6_PTST_LAYERS; l++)
      {
         double mean_in_l[FX6_PTST_D_MODEL];
         double mean_ctx_l[FX6_PTST_D_MODEL];
         double mean_ff_l[FX6_PTST_D_FF];

         AttentionLayer(l,
                        tokens_a,
                        token_count,
                        tokens_b,
                        mean_in_l,
                        mean_ctx_l,
                        mean_ff_l);

         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            layer_in_mean[l][d] = mean_in_l[d];
            layer_ctx_mean[l][d] = mean_ctx_l[d];
         }
         for(int r=0; r<FX6_PTST_D_FF; r++)
            layer_ff_mean[l][r] = mean_ff_l[r];

         for(int t=0; t<token_count; t++)
         {
            for(int d=0; d<FX6_PTST_D_MODEL; d++)
               tokens_a[t][d] = tokens_b[t][d];
         }
      }

      PoolRepresentation(tokens_a, token_count, rep);

      double logits[FX6_PTST_CLASS_COUNT];
      ComputeHeads(rep, logits, p_raw, mu, logv, q25, q75);

      if(commit)
         PushSequence(xn);
   }

   double ExpectedMoveFromHeads(const double mu,
                                const double logv,
                                const double q25,
                                const double q75,
                                const double skip_prob) const
   {
      double sigma = MathExp(0.5 * FX6_Clamp(logv, -4.0, 4.0));
      sigma = FX6_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double ev = (0.68 * MathAbs(mu) + 0.22 * sigma + 0.10 * iqr) * FX6_Clamp(1.0 - skip_prob, 0.0, 1.0);
      return ev;
   }

public:
   CFX6AIPatchTST(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_PATCHTST; }
   virtual string AIName(void) const { return "patchtst"; }
   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_adam_t = 0;
      ResetSequence();
      ResetInputNorm();
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
         m_cls_ema[c] = 1.0;
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
   }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double rep[FX6_PTST_D_MODEL];
      double p_raw[FX6_PTST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double patch_stat[FX6_AI_FEATURES][FX6_PTST_PATCH_LEN];
      double layer_in_mean[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
      double layer_ctx_mean[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
      double layer_ff_mean[FX6_PTST_LAYERS][FX6_PTST_D_FF];
      int token_count = 0;

      ForwardPass(x,
                  false,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  patch_stat,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  token_count);

      Calibrate3(p_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FX6_LABEL_SKIP]);
      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);

      if(ev > 0.0 && base_ev > 0.0)
         expected_move_points = 0.70 * ev + 0.30 * base_ev;
      else if(ev > 0.0)
         expected_move_points = ev;
      else
         expected_move_points = base_ev;

      if(expected_move_points <= 0.0)
         expected_move_points = MathMax(ResolveMinMovePoints(), 0.10);

      return true;
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      UpdateWithMove(cls, x, hp, pseudo_move);
   }

protected:
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      m_step++;
      m_adam_t++;

      // Controlled reset policy to reduce state bleed across sharp regime jumps.
      if((m_step % 4096) == 0)
         ResetSequence();
      if(MathAbs(x[1]) > 9.0 || MathAbs(x[2]) > 9.0)
         ResetSequence();

      UpdateInputStats(x);

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
         cls = (int)FX6_LABEL_SKIP;

      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FX6_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double base_lr = ScheduledLR(h);
      double l2 = FX6_Clamp(h.l2, 0.0, 0.0800);

      double cost = InputCostProxyPoints(x);
      double sample_w = MoveSampleWeight(x, move_points);
      sample_w = FX6_Clamp(sample_w * cls_bal, 0.10, 6.00);

      double rep[FX6_PTST_D_MODEL];
      double p_raw[FX6_PTST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double patch_stat[FX6_AI_FEATURES][FX6_PTST_PATCH_LEN];
      double layer_in_mean[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
      double layer_ctx_mean[FX6_PTST_LAYERS][FX6_PTST_D_MODEL];
      double layer_ff_mean[FX6_PTST_LAYERS][FX6_PTST_D_FF];
      int token_count = 0;

      ForwardPass(x,
                  true,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  patch_stat,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  token_count);

      double cal_lr = FX6_Clamp(0.02 + 0.12 * base_lr, 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, sample_w, cal_lr);

      // Keep binary calibrator aligned for legacy paths.
      double den_dir = p_raw[(int)FX6_LABEL_BUY] + p_raw[(int)FX6_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FX6_LABEL_BUY] / den_dir;
      if(cls == (int)FX6_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, sample_w);
      else if(cls == (int)FX6_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, sample_w);

      double target_cls[FX6_PTST_CLASS_COUNT];
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
         target_cls[c] = (c == cls ? 1.0 : 0.0);

      // Cross-entropy gradient.
      double err_cls[FX6_PTST_CLASS_COUNT];
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
         err_cls[c] = (p_raw[c] - target_cls[c]);

      double g_rep[FX6_PTST_D_MODEL];
      for(int d=0; d<FX6_PTST_D_MODEL; d++) g_rep[d] = 0.0;

      double lr_head = AdamGroupLR(0, MathAbs(err_cls[0]) + MathAbs(err_cls[1]) + MathAbs(err_cls[2]), base_lr);
      for(int c=0; c<FX6_PTST_CLASS_COUNT; c++)
      {
         m_b_cls[c] -= lr_head * sample_w * err_cls[c];
         m_b_cls[c] = FX6_ClipSym(m_b_cls[c], 4.0);

         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            double w_old = m_w_cls[c][d];
            double grad = err_cls[c] * rep[d] + l2 * 0.20 * m_w_cls[c][d];
            m_w_cls[c][d] -= lr_head * sample_w * grad;
            g_rep[d] += err_cls[c] * w_old;
         }
      }

      // Distributional move head gradients.
      double move_tgt = MathAbs(move_points);
      double sigma = MathExp(0.5 * FX6_Clamp(logv, -4.0, 4.0));
      sigma = FX6_Clamp(sigma, 0.05, 30.0);
      double sig2 = sigma * sigma;

      double diff = mu - move_tgt;
      double g_mu = FX6_ClipSym(diff / (sig2 + 1e-6), 4.0);
      double g_logv = FX6_ClipSym(0.5 * (1.0 - (diff * diff) / (sig2 + 1e-6)), 4.0);
      double e25 = move_tgt - q25;
      double e75 = move_tgt - q75;
      double g_q25 = (e25 >= 0.0 ? -0.25 : 0.75);
      double g_q75 = (e75 >= 0.0 ? -0.75 : 0.25);

      double edge = MathAbs(move_points) - cost;
      double move_w = FX6_Clamp(sample_w * (0.50 + edge / MathMax(cost, 1.0)), 0.10, 8.00);
      double lr_move = AdamGroupLR(1, MathAbs(g_mu) + MathAbs(g_logv), base_lr * 0.70);

      m_b_mu -= lr_move * move_w * g_mu;
      m_b_logv -= lr_move * move_w * g_logv;
      m_b_q25 -= lr_move * move_w * g_q25;
      m_b_q75 -= lr_move * move_w * g_q75;

      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         double wmu_old = m_w_mu[d];
         double wlv_old = m_w_logv[d];
         double w25_old = m_w_q25[d];
         double w75_old = m_w_q75[d];

         m_w_mu[d] -= lr_move * move_w * (g_mu * rep[d] + l2 * 0.10 * m_w_mu[d]);
         m_w_logv[d] -= lr_move * move_w * (g_logv * rep[d] + l2 * 0.10 * m_w_logv[d]);
         m_w_q25[d] -= lr_move * move_w * (g_q25 * rep[d] + l2 * 0.10 * m_w_q25[d]);
         m_w_q75[d] -= lr_move * move_w * (g_q75 * rep[d] + l2 * 0.10 * m_w_q75[d]);

         g_rep[d] += move_w * (g_mu * wmu_old + 0.35 * g_logv * wlv_old + 0.20 * g_q25 * w25_old + 0.20 * g_q75 * w75_old);
      }

      // Gradient clipping on shared representation gradient.
      double gnorm2 = 0.0;
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
         gnorm2 += g_rep[d] * g_rep[d];
      double gnorm = MathSqrt(gnorm2);
      if(gnorm > 3.0)
      {
         double s = 3.0 / MathMax(gnorm, 1e-9);
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            g_rep[d] *= s;
      }

      // Update patch embedding + channel gates.
      double lr_patch = AdamGroupLR(2, gnorm, base_lr * 0.45);
      for(int d=0; d<FX6_PTST_D_MODEL; d++)
      {
         for(int f=0; f<FX6_AI_FEATURES; f++)
         {
            for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
            {
               double grad = (g_rep[d] * patch_stat[f][t] / (double)MathMax(token_count, 1)) + l2 * 0.10 * m_w_patch[d][f][t];
               m_w_patch[d][f][t] -= lr_patch * move_w * grad;
            }
         }
         m_b_patch[d] -= lr_patch * move_w * g_rep[d] * 0.15;
      }

      for(int f=0; f<FX6_AI_FEATURES; f++)
      {
         double pm = 0.0;
         for(int t=0; t<FX6_PTST_PATCH_LEN; t++)
            pm += patch_stat[f][t];
         pm /= (double)FX6_PTST_PATCH_LEN;

         double gf = 0.0;
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
            gf += g_rep[d] * pm;

         m_ch_gate[f] -= lr_patch * move_w * (0.25 * gf + l2 * 0.02 * (m_ch_gate[f] - 1.0));
         m_ch_gate[f] = FX6_Clamp(m_ch_gate[f], 0.10, 4.00);
      }

      // Update positional embeddings with recency focus.
      double lr_pos = AdamGroupLR(3, gnorm, base_lr * 0.20);
      for(int p=0; p<token_count; p++)
      {
         double rw = (p == token_count - 1 ? 0.40 : 0.12);
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            double grad = rw * g_rep[d] / (double)MathMax(token_count, 1);
            m_pos[p][d] -= lr_pos * move_w * (grad + l2 * 0.02 * m_pos[p][d]);
         }
      }

      // Encoder-weight updates using layer summary statistics.
      for(int l=0; l<FX6_PTST_LAYERS; l++)
      {
         double lr_enc = AdamGroupLR(4 + l, gnorm, base_lr * 0.25);

         // Output projection from attention contexts.
         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            for(int od=0; od<FX6_PTST_D_MODEL; od++)
            {
               double grad = g_rep[d] * layer_ctx_mean[l][od] + l2 * 0.05 * m_wo[l][d][od];
               m_wo[l][d][od] -= lr_enc * move_w * grad;
            }
         }

         // FFN2.
         double dff[FX6_PTST_D_FF];
         for(int r=0; r<FX6_PTST_D_FF; r++) dff[r] = 0.0;

         for(int d=0; d<FX6_PTST_D_MODEL; d++)
         {
            m_bff2[l][d] -= lr_enc * move_w * g_rep[d] * 0.30;
            for(int r=0; r<FX6_PTST_D_FF; r++)
            {
               double grad = g_rep[d] * layer_ff_mean[l][r] + l2 * 0.05 * m_wff2[l][d][r];
               m_wff2[l][d][r] -= lr_enc * move_w * grad;
               dff[r] += g_rep[d] * m_wff2[l][d][r];
            }
         }

         // FFN1 (approximate backprop through GELU).
         for(int r=0; r<FX6_PTST_D_FF; r++)
         {
            double act = layer_ff_mean[l][r];
            double dg = GELUDerivApprox(act);
            double dr = dff[r] * dg;
            m_bff1[l][r] -= lr_enc * move_w * dr * 0.25;
            for(int d=0; d<FX6_PTST_D_MODEL; d++)
            {
               double grad = dr * layer_in_mean[l][d] + l2 * 0.05 * m_wff1[l][r][d];
               m_wff1[l][r][d] -= lr_enc * move_w * grad;
            }
         }

         // Q/K/V tiny corrective step keeps attention adaptable without heavy backprop.
         for(int hdx=0; hdx<FX6_PTST_HEADS; hdx++)
         {
            for(int dh=0; dh<FX6_PTST_D_HEAD; dh++)
            {
               for(int d=0; d<FX6_PTST_D_MODEL; d++)
               {
                  double corr = g_rep[d] * layer_in_mean[l][d] * 0.015;
                  m_wq[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wq[l][hdx][dh][d]);
                  m_wk[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wk[l][hdx][dh][d]);
                  m_wv[l][hdx][dh][d] -= lr_enc * move_w * (corr + l2 * 0.02 * m_wv[l][hdx][dh][d]);
               }
            }
         }
      }

      // Update shared move estimators in base plugin.
      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, sample_w);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double expected_move = 0.0;
      if(!PredictNativeClassProbs(x, hp, probs, expected_move))
         return 0.5;

      double den = probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FX6_Clamp(probs[(int)FX6_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = -1.0;
      if(PredictNativeClassProbs(x, hp, probs, ev) && ev > 0.0)
         return ev;

      return CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FX6_AI_PATCHTST_MQH__
