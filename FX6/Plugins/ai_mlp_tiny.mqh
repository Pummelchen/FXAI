#ifndef __FX6_AI_MLP_TINY_MQH__
#define __FX6_AI_MLP_TINY_MQH__

#include "..\plugin_base.mqh"

#define FX6_MLP_H1 16
#define FX6_MLP_H2 16
#define FX6_MLP_CTX 6
#define FX6_MLP_CLASSES 3
#define FX6_MLP_HIST 8

class CFX6AIMLPTiny : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   bool   m_shadow_ready;
   int    m_step;
   int    m_adam_step;
   int    m_hist_len;
   int    m_hist_ptr;

   double m_shadow_decay;

   // Core network.
   double m_w1[FX6_MLP_H1][FX6_AI_WEIGHTS];
   double m_w1c[FX6_MLP_H1][FX6_MLP_CTX];
   double m_b1[FX6_MLP_H1];

   double m_w2[FX6_MLP_H2][FX6_MLP_H1];
   double m_b2[FX6_MLP_H2];

   // 3-class head.
   double m_w_cls[FX6_MLP_CLASSES][FX6_MLP_H2];
   double m_b_cls[FX6_MLP_CLASSES];

   // Distributional move heads (absolute move points).
   double m_w_mu[FX6_MLP_H2];
   double m_b_mu;
   double m_w_logv[FX6_MLP_H2];
   double m_b_logv;
   double m_w_q25[FX6_MLP_H2];
   double m_b_q25;
   double m_w_q75[FX6_MLP_H2];
   double m_b_q75;

   // AdamW moments.
   double m_m_w1[FX6_MLP_H1][FX6_AI_WEIGHTS], m_v_w1[FX6_MLP_H1][FX6_AI_WEIGHTS];
   double m_m_w1c[FX6_MLP_H1][FX6_MLP_CTX],   m_v_w1c[FX6_MLP_H1][FX6_MLP_CTX];
   double m_m_b1[FX6_MLP_H1], m_v_b1[FX6_MLP_H1];

   double m_m_w2[FX6_MLP_H2][FX6_MLP_H1], m_v_w2[FX6_MLP_H2][FX6_MLP_H1];
   double m_m_b2[FX6_MLP_H2], m_v_b2[FX6_MLP_H2];

   double m_m_w_cls[FX6_MLP_CLASSES][FX6_MLP_H2], m_v_w_cls[FX6_MLP_CLASSES][FX6_MLP_H2];
   double m_m_b_cls[FX6_MLP_CLASSES], m_v_b_cls[FX6_MLP_CLASSES];

   double m_m_w_mu[FX6_MLP_H2], m_v_w_mu[FX6_MLP_H2];
   double m_m_b_mu, m_v_b_mu;
   double m_m_w_logv[FX6_MLP_H2], m_v_w_logv[FX6_MLP_H2];
   double m_m_b_logv, m_v_b_logv;
   double m_m_w_q25[FX6_MLP_H2], m_v_w_q25[FX6_MLP_H2];
   double m_m_b_q25, m_v_b_q25;
   double m_m_w_q75[FX6_MLP_H2], m_v_w_q75[FX6_MLP_H2];
   double m_m_b_q75, m_v_b_q75;

   // EMA shadow params for stable inference.
   double m_sw1[FX6_MLP_H1][FX6_AI_WEIGHTS];
   double m_sw1c[FX6_MLP_H1][FX6_MLP_CTX];
   double m_sb1[FX6_MLP_H1];

   double m_sw2[FX6_MLP_H2][FX6_MLP_H1];
   double m_sb2[FX6_MLP_H2];

   double m_sw_cls[FX6_MLP_CLASSES][FX6_MLP_H2];
   double m_sb_cls[FX6_MLP_CLASSES];

   double m_sw_mu[FX6_MLP_H2];
   double m_sb_mu;
   double m_sw_logv[FX6_MLP_H2];
   double m_sb_logv;
   double m_sw_q25[FX6_MLP_H2];
   double m_sb_q25;
   double m_sw_q75[FX6_MLP_H2];
   double m_sb_q75;

   // Temporal context ring buffer.
   double m_hist_x[FX6_MLP_HIST][FX6_AI_WEIGHTS];

   // Class-balance EMA.
   double m_class_ema[FX6_MLP_CLASSES];

   // Plugin-local directional calibration.
   double m_dir_scale;
   double m_dir_bias;
   double m_dir_temp;

   int HistIndex(const int back) const
   {
      if(m_hist_len <= 0) return -1;
      int b = back;
      if(b < 0) b = 0;
      int max_back = m_hist_len - 1;
      if(max_back < 0) max_back = 0;
      if(b > max_back) b = max_back;

      int idx = m_hist_ptr - 1 - b;
      while(idx < 0) idx += FX6_MLP_HIST;
      while(idx >= FX6_MLP_HIST) idx -= FX6_MLP_HIST;
      return idx;
   }

   void ResetHistory(void)
   {
      m_hist_len = 0;
      m_hist_ptr = 0;
      for(int t=0; t<FX6_MLP_HIST; t++)
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_hist_x[t][i] = 0.0;
   }

   void PushHistory(const double &x[])
   {
      int p = m_hist_ptr;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_hist_x[p][i] = x[i];

      m_hist_ptr++;
      if(m_hist_ptr >= FX6_MLP_HIST) m_hist_ptr = 0;
      if(m_hist_len < FX6_MLP_HIST) m_hist_len++;
   }

   void BuildTemporalContext(const double &x[],
                             double &ctx[]) const
   {
      for(int i=0; i<FX6_MLP_CTX; i++) ctx[i] = 0.0;
      if(m_hist_len <= 0)
      {
         ctx[4] = FX6_Clamp(MathAbs(x[7]), 0.0, 8.0);
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

      ctx[0] = x[1] - x1_prev;
      ctx[1] = x[1] - m_hist_x[i2][1];
      ctx[2] = x[1] - m_hist_x[i4][1];

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
      ctx[3] = vol;

      ctx[4] = MathAbs(x[7]); // spread/cost-aware feature channel.

      double regime = MathAbs(x[2] - x2_prev) + 0.5 * MathAbs(x[3] - x3_prev);
      regime += 0.35 * MathAbs(x[1] - x1_prev);
      ctx[5] = regime;

      for(int i=0; i<FX6_MLP_CTX; i++)
         ctx[i] = FX6_ClipSym(ctx[i], 8.0);
   }

   int ResolveClass(const int y,
                    const double &x[],
                    const double move_points) const
   {
      if(y == (int)FX6_LABEL_SELL || y == (int)FX6_LABEL_BUY || y == (int)FX6_LABEL_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return (int)FX6_LABEL_SKIP;

      if(y > 0) return (int)FX6_LABEL_BUY;
      if(y == 0) return (int)FX6_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
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
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);

      double tot = m_class_ema[0] + m_class_ema[1] + m_class_ema[2];
      if(tot <= 0.0) tot = 3.0;
      double cnt = m_class_ema[cls];
      if(cnt < 1e-6) cnt = 1e-6;
      double balance = tot / (3.0 * cnt);
      balance = FX6_Clamp(balance, 0.60, 2.60);

      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);
      double edge_scale = (edge > 0.0 ? 1.0 + 0.04 * MathMin(edge, 20.0) : 0.75);

      if(cls == (int)FX6_LABEL_SKIP)
      {
         if(edge <= 0.0) edge_scale *= 1.20;
         else            edge_scale *= 0.80;
      }

      return FX6_Clamp(sw * balance * edge_scale, 0.20, 6.00);
   }

   double MoveWeight(const double move_points,
                     const double cost_points,
                     const double sample_w) const
   {
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double edge_w = FX6_MoveEdgeWeight(move_points, cost_points);
      double edge = MathMax(0.0, MathAbs(move_points) - MathMax(cost_points, 0.0));
      double scale = 1.0 + 0.05 * MathMin(edge, 20.0);
      return FX6_Clamp(sw * edge_w * scale, 0.20, 8.00);
   }

   double LocalDirCalibrate(const double p_raw) const
   {
      double p = FX6_Clamp(p_raw, 0.001, 0.999);
      double z = FX6_Logit(p);
      double t = FX6_Clamp(m_dir_temp, 0.50, 3.00);
      double pc = FX6_Sigmoid((m_dir_scale * z / t) + m_dir_bias);
      return FX6_Clamp(pc, 0.001, 0.999);
   }

   void UpdateLocalDirCalib(const double p_raw,
                            const int y_dir,
                            const double sample_w)
   {
      double p = FX6_Clamp(p_raw, 0.001, 0.999);
      double z = FX6_Logit(p);
      double t = FX6_Clamp(m_dir_temp, 0.50, 3.00);
      double pc = FX6_Sigmoid((m_dir_scale * z / t) + m_dir_bias);
      double e = ((double)y_dir - pc);

      double lr = FX6_Clamp(0.010 * sample_w, 0.0010, 0.0300);
      m_dir_scale = FX6_Clamp(m_dir_scale + lr * (e * z / t - 0.001 * (m_dir_scale - 1.0)), 0.20, 6.00);
      m_dir_bias  = FX6_Clamp(m_dir_bias + lr * e, -6.0, 6.0);
      m_dir_temp  = FX6_Clamp(m_dir_temp + lr * (0.50 * e * (MathAbs(z) - 1.0)), 0.50, 3.00);
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

   double ScheduledLR(const FX6AIHyperParams &hp,
                      const double sample_w) const
   {
      double base = FX6_Clamp(hp.mlp_lr, 0.00005, 1.00000);
      double st = (double)MathMax(m_step, 1);

      double warm = FX6_Clamp(st / 128.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0010 * MathMax(0.0, st - 128.0));

      double period = 2048.0;
      double phase = MathMod(st, period) / period;
      double cosine = 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
      double cosine_floor = 0.20 + 0.80 * cosine;

      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double sw_scale = FX6_Clamp(0.80 + 0.20 * MathSqrt(sw), 0.70, 1.60);

      double lr = base * warm * invsqrt * cosine_floor * sw_scale;
      return FX6_Clamp(lr, 0.00001, 0.08000);
   }

   void AdamWStep(double &p,
                  double &m,
                  double &v,
                  const double g,
                  const double lr,
                  const double wd)
   {
      const double beta1 = 0.90;
      const double beta2 = 0.999;
      const double eps = 1e-8;

      double grad = FX6_ClipSym(g, 10.0);
      m = beta1 * m + (1.0 - beta1) * grad;
      v = beta2 * v + (1.0 - beta2) * grad * grad;

      double t = (double)MathMax(m_adam_step, 1);
      double mhat = m / (1.0 - MathPow(beta1, t));
      double vhat = v / (1.0 - MathPow(beta2, t));

      p -= lr * (mhat / (MathSqrt(vhat) + eps));
      if(wd > 0.0)
         p -= lr * wd * p;
   }

   void InitMoments(void)
   {
      for(int h=0; h<FX6_MLP_H1; h++)
      {
         m_m_b1[h] = 0.0;
         m_v_b1[h] = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_m_w1[h][i] = 0.0;
            m_v_w1[h][i] = 0.0;
         }
         for(int c=0; c<FX6_MLP_CTX; c++)
         {
            m_m_w1c[h][c] = 0.0;
            m_v_w1c[h][c] = 0.0;
         }
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         m_m_b2[h] = 0.0;
         m_v_b2[h] = 0.0;

         m_m_w_mu[h] = 0.0;
         m_v_w_mu[h] = 0.0;
         m_m_w_logv[h] = 0.0;
         m_v_w_logv[h] = 0.0;
         m_m_w_q25[h] = 0.0;
         m_v_w_q25[h] = 0.0;
         m_m_w_q75[h] = 0.0;
         m_v_w_q75[h] = 0.0;

         for(int j=0; j<FX6_MLP_H1; j++)
         {
            m_m_w2[h][j] = 0.0;
            m_v_w2[h][j] = 0.0;
         }
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         m_m_b_cls[c] = 0.0;
         m_v_b_cls[c] = 0.0;
         for(int h=0; h<FX6_MLP_H2; h++)
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
      double s = FX6_Clamp(init_scale, 0.01, 0.60);

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double a = (double)((h + 1) * (i + 3));
            m_w1[h][i] = 0.04 * s * MathSin(0.87 * a);
         }
         for(int c=0; c<FX6_MLP_CTX; c++)
         {
            double b = (double)((h + 2) * (c + 5));
            m_w1c[h][c] = 0.05 * s * MathCos(0.79 * b);
         }
         m_b1[h] = 0.0;
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         for(int j=0; j<FX6_MLP_H1; j++)
         {
            double a = (double)((h + 3) * (j + 2));
            m_w2[h][j] = 0.035 * s * MathCos(0.93 * a);
         }
         m_b2[h] = 0.0;
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         m_b_cls[c] = 0.0;
         for(int h=0; h<FX6_MLP_H2; h++)
         {
            double a = (double)((c + 2) * (h + 1));
            m_w_cls[c][h] = 0.04 * s * MathSin(1.03 * a);
         }
      }

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.5;
      for(int h=0; h<FX6_MLP_H2; h++)
      {
         m_w_mu[h] = 0.03 * s * MathCos((double)(h + 1) * 1.11);
         m_w_logv[h] = 0.02 * s * MathSin((double)(h + 1) * 1.07);
         m_w_q25[h] = 0.02 * s * MathSin((double)(h + 1) * 1.17);
         m_w_q75[h] = 0.02 * s * MathCos((double)(h + 1) * 1.19);
      }

      InitMoments();

      m_shadow_ready = false;
      m_shadow_decay = 0.995;
      SyncShadow(true);

      m_initialized = true;
   }

   void SyncShadow(const bool hard_copy)
   {
      double a = (hard_copy ? 0.0 : FX6_Clamp(m_shadow_decay, 0.90, 0.9999));
      double b = (hard_copy ? 1.0 : (1.0 - a));

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         m_sb1[h] = (hard_copy ? m_b1[h] : a * m_sb1[h] + b * m_b1[h]);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_sw1[h][i] = (hard_copy ? m_w1[h][i] : a * m_sw1[h][i] + b * m_w1[h][i]);
         for(int c=0; c<FX6_MLP_CTX; c++)
            m_sw1c[h][c] = (hard_copy ? m_w1c[h][c] : a * m_sw1c[h][c] + b * m_w1c[h][c]);
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         m_sb2[h] = (hard_copy ? m_b2[h] : a * m_sb2[h] + b * m_b2[h]);
         m_sw_mu[h] = (hard_copy ? m_w_mu[h] : a * m_sw_mu[h] + b * m_w_mu[h]);
         m_sw_logv[h] = (hard_copy ? m_w_logv[h] : a * m_sw_logv[h] + b * m_w_logv[h]);
         m_sw_q25[h] = (hard_copy ? m_w_q25[h] : a * m_sw_q25[h] + b * m_w_q25[h]);
         m_sw_q75[h] = (hard_copy ? m_w_q75[h] : a * m_sw_q75[h] + b * m_w_q75[h]);

         for(int j=0; j<FX6_MLP_H1; j++)
            m_sw2[h][j] = (hard_copy ? m_w2[h][j] : a * m_sw2[h][j] + b * m_w2[h][j]);
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         m_sb_cls[c] = (hard_copy ? m_b_cls[c] : a * m_sb_cls[c] + b * m_b_cls[c]);
         for(int h=0; h<FX6_MLP_H2; h++)
            m_sw_cls[c][h] = (hard_copy ? m_w_cls[c][h] : a * m_sw_cls[c][h] + b * m_w_cls[c][h]);
      }

      m_sb_mu = (hard_copy ? m_b_mu : a * m_sb_mu + b * m_b_mu);
      m_sb_logv = (hard_copy ? m_b_logv : a * m_sb_logv + b * m_b_logv);
      m_sb_q25 = (hard_copy ? m_b_q25 : a * m_sb_q25 + b * m_b_q25);
      m_sb_q75 = (hard_copy ? m_b_q75 : a * m_sb_q75 + b * m_b_q75);
   }

   void Forward(const double &x[],
                const double &ctx[],
                const bool use_shadow,
                const bool training,
                double &h1_raw[],
                double &h1_out[],
                double &h2_out[],
                double &drop_mask[],
                double &logits[],
                double &probs[],
                double &mu,
                double &logv,
                double &q25,
                double &q75) const
   {
      double drop_rate = (training ? 0.10 : 0.0);

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         double s = (use_shadow ? m_sb1[h] : m_b1[h]);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            s += (use_shadow ? m_sw1[h][i] : m_w1[h][i]) * x[i];
         for(int c=0; c<FX6_MLP_CTX; c++)
            s += (use_shadow ? m_sw1c[h][c] : m_w1c[h][c]) * ctx[c];

         double a = FX6_Tanh(FX6_ClipSym(s, 8.0));
         double m = (training ? DropMask(h, 17, drop_rate) : 1.0);

         h1_raw[h] = a;
         drop_mask[h] = m;
         h1_out[h] = a * m;
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         double s = (use_shadow ? m_sb2[h] : m_b2[h]);
         for(int j=0; j<FX6_MLP_H1; j++)
            s += (use_shadow ? m_sw2[h][j] : m_w2[h][j]) * h1_out[j];
         h2_out[h] = FX6_Tanh(FX6_ClipSym(s, 8.0));
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         double z = (use_shadow ? m_sb_cls[c] : m_b_cls[c]);
         for(int h=0; h<FX6_MLP_H2; h++)
            z += (use_shadow ? m_sw_cls[c][h] : m_w_cls[c][h]) * h2_out[h];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = (use_shadow ? m_sb_mu : m_b_mu);
      logv = (use_shadow ? m_sb_logv : m_b_logv);
      q25 = (use_shadow ? m_sb_q25 : m_b_q25);
      q75 = (use_shadow ? m_sb_q75 : m_b_q75);
      for(int h=0; h<FX6_MLP_H2; h++)
      {
         mu += (use_shadow ? m_sw_mu[h] : m_w_mu[h]) * h2_out[h];
         logv += (use_shadow ? m_sw_logv[h] : m_w_logv[h]) * h2_out[h];
         q25 += (use_shadow ? m_sw_q25[h] : m_w_q25[h]) * h2_out[h];
         q75 += (use_shadow ? m_sw_q75[h] : m_w_q75[h]) * h2_out[h];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double tmp = q25;
         q25 = q75;
         q75 = tmp;
      }
   }

   void UpdateWeighted(const int y,
                       const double &x[],
                       const FX6AIHyperParams &hp,
                       const double sample_w,
                       const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;
      m_adam_step++;

      int cls = ResolveClass(y, x, move_points);
      if(cls < 0) cls = 0;
      if(cls >= FX6_MLP_CLASSES) cls = FX6_MLP_CLASSES - 1;

      for(int k=0; k<FX6_MLP_CLASSES; k++)
         m_class_ema[k] = 0.995 * m_class_ema[k] + (k == cls ? 0.005 : 0.0);

      double cost = InputCostProxyPoints(x);
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double lr = ScheduledLR(hp, sw);
      double wd = FX6_Clamp(hp.mlp_l2, 0.0, 0.0500);

      double ctx[FX6_MLP_CTX];
      BuildTemporalContext(x, ctx);

      double h1_raw[FX6_MLP_H1], h1[FX6_MLP_H1], h2[FX6_MLP_H2], drop1[FX6_MLP_H1];
      double logits[FX6_MLP_CLASSES], probs[FX6_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x, ctx, false, true, h1_raw, h1, h2, drop1, logits, probs, mu, logv, q25, q75);

      double cls_w = ClassWeight(cls, move_points, cost, sw);
      double mv_w = MoveWeight(move_points, cost, sw);

      double pt = FX6_Clamp(probs[cls], 0.001, 0.999);
      double focal = MathPow(FX6_Clamp(1.0 - pt, 0.02, 1.0), 1.50);

      double dlogits[FX6_MLP_CLASSES];
      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         double yk = (c == cls ? 1.0 : 0.0);
         dlogits[c] = (probs[c] - yk) * cls_w * focal;
      }

      double target = MathAbs(move_points);
      double err = mu - target;
      double var = MathExp(logv);
      var = FX6_Clamp(var, 0.05, 100.0);

      double dmu = FX6_ClipSym((HuberGrad(err, 6.0) / MathMax(var, 0.25)) * mv_w, 5.0);
      double dlogv = FX6_ClipSym(0.5 * mv_w * (1.0 - (err * err) / MathMax(var, 0.25)), 4.0);
      double dq25 = FX6_ClipSym(PinballGrad(target, q25, 0.25) * 0.25 * mv_w, 3.0);
      double dq75 = FX6_ClipSym(PinballGrad(target, q75, 0.75) * 0.25 * mv_w, 3.0);
      if(q25 > q75)
      {
         double pen = 0.25 * FX6_ClipSym(q25 - q75, 4.0);
         dq25 += pen;
         dq75 -= pen;
      }

      // Gradient accumulators.
      double g_w1[FX6_MLP_H1][FX6_AI_WEIGHTS];
      double g_w1c[FX6_MLP_H1][FX6_MLP_CTX];
      double g_b1[FX6_MLP_H1];

      double g_w2[FX6_MLP_H2][FX6_MLP_H1];
      double g_b2[FX6_MLP_H2];

      double g_w_cls[FX6_MLP_CLASSES][FX6_MLP_H2];
      double g_b_cls[FX6_MLP_CLASSES];

      double g_w_mu[FX6_MLP_H2], g_w_logv[FX6_MLP_H2], g_w_q25[FX6_MLP_H2], g_w_q75[FX6_MLP_H2];
      double g_b_mu = dmu, g_b_logv = dlogv, g_b_q25 = dq25, g_b_q75 = dq75;

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         g_b1[h] = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g_w1[h][i] = 0.0;
         for(int c=0; c<FX6_MLP_CTX; c++) g_w1c[h][c] = 0.0;
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         g_b2[h] = 0.0;
         g_w_mu[h] = dmu * h2[h];
         g_w_logv[h] = dlogv * h2[h];
         g_w_q25[h] = dq25 * h2[h];
         g_w_q75[h] = dq75 * h2[h];

         for(int j=0; j<FX6_MLP_H1; j++) g_w2[h][j] = 0.0;
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         g_b_cls[c] = dlogits[c];
         for(int h=0; h<FX6_MLP_H2; h++)
            g_w_cls[c][h] = dlogits[c] * h2[h];
      }

      // Backprop to hidden layer 2.
      double dh2[FX6_MLP_H2];
      for(int h=0; h<FX6_MLP_H2; h++)
      {
         double d = dmu * m_w_mu[h] + dlogv * m_w_logv[h] + dq25 * m_w_q25[h] + dq75 * m_w_q75[h];
         for(int c=0; c<FX6_MLP_CLASSES; c++)
            d += dlogits[c] * m_w_cls[c][h];
         dh2[h] = d;
      }

      double dz2[FX6_MLP_H2];
      double dh1[FX6_MLP_H1];
      for(int h=0; h<FX6_MLP_H1; h++) dh1[h] = 0.0;

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         dz2[h] = dh2[h] * (1.0 - h2[h] * h2[h]);
         g_b2[h] += dz2[h];

         for(int j=0; j<FX6_MLP_H1; j++)
         {
            g_w2[h][j] += dz2[h] * h1[j];
            dh1[j] += dz2[h] * m_w2[h][j];
         }
      }

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         dh1[h] *= drop1[h];
         double dz1 = dh1[h] * (1.0 - h1_raw[h] * h1_raw[h]);
         g_b1[h] += dz1;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g_w1[h][i] += dz1 * x[i];
         for(int c=0; c<FX6_MLP_CTX; c++)
            g_w1c[h][c] += dz1 * ctx[c];
      }

      // Per-layer clipping.
      for(int h=0; h<FX6_MLP_H1; h++)
      {
         g_b1[h] = FX6_ClipSym(g_b1[h], 3.0);
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g_w1[h][i] = FX6_ClipSym(g_w1[h][i], 3.0);
         for(int c=0; c<FX6_MLP_CTX; c++) g_w1c[h][c] = FX6_ClipSym(g_w1c[h][c], 3.0);
      }
      for(int h=0; h<FX6_MLP_H2; h++)
      {
         g_b2[h] = FX6_ClipSym(g_b2[h], 3.5);
         g_w_mu[h] = FX6_ClipSym(g_w_mu[h], 4.0);
         g_w_logv[h] = FX6_ClipSym(g_w_logv[h], 4.0);
         g_w_q25[h] = FX6_ClipSym(g_w_q25[h], 4.0);
         g_w_q75[h] = FX6_ClipSym(g_w_q75[h], 4.0);
         for(int j=0; j<FX6_MLP_H1; j++) g_w2[h][j] = FX6_ClipSym(g_w2[h][j], 3.5);
      }
      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         g_b_cls[c] = FX6_ClipSym(g_b_cls[c], 4.0);
         for(int h=0; h<FX6_MLP_H2; h++) g_w_cls[c][h] = FX6_ClipSym(g_w_cls[c][h], 4.0);
      }
      g_b_mu = FX6_ClipSym(g_b_mu, 4.0);
      g_b_logv = FX6_ClipSym(g_b_logv, 4.0);
      g_b_q25 = FX6_ClipSym(g_b_q25, 4.0);
      g_b_q75 = FX6_ClipSym(g_b_q75, 4.0);

      // Global norm clip.
      double g2 = g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int h=0; h<FX6_MLP_H1; h++)
      {
         g2 += g_b1[h] * g_b1[h];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g2 += g_w1[h][i] * g_w1[h][i];
         for(int c=0; c<FX6_MLP_CTX; c++) g2 += g_w1c[h][c] * g_w1c[h][c];
      }
      for(int h=0; h<FX6_MLP_H2; h++)
      {
         g2 += g_b2[h] * g_b2[h] + g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];
         for(int j=0; j<FX6_MLP_H1; j++) g2 += g_w2[h][j] * g_w2[h][j];
      }
      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         g2 += g_b_cls[c] * g_b_cls[c];
         for(int h=0; h<FX6_MLP_H2; h++) g2 += g_w_cls[c][h] * g_w_cls[c][h];
      }

      double gnorm = MathSqrt(g2 + 1e-12);
      double gscale = (gnorm > 3.20 ? (3.20 / gnorm) : 1.0);

      // AdamW updates.
      for(int h=0; h<FX6_MLP_H1; h++)
      {
         AdamWStep(m_b1[h], m_m_b1[h], m_v_b1[h], gscale * g_b1[h], lr, 0.0);
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            AdamWStep(m_w1[h][i], m_m_w1[h][i], m_v_w1[h][i], gscale * g_w1[h][i], lr, wd);
         for(int c=0; c<FX6_MLP_CTX; c++)
            AdamWStep(m_w1c[h][c], m_m_w1c[h][c], m_v_w1c[h][c], gscale * g_w1c[h][c], lr, wd);
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         AdamWStep(m_b2[h], m_m_b2[h], m_v_b2[h], gscale * g_b2[h], lr, 0.0);
         for(int j=0; j<FX6_MLP_H1; j++)
            AdamWStep(m_w2[h][j], m_m_w2[h][j], m_v_w2[h][j], gscale * g_w2[h][j], lr, wd);

         AdamWStep(m_w_mu[h],   m_m_w_mu[h],   m_v_w_mu[h],   gscale * g_w_mu[h],   lr, wd);
         AdamWStep(m_w_logv[h], m_m_w_logv[h], m_v_w_logv[h], gscale * g_w_logv[h], lr, wd);
         AdamWStep(m_w_q25[h],  m_m_w_q25[h],  m_v_w_q25[h],  gscale * g_w_q25[h],  lr, wd);
         AdamWStep(m_w_q75[h],  m_m_w_q75[h],  m_v_w_q75[h],  gscale * g_w_q75[h],  lr, wd);
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         AdamWStep(m_b_cls[c], m_m_b_cls[c], m_v_b_cls[c], gscale * g_b_cls[c], lr, 0.0);
         for(int h=0; h<FX6_MLP_H2; h++)
            AdamWStep(m_w_cls[c][h], m_m_w_cls[c][h], m_v_w_cls[c][h], gscale * g_w_cls[c][h], lr, wd);
      }

      AdamWStep(m_b_mu,   m_m_b_mu,   m_v_b_mu,   gscale * g_b_mu,   lr, 0.0);
      AdamWStep(m_b_logv, m_m_b_logv, m_v_b_logv, gscale * g_b_logv, lr, 0.0);
      AdamWStep(m_b_q25,  m_m_b_q25,  m_v_b_q25,  gscale * g_b_q25,  lr, 0.0);
      AdamWStep(m_b_q75,  m_m_b_q75,  m_v_b_q75,  gscale * g_b_q75,  lr, 0.0);

      m_b_logv = FX6_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         m_w_mu[h] = FX6_ClipSym(m_w_mu[h], 6.0);
         m_w_logv[h] = FX6_ClipSym(m_w_logv[h], 6.0);
         m_w_q25[h] = FX6_ClipSym(m_w_q25[h], 6.0);
         m_w_q75[h] = FX6_ClipSym(m_w_q75[h], 6.0);
         for(int c=0; c<FX6_MLP_CLASSES; c++)
            m_w_cls[c][h] = FX6_ClipSym(m_w_cls[c][h], 6.0);
      }

      SyncShadow(false);
      if(!m_shadow_ready && m_step >= 64)
         m_shadow_ready = true;

      // Directional calibration updates.
      double den = probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[(int)FX6_LABEL_BUY] / den;
      int y_dir = (cls == (int)FX6_LABEL_BUY ? 1 : 0);
      if(cls == (int)FX6_LABEL_SKIP)
         y_dir = (move_points >= 0.0 ? 1 : 0);

      UpdateLocalDirCalib(p_dir_raw, y_dir, cls_w);
      double p_dir_local = LocalDirCalibrate(p_dir_raw);
      if(cls == (int)FX6_LABEL_SKIP)
         UpdateCalibration(p_dir_local, y_dir, 0.25 * cls_w);
      else
         UpdateCalibration(p_dir_local, y_dir, cls_w);

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hp, sw);

      PushHistory(x);
   }

public:
   CFX6AIMLPTiny(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_MLP_TINY; }
   virtual string AIName(void) const { return "mlp_tiny"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_shadow_ready = false;
      m_step = 0;
      m_adam_step = 0;
      m_shadow_decay = 0.995;

      ResetHistory();

      for(int c=0; c<FX6_MLP_CLASSES; c++)
         m_class_ema[c] = 1.0;

      m_dir_scale = 1.0;
      m_dir_bias = 0.0;
      m_dir_temp = 1.0;

      for(int h=0; h<FX6_MLP_H1; h++)
      {
         m_b1[h] = 0.0;
         m_sb1[h] = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_w1[h][i] = 0.0;
            m_sw1[h][i] = 0.0;
         }
         for(int c=0; c<FX6_MLP_CTX; c++)
         {
            m_w1c[h][c] = 0.0;
            m_sw1c[h][c] = 0.0;
         }
      }

      for(int h=0; h<FX6_MLP_H2; h++)
      {
         m_b2[h] = 0.0;
         m_sb2[h] = 0.0;

         m_w_mu[h] = 0.0; m_sw_mu[h] = 0.0;
         m_w_logv[h] = 0.0; m_sw_logv[h] = 0.0;
         m_w_q25[h] = 0.0; m_sw_q25[h] = 0.0;
         m_w_q75[h] = 0.0; m_sw_q75[h] = 0.0;

         for(int j=0; j<FX6_MLP_H1; j++)
         {
            m_w2[h][j] = 0.0;
            m_sw2[h][j] = 0.0;
         }
      }

      for(int c=0; c<FX6_MLP_CLASSES; c++)
      {
         m_b_cls[c] = 0.0;
         m_sb_cls[c] = 0.0;
         for(int h=0; h<FX6_MLP_H2; h++)
         {
            m_w_cls[c][h] = 0.0;
            m_sw_cls[c][h] = 0.0;
         }
      }

      m_b_mu = 0.0; m_sb_mu = 0.0;
      m_b_logv = 0.0; m_sb_logv = 0.0;
      m_b_q25 = 0.0; m_sb_q25 = 0.0;
      m_b_q75 = 0.0; m_sb_q75 = 0.0;

      InitMoments();
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights(hp.mlp_init);
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
      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(x, move_points);
      UpdateWeighted(y, x, h, w, move_points);
   }

   virtual double PredictProb(const double &x[],
                              const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double ctx[FX6_MLP_CTX];
      BuildTemporalContext(x, ctx);

      double h1_raw[FX6_MLP_H1], h1[FX6_MLP_H1], h2[FX6_MLP_H2], drop1[FX6_MLP_H1];
      double logits[FX6_MLP_CLASSES], probs[FX6_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x, ctx, m_shadow_ready, false, h1_raw, h1, h2, drop1, logits, probs, mu, logv, q25, q75);

      double den = probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[(int)FX6_LABEL_BUY] / den;
      double p_dir_local = LocalDirCalibrate(p_dir_raw);
      double p_dir_cal = CalibrateProb(p_dir_local);

      double p_up = p_dir_cal * FX6_Clamp(1.0 - probs[(int)FX6_LABEL_SKIP], 0.0, 1.0);
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[],
                                            const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double ctx[FX6_MLP_CTX];
      BuildTemporalContext(x, ctx);

      double h1_raw[FX6_MLP_H1], h1[FX6_MLP_H1], h2[FX6_MLP_H2], drop1[FX6_MLP_H1];
      double logits[FX6_MLP_CLASSES], probs[FX6_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x, ctx, m_shadow_ready, false, h1_raw, h1, h2, drop1, logits, probs, mu, logv, q25, q75);

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr);
      double active = FX6_Clamp(1.0 - probs[(int)FX6_LABEL_SKIP], 0.0, 1.0);
      double ev = amp * active;

      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      if(ev > 0.0 && base_ev > 0.0) return 0.65 * ev + 0.35 * base_ev;
      if(ev > 0.0) return ev;
      return base_ev;
   }
};

#endif // __FX6_AI_MLP_TINY_MQH__
