// FXAI v1
#ifndef __FX6_AI_FTRL_MQH__
#define __FX6_AI_FTRL_MQH__

#include "..\plugin_base.mqh"

#define FX6_FTRL_CLASS_COUNT 3
#define FX6_FTRL_HASH2_BUCKETS 97
#define FX6_FTRL_CAL_BINS 10

class CFX6AIFTRL : public CFX6AIPlugin
{
private:
   // Main multiclass FTRL parameters.
   double m_z[FX6_FTRL_CLASS_COUNT][FX6_AI_WEIGHTS];
   double m_n[FX6_FTRL_CLASS_COUNT][FX6_AI_WEIGHTS];

   // Two hashed interaction spaces (second space mitigates collisions).
   double m_hz1[FX6_FTRL_CLASS_COUNT][FX6_ENHASH_BUCKETS];
   double m_hn1[FX6_FTRL_CLASS_COUNT][FX6_ENHASH_BUCKETS];
   double m_hz2[FX6_FTRL_CLASS_COUNT][FX6_FTRL_HASH2_BUCKETS];
   double m_hn2[FX6_FTRL_CLASS_COUNT][FX6_FTRL_HASH2_BUCKETS];

   // Runtime weight views rebuilt from z/n.
   double m_w[FX6_FTRL_CLASS_COUNT][FX6_AI_WEIGHTS];
   double m_hw1[FX6_FTRL_CLASS_COUNT][FX6_ENHASH_BUCKETS];
   double m_hw2[FX6_FTRL_CLASS_COUNT][FX6_FTRL_HASH2_BUCKETS];

   // Collision monitoring and rebalancing.
   double m_hload1[FX6_ENHASH_BUCKETS];
   double m_hload2[FX6_FTRL_HASH2_BUCKETS];
   double m_hmean1;
   double m_hmean2;

   // Move-amplitude FTRL head.
   double m_mv_z[FX6_AI_WEIGHTS];
   double m_mv_n[FX6_AI_WEIGHTS];
   double m_mv_hz1[FX6_ENHASH_BUCKETS];
   double m_mv_hn1[FX6_ENHASH_BUCKETS];
   double m_mv_hz2[FX6_FTRL_HASH2_BUCKETS];
   double m_mv_hn2[FX6_FTRL_HASH2_BUCKETS];
   double m_mv_w[FX6_AI_WEIGHTS];
   double m_mv_hw1[FX6_ENHASH_BUCKETS];
   double m_mv_hw2[FX6_FTRL_HASH2_BUCKETS];
   int    m_mv_steps;

   // Online balancing and drift.
   int    m_step;
   double m_cls_ema[FX6_FTRL_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration.
   double m_cal3_temp;
   double m_cal3_bias[FX6_FTRL_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_FTRL_CLASS_COUNT][FX6_FTRL_CAL_BINS];
   double m_cal3_iso_cnt[FX6_FTRL_CLASS_COUNT][FX6_FTRL_CAL_BINS];
   int    m_cal3_steps;

   bool   m_use_hash;
   bool   m_use_hash2;

   int HashIndex(const int i, const int j) const
   {
      uint h = ((uint)(i * 73856093)) ^ ((uint)(j * 19349663));
      return (int)(h % (uint)FX6_ENHASH_BUCKETS);
   }

   int HashIndex2(const int i, const int j) const
   {
      uint h = ((uint)(i * 83492791)) ^ ((uint)(j * 2654435761U));
      return (int)(h % (uint)FX6_FTRL_HASH2_BUCKETS);
   }

   double HashSign(const int i, const int j) const
   {
      int v = (i * 31 + j * 17) & 1;
      return (v == 0 ? 1.0 : -1.0);
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

   void ComputeAdaptiveParams(const FX6AIHyperParams &hp,
                              const double sample_w,
                              double &alpha,
                              double &beta,
                              double &l1,
                              double &l2) const
   {
      double base_alpha = FX6_Clamp(hp.ftrl_alpha, 0.0001, 5.0000);
      double base_beta  = FX6_Clamp(hp.ftrl_beta,  0.0000, 8.0000);
      double base_l1    = FX6_Clamp(hp.ftrl_l1,    0.0000, 0.3000);
      double base_l2    = FX6_Clamp(hp.ftrl_l2,    0.0000, 2.0000);

      double st = (double)MathMax(m_step, 1);
      double warmup = FX6_Clamp(st / 128.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.0025 * MathMax(0.0, st - 128.0));
      double sw = FX6_Clamp(sample_w, 0.25, 6.00);
      double sw_scale = FX6_Clamp(0.80 + 0.20 * sw, 0.70, 1.80);

      alpha = FX6_Clamp(base_alpha * warmup * invsqrt * sw_scale, 0.00005, 5.0000);
      beta  = FX6_Clamp(base_beta * (1.0 + 0.15 * (1.0 - warmup)), 0.0000, 10.0000);
      l1    = base_l1;
      l2    = base_l2;

      if(m_drift_cooldown > 0)
      {
         alpha = FX6_Clamp(alpha * 0.70, 0.00005, 5.0000);
         beta  = FX6_Clamp(beta * 1.30, 0.0000, 10.0000);
         l2    = FX6_Clamp(l2 * 1.20, 0.0000, 2.5000);
      }
   }

   void BuildWeights(const double alpha,
                     const double beta,
                     const double l1,
                     const double l2)
   {
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double zi = m_z[c][i];
            double ni = m_n[c][i];
            double absz = MathAbs(zi);
            if(absz <= l1)
               m_w[c][i] = 0.0;
            else
            {
               double signz = (zi < 0.0 ? -1.0 : 1.0);
               double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
               m_w[c][i] = -(zi - signz * l1) / denom;
            }
         }

         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         {
            double zi = m_hz1[c][i];
            double ni = m_hn1[c][i];
            double absz = MathAbs(zi);
            if(absz <= l1)
               m_hw1[c][i] = 0.0;
            else
            {
               double signz = (zi < 0.0 ? -1.0 : 1.0);
               double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
               m_hw1[c][i] = -(zi - signz * l1) / denom;
            }
         }

         for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
         {
            double zi = m_hz2[c][i];
            double ni = m_hn2[c][i];
            double absz = MathAbs(zi);
            if(absz <= l1)
               m_hw2[c][i] = 0.0;
            else
            {
               double signz = (zi < 0.0 ? -1.0 : 1.0);
               double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
               m_hw2[c][i] = -(zi - signz * l1) / denom;
            }
         }
      }
   }

   void BuildMoveWeights(const double alpha,
                         const double beta,
                         const double l2)
   {
      const double l1 = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double zi = m_mv_z[i];
         double ni = m_mv_n[i];
         double absz = MathAbs(zi);
         if(absz <= l1)
            m_mv_w[i] = 0.0;
         else
         {
            double signz = (zi < 0.0 ? -1.0 : 1.0);
            double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
            m_mv_w[i] = -(zi - signz * l1) / denom;
         }
      }

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
      {
         double zi = m_mv_hz1[i];
         double ni = m_mv_hn1[i];
         double absz = MathAbs(zi);
         if(absz <= l1)
            m_mv_hw1[i] = 0.0;
         else
         {
            double signz = (zi < 0.0 ? -1.0 : 1.0);
            double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
            m_mv_hw1[i] = -(zi - signz * l1) / denom;
         }
      }

      for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
      {
         double zi = m_mv_hz2[i];
         double ni = m_mv_hn2[i];
         double absz = MathAbs(zi);
         if(absz <= l1)
            m_mv_hw2[i] = 0.0;
         else
         {
            double signz = (zi < 0.0 ? -1.0 : 1.0);
            double denom = ((beta + MathSqrt(ni)) / alpha) + l2;
            m_mv_hw2[i] = -(zi - signz * l1) / denom;
         }
      }
   }

   double CollisionRebalance1(const int h1) const
   {
      double overload = MathMax(0.0, m_hload1[h1] - m_hmean1);
      return 1.0 / MathSqrt(1.0 + overload);
   }

   double CollisionRebalance2(const int h2) const
   {
      double overload = MathMax(0.0, m_hload2[h2] - m_hmean2);
      return 1.0 / MathSqrt(1.0 + overload);
   }

   double ScoreRawClass(const double &x[],
                        const int cls) const
   {
      if(cls < 0 || cls >= FX6_FTRL_CLASS_COUNT) return 0.0;
      double z = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         z += m_w[cls][i] * x[i];
      if(!m_use_hash) return z;

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            int hi1 = HashIndex(i, j);
            double hv = HashSign(i, j) * x[i] * x[j];
            z += m_hw1[cls][hi1] * hv;
            if(m_use_hash2)
            {
               int hi2 = HashIndex2(i, j);
               z += 0.70 * m_hw2[cls][hi2] * hv;
            }
         }
      }
      return z;
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_FTRL_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_FTRL_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_FTRL_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_FTRL_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_FTRL_CAL_BINS) bi = FX6_FTRL_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
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
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }

      double p_cal[FX6_FTRL_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.18 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FX6_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_FTRL_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_FTRL_CAL_BINS) bi = FX6_FTRL_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_temp = FX6_Clamp(m_cal3_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void UpdateLossDrift(const double ce_loss)
   {
      if(!m_loss_ready)
      {
         m_loss_fast = ce_loss;
         m_loss_slow = ce_loss;
         m_loss_ready = true;
         return;
      }

      m_loss_fast = 0.90 * m_loss_fast + 0.10 * ce_loss;
      m_loss_slow = 0.995 * m_loss_slow + 0.005 * ce_loss;

      if(m_drift_cooldown > 0) m_drift_cooldown--;
      if(m_step < 256 || m_drift_cooldown > 0) return;

      if(m_loss_fast > 1.7 * MathMax(m_loss_slow, 0.10))
         m_drift_cooldown = 96;
   }

   void UpdateMoveHeadFTRL(const double &x[],
                           const double move_points,
                           const double alpha,
                           const double beta,
                           const double l2,
                           const double sample_w)
   {
      double target = MathAbs(move_points);
      if(!MathIsValidNumber(target)) return;

      double alpha_m = FX6_Clamp(0.70 * alpha, 0.00005, 3.0000);
      double beta_m  = FX6_Clamp(0.80 * beta + 0.05, 0.0000, 10.0000);
      double l2_m    = FX6_Clamp(0.50 * l2 + 0.001, 0.0000, 2.0000);
      BuildMoveWeights(alpha_m, beta_m, l2_m);

      double pred = FX6_DotLinear(m_mv_w, x);
      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               pred += m_mv_hw1[h1] * hv;
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  pred += 0.70 * m_mv_hw2[h2] * hv;
               }
            }
         }
      }
      if(pred < 0.0) pred = 0.0;

      double err = pred - target;
      double delta = 1.0;
      double g = (MathAbs(err) <= delta ? err : FX6_Sign(err) * delta);
      double sw = FX6_Clamp(sample_w, 0.25, 6.00);

      double g2 = 0.0;
      double g_lin[FX6_AI_WEIGHTS];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         g_lin[i] = FX6_ClipSym(sw * g * x[i], 6.0);
         g2 += g_lin[i] * g_lin[i];
      }

      double g_hash1[FX6_ENHASH_BUCKETS];
      double g_hash2[FX6_FTRL_HASH2_BUCKETS];
      for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g_hash1[i] = 0.0;
      for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++) g_hash2[i] = 0.0;

      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               g_hash1[h1] += FX6_ClipSym(sw * g * hv, 6.0);
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  g_hash2[h2] += FX6_ClipSym(sw * g * 0.70 * hv, 6.0);
               }
            }
         }
      }

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g2 += g_hash1[i] * g_hash1[i];
      for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++) g2 += g_hash2[i] * g_hash2[i];

      double gscale = 1.0;
      if(g2 > 0.0)
      {
         double gn = MathSqrt(g2);
         double clip = 6.0;
         if(gn > clip) gscale = clip / gn;
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double gi = g_lin[i] * gscale;
         double ni_old = m_mv_n[i];
         double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha_m;
         m_mv_z[i] += gi - sigma * m_mv_w[i];
         m_mv_n[i]  = ni_old + gi * gi;
      }

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
      {
         double gi = g_hash1[i] * gscale;
         double ni_old = m_mv_hn1[i];
         double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha_m;
         m_mv_hz1[i] += gi - sigma * m_mv_hw1[i];
         m_mv_hn1[i]  = ni_old + gi * gi;
      }

      for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
      {
         double gi = g_hash2[i] * gscale;
         double ni_old = m_mv_hn2[i];
         double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha_m;
         m_mv_hz2[i] += gi - sigma * m_mv_hw2[i];
         m_mv_hn2[i]  = ni_old + gi * gi;
      }

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      m_mv_steps++;
   }

   void UpdateWeighted(const int cls,
                       const double &x[],
                       const FX6AIHyperParams &hp,
                       const double sample_w,
                       const double move_points)
   {
      if(cls < 0 || cls >= FX6_FTRL_CLASS_COUNT) return;

      m_step++;

      // Class-balance priors and recency weighting.
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FX6_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);
      double recency = 0.85 + 0.30 * (1.0 - MathExp(-(double)m_step / 512.0));
      double w = FX6_Clamp(sample_w * cls_bal * recency, 0.10, 6.00);

      double alpha,beta,l1,l2;
      ComputeAdaptiveParams(hp, w, alpha, beta, l1, l2);
      BuildWeights(alpha, beta, l1, l2);

      double logits[FX6_FTRL_CLASS_COUNT];
      double p_raw[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
         logits[c] = FX6_ClipSym(ScoreRawClass(x, c), 35.0);
      Softmax3(logits, p_raw);

      double g_logits[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         g_logits[c] = FX6_ClipSym((p_raw[c] - target) * w, 4.0);
      }

      double g_lin[FX6_FTRL_CLASS_COUNT][FX6_AI_WEIGHTS];
      double g_hash1[FX6_FTRL_CLASS_COUNT][FX6_ENHASH_BUCKETS];
      double g_hash2[FX6_FTRL_CLASS_COUNT][FX6_FTRL_HASH2_BUCKETS];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g_lin[c][i] = 0.0;
         for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g_hash1[c][i] = 0.0;
         for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++) g_hash2[c][i] = 0.0;
      }

      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g_lin[c][i] = FX6_ClipSym(g_logits[c] * x[i], 8.0);
      }

      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);

               m_hload1[h1] = 0.997 * m_hload1[h1] + 0.003;
               m_hmean1 = 0.999 * m_hmean1 + 0.001 * m_hload1[h1];
               double rb1 = CollisionRebalance1(h1);

               int h2 = -1;
               double rb2 = 1.0;
               if(m_use_hash2)
               {
                  h2 = HashIndex2(i, j);
                  m_hload2[h2] = 0.997 * m_hload2[h2] + 0.003;
                  m_hmean2 = 0.999 * m_hmean2 + 0.001 * m_hload2[h2];
                  rb2 = CollisionRebalance2(h2);
               }

               for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
               {
                  double gh = FX6_ClipSym(g_logits[c] * hv, 8.0);
                  g_hash1[c][h1] += 1.00 * rb1 * gh;
                  if(m_use_hash2 && h2 >= 0)
                     g_hash2[c][h2] += 0.70 * rb2 * gh;
               }
            }
         }
      }

      // Full gradient-norm clipping (linear + both hash spaces).
      double g2 = 0.0;
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g2 += g_lin[c][i] * g_lin[c][i];
         for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g2 += g_hash1[c][i] * g_hash1[c][i];
         for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++) g2 += g_hash2[c][i] * g_hash2[c][i];
      }

      double gscale = 1.0;
      if(g2 > 0.0)
      {
         double gn = MathSqrt(g2);
         double clip = FX6_Clamp(7.0 + MathSqrt(w), 6.0, 12.0);
         if(gn > clip) gscale = clip / gn;
      }

      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double gi = g_lin[c][i] * gscale;
            double ni_old = m_n[c][i];
            double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha;
            m_z[c][i] += gi - sigma * m_w[c][i];
            m_n[c][i]  = ni_old + gi * gi;
         }

         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         {
            double gi = g_hash1[c][i] * gscale;
            double ni_old = m_hn1[c][i];
            double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha;
            m_hz1[c][i] += gi - sigma * m_hw1[c][i];
            m_hn1[c][i]  = ni_old + gi * gi;
         }

         for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
         {
            double gi = g_hash2[c][i] * gscale;
            double ni_old = m_hn2[c][i];
            double sigma = (MathSqrt(ni_old + gi * gi) - MathSqrt(ni_old)) / alpha;
            m_hz2[c][i] += gi - sigma * m_hw2[c][i];
            m_hn2[c][i]  = ni_old + gi * gi;
         }
      }

      double ce = -MathLog(FX6_Clamp(p_raw[cls], 1e-6, 1.0));
      UpdateLossDrift(ce);
      double cal_lr = FX6_Clamp(alpha * 0.15, 0.0002, 0.0200);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      UpdateMoveHeadFTRL(x, move_points, alpha, beta, l2, w);
   }

public:
   CFX6AIFTRL(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_FTRL_LOGIT; }
   virtual string AIName(void) const { return "ftrl_logit"; }
   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      double alpha,beta,l1,l2;
      ComputeAdaptiveParams(hp, 1.0, alpha, beta, l1, l2);
      BuildWeights(alpha, beta, l1, l2);

      double logits[FX6_FTRL_CLASS_COUNT];
      double p_raw[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
         logits[c] = FX6_ClipSym(ScoreRawClass(x, c), 35.0);
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points <= 0.0)
         expected_move_points = ResolveMinMovePoints();
      if(expected_move_points <= 0.0)
         expected_move_points = 0.10;
      return true;
   }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();

      m_step = 0;
      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;
      m_mv_steps = 0;

      m_use_hash = true;
      m_use_hash2 = true;
      m_hmean1 = 0.0;
      m_hmean2 = 0.0;

      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         m_cls_ema[c] = 1.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_z[c][i] = 0.0;
            m_n[c][i] = 0.0;
            m_w[c][i] = 0.0;
         }
         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         {
            m_hz1[c][i] = 0.0;
            m_hn1[c][i] = 0.0;
            m_hw1[c][i] = 0.0;
         }
         for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
         {
            m_hz2[c][i] = 0.0;
            m_hn2[c][i] = 0.0;
            m_hw2[c][i] = 0.0;
         }
         for(int b=0; b<FX6_FTRL_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal3_temp = 1.0;
      m_cal3_steps = 0;

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
      {
         m_hload1[i] = 0.0;
         m_mv_hz1[i] = 0.0;
         m_mv_hn1[i] = 0.0;
         m_mv_hw1[i] = 0.0;
      }
      for(int i=0; i<FX6_FTRL_HASH2_BUCKETS; i++)
      {
         m_hload2[i] = 0.0;
         m_mv_hz2[i] = 0.0;
         m_mv_hn2[i] = 0.0;
         m_mv_hw2[i] = 0.0;
      }
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_mv_z[i] = 0.0;
         m_mv_n[i] = 0.0;
         m_mv_w[i] = 0.0;
      }
   }

protected:
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost_proxy = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double edge = MathMax(0.0, abs_move - cost_proxy);
      double denom = MathMax(cost_proxy, 0.50);
      double ev_w = FX6_Clamp(0.35 + (edge / denom), 0.15, 6.00);
      if(cls == (int)FX6_LABEL_SKIP) ev_w *= 0.90;
      double w = FX6_Clamp(ev_w, 0.10, 6.00);

      UpdateWeighted(cls, x, h, w, move_points);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      double alpha,beta,l1,l2;
      ComputeAdaptiveParams(hp, 1.0, alpha, beta, l1, l2);
      BuildWeights(alpha, beta, l1, l2);

      double logits[FX6_FTRL_CLASS_COUNT];
      double p_raw[FX6_FTRL_CLASS_COUNT];
      double p_cal[FX6_FTRL_CLASS_COUNT];
      for(int c=0; c<FX6_FTRL_CLASS_COUNT; c++)
         logits[c] = FX6_ClipSym(ScoreRawClass(x, c), 35.0);
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, p_cal);

      double den = p_cal[(int)FX6_LABEL_BUY] + p_cal[(int)FX6_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FX6_Clamp(p_cal[(int)FX6_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      double alpha,beta,l1,l2;
      ComputeAdaptiveParams(hp, 1.0, alpha, beta, l1, l2);
      BuildMoveWeights(FX6_Clamp(0.70 * alpha, 0.00005, 3.0000),
                       FX6_Clamp(0.80 * beta + 0.05, 0.0000, 10.0000),
                       FX6_Clamp(0.50 * l2 + 0.001, 0.0000, 2.0000));

      double pred = FX6_DotLinear(m_mv_w, x);
      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               pred += m_mv_hw1[h1] * hv;
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  pred += 0.70 * m_mv_hw2[h2] * hv;
               }
            }
         }
      }

      if(pred < 0.0) pred = 0.0;

      double base = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      if(m_mv_steps >= 24 && base > 0.0) return 0.65 * pred + 0.35 * base;
      if(m_mv_steps >= 24) return pred;
      if(base > 0.0) return base;
      return pred;
   }
};

#endif // __FX6_AI_FTRL_MQH__
