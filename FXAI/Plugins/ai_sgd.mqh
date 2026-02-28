#ifndef __FX6_AI_SGD_MQH__
#define __FX6_AI_SGD_MQH__

#include "..\plugin_base.mqh"

#define FX6_SGD_CLASS_COUNT 3
#define FX6_SGD_HASH_BUCKETS 192
#define FX6_SGD_ISO_BINS 10

class CFX6AISGD : public CFX6AIPlugin
{
private:
   double m_w[FX6_SGD_CLASS_COUNT][FX6_AI_WEIGHTS];
   double m_m[FX6_SGD_CLASS_COUNT][FX6_AI_WEIGHTS];
   double m_v[FX6_SGD_CLASS_COUNT][FX6_AI_WEIGHTS];

   double m_hw[FX6_SGD_CLASS_COUNT][FX6_SGD_HASH_BUCKETS];
   double m_hm[FX6_SGD_CLASS_COUNT][FX6_SGD_HASH_BUCKETS];
   double m_hv[FX6_SGD_CLASS_COUNT][FX6_SGD_HASH_BUCKETS];

   // Joint move-amplitude head.
   double m_mv_w[FX6_AI_WEIGHTS];
   double m_mv_m[FX6_AI_WEIGHTS];
   double m_mv_v[FX6_AI_WEIGHTS];
   double m_mv_hw[FX6_SGD_HASH_BUCKETS];
   double m_mv_hm[FX6_SGD_HASH_BUCKETS];
   double m_mv_hv[FX6_SGD_HASH_BUCKETS];
   bool   m_mv_ready;
   double m_mv_ema_abs;

   // Multiclass calibration.
   double m_cal_temp;
   double m_cal_bias[FX6_SGD_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_SGD_CLASS_COUNT][FX6_SGD_ISO_BINS];
   double m_cal3_iso_cnt[FX6_SGD_CLASS_COUNT][FX6_SGD_ISO_BINS];
   int    m_cal3_steps;

   // Drift guard and balancing.
   int    m_step;
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;
   double m_class_ema[FX6_SGD_CLASS_COUNT];

   int HashIndex(const int i, const int j) const
   {
      uint h = ((uint)(i * 73856093U)) ^ ((uint)(j * 19349663U));
      return (int)(h % (uint)FX6_SGD_HASH_BUCKETS);
   }

   double HashSign(const int i, const int j) const
   {
      int v = (i * 31 + j * 17) & 1;
      return (v == 0 ? 1.0 : -1.0);
   }

   double ScheduledLR(const FX6AIHyperParams &hp,
                      const double sample_w) const
   {
      double base = FX6_Clamp(hp.lr, 0.00002, 0.20000);
      double st = (double)MathMax(m_step, 1);

      double warmup = FX6_Clamp(st / 128.0, 0.10, 1.00);
      double invsqrt = 1.0 / MathSqrt(1.0 + 0.004 * MathMax(0.0, st - 128.0));
      double cyc = 2048.0;
      double ph = MathMod(st, cyc) / cyc;
      double cosine = 0.60 + 0.40 * (0.5 * (1.0 + MathCos(3.141592653589793 * ph)));
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      double sw_scale = FX6_Clamp(0.80 + 0.20 * sw, 0.70, 1.60);

      return FX6_Clamp(base * warmup * invsqrt * cosine * sw_scale, 0.00001, 0.08000);
   }

   double ScheduledMoveLR(const FX6AIHyperParams &hp,
                          const double sample_w) const
   {
      double base = FX6_Clamp(0.65 * hp.lr, 0.00001, 0.08000);
      double sw = FX6_Clamp(sample_w, 0.25, 4.00);
      return FX6_Clamp(base * (0.90 + 0.10 * sw), 0.00001, 0.05000);
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

   void BuildClassLogits(const double &x[],
                         double &logits[]) const
   {
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double z = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            z += m_w[c][i] * x[i];

         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int hi = HashIndex(i, j);
               z += m_hw[c][hi] * hv;
            }
         }

         logits[c] = FX6_ClipSym(z, 35.0);
      }
   }

   double PredictMoveRaw(const double &x[]) const
   {
      double y = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         y += m_mv_w[i] * x[i];

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            int hi = HashIndex(i, j);
            double hv = HashSign(i, j) * x[i] * x[j];
            y += m_mv_hw[hi] * hv;
         }
      }

      if(y < 0.0) y = 0.0;
      return y;
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal_temp, 0.50, 3.00);
      double logits[FX6_SGD_CLASS_COUNT];
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_SGD_CLASS_COUNT];
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_SGD_ISO_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_SGD_ISO_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_SGD_ISO_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_SGD_ISO_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_SGD_ISO_BINS) bi = FX6_SGD_ISO_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
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
      double logits[FX6_SGD_CLASS_COUNT];
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal_bias[c];
      }

      double p_cal[FX6_SGD_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double cal_lr = FX6_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal_bias[c] = FX6_ClipSym(m_cal_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_SGD_ISO_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_SGD_ISO_BINS) bi = FX6_SGD_ISO_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal_temp = FX6_Clamp(m_cal_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void ApplyDriftGuard(const double ce_loss)
   {
      if(!m_loss_ready)
      {
         m_loss_fast = ce_loss;
         m_loss_slow = ce_loss;
         m_loss_ready = true;
         return;
      }

      m_loss_fast = 0.90 * m_loss_fast + 0.10 * ce_loss;
      m_loss_slow = 0.99 * m_loss_slow + 0.01 * ce_loss;

      if(m_drift_cooldown > 0) m_drift_cooldown--;
      if(m_step < 256 || m_drift_cooldown > 0) return;

      if(m_loss_fast > 1.8 * MathMax(m_loss_slow, 0.10))
      {
         // Keep weights, but purge stale optimizer moments for regime reset.
         for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
         {
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               m_m[c][i] *= 0.20;
               m_v[c][i] *= 0.20;
            }
            for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++)
            {
               m_hm[c][b] *= 0.20;
               m_hv[c][b] *= 0.20;
            }
         }

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_mv_m[i] *= 0.20;
            m_mv_v[i] *= 0.20;
         }
         for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++)
         {
            m_mv_hm[b] *= 0.20;
            m_mv_hv[b] *= 0.20;
         }

         m_drift_cooldown = 64;
      }
   }

   void UpdateMoveHeadJoint(const double &x[],
                            const double move_points,
                            const FX6AIHyperParams &hp,
                            const double sample_w)
   {
      double target = MathAbs(move_points);
      if(!MathIsValidNumber(target)) return;

      double pred = PredictMoveRaw(x);
      double err = target - pred;
      double delta = 1.0;
      double g = (MathAbs(err) <= delta ? err : (err > 0.0 ? delta : -delta));
      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double lr = ScheduledMoveLR(hp, w);
      double wd = FX6_Clamp(0.20 * hp.l2, 0.0, 0.0500);

      const double b1 = 0.90;
      const double b2 = 0.999;
      const double eps = 1e-8;
      double t = (double)MathMax(m_step, 1);
      double bc1 = 1.0 - MathPow(b1, t);
      double bc2 = 1.0 - MathPow(b2, t);
      if(bc1 < 1e-8) bc1 = 1e-8;
      if(bc2 < 1e-8) bc2 = 1e-8;

      double gnorm2 = 0.0;
      double glin[FX6_AI_WEIGHTS];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         glin[i] = FX6_ClipSym(w * g * x[i], 6.0);
         gnorm2 += glin[i] * glin[i];
      }

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            double hv = HashSign(i, j) * x[i] * x[j];
            double gh = FX6_ClipSym(w * g * hv, 6.0);
            gnorm2 += gh * gh;
         }
      }

      double grad_scale = 1.0;
      if(gnorm2 > 0.0)
      {
         double gnorm = MathSqrt(gnorm2);
         double clip = 6.0;
         if(gnorm > clip)
         {
            grad_scale = clip / gnorm;
            for(int i=0; i<FX6_AI_WEIGHTS; i++) glin[i] *= grad_scale;
         }
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         if(i != 0) m_mv_w[i] *= (1.0 - lr * wd);
         m_mv_m[i] = b1 * m_mv_m[i] + (1.0 - b1) * glin[i];
         m_mv_v[i] = b2 * m_mv_v[i] + (1.0 - b2) * glin[i] * glin[i];
         double mhat = m_mv_m[i] / bc1;
         double vhat = m_mv_v[i] / bc2;
         m_mv_w[i] += lr * (mhat / (MathSqrt(vhat) + eps));
         m_mv_w[i] = FX6_ClipSym(m_mv_w[i], 20.0);
      }

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            int hi = HashIndex(i, j);
            double hv = HashSign(i, j) * x[i] * x[j];
            double gh = FX6_ClipSym(w * g * hv, 6.0) * grad_scale;

            m_mv_hw[hi] *= (1.0 - lr * wd);
            m_mv_hm[hi] = b1 * m_mv_hm[hi] + (1.0 - b1) * gh;
            m_mv_hv[hi] = b2 * m_mv_hv[hi] + (1.0 - b2) * gh * gh;
            double mhat = m_mv_hm[hi] / bc1;
            double vhat = m_mv_hv[hi] / bc2;
            m_mv_hw[hi] += lr * (mhat / (MathSqrt(vhat) + eps));
            m_mv_hw[hi] = FX6_ClipSym(m_mv_hw[hi], 15.0);
         }
      }

      FX6_UpdateMoveEMA(m_mv_ema_abs, m_mv_ready, move_points, 0.05);
   }

   void UpdateWeighted(const int cls,
                       const double &x[],
                       const FX6AIHyperParams &hp,
                       const double sample_w,
                       const double move_points)
   {
      if(cls < 0 || cls >= FX6_SGD_CLASS_COUNT) return;

      m_step++;

      // Class-balance weight from EMA priors.
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
         m_class_ema[c] = 0.995 * m_class_ema[c] + (c == cls ? 0.005 : 0.0);
      double mean_cnt = (m_class_ema[0] + m_class_ema[1] + m_class_ema[2]) / 3.0;
      double cls_bal = FX6_Clamp(mean_cnt / MathMax(m_class_ema[cls], 0.01), 0.60, 2.20);
      double w = FX6_Clamp(sample_w * cls_bal, 0.10, 6.00);

      double lr = ScheduledLR(hp, w);
      double wd = FX6_Clamp(hp.l2, 0.0, 0.0500);

      double logits[FX6_SGD_CLASS_COUNT];
      double probs_raw[FX6_SGD_CLASS_COUNT];
      BuildClassLogits(x, logits);
      Softmax3(logits, probs_raw);

      double g_logits[FX6_SGD_CLASS_COUNT];
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         g_logits[c] = FX6_ClipSym((target - probs_raw[c]) * w, 4.0);
      }

      // Build grads.
      double g_lin[FX6_SGD_CLASS_COUNT][FX6_AI_WEIGHTS];
      double g_hash[FX6_SGD_CLASS_COUNT][FX6_SGD_HASH_BUCKETS];
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g_lin[c][i] = 0.0;
         for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++) g_hash[c][b] = 0.0;
      }

      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g_lin[c][i] = FX6_ClipSym(g_logits[c] * x[i], 8.0);
      }

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            double hv = HashSign(i, j) * x[i] * x[j];
            int hi = HashIndex(i, j);
            for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
               g_hash[c][hi] += FX6_ClipSym(g_logits[c] * hv, 8.0);
         }
      }

      // Global norm clipping.
      double g2 = 0.0;
      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g2 += g_lin[c][i] * g_lin[c][i];
         for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++) g2 += g_hash[c][b] * g_hash[c][b];
      }
      if(g2 > 0.0)
      {
         double gn = MathSqrt(g2);
         double clip = FX6_Clamp(8.0 + MathSqrt(w), 6.0, 12.0);
         if(gn > clip)
         {
            double s = clip / gn;
            for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
            {
               for(int i=0; i<FX6_AI_WEIGHTS; i++) g_lin[c][i] *= s;
               for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++) g_hash[c][b] *= s;
            }
         }
      }

      // AdamW-lite update.
      const double b1 = 0.90;
      const double b2 = 0.999;
      const double eps = 1e-8;
      double t = (double)MathMax(m_step, 1);
      double bc1 = 1.0 - MathPow(b1, t);
      double bc2 = 1.0 - MathPow(b2, t);
      if(bc1 < 1e-8) bc1 = 1e-8;
      if(bc2 < 1e-8) bc2 = 1e-8;

      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            if(i != 0) m_w[c][i] *= (1.0 - lr * wd);

            double g = g_lin[c][i];
            m_m[c][i] = b1 * m_m[c][i] + (1.0 - b1) * g;
            m_v[c][i] = b2 * m_v[c][i] + (1.0 - b2) * g * g;
            double mhat = m_m[c][i] / bc1;
            double vhat = m_v[c][i] / bc2;
            m_w[c][i] += lr * (mhat / (MathSqrt(vhat) + eps));
            m_w[c][i] = FX6_ClipSym(m_w[c][i], 25.0);
         }

         for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++)
         {
            m_hw[c][b] *= (1.0 - lr * wd);

            double g = g_hash[c][b];
            m_hm[c][b] = b1 * m_hm[c][b] + (1.0 - b1) * g;
            m_hv[c][b] = b2 * m_hv[c][b] + (1.0 - b2) * g * g;
            double mhat = m_hm[c][b] / bc1;
            double vhat = m_hv[c][b] / bc2;
            m_hw[c][b] += lr * (mhat / (MathSqrt(vhat) + eps));
            m_hw[c][b] = FX6_ClipSym(m_hw[c][b], 20.0);
         }
      }

      double ce = -MathLog(FX6_Clamp(probs_raw[cls], 1e-6, 1.0));
      ApplyDriftGuard(ce);
      UpdateCalibrator3(probs_raw, cls, w, lr);

      double den = probs_raw[(int)FX6_LABEL_BUY] + probs_raw[(int)FX6_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs_raw[(int)FX6_LABEL_BUY] / den;
      if(cls == (int)FX6_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FX6_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hp, w);
      UpdateMoveHeadJoint(x, move_points, hp, w);
   }

public:
   CFX6AISGD(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_SGD_LOGIT; }
   virtual string AIName(void) const { return "sgd_logit"; }

   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      double logits[FX6_SGD_CLASS_COUNT];
      double p_raw[FX6_SGD_CLASS_COUNT];
      BuildClassLogits(x, logits);
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

      for(int c=0; c<FX6_SGD_CLASS_COUNT; c++)
      {
         m_cal_bias[c] = 0.0;
         m_class_ema[c] = 1.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_w[c][i] = 0.0;
            m_m[c][i] = 0.0;
            m_v[c][i] = 0.0;
         }
         for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++)
         {
            m_hw[c][b] = 0.0;
            m_hm[c][b] = 0.0;
            m_hv[c][b] = 0.0;
         }
         for(int k=0; k<FX6_SGD_ISO_BINS; k++)
         {
            m_cal3_iso_pos[c][k] = 0.0;
            m_cal3_iso_cnt[c][k] = 0.0;
         }
      }
      m_cal_temp = 1.0;
      m_cal3_steps = 0;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_mv_w[i] = 0.0;
         m_mv_m[i] = 0.0;
         m_mv_v[i] = 0.0;
      }
      for(int b=0; b<FX6_SGD_HASH_BUCKETS; b++)
      {
         m_mv_hw[b] = 0.0;
         m_mv_hm[b] = 0.0;
         m_mv_hv[b] = 0.0;
      }
      m_mv_ready = false;
      m_mv_ema_abs = 0.0;
   }

protected:
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost = InputCostProxyPoints(x);
      double edge_w = FX6_MoveEdgeWeight(move_points, cost);
      double abs_move = MathAbs(move_points);
      double excess = MathMax(0.0, abs_move - cost);

      h.lr = FX6_Clamp(h.lr * (1.0 + 0.08 * excess), 0.00002, 0.25000);
      h.l2 = FX6_Clamp(h.l2 * (1.0 - 0.15 * FX6_Clamp(excess / 12.0, 0.0, 1.0)), 0.0, 0.10);

      double cls_w = (cls == (int)FX6_LABEL_SKIP ? 0.80 : 1.0);
      double w = FX6_Clamp(edge_w * cls_w, 0.10, 6.00);
      UpdateWeighted(cls, x, h, w, move_points);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      double logits[FX6_SGD_CLASS_COUNT];
      double p_raw[FX6_SGD_CLASS_COUNT];
      double p_cal[FX6_SGD_CLASS_COUNT];
      BuildClassLogits(x, logits);
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, p_cal);

      double den = p_cal[(int)FX6_LABEL_BUY] + p_cal[(int)FX6_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FX6_Clamp(p_cal[(int)FX6_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      double base_est = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      double head_est = PredictMoveRaw(x);
      if(m_mv_ready && m_mv_ema_abs > 0.0)
         head_est = 0.65 * head_est + 0.35 * m_mv_ema_abs;

      if(head_est > 0.0 && base_est > 0.0) return 0.65 * head_est + 0.35 * base_est;
      if(head_est > 0.0) return head_est;
      return base_est;
   }
};

#endif // __FX6_AI_SGD_MQH__
