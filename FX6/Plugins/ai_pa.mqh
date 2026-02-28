#ifndef __FX6_AI_PA_MQH__
#define __FX6_AI_PA_MQH__

#include "..\plugin_base.mqh"

#define FX6_PA_CLASS_COUNT 3
#define FX6_PA_HASH2_BUCKETS 97
#define FX6_PA_CAL_BINS 10

class CFX6AIPA : public CFX6AIPlugin
{
private:
   // Crammer-Singer multiclass PA weights.
   double m_w[FX6_PA_CLASS_COUNT][FX6_AI_WEIGHTS];
   double m_w_avg[FX6_PA_CLASS_COUNT][FX6_AI_WEIGHTS];

   // Optional hashed interactions (two spaces to reduce collisions).
   double m_hw1[FX6_PA_CLASS_COUNT][FX6_ENHASH_BUCKETS];
   double m_hw1_avg[FX6_PA_CLASS_COUNT][FX6_ENHASH_BUCKETS];
   double m_hw2[FX6_PA_CLASS_COUNT][FX6_PA_HASH2_BUCKETS];
   double m_hw2_avg[FX6_PA_CLASS_COUNT][FX6_PA_HASH2_BUCKETS];

   // Dedicated move-amplitude head (plugin-level, not base-only EMA).
   double m_mv_w[FX6_AI_WEIGHTS];
   double m_mv_hw1[FX6_ENHASH_BUCKETS];
   double m_mv_hw2[FX6_PA_HASH2_BUCKETS];
   int    m_mv_steps;

   int    m_steps;
   bool   m_margin_ready;
   double m_margin_ema;
   bool   m_use_hash;
   bool   m_use_hash2;

   // Online balancing and drift guard.
   double m_cls_ema[FX6_PA_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration.
   double m_cal3_temp;
   double m_cal3_bias[FX6_PA_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_PA_CLASS_COUNT][FX6_PA_CAL_BINS];
   double m_cal3_iso_cnt[FX6_PA_CLASS_COUNT][FX6_PA_CAL_BINS];
   int    m_cal3_steps;

   int HashIndex(const int i, const int j) const
   {
      uint h = ((uint)(i * 73856093U)) ^ ((uint)(j * 19349663U));
      return (int)(h % (uint)FX6_ENHASH_BUCKETS);
   }

   int HashIndex2(const int i, const int j) const
   {
      uint h = ((uint)(i * 83492791U)) ^ ((uint)(j * 2654435761U));
      return (int)(h % (uint)FX6_PA_HASH2_BUCKETS);
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

   double ScoreClass(const double &x[],
                     const int cls,
                     const bool averaged) const
   {
      if(cls < 0 || cls >= FX6_PA_CLASS_COUNT) return 0.0;

      double z = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         z += (averaged ? m_w_avg[cls][i] : m_w[cls][i]) * x[i];

      if(!m_use_hash) return z;

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            double hv = HashSign(i, j) * x[i] * x[j];
            int h1 = HashIndex(i, j);
            z += (averaged ? m_hw1_avg[cls][h1] : m_hw1[cls][h1]) * hv;

            if(m_use_hash2)
            {
               int h2 = HashIndex2(i, j);
               z += 0.70 * (averaged ? m_hw2_avg[cls][h2] : m_hw2[cls][h2]) * hv;
            }
         }
      }
      return z;
   }

   void ComputeScores(const double &x[],
                      const bool averaged,
                      double &scores[]) const
   {
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
         scores[c] = ScoreClass(x, c, averaged);
   }

   double ComputePhiNorm2(const double &x[]) const
   {
      double norm2 = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         norm2 += x[i] * x[i];

      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               norm2 += hv * hv;
               if(m_use_hash2)
                  norm2 += 0.49 * hv * hv;
            }
         }
      }

      if(norm2 <= 1e-9) norm2 = 1e-9;
      return norm2;
   }

   void ApplyDecay(const double decay)
   {
      if(decay <= 0.0) return;

      double k = 1.0 - 0.01 * decay;
      if(k < 0.90) k = 0.90;

      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            m_w[c][i] *= k;
            m_w_avg[c][i] *= k;
         }
         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] *= k;
            m_hw1_avg[c][i] *= k;
         }
         for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] *= k;
            m_hw2_avg[c][i] *= k;
         }
      }

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
         m_mv_w[i] *= k;
      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         m_mv_hw1[i] *= k;
      for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
         m_mv_hw2[i] *= k;
   }

   double PredictMoveRaw(const double &x[]) const
   {
      double y = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         y += m_mv_w[i] * x[i];

      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               y += m_mv_hw1[h1] * hv;
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  y += 0.70 * m_mv_hw2[h2] * hv;
               }
            }
         }
      }

      if(y < 0.0) y = 0.0;
      return y;
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_PA_CLASS_COUNT];
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_PA_CLASS_COUNT];
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_PA_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_PA_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_PA_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_PA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_PA_CAL_BINS) bi = FX6_PA_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
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
      double logits[FX6_PA_CLASS_COUNT];
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = (MathLog(pr) * inv_temp) + m_cal3_bias[c];
      }

      double p_cal[FX6_PA_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FX6_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_PA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_PA_CAL_BINS) bi = FX6_PA_CAL_BINS - 1;
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
      if(m_steps < 256 || m_drift_cooldown > 0) return;

      if(m_loss_fast > 1.7 * MathMax(m_loss_slow, 0.10))
         m_drift_cooldown = 96;
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
      double g = (MathAbs(err) <= delta ? err : FX6_Sign(err) * delta);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double lr = FX6_Clamp(0.02 * hp.lr * (0.85 + 0.15 * w), 0.00005, 0.02000);
      double wd = FX6_Clamp(0.25 * hp.l2, 0.0, 0.0500);

      double g2 = 0.0;
      double g_lin[FX6_AI_WEIGHTS];
      double g_h1[FX6_ENHASH_BUCKETS];
      double g_h2[FX6_PA_HASH2_BUCKETS];

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         g_lin[i] = FX6_ClipSym(w * g * x[i], 6.0);
         g2 += g_lin[i] * g_lin[i];
      }
      for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g_h1[i] = 0.0;
      for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++) g_h2[i] = 0.0;

      if(m_use_hash)
      {
         for(int i=1; i<FX6_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               g_h1[h1] += FX6_ClipSym(w * g * hv, 6.0);

               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  g_h2[h2] += FX6_ClipSym(w * g * 0.70 * hv, 6.0);
               }
            }
         }
      }

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++) g2 += g_h1[i] * g_h1[i];
      for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++) g2 += g_h2[i] * g_h2[i];

      double gscale = 1.0;
      if(g2 > 0.0)
      {
         double gn = MathSqrt(g2);
         double clip = 6.0;
         if(gn > clip) gscale = clip / gn;
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         if(i != 0) m_mv_w[i] *= (1.0 - lr * wd);
         m_mv_w[i] += lr * (g_lin[i] * gscale);
         m_mv_w[i] = FX6_ClipSym(m_mv_w[i], 20.0);
      }

      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
      {
         m_mv_hw1[i] *= (1.0 - lr * wd);
         m_mv_hw1[i] += lr * (g_h1[i] * gscale);
         m_mv_hw1[i] = FX6_ClipSym(m_mv_hw1[i], 15.0);
      }

      for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
      {
         m_mv_hw2[i] *= (1.0 - lr * wd);
         m_mv_hw2[i] += lr * (g_h2[i] * gscale);
         m_mv_hw2[i] = FX6_ClipSym(m_mv_hw2[i], 12.0);
      }

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      m_mv_steps++;
   }

   void UpdateWeighted(const int cls,
                       const double &x[],
                       const FX6AIHyperParams &hp,
                       const double margin_scale,
                       const double sample_w,
                       const double move_points)
   {
      if(cls < 0 || cls >= FX6_PA_CLASS_COUNT) return;

      m_steps++;

      // Class balance and recency weighting for online stability.
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FX6_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);
      double recency = 0.85 + 0.30 * (1.0 - MathExp(-(double)m_steps / 512.0));
      double w = FX6_Clamp(sample_w * cls_bal * recency, 0.10, 6.00);

      double c = FX6_Clamp(hp.pa_c, 0.001, 100.0);
      double margin0 = FX6_Clamp(hp.pa_margin, 0.05, 4.0);
      double margin = FX6_Clamp(margin0 * margin_scale, 0.05, 8.0);
      double decay = FX6_Clamp(hp.l2, 0.0, 0.2);

      if(m_drift_cooldown > 0)
      {
         c = FX6_Clamp(c * 0.70, 0.001, 100.0);
         margin = FX6_Clamp(margin * 1.10, 0.05, 8.0);
      }

      ApplyDecay(decay);

      double scores[FX6_PA_CLASS_COUNT];
      ComputeScores(x, false, scores);

      int rival = -1;
      double rival_score = -DBL_MAX;
      for(int cidx=0; cidx<FX6_PA_CLASS_COUNT; cidx++)
      {
         if(cidx == cls) continue;
         if(scores[cidx] > rival_score)
         {
            rival_score = scores[cidx];
            rival = cidx;
         }
      }
      if(rival < 0) return;

      double loss = margin - scores[cls] + scores[rival];
      if(loss > 0.0)
      {
         double norm2 = ComputePhiNorm2(x);
         double tau = loss / (2.0 * norm2 + (1.0 / (2.0 * c)));
         tau = FX6_Clamp(tau * w, 0.0, c);
         if(m_drift_cooldown > 0)
            tau = FX6_Clamp(tau * 0.80, 0.0, c);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double dx = tau * x[i];
            m_w[cls][i] += dx;
            m_w[rival][i] -= dx;
         }

         if(m_use_hash)
         {
            for(int i=1; i<FX6_AI_WEIGHTS; i++)
            {
               for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
               {
                  double hv = HashSign(i, j) * x[i] * x[j];
                  int h1 = HashIndex(i, j);
                  double dh1 = tau * hv;
                  m_hw1[cls][h1] += dh1;
                  m_hw1[rival][h1] -= dh1;

                  if(m_use_hash2)
                  {
                     int h2 = HashIndex2(i, j);
                     double dh2 = tau * 0.70 * hv;
                     m_hw2[cls][h2] += dh2;
                     m_hw2[rival][h2] -= dh2;
                  }
               }
            }
         }
      }

      // Averaged-PA weights.
      double beta = 1.0 / (double)m_steps;
      for(int cidx=0; cidx<FX6_PA_CLASS_COUNT; cidx++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_w_avg[cidx][i] += beta * (m_w[cidx][i] - m_w_avg[cidx][i]);
         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
            m_hw1_avg[cidx][i] += beta * (m_hw1[cidx][i] - m_hw1_avg[cidx][i]);
         for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
            m_hw2_avg[cidx][i] += beta * (m_hw2[cidx][i] - m_hw2_avg[cidx][i]);
      }

      double scores_post[FX6_PA_CLASS_COUNT];
      ComputeScores(x, false, scores_post);
      double margin_gap = scores_post[cls] - scores_post[rival];
      double abs_margin = MathAbs(margin_gap);
      if(!m_margin_ready)
      {
         m_margin_ema = MathMax(abs_margin, 0.25);
         m_margin_ready = true;
      }
      else
      {
         m_margin_ema = 0.95 * m_margin_ema + 0.05 * abs_margin;
         if(m_margin_ema < 0.25) m_margin_ema = 0.25;
      }

      double den = MathMax(m_margin_ema, 0.25);
      double logits[FX6_PA_CLASS_COUNT];
      for(int cidx=0; cidx<FX6_PA_CLASS_COUNT; cidx++)
         logits[cidx] = FX6_ClipSym(scores_post[cidx] / den, 20.0);
      double p_raw[FX6_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);

      double ce = -MathLog(FX6_Clamp(p_raw[cls], 1e-6, 1.0));
      UpdateLossDrift(ce);
      double cal_lr = FX6_Clamp(0.01 * MathSqrt(w), 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      // Keep legacy binary calibrator aligned for compatibility paths.
      double den_dir = p_raw[(int)FX6_LABEL_BUY] + p_raw[(int)FX6_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FX6_LABEL_BUY] / den_dir;
      if(cls == (int)FX6_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FX6_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      UpdateMoveHeadJoint(x, move_points, hp, w);
   }

public:
   CFX6AIPA(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_PA_LINEAR; }
   virtual string AIName(void) const { return "pa_linear"; }
   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      bool averaged = (m_steps > 24);
      double scores[FX6_PA_CLASS_COUNT];
      ComputeScores(x, averaged, scores);

      double den = (m_margin_ready ? m_margin_ema : FX6_Clamp(hp.pa_margin, 0.25, 4.0));
      if(den < 0.25) den = 0.25;

      double logits[FX6_PA_CLASS_COUNT];
      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
         logits[c] = FX6_ClipSym(scores[c] / den, 20.0);

      double p_raw[FX6_PA_CLASS_COUNT];
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

      m_steps = 0;
      m_mv_steps = 0;
      m_margin_ready = false;
      m_margin_ema = 0.0;
      m_use_hash = true;
      m_use_hash2 = true;

      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;

      for(int c=0; c<FX6_PA_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_cal3_bias[c] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_w[c][i] = 0.0;
            m_w_avg[c][i] = 0.0;
         }

         for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] = 0.0;
            m_hw1_avg[c][i] = 0.0;
         }

         for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] = 0.0;
            m_hw2_avg[c][i] = 0.0;
         }

         for(int b=0; b<FX6_PA_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_mv_w[i] = 0.0;
      for(int i=0; i<FX6_ENHASH_BUCKETS; i++)
         m_mv_hw1[i] = 0.0;
      for(int i=0; i<FX6_PA_HASH2_BUCKETS; i++)
         m_mv_hw2[i] = 0.0;
   }

   // Legacy compatibility path.
   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      UpdateWeighted(cls, x, hp, 1.0, 1.0, pseudo_move);
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
      double excess = MathMax(0.0, abs_move - cost_proxy);

      double ev_w = FX6_Clamp(0.35 + (excess / MathMax(cost_proxy, 0.50)), 0.10, 6.00);
      if(cls == (int)FX6_LABEL_SKIP) ev_w *= 0.90;

      double move_scale = FX6_Clamp(1.0 + 0.10 * excess, 0.70, 3.50);
      double margin_scale = FX6_Clamp(move_scale * (1.0 + 0.25 * (ev_w - 1.0)), 0.60, 4.00);

      UpdateWeighted(cls, x, h, margin_scale, ev_w, move_points);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      double probs[FX6_PA_CLASS_COUNT];
      double expected = -1.0;
      if(!PredictNativeClassProbs(x, hp, probs, expected))
         return 0.5;

      double den = probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FX6_Clamp(probs[(int)FX6_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      double head = PredictMoveRaw(x);
      double base = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);

      if(m_mv_steps >= 24 && m_move_ready && m_move_ema_abs > 0.0)
         head = 0.70 * head + 0.30 * m_move_ema_abs;

      if(head > 0.0 && base > 0.0) return 0.65 * head + 0.35 * base;
      if(head > 0.0) return head;
      return base;
   }
};

#endif // __FX6_AI_PA_MQH__
