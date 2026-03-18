#ifndef __FXAI_AI_PA_MQH__
#define __FXAI_AI_PA_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_PA_CLASS_COUNT 3
#define FXAI_PA_HASH2_BUCKETS 97
#define FXAI_PA_CAL_BINS 10
#define FXAI_PA_REPLAY 192
#define FXAI_PA_TOP_RIVALS 2

class CFXAIAIPA : public CFXAIAIPlugin
{
private:
   CFXAINativeQualityHeads m_quality_heads;
   // Crammer-Singer multiclass PA weights.
   double m_w[FXAI_PA_CLASS_COUNT][FXAI_AI_WEIGHTS];
   double m_w_avg[FXAI_PA_CLASS_COUNT][FXAI_AI_WEIGHTS];

   // Diagonal confidence for AROW/SCW-like scaled PA updates.
   double m_sigma[FXAI_PA_CLASS_COUNT][FXAI_AI_WEIGHTS];

   // Optional hashed interactions (two spaces to reduce collisions).
   double m_hw1[FXAI_PA_CLASS_COUNT][FXAI_ENHASH_BUCKETS];
   double m_hw1_avg[FXAI_PA_CLASS_COUNT][FXAI_ENHASH_BUCKETS];
   double m_hw2[FXAI_PA_CLASS_COUNT][FXAI_PA_HASH2_BUCKETS];
   double m_hw2_avg[FXAI_PA_CLASS_COUNT][FXAI_PA_HASH2_BUCKETS];
   double m_hsigma1[FXAI_PA_CLASS_COUNT][FXAI_ENHASH_BUCKETS];
   double m_hsigma2[FXAI_PA_CLASS_COUNT][FXAI_PA_HASH2_BUCKETS];

   // Collision monitoring / adaptive rebalancing.
   int    m_hash_occ1[FXAI_ENHASH_BUCKETS];
   int    m_hash_occ2[FXAI_PA_HASH2_BUCKETS];
   double m_hash_bw1[FXAI_ENHASH_BUCKETS];
   double m_hash_bw2[FXAI_PA_HASH2_BUCKETS];
   double m_hash2_scale;
   bool   m_hash_occ_ready;

   // Distributional move head (mu + log-variance).
   double m_mv_mu_w[FXAI_AI_WEIGHTS];
   double m_mv_mu_hw1[FXAI_ENHASH_BUCKETS];
   double m_mv_mu_hw2[FXAI_PA_HASH2_BUCKETS];
   double m_mv_lv_w[FXAI_AI_WEIGHTS];
   double m_mv_lv_hw1[FXAI_ENHASH_BUCKETS];
   double m_mv_lv_hw2[FXAI_PA_HASH2_BUCKETS];
   double m_mv_lv_b;
   int    m_mv_steps;

   int    m_steps;
   bool   m_margin_ready;
   double m_margin_ema;
   bool   m_use_hash;
   bool   m_use_hash2;
   int    m_pa_mode;
   double m_conf_r;

   // Online balancing and drift guard.
   double m_cls_ema[FXAI_PA_CLASS_COUNT];
   bool   m_loss_ready;
   double m_loss_fast;
   double m_loss_slow;
   int    m_drift_cooldown;

   // Plugin-native multiclass calibration (matrix scaling + isotonic).
   double m_cal3_w[FXAI_PA_CLASS_COUNT][FXAI_PA_CLASS_COUNT];
   double m_cal3_b[FXAI_PA_CLASS_COUNT];
   double m_cal3_iso_pos[FXAI_PA_CLASS_COUNT][FXAI_PA_CAL_BINS];
   double m_cal3_iso_cnt[FXAI_PA_CLASS_COUNT][FXAI_PA_CAL_BINS];
   int    m_cal3_steps;

   // A/B guard (live vs averaged weights).
   bool   m_guard_ready;
   bool   m_guard_use_avg;
   double m_guard_live_fast;
   double m_guard_live_slow;
   double m_guard_avg_fast;
   double m_guard_avg_slow;

   // Hard replay buffer.
   double m_rep_x[FXAI_PA_REPLAY][FXAI_AI_WEIGHTS];
   int    m_rep_cls[FXAI_PA_REPLAY];
   double m_rep_move[FXAI_PA_REPLAY];
   double m_rep_w[FXAI_PA_REPLAY];
   double m_rep_hard[FXAI_PA_REPLAY];
   int    m_rep_head;
   int    m_rep_size;

   int HashIndex(const int i, const int j) const
   {
      uint h = ((uint)(i * 73856093U)) ^ ((uint)(j * 19349663U));
      return (int)(h % (uint)FXAI_ENHASH_BUCKETS);
   }

   int HashIndex2(const int i, const int j) const
   {
      uint h = ((uint)(i * 83492791U)) ^ ((uint)(j * 2654435761U));
      return (int)(h % (uint)FXAI_PA_HASH2_BUCKETS);
   }

   double HashSign(const int i, const int j) const
   {
      int v = (i * 31 + j * 17) & 1;
      return (v == 0 ? 1.0 : -1.0);
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

   void BuildCollisionProfile(void)
   {
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_hash_occ1[i] = 0;
         m_hash_bw1[i] = 1.0;
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_hash_occ2[i] = 0;
         m_hash_bw2[i] = 1.0;
      }

      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
         {
            int h1 = HashIndex(i, j);
            int h2 = HashIndex2(i, j);
            m_hash_occ1[h1]++;
            m_hash_occ2[h2]++;
         }
      }

      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         double occ = (double)MathMax(1, m_hash_occ1[i]);
         m_hash_bw1[i] = FXAI_Clamp(1.0 / MathSqrt(occ), 0.20, 1.00);
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         double occ = (double)MathMax(1, m_hash_occ2[i]);
         m_hash_bw2[i] = FXAI_Clamp(1.0 / MathSqrt(occ), 0.20, 1.00);
      }
      m_hash2_scale = 1.0;
      m_hash_occ_ready = true;
   }

   void UpdateCollisionRebalance(void)
   {
      if(!m_hash_occ_ready || (m_steps % 128) != 0) return;

      double occ1_mean = 0.0, occ2_mean = 0.0;
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++) occ1_mean += (double)m_hash_occ1[i];
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++) occ2_mean += (double)m_hash_occ2[i];
      occ1_mean /= (double)FXAI_ENHASH_BUCKETS;
      occ2_mean /= (double)FXAI_PA_HASH2_BUCKETS;

      double over1 = 0.0, over2 = 0.0;
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         double occ = (double)m_hash_occ1[i];
         over1 += MathMax(0.0, occ - occ1_mean);
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         double occ = (double)m_hash_occ2[i];
         over2 += MathMax(0.0, occ - occ2_mean);
      }
      double coll_ratio = 0.0;
      if(occ1_mean > 1e-9) coll_ratio = (over1 / (occ1_mean * FXAI_ENHASH_BUCKETS));
      if(occ2_mean > 1e-9) coll_ratio = 0.5 * coll_ratio + 0.5 * (over2 / (occ2_mean * FXAI_PA_HASH2_BUCKETS));
      coll_ratio = FXAI_Clamp(coll_ratio, 0.0, 2.0);

      m_hash2_scale = FXAI_Clamp(0.70 + 0.45 * coll_ratio, 0.60, 1.40);

      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         double occ = (double)MathMax(1, m_hash_occ1[i]);
         double target = FXAI_Clamp(1.0 / MathSqrt(occ), 0.20, 1.00);
         double mag = 0.0;
         for(int c=0; c<FXAI_PA_CLASS_COUNT; c++) mag += MathAbs(m_hw1[c][i]);
         if(mag > 9.0) target *= 0.85; // prune unstable overloaded buckets
         m_hash_bw1[i] = FXAI_Clamp(0.98 * m_hash_bw1[i] + 0.02 * target, 0.15, 1.10);
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         double occ = (double)MathMax(1, m_hash_occ2[i]);
         double target = FXAI_Clamp(1.0 / MathSqrt(occ), 0.20, 1.00);
         double mag = 0.0;
         for(int c=0; c<FXAI_PA_CLASS_COUNT; c++) mag += MathAbs(m_hw2[c][i]);
         if(mag > 7.0) target *= 0.85;
         m_hash_bw2[i] = FXAI_Clamp(0.98 * m_hash_bw2[i] + 0.02 * target, 0.15, 1.10);
      }
   }

   double ComputeDiffNorm2(const double &x[],
                           const int cls,
                           const int rival) const
   {
      double n2 = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double xi = x[i];
         double sig = m_sigma[cls][i] + m_sigma[rival][i];
         n2 += sig * xi * xi;
      }
      if(m_use_hash)
      {
         for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               double v1 = m_hash_bw1[h1] * hv;
               n2 += (m_hsigma1[cls][h1] + m_hsigma1[rival][h1]) * v1 * v1;
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  double v2 = m_hash2_scale * m_hash_bw2[h2] * hv;
                  n2 += (m_hsigma2[cls][h2] + m_hsigma2[rival][h2]) * v2 * v2;
               }
            }
         }
      }
      if(n2 <= 1e-9) n2 = 1e-9;
      return n2;
   }

   int SelectPAMode(const double loss,
                    const double excess_points) const
   {
      if(m_drift_cooldown > 0) return 1; // PA-II under drift
      if(m_steps < 192) return 0;        // PA-I early
      if(loss > 1.20 && excess_points > 0.0) return 2; // MIRA for hard directional
      if(m_loss_ready && m_loss_fast > 1.18 * MathMax(0.05, m_loss_slow)) return 1;
      return 0;
   }

   double ComputeTau(const int mode,
                     const double loss,
                     const double c,
                     const double norm2) const
   {
      if(loss <= 0.0) return 0.0;
      double tau = 0.0;
      switch(mode)
      {
         case 1: // PA-II
            tau = loss / (2.0 * norm2 + (1.0 / (2.0 * c)));
            break;
         case 2: // MIRA-like capped
            tau = MathMin(c, loss / (2.0 * norm2 + 1e-9));
            break;
         case 0: // PA-I
         default:
            tau = MathMin(c, loss / (2.0 * norm2 + 1e-9));
            break;
      }
      return FXAI_Clamp(tau, 0.0, c);
   }

   void BuildCalLogits(const double &p_raw[],
                       double &logits[]) const
   {
      double lraw[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         double z = m_cal3_b[c];
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
            z += m_cal3_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

   void PredictMoveDistRaw(const double &x[],
                           double &mu,
                           double &logv) const
   {
      mu = 0.0;
      logv = m_mv_lv_b;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         mu += m_mv_mu_w[i] * x[i];
         logv += m_mv_lv_w[i] * x[i];
      }
      if(m_use_hash)
      {
         for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               double v1 = m_hash_bw1[h1] * hv;
               mu += m_mv_mu_hw1[h1] * v1;
               logv += m_mv_lv_hw1[h1] * v1;
               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  double v2 = m_hash2_scale * m_hash_bw2[h2] * hv;
                  mu += m_mv_mu_hw2[h2] * v2;
                  logv += m_mv_lv_hw2[h2] * v2;
               }
            }
         }
      }
      if(mu < 0.0) mu = 0.0;
      logv = FXAI_Clamp(logv, -4.0, 4.0);
   }

   void PushReplay(const int cls,
                   const double &x[],
                   const double move_points,
                   const double sample_w,
                   const double hardness)
   {
      int p = m_rep_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_rep_x[p][i] = x[i];
      m_rep_cls[p] = cls;
      m_rep_move[p] = move_points;
      m_rep_w[p] = sample_w;
      m_rep_hard[p] = FXAI_Clamp(hardness, 0.0, 20.0);
      m_rep_head++;
      if(m_rep_head >= FXAI_PA_REPLAY) m_rep_head = 0;
      if(m_rep_size < FXAI_PA_REPLAY) m_rep_size++;
   }

   int PickHardReplay(void)
   {
      if(m_rep_size <= 0) return -1;
      int best = PluginRandIndex(m_rep_size);
      double best_h = m_rep_hard[best];
      for(int k=1; k<6; k++)
      {
         int idx = PluginRandIndex(m_rep_size);
         if(m_rep_hard[idx] > best_h)
         {
            best = idx;
            best_h = m_rep_hard[idx];
         }
      }
      return best;
   }

   void UpdateABGuard(const double ce_live,
                      const double ce_avg)
   {
      if(!m_guard_ready)
      {
         m_guard_live_fast = ce_live;
         m_guard_live_slow = ce_live;
         m_guard_avg_fast = ce_avg;
         m_guard_avg_slow = ce_avg;
         m_guard_use_avg = false;
         m_guard_ready = true;
         return;
      }
      m_guard_live_fast = 0.92 * m_guard_live_fast + 0.08 * ce_live;
      m_guard_live_slow = 0.995 * m_guard_live_slow + 0.005 * ce_live;
      m_guard_avg_fast = 0.92 * m_guard_avg_fast + 0.08 * ce_avg;
      m_guard_avg_slow = 0.995 * m_guard_avg_slow + 0.005 * ce_avg;

      if(m_steps > 64)
      {
         if(m_guard_avg_fast < 0.985 * m_guard_live_fast && m_guard_avg_slow <= 1.03 * m_guard_live_slow)
            m_guard_use_avg = true;
         else if(m_guard_live_fast < 0.97 * m_guard_avg_fast)
            m_guard_use_avg = false;
      }
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

   double ScoreClass(const double &x[],
                     const int cls,
                     const bool averaged) const
   {
      if(cls < 0 || cls >= FXAI_PA_CLASS_COUNT) return 0.0;

      double z = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         z += (averaged ? m_w_avg[cls][i] : m_w[cls][i]) * x[i];

      if(!m_use_hash) return z;

      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
         {
            double hv = HashSign(i, j) * x[i] * x[j];
            int h1 = HashIndex(i, j);
            double v1 = m_hash_bw1[h1] * hv;
            z += (averaged ? m_hw1_avg[cls][h1] : m_hw1[cls][h1]) * v1;

            if(m_use_hash2)
            {
               int h2 = HashIndex2(i, j);
               double v2 = m_hash2_scale * m_hash_bw2[h2] * hv;
               z += 0.70 * (averaged ? m_hw2_avg[cls][h2] : m_hw2[cls][h2]) * v2;
            }
         }
      }
      return z;
   }

   void ComputeScores(const double &x[],
                      const bool averaged,
                      double &scores[]) const
   {
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         scores[c] = ScoreClass(x, c, averaged);
   }

   void ApplyDecay(const double decay)
   {
      if(decay <= 0.0) return;

      double k = 1.0 - 0.01 * decay;
      if(k < 0.90) k = 0.90;

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w[c][i] *= k;
            m_w_avg[c][i] *= k;
         }
         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] *= k;
            m_hw1_avg[c][i] *= k;
         }
         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] *= k;
            m_hw2_avg[c][i] *= k;
         }
      }

      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         m_mv_mu_w[i] *= k;
         m_mv_lv_w[i] *= k;
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_mv_mu_hw1[i] *= k;
         m_mv_lv_hw1[i] *= k;
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_mv_mu_hw2[i] *= k;
         m_mv_lv_hw2[i] *= k;
      }
   }

   double PredictMoveRaw(const double &x[]) const
   {
      double mu = 0.0, logv = 0.0;
      PredictMoveDistRaw(x, mu, logv);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      return MathMax(0.0, mu + 0.30 * sigma);
   }

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double logits[FXAI_PA_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_PA_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_PA_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_PA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_PA_CAL_BINS) bi = FXAI_PA_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
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
      double logits[FXAI_PA_CLASS_COUNT];
      BuildCalLogits(p_raw, logits);

      double p_cal[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double lraw[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FXAI_Clamp(0.20 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;

      // Decay old isotonic stats so calibration follows regime changes.
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            m_cal3_iso_cnt[c][b] *= 0.9995;
            m_cal3_iso_pos[c][b] *= 0.9995;
         }
      }

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_b[c] = FXAI_ClipSym(m_cal3_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double g = e * lraw[j] - reg_l2 * (m_cal3_w[c][j] - target_w);
            m_cal3_w[c][j] = FXAI_ClipSym(m_cal3_w[c][j] + cal_lr * g, 5.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_PA_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_PA_CAL_BINS) bi = FXAI_PA_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

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
                            const FXAIAIHyperParams &hp,
                            const double sample_w)
   {
      double target = MathAbs(move_points);
      if(!MathIsValidNumber(target)) return;

      double mu = 0.0, logv = 0.0;
      PredictMoveDistRaw(x, mu, logv);
      double var = MathMax(MathExp(logv), 0.05);
      double err = mu - target;
      double g_mu = err;
      if(g_mu > 2.0) g_mu = 2.0;
      if(g_mu < -2.0) g_mu = -2.0;
      g_mu /= MathMax(var, 0.25);
      double g_lv = 0.5 * (1.0 - (err * err) / MathMax(var, 0.25));
      g_lv = FXAI_ClipSym(g_lv, 2.0);

      double w = FXAI_Clamp(sample_w, 0.25, 6.00);
      double lr = FXAI_Clamp(0.02 * hp.lr * (0.85 + 0.15 * w), 0.00005, 0.02000);
      double wd = FXAI_Clamp(0.25 * hp.l2, 0.0, 0.0500);

      double g2 = 0.0;
      double g_mu_lin[FXAI_AI_WEIGHTS], g_lv_lin[FXAI_AI_WEIGHTS];
      double g_mu_h1[FXAI_ENHASH_BUCKETS], g_lv_h1[FXAI_ENHASH_BUCKETS];
      double g_mu_h2[FXAI_PA_HASH2_BUCKETS], g_lv_h2[FXAI_PA_HASH2_BUCKETS];

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         g_mu_lin[i] = FXAI_ClipSym(w * g_mu * x[i], 6.0);
         g_lv_lin[i] = FXAI_ClipSym(w * g_lv * x[i], 6.0);
         g2 += g_mu_lin[i] * g_mu_lin[i] + g_lv_lin[i] * g_lv_lin[i];
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++) { g_mu_h1[i] = 0.0; g_lv_h1[i] = 0.0; }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++) { g_mu_h2[i] = 0.0; g_lv_h2[i] = 0.0; }

      if(m_use_hash)
      {
         for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         {
            for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
            {
               double hv = HashSign(i, j) * x[i] * x[j];
               int h1 = HashIndex(i, j);
               double v1 = m_hash_bw1[h1] * hv;
               g_mu_h1[h1] += FXAI_ClipSym(w * g_mu * v1, 6.0);
               g_lv_h1[h1] += FXAI_ClipSym(w * g_lv * v1, 6.0);

               if(m_use_hash2)
               {
                  int h2 = HashIndex2(i, j);
                  double v2 = m_hash2_scale * m_hash_bw2[h2] * hv;
                  g_mu_h2[h2] += FXAI_ClipSym(w * g_mu * v2, 6.0);
                  g_lv_h2[h2] += FXAI_ClipSym(w * g_lv * v2, 6.0);
               }
            }
         }
      }

      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++) g2 += g_mu_h1[i] * g_mu_h1[i] + g_lv_h1[i] * g_lv_h1[i];
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++) g2 += g_mu_h2[i] * g_mu_h2[i] + g_lv_h2[i] * g_lv_h2[i];

      double gscale = 1.0;
      if(g2 > 0.0)
      {
         double gn = MathSqrt(g2);
         double clip = 6.0;
         if(gn > clip) gscale = clip / gn;
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         if(i != 0)
         {
            m_mv_mu_w[i] *= (1.0 - lr * wd);
            m_mv_lv_w[i] *= (1.0 - lr * wd);
         }
         m_mv_mu_w[i] -= lr * (g_mu_lin[i] * gscale);
         m_mv_lv_w[i] -= lr * (g_lv_lin[i] * gscale);
         m_mv_mu_w[i] = FXAI_ClipSym(m_mv_mu_w[i], 20.0);
         m_mv_lv_w[i] = FXAI_ClipSym(m_mv_lv_w[i], 20.0);
      }
      m_mv_lv_b = FXAI_ClipSym(m_mv_lv_b - lr * (w * g_lv) * gscale, 4.0);

      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_mv_mu_hw1[i] *= (1.0 - lr * wd);
         m_mv_lv_hw1[i] *= (1.0 - lr * wd);
         m_mv_mu_hw1[i] -= lr * (g_mu_h1[i] * gscale);
         m_mv_lv_hw1[i] -= lr * (g_lv_h1[i] * gscale);
         m_mv_mu_hw1[i] = FXAI_ClipSym(m_mv_mu_hw1[i], 15.0);
         m_mv_lv_hw1[i] = FXAI_ClipSym(m_mv_lv_hw1[i], 15.0);
      }

      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_mv_mu_hw2[i] *= (1.0 - lr * wd);
         m_mv_lv_hw2[i] *= (1.0 - lr * wd);
         m_mv_mu_hw2[i] -= lr * (g_mu_h2[i] * gscale);
         m_mv_lv_hw2[i] -= lr * (g_lv_h2[i] * gscale);
         m_mv_mu_hw2[i] = FXAI_ClipSym(m_mv_mu_hw2[i], 12.0);
         m_mv_lv_hw2[i] = FXAI_ClipSym(m_mv_lv_hw2[i], 12.0);
      }

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      m_mv_steps++;
   }

   void UpdateWeighted(const int cls,
                       const double &x[],
                       const FXAIAIHyperParams &hp,
                       const double margin_scale,
                       const double sample_w,
                       const double move_points,
                       const bool from_replay)
   {
      if(cls < 0 || cls >= FXAI_PA_CLASS_COUNT) return;

      m_steps++;

      // Class balance and recency weighting for online stability.
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FXAI_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);
      double recency = 0.85 + 0.30 * (1.0 - MathExp(-(double)m_steps / 512.0));
      double w = FXAI_Clamp(sample_w * cls_bal * recency, 0.10, 6.00);

      // Adaptive confidence prior by current volatility context.
      double vol_proxy = MathAbs(x[1]) + 0.5 * MathAbs(x[2]) + 0.35 * MathAbs(x[3]);
      double conf_target = FXAI_Clamp(0.50 + vol_proxy, 0.25, 4.00);
      m_conf_r = FXAI_Clamp(0.995 * m_conf_r + 0.005 * conf_target, 0.20, 5.00);

      double c = FXAI_Clamp(hp.pa_c, 0.001, 100.0);
      double margin0 = FXAI_Clamp(hp.pa_margin, 0.05, 4.0);
      double margin = FXAI_Clamp(margin0 * margin_scale, 0.05, 8.0);
      double decay = FXAI_Clamp(hp.l2, 0.0, 0.2);

      if(from_replay)
      {
         c = FXAI_Clamp(c * 0.85, 0.001, 100.0);
         margin = FXAI_Clamp(margin * 1.05, 0.05, 8.0);
      }
      if(m_drift_cooldown > 0)
      {
         c = FXAI_Clamp(c * 0.70, 0.001, 100.0);
         margin = FXAI_Clamp(margin * 1.10, 0.05, 8.0);
      }

      ApplyDecay(decay);
      UpdateCollisionRebalance();

      double scores[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, false, scores);

      int rivals[FXAI_PA_TOP_RIVALS];
      double rival_scores[FXAI_PA_TOP_RIVALS];
      for(int k=0; k<FXAI_PA_TOP_RIVALS; k++)
      {
         rivals[k] = -1;
         rival_scores[k] = -DBL_MAX;
      }
      for(int cidx=0; cidx<FXAI_PA_CLASS_COUNT; cidx++)
      {
         if(cidx == cls) continue;
         if(scores[cidx] > rival_scores[0])
         {
            rival_scores[1] = rival_scores[0];
            rivals[1] = rivals[0];
            rival_scores[0] = scores[cidx];
            rivals[0] = cidx;
         }
         else if(scores[cidx] > rival_scores[1])
         {
            rival_scores[1] = scores[cidx];
            rivals[1] = cidx;
         }
      }
      if(rivals[0] < 0) return;

      double total_hard = 0.0;
      for(int rk=0; rk<FXAI_PA_TOP_RIVALS; rk++)
      {
         int rival = rivals[rk];
         if(rival < 0) continue;
         double loss = margin - scores[cls] + scores[rival];
         if(loss <= 0.0) continue;
         total_hard += loss;

         double excess = MathMax(0.0, MathAbs(move_points) - InputCostProxyPoints(x));
         int pa_mode = SelectPAMode(loss, excess);
         m_pa_mode = pa_mode;

         double norm2 = ComputeDiffNorm2(x, cls, rival);
         double tau = ComputeTau(pa_mode, loss, c, norm2);
         tau *= w;
         if(rk == 1) tau *= 0.60;
         if(m_drift_cooldown > 0) tau *= 0.80;
         tau = FXAI_Clamp(tau, 0.0, c);
         if(tau <= 0.0) continue;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double xi = x[i];
            double s_cls = m_sigma[cls][i];
            double s_riv = m_sigma[rival][i];
            m_w[cls][i] += tau * s_cls * xi;
            m_w[rival][i] -= tau * s_riv * xi;

            double x2 = xi * xi;
            m_sigma[cls][i] = 1.0 / (1.0 / MathMax(s_cls, 1e-6) + x2 / MathMax(m_conf_r, 1e-3));
            m_sigma[rival][i] = 1.0 / (1.0 / MathMax(s_riv, 1e-6) + x2 / MathMax(m_conf_r, 1e-3));
            m_sigma[cls][i] = FXAI_Clamp(m_sigma[cls][i], 1e-5, 10.0);
            m_sigma[rival][i] = FXAI_Clamp(m_sigma[rival][i], 1e-5, 10.0);
         }

         if(m_use_hash)
         {
            for(int i=1; i<FXAI_AI_WEIGHTS; i++)
            {
               for(int j=i+1; j<FXAI_AI_WEIGHTS; j++)
               {
                  double hv = HashSign(i, j) * x[i] * x[j];
                  int h1 = HashIndex(i, j);
                  double v1 = m_hash_bw1[h1] * hv;

                  double s1c = m_hsigma1[cls][h1];
                  double s1r = m_hsigma1[rival][h1];
                  m_hw1[cls][h1] += tau * s1c * v1;
                  m_hw1[rival][h1] -= tau * s1r * v1;

                  double v12 = v1 * v1;
                  m_hsigma1[cls][h1] = 1.0 / (1.0 / MathMax(s1c, 1e-6) + v12 / MathMax(m_conf_r, 1e-3));
                  m_hsigma1[rival][h1] = 1.0 / (1.0 / MathMax(s1r, 1e-6) + v12 / MathMax(m_conf_r, 1e-3));
                  m_hsigma1[cls][h1] = FXAI_Clamp(m_hsigma1[cls][h1], 1e-5, 10.0);
                  m_hsigma1[rival][h1] = FXAI_Clamp(m_hsigma1[rival][h1], 1e-5, 10.0);

                  if(m_use_hash2)
                  {
                     int h2 = HashIndex2(i, j);
                     double v2 = m_hash2_scale * m_hash_bw2[h2] * hv;
                     double s2c = m_hsigma2[cls][h2];
                     double s2r = m_hsigma2[rival][h2];
                     m_hw2[cls][h2] += tau * s2c * v2;
                     m_hw2[rival][h2] -= tau * s2r * v2;

                     double v22 = v2 * v2;
                     m_hsigma2[cls][h2] = 1.0 / (1.0 / MathMax(s2c, 1e-6) + v22 / MathMax(m_conf_r, 1e-3));
                     m_hsigma2[rival][h2] = 1.0 / (1.0 / MathMax(s2r, 1e-6) + v22 / MathMax(m_conf_r, 1e-3));
                     m_hsigma2[cls][h2] = FXAI_Clamp(m_hsigma2[cls][h2], 1e-5, 10.0);
                     m_hsigma2[rival][h2] = FXAI_Clamp(m_hsigma2[rival][h2], 1e-5, 10.0);
                  }
               }
            }
         }
      }

      // Averaged-PA weights.
      double beta = 1.0 / (double)m_steps;
      for(int cidx=0; cidx<FXAI_PA_CLASS_COUNT; cidx++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_w_avg[cidx][i] += beta * (m_w[cidx][i] - m_w_avg[cidx][i]);
         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
            m_hw1_avg[cidx][i] += beta * (m_hw1[cidx][i] - m_hw1_avg[cidx][i]);
         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
            m_hw2_avg[cidx][i] += beta * (m_hw2[cidx][i] - m_hw2_avg[cidx][i]);
      }

      double scores_post[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, false, scores_post);
      double scores_avg[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, true, scores_avg);

      int rival_best = -1;
      double rival_best_score = -DBL_MAX;
      for(int cidx=0; cidx<FXAI_PA_CLASS_COUNT; cidx++)
      {
         if(cidx == cls) continue;
         if(scores_post[cidx] > rival_best_score)
         {
            rival_best_score = scores_post[cidx];
            rival_best = cidx;
         }
      }
      if(rival_best < 0) return;
      double margin_gap = scores_post[cls] - scores_post[rival_best];
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
      double logits[FXAI_PA_CLASS_COUNT];
      for(int cidx=0; cidx<FXAI_PA_CLASS_COUNT; cidx++)
         logits[cidx] = FXAI_ClipSym(scores_post[cidx] / den, 20.0);
      double p_raw[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);

      double logits_avg[FXAI_PA_CLASS_COUNT];
      for(int cidx=0; cidx<FXAI_PA_CLASS_COUNT; cidx++)
         logits_avg[cidx] = FXAI_ClipSym(scores_avg[cidx] / den, 20.0);
      double p_raw_avg[FXAI_PA_CLASS_COUNT];
      Softmax3(logits_avg, p_raw_avg);

      double ce = -MathLog(FXAI_Clamp(p_raw[cls], 1e-6, 1.0));
      double ce_avg = -MathLog(FXAI_Clamp(p_raw_avg[cls], 1e-6, 1.0));
      UpdateABGuard(ce, ce_avg);
      UpdateLossDrift(ce);
      double cal_lr = FXAI_Clamp(0.01 * MathSqrt(w), 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      // Keep legacy binary calibrator aligned for compatibility paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      UpdateMoveHeadJoint(x, move_points, hp, w);

      if(!from_replay)
      {
         double hardness = ce + 0.30 * total_hard;
         PushReplay(cls, x, move_points, w, hardness);
      }
   }

public:
   CFXAIAIPA(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_PA_LINEAR; }
   virtual string AIName(void) const { return "lin_pa"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_LINEAR, caps, 1, 1);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      bool averaged = (m_guard_ready ? m_guard_use_avg : (m_steps > 24));
      double scores[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, averaged, scores);

      double den = (m_margin_ready ? m_margin_ema : FXAI_Clamp(hp.pa_margin, 0.25, 4.0));
      if(den < 0.25) den = 0.25;

      double logits[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         logits[c] = FXAI_ClipSym(scores[c] / den, 20.0);

      double p_raw[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points < 0.0)
         expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      bool averaged = (m_guard_ready ? m_guard_use_avg : (m_steps > 24));
      double scores[FXAI_PA_CLASS_COUNT];
      ComputeScores(x, averaged, scores);
      double den = (m_margin_ready ? m_margin_ema : FXAI_Clamp(hp.pa_margin, 0.25, 4.0));
      if(den < 0.25) den = 0.25;
      double logits[FXAI_PA_CLASS_COUNT];
      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
         logits[c] = FXAI_ClipSym(scores[c] / den, 20.0);
      double p_raw[FXAI_PA_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double mu = 0.0, logv = 0.0;
      PredictMoveDistRaw(x, mu, logv);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double head = PredictMoveRaw(x);
      if(head <= 0.0 && m_move_ready) head = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, head);
      out.move_q25_points = MathMax(0.0, MathMax(0.0, mu - 0.60 * sigma));
      out.move_q50_points = MathMax(out.move_q25_points, MathMax(0.0, mu));
      out.move_q75_points = MathMax(out.move_q50_points, MathMax(0.0, mu + 0.60 * sigma));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * (m_move_ready ? 1.0 : 0.0) + 0.30 * MathMin((double)m_mv_steps / 64.0, 1.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_quality_heads.Reset();

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
      m_pa_mode = 0;
      m_conf_r = 1.0;

      m_guard_ready = false;
      m_guard_use_avg = false;
      m_guard_live_fast = 0.0;
      m_guard_live_slow = 0.0;
      m_guard_avg_fast = 0.0;
      m_guard_avg_slow = 0.0;

      m_hash_occ_ready = false;
      m_hash2_scale = 1.0;
      m_rep_head = 0;
      m_rep_size = 0;

      m_mv_lv_b = 0.0;
      m_cal3_steps = 0;

      for(int c=0; c<FXAI_PA_CLASS_COUNT; c++)
      {
         m_cls_ema[c] = 1.0;
         m_cal3_b[c] = 0.0;
         for(int j=0; j<FXAI_PA_CLASS_COUNT; j++)
            m_cal3_w[c][j] = (c == j ? 1.0 : 0.0);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w[c][i] = 0.0;
            m_w_avg[c][i] = 0.0;
            m_sigma[c][i] = 1.0;
         }

         for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
         {
            m_hw1[c][i] = 0.0;
            m_hw1_avg[c][i] = 0.0;
            m_hsigma1[c][i] = 1.0;
         }

         for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
         {
            m_hw2[c][i] = 0.0;
            m_hw2_avg[c][i] = 0.0;
            m_hsigma2[c][i] = 1.0;
         }

         for(int b=0; b<FXAI_PA_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_mv_mu_w[i] = 0.0;
         m_mv_lv_w[i] = 0.0;
      }
      for(int i=0; i<FXAI_ENHASH_BUCKETS; i++)
      {
         m_mv_mu_hw1[i] = 0.0;
         m_mv_lv_hw1[i] = 0.0;
      }
      for(int i=0; i<FXAI_PA_HASH2_BUCKETS; i++)
      {
         m_mv_mu_hw2[i] = 0.0;
         m_mv_lv_hw2[i] = 0.0;
      }

      for(int i=0; i<FXAI_PA_REPLAY; i++)
      {
         m_rep_cls[i] = (int)FXAI_LABEL_SKIP;
         m_rep_move[i] = 0.0;
         m_rep_w[i] = 1.0;
         m_rep_hard[i] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_rep_x[i][k] = 0.0;
      }

      BuildCollisionProfile();
   }

   // Legacy compatibility path.
   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      UpdateWeighted(cls, x, hp, 1.0, 1.0, pseudo_move, false);
   }

protected:
   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost_proxy = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double excess = MathMax(0.0, abs_move - cost_proxy);

      double edge_ratio = excess / MathMax(cost_proxy, 0.50);

      // Explicit skip EV prior.
      if(cls == (int)FXAI_LABEL_SKIP)
      {
         double skip_prior = FXAI_Clamp(1.35 - 0.30 * edge_ratio, 0.25, 1.50);
         if(edge_ratio > 1.8 && abs_move > 1.2 * MathMax(cost_proxy, 0.5))
            cls = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
         else
            edge_ratio *= skip_prior;
      }
      else
      {
         if(edge_ratio < 0.05) cls = (int)FXAI_LABEL_SKIP;
      }

      double ev_w = FXAI_Clamp(0.35 + edge_ratio, 0.10, 6.00);
      if(cls == (int)FXAI_LABEL_SKIP)
         ev_w *= FXAI_Clamp(1.20 - 0.20 * edge_ratio, 0.25, 1.40);
      else
         ev_w *= FXAI_Clamp(0.65 + 0.40 * edge_ratio, 0.40, 2.50);

      double move_scale = FXAI_Clamp(1.0 + 0.10 * excess, 0.70, 3.50);
      double margin_scale = FXAI_Clamp(move_scale * (1.0 + 0.25 * (ev_w - 1.0)), 0.60, 4.00);
      // Regime/session-aware margin schedule.
      datetime t = ResolveContextTime();
      if(t <= 0) t = TimeCurrent();
      int sess = SessionBucket(t);
      double sess_scale = 1.0;
      if(sess == 0) sess_scale = 1.10;
      else if(sess == 1) sess_scale = 0.95;
      else if(sess == 2) sess_scale = 0.92;
      else sess_scale = 1.05;

      double vol_proxy = MathAbs(x[1]) + 0.7 * MathAbs(x[2]) + 0.5 * MathAbs(x[3]);
      double regime_scale = 1.0;
      if(vol_proxy < 0.20) regime_scale = 1.10;
      else if(vol_proxy > 2.00) regime_scale = 0.90;

      double spread_scale = FXAI_Clamp(1.0 + 0.08 * (cost_proxy - 1.0), 0.80, 1.30);
      margin_scale = FXAI_Clamp(margin_scale * sess_scale * regime_scale * spread_scale, 0.50, 5.00);
      UpdateNativeQualityHeads(x, ev_w, h.lr, h.l2);

      UpdateWeighted(cls, x, h, margin_scale, ev_w, move_points, false);

      // Hard-example replay.
      int replay_n = 0;
      if(m_rep_size >= 96) replay_n = 2;
      else if(m_rep_size >= 24) replay_n = 1;
      for(int r=0; r<replay_n; r++)
      {
         int idx = PickHardReplay();
         if(idx < 0) break;
         double rw = FXAI_Clamp(0.75 * m_rep_w[idx], 0.10, 4.00);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            replay_x[i] = m_rep_x[idx][i];
         UpdateWeighted(m_rep_cls[idx], replay_x, h, margin_scale, rw, m_rep_move[idx], true);
      }
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_PA_CLASS_COUNT];
      double expected = -1.0;
      if(!PredictModelCore(x, hp, probs, expected))
         return 0.5;

      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double head = PredictMoveRaw(x);

      if(m_mv_steps >= 24 && m_move_ready && m_move_ema_abs > 0.0)
         head = 0.70 * head + 0.30 * m_move_ema_abs;

      if(head > 0.0) return head;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};

#endif // __FXAI_AI_PA_MQH__
