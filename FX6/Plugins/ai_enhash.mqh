#ifndef __FX6_AI_ENHASH_MQH__
#define __FX6_AI_ENHASH_MQH__

#include "..\plugin_base.mqh"

#define FX6_ENH_CLASSES 3
#define FX6_ENH_TABLES 2
#define FX6_ENH_FIELDS 5
#define FX6_ENH_FPAIRS (FX6_ENH_FIELDS * FX6_ENH_FIELDS)
#define FX6_ENH_SESSIONS 4
#define FX6_ENH_REGIMES 3
#define FX6_ENH_REHASH_PERIOD 512

class CFX6AIENHash : public CFX6AIPlugin
{
private:
   // Native 3-class FTRL-prox weights.
   double m_z_lin[FX6_ENH_CLASSES][FX6_AI_WEIGHTS];
   double m_n_lin[FX6_ENH_CLASSES][FX6_AI_WEIGHTS];
   double m_w_lin[FX6_ENH_CLASSES][FX6_AI_WEIGHTS];

   // Field-aware hashed interactions with dual-table rehash windows.
   double m_z_hash[FX6_ENH_CLASSES][FX6_ENH_TABLES][FX6_ENH_FPAIRS][FX6_ENHASH_BUCKETS];
   double m_n_hash[FX6_ENH_CLASSES][FX6_ENH_TABLES][FX6_ENH_FPAIRS][FX6_ENHASH_BUCKETS];
   double m_w_hash[FX6_ENH_CLASSES][FX6_ENH_TABLES][FX6_ENH_FPAIRS][FX6_ENHASH_BUCKETS];

   // Continuous collision telemetry.
   double m_bucket_use_ema[FX6_ENH_TABLES][FX6_ENH_FPAIRS][FX6_ENHASH_BUCKETS];
   double m_pair_collision_ema[FX6_ENH_FPAIRS];

   // Session/regime conditioning.
   double m_bias[FX6_ENH_CLASSES][FX6_ENH_REGIMES][FX6_ENH_SESSIONS];
   double m_bias_g2[FX6_ENH_CLASSES][FX6_ENH_REGIMES][FX6_ENH_SESSIONS];

   // Lightweight online normalization.
   bool   m_norm_ready;
   int    m_norm_steps;
   double m_x_mean[FX6_AI_WEIGHTS];
   double m_x_var[FX6_AI_WEIGHTS];

   // Diagnostics-driven self tuning.
   int    m_step;
   uint   m_seed_a;
   uint   m_seed_b;
   double m_diag_collision_ema;
   double m_diag_calerr_ema;
   double m_diag_edgehit_ema;
   double m_diag_uncert_ema;

   // Interaction uncertainty statistics for expected-move head.
   int    m_inter_n;
   double m_inter_mean;
   double m_inter_m2;

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      int h = dt.hour;
      if(h >= 0 && h < 6) return 0;
      if(h >= 6 && h < 12) return 1;
      if(h >= 12 && h < 20) return 2;
      return 3;
   }

   int RegimeBucket(const double &x[]) const
   {
      double vol = 0.0;
      if(ArraySize(x) > 6)
         vol = MathAbs(x[6]);
      double cost = InputCostProxyPoints(x);
      double s = vol + (0.20 * cost);
      if(s < 0.90) return 0;
      if(s < 1.80) return 1;
      return 2;
   }

   int ResolveClass(const int y,
                    const double move_points,
                    const double cost_points) const
   {
      if(y >= (int)FX6_LABEL_SELL && y <= (int)FX6_LABEL_SKIP)
         return y;

      double edge = MathAbs(move_points) - MathMax(cost_points, 0.0);
      double skip_band = 0.10 + (0.25 * MathMax(cost_points, 0.0));
      if(edge <= skip_band)
         return (int)FX6_LABEL_SKIP;

      if(y > 0) return (int)FX6_LABEL_BUY;
      if(y == 0) return (int)FX6_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
   }

   int FeatureField(const int idx) const
   {
      if(idx <= 3)  return 0;
      if(idx <= 6)  return 1;
      if(idx <= 8)  return 2;
      if(idx <= 12) return 3;
      return 4;
   }

   int FieldPairIndex(const int fa_in, const int fb_in) const
   {
      int fa = fa_in;
      int fb = fb_in;
      if(fa > fb)
      {
         int t = fa;
         fa = fb;
         fb = t;
      }
      return (fa * FX6_ENH_FIELDS + fb);
   }

   int HashIndex(const int table,
                 const int fpair,
                 const int i,
                 const int j) const
   {
      uint seed = (table == 0 ? m_seed_a : m_seed_b);
      uint h = ((uint)(i * 73856093u)) ^ ((uint)(j * 19349663u)) ^ ((uint)(fpair * 83492791u)) ^ seed;
      return (int)(h % (uint)FX6_ENHASH_BUCKETS);
   }

   double HashSign(const int table,
                   const int fpair,
                   const int i,
                   const int j) const
   {
      uint h = ((uint)(i * 31u)) ^ ((uint)(j * 17u)) ^ ((uint)(fpair * 131u)) ^ ((uint)(table * 2654435761u));
      return ((h & 1u) == 0u ? 1.0 : -1.0);
   }

   double FTRLWeight(const double z,
                     const double n,
                     const double alpha,
                     const double beta,
                     const double l1,
                     const double l2) const
   {
      double az = MathAbs(z);
      if(az <= l1) return 0.0;
      double denom = (beta + MathSqrt(n)) / alpha + l2;
      if(denom <= 1e-12) return 0.0;
      return -(z - FX6_Sign(z) * l1) / denom;
   }

   void FTRLUpdate(double &z,
                   double &n,
                   double &w,
                   const double g,
                   const double alpha,
                   const double beta,
                   const double l1,
                   const double l2)
   {
      double gn = g;
      if(!MathIsValidNumber(gn)) return;
      if(gn > 10.0) gn = 10.0;
      if(gn < -10.0) gn = -10.0;

      double n_old = n;
      double n_new = n_old + gn * gn;
      double sigma = (MathSqrt(n_new) - MathSqrt(n_old)) / alpha;
      z += gn - sigma * w;
      n = n_new;
      w = FTRLWeight(z, n, alpha, beta, l1, l2);
      w = FX6_ClipSym(w, 8.0);
   }

   void ResetNorm(void)
   {
      m_norm_ready = false;
      m_norm_steps = 0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateNorm(const double &x[])
   {
      double a = (m_norm_steps < 128 ? 0.04 : 0.010);
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_norm_steps++;
      if(m_norm_steps >= 32) m_norm_ready = true;
   }

   void NormalizeInput(const double &x[],
                       const bool update_stats,
                       double &xn[])
   {
      if(update_stats)
         UpdateNorm(x);

      xn[0] = 1.0;
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         double v = x[i];
         if(m_norm_ready)
         {
            double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
            v = (x[i] - m_x_mean[i]) * inv;
         }
         xn[i] = FX6_ClipSym(v, 6.0);
      }
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      for(int c=1; c<FX6_ENH_CLASSES; c++)
         if(logits[c] > m) m = logits[c];

      double den = 0.0;
      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         probs[c] = MathExp(FX6_ClipSym(logits[c] - m, 30.0));
         den += probs[c];
      }
      if(den <= 0.0) den = 1.0;
      for(int c=0; c<FX6_ENH_CLASSES; c++) probs[c] /= den;
   }

   void EvalModel(const double &xn[],
                  const int sess,
                  const int reg,
                  const bool touch_usage,
                  double &probs[],
                  double &p_dir_raw,
                  double &p_skip,
                  double &inter_amp,
                  double &inter_std,
                  double &collision_metric)
   {
      double logits[FX6_ENH_CLASSES];
      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         double z = m_bias[c][reg][sess];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            z += m_w_lin[c][i] * xn[i];
         logits[c] = z;
      }

      double diff_sum = 0.0;
      double diff_sum2 = 0.0;
      double coll_sum = 0.0;
      int pair_n = 0;
      inter_amp = 0.0;

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            double hv0 = xn[i] * xn[j];
            if(MathAbs(hv0) < 1e-12) continue;

            int fi = FeatureField(i);
            int fj = FeatureField(j);
            int fp = FieldPairIndex(fi, fj);

            int idx0 = HashIndex(0, fp, i, j);
            int idx1 = HashIndex(1, fp, i, j);

            double u0 = m_bucket_use_ema[0][fp][idx0];
            double u1 = m_bucket_use_ema[1][fp][idx1];

            double inv0 = 1.0 / (0.05 + u0);
            double inv1 = 1.0 / (0.05 + u1);
            double invs = inv0 + inv1;
            if(invs <= 1e-12) invs = 1.0;

            // Continuous collision-aware weighting.
            double pair_scale = 1.0 / (1.0 + 0.75 * (u0 + u1));
            double a0 = pair_scale * (inv0 / invs);
            double a1 = pair_scale * (inv1 / invs);

            if(touch_usage)
            {
               m_bucket_use_ema[0][fp][idx0] = 0.995 * m_bucket_use_ema[0][fp][idx0] + 0.005;
               m_bucket_use_ema[1][fp][idx1] = 0.995 * m_bucket_use_ema[1][fp][idx1] + 0.005;
               double cpair = 0.5 * (u0 + u1);
               m_pair_collision_ema[fp] = 0.995 * m_pair_collision_ema[fp] + 0.005 * cpair;
            }

            double v0 = HashSign(0, fp, i, j) * hv0;
            double v1 = HashSign(1, fp, i, j) * hv0;

            double d_buy = 0.0;
            double d_sell = 0.0;

            for(int c=0; c<FX6_ENH_CLASSES; c++)
            {
               double c0 = a0 * m_w_hash[c][0][fp][idx0] * v0;
               double c1 = a1 * m_w_hash[c][1][fp][idx1] * v1;
               logits[c] += c0 + c1;
               if(c == (int)FX6_LABEL_BUY) d_buy = c0 + c1;
               if(c == (int)FX6_LABEL_SELL) d_sell = c0 + c1;
            }

            double dd = d_buy - d_sell;
            diff_sum += dd;
            diff_sum2 += dd * dd;
            inter_amp += 0.5 * (MathAbs(d_buy) + MathAbs(d_sell));

            coll_sum += 0.5 * (u0 + u1);
            pair_n++;
         }
      }

      Softmax3(logits, probs);

      p_skip = probs[(int)FX6_LABEL_SKIP];
      double den = probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL];
      if(den <= 1e-9) den = 1e-9;
      p_dir_raw = probs[(int)FX6_LABEL_BUY] / den;

      if(pair_n > 0)
      {
         inter_amp /= (double)pair_n;
         double md = diff_sum / (double)pair_n;
         double vd = diff_sum2 / (double)pair_n - md * md;
         if(vd < 0.0) vd = 0.0;
         inter_std = MathSqrt(vd);
         collision_metric = FX6_Clamp(coll_sum / (double)pair_n, 0.0, 1.0);
      }
      else
      {
         inter_amp = 0.0;
         inter_std = 0.0;
         collision_metric = 0.0;
      }
   }

   void AdaptHyperParams(const FX6AIHyperParams &hp,
                         double &alpha,
                         double &beta,
                         double &l1,
                         double &l2) const
   {
      alpha = FX6_Clamp(hp.enhash_lr, 0.0002, 0.1000);
      beta  = 1.0 + 4.0 * FX6_Clamp(hp.enhash_l2, 0.0000, 0.1000);
      l1    = FX6_Clamp(hp.enhash_l1, 0.0000, 0.1000);
      l2    = FX6_Clamp(hp.enhash_l2, 0.0000, 0.1000);

      // Diagnostics-driven self tuning.
      double penalty = 0.45 * m_diag_collision_ema + 0.35 * m_diag_calerr_ema + 0.20 * m_diag_uncert_ema;
      double reward  = 0.30 * (m_diag_edgehit_ema - 0.50);

      alpha *= FX6_Clamp(1.0 - penalty + reward, 0.35, 1.50);
      beta  *= FX6_Clamp(1.0 + 0.80 * m_diag_collision_ema, 0.80, 3.00);
      l1    *= FX6_Clamp(1.0 + 1.50 * m_diag_collision_ema, 0.80, 3.00);
      l2    *= FX6_Clamp(1.0 + m_diag_calerr_ema + 0.80 * m_diag_collision_ema, 0.80, 3.50);

      alpha = FX6_Clamp(alpha, 0.00005, 0.1500);
      beta  = FX6_Clamp(beta, 0.25000, 12.0000);
      l1    = FX6_Clamp(l1, 0.00000, 0.2000);
      l2    = FX6_Clamp(l2, 0.00000, 0.2000);
   }

   void RotateSecondaryTable(void)
   {
      m_seed_b = m_seed_b * 1664525u + 1013904223u;

      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         for(int fp=0; fp<FX6_ENH_FPAIRS; fp++)
         {
            for(int b=0; b<FX6_ENHASH_BUCKETS; b++)
            {
               // Rehash window: fade secondary table and refill under new seed.
               m_w_hash[c][1][fp][b] *= 0.75;
               m_z_hash[c][1][fp][b] *= 0.75;
               m_n_hash[c][1][fp][b] *= 0.90;

               m_bucket_use_ema[1][fp][b] *= 0.50;
               m_bucket_use_ema[0][fp][b] *= 0.98;
            }
         }
      }
   }

public:
   CFX6AIENHash(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_ENHASH; }
   virtual string AIName(void) const { return "enhash"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();

      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_z_lin[c][i] = 0.0;
            m_n_lin[c][i] = 0.0;
            m_w_lin[c][i] = 0.0;
         }
      }

      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         for(int t=0; t<FX6_ENH_TABLES; t++)
         {
            for(int fp=0; fp<FX6_ENH_FPAIRS; fp++)
            {
               for(int b=0; b<FX6_ENHASH_BUCKETS; b++)
               {
                  m_z_hash[c][t][fp][b] = 0.0;
                  m_n_hash[c][t][fp][b] = 0.0;
                  m_w_hash[c][t][fp][b] = 0.0;
               }
            }
         }
      }

      for(int t=0; t<FX6_ENH_TABLES; t++)
      {
         for(int fp=0; fp<FX6_ENH_FPAIRS; fp++)
         {
            m_pair_collision_ema[fp] = 0.0;
            for(int b=0; b<FX6_ENHASH_BUCKETS; b++)
               m_bucket_use_ema[t][fp][b] = 0.0;
         }
      }

      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         for(int r=0; r<FX6_ENH_REGIMES; r++)
         {
            for(int s=0; s<FX6_ENH_SESSIONS; s++)
            {
               m_bias[c][r][s] = 0.0;
               m_bias_g2[c][r][s] = 0.0;
            }
         }
      }
      // Conservative start: small skip prior.
      for(int r=0; r<FX6_ENH_REGIMES; r++)
         for(int s=0; s<FX6_ENH_SESSIONS; s++)
            m_bias[(int)FX6_LABEL_SKIP][r][s] = 0.20;

      ResetNorm();

      m_step = 0;
      m_seed_a = 2166136261u;
      m_seed_b = 3735928559u;

      m_diag_collision_ema = 0.0;
      m_diag_calerr_ema = 0.0;
      m_diag_edgehit_ema = 0.50;
      m_diag_uncert_ema = 0.0;

      m_inter_n = 0;
      m_inter_mean = 0.0;
      m_inter_m2 = 0.0;
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      double pseudo_move = (y == (int)FX6_LABEL_BUY ? 1.0 : (y == (int)FX6_LABEL_SELL ? -1.0 : 0.0));
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, true, xn);

      int sess = SessionBucket(ResolveContextTime());
      int reg  = RegimeBucket(x);
      double cost = InputCostProxyPoints(x);
      int cls = ResolveClass(y, move_points, cost);

      double probs[FX6_ENH_CLASSES];
      double p_dir_raw = 0.5;
      double p_skip = 0.0;
      double inter_amp = 0.0;
      double inter_std = 0.0;
      double collision = 0.0;

      EvalModel(xn, sess, reg, true, probs, p_dir_raw, p_skip, inter_amp, inter_std, collision);

      double sw = FX6_Clamp(MoveSampleWeight(x, move_points), 0.10, 6.00);

      double alpha, beta, l1, l2;
      AdaptHyperParams(hp, alpha, beta, l1, l2);
      alpha *= FX6_Clamp(sw, 0.25, 4.00);
      alpha = FX6_Clamp(alpha, 0.00005, 0.1500);

      // FTRL-prox multiclass updates (native 3-class training).
      for(int c=0; c<FX6_ENH_CLASSES; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double gcls = FX6_ClipSym(sw * (target - probs[c]), 4.0);

         // Session/regime bias (lightweight AdaGrad update).
         m_bias_g2[c][reg][sess] += gcls * gcls;
         double lrb = alpha / MathSqrt(m_bias_g2[c][reg][sess] + 1e-8);
         m_bias[c][reg][sess] += lrb * gcls;
         m_bias[c][reg][sess] = FX6_ClipSym(m_bias[c][reg][sess], 8.0);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double g = gcls * xn[i];
            FTRLUpdate(m_z_lin[c][i], m_n_lin[c][i], m_w_lin[c][i], g, alpha, beta, l1, l2);
         }
      }

      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         for(int j=i+1; j<FX6_AI_WEIGHTS; j++)
         {
            double hv0 = xn[i] * xn[j];
            if(MathAbs(hv0) < 1e-12) continue;

            int fi = FeatureField(i);
            int fj = FeatureField(j);
            int fp = FieldPairIndex(fi, fj);

            int idx0 = HashIndex(0, fp, i, j);
            int idx1 = HashIndex(1, fp, i, j);

            double u0 = m_bucket_use_ema[0][fp][idx0];
            double u1 = m_bucket_use_ema[1][fp][idx1];

            double inv0 = 1.0 / (0.05 + u0);
            double inv1 = 1.0 / (0.05 + u1);
            double invs = inv0 + inv1;
            if(invs <= 1e-12) invs = 1.0;

            double pair_scale = 1.0 / (1.0 + 0.75 * (u0 + u1));
            double a0 = pair_scale * (inv0 / invs);
            double a1 = pair_scale * (inv1 / invs);

            double v0 = HashSign(0, fp, i, j) * hv0;
            double v1 = HashSign(1, fp, i, j) * hv0;

            for(int c=0; c<FX6_ENH_CLASSES; c++)
            {
               double target = (c == cls ? 1.0 : 0.0);
               double gcls = FX6_ClipSym(sw * (target - probs[c]), 4.0);

               double g0 = gcls * a0 * v0;
               double g1 = gcls * a1 * v1;

               FTRLUpdate(m_z_hash[c][0][fp][idx0], m_n_hash[c][0][fp][idx0], m_w_hash[c][0][fp][idx0], g0, alpha, beta, l1, l2);
               FTRLUpdate(m_z_hash[c][1][fp][idx1], m_n_hash[c][1][fp][idx1], m_w_hash[c][1][fp][idx1], g1, alpha, beta, l1, l2);
            }
         }
      }

      // Second-order confidence from interaction uncertainty.
      double conf = 1.0 / (1.0 + 0.60 * inter_std + 0.50 * collision + 0.20 * m_diag_calerr_ema);
      conf = FX6_Clamp(conf, 0.25, 1.00);
      double p_dir_conf = 0.5 + (p_dir_raw - 0.5) * conf;
      double p_up_raw = FX6_Clamp(p_dir_conf * (1.0 - p_skip), 0.001, 0.999);

      int y_dir = (cls == (int)FX6_LABEL_BUY ? 1 : (cls == (int)FX6_LABEL_SELL ? 0 : (move_points >= 0.0 ? 1 : 0)));
      double w_dir = (cls == (int)FX6_LABEL_SKIP ? 0.35 * sw : sw);
      UpdateCalibration(p_up_raw, y_dir, w_dir);

      // Diagnostics-driven online adaptation metrics.
      double cal_err = MathAbs((double)y_dir - p_dir_conf);
      m_diag_calerr_ema = 0.98 * m_diag_calerr_ema + 0.02 * cal_err;
      m_diag_collision_ema = 0.98 * m_diag_collision_ema + 0.02 * collision;
      m_diag_uncert_ema = 0.98 * m_diag_uncert_ema + 0.02 * FX6_Clamp(inter_std, 0.0, 2.0);

      int pred_cls = 0;
      for(int c=1; c<FX6_ENH_CLASSES; c++) if(probs[c] > probs[pred_cls]) pred_cls = c;

      double edge = MathAbs(move_points) - cost;
      double edge_hit = 0.0;
      if(pred_cls == cls)
      {
         if(cls == (int)FX6_LABEL_SKIP) edge_hit = (edge <= 0.0 ? 1.0 : 0.25);
         else edge_hit = (edge > 0.0 ? 1.0 : 0.0);
      }
      m_diag_edgehit_ema = 0.98 * m_diag_edgehit_ema + 0.02 * edge_hit;

      // Interaction-move uncertainty head statistics.
      double inter_score = inter_amp + 0.35 * inter_std;
      m_inter_n++;
      double d = inter_score - m_inter_mean;
      m_inter_mean += d / (double)m_inter_n;
      m_inter_m2 += d * (inter_score - m_inter_mean);

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(xn, move_points, hp, sw);

      m_step++;

      // Periodic rehash windows.
      if((m_step % FX6_ENH_REHASH_PERIOD) == 0 || m_diag_collision_ema > 0.65)
         RotateSecondaryTable();
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, false, xn);

      int sess = SessionBucket(ResolveContextTime());
      int reg  = RegimeBucket(x);

      double probs[FX6_ENH_CLASSES];
      double p_dir_raw = 0.5;
      double p_skip = 0.0;
      double inter_amp = 0.0;
      double inter_std = 0.0;
      double collision = 0.0;
      EvalModel(xn, sess, reg, false, probs, p_dir_raw, p_skip, inter_amp, inter_std, collision);

      double conf = 1.0 / (1.0 + 0.60 * inter_std + 0.50 * collision + 0.20 * m_diag_calerr_ema);
      conf = FX6_Clamp(conf, 0.25, 1.00);

      double p_dir_conf = 0.5 + (p_dir_raw - 0.5) * conf;
      double p_up_raw = FX6_Clamp(p_dir_conf * (1.0 - p_skip), 0.001, 0.999);
      return CalibrateProb(p_up_raw);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, false, xn);

      int sess = SessionBucket(ResolveContextTime());
      int reg  = RegimeBucket(x);

      double probs[FX6_ENH_CLASSES];
      double p_dir_raw = 0.5;
      double p_skip = 0.0;
      double inter_amp = 0.0;
      double inter_std = 0.0;
      double collision = 0.0;
      EvalModel(xn, sess, reg, false, probs, p_dir_raw, p_skip, inter_amp, inter_std, collision);

      double ev = (probs[(int)FX6_LABEL_BUY] + probs[(int)FX6_LABEL_SELL]) * (inter_amp + 0.35 * inter_std);
      if(m_inter_n > 1)
      {
         double var = m_inter_m2 / (double)(m_inter_n - 1);
         if(var > 0.0)
            ev += 0.20 * MathSqrt(var);
      }

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.60 * ev + 0.40 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return CFX6AIPlugin::PredictExpectedMovePoints(xn, hp);
   }
};

#endif // __FX6_AI_ENHASH_MQH__
