// FXAI v1
#ifndef __FX6_AI_QUANTILE_MQH__
#define __FX6_AI_QUANTILE_MQH__

#include "..\plugin_base.mqh"

#define FX6_QT_Q      9
#define FX6_QT_MID    4
#define FX6_QT_SIDE   4
#define FX6_QT_SESS   4
#define FX6_QT_REG    3
#define FX6_QT_ZF     8
#define FX6_QT_CALF   8

class CFX6AIQuantile : public CFX6AIPlugin
{
private:
   bool   m_initialized;

   // Quantile levels: 5%..95% (dense tails for FX).
   double m_tau[FX6_QT_Q];

   // Short-horizon head.
   double m_w_med_s[FX6_AI_WEIGHTS];
   double m_w_up_s[FX6_QT_SIDE][FX6_AI_WEIGHTS];
   double m_w_dn_s[FX6_QT_SIDE][FX6_AI_WEIGHTS];

   // Medium-horizon head.
   double m_w_med_m[FX6_AI_WEIGHTS];
   double m_w_up_m[FX6_QT_SIDE][FX6_AI_WEIGHTS];
   double m_w_dn_m[FX6_QT_SIDE][FX6_AI_WEIGHTS];

   // AdaGrad accumulators.
   double m_g2_med_s[FX6_AI_WEIGHTS];
   double m_g2_up_s[FX6_QT_SIDE][FX6_AI_WEIGHTS];
   double m_g2_dn_s[FX6_QT_SIDE][FX6_AI_WEIGHTS];

   double m_g2_med_m[FX6_AI_WEIGHTS];
   double m_g2_up_m[FX6_QT_SIDE][FX6_AI_WEIGHTS];
   double m_g2_dn_m[FX6_QT_SIDE][FX6_AI_WEIGHTS];

   // Session/regime conditioning.
   double m_sess_bias[2][FX6_QT_SESS];
   double m_g2_sess_bias[2][FX6_QT_SESS];

   double m_reg_scale[2][FX6_QT_REG];
   double m_g2_reg_scale[2][FX6_QT_REG];

   // Native 3-class head on quantile features.
   double m_cls_w[3][FX6_QT_ZF];
   double m_cls_g2[3][FX6_QT_ZF];

   // Learned direction calibrator over quantile features.
   double m_cal_w_q[FX6_QT_CALF];
   double m_cal_g2_q[FX6_QT_CALF];

   // Diagnostics and adaptive reliability.
   int    m_diag_n;
   double m_pit_mean;
   double m_pit_m2;
   double m_cross_ema;
   double m_cal_err_ema;
   double m_rel_weight;

   bool   m_medium_ready;
   double m_medium_target_ema;

   double DotVec(const double &w[], const double &x[]) const
   {
      double z = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         z += w[i] * x[i];
      return z;
   }

   double DotClass(const int c, const double &zf[]) const
   {
      double z = 0.0;
      for(int i=0; i<FX6_QT_ZF; i++)
         z += m_cls_w[c][i] * zf[i];
      return z;
   }

   double DotCal(const double &cf[]) const
   {
      double z = 0.0;
      for(int i=0; i<FX6_QT_CALF; i++)
         z += m_cal_w_q[i] * cf[i];
      return z;
   }

   double Softplus(const double z) const
   {
      if(z > 30.0) return z;
      if(z < -30.0) return MathExp(z);
      return MathLog(1.0 + MathExp(z));
   }

   double SoftplusPrime(const double z) const
   {
      return FX6_Sigmoid(z);
   }

   int SessionBucket(const datetime t) const
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      int h = dt.hour;
      if(h >= 0 && h < 6) return 0;   // Asia early
      if(h >= 6 && h < 12) return 1;  // Europe
      if(h >= 12 && h < 20) return 2; // US overlap
      return 3;                       // late session/off hours
   }

   int RegimeBucket(const double &x[]) const
   {
      // x[6] ~ volatility z-score, x[7] ~ cost/spread feature in this project.
      double vol = 0.0;
      if(ArraySize(x) > 6) vol = MathAbs(x[6]);
      double cost = InputCostProxyPoints(x);
      double score = vol + (0.20 * cost);

      if(score < 0.90) return 0;
      if(score < 1.80) return 1;
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

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      for(int c=1; c<3; c++) if(logits[c] > m) m = logits[c];

      double den = 0.0;
      for(int c=0; c<3; c++)
      {
         probs[c] = MathExp(FX6_ClipSym(logits[c] - m, 30.0));
         den += probs[c];
      }
      if(den <= 0.0) den = 1.0;
      for(int c=0; c<3; c++) probs[c] /= den;
   }

   void BuildHead(const int h,
                  const double &x[],
                  const int sess,
                  const int reg,
                  double &q[]) const
   {
      double med = 0.0;
      double scale = 1.0;

      if(h == 0)
      {
         med = DotVec(m_w_med_s, x) + m_sess_bias[0][sess];
         scale = FX6_Clamp(m_reg_scale[0][reg], 0.40, 2.80);
      }
      else
      {
         med = DotVec(m_w_med_m, x) + m_sess_bias[1][sess];
         scale = FX6_Clamp(m_reg_scale[1][reg], 0.40, 2.80);
      }

      q[FX6_QT_MID] = med;

      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         double z_up = 0.0;
         double z_dn = 0.0;

         if(h == 0)
         {
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               z_up += m_w_up_s[j][i] * x[i];
               z_dn += m_w_dn_s[j][i] * x[i];
            }
         }
         else
         {
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               z_up += m_w_up_m[j][i] * x[i];
               z_dn += m_w_dn_m[j][i] * x[i];
            }
         }

         double gap_up = scale * Softplus(z_up);
         double gap_dn = scale * Softplus(z_dn);

         q[FX6_QT_MID + 1 + j] = q[FX6_QT_MID + j] - 0.0 + gap_up;
         q[FX6_QT_MID - 1 - j] = q[FX6_QT_MID - j] - gap_dn;
      }

      // Hard monotonic guard for numerical safety.
      for(int k=1; k<FX6_QT_Q; k++)
      {
         if(q[k] < q[k - 1] + 1e-4)
            q[k] = q[k - 1] + 1e-4;
      }
   }

   void BuildBlendedQuantiles(const double &x[],
                              const int sess,
                              const int reg,
                              double &q_short[],
                              double &q_medium[],
                              double &q_blend[]) const
   {
      BuildHead(0, x, sess, reg, q_short);
      BuildHead(1, x, sess, reg, q_medium);

      double ws = 0.55;
      if(reg == 0) ws = 0.45;
      if(reg == 2) ws = 0.70;

      for(int k=0; k<FX6_QT_Q; k++)
         q_blend[k] = (ws * q_short[k]) + ((1.0 - ws) * q_medium[k]);

      for(int k=1; k<FX6_QT_Q; k++)
      {
         if(q_blend[k] < q_blend[k - 1] + 1e-4)
            q_blend[k] = q_blend[k - 1] + 1e-4;
      }
   }

   double QuantileGradSignal(const double target,
                             const double pred,
                             const double tau,
                             const double huber_delta) const
   {
      double err = target - pred;
      double dir = (err >= 0.0 ? tau : (tau - 1.0));
      double aerr = MathAbs(err);
      if(aerr <= huber_delta)
         return dir;
      return dir * (huber_delta / aerr);
   }

   void ApplyMedUpdate(const int h,
                       const double &x[],
                       const double grad_scalar,
                       const double lr,
                       const double wd)
   {
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double g = grad_scalar * x[i];
         if(h == 0)
         {
            m_g2_med_s[i] += g * g;
            double step = lr / MathSqrt(m_g2_med_s[i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_med_s[i] *= (1.0 - step * wd);
            m_w_med_s[i] += step * g;
            m_w_med_s[i] = FX6_ClipSym(m_w_med_s[i], 10.0);
         }
         else
         {
            m_g2_med_m[i] += g * g;
            double step = lr / MathSqrt(m_g2_med_m[i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_med_m[i] *= (1.0 - step * wd);
            m_w_med_m[i] += step * g;
            m_w_med_m[i] = FX6_ClipSym(m_w_med_m[i], 10.0);
         }
      }
   }

   void ApplyUpUpdate(const int h,
                      const int j,
                      const double &x[],
                      const double grad_scalar,
                      const double lr,
                      const double wd)
   {
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double g = grad_scalar * x[i];
         if(h == 0)
         {
            m_g2_up_s[j][i] += g * g;
            double step = lr / MathSqrt(m_g2_up_s[j][i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_up_s[j][i] *= (1.0 - step * wd);
            m_w_up_s[j][i] += step * g;
            m_w_up_s[j][i] = FX6_ClipSym(m_w_up_s[j][i], 10.0);
         }
         else
         {
            m_g2_up_m[j][i] += g * g;
            double step = lr / MathSqrt(m_g2_up_m[j][i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_up_m[j][i] *= (1.0 - step * wd);
            m_w_up_m[j][i] += step * g;
            m_w_up_m[j][i] = FX6_ClipSym(m_w_up_m[j][i], 10.0);
         }
      }
   }

   void ApplyDnUpdate(const int h,
                      const int j,
                      const double &x[],
                      const double grad_scalar,
                      const double lr,
                      const double wd)
   {
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double g = grad_scalar * x[i];
         if(h == 0)
         {
            m_g2_dn_s[j][i] += g * g;
            double step = lr / MathSqrt(m_g2_dn_s[j][i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_dn_s[j][i] *= (1.0 - step * wd);
            m_w_dn_s[j][i] += step * g;
            m_w_dn_s[j][i] = FX6_ClipSym(m_w_dn_s[j][i], 10.0);
         }
         else
         {
            m_g2_dn_m[j][i] += g * g;
            double step = lr / MathSqrt(m_g2_dn_m[j][i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_w_dn_m[j][i] *= (1.0 - step * wd);
            m_w_dn_m[j][i] += step * g;
            m_w_dn_m[j][i] = FX6_ClipSym(m_w_dn_m[j][i], 10.0);
         }
      }
   }

   void ApplyScalarUpdate(double &w,
                          double &g2,
                          const double grad,
                          const double lr,
                          const double wd,
                          const double lo,
                          const double hi)
   {
      g2 += grad * grad;
      double step = lr / MathSqrt(g2 + 1e-8);
      if(wd > 0.0)
         w *= (1.0 - step * wd);
      w += step * grad;
      w = FX6_Clamp(w, lo, hi);
   }

   void BuildClassFeatures(const double &q_short[],
                           const double &q_medium[],
                           const double &q_blend[],
                           const double cost,
                           const double &x[],
                           double &zf[]) const
   {
      double spread_s = q_short[FX6_QT_Q - 1] - q_short[0];
      if(spread_s < 0.05) spread_s = 0.05;

      double spread_m = q_medium[FX6_QT_Q - 1] - q_medium[0];
      if(spread_m < 0.05) spread_m = 0.05;

      double spread_b = q_blend[FX6_QT_Q - 1] - q_blend[0];
      if(spread_b < 0.05) spread_b = 0.05;

      double ev_buy = 0.5 * (q_blend[FX6_QT_MID] + q_blend[FX6_QT_Q - 1]) - cost;
      double ev_sell = 0.5 * (-q_blend[FX6_QT_MID] - q_blend[0]) - cost;

      zf[0] = 1.0;
      zf[1] = FX6_ClipSym(q_short[FX6_QT_MID] / spread_s, 8.0);
      zf[2] = FX6_ClipSym(spread_s, 20.0);
      zf[3] = FX6_ClipSym(q_medium[FX6_QT_MID] / spread_m, 8.0);
      zf[4] = FX6_ClipSym(spread_m, 20.0);
      zf[5] = FX6_ClipSym((ev_buy - ev_sell) / spread_b, 8.0);
      zf[6] = FX6_ClipSym(cost, 12.0);
      zf[7] = FX6_ClipSym((ArraySize(x) > 6 ? MathAbs(x[6]) : 0.0), 8.0);
   }

   void PredictClassProbs(const double &zf[], double &probs[]) const
   {
      double logits[3];
      for(int c=0; c<3; c++)
         logits[c] = DotClass(c, zf);
      Softmax3(logits, probs);
   }

   void UpdateClassHead(const int cls,
                        const double &zf[],
                        const double sample_w,
                        const double lr,
                        const double wd,
                        double &probs_out[])
   {
      double probs[3];
      PredictClassProbs(zf, probs);
      for(int c=0; c<3; c++) probs_out[c] = probs[c];

      double zf_norm2 = 0.0;
      for(int i=0; i<FX6_QT_ZF; i++) zf_norm2 += zf[i] * zf[i];
      if(zf_norm2 < 1.0) zf_norm2 = 1.0;

      double gc[3];
      double norm2 = 0.0;
      for(int c=0; c<3; c++)
      {
         double t = (c == cls ? 1.0 : 0.0);
         gc[c] = sample_w * (t - probs[c]);
         norm2 += gc[c] * gc[c] * zf_norm2;
      }

      double gnorm = MathSqrt(norm2);
      double clip = 6.0;
      double gscale = (gnorm > clip && gnorm > 1e-9 ? clip / gnorm : 1.0);

      for(int c=0; c<3; c++)
      {
         double gcs = gc[c] * gscale;
         for(int i=0; i<FX6_QT_ZF; i++)
         {
            double g = gcs * zf[i];
            m_cls_g2[c][i] += g * g;
            double step = lr / MathSqrt(m_cls_g2[c][i] + 1e-8);
            if(i > 0 && wd > 0.0)
               m_cls_w[c][i] *= (1.0 - step * wd);
            m_cls_w[c][i] += step * g;
            m_cls_w[c][i] = FX6_ClipSym(m_cls_w[c][i], 10.0);
         }
      }
   }

   void BuildCalFeatures(const double &q_blend[],
                         const double &class_probs[],
                         const double cost,
                         const double &x[],
                         double &cf[]) const
   {
      double spread = q_blend[FX6_QT_Q - 1] - q_blend[0];
      if(spread < 0.05) spread = 0.05;

      double ev_buy = 0.5 * (q_blend[FX6_QT_MID] + q_blend[FX6_QT_Q - 1]) - cost;
      double ev_sell = 0.5 * (-q_blend[FX6_QT_MID] - q_blend[0]) - cost;

      cf[0] = 1.0;
      cf[1] = FX6_ClipSym(q_blend[FX6_QT_MID] / spread, 8.0);
      cf[2] = FX6_ClipSym((ev_buy - ev_sell) / spread, 8.0);
      cf[3] = FX6_ClipSym(class_probs[(int)FX6_LABEL_BUY] - class_probs[(int)FX6_LABEL_SELL], 1.0);
      cf[4] = FX6_ClipSym(class_probs[(int)FX6_LABEL_SKIP], 1.0);
      cf[5] = FX6_ClipSym(cost, 12.0);
      cf[6] = FX6_ClipSym((ArraySize(x) > 6 ? MathAbs(x[6]) : 0.0), 8.0);
      cf[7] = FX6_ClipSym(1.0 / (spread + 0.10), 8.0);
   }

   double PredictDirectionCal(const double &cf[]) const
   {
      return FX6_Sigmoid(DotCal(cf));
   }

   void UpdateDirectionCalibrator(const int y_dir,
                                  const double &cf[],
                                  const double sample_w,
                                  const double lr,
                                  const double wd)
   {
      double p = PredictDirectionCal(cf);
      double err = sample_w * ((double)y_dir - p);

      double cf_norm2 = 0.0;
      for(int i=0; i<FX6_QT_CALF; i++) cf_norm2 += cf[i] * cf[i];
      if(cf_norm2 < 1.0) cf_norm2 = 1.0;

      double gnorm = MathSqrt(err * err * cf_norm2);
      double gscale = (gnorm > 4.0 && gnorm > 1e-9 ? 4.0 / gnorm : 1.0);
      double es = err * gscale;

      for(int i=0; i<FX6_QT_CALF; i++)
      {
         double g = es * cf[i];
         m_cal_g2_q[i] += g * g;
         double step = lr / MathSqrt(m_cal_g2_q[i] + 1e-8);
         if(i > 0 && wd > 0.0)
            m_cal_w_q[i] *= (1.0 - step * wd);
         m_cal_w_q[i] += step * g;
         m_cal_w_q[i] = FX6_ClipSym(m_cal_w_q[i], 10.0);
      }
   }

   double ApproxPIT(const double target,
                    const double &q[]) const
   {
      if(target <= q[0])
         return 0.01;
      if(target >= q[FX6_QT_Q - 1])
         return 0.99;

      for(int k=0; k<FX6_QT_Q - 1; k++)
      {
         if(target >= q[k] && target <= q[k + 1])
         {
            double span = q[k + 1] - q[k];
            if(span < 1e-9) span = 1e-9;
            double u = (target - q[k]) / span;
            double pit = m_tau[k] + (m_tau[k + 1] - m_tau[k]) * u;
            return FX6_Clamp(pit, 0.001, 0.999);
         }
      }
      return 0.50;
   }

   void UpdateDiagnostics(const double move_points,
                          const double &q_blend[],
                          const int cls,
                          const double &class_probs[])
   {
      double pit = ApproxPIT(move_points, q_blend);

      m_diag_n++;
      double dn = (double)m_diag_n;
      double d = pit - m_pit_mean;
      m_pit_mean += d / dn;
      m_pit_m2 += d * (pit - m_pit_mean);

      int crossing = 0;
      for(int k=1; k<FX6_QT_Q; k++)
      {
         if(q_blend[k] < q_blend[k - 1])
         {
            crossing = 1;
            break;
         }
      }
      m_cross_ema = (0.98 * m_cross_ema) + (0.02 * (double)crossing);

      int c = cls;
      if(c < 0 || c > 2) c = (int)FX6_LABEL_SKIP;
      double p_true = FX6_Clamp(class_probs[c], 0.0, 1.0);
      double cal_err = 1.0 - p_true;
      m_cal_err_ema = (0.98 * m_cal_err_ema) + (0.02 * cal_err);

      if(m_diag_n >= 25)
      {
         double var = (m_diag_n > 1 ? (m_pit_m2 / (double)(m_diag_n - 1)) : (1.0 / 12.0));
         double pit_pen = MathAbs(m_pit_mean - 0.5) + (6.0 * MathAbs(var - (1.0 / 12.0)));
         double drift = pit_pen + (0.8 * m_cross_ema) + (0.9 * m_cal_err_ema);
         double target_rel = FX6_Clamp(1.25 - drift, 0.35, 1.50);
         m_rel_weight = FX6_Clamp((0.97 * m_rel_weight) + (0.03 * target_rel), 0.25, 1.75);
      }
   }

   void TrainQuantileHead(const int h,
                          const double &x[],
                          const int sess,
                          const int reg,
                          const double target,
                          const double sample_w,
                          const double lr,
                          const double wd)
   {
      double q[FX6_QT_Q];
      BuildHead(h, x, sess, reg, q);

      // Raw delta states are needed for monotonic-parameterized gradients.
      double z_up[FX6_QT_SIDE], z_dn[FX6_QT_SIDE];
      double sp_up[FX6_QT_SIDE], sp_dn[FX6_QT_SIDE];
      double sig_up[FX6_QT_SIDE], sig_dn[FX6_QT_SIDE];
      double scale = FX6_Clamp(m_reg_scale[h][reg], 0.40, 2.80);

      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         if(h == 0)
         {
            double zu = 0.0, zd = 0.0;
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               zu += m_w_up_s[j][i] * x[i];
               zd += m_w_dn_s[j][i] * x[i];
            }
            z_up[j] = zu;
            z_dn[j] = zd;
         }
         else
         {
            double zu = 0.0, zd = 0.0;
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               zu += m_w_up_m[j][i] * x[i];
               zd += m_w_dn_m[j][i] * x[i];
            }
            z_up[j] = zu;
            z_dn[j] = zd;
         }

         sp_up[j] = Softplus(z_up[j]);
         sp_dn[j] = Softplus(z_dn[j]);
         sig_up[j] = SoftplusPrime(z_up[j]);
         sig_dn[j] = SoftplusPrime(z_dn[j]);
      }

      double gq[FX6_QT_Q];
      const double huber_delta = 5.0;
      for(int k=0; k<FX6_QT_Q; k++)
      {
         gq[k] = sample_w * QuantileGradSignal(target, q[k], m_tau[k], huber_delta);
         gq[k] = FX6_ClipSym(gq[k], 2.0);
      }

      double g_med = 0.0;
      for(int k=0; k<FX6_QT_Q; k++) g_med += gq[k];

      double g_up[FX6_QT_SIDE];
      double g_dn[FX6_QT_SIDE];
      double g_scale = 0.0;

      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         double sum_up = 0.0;
         for(int k=FX6_QT_MID + 1 + j; k<FX6_QT_Q; k++)
            sum_up += gq[k];

         double sum_dn = 0.0;
         for(int k=0; k<=FX6_QT_MID - 1 - j; k++)
            sum_dn += gq[k];

         g_up[j] = scale * sig_up[j] * sum_up;
         g_dn[j] = -scale * sig_dn[j] * sum_dn;

         g_scale += (sp_up[j] * sum_up) - (sp_dn[j] * sum_dn);
      }

      double x_norm2 = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++) x_norm2 += x[i] * x[i];
      if(x_norm2 < 1.0) x_norm2 = 1.0;

      double norm2 = g_med * g_med * x_norm2;
      norm2 += g_scale * g_scale;
      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         norm2 += g_up[j] * g_up[j] * x_norm2;
         norm2 += g_dn[j] * g_dn[j] * x_norm2;
      }

      double gnorm = MathSqrt(norm2);
      double clip = 8.0;
      double gscale_clip = (gnorm > clip && gnorm > 1e-9 ? clip / gnorm : 1.0);

      g_med *= gscale_clip;
      g_scale *= gscale_clip;
      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         g_up[j] *= gscale_clip;
         g_dn[j] *= gscale_clip;
      }

      ApplyMedUpdate(h, x, g_med, lr, wd);
      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         ApplyUpUpdate(h, j, x, g_up[j], lr, wd);
         ApplyDnUpdate(h, j, x, g_dn[j], lr, wd);
      }

      ApplyScalarUpdate(m_sess_bias[h][sess],
                        m_g2_sess_bias[h][sess],
                        g_med,
                        0.60 * lr,
                        0.10 * wd,
                        -8.0,
                        8.0);

      ApplyScalarUpdate(m_reg_scale[h][reg],
                        m_g2_reg_scale[h][reg],
                        g_scale,
                        0.40 * lr,
                        0.00,
                        0.30,
                        3.20);
   }

public:
   CFX6AIQuantile(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_QUANTILE; }
   virtual string AIName(void) const { return "quantile"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;

      m_tau[0] = 0.05;
      m_tau[1] = 0.15;
      m_tau[2] = 0.25;
      m_tau[3] = 0.35;
      m_tau[4] = 0.50;
      m_tau[5] = 0.65;
      m_tau[6] = 0.75;
      m_tau[7] = 0.85;
      m_tau[8] = 0.95;

      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_w_med_s[i] = 0.0;
         m_w_med_m[i] = 0.0;
         m_g2_med_s[i] = 0.0;
         m_g2_med_m[i] = 0.0;
      }

      for(int j=0; j<FX6_QT_SIDE; j++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_w_up_s[j][i] = 0.0;
            m_w_dn_s[j][i] = 0.0;
            m_w_up_m[j][i] = 0.0;
            m_w_dn_m[j][i] = 0.0;

            m_g2_up_s[j][i] = 0.0;
            m_g2_dn_s[j][i] = 0.0;
            m_g2_up_m[j][i] = 0.0;
            m_g2_dn_m[j][i] = 0.0;
         }
      }

      for(int h=0; h<2; h++)
      {
         for(int s=0; s<FX6_QT_SESS; s++)
         {
            m_sess_bias[h][s] = 0.0;
            m_g2_sess_bias[h][s] = 0.0;
         }
         for(int r=0; r<FX6_QT_REG; r++)
         {
            m_reg_scale[h][r] = 1.0;
            m_g2_reg_scale[h][r] = 0.0;
         }
      }

      for(int c=0; c<3; c++)
      {
         for(int i=0; i<FX6_QT_ZF; i++)
         {
            m_cls_w[c][i] = 0.0;
            m_cls_g2[c][i] = 0.0;
         }
      }
      // Prior toward skip in early training to reduce overtrading.
      m_cls_w[(int)FX6_LABEL_SKIP][0] = 0.20;

      for(int i=0; i<FX6_QT_CALF; i++)
      {
         m_cal_w_q[i] = 0.0;
         m_cal_g2_q[i] = 0.0;
      }

      m_diag_n = 0;
      m_pit_mean = 0.50;
      m_pit_m2 = 0.0;
      m_cross_ema = 0.0;
      m_cal_err_ema = 0.0;
      m_rel_weight = 1.0;

      m_medium_ready = false;
      m_medium_target_ema = 0.0;
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
         m_initialized = true;
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FX6AIHyperParams &hp)
   {
      double pseudo_move = (y == (int)FX6_LABEL_BUY ? 1.0 : (y == (int)FX6_LABEL_SELL ? -1.0 : 0.0));
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);

      double cost = InputCostProxyPoints(x);
      int cls = ResolveClass(y, move_points, cost);

      if(!m_medium_ready)
      {
         m_medium_target_ema = move_points;
         m_medium_ready = true;
      }
      else
      {
         m_medium_target_ema = (0.98 * m_medium_target_ema) + (0.02 * move_points);
      }

      int sess = SessionBucket(ResolveContextTime());
      int reg = RegimeBucket(x);

      double lr = FX6_Clamp(hp.quantile_lr, 0.00005, 0.05000);
      double wd = FX6_Clamp(hp.quantile_l2, 0.00000, 0.05000);

      if(m_rel_weight < 0.75)
         lr *= 0.85;

      double sw = FX6_Clamp(MoveSampleWeight(x, move_points) * m_rel_weight, 0.10, 6.00);

      // Short and medium horizon quantile heads.
      TrainQuantileHead(0, x, sess, reg, move_points, sw, lr, wd);
      TrainQuantileHead(1, x, sess, reg, m_medium_target_ema, 0.60 * sw, 0.80 * lr, 0.80 * wd);

      double q_short[FX6_QT_Q], q_medium[FX6_QT_Q], q_blend[FX6_QT_Q];
      BuildBlendedQuantiles(x, sess, reg, q_short, q_medium, q_blend);

      // Native 3-class training.
      double zf[FX6_QT_ZF];
      BuildClassFeatures(q_short, q_medium, q_blend, cost, x, zf);
      double class_probs[3];
      UpdateClassHead(cls, zf, sw, 0.70 * lr, 0.50 * wd, class_probs);

      // Learned direction calibration over quantile/class features.
      double cf[FX6_QT_CALF];
      BuildCalFeatures(q_blend, class_probs, cost, x, cf);
      int y_dir = (cls == (int)FX6_LABEL_BUY ? 1 : (cls == (int)FX6_LABEL_SELL ? 0 : (move_points >= 0.0 ? 1 : 0)));
      double w_dir = (cls == (int)FX6_LABEL_SKIP ? 0.35 * sw : sw);
      UpdateDirectionCalibrator(y_dir, cf, w_dir, 0.50 * lr, 0.30 * wd);

      // Plugin-level calibration anchor for directional probability.
      double den = class_probs[(int)FX6_LABEL_BUY] + class_probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = class_probs[(int)FX6_LABEL_BUY] / den;
      p_dir_raw = 0.65 * p_dir_raw + 0.35 * PredictDirectionCal(cf);
      UpdateCalibration(p_dir_raw, y_dir, w_dir);

      // Diagnostics/reliability tracking and auto-reweight.
      UpdateDiagnostics(move_points, q_blend, cls, class_probs);

      // Keep shared move head for ensemble comparability.
      UpdateMoveHead(x, move_points, hp, sw);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      int sess = SessionBucket(ResolveContextTime());
      int reg = RegimeBucket(x);
      double cost = InputCostProxyPoints(x);

      double q_short[FX6_QT_Q], q_medium[FX6_QT_Q], q_blend[FX6_QT_Q];
      BuildBlendedQuantiles(x, sess, reg, q_short, q_medium, q_blend);

      double zf[FX6_QT_ZF];
      BuildClassFeatures(q_short, q_medium, q_blend, cost, x, zf);
      double class_probs[3];
      PredictClassProbs(zf, class_probs);

      double spread = q_blend[FX6_QT_Q - 1] - q_blend[0];
      if(spread < 0.05) spread = 0.05;

      double buy_ev = 0.5 * (q_blend[FX6_QT_MID] + q_blend[FX6_QT_Q - 1]) - cost;
      double sell_ev = 0.5 * (-q_blend[FX6_QT_MID] - q_blend[0]) - cost;

      double den = class_probs[(int)FX6_LABEL_BUY] + class_probs[(int)FX6_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_cls = class_probs[(int)FX6_LABEL_BUY] / den;

      double p_dir_ev = FX6_Sigmoid(FX6_ClipSym((buy_ev - sell_ev) / (spread + 0.10), 12.0));

      double cf[FX6_QT_CALF];
      BuildCalFeatures(q_blend, class_probs, cost, x, cf);
      double p_dir_cal = PredictDirectionCal(cf);

      double p_dir = (0.45 * p_dir_cls) + (0.35 * p_dir_ev) + (0.20 * p_dir_cal);
      p_dir = FX6_Clamp(p_dir, 0.001, 0.999);

      double p_skip = FX6_Clamp(class_probs[(int)FX6_LABEL_SKIP], 0.0, 0.98);
      if(buy_ev <= 0.0 && sell_ev <= 0.0)
         p_skip = MathMax(p_skip, 0.70);

      double p_up_raw = p_dir * (1.0 - p_skip);
      p_up_raw = FX6_Clamp(p_up_raw, 0.001, 0.999);
      return CalibrateProb(p_up_raw);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      int sess = SessionBucket(ResolveContextTime());
      int reg = RegimeBucket(x);
      double cost = InputCostProxyPoints(x);

      double q_short[FX6_QT_Q], q_medium[FX6_QT_Q], q_blend[FX6_QT_Q];
      BuildBlendedQuantiles(x, sess, reg, q_short, q_medium, q_blend);

      double zf[FX6_QT_ZF];
      BuildClassFeatures(q_short, q_medium, q_blend, cost, x, zf);
      double class_probs[3];
      PredictClassProbs(zf, class_probs);

      double buy_tail = 0.5 * MathMax(0.0, q_blend[FX6_QT_Q - 1] - cost)
                      + 0.5 * MathMax(0.0, q_blend[FX6_QT_Q - 2] - cost);

      double sell_tail = 0.5 * MathMax(0.0, -q_blend[0] - cost)
                       + 0.5 * MathMax(0.0, -q_blend[1] - cost);

      double ev = class_probs[(int)FX6_LABEL_BUY] * buy_tail
                + class_probs[(int)FX6_LABEL_SELL] * sell_tail;

      double amp = 0.5 * (MathAbs(q_blend[FX6_QT_Q - 1]) + MathAbs(q_blend[0]));
      double pred = MathMax(ev, 0.35 * amp);
      pred *= FX6_Clamp(m_rel_weight, 0.50, 1.50);

      double base = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      if(pred > 0.0 && base > 0.0)
         return 0.65 * pred + 0.35 * base;
      if(pred > 0.0)
         return pred;
      return base;
   }
};

#endif // __FX6_AI_QUANTILE_MQH__
