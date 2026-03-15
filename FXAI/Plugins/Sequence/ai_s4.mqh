#ifndef __FXAI_AI_S4_MQH__
#define __FXAI_AI_S4_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_S4_RANK 4
#define FXAI_S4_TBPTT 20
#define FXAI_S4_HORIZONS 3

class CFXAIAIS4 : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_train_steps;
   int    m_seen_updates;

   // Continuous-time diagonal + low-rank complex SSM parameters.
   // lambda_h = -exp(log_tau_h) + i*omega_h, A = diag(lambda) + U*V^T
   double m_log_tau[FXAI_AI_MLP_HIDDEN];
   double m_omega[FXAI_AI_MLP_HIDDEN];
   double m_log_dt[FXAI_AI_MLP_HIDDEN];

   double m_b_re[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_b_im[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];

   double m_u_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_u_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_v_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_v_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];

   // Output + skip heads.
   double m_c_re[FXAI_AI_MLP_HIDDEN];
   double m_c_im[FXAI_AI_MLP_HIDDEN];
   double m_d_skip[FXAI_AI_WEIGHTS];
   double m_b_out;

   // Plugin-level logit calibration.
   bool   m_margin_ready;
   double m_margin_ema;
   double m_prob_scale;
   double m_prob_bias;
   CFXAITernaryCalibrator m_cal3;

   // Distributional multi-horizon move heads: means for {1,3,5}, shared log-variance.
   double m_w_move_mu[FXAI_S4_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_b_move_mu[FXAI_S4_HORIZONS];
   double m_w_move_logv[FXAI_AI_MLP_HIDDEN];
   double m_b_move_logv;
   CFXAINativeQualityHeads m_quality_heads;

   // Recurrent complex state.
   double m_state_re[FXAI_AI_MLP_HIDDEN];
   double m_state_im[FXAI_AI_MLP_HIDDEN];

   // Input normalization stats.
   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   // State normalization stats.
   bool   m_s_norm_ready;
   int    m_s_norm_steps;
   double m_sr_mean[FXAI_AI_MLP_HIDDEN];
   double m_sr_var[FXAI_AI_MLP_HIDDEN];
   double m_si_mean[FXAI_AI_MLP_HIDDEN];
   double m_si_var[FXAI_AI_MLP_HIDDEN];

   // TBPTT buffer.
   int    m_batch_size;
   double m_batch_s0_re[FXAI_AI_MLP_HIDDEN];
   double m_batch_s0_im[FXAI_AI_MLP_HIDDEN];
   double m_batch_x[FXAI_S4_TBPTT][FXAI_AI_WEIGHTS];
   int    m_batch_y[FXAI_S4_TBPTT];
   double m_batch_move[FXAI_S4_TBPTT];
   double m_batch_w[FXAI_S4_TBPTT];

   // RMSProp accumulators for SSM core.
   double m_r_log_tau[FXAI_AI_MLP_HIDDEN];
   double m_r_omega[FXAI_AI_MLP_HIDDEN];
   double m_r_log_dt[FXAI_AI_MLP_HIDDEN];
   double m_r_b_re[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_r_b_im[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_r_u_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_r_u_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_r_v_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
   double m_r_v_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];

   // AdamW moments for heads.
   double m_m_c_re[FXAI_AI_MLP_HIDDEN], m_v_c_re[FXAI_AI_MLP_HIDDEN];
   double m_m_c_im[FXAI_AI_MLP_HIDDEN], m_v_c_im[FXAI_AI_MLP_HIDDEN];
   double m_m_d_skip[FXAI_AI_WEIGHTS], m_v_d_skip[FXAI_AI_WEIGHTS];
   double m_m_b_out, m_v_b_out;

   double m_m_prob_scale, m_v_prob_scale;
   double m_m_prob_bias,  m_v_prob_bias;

   double m_m_w_move_mu[FXAI_S4_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_v_w_move_mu[FXAI_S4_HORIZONS][FXAI_AI_MLP_HIDDEN];
   double m_m_b_move_mu[FXAI_S4_HORIZONS];
   double m_v_b_move_mu[FXAI_S4_HORIZONS];

   double m_m_w_move_logv[FXAI_AI_MLP_HIDDEN];
   double m_v_w_move_logv[FXAI_AI_MLP_HIDDEN];
   double m_m_b_move_logv, m_v_b_move_logv;

   void ComplexMul(const double ar, const double ai,
                   const double br, const double bi,
                   double &cr, double &ci) const
   {
      cr = ar * br - ai * bi;
      ci = ar * bi + ai * br;
   }

   void ComplexDiv(const double ar, const double ai,
                   const double br, const double bi,
                   double &cr, double &ci) const
   {
      double den = br * br + bi * bi;
      if(den < 1e-12) den = 1e-12;
      cr = (ar * br + ai * bi) / den;
      ci = (ai * br - ar * bi) / den;
   }

   void LayerNorm(double &v[]) const
   {
      double g[FXAI_AI_MLP_HIDDEN];
      double b[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g[h] = 1.0;
         b[h] = 0.0;
      }
      FXAI_ModuleLayerNormAffine(v, FXAI_AI_MLP_HIDDEN, g, b);
   }

   void ResetState(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_state_re[h] = 0.0;
         m_state_im[h] = 0.0;
      }
   }

   void ResetBatch(void)
   {
      m_batch_size = 0;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_batch_s0_re[h] = m_state_re[h];
         m_batch_s0_im[h] = m_state_im[h];
      }
   }

   void ResetNormStats(void)
   {
      m_x_norm_ready = false;
      m_x_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }

      m_s_norm_ready = false;
      m_s_norm_steps = 0;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_sr_mean[h] = 0.0;
         m_sr_var[h] = 1.0;
         m_si_mean[h] = 0.0;
         m_si_var[h] = 1.0;
      }
   }

   void ResetOptimizers(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_r_log_tau[h] = 0.0;
         m_r_omega[h] = 0.0;
         m_r_log_dt[h] = 0.0;

         m_m_c_re[h] = 0.0; m_v_c_re[h] = 0.0;
         m_m_c_im[h] = 0.0; m_v_c_im[h] = 0.0;
         m_m_w_move_logv[h] = 0.0; m_v_w_move_logv[h] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_r_b_re[h][i] = 0.0;
            m_r_b_im[h][i] = 0.0;
         }

         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            m_r_u_re[h][r] = 0.0;
            m_r_u_im[h][r] = 0.0;
            m_r_v_re[h][r] = 0.0;
            m_r_v_im[h][r] = 0.0;
         }
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_m_d_skip[i] = 0.0;
         m_v_d_skip[i] = 0.0;
      }

      for(int k=0; k<FXAI_S4_HORIZONS; k++)
      {
         m_m_b_move_mu[k] = 0.0;
         m_v_b_move_mu[k] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_m_w_move_mu[k][h] = 0.0;
            m_v_w_move_mu[k][h] = 0.0;
         }
      }

      m_m_b_out = 0.0; m_v_b_out = 0.0;
      m_m_prob_scale = 0.0; m_v_prob_scale = 0.0;
      m_m_prob_bias  = 0.0; m_v_prob_bias = 0.0;
      m_m_b_move_logv = 0.0; m_v_b_move_logv = 0.0;
   }

   void InitWeights(void)
   {
      m_train_steps = 0;
      m_seen_updates = 0;
      ResetNormStats();
      ResetState();
      ResetBatch();
      ResetOptimizers();

      m_margin_ready = false;
      m_margin_ema = 1.0;
      m_prob_scale = 1.0;
      m_prob_bias = 0.0;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_log_tau[h] = MathLog(0.7 + 0.4 * MathAbs(MathSin((double)(h + 1) * 0.47)));
         m_omega[h] = 0.25 * MathSin((double)(h + 1) * 0.73);
         m_log_dt[h] = 0.0;

         m_c_re[h] = 0.04 * MathCos((double)(h + 2) * 0.91);
         m_c_im[h] = 0.04 * MathSin((double)(h + 3) * 0.95);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double s = (double)((h + 1) * (i + 2));
            m_b_re[h][i] = 0.03 * MathSin(0.83 * s);
            m_b_im[h][i] = 0.03 * MathCos(0.89 * s);
         }

         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            double u = (double)((h + 2) * (r + 3));
            double v = (double)((h + 3) * (r + 5));
            m_u_re[h][r] = 0.015 * MathSin(0.79 * u);
            m_u_im[h][r] = 0.015 * MathCos(0.81 * u);
            m_v_re[h][r] = 0.015 * MathSin(0.87 * v);
            m_v_im[h][r] = 0.015 * MathCos(0.93 * v);
         }

         m_w_move_logv[h] = 0.03 * MathCos((double)(h + 1) * 1.07);
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_d_skip[i] = 0.02 * MathCos((double)(i + 1) * 1.11);

      m_b_out = 0.0;

      for(int k=0; k<FXAI_S4_HORIZONS; k++)
      {
         m_b_move_mu[k] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            m_w_move_mu[k][h] = 0.03 * MathSin((double)((k + 2) * (h + 1)) * 0.61);
      }

      m_b_move_logv = MathLog(0.7);

      m_initialized = true;
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 64 ? 0.06 : 0.02);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }
      m_x_norm_steps++;
      if(m_x_norm_steps >= 24) m_x_norm_ready = true;
   }

   void NormalizeInput(const double &x[], double &xn[], const bool update_stats)
   {
      if(update_stats) UpdateInputStats(x);

      xn[0] = 1.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = x[i];
         if(m_x_norm_ready)
            v = (x[i] - m_x_mean[i]) / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym(v, 8.0);
      }
   }

   void UpdateStateStats(const double &sr_raw[], const double &si_raw[])
   {
      double a = (m_s_norm_steps < 96 ? 0.05 : 0.015);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double dr = sr_raw[h] - m_sr_mean[h];
         m_sr_mean[h] += a * dr;
         double drr = sr_raw[h] - m_sr_mean[h];
         m_sr_var[h] = (1.0 - a) * m_sr_var[h] + a * drr * drr;
         if(m_sr_var[h] < 1e-6) m_sr_var[h] = 1e-6;

         double di = si_raw[h] - m_si_mean[h];
         m_si_mean[h] += a * di;
         double dii = si_raw[h] - m_si_mean[h];
         m_si_var[h] = (1.0 - a) * m_si_var[h] + a * dii * dii;
         if(m_si_var[h] < 1e-6) m_si_var[h] = 1e-6;
      }
      m_s_norm_steps++;
      if(m_s_norm_steps >= 32) m_s_norm_ready = true;
   }

   void NormalizeState(const double &sr_raw[],
                       const double &si_raw[],
                       double &sr[],
                       double &si[],
                       const bool update_stats)
   {
      if(update_stats) UpdateStateStats(sr_raw, si_raw);

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double r = sr_raw[h];
         double im = si_raw[h];

         if(m_s_norm_ready)
         {
            r = (r - m_sr_mean[h]) / MathSqrt(m_sr_var[h] + 1e-6);
            im = (im - m_si_mean[h]) / MathSqrt(m_si_var[h] + 1e-6);
         }

         sr[h] = FXAI_ClipSym(r, 8.0);
         si[h] = FXAI_ClipSym(im, 8.0);
      }

      LayerNorm(sr);
      LayerNorm(si);
   }

   void DiscretizeOne(const int h,
                      double &lam_re,
                      double &lam_im,
                      double &dt,
                      double &a_re,
                      double &a_im,
                      double &den_re,
                      double &den_im,
                      double &A_re,
                      double &A_im,
                      double &F_re,
                      double &F_im) const
   {
      lam_re = -MathExp(FXAI_Clamp(m_log_tau[h], -5.0, 4.0));
      lam_im = FXAI_Clamp(m_omega[h], -16.0, 16.0);
      dt = MathExp(FXAI_Clamp(m_log_dt[h], -3.5, 1.2));

      a_re = 0.5 * dt * lam_re;
      a_im = 0.5 * dt * lam_im;
      den_re = 1.0 - a_re;
      den_im = -a_im;

      // Bilinear (Tustin) discretization.
      double num_re = 1.0 + a_re;
      double num_im = a_im;
      ComplexDiv(num_re, num_im, den_re, den_im, A_re, A_im);
      ComplexDiv(dt, 0.0, den_re, den_im, F_re, F_im);

      double mag = MathSqrt(A_re * A_re + A_im * A_im);
      if(mag > 0.999)
      {
         double s = 0.999 / mag;
         A_re *= s;
         A_im *= s;
      }
   }

   void RMSPropApply(double &p,
                     double &acc,
                     const double grad,
                     const double lr,
                     const double decay,
                     const double wd)
   {
      FXAI_OptRMSPropStep(p, acc, grad, lr, decay, wd);
   }

   void AdamWApply(double &p,
                   double &m,
                   double &v,
                   const double grad,
                   const double lr,
                   const double beta1,
                   const double beta2,
                   const double wd,
                   const int t)
   {
      FXAI_OptAdamWStep(p, m, v, grad, lr, beta1, beta2, wd, t);
   }

   void AppendBatch(const int y,
                    const double &x[],
                    const double move_points,
                    const double sample_w)
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_batch_s0_re[h] = m_state_re[h];
            m_batch_s0_im[h] = m_state_im[h];
         }
      }

      if(m_batch_size < FXAI_S4_TBPTT)
      {
         int p = m_batch_size;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[p][i] = x[i];
         m_batch_y[p] = y;
         m_batch_move[p] = move_points;
         m_batch_w[p] = sample_w;
         m_batch_size++;
         return;
      }

      for(int t=1; t<FXAI_S4_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[t - 1][i] = m_batch_x[t][i];
         m_batch_y[t - 1] = m_batch_y[t];
         m_batch_move[t - 1] = m_batch_move[t];
         m_batch_w[t - 1] = m_batch_w[t];
      }
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[FXAI_S4_TBPTT - 1][i] = x[i];
      m_batch_y[FXAI_S4_TBPTT - 1] = y;
      m_batch_move[FXAI_S4_TBPTT - 1] = move_points;
      m_batch_w[FXAI_S4_TBPTT - 1] = sample_w;
      m_batch_size = FXAI_S4_TBPTT;
   }

   void TrainBatch(const FXAIAIHyperParams &hp)
   {
      int len = m_batch_size;
      if(len <= 0) return;

      m_train_steps += len;

      double lr_base = FXAI_Clamp(hp.lr, 0.00005, 0.20000);
      double lr_core = FXAI_Clamp(0.45 * lr_base, 0.00001, 0.03000);
      double lr_head = FXAI_Clamp(0.80 * lr_base, 0.00001, 0.06000);
      double wd_core = FXAI_Clamp(0.15 * hp.l2, 0.0, 0.0300);
      double wd_head = FXAI_Clamp(0.35 * hp.l2, 0.0, 0.0600);

      // Precompute discretization constants.
      double lam_re[FXAI_AI_MLP_HIDDEN], lam_im[FXAI_AI_MLP_HIDDEN], dtv[FXAI_AI_MLP_HIDDEN];
      double a_re[FXAI_AI_MLP_HIDDEN], a_im[FXAI_AI_MLP_HIDDEN];
      double den_re[FXAI_AI_MLP_HIDDEN], den_im[FXAI_AI_MLP_HIDDEN];
      double A_re[FXAI_AI_MLP_HIDDEN], A_im[FXAI_AI_MLP_HIDDEN];
      double F_re[FXAI_AI_MLP_HIDDEN], F_im[FXAI_AI_MLP_HIDDEN];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         DiscretizeOne(h,
                       lam_re[h], lam_im[h], dtv[h],
                       a_re[h], a_im[h], den_re[h], den_im[h],
                       A_re[h], A_im[h], F_re[h], F_im[h]);
      }

      // Forward caches.
      double xnorm[FXAI_S4_TBPTT][FXAI_AI_WEIGHTS];
      double prev_re[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN], prev_im[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];
      double u_re[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN], u_im[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];
      double z_re[FXAI_S4_TBPTT][FXAI_S4_RANK], z_im[FXAI_S4_TBPTT][FXAI_S4_RANK];
      double q_re[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN], q_im[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];
      double raw_re[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN], raw_im[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];
      double st_re[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN], st_im[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];

      double z_cls[FXAI_S4_TBPTT], z_norm[FXAI_S4_TBPTT], p_local[FXAI_S4_TBPTT], norm_denom[FXAI_S4_TBPTT];
      double amp[FXAI_S4_TBPTT][FXAI_AI_MLP_HIDDEN];
      double mu[FXAI_S4_TBPTT][FXAI_S4_HORIZONS];
      double logv[FXAI_S4_TBPTT];

      double t_move[FXAI_S4_TBPTT][FXAI_S4_HORIZONS];
      double t_w[FXAI_S4_TBPTT][FXAI_S4_HORIZONS];

      double cur_re[FXAI_AI_MLP_HIDDEN], cur_im[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         cur_re[h] = m_batch_s0_re[h];
         cur_im[h] = m_batch_s0_im[h];
      }

      for(int t=0; t<len; t++)
      {
         double xin[FXAI_AI_WEIGHTS];
         double xout[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xin[i] = m_batch_x[t][i];
         NormalizeInput(xin, xout, true);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xnorm[t][i] = xout[i];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            prev_re[t][h] = cur_re[h];
            prev_im[t][h] = cur_im[h];

            double ur = 0.0, ui = 0.0;
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               ur += m_b_re[h][i] * xnorm[t][i];
               ui += m_b_im[h][i] * xnorm[t][i];
            }
            u_re[t][h] = ur;
            u_im[t][h] = ui;
         }

         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            double zr = 0.0, zi = 0.0;
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
            {
               double pr = prev_re[t][j], pi = prev_im[t][j];
               zr += m_v_re[j][r] * pr - m_v_im[j][r] * pi;
               zi += m_v_re[j][r] * pi + m_v_im[j][r] * pr;
            }
            z_re[t][r] = zr;
            z_im[t][r] = zi;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double qr = 0.0, qi = 0.0;
            for(int r=0; r<FXAI_S4_RANK; r++)
            {
               double zr = z_re[t][r], zi = z_im[t][r];
               qr += m_u_re[h][r] * zr - m_u_im[h][r] * zi;
               qi += m_u_re[h][r] * zi + m_u_im[h][r] * zr;
            }
            q_re[t][h] = qr;
            q_im[t][h] = qi;

            double memr, memi, inr, ini;
            ComplexMul(A_re[h], A_im[h], prev_re[t][h], prev_im[t][h], memr, memi);
            ComplexMul(F_re[h], F_im[h], u_re[t][h], u_im[t][h], inr, ini);

            double nr = memr + inr + dtv[h] * qr;
            double ni = memi + ini + dtv[h] * qi;

            double mag = MathSqrt(nr * nr + ni * ni);
            if(mag > 20.0)
            {
               double s = 20.0 / mag;
               nr *= s;
               ni *= s;
            }

            raw_re[t][h] = nr;
            raw_im[t][h] = ni;
         }

         double rawr[FXAI_AI_MLP_HIDDEN], rawi[FXAI_AI_MLP_HIDDEN];
         double str[FXAI_AI_MLP_HIDDEN], sti[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            rawr[h] = raw_re[t][h];
            rawi[h] = raw_im[t][h];
         }
         NormalizeState(rawr, rawi, str, sti, true);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            st_re[t][h] = str[h];
            st_im[t][h] = sti[h];
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            cur_re[h] = raw_re[t][h];
            cur_im[h] = raw_im[t][h];
         }

         double z = m_b_out;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            z += m_c_re[h] * st_re[t][h] - m_c_im[h] * st_im[t][h];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            z += m_d_skip[i] * xnorm[t][i];

         z_cls[t] = z;

         double dnm = (m_margin_ready ? m_margin_ema : 1.0);
         if(dnm < 0.25) dnm = 0.25;
         norm_denom[t] = dnm;
         z_norm[t] = z / dnm;

         double pl = FXAI_Sigmoid((m_prob_scale * z_norm[t]) + m_prob_bias);
         p_local[t] = pl;

         double lv = m_b_move_logv;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double a = MathSqrt(st_re[t][h] * st_re[t][h] + st_im[t][h] * st_im[t][h] + 1e-8);
            amp[t][h] = a;
            lv += m_w_move_logv[h] * a;
         }
         logv[t] = FXAI_Clamp(lv, -4.0, 4.0);

         for(int k=0; k<FXAI_S4_HORIZONS; k++)
         {
            double m = m_b_move_mu[k];
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               m += m_w_move_mu[k][h] * amp[t][h];
            mu[t][k] = m;
         }
      }

      // Keep latest recurrent state.
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_state_re[h] = cur_re[h];
         m_state_im[h] = cur_im[h];
      }

      // Build multi-horizon targets from forward-looking labels in TBPTT window.
      for(int t=0; t<len; t++)
      {
         double m1 = MathAbs(m_batch_move[t]);

         int n3 = 0;
         double s3 = 0.0;
         for(int k=0; k<3; k++)
         {
            int idx = t + k;
            if(idx >= len) break;
            s3 += MathAbs(m_batch_move[idx]);
            n3++;
         }

         int n5 = 0;
         double s5 = 0.0;
         for(int k=0; k<5; k++)
         {
            int idx = t + k;
            if(idx >= len) break;
            s5 += MathAbs(m_batch_move[idx]);
            n5++;
         }

         double m3 = (n3 > 0 ? s3 / (double)n3 : m1);
         double m5 = (n5 > 0 ? s5 / (double)n5 : m3);

         t_move[t][0] = m1;
         t_move[t][1] = m3;
         t_move[t][2] = m5;

         // Uncertainty-aware horizon weighting baseline (near > mid > far), adjusted by support.
         double w1 = 0.50;
         double w3 = 0.30 * ((double)n3 / 3.0);
         double w5 = 0.20 * ((double)n5 / 5.0);
         double ws = w1 + w3 + w5;
         if(ws <= 0.0) ws = 1.0;
         t_w[t][0] = w1 / ws;
         t_w[t][1] = w3 / ws;
         t_w[t][2] = w5 / ws;
      }

      // Gradient accumulators.
      double g_log_tau[FXAI_AI_MLP_HIDDEN], g_omega[FXAI_AI_MLP_HIDDEN], g_log_dt[FXAI_AI_MLP_HIDDEN];
      double g_b_re[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS], g_b_im[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_u_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK], g_u_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];
      double g_v_re[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK], g_v_im[FXAI_AI_MLP_HIDDEN][FXAI_S4_RANK];

      double g_c_re[FXAI_AI_MLP_HIDDEN], g_c_im[FXAI_AI_MLP_HIDDEN], g_d_skip[FXAI_AI_WEIGHTS], g_b_out;
      double g_prob_scale, g_prob_bias;

      double g_w_move_mu[FXAI_S4_HORIZONS][FXAI_AI_MLP_HIDDEN], g_b_move_mu[FXAI_S4_HORIZONS];
      double g_w_move_logv[FXAI_AI_MLP_HIDDEN], g_b_move_logv;

      double gA_re[FXAI_AI_MLP_HIDDEN], gA_im[FXAI_AI_MLP_HIDDEN];
      double gF_re[FXAI_AI_MLP_HIDDEN], gF_im[FXAI_AI_MLP_HIDDEN];
      double g_dt_direct[FXAI_AI_MLP_HIDDEN];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g_log_tau[h] = 0.0;
         g_omega[h] = 0.0;
         g_log_dt[h] = 0.0;

         g_c_re[h] = 0.0;
         g_c_im[h] = 0.0;
         g_w_move_logv[h] = 0.0;
         gA_re[h] = 0.0;
         gA_im[h] = 0.0;
         gF_re[h] = 0.0;
         gF_im[h] = 0.0;
         g_dt_direct[h] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            g_b_re[h][i] = 0.0;
            g_b_im[h][i] = 0.0;
         }
         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            g_u_re[h][r] = 0.0;
            g_u_im[h][r] = 0.0;
            g_v_re[h][r] = 0.0;
            g_v_im[h][r] = 0.0;
         }
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_d_skip[i] = 0.0;
      for(int k=0; k<FXAI_S4_HORIZONS; k++)
      {
         g_b_move_mu[k] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            g_w_move_mu[k][h] = 0.0;
      }
      g_b_out = 0.0;
      g_prob_scale = 0.0;
      g_prob_bias = 0.0;
      g_b_move_logv = 0.0;

      double ds_next_re[FXAI_AI_MLP_HIDDEN], ds_next_im[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         ds_next_re[h] = 0.0;
         ds_next_im[h] = 0.0;
      }

      // Backward through truncated sequence.
      for(int t=len - 1; t>=0; t--)
      {
         double sw = FXAI_Clamp(m_batch_w[t], 0.25, 4.00);

         // Class objective via plugin-local calibration layer.
         double y = (double)m_batch_y[t];
         double p = p_local[t];
         double dprob = (p - y) * sw;
         double dcal = dprob * p * (1.0 - p);

         g_prob_scale += dcal * z_norm[t];
         g_prob_bias += dcal;

         double dz = dcal * m_prob_scale / norm_denom[t];

         g_b_out += dz;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            g_d_skip[i] += dz * xnorm[t][i];

         double ds_re[FXAI_AI_MLP_HIDDEN], ds_im[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_c_re[h] += dz * st_re[t][h];
            g_c_im[h] += dz * (-st_im[t][h]);
            ds_re[h] = dz * m_c_re[h];
            ds_im[h] = dz * (-m_c_im[h]);
         }

         // Multi-horizon distributional objective with shared uncertainty.
         double inv_var = MathExp(-logv[t]);
         inv_var = FXAI_Clamp(inv_var, 0.05, 20.0);

         double sq = 0.0;
         double d_amp[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) d_amp[h] = 0.0;

         for(int k=0; k<FXAI_S4_HORIZONS; k++)
         {
            double err = mu[t][k] - t_move[t][k];
            double wk = t_w[t][k];
            sq += wk * err * err;

            double dmu = sw * inv_var * wk * err;
            g_b_move_mu[k] += dmu;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               g_w_move_mu[k][h] += dmu * amp[t][h];
               d_amp[h] += dmu * m_w_move_mu[k][h];
            }
         }

         double dlogv = 0.5 * sw * (1.0 - inv_var * sq);
         g_b_move_logv += dlogv;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_w_move_logv[h] += dlogv * amp[t][h];
            d_amp[h] += dlogv * m_w_move_logv[h];
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double a = amp[t][h];
            double inv_a = (a > 1e-6 ? 1.0 / a : 0.0);
            ds_re[h] += d_amp[h] * st_re[t][h] * inv_a;
            ds_im[h] += d_amp[h] * st_im[t][h] * inv_a;
         }

         // Recurrence backprop.
         double ds_prev_re[FXAI_AI_MLP_HIDDEN], ds_prev_im[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            ds_prev_re[h] = 0.0;
            ds_prev_im[h] = 0.0;
         }

         double dz_re[FXAI_S4_RANK], dz_im[FXAI_S4_RANK];
         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            dz_re[r] = 0.0;
            dz_im[r] = 0.0;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double gr = ds_re[h] + ds_next_re[h];
            double gi = ds_im[h] + ds_next_im[h];

            double pr = prev_re[t][h], pi = prev_im[t][h];
            double ur = u_re[t][h], ui = u_im[t][h];

            // Exact local grads wrt A/F contributions.
            gA_re[h] += gr * pr + gi * pi;
            gA_im[h] += -gr * pi + gi * pr;
            gF_re[h] += gr * ur + gi * ui;
            gF_im[h] += -gr * ui + gi * ur;

            ds_prev_re[h] += gr * A_re[h] + gi * A_im[h];
            ds_prev_im[h] += -gr * A_im[h] + gi * A_re[h];

            double du_re = gr * F_re[h] + gi * F_im[h];
            double du_im = -gr * F_im[h] + gi * F_re[h];
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               g_b_re[h][i] += du_re * xnorm[t][i];
               g_b_im[h][i] += du_im * xnorm[t][i];
            }

            // Low-rank path: dt * U * (V^T s_prev)
            double qr = q_re[t][h], qi = q_im[t][h];
            g_dt_direct[h] += gr * qr + gi * qi;

            double dqr = gr * dtv[h];
            double dqi = gi * dtv[h];
            for(int r=0; r<FXAI_S4_RANK; r++)
            {
               double zr = z_re[t][r], zi = z_im[t][r];
               g_u_re[h][r] += dqr * zr + dqi * zi;
               g_u_im[h][r] += -dqr * zi + dqi * zr;

               dz_re[r] += dqr * m_u_re[h][r] + dqi * m_u_im[h][r];
               dz_im[r] += -dqr * m_u_im[h][r] + dqi * m_u_re[h][r];
            }
         }

         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
            {
               double pr = prev_re[t][j], pi = prev_im[t][j];
               g_v_re[j][r] += dz_re[r] * pr + dz_im[r] * pi;
               g_v_im[j][r] += -dz_re[r] * pi + dz_im[r] * pr;

               ds_prev_re[j] += dz_re[r] * m_v_re[j][r] + dz_im[r] * m_v_im[j][r];
               ds_prev_im[j] += -dz_re[r] * m_v_im[j][r] + dz_im[r] * m_v_re[j][r];
            }
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            ds_next_re[h] = ds_prev_re[h];
            ds_next_im[h] = ds_prev_im[h];
         }
      }

      // Exact gradient path through discretization (A,F from lambda,dt).
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double den2_re = den_re[h] * den_re[h] - den_im[h] * den_im[h];
         double den2_im = 2.0 * den_re[h] * den_im[h];

         double inv_den_re, inv_den_im;
         ComplexDiv(1.0, 0.0, den_re[h], den_im[h], inv_den_re, inv_den_im);

         double inv_den2_re, inv_den2_im;
         ComplexDiv(1.0, 0.0, den2_re, den2_im, inv_den2_re, inv_den2_im);

         // dA/dlambda = dt / (1 - a)^2
         double dAdl_re = dtv[h] * inv_den2_re;
         double dAdl_im = dtv[h] * inv_den2_im;

         // dF/dlambda = 0.5 * dt^2 / (1 - a)^2
         double dFdl_re = 0.5 * dtv[h] * dtv[h] * inv_den2_re;
         double dFdl_im = 0.5 * dtv[h] * dtv[h] * inv_den2_im;

         // dA/ddt = lambda / (1 - a)^2
         double dAddt_re, dAddt_im;
         ComplexMul(lam_re[h], lam_im[h], inv_den2_re, inv_den2_im, dAddt_re, dAddt_im);

         // dF/ddt = 1/(1-a) + 0.5*dt*lambda/(1-a)^2
         double tmp_re, tmp_im;
         ComplexMul(0.5 * dtv[h] * lam_re[h], 0.5 * dtv[h] * lam_im[h], inv_den2_re, inv_den2_im, tmp_re, tmp_im);
         double dFddt_re = inv_den_re + tmp_re;
         double dFddt_im = inv_den_im + tmp_im;

         double glam_re = 0.0, glam_im = 0.0;
         glam_re += gA_re[h] * dAdl_re + gA_im[h] * dAdl_im;
         glam_im += -gA_re[h] * dAdl_im + gA_im[h] * dAdl_re;
         glam_re += gF_re[h] * dFdl_re + gF_im[h] * dFdl_im;
         glam_im += -gF_re[h] * dFdl_im + gF_im[h] * dFdl_re;

         double gdt = 0.0;
         gdt += gA_re[h] * dAddt_re + gA_im[h] * dAddt_im;
         gdt += gF_re[h] * dFddt_re + gF_im[h] * dFddt_im;
         gdt += g_dt_direct[h];

         // lambda_re = -exp(log_tau) => dlambda_re/dlog_tau = lambda_re.
         g_log_tau[h] += glam_re * lam_re[h];
         g_omega[h] += glam_im;
         g_log_dt[h] += gdt * dtv[h];
      }

      // Global gradient norm clip.
      double gnorm2 = g_b_out * g_b_out +
                      g_prob_scale * g_prob_scale +
                      g_prob_bias * g_prob_bias +
                      g_b_move_logv * g_b_move_logv;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         gnorm2 += g_d_skip[i] * g_d_skip[i];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         gnorm2 += g_log_tau[h] * g_log_tau[h] + g_omega[h] * g_omega[h] + g_log_dt[h] * g_log_dt[h];
         gnorm2 += g_c_re[h] * g_c_re[h] + g_c_im[h] * g_c_im[h] + g_w_move_logv[h] * g_w_move_logv[h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            gnorm2 += g_b_re[h][i] * g_b_re[h][i] + g_b_im[h][i] * g_b_im[h][i];

         for(int r=0; r<FXAI_S4_RANK; r++)
            gnorm2 += g_u_re[h][r] * g_u_re[h][r] + g_u_im[h][r] * g_u_im[h][r] +
                      g_v_re[h][r] * g_v_re[h][r] + g_v_im[h][r] * g_v_im[h][r];
      }

      for(int k=0; k<FXAI_S4_HORIZONS; k++)
      {
         gnorm2 += g_b_move_mu[k] * g_b_move_mu[k];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            gnorm2 += g_w_move_mu[k][h] * g_w_move_mu[k][h];
      }

      double gnorm = MathSqrt(gnorm2);
      double gscale = (gnorm > 10.0 ? (10.0 / gnorm) : 1.0);
      if(gscale < 1.0)
      {
         g_b_out *= gscale;
         g_prob_scale *= gscale;
         g_prob_bias *= gscale;
         g_b_move_logv *= gscale;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_d_skip[i] *= gscale;

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_log_tau[h] *= gscale;
            g_omega[h] *= gscale;
            g_log_dt[h] *= gscale;
            g_c_re[h] *= gscale;
            g_c_im[h] *= gscale;
            g_w_move_logv[h] *= gscale;

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               g_b_re[h][i] *= gscale;
               g_b_im[h][i] *= gscale;
            }
            for(int r=0; r<FXAI_S4_RANK; r++)
            {
               g_u_re[h][r] *= gscale;
               g_u_im[h][r] *= gscale;
               g_v_re[h][r] *= gscale;
               g_v_im[h][r] *= gscale;
            }
         }

         for(int k=0; k<FXAI_S4_HORIZONS; k++)
         {
            g_b_move_mu[k] *= gscale;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               g_w_move_mu[k][h] *= gscale;
         }
      }

      // Optimizer step: RMSProp on SSM core.
      const double rms_decay = 0.99;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         RMSPropApply(m_log_tau[h], m_r_log_tau[h], g_log_tau[h], lr_core, rms_decay, wd_core);
         RMSPropApply(m_omega[h],   m_r_omega[h],   g_omega[h],   lr_core, rms_decay, wd_core);
         RMSPropApply(m_log_dt[h],  m_r_log_dt[h],  g_log_dt[h],  lr_core, rms_decay, wd_core);

         m_log_tau[h] = FXAI_Clamp(m_log_tau[h], -5.0, 4.0);
         m_omega[h] = FXAI_Clamp(m_omega[h], -20.0, 20.0);
         m_log_dt[h] = FXAI_Clamp(m_log_dt[h], -3.5, 1.2);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            RMSPropApply(m_b_re[h][i], m_r_b_re[h][i], g_b_re[h][i], lr_core, rms_decay, wd_core);
            RMSPropApply(m_b_im[h][i], m_r_b_im[h][i], g_b_im[h][i], lr_core, rms_decay, wd_core);
            m_b_re[h][i] = FXAI_ClipSym(m_b_re[h][i], 3.0);
            m_b_im[h][i] = FXAI_ClipSym(m_b_im[h][i], 3.0);
         }

         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            RMSPropApply(m_u_re[h][r], m_r_u_re[h][r], g_u_re[h][r], lr_core, rms_decay, wd_core);
            RMSPropApply(m_u_im[h][r], m_r_u_im[h][r], g_u_im[h][r], lr_core, rms_decay, wd_core);
            RMSPropApply(m_v_re[h][r], m_r_v_re[h][r], g_v_re[h][r], lr_core, rms_decay, wd_core);
            RMSPropApply(m_v_im[h][r], m_r_v_im[h][r], g_v_im[h][r], lr_core, rms_decay, wd_core);

            m_u_re[h][r] = FXAI_ClipSym(m_u_re[h][r], 2.0);
            m_u_im[h][r] = FXAI_ClipSym(m_u_im[h][r], 2.0);
            m_v_re[h][r] = FXAI_ClipSym(m_v_re[h][r], 2.0);
            m_v_im[h][r] = FXAI_ClipSym(m_v_im[h][r], 2.0);
         }
      }

      // Optimizer step: AdamW on heads.
      const double b1 = 0.90;
      const double b2 = 0.999;
      int tstep = MathMax(1, m_train_steps);

      AdamWApply(m_b_out, m_m_b_out, m_v_b_out, g_b_out, lr_head, b1, b2, 0.0, tstep);

      AdamWApply(m_prob_scale, m_m_prob_scale, m_v_prob_scale, g_prob_scale, 0.5 * lr_head, b1, b2, 0.0, tstep);
      AdamWApply(m_prob_bias,  m_m_prob_bias,  m_v_prob_bias,  g_prob_bias,  0.5 * lr_head, b1, b2, 0.0, tstep);
      m_prob_scale = FXAI_Clamp(m_prob_scale, 0.20, 6.00);
      m_prob_bias = FXAI_Clamp(m_prob_bias, -6.0, 6.0);

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         AdamWApply(m_c_re[h], m_m_c_re[h], m_v_c_re[h], g_c_re[h], lr_head, b1, b2, wd_head, tstep);
         AdamWApply(m_c_im[h], m_m_c_im[h], m_v_c_im[h], g_c_im[h], lr_head, b1, b2, wd_head, tstep);
         AdamWApply(m_w_move_logv[h], m_m_w_move_logv[h], m_v_w_move_logv[h], g_w_move_logv[h], lr_head, b1, b2, wd_head, tstep);

         m_c_re[h] = FXAI_ClipSym(m_c_re[h], 4.0);
         m_c_im[h] = FXAI_ClipSym(m_c_im[h], 4.0);
         m_w_move_logv[h] = FXAI_ClipSym(m_w_move_logv[h], 4.0);
      }

      AdamWApply(m_b_move_logv, m_m_b_move_logv, m_v_b_move_logv, g_b_move_logv, lr_head, b1, b2, 0.0, tstep);
      m_b_move_logv = FXAI_Clamp(m_b_move_logv, -4.0, 3.0);

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         AdamWApply(m_d_skip[i], m_m_d_skip[i], m_v_d_skip[i], g_d_skip[i], 0.7 * lr_head, b1, b2, wd_head, tstep);
         m_d_skip[i] = FXAI_ClipSym(m_d_skip[i], 4.0);
      }

      for(int k=0; k<FXAI_S4_HORIZONS; k++)
      {
         AdamWApply(m_b_move_mu[k], m_m_b_move_mu[k], m_v_b_move_mu[k], g_b_move_mu[k], lr_head, b1, b2, 0.0, tstep);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            AdamWApply(m_w_move_mu[k][h],
                       m_m_w_move_mu[k][h],
                       m_v_w_move_mu[k][h],
                       g_w_move_mu[k][h],
                       lr_head,
                       b1,
                       b2,
                       wd_head,
                       tstep);
            m_w_move_mu[k][h] = FXAI_ClipSym(m_w_move_mu[k][h], 4.0);
         }
      }

      // Update local margin scale and shared calibrators/heads on realized samples.
      for(int t=0; t<len; t++)
      {
         double absz = MathAbs(z_cls[t]);
         if(!m_margin_ready)
         {
            m_margin_ema = MathMax(absz, 0.25);
            m_margin_ready = true;
         }
         else
         {
            m_margin_ema = 0.97 * m_margin_ema + 0.03 * absz;
            if(m_margin_ema < 0.25) m_margin_ema = 0.25;
         }

         double sw = FXAI_Clamp(m_batch_w[t], 0.25, 4.00);
         UpdateCalibration(p_local[t], m_batch_y[t], sw);
         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, m_batch_move[t], 0.05);

         double xloc[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];
         UpdateMoveHead(xloc, m_batch_move[t], hp, sw);
      }

      ResetBatch();
   }

   void ForwardNoCommit(const double &x[],
                        const bool update_norm,
                        double &st_re[],
                        double &st_im[],
                        double &z_cls,
                        double &p_local,
                        double &mu1,
                        double &mu3,
                        double &mu5,
                        double &logv) const
   {
      double lam_re[FXAI_AI_MLP_HIDDEN], lam_im[FXAI_AI_MLP_HIDDEN], dtv[FXAI_AI_MLP_HIDDEN];
      double a_re[FXAI_AI_MLP_HIDDEN], a_im[FXAI_AI_MLP_HIDDEN];
      double den_re[FXAI_AI_MLP_HIDDEN], den_im[FXAI_AI_MLP_HIDDEN];
      double A_re[FXAI_AI_MLP_HIDDEN], A_im[FXAI_AI_MLP_HIDDEN];
      double F_re[FXAI_AI_MLP_HIDDEN], F_im[FXAI_AI_MLP_HIDDEN];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         // const-safe duplication of DiscretizeOne
         lam_re[h] = -MathExp(FXAI_Clamp(m_log_tau[h], -5.0, 4.0));
         lam_im[h] = FXAI_Clamp(m_omega[h], -16.0, 16.0);
         dtv[h] = MathExp(FXAI_Clamp(m_log_dt[h], -3.5, 1.2));

         a_re[h] = 0.5 * dtv[h] * lam_re[h];
         a_im[h] = 0.5 * dtv[h] * lam_im[h];
         den_re[h] = 1.0 - a_re[h];
         den_im[h] = -a_im[h];

         double num_re = 1.0 + a_re[h];
         double num_im = a_im[h];
         ComplexDiv(num_re, num_im, den_re[h], den_im[h], A_re[h], A_im[h]);
         ComplexDiv(dtv[h], 0.0, den_re[h], den_im[h], F_re[h], F_im[h]);

         double mag = MathSqrt(A_re[h] * A_re[h] + A_im[h] * A_im[h]);
         if(mag > 0.999)
         {
            double s = 0.999 / mag;
            A_re[h] *= s;
            A_im[h] *= s;
         }
      }

      double xn[FXAI_AI_WEIGHTS];
      // const function: no stats updates even if update_norm requested.
      xn[0] = 1.0;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = x[i];
         if(m_x_norm_ready)
            v = (x[i] - m_x_mean[i]) / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym(v, 8.0);
      }

      double prev_re[FXAI_AI_MLP_HIDDEN], prev_im[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         prev_re[h] = m_state_re[h];
         prev_im[h] = m_state_im[h];
      }

      double zr[FXAI_S4_RANK], zi[FXAI_S4_RANK];
      for(int r=0; r<FXAI_S4_RANK; r++)
      {
         zr[r] = 0.0;
         zi[r] = 0.0;
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            zr[r] += m_v_re[j][r] * prev_re[j] - m_v_im[j][r] * prev_im[j];
            zi[r] += m_v_re[j][r] * prev_im[j] + m_v_im[j][r] * prev_re[j];
         }
      }

      double raw_re[FXAI_AI_MLP_HIDDEN], raw_im[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double ur = 0.0, ui = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            ur += m_b_re[h][i] * xn[i];
            ui += m_b_im[h][i] * xn[i];
         }

         double qr = 0.0, qi = 0.0;
         for(int r=0; r<FXAI_S4_RANK; r++)
         {
            qr += m_u_re[h][r] * zr[r] - m_u_im[h][r] * zi[r];
            qi += m_u_re[h][r] * zi[r] + m_u_im[h][r] * zr[r];
         }

         double memr, memi, inr, ini;
         ComplexMul(A_re[h], A_im[h], prev_re[h], prev_im[h], memr, memi);
         ComplexMul(F_re[h], F_im[h], ur, ui, inr, ini);

         raw_re[h] = memr + inr + dtv[h] * qr;
         raw_im[h] = memi + ini + dtv[h] * qi;
      }

      // const normalization path.
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double rr = raw_re[h];
         double ii = raw_im[h];
         if(m_s_norm_ready)
         {
            rr = (rr - m_sr_mean[h]) / MathSqrt(m_sr_var[h] + 1e-6);
            ii = (ii - m_si_mean[h]) / MathSqrt(m_si_var[h] + 1e-6);
         }
         st_re[h] = FXAI_ClipSym(rr, 8.0);
         st_im[h] = FXAI_ClipSym(ii, 8.0);
      }

      LayerNorm(st_re);
      LayerNorm(st_im);

      z_cls = m_b_out;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         z_cls += m_c_re[h] * st_re[h] - m_c_im[h] * st_im[h];
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         z_cls += m_d_skip[i] * xn[i];

      double denom = (m_margin_ready ? m_margin_ema : 1.0);
      if(denom < 0.25) denom = 0.25;
      double zloc = z_cls / denom;
      p_local = FXAI_Sigmoid((m_prob_scale * zloc) + m_prob_bias);

      double amps[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         amps[h] = MathSqrt(st_re[h] * st_re[h] + st_im[h] * st_im[h] + 1e-8);

      mu1 = m_b_move_mu[0];
      mu3 = m_b_move_mu[1];
      mu5 = m_b_move_mu[2];
      logv = m_b_move_logv;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         mu1 += m_w_move_mu[0][h] * amps[h];
         mu3 += m_w_move_mu[1][h] * amps[h];
         mu5 += m_w_move_mu[2][h] * amps[h];
         logv += m_w_move_logv[h] * amps[h];
      }
      logv = FXAI_Clamp(logv, -4.0, 4.0);
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(80, 48);
   }

   void ForwardSequenceContext(const double &x[],
                               double &st_re[],
                               double &st_im[],
                               double &z_cls,
                               double &p_local,
                               double &mu1,
                               double &mu3,
                               double &mu5,
                               double &logv)
   {
      double saved_re[FXAI_AI_MLP_HIDDEN];
      double saved_im[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         saved_re[h] = m_state_re[h];
         saved_im[h] = m_state_im[h];
      }

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_STATE_SPACE, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildChronologicalSequenceTensorConfigured(x, seq_cfg, seq, seq_len);

      for(int t=0; t<seq_len; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         ForwardNoCommit(x_step, false, st_re, st_im, z_cls, p_local, mu1, mu3, mu5, logv);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_state_re[h] = st_re[h];
            m_state_im[h] = st_im[h];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_state_re[h] = saved_re[h];
         m_state_im[h] = saved_im[h];
      }
   }

public:
   CFXAIAIS4(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_S4; }
   virtual string AIName(void) const { return "ai_s4"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_STATE_SPACE, caps, 16, 256);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);

      double mean_combo = 0.50 * mu1 + 0.30 * mu3 + 0.20 * mu5;
      if(mean_combo < 0.0) mean_combo = 0.0;
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 20.0);
      expected_move_points = mean_combo + 0.30 * sigma;
      if(expected_move_points <= 0.0)
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      double min_move = MathMax(ResolveMinMovePoints(), 0.10);
      double cost = MathMax(ResolveCostPoints(x), 0.0);
      double active = FXAI_Clamp((expected_move_points - 0.35 * cost) / MathMax(min_move, 0.10), 0.0, 1.0);
      double p_dir = FXAI_Clamp(p_local, 0.001, 0.999);
      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_BUY] = p_dir * active;
      p_raw3[(int)FXAI_LABEL_SELL] = (1.0 - p_dir) * active;
      p_raw3[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      m_cal3.Calibrate(p_raw3, class_probs);
      NormalizeClassDistribution(class_probs);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);

      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);

      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return false;
      for(int c=0; c<3; c++)
         out.class_probs[c] = probs[c];

      double mean_combo = MathMax(0.0, 0.50 * mu1 + 0.30 * mu3 + 0.20 * mu5);
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 20.0);
      out.move_mean_points = MathMax(ev, (m_move_ready ? m_move_ema_abs : mean_combo));
      out.move_q25_points = MathMax(0.0, mean_combo - 0.60 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, mean_combo);
      out.move_q75_points = MathMax(out.move_q50_points, mean_combo + 0.60 * sigma);
      out.confidence = FXAI_Clamp(0.60 * MathMax(probs[(int)FXAI_LABEL_BUY], probs[(int)FXAI_LABEL_SELL]) + 0.20 * (1.0 - probs[(int)FXAI_LABEL_SKIP]) + 0.20 * (1.0 - FXAI_Clamp(m_margin_ema / 2.0, 0.0, 1.0)), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.55 + 0.20 * (m_margin_ready ? 1.0 - MathAbs(m_prob_bias) / 2.0 : 0.0) + 0.15 * (m_move_ready ? 1.0 : 0.0) + 0.10 * (1.0 - FXAI_Clamp(MathAbs(m_prob_scale - 1.0), 0.0, 1.0)), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(xa,
                              out.move_mean_points,
                              FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                              out.reliability,
                              out.confidence,
                              bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust,
                              out);
      return true;
   }


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_train_steps = 0;
      m_seen_updates = 0;
      ResetNormStats();
      ResetState();
      ResetBatch();
      ResetOptimizers();

      m_margin_ready = false;
      m_margin_ema = 1.0;
      m_prob_scale = 1.0;
      m_prob_bias = 0.0;
      m_quality_heads.Reset();
      m_cal3.Reset();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights();
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen_updates++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls == (int)FXAI_LABEL_SKIP) return;
      int y_dir = (cls == (int)FXAI_LABEL_BUY ? 1 : 0);
      double cls_w = 1.0;

      // Controlled reset policy to prevent state bleed in long runs and shock bars.
      if((m_seen_updates % 512) == 0 ||
         MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
      {
         ResetState();
         ResetBatch();
      }

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = FXAI_Clamp(MoveSampleWeight(xa, move_points) * cls_w, 0.10, 4.00);
      m_quality_heads.Update(xa,
                             w,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             h.lr,
                             h.l2);
      double pre_probs[3];
      double pre_ev = 0.0;
      if(PredictModelCore(x, h, pre_probs, pre_ev))
         m_cal3.Update(pre_probs, cls, w, h.lr);
      AppendBatch(y_dir, xa, move_points, w);

      if(m_batch_size >= FXAI_S4_TBPTT)
         TrainBatch(h);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);
      return CalibrateProb(p_local);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0)
         return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
};

#endif // __FXAI_AI_S4_MQH__
