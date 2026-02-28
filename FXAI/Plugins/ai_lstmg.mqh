// FXAI v1
#ifndef __FX6_AI_LSTMG_MQH__
#define __FX6_AI_LSTMG_MQH__

#include "..\plugin_base.mqh"

#define FX6_LSTMG_TBPTT 24
#define FX6_LSTMG_CLASS_COUNT 3
#define FX6_LSTMG_CAL_BINS 12
#define FX6_LSTMG_LN_EPS 0.00001
#define FX6_LSTMG_DROP_RATE 0.10
#define FX6_LSTMG_ZONEOUT 0.06
#define FX6_LSTMG_OPT_GROUPS 6

class CFX6AILSTMG : public CFX6AIPlugin
{
private:
   enum ENUM_LSTMG_GATE
   {
      LSTMG_GATE_I = 0,
      LSTMG_GATE_F,
      LSTMG_GATE_O,
      LSTMG_GATE_G
   };

   bool   m_initialized;
   int    m_step;
   int    m_seen_updates;
   int    m_adam_t;

   // Regime/session state.
   datetime m_last_update_time;
   int      m_last_session_bucket;
   bool     m_vol_ready;
   double   m_vol_ema;

   // Recurrent state.
   double m_h[FX6_AI_MLP_HIDDEN];
   double m_c[FX6_AI_MLP_HIDDEN];

   // Input/recurrent gate weights (LSTM-G with peepholes).
   double m_wi[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wf[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wo[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wg[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];

   double m_ui[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_uf[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_uo[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ug[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];

   double m_pi[FX6_AI_MLP_HIDDEN];
   double m_pf[FX6_AI_MLP_HIDDEN];
   double m_po[FX6_AI_MLP_HIDDEN];

   double m_bi[FX6_AI_MLP_HIDDEN];
   double m_bf[FX6_AI_MLP_HIDDEN];
   double m_bo[FX6_AI_MLP_HIDDEN];
   double m_bg[FX6_AI_MLP_HIDDEN];

   // Gate-wise LayerNorm parameters.
   double m_ln_gi[FX6_AI_MLP_HIDDEN];
   double m_ln_gf[FX6_AI_MLP_HIDDEN];
   double m_ln_go[FX6_AI_MLP_HIDDEN];
   double m_ln_gg[FX6_AI_MLP_HIDDEN];

   double m_ln_bi[FX6_AI_MLP_HIDDEN];
   double m_ln_bf[FX6_AI_MLP_HIDDEN];
   double m_ln_bo[FX6_AI_MLP_HIDDEN];
   double m_ln_bg[FX6_AI_MLP_HIDDEN];

   // Native 3-class directional head.
   double m_w_cls[FX6_LSTMG_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_LSTMG_CLASS_COUNT];

   // Distributional move head.
   double m_w_mu[FX6_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN];
   double m_b_q75;

   // EMA shadow weights for inference smoothing.
   bool   m_ema_ready;
   int    m_ema_steps;

   double m_ema_wi[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_ema_wf[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_ema_wo[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_ema_wg[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];

   double m_ema_ui[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ema_uf[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ema_uo[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_ema_ug[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];

   double m_ema_pi[FX6_AI_MLP_HIDDEN];
   double m_ema_pf[FX6_AI_MLP_HIDDEN];
   double m_ema_po[FX6_AI_MLP_HIDDEN];

   double m_ema_bi[FX6_AI_MLP_HIDDEN];
   double m_ema_bf[FX6_AI_MLP_HIDDEN];
   double m_ema_bo[FX6_AI_MLP_HIDDEN];
   double m_ema_bg[FX6_AI_MLP_HIDDEN];

   double m_ema_ln_gi[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_gf[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_go[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_gg[FX6_AI_MLP_HIDDEN];

   double m_ema_ln_bi[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_bf[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_bo[FX6_AI_MLP_HIDDEN];
   double m_ema_ln_bg[FX6_AI_MLP_HIDDEN];

   double m_ema_w_cls[FX6_LSTMG_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_ema_b_cls[FX6_LSTMG_CLASS_COUNT];

   double m_ema_w_mu[FX6_AI_MLP_HIDDEN];
   double m_ema_b_mu;
   double m_ema_w_logv[FX6_AI_MLP_HIDDEN];
   double m_ema_b_logv;
   double m_ema_w_q25[FX6_AI_MLP_HIDDEN];
   double m_ema_b_q25;
   double m_ema_w_q75[FX6_AI_MLP_HIDDEN];
   double m_ema_b_q75;

   // Sequence batching (TBPTT).
   int    m_batch_size;
   double m_batch_h0[FX6_AI_MLP_HIDDEN];
   double m_batch_c0[FX6_AI_MLP_HIDDEN];
   double m_batch_x[FX6_LSTMG_TBPTT][FX6_AI_WEIGHTS];
   int    m_batch_cls[FX6_LSTMG_TBPTT];
   double m_batch_move[FX6_LSTMG_TBPTT];
   double m_batch_cost[FX6_LSTMG_TBPTT];
   double m_batch_w[FX6_LSTMG_TBPTT];
   double m_batch_reset[FX6_LSTMG_TBPTT];

   // Optimizer moments (grouped AdamW-style scaling).
   double m_opt_m[FX6_LSTMG_OPT_GROUPS];
   double m_opt_v[FX6_LSTMG_OPT_GROUPS];

   // Native multiclass calibrator.
   double m_cal3_temp;
   double m_cal3_bias[FX6_LSTMG_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_LSTMG_CLASS_COUNT][FX6_LSTMG_CAL_BINS];
   double m_cal3_iso_cnt[FX6_LSTMG_CLASS_COUNT][FX6_LSTMG_CAL_BINS];
   int    m_cal3_steps;

   // EV calibration for move output.
   bool   m_ev_ready;
   int    m_ev_steps;
   double m_ev_a;
   double m_ev_b;

   // Session/class balancing stats.
   double m_cls_count[4][FX6_LSTMG_CLASS_COUNT];
   double m_sess_total[4];

   // Built-in correctness checks.
   bool   m_selftest_done;
   bool   m_selftest_ok;

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

   double Hash01(const int a, const int b, const int c) const
   {
      double z = MathSin(12.9898 * (double)(a + 1) +
                         78.2330 * (double)(b + 2) +
                         37.7190 * (double)(c + 3));
      double x = MathAbs(z * 43758.5453);
      return x - MathFloor(x);
   }

   double DropScale(const int salt, const int idx) const
   {
      double keep = 1.0 - FX6_LSTMG_DROP_RATE;
      if(keep <= 0.05) keep = 0.05;
      double u = Hash01(m_step + 7, salt, idx);
      if(u < FX6_LSTMG_DROP_RATE) return 0.0;
      return 1.0 / keep;
   }

   bool ZoneKeep(const int salt, const int idx) const
   {
      double u = Hash01(m_seen_updates + 11, salt, idx);
      return (u < FX6_LSTMG_ZONEOUT);
   }

   double BlendParam(const double base, const double ema, const bool use_ema) const
   {
      if(!use_ema) return base;
      return 0.15 * base + 0.85 * ema;
   }

   bool UseEMAInference(void) const
   {
      return (m_ema_ready && m_ema_steps >= 12);
   }

   double AdamGroupLR(const int group_idx,
                      const double grad_mag,
                      const double base_lr)
   {
      int g = group_idx;
      if(g < 0) g = 0;
      if(g >= FX6_LSTMG_OPT_GROUPS) g = FX6_LSTMG_OPT_GROUPS - 1;

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

      return FX6_Clamp(base_lr * (0.55 + 0.45 * scale), 0.000002, 0.060000);
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

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FX6_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FX6_LSTMG_CLASS_COUNT];
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_LSTMG_CLASS_COUNT];
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_LSTMG_CAL_BINS; b++) total += m_cal3_iso_cnt[c][b];
         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_LSTMG_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_LSTMG_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_LSTMG_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_LSTMG_CAL_BINS) bi = FX6_LSTMG_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
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
      double logits[FX6_LSTMG_CLASS_COUNT];
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }

      double p_cal[FX6_LSTMG_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FX6_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_LSTMG_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_LSTMG_CAL_BINS) bi = FX6_LSTMG_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_temp = FX6_Clamp(m_cal3_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   double CalibrateEV(const double raw_ev) const
   {
      double x = MathMax(0.0, raw_ev);
      if(!m_ev_ready) return x;
      double y = (m_ev_a * x) + m_ev_b;
      return FX6_Clamp(y, 0.0, 300.0);
   }

   void UpdateEVCalibration(const double raw_ev,
                            const double target_ev,
                            const double sample_w,
                            const double base_lr)
   {
      double x = MathMax(0.0, raw_ev);
      double y = MathMax(0.0, target_ev);
      double p = (m_ev_a * x) + m_ev_b;
      double err = FX6_ClipSym(y - p, 40.0);

      double w = FX6_Clamp(sample_w, 0.25, 6.0);
      double lr = FX6_Clamp(0.08 * base_lr * w, 0.00005, 0.01000);
      double reg = 0.0005;

      m_ev_a += lr * (err * x - reg * (m_ev_a - 1.0));
      m_ev_b += lr * err;

      m_ev_a = FX6_Clamp(m_ev_a, 0.10, 4.00);
      m_ev_b = FX6_Clamp(m_ev_b, -25.0, 25.0);
      m_ev_steps++;
      if(m_ev_steps >= 24) m_ev_ready = true;
   }

   void LayerNormForward(const double &z_in[],
                         const double &g_base[],
                         const double &b_base[],
                         const double &g_ema[],
                         const double &b_ema[],
                         const bool use_ema,
                         double &z_hat[],
                         double &z_out[],
                         double &mean,
                         double &inv_std) const
   {
      mean = 0.0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) mean += z_in[h];
      mean /= (double)FX6_AI_MLP_HIDDEN;

      double var = 0.0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double d = z_in[h] - mean;
         var += d * d;
      }
      var /= (double)FX6_AI_MLP_HIDDEN;

      inv_std = 1.0 / MathSqrt(var + FX6_LSTMG_LN_EPS);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         z_hat[h] = (z_in[h] - mean) * inv_std;
         double g = BlendParam(g_base[h], g_ema[h], use_ema);
         double b = BlendParam(b_base[h], b_ema[h], use_ema);
         z_out[h] = g * z_hat[h] + b;
      }
   }

   void LayerNormBackward(const double &dy_ln[],
                          const double &z_hat[],
                          const double inv_std,
                          const double &gamma[],
                          double &g_gamma[],
                          double &g_beta[],
                          double &dz_raw[]) const
   {
      double dzhat[FX6_AI_MLP_HIDDEN];
      double sum1 = 0.0;
      double sum2 = 0.0;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double dy = dy_ln[h];
         g_gamma[h] += dy * z_hat[h];
         g_beta[h] += dy;

         dzhat[h] = dy * gamma[h];
         sum1 += dzhat[h];
         sum2 += dzhat[h] * z_hat[h];
      }

      double n = (double)FX6_AI_MLP_HIDDEN;
      double k = inv_std / n;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double v = (n * dzhat[h]) - sum1 - z_hat[h] * sum2;
         dz_raw[h] = k * v;
      }
   }

   int CheckRegimeBoundary(const double &x[], int &session_bucket)
   {
      datetime now = ResolveContextTime();
      if(now <= 0) now = TimeCurrent();

      session_bucket = SessionBucket(now);
      int boundary = 0;

      if(m_last_update_time > 0)
      {
         int gap_sec = (int)(now - m_last_update_time);
         if(gap_sec > 90 * 60)
            boundary = 2;
      }

      double shock = MathAbs(x[1]) + MathAbs(x[2]) + 0.5 * MathAbs(x[5]);
      if(!m_vol_ready)
      {
         m_vol_ema = shock;
         m_vol_ready = true;
      }
      else
      {
         m_vol_ema = 0.97 * m_vol_ema + 0.03 * shock;
         if(m_vol_ema > 1e-6 && shock > 3.8 * m_vol_ema)
            boundary = 2;
      }

      if(boundary == 0 && m_last_session_bucket >= 0 && session_bucket != m_last_session_bucket)
         boundary = 1;

      m_last_update_time = now;
      m_last_session_bucket = session_bucket;
      return boundary;
   }

   void ResetState(void)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = 0.0;
         m_c[h] = 0.0;
      }
   }

   void DecayState(const double k)
   {
      double d = FX6_Clamp(k, 0.0, 1.0);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_h[h] *= d;
         m_c[h] *= d;
      }
   }

   void ResetBatch(void)
   {
      m_batch_size = 0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_batch_h0[h] = m_h[h];
         m_batch_c0[h] = m_c[h];
      }
   }

   void SyncEMAWithParams(void)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_ema_pi[h] = m_pi[h];
         m_ema_pf[h] = m_pf[h];
         m_ema_po[h] = m_po[h];

         m_ema_bi[h] = m_bi[h];
         m_ema_bf[h] = m_bf[h];
         m_ema_bo[h] = m_bo[h];
         m_ema_bg[h] = m_bg[h];

         m_ema_ln_gi[h] = m_ln_gi[h];
         m_ema_ln_gf[h] = m_ln_gf[h];
         m_ema_ln_go[h] = m_ln_go[h];
         m_ema_ln_gg[h] = m_ln_gg[h];

         m_ema_ln_bi[h] = m_ln_bi[h];
         m_ema_ln_bf[h] = m_ln_bf[h];
         m_ema_ln_bo[h] = m_ln_bo[h];
         m_ema_ln_bg[h] = m_ln_bg[h];

         m_ema_w_mu[h] = m_w_mu[h];
         m_ema_w_logv[h] = m_w_logv[h];
         m_ema_w_q25[h] = m_w_q25[h];
         m_ema_w_q75[h] = m_w_q75[h];

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_ema_w_cls[c][h] = m_w_cls[c][h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_ema_wi[h][i] = m_wi[h][i];
            m_ema_wf[h][i] = m_wf[h][i];
            m_ema_wo[h][i] = m_wo[h][i];
            m_ema_wg[h][i] = m_wg[h][i];
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_ema_ui[h][j] = m_ui[h][j];
            m_ema_uf[h][j] = m_uf[h][j];
            m_ema_uo[h][j] = m_uo[h][j];
            m_ema_ug[h][j] = m_ug[h][j];
         }
      }

      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
         m_ema_b_cls[c] = m_b_cls[c];

      m_ema_b_mu = m_b_mu;
      m_ema_b_logv = m_b_logv;
      m_ema_b_q25 = m_b_q25;
      m_ema_b_q75 = m_b_q75;

      m_ema_steps = 0;
      m_ema_ready = false;
   }

   void UpdateEMAFromParams(const double decay)
   {
      double d = FX6_Clamp(decay, 0.9000, 0.9999);
      double one = 1.0 - d;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_ema_pi[h] = d * m_ema_pi[h] + one * m_pi[h];
         m_ema_pf[h] = d * m_ema_pf[h] + one * m_pf[h];
         m_ema_po[h] = d * m_ema_po[h] + one * m_po[h];

         m_ema_bi[h] = d * m_ema_bi[h] + one * m_bi[h];
         m_ema_bf[h] = d * m_ema_bf[h] + one * m_bf[h];
         m_ema_bo[h] = d * m_ema_bo[h] + one * m_bo[h];
         m_ema_bg[h] = d * m_ema_bg[h] + one * m_bg[h];

         m_ema_ln_gi[h] = d * m_ema_ln_gi[h] + one * m_ln_gi[h];
         m_ema_ln_gf[h] = d * m_ema_ln_gf[h] + one * m_ln_gf[h];
         m_ema_ln_go[h] = d * m_ema_ln_go[h] + one * m_ln_go[h];
         m_ema_ln_gg[h] = d * m_ema_ln_gg[h] + one * m_ln_gg[h];

         m_ema_ln_bi[h] = d * m_ema_ln_bi[h] + one * m_ln_bi[h];
         m_ema_ln_bf[h] = d * m_ema_ln_bf[h] + one * m_ln_bf[h];
         m_ema_ln_bo[h] = d * m_ema_ln_bo[h] + one * m_ln_bo[h];
         m_ema_ln_bg[h] = d * m_ema_ln_bg[h] + one * m_ln_bg[h];

         m_ema_w_mu[h] = d * m_ema_w_mu[h] + one * m_w_mu[h];
         m_ema_w_logv[h] = d * m_ema_w_logv[h] + one * m_w_logv[h];
         m_ema_w_q25[h] = d * m_ema_w_q25[h] + one * m_w_q25[h];
         m_ema_w_q75[h] = d * m_ema_w_q75[h] + one * m_w_q75[h];

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_ema_w_cls[c][h] = d * m_ema_w_cls[c][h] + one * m_w_cls[c][h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_ema_wi[h][i] = d * m_ema_wi[h][i] + one * m_wi[h][i];
            m_ema_wf[h][i] = d * m_ema_wf[h][i] + one * m_wf[h][i];
            m_ema_wo[h][i] = d * m_ema_wo[h][i] + one * m_wo[h][i];
            m_ema_wg[h][i] = d * m_ema_wg[h][i] + one * m_wg[h][i];
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_ema_ui[h][j] = d * m_ema_ui[h][j] + one * m_ui[h][j];
            m_ema_uf[h][j] = d * m_ema_uf[h][j] + one * m_uf[h][j];
            m_ema_uo[h][j] = d * m_ema_uo[h][j] + one * m_uo[h][j];
            m_ema_ug[h][j] = d * m_ema_ug[h][j] + one * m_ug[h][j];
         }
      }

      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
         m_ema_b_cls[c] = d * m_ema_b_cls[c] + one * m_b_cls[c];

      m_ema_b_mu = d * m_ema_b_mu + one * m_b_mu;
      m_ema_b_logv = d * m_ema_b_logv + one * m_b_logv;
      m_ema_b_q25 = d * m_ema_b_q25 + one * m_b_q25;
      m_ema_b_q75 = d * m_ema_b_q75 + one * m_b_q75;

      m_ema_steps++;
      if(m_ema_steps >= 8) m_ema_ready = true;
   }

   void InitWeights(void)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double s = (double)((h + 1) * (i + 7));
            m_wi[h][i] = 0.028 * MathSin(s * 0.71);
            m_wf[h][i] = 0.028 * MathCos(s * 0.73);
            m_wo[h][i] = 0.028 * MathSin(s * 0.79);
            m_wg[h][i] = 0.028 * MathCos(s * 0.83);
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            double t = (double)((h + 3) * (j + 2));
            m_ui[h][j] = 0.018 * MathSin(t * 0.89);
            m_uf[h][j] = 0.018 * MathCos(t * 0.91);
            m_uo[h][j] = 0.018 * MathSin(t * 0.97);
            m_ug[h][j] = 0.018 * MathCos(t * 1.01);
         }

         m_pi[h] = 0.01 * MathSin((double)(h + 1) * 1.27);
         m_pf[h] = 0.01 * MathCos((double)(h + 1) * 1.31);
         m_po[h] = 0.01 * MathSin((double)(h + 1) * 1.37);

         m_bi[h] = 0.0;
         m_bf[h] = 0.3;
         m_bo[h] = 0.0;
         m_bg[h] = 0.0;

         m_ln_gi[h] = 1.0;
         m_ln_gf[h] = 1.0;
         m_ln_go[h] = 1.0;
         m_ln_gg[h] = 1.0;

         m_ln_bi[h] = 0.0;
         m_ln_bf[h] = 0.0;
         m_ln_bo[h] = 0.0;
         m_ln_bg[h] = 0.0;

         m_w_mu[h] = 0.03 * MathSin((double)(h + 1) * 1.29);
         m_w_logv[h] = 0.03 * MathCos((double)(h + 2) * 1.23);
         m_w_q25[h] = 0.02 * MathSin((double)(h + 3) * 1.17);
         m_w_q75[h] = 0.02 * MathCos((double)(h + 4) * 1.11);

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_w_cls[c][h] = 0.04 * MathSin((double)((c + 2) * (h + 1)) * 1.07);
      }

      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++) m_b_cls[c] = 0.0;
      m_b_cls[(int)FX6_LABEL_SKIP] = 0.15;

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FX6_LSTMG_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      m_ev_ready = false;
      m_ev_steps = 0;
      m_ev_a = 1.0;
      m_ev_b = 0.0;

      for(int s=0; s<4; s++)
      {
         m_sess_total[s] = 0.0;
         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_cls_count[s][c] = 0.0;
      }

      for(int g=0; g<FX6_LSTMG_OPT_GROUPS; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      SyncEMAWithParams();
      ResetState();
      ResetBatch();
      m_initialized = true;
   }

   double DotInput(const int gate,
                   const int row,
                   const double &x[],
                   const bool use_ema) const
   {
      double s = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         double b = 0.0;
         double e = 0.0;
         if(gate == (int)LSTMG_GATE_I)      { b = m_wi[row][i]; e = m_ema_wi[row][i]; }
         else if(gate == (int)LSTMG_GATE_F) { b = m_wf[row][i]; e = m_ema_wf[row][i]; }
         else if(gate == (int)LSTMG_GATE_O) { b = m_wo[row][i]; e = m_ema_wo[row][i]; }
         else                               { b = m_wg[row][i]; e = m_ema_wg[row][i]; }

         s += BlendParam(b, e, use_ema) * x[i];
      }
      return s;
   }

   double DotState(const int gate,
                   const int row,
                   const double &hprev[],
                   const bool use_ema) const
   {
      double s = 0.0;
      for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
      {
         double b = 0.0;
         double e = 0.0;
         if(gate == (int)LSTMG_GATE_I)      { b = m_ui[row][j]; e = m_ema_ui[row][j]; }
         else if(gate == (int)LSTMG_GATE_F) { b = m_uf[row][j]; e = m_ema_uf[row][j]; }
         else if(gate == (int)LSTMG_GATE_O) { b = m_uo[row][j]; e = m_ema_uo[row][j]; }
         else                               { b = m_ug[row][j]; e = m_ema_ug[row][j]; }

         s += BlendParam(b, e, use_ema) * hprev[j];
      }
      return s;
   }

   void ForwardOne(const double &x[],
                   const double &h_prev[],
                   const double &c_prev[],
                   const bool training,
                   const bool use_ema,
                   const int salt,
                   double &drop_mask[],
                   double &zone_mask[],
                   double &zi_hat[],
                   double &zf_hat[],
                   double &zo_hat[],
                   double &zg_hat[],
                   double &inv_i,
                   double &inv_f,
                   double &inv_o,
                   double &inv_g,
                   double &ig[],
                   double &fg[],
                   double &og[],
                   double &gg[],
                   double &c_new[],
                   double &h_new[]) const
   {
      double h_in[FX6_AI_MLP_HIDDEN];
      for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
      {
         double dm = 1.0;
         if(training) dm = DropScale(salt, j);
         drop_mask[j] = dm;
         h_in[j] = h_prev[j] * dm;
         zone_mask[j] = 0.0;
      }

      double zi_raw[FX6_AI_MLP_HIDDEN];
      double zf_raw[FX6_AI_MLP_HIDDEN];
      double zg_raw[FX6_AI_MLP_HIDDEN];
      double zo_raw[FX6_AI_MLP_HIDDEN];

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double pi = BlendParam(m_pi[h], m_ema_pi[h], use_ema);
         double pf = BlendParam(m_pf[h], m_ema_pf[h], use_ema);

         zi_raw[h] = DotInput((int)LSTMG_GATE_I, h, x, use_ema) +
                     DotState((int)LSTMG_GATE_I, h, h_in, use_ema) +
                     pi * c_prev[h] +
                     BlendParam(m_bi[h], m_ema_bi[h], use_ema);

         zf_raw[h] = DotInput((int)LSTMG_GATE_F, h, x, use_ema) +
                     DotState((int)LSTMG_GATE_F, h, h_in, use_ema) +
                     pf * c_prev[h] +
                     BlendParam(m_bf[h], m_ema_bf[h], use_ema);

         zg_raw[h] = DotInput((int)LSTMG_GATE_G, h, x, use_ema) +
                     DotState((int)LSTMG_GATE_G, h, h_in, use_ema) +
                     BlendParam(m_bg[h], m_ema_bg[h], use_ema);
      }

      double zi_ln[FX6_AI_MLP_HIDDEN];
      double zf_ln[FX6_AI_MLP_HIDDEN];
      double zg_ln[FX6_AI_MLP_HIDDEN];
      double mean_i, mean_f, mean_g;
      LayerNormForward(zi_raw, m_ln_gi, m_ln_bi, m_ema_ln_gi, m_ema_ln_bi, use_ema,
                       zi_hat, zi_ln, mean_i, inv_i);
      LayerNormForward(zf_raw, m_ln_gf, m_ln_bf, m_ema_ln_gf, m_ema_ln_bf, use_ema,
                       zf_hat, zf_ln, mean_f, inv_f);
      LayerNormForward(zg_raw, m_ln_gg, m_ln_bg, m_ema_ln_gg, m_ema_ln_bg, use_ema,
                       zg_hat, zg_ln, mean_g, inv_g);

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         ig[h] = FX6_Sigmoid(zi_ln[h]);
         fg[h] = FX6_Sigmoid(zf_ln[h]);
         gg[h] = FX6_Tanh(zg_ln[h]);

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         c_new[h] = FX6_ClipSym(c_new[h], 10.0);
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double po = BlendParam(m_po[h], m_ema_po[h], use_ema);
         zo_raw[h] = DotInput((int)LSTMG_GATE_O, h, x, use_ema) +
                     DotState((int)LSTMG_GATE_O, h, h_in, use_ema) +
                     po * c_new[h] +
                     BlendParam(m_bo[h], m_ema_bo[h], use_ema);
      }

      double zo_ln[FX6_AI_MLP_HIDDEN];
      double mean_o;
      LayerNormForward(zo_raw, m_ln_go, m_ln_bo, m_ema_ln_go, m_ema_ln_bo, use_ema,
                       zo_hat, zo_ln, mean_o, inv_o);

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         og[h] = FX6_Sigmoid(zo_ln[h]);
         h_new[h] = og[h] * FX6_Tanh(c_new[h]);

         if(training && ZoneKeep(salt, h))
         {
            zone_mask[h] = 1.0;
            c_new[h] = c_prev[h];
            h_new[h] = h_prev[h];
         }
      }
   }

   void ComputeHeads(const double &hvec[],
                     const bool use_ema,
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75) const
   {
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         double z = BlendParam(m_b_cls[c], m_ema_b_cls[c], use_ema);
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            z += BlendParam(m_w_cls[c][h], m_ema_w_cls[c][h], use_ema) * hvec[h];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = BlendParam(m_b_mu, m_ema_b_mu, use_ema);
      logv = BlendParam(m_b_logv, m_ema_b_logv, use_ema);
      q25 = BlendParam(m_b_q25, m_ema_b_q25, use_ema);
      q75 = BlendParam(m_b_q75, m_ema_b_q75, use_ema);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         mu += BlendParam(m_w_mu[h], m_ema_w_mu[h], use_ema) * hvec[h];
         logv += BlendParam(m_w_logv[h], m_ema_w_logv[h], use_ema) * hvec[h];
         q25 += BlendParam(m_w_q25[h], m_ema_w_q25[h], use_ema) * hvec[h];
         q75 += BlendParam(m_w_q75[h], m_ema_w_q75[h], use_ema) * hvec[h];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double t = q25;
         q25 = q75;
         q75 = t;
      }
   }

   double ExpectedMoveFromHeads(const double mu,
                                const double logv,
                                const double q25,
                                const double q75,
                                const double p_skip) const
   {
      double sigma = MathExp(0.5 * FX6_Clamp(logv, -4.0, 4.0));
      sigma = FX6_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);

      double ev = (0.68 * MathAbs(mu) + 0.22 * sigma + 0.10 * iqr) *
                  FX6_Clamp(1.0 - p_skip, 0.0, 1.0);
      return ev;
   }

   void AppendBatch(const int cls,
                    const double &x[],
                    const double move_points,
                    const double cost_points,
                    const double sample_w,
                    const double reset_flag)
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_batch_h0[h] = m_h[h];
            m_batch_c0[h] = m_c[h];
         }
      }

      if(m_batch_size < FX6_LSTMG_TBPTT)
      {
         int p = m_batch_size;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[p][i] = x[i];
         m_batch_cls[p] = cls;
         m_batch_move[p] = move_points;
         m_batch_cost[p] = cost_points;
         m_batch_w[p] = sample_w;
         m_batch_reset[p] = reset_flag;
         m_batch_size++;
         return;
      }

      for(int t=1; t<FX6_LSTMG_TBPTT; t++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[t - 1][i] = m_batch_x[t][i];
         m_batch_cls[t - 1] = m_batch_cls[t];
         m_batch_move[t - 1] = m_batch_move[t];
         m_batch_cost[t - 1] = m_batch_cost[t];
         m_batch_w[t - 1] = m_batch_w[t];
         m_batch_reset[t - 1] = m_batch_reset[t];
      }

      int last = FX6_LSTMG_TBPTT - 1;
      for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[last][i] = x[i];
      m_batch_cls[last] = cls;
      m_batch_move[last] = move_points;
      m_batch_cost[last] = cost_points;
      m_batch_w[last] = sample_w;
      m_batch_reset[last] = reset_flag;
      m_batch_size = FX6_LSTMG_TBPTT;
   }

   void BuildStateWithPending(double &h_out[],
                              double &c_out[],
                              const bool use_ema) const
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            h_out[h] = m_h[h];
            c_out[h] = m_c[h];
         }
         return;
      }

      double h_cur[FX6_AI_MLP_HIDDEN];
      double c_cur[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         h_cur[h] = m_batch_h0[h];
         c_cur[h] = m_batch_c0[h];
      }

      for(int t=0; t<m_batch_size; t++)
      {
         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               h_cur[h] = 0.0;
               c_cur[h] = 0.0;
            }
         }

         double drop_mask[FX6_AI_MLP_HIDDEN];
         double zone_mask[FX6_AI_MLP_HIDDEN];
         double zi_hat[FX6_AI_MLP_HIDDEN];
         double zf_hat[FX6_AI_MLP_HIDDEN];
         double zo_hat[FX6_AI_MLP_HIDDEN];
         double zg_hat[FX6_AI_MLP_HIDDEN];
         double inv_i, inv_f, inv_o, inv_g;
         double ig[FX6_AI_MLP_HIDDEN];
         double fg[FX6_AI_MLP_HIDDEN];
         double og[FX6_AI_MLP_HIDDEN];
         double gg[FX6_AI_MLP_HIDDEN];
         double c_new[FX6_AI_MLP_HIDDEN];
         double h_new[FX6_AI_MLP_HIDDEN];

         double xloc[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];

         ForwardOne(xloc, h_cur, c_cur, false, use_ema, t,
                    drop_mask, zone_mask,
                    zi_hat, zf_hat, zo_hat, zg_hat,
                    inv_i, inv_f, inv_o, inv_g,
                    ig, fg, og, gg, c_new, h_new);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            h_cur[h] = h_new[h];
            c_cur[h] = c_new[h];
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         h_out[h] = h_cur[h];
         c_out[h] = c_cur[h];
      }
   }

   double ClassBalanceWeight(const int sess,
                             const int cls) const
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s > 3) s = 3;
      int c = cls;
      if(c < 0) c = 0;
      if(c >= FX6_LSTMG_CLASS_COUNT) c = FX6_LABEL_SKIP;

      double cnt = m_cls_count[s][c] + 1.0;
      double tot = m_sess_total[s] + (double)FX6_LSTMG_CLASS_COUNT;
      double ideal = tot / (double)FX6_LSTMG_CLASS_COUNT;
      double w = MathSqrt(ideal / cnt);
      return FX6_Clamp(w, 0.55, 2.50);
   }

   void UpdateClassStats(const int sess,
                         const int cls)
   {
      int s = sess;
      if(s < 0) s = 0;
      if(s > 3) s = 3;
      int c = cls;
      if(c < 0) c = 0;
      if(c >= FX6_LSTMG_CLASS_COUNT) c = FX6_LABEL_SKIP;

      m_cls_count[s][c] += 1.0;
      m_sess_total[s] += 1.0;
   }

   int DeriveClassLabel(const int y,
                        const double &x[],
                        const double move_points,
                        const double cost_points,
                        const double min_move_points) const
   {
      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
      {
         if(y == 1) cls = (int)FX6_LABEL_BUY;
         else if(y == 0) cls = (int)FX6_LABEL_SELL;
         else cls = (move_points >= 0.0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
      }

      double cost = MathMax(0.0, cost_points);
      double mm = MathMax(min_move_points, 0.10);
      double edge = MathAbs(move_points) - cost;
      double skip_band = MathMax(0.25 * mm, 0.30 * cost + 0.10);

      if(edge <= skip_band)
         return (int)FX6_LABEL_SKIP;

      if(cls == (int)FX6_LABEL_SKIP)
         return (move_points >= 0.0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);

      return cls;
   }

   bool RunSelfTests(void)
   {
      if(m_selftest_done) return m_selftest_ok;

      bool ok = true;

      // 1) Deterministic replay for forward path.
      double x[FX6_AI_WEIGHTS];
      double h0[FX6_AI_MLP_HIDDEN];
      double c0[FX6_AI_MLP_HIDDEN];
      for(int i=0; i<FX6_AI_WEIGHTS; i++) x[i] = 0.05 * MathSin((double)(i + 1) * 0.31);
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         h0[h] = 0.07 * MathCos((double)(h + 1) * 0.19);
         c0[h] = 0.06 * MathSin((double)(h + 2) * 0.23);
      }

      double drop1[FX6_AI_MLP_HIDDEN], drop2[FX6_AI_MLP_HIDDEN];
      double zone1[FX6_AI_MLP_HIDDEN], zone2[FX6_AI_MLP_HIDDEN];
      double zih1[FX6_AI_MLP_HIDDEN], zfh1[FX6_AI_MLP_HIDDEN], zoh1[FX6_AI_MLP_HIDDEN], zgh1[FX6_AI_MLP_HIDDEN];
      double zih2[FX6_AI_MLP_HIDDEN], zfh2[FX6_AI_MLP_HIDDEN], zoh2[FX6_AI_MLP_HIDDEN], zgh2[FX6_AI_MLP_HIDDEN];
      double ii1, if1, io1, ig1, ii2, if2, io2, ig2;
      double i1[FX6_AI_MLP_HIDDEN], f1[FX6_AI_MLP_HIDDEN], o1[FX6_AI_MLP_HIDDEN], g1[FX6_AI_MLP_HIDDEN], c1[FX6_AI_MLP_HIDDEN], h1[FX6_AI_MLP_HIDDEN];
      double i2[FX6_AI_MLP_HIDDEN], f2[FX6_AI_MLP_HIDDEN], o2[FX6_AI_MLP_HIDDEN], g2[FX6_AI_MLP_HIDDEN], c2[FX6_AI_MLP_HIDDEN], h2[FX6_AI_MLP_HIDDEN];

      ForwardOne(x, h0, c0, false, false, 11,
                 drop1, zone1, zih1, zfh1, zoh1, zgh1, ii1, if1, io1, ig1,
                 i1, f1, o1, g1, c1, h1);
      ForwardOne(x, h0, c0, false, false, 11,
                 drop2, zone2, zih2, zfh2, zoh2, zgh2, ii2, if2, io2, ig2,
                 i2, f2, o2, g2, c2, h2);

      double maxd = 0.0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double d = MathAbs(h1[h] - h2[h]) + MathAbs(c1[h] - c2[h]);
         if(d > maxd) maxd = d;
      }
      if(maxd > 1e-12) ok = false;

      // 2) Finite-difference check on class-head gradient.
      double logits[FX6_LSTMG_CLASS_COUNT];
      double probs[FX6_LSTMG_CLASS_COUNT];
      double mu, lv, q25, q75;
      ComputeHeads(h1, false, logits, probs, mu, lv, q25, q75);

      int cls = (int)FX6_LABEL_BUY;
      double analytic = (1.0 - probs[cls]) * h1[0];

      double eps = 1e-4;
      double orig = m_w_cls[cls][0];
      m_w_cls[cls][0] = orig + eps;
      double l1[FX6_LSTMG_CLASS_COUNT], p1[FX6_LSTMG_CLASS_COUNT];
      ComputeHeads(h1, false, l1, p1, mu, lv, q25, q75);
      double lp1 = MathLog(FX6_Clamp(p1[cls], 1e-8, 1.0));

      m_w_cls[cls][0] = orig - eps;
      double l2[FX6_LSTMG_CLASS_COUNT], p2[FX6_LSTMG_CLASS_COUNT];
      ComputeHeads(h1, false, l2, p2, mu, lv, q25, q75);
      double lp2 = MathLog(FX6_Clamp(p2[cls], 1e-8, 1.0));

      m_w_cls[cls][0] = orig;
      double numeric = (lp1 - lp2) / (2.0 * eps);
      if(MathAbs(numeric - analytic) > 0.15) ok = false;

      // 3) Sequence reset consistency.
      double x1[FX6_AI_WEIGHTS], x2[FX6_AI_WEIGHTS];
      double hs[FX6_AI_MLP_HIDDEN], cs[FX6_AI_MLP_HIDDEN];
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         x1[i] = 0.06 * MathSin((double)(i + 3) * 0.27);
         x2[i] = 0.05 * MathCos((double)(i + 5) * 0.33);
      }
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         hs[h] = 0.11 * MathSin((double)(h + 1) * 0.37);
         cs[h] = 0.09 * MathCos((double)(h + 2) * 0.41);
      }

      double h_after1[FX6_AI_MLP_HIDDEN], c_after1[FX6_AI_MLP_HIDDEN];
      ForwardOne(x1, hs, cs, false, false, 5,
                 drop1, zone1, zih1, zfh1, zoh1, zgh1, ii1, if1, io1, ig1,
                 i1, f1, o1, g1, c_after1, h_after1);

      double h_noreset[FX6_AI_MLP_HIDDEN], c_noreset[FX6_AI_MLP_HIDDEN];
      ForwardOne(x2, h_after1, c_after1, false, false, 6,
                 drop1, zone1, zih1, zfh1, zoh1, zgh1, ii1, if1, io1, ig1,
                 i1, f1, o1, g1, c_noreset, h_noreset);

      double hz[FX6_AI_MLP_HIDDEN], cz[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) { hz[h] = 0.0; cz[h] = 0.0; }
      double h_reset[FX6_AI_MLP_HIDDEN], c_reset[FX6_AI_MLP_HIDDEN];
      ForwardOne(x2, hz, cz, false, false, 6,
                 drop1, zone1, zih1, zfh1, zoh1, zgh1, ii1, if1, io1, ig1,
                 i1, f1, o1, g1, c_reset, h_reset);

      // "Masked reset" behavior should match zero-state pass exactly.
      double h_mask[FX6_AI_MLP_HIDDEN], c_mask[FX6_AI_MLP_HIDDEN];
      ForwardOne(x2, hz, cz, false, false, 6,
                 drop2, zone2, zih2, zfh2, zoh2, zgh2, ii2, if2, io2, ig2,
                 i2, f2, o2, g2, c_mask, h_mask);

      double d_reset = 0.0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double d = MathAbs(h_reset[h] - h_mask[h]) + MathAbs(c_reset[h] - c_mask[h]);
         if(d > d_reset) d_reset = d;
      }
      if(d_reset > 1e-12) ok = false;

      m_selftest_done = true;
      m_selftest_ok = ok;
      if(!ok)
         Print("FX6 lstmg self-test warning: one or more checks failed");
      return ok;
   }

   void TrainBatch(const FX6AIHyperParams &hp)
   {
      int len = m_batch_size;
      if(len <= 0) return;

      m_step += len;
      m_adam_t++;

      double lr0 = FX6_Clamp(hp.lr, 0.00005, 0.30000);
      double warm = FX6_Clamp((double)m_step / 512.0, 0.05, 1.00);
      double invs = 1.0 / MathSqrt(1.0 + 0.0008 * MathMax(0.0, (double)m_step - 256.0));
      double phase = MathMod((double)m_step, 4096.0) / 4096.0;
      double cosine = 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
      double base_lr = FX6_Clamp(lr0 * warm * invs * (0.65 + 0.35 * cosine), 0.00001, 0.05000);
      double l2 = FX6_Clamp(hp.l2, 0.0000, 0.2000);

      double h_prev[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double c_prev[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];

      double drop_mask[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double zone_mask[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];

      double zi_hat[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double zf_hat[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double zo_hat[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double zg_hat[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];

      double inv_i[FX6_LSTMG_TBPTT];
      double inv_f[FX6_LSTMG_TBPTT];
      double inv_o[FX6_LSTMG_TBPTT];
      double inv_g[FX6_LSTMG_TBPTT];

      double ig[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double fg[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double og[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double gg[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];

      double c_t[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];
      double h_t[FX6_LSTMG_TBPTT][FX6_AI_MLP_HIDDEN];

      double logits[FX6_LSTMG_TBPTT][FX6_LSTMG_CLASS_COUNT];
      double probs[FX6_LSTMG_TBPTT][FX6_LSTMG_CLASS_COUNT];
      double mu[FX6_LSTMG_TBPTT];
      double logv[FX6_LSTMG_TBPTT];
      double q25[FX6_LSTMG_TBPTT];
      double q75[FX6_LSTMG_TBPTT];

      double h_cur[FX6_AI_MLP_HIDDEN];
      double c_cur[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         h_cur[h] = m_batch_h0[h];
         c_cur[h] = m_batch_c0[h];
      }

      // Forward unroll with reset masks.
      for(int t=0; t<len; t++)
      {
         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               h_cur[h] = 0.0;
               c_cur[h] = 0.0;
            }
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            h_prev[t][h] = h_cur[h];
            c_prev[t][h] = c_cur[h];
         }

         double xloc[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];

         double ig_loc[FX6_AI_MLP_HIDDEN];
         double fg_loc[FX6_AI_MLP_HIDDEN];
         double og_loc[FX6_AI_MLP_HIDDEN];
         double gg_loc[FX6_AI_MLP_HIDDEN];
         double c_new[FX6_AI_MLP_HIDDEN];
         double h_new[FX6_AI_MLP_HIDDEN];

         double drop_loc[FX6_AI_MLP_HIDDEN];
         double zone_loc[FX6_AI_MLP_HIDDEN];
         double zi_loc[FX6_AI_MLP_HIDDEN];
         double zf_loc[FX6_AI_MLP_HIDDEN];
         double zo_loc[FX6_AI_MLP_HIDDEN];
         double zg_loc[FX6_AI_MLP_HIDDEN];
         double inv_i_loc, inv_f_loc, inv_o_loc, inv_g_loc;

         ForwardOne(xloc, h_cur, c_cur, true, false, t,
                    drop_loc, zone_loc,
                    zi_loc, zf_loc, zo_loc, zg_loc,
                    inv_i_loc, inv_f_loc, inv_o_loc, inv_g_loc,
                    ig_loc, fg_loc, og_loc, gg_loc, c_new, h_new);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            drop_mask[t][h] = drop_loc[h];
            zone_mask[t][h] = zone_loc[h];
            zi_hat[t][h] = zi_loc[h];
            zf_hat[t][h] = zf_loc[h];
            zo_hat[t][h] = zo_loc[h];
            zg_hat[t][h] = zg_loc[h];

            ig[t][h] = ig_loc[h];
            fg[t][h] = fg_loc[h];
            og[t][h] = og_loc[h];
            gg[t][h] = gg_loc[h];
            c_t[t][h] = c_new[h];
            h_t[t][h] = h_new[h];

            h_cur[h] = h_new[h];
            c_cur[h] = c_new[h];
         }

         inv_i[t] = inv_i_loc;
         inv_f[t] = inv_f_loc;
         inv_o[t] = inv_o_loc;
         inv_g[t] = inv_g_loc;

         double lo[FX6_LSTMG_CLASS_COUNT];
         double pr[FX6_LSTMG_CLASS_COUNT];
         double mu_loc, lv_loc, q25_loc, q75_loc;
         ComputeHeads(h_new, false, lo, pr, mu_loc, lv_loc, q25_loc, q75_loc);
         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
         {
            logits[t][c] = lo[c];
            probs[t][c] = pr[c];
         }
         mu[t] = mu_loc;
         logv[t] = lv_loc;
         q25[t] = q25_loc;
         q75[t] = q75_loc;
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = h_cur[h];
         m_c[h] = c_cur[h];
      }

      // Gradients.
      double g_wi[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wf[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wo[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wg[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];

      double g_ui[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_uf[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_uo[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_ug[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];

      double g_pi[FX6_AI_MLP_HIDDEN];
      double g_pf[FX6_AI_MLP_HIDDEN];
      double g_po[FX6_AI_MLP_HIDDEN];

      double g_bi[FX6_AI_MLP_HIDDEN];
      double g_bf[FX6_AI_MLP_HIDDEN];
      double g_bo[FX6_AI_MLP_HIDDEN];
      double g_bg[FX6_AI_MLP_HIDDEN];

      double g_ln_gi[FX6_AI_MLP_HIDDEN];
      double g_ln_gf[FX6_AI_MLP_HIDDEN];
      double g_ln_go[FX6_AI_MLP_HIDDEN];
      double g_ln_gg[FX6_AI_MLP_HIDDEN];

      double g_ln_bi[FX6_AI_MLP_HIDDEN];
      double g_ln_bf[FX6_AI_MLP_HIDDEN];
      double g_ln_bo[FX6_AI_MLP_HIDDEN];
      double g_ln_bg[FX6_AI_MLP_HIDDEN];

      double g_w_cls[FX6_LSTMG_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
      double g_b_cls[FX6_LSTMG_CLASS_COUNT];

      double g_w_mu[FX6_AI_MLP_HIDDEN];
      double g_w_logv[FX6_AI_MLP_HIDDEN];
      double g_w_q25[FX6_AI_MLP_HIDDEN];
      double g_w_q75[FX6_AI_MLP_HIDDEN];
      double g_b_mu = 0.0;
      double g_b_logv = 0.0;
      double g_b_q25 = 0.0;
      double g_b_q75 = 0.0;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         g_pi[h] = 0.0; g_pf[h] = 0.0; g_po[h] = 0.0;
         g_bi[h] = 0.0; g_bf[h] = 0.0; g_bo[h] = 0.0; g_bg[h] = 0.0;

         g_ln_gi[h] = 0.0; g_ln_gf[h] = 0.0; g_ln_go[h] = 0.0; g_ln_gg[h] = 0.0;
         g_ln_bi[h] = 0.0; g_ln_bf[h] = 0.0; g_ln_bo[h] = 0.0; g_ln_bg[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            g_w_cls[c][h] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            g_wi[h][i] = 0.0;
            g_wf[h][i] = 0.0;
            g_wo[h][i] = 0.0;
            g_wg[h][i] = 0.0;
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g_ui[h][j] = 0.0;
            g_uf[h][j] = 0.0;
            g_uo[h][j] = 0.0;
            g_ug[h][j] = 0.0;
         }
      }
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++) g_b_cls[c] = 0.0;

      double dh_next[FX6_AI_MLP_HIDDEN];
      double dc_next[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         dh_next[h] = 0.0;
         dc_next[h] = 0.0;
      }

      // Backward through sequence (TBPTT).
      for(int t=len - 1; t>=0; t--)
      {
         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               dh_next[h] = 0.0;
               dc_next[h] = 0.0;
            }
         }

         int cls = m_batch_cls[t];
         if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
            cls = (int)FX6_LABEL_SKIP;

         double sw = FX6_Clamp(m_batch_w[t], 0.20, 6.00);

         double err_cls[FX6_LSTMG_CLASS_COUNT];
         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
         {
            double target = (c == cls ? 1.0 : 0.0);
            err_cls[c] = FX6_ClipSym((target - probs[t][c]) * sw, 3.0);
            g_b_cls[c] += err_cls[c];
         }

         double target_move = MathAbs(m_batch_move[t]);
         double sigma = MathExp(0.5 * FX6_Clamp(logv[t], -4.0, 4.0));
         sigma = FX6_Clamp(sigma, 0.05, 30.0);
         double sig2 = sigma * sigma;

         double err_mu = FX6_ClipSym((target_move - mu[t]) / MathMax(1.0, target_move), 4.0);
         double err_lv = FX6_ClipSym(((target_move - mu[t]) * (target_move - mu[t]) / (sig2 + 1e-6)) - 1.0, 4.0);
         double err_q25 = (target_move >= q25[t] ? 0.25 : -0.75);
         double err_q75 = (target_move >= q75[t] ? 0.75 : -0.25);

         double move_scale = FX6_Clamp(0.65 * sw, 0.10, 8.00);

         g_b_mu += move_scale * err_mu;
         g_b_logv += move_scale * err_lv;
         g_b_q25 += move_scale * err_q25;
         g_b_q75 += move_scale * err_q75;

         double dh_total[FX6_AI_MLP_HIDDEN];
         double dzo_ln[FX6_AI_MLP_HIDDEN];
         double dc_mid[FX6_AI_MLP_HIDDEN];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
               g_w_cls[c][h] += err_cls[c] * h_t[t][h];

            g_w_mu[h] += move_scale * err_mu * h_t[t][h];
            g_w_logv[h] += move_scale * err_lv * h_t[t][h];
            g_w_q25[h] += move_scale * err_q25 * h_t[t][h];
            g_w_q75[h] += move_scale * err_q75 * h_t[t][h];

            double dh = dh_next[h];
            for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
               dh += err_cls[c] * m_w_cls[c][h];

            dh += move_scale * (err_mu * m_w_mu[h] +
                                0.30 * err_lv * m_w_logv[h] +
                                0.18 * err_q25 * m_w_q25[h] +
                                0.18 * err_q75 * m_w_q75[h]);

            dh_total[h] = dh;

            double tanh_c = FX6_Tanh(c_t[t][h]);
            dzo_ln[h] = FX6_ClipSym((dh * tanh_c) * og[t][h] * (1.0 - og[t][h]), 4.0);
            dc_mid[h] = dc_next[h] + dh * og[t][h] * (1.0 - tanh_c * tanh_c);
         }

         double dzo_raw[FX6_AI_MLP_HIDDEN];
         double zo_row[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) zo_row[h] = zo_hat[t][h];
         LayerNormBackward(dzo_ln, zo_row, inv_o[t], m_ln_go, g_ln_go, g_ln_bo, dzo_raw);

         double di_ln[FX6_AI_MLP_HIDDEN];
         double df_ln[FX6_AI_MLP_HIDDEN];
         double dg_ln[FX6_AI_MLP_HIDDEN];
         double dc_vec[FX6_AI_MLP_HIDDEN];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            if(zone_mask[t][h] > 0.5)
            {
               di_ln[h] = 0.0;
               df_ln[h] = 0.0;
               dg_ln[h] = 0.0;
               dc_vec[h] = dc_next[h];
               dzo_raw[h] = 0.0;
               continue;
            }

            double dc = dc_mid[h] + dzo_raw[h] * m_po[h];
            dc_vec[h] = dc;

            di_ln[h] = FX6_ClipSym((dc * gg[t][h]) * ig[t][h] * (1.0 - ig[t][h]), 4.0);
            df_ln[h] = FX6_ClipSym((dc * c_prev[t][h]) * fg[t][h] * (1.0 - fg[t][h]), 4.0);
            dg_ln[h] = FX6_ClipSym((dc * ig[t][h]) * (1.0 - gg[t][h] * gg[t][h]), 4.0);
         }

         double dzi_raw[FX6_AI_MLP_HIDDEN];
         double dzf_raw[FX6_AI_MLP_HIDDEN];
         double dzg_raw[FX6_AI_MLP_HIDDEN];
         double zi_row[FX6_AI_MLP_HIDDEN];
         double zf_row[FX6_AI_MLP_HIDDEN];
         double zg_row[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            zi_row[h] = zi_hat[t][h];
            zf_row[h] = zf_hat[t][h];
            zg_row[h] = zg_hat[t][h];
         }
         LayerNormBackward(di_ln, zi_row, inv_i[t], m_ln_gi, g_ln_gi, g_ln_bi, dzi_raw);
         LayerNormBackward(df_ln, zf_row, inv_f[t], m_ln_gf, g_ln_gf, g_ln_bf, dzf_raw);
         LayerNormBackward(dg_ln, zg_row, inv_g[t], m_ln_gg, g_ln_gg, g_ln_bg, dzg_raw);

         double dh_prev[FX6_AI_MLP_HIDDEN];
         double dc_prev[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            dh_prev[h] = 0.0;
            dc_prev[h] = 0.0;
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            if(zone_mask[t][h] > 0.5)
            {
               dh_prev[h] += dh_total[h];
               dc_prev[h] = dc_next[h];
               continue;
            }

            double dzi = FX6_ClipSym(dzi_raw[h], 4.0);
            double dzf = FX6_ClipSym(dzf_raw[h], 4.0);
            double dzo = FX6_ClipSym(dzo_raw[h], 4.0);
            double dzg = FX6_ClipSym(dzg_raw[h], 4.0);

            g_pi[h] += dzi * c_prev[t][h];
            g_pf[h] += dzf * c_prev[t][h];
            g_po[h] += dzo * c_t[t][h];

            g_bi[h] += dzi;
            g_bf[h] += dzf;
            g_bo[h] += dzo;
            g_bg[h] += dzg;

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               double xv = m_batch_x[t][i];
               g_wi[h][i] += dzi * xv;
               g_wf[h][i] += dzf * xv;
               g_wo[h][i] += dzo * xv;
               g_wg[h][i] += dzg * xv;
            }

            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            {
               double hpv = h_prev[t][j] * drop_mask[t][j];
               g_ui[h][j] += dzi * hpv;
               g_uf[h][j] += dzf * hpv;
               g_uo[h][j] += dzo * hpv;
               g_ug[h][j] += dzg * hpv;

               double dh_in = m_ui[h][j] * dzi +
                              m_uf[h][j] * dzf +
                              m_uo[h][j] * dzo +
                              m_ug[h][j] * dzg;
               dh_prev[j] += dh_in * drop_mask[t][j];
            }

            dc_prev[h] = dc_vec[h] * fg[t][h] + dzi * m_pi[h] + dzf * m_pf[h];
         }

         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               dh_next[h] = 0.0;
               dc_next[h] = 0.0;
            }
         }
         else
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               dh_next[h] = dh_prev[h];
               dc_next[h] = dc_prev[h];
            }
         }
      }

      // Global grad norm clipping.
      double gnorm2 = g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++) gnorm2 += g_b_cls[c] * g_b_cls[c];

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         gnorm2 += g_pi[h] * g_pi[h] + g_pf[h] * g_pf[h] + g_po[h] * g_po[h];
         gnorm2 += g_bi[h] * g_bi[h] + g_bf[h] * g_bf[h] + g_bo[h] * g_bo[h] + g_bg[h] * g_bg[h];

         gnorm2 += g_ln_gi[h] * g_ln_gi[h] + g_ln_gf[h] * g_ln_gf[h] + g_ln_go[h] * g_ln_go[h] + g_ln_gg[h] * g_ln_gg[h];
         gnorm2 += g_ln_bi[h] * g_ln_bi[h] + g_ln_bf[h] * g_ln_bf[h] + g_ln_bo[h] * g_ln_bo[h] + g_ln_bg[h] * g_ln_bg[h];

         gnorm2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++) gnorm2 += g_w_cls[c][h] * g_w_cls[c][h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            gnorm2 += g_wi[h][i] * g_wi[h][i] + g_wf[h][i] * g_wf[h][i] +
                      g_wo[h][i] * g_wo[h][i] + g_wg[h][i] * g_wg[h][i];
         }
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            gnorm2 += g_ui[h][j] * g_ui[h][j] + g_uf[h][j] * g_uf[h][j] +
                      g_uo[h][j] * g_uo[h][j] + g_ug[h][j] * g_ug[h][j];
         }
      }

      double gnorm = MathSqrt(gnorm2);
      double gscale = (gnorm > 3.0 ? (3.0 / MathMax(gnorm, 1e-9)) : 1.0);

      double gmag_gate = 0.0;
      double gmag_peep = 0.0;
      double gmag_ln = 0.0;
      double gmag_cls = 0.0;
      double gmag_move = 0.0;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         gmag_peep += MathAbs(g_pi[h]) + MathAbs(g_pf[h]) + MathAbs(g_po[h]);
         gmag_ln += MathAbs(g_ln_gi[h]) + MathAbs(g_ln_gf[h]) + MathAbs(g_ln_go[h]) + MathAbs(g_ln_gg[h]) +
                    MathAbs(g_ln_bi[h]) + MathAbs(g_ln_bf[h]) + MathAbs(g_ln_bo[h]) + MathAbs(g_ln_bg[h]);

         gmag_move += MathAbs(g_w_mu[h]) + MathAbs(g_w_logv[h]) + MathAbs(g_w_q25[h]) + MathAbs(g_w_q75[h]);
         gmag_gate += MathAbs(g_bi[h]) + MathAbs(g_bf[h]) + MathAbs(g_bo[h]) + MathAbs(g_bg[h]);

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            gmag_cls += MathAbs(g_w_cls[c][h]);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            gmag_gate += MathAbs(g_wi[h][i]) + MathAbs(g_wf[h][i]) + MathAbs(g_wo[h][i]) + MathAbs(g_wg[h][i]);
         }
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            gmag_gate += MathAbs(g_ui[h][j]) + MathAbs(g_uf[h][j]) + MathAbs(g_uo[h][j]) + MathAbs(g_ug[h][j]);
         }
      }

      double lr_gate = AdamGroupLR(0, gmag_gate, base_lr);
      double lr_peep = AdamGroupLR(1, gmag_peep, base_lr * 0.85);
      double lr_ln = AdamGroupLR(2, gmag_ln, base_lr * 0.70);
      double lr_cls = AdamGroupLR(3, gmag_cls, base_lr * 0.80);
      double lr_move = AdamGroupLR(4, gmag_move, base_lr * 0.72);

      double wd_gate = FX6_Clamp(l2 * 0.18, 0.0, 0.20);
      double wd_peep = FX6_Clamp(l2 * 0.10, 0.0, 0.12);
      double wd_ln = FX6_Clamp(l2 * 0.04, 0.0, 0.08);
      double wd_cls = FX6_Clamp(l2 * 0.08, 0.0, 0.12);
      double wd_move = FX6_Clamp(l2 * 0.06, 0.0, 0.10);

      // Parameter update.
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_bi[h] += lr_gate * gscale * g_bi[h];
         m_bf[h] += lr_gate * gscale * g_bf[h];
         m_bo[h] += lr_gate * gscale * g_bo[h];
         m_bg[h] += lr_gate * gscale * g_bg[h];

         m_pi[h] = FX6_ClipSym(m_pi[h] * (1.0 - lr_peep * wd_peep) + lr_peep * gscale * g_pi[h], 4.0);
         m_pf[h] = FX6_ClipSym(m_pf[h] * (1.0 - lr_peep * wd_peep) + lr_peep * gscale * g_pf[h], 4.0);
         m_po[h] = FX6_ClipSym(m_po[h] * (1.0 - lr_peep * wd_peep) + lr_peep * gscale * g_po[h], 4.0);

         m_ln_gi[h] = FX6_Clamp(m_ln_gi[h] * (1.0 - lr_ln * wd_ln) + lr_ln * gscale * g_ln_gi[h], 0.20, 3.00);
         m_ln_gf[h] = FX6_Clamp(m_ln_gf[h] * (1.0 - lr_ln * wd_ln) + lr_ln * gscale * g_ln_gf[h], 0.20, 3.00);
         m_ln_go[h] = FX6_Clamp(m_ln_go[h] * (1.0 - lr_ln * wd_ln) + lr_ln * gscale * g_ln_go[h], 0.20, 3.00);
         m_ln_gg[h] = FX6_Clamp(m_ln_gg[h] * (1.0 - lr_ln * wd_ln) + lr_ln * gscale * g_ln_gg[h], 0.20, 3.00);

         m_ln_bi[h] = FX6_ClipSym(m_ln_bi[h] + lr_ln * gscale * g_ln_bi[h], 3.0);
         m_ln_bf[h] = FX6_ClipSym(m_ln_bf[h] + lr_ln * gscale * g_ln_bf[h], 3.0);
         m_ln_bo[h] = FX6_ClipSym(m_ln_bo[h] + lr_ln * gscale * g_ln_bo[h], 3.0);
         m_ln_bg[h] = FX6_ClipSym(m_ln_bg[h] + lr_ln * gscale * g_ln_bg[h], 3.0);

         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_w_cls[c][h] = m_w_cls[c][h] * (1.0 - lr_cls * wd_cls) + lr_cls * gscale * g_w_cls[c][h];

         m_w_mu[h] = m_w_mu[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_mu[h];
         m_w_logv[h] = m_w_logv[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_logv[h];
         m_w_q25[h] = m_w_q25[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q25[h];
         m_w_q75[h] = m_w_q75[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q75[h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_wi[h][i] = m_wi[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wi[h][i];
            m_wf[h][i] = m_wf[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wf[h][i];
            m_wo[h][i] = m_wo[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wo[h][i];
            m_wg[h][i] = m_wg[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wg[h][i];
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_ui[h][j] = m_ui[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_ui[h][j];
            m_uf[h][j] = m_uf[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_uf[h][j];
            m_uo[h][j] = m_uo[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_uo[h][j];
            m_ug[h][j] = m_ug[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_ug[h][j];
         }
      }

      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         m_b_cls[c] = FX6_ClipSym(m_b_cls[c] + lr_cls * gscale * g_b_cls[c], 6.0);
      }

      m_b_mu += lr_move * gscale * g_b_mu;
      m_b_logv = FX6_ClipSym(m_b_logv + lr_move * gscale * g_b_logv, 6.0);
      m_b_q25 += lr_move * gscale * g_b_q25;
      m_b_q75 += lr_move * gscale * g_b_q75;

      UpdateEMAFromParams(0.995);

      // Calibration and auxiliary updates.
      for(int t=0; t<len; t++)
      {
         int cls = m_batch_cls[t];
         if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
            cls = (int)FX6_LABEL_SKIP;

         double sw = FX6_Clamp(m_batch_w[t], 0.25, 6.00);
         double p_raw[FX6_LSTMG_CLASS_COUNT];
         p_raw[0] = probs[t][0];
         p_raw[1] = probs[t][1];
         p_raw[2] = probs[t][2];

         UpdateCalibrator3(p_raw, cls, sw, base_lr);

         double edge_tgt = MathMax(0.0, MathAbs(m_batch_move[t]) - MathMax(0.0, m_batch_cost[t]));
         double ev_raw = ExpectedMoveFromHeads(mu[t], logv[t], q25[t], q75[t], p_raw[(int)FX6_LABEL_SKIP]);
         UpdateEVCalibration(ev_raw, edge_tgt, sw, base_lr);

         double den = p_raw[(int)FX6_LABEL_BUY] + p_raw[(int)FX6_LABEL_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir = p_raw[(int)FX6_LABEL_BUY] / den;
         if(cls == (int)FX6_LABEL_BUY) UpdateCalibration(p_dir, 1, sw);
         else if(cls == (int)FX6_LABEL_SELL) UpdateCalibration(p_dir, 0, sw);

         FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, m_batch_move[t], 0.05);

         double xloc[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];
         UpdateMoveHead(xloc, m_batch_move[t], hp, sw);
      }

      ResetBatch();
   }

public:
   CFX6AILSTMG(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_LSTMG; }
   virtual string AIName(void) const { return "lstmg"; }
   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_seen_updates = 0;
      m_adam_t = 0;

      m_last_update_time = 0;
      m_last_session_bucket = -1;
      m_vol_ready = false;
      m_vol_ema = 0.0;

      m_ev_ready = false;
      m_ev_steps = 0;
      m_ev_a = 1.0;
      m_ev_b = 0.0;

      m_selftest_done = false;
      m_selftest_ok = false;

      for(int g=0; g<FX6_LSTMG_OPT_GROUPS; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      for(int s=0; s<4; s++)
      {
         m_sess_total[s] = 0.0;
         for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
            m_cls_count[s][c] = 0.0;
      }

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FX6_LSTMG_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FX6_LSTMG_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      m_ema_ready = false;
      m_ema_steps = 0;

      ResetState();
      ResetBatch();
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights();

      if(!m_selftest_done)
         RunSelfTests();
   }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      if(ArraySize(class_probs) < FX6_LSTMG_CLASS_COUNT)
         ArrayResize(class_probs, FX6_LSTMG_CLASS_COUNT);

      bool use_ema = UseEMAInference();

      double h_state[FX6_AI_MLP_HIDDEN];
      double c_state[FX6_AI_MLP_HIDDEN];
      BuildStateWithPending(h_state, c_state, use_ema);

      double drop_mask[FX6_AI_MLP_HIDDEN];
      double zone_mask[FX6_AI_MLP_HIDDEN];
      double zi_hat[FX6_AI_MLP_HIDDEN];
      double zf_hat[FX6_AI_MLP_HIDDEN];
      double zo_hat[FX6_AI_MLP_HIDDEN];
      double zg_hat[FX6_AI_MLP_HIDDEN];
      double inv_i, inv_f, inv_o, inv_g;
      double ig[FX6_AI_MLP_HIDDEN];
      double fg[FX6_AI_MLP_HIDDEN];
      double og[FX6_AI_MLP_HIDDEN];
      double gg[FX6_AI_MLP_HIDDEN];
      double c_new[FX6_AI_MLP_HIDDEN];
      double h_new[FX6_AI_MLP_HIDDEN];

      ForwardOne(x, h_state, c_state, false, use_ema, 0,
                 drop_mask, zone_mask,
                 zi_hat, zf_hat, zo_hat, zg_hat,
                 inv_i, inv_f, inv_o, inv_g,
                 ig, fg, og, gg, c_new, h_new);

      double logits[FX6_LSTMG_CLASS_COUNT];
      double probs_raw[FX6_LSTMG_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_new, use_ema, logits, probs_raw, mu, logv, q25, q75);

      Calibrate3(probs_raw, class_probs);

      double ev_raw = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FX6_LABEL_SKIP]);
      double ev_cal = CalibrateEV(ev_raw);
      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);

      if(ev_cal > 0.0 && base_ev > 0.0) expected_move_points = 0.70 * ev_cal + 0.30 * base_ev;
      else if(ev_cal > 0.0) expected_move_points = ev_cal;
      else expected_move_points = base_ev;

      if(expected_move_points <= 0.0)
         expected_move_points = MathMax(ResolveMinMovePoints(), 0.10);

      return true;
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      double pseudo_move = 0.0;
      if(y == (int)FX6_LABEL_BUY) pseudo_move = 1.0;
      else if(y == (int)FX6_LABEL_SELL) pseudo_move = -1.0;
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen_updates++;

      double cost_points = ResolveCostPoints(x);
      if(cost_points < 0.0) cost_points = 0.0;

      double min_move_points = ResolveMinMovePoints();
      if(min_move_points <= 0.0)
         min_move_points = MathMax(0.10, cost_points + 0.10);

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      int sess = 0;
      int boundary = CheckRegimeBoundary(x, sess);
      if(boundary > 0 && m_batch_size > 0)
         TrainBatch(h);

      if(boundary == 2)
      {
         ResetState();
         ResetBatch();
      }
      else if(boundary == 1)
      {
         DecayState(0.65);
         ResetBatch();
      }

      double reset_flag = (boundary > 0 ? 1.0 : 0.0);

      int cls = DeriveClassLabel(y, x, move_points, cost_points, min_move_points);
      double w = MoveSampleWeight(x, move_points) * ClassBalanceWeight(sess, cls);
      if(cls == (int)FX6_LABEL_SKIP) w *= 0.80;
      w = FX6_Clamp(w, 0.10, 6.00);

      AppendBatch(cls, x, move_points, cost_points, w, reset_flag);
      UpdateClassStats(sess, cls);

      if(m_batch_size >= FX6_LSTMG_TBPTT ||
         (m_batch_size >= 8 && (m_seen_updates % 4) == 0))
      {
         TrainBatch(h);
      }
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = 0.0;
      if(!PredictNativeClassProbs(x, hp, probs, ev))
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

#endif // __FX6_AI_LSTMG_MQH__
