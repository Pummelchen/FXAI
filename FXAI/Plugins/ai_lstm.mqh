#ifndef __FX6_AI_LSTM_MQH__
#define __FX6_AI_LSTM_MQH__

#include "..\plugin_base.mqh"

#define FX6_LSTM_TBPTT 16
#define FX6_LSTM_CLASS_COUNT 3
#define FX6_LSTM_CAL_BINS 12
#define FX6_LSTM_DROP_RATE 0.08
#define FX6_LSTM_ZONEOUT 0.05

class CFX6AILSTM : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   int    m_seen_updates;
   int    m_adam_t;

   // Regime/session-aware state control.
   datetime m_last_update_time;
   int      m_last_session_bucket;
   bool     m_vol_ready;
   double   m_vol_ema;

   double m_h[FX6_AI_MLP_HIDDEN];
   double m_c[FX6_AI_MLP_HIDDEN];

   double m_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];

   double m_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];

   double m_bi[FX6_AI_MLP_HIDDEN];
   double m_bf[FX6_AI_MLP_HIDDEN];
   double m_bo[FX6_AI_MLP_HIDDEN];
   double m_bg[FX6_AI_MLP_HIDDEN];

   // Native 3-class head.
   double m_w_cls[FX6_LSTM_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_LSTM_CLASS_COUNT];

   // Distributional move head.
   double m_w_mu[FX6_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN];
   double m_b_q75;

   // Sequence batching buffer for truncated BPTT.
   int    m_batch_size;
   double m_batch_h0[FX6_AI_MLP_HIDDEN];
   double m_batch_c0[FX6_AI_MLP_HIDDEN];
   double m_batch_x[FX6_LSTM_TBPTT][FX6_AI_WEIGHTS];
   int    m_batch_y[FX6_LSTM_TBPTT];
   double m_batch_move[FX6_LSTM_TBPTT];
   double m_batch_w[FX6_LSTM_TBPTT];

   // Grouped AdamW moments.
   double m_opt_m[6];
   double m_opt_v[6];

   // Native multiclass calibration.
   double m_cal3_temp;
   double m_cal3_bias[FX6_LSTM_CLASS_COUNT];
   double m_cal3_iso_pos[FX6_LSTM_CLASS_COUNT][FX6_LSTM_CAL_BINS];
   double m_cal3_iso_cnt[FX6_LSTM_CLASS_COUNT][FX6_LSTM_CAL_BINS];
   int    m_cal3_steps;

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
      double keep = 1.0 - FX6_LSTM_DROP_RATE;
      if(keep <= 0.05) keep = 0.05;
      double u = Hash01(m_step, salt, idx);
      if(u < FX6_LSTM_DROP_RATE) return 0.0;
      return 1.0 / keep;
   }

   bool ZoneKeep(const int salt, const int idx) const
   {
      double u = Hash01(m_seen_updates + 17, salt, idx);
      return (u < FX6_LSTM_ZONEOUT);
   }

   double AdamGroupLR(const int group_idx,
                      const double grad_mag,
                      const double base_lr)
   {
      int g = group_idx;
      if(g < 0) g = 0;
      if(g > 5) g = 5;

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
      double logits[FX6_LSTM_CLASS_COUNT];
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FX6_LSTM_CLASS_COUNT];
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FX6_LSTM_CAL_BINS; b++)
            total += m_cal3_iso_cnt[c][b];

         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FX6_LSTM_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FX6_LSTM_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FX6_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_LSTM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_LSTM_CAL_BINS) bi = FX6_LSTM_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
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
      double logits[FX6_LSTM_CLASS_COUNT];
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         double pr = FX6_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }

      double p_cal[FX6_LSTM_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FX6_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FX6_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FX6_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FX6_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FX6_LSTM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FX6_LSTM_CAL_BINS) bi = FX6_LSTM_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_temp = FX6_Clamp(m_cal3_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void MaybeResetForRegime(const double &x[])
   {
      datetime now = ResolveContextTime();
      if(now <= 0) now = TimeCurrent();

      int sess = SessionBucket(now);
      bool hard_reset = false;

      if(m_last_update_time > 0)
      {
         int gap_sec = (int)(now - m_last_update_time);
         if(gap_sec > 90 * 60)
            hard_reset = true;
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
         if(m_vol_ema > 1e-6 && shock > 3.5 * m_vol_ema)
            hard_reset = true;
      }

      if(m_last_session_bucket >= 0 && sess != m_last_session_bucket)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_h[h] *= 0.70;
            m_c[h] *= 0.70;
         }
         ResetBatch();
      }

      if(hard_reset)
      {
         ResetState();
         ResetBatch();
      }

      m_last_update_time = now;
      m_last_session_bucket = sess;
   }

   void ResetState(void)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = 0.0;
         m_c[h] = 0.0;
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

   void InitWeights(void)
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double seed = (double)((h + 1) * (i + 3));
            m_wi_x[h][i] = 0.03 * MathSin(seed * 1.03);
            m_wf_x[h][i] = 0.03 * MathCos(seed * 1.07);
            m_wo_x[h][i] = 0.03 * MathSin(seed * 1.11);
            m_wg_x[h][i] = 0.03 * MathCos(seed * 1.13);
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            double seed2 = (double)((h + 2) * (j + 5));
            m_wi_h[h][j] = 0.02 * MathSin(seed2 * 0.91);
            m_wf_h[h][j] = 0.02 * MathCos(seed2 * 0.93);
            m_wo_h[h][j] = 0.02 * MathSin(seed2 * 0.97);
            m_wg_h[h][j] = 0.02 * MathCos(seed2 * 0.99);
         }

         m_bi[h] = 0.0;
         m_bf[h] = 0.3;
         m_bo[h] = 0.0;
         m_bg[h] = 0.0;

         m_w_mu[h] = 0.03 * MathSin((double)(h + 1) * 1.29);
         m_w_logv[h] = 0.03 * MathCos((double)(h + 2) * 1.23);
         m_w_q25[h] = 0.02 * MathSin((double)(h + 3) * 1.17);
         m_w_q75[h] = 0.02 * MathCos((double)(h + 4) * 1.11);

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
            m_w_cls[c][h] = 0.04 * MathSin((double)((c + 2) * (h + 1)) * 1.07);
      }

      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;
      m_b_cls[(int)FX6_LABEL_SKIP] = 0.15;

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FX6_LSTM_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int g=0; g<6; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_initialized = true;
   }

   double DotX(const double &w[][FX6_AI_WEIGHTS], const int row, const double &x[]) const
   {
      double s = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++) s += w[row][i] * x[i];
      return s;
   }

   double DotH(const double &w[][FX6_AI_MLP_HIDDEN], const int row, const double &hprev[]) const
   {
      double s = 0.0;
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++) s += w[row][i] * hprev[i];
      return s;
   }

   void ForwardOne(const double &x[],
                   const double &h_prev[],
                   const double &c_prev[],
                   const bool training,
                   const int salt,
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
         if(training)
            h_in[j] = h_prev[j] * DropScale(salt, j);
         else
            h_in[j] = h_prev[j];
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         ig[h] = FX6_Sigmoid(DotX(m_wi_x, h, x) + DotH(m_wi_h, h, h_in) + m_bi[h]);
         fg[h] = FX6_Sigmoid(DotX(m_wf_x, h, x) + DotH(m_wf_h, h, h_in) + m_bf[h]);
         og[h] = FX6_Sigmoid(DotX(m_wo_x, h, x) + DotH(m_wo_h, h, h_in) + m_bo[h]);
         gg[h] = FX6_Tanh(DotX(m_wg_x, h, x) + DotH(m_wg_h, h, h_in) + m_bg[h]);

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         c_new[h] = FX6_ClipSym(c_new[h], 10.0);
         h_new[h] = og[h] * FX6_Tanh(c_new[h]);

         if(training && ZoneKeep(salt, h))
         {
            c_new[h] = c_prev[h];
            h_new[h] = h_prev[h];
         }
      }
   }

   void ComputeHeads(const double &hvec[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75) const
   {
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            z += m_w_cls[c][h] * hvec[h];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         mu += m_w_mu[h] * hvec[h];
         logv += m_w_logv[h] * hvec[h];
         q25 += m_w_q25[h] * hvec[h];
         q75 += m_w_q75[h] * hvec[h];
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
      double ev = (0.68 * MathAbs(mu) + 0.22 * sigma + 0.10 * iqr) * FX6_Clamp(1.0 - p_skip, 0.0, 1.0);
      return ev;
   }

   void AppendBatch(const int y_cls,
                    const double &x[],
                    const double move_points,
                    const double sample_w)
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_batch_h0[h] = m_h[h];
            m_batch_c0[h] = m_c[h];
         }
      }

      if(m_batch_size < FX6_LSTM_TBPTT)
      {
         int p = m_batch_size;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[p][i] = x[i];
         m_batch_y[p] = y_cls;
         m_batch_move[p] = move_points;
         m_batch_w[p] = sample_w;
         m_batch_size++;
         return;
      }

      for(int t=1; t<FX6_LSTM_TBPTT; t++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[t - 1][i] = m_batch_x[t][i];
         m_batch_y[t - 1] = m_batch_y[t];
         m_batch_move[t - 1] = m_batch_move[t];
         m_batch_w[t - 1] = m_batch_w[t];
      }

      for(int i=0; i<FX6_AI_WEIGHTS; i++) m_batch_x[FX6_LSTM_TBPTT - 1][i] = x[i];
      m_batch_y[FX6_LSTM_TBPTT - 1] = y_cls;
      m_batch_move[FX6_LSTM_TBPTT - 1] = move_points;
      m_batch_w[FX6_LSTM_TBPTT - 1] = sample_w;
      m_batch_size = FX6_LSTM_TBPTT;
   }

   void BuildStateWithPending(double &h_out[], double &c_out[]) const
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
         double ig[FX6_AI_MLP_HIDDEN];
         double fg[FX6_AI_MLP_HIDDEN];
         double og[FX6_AI_MLP_HIDDEN];
         double gg[FX6_AI_MLP_HIDDEN];
         double c_new[FX6_AI_MLP_HIDDEN];
         double h_new[FX6_AI_MLP_HIDDEN];

         double xloc[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            xloc[i] = m_batch_x[t][i];

         ForwardOne(xloc, h_cur, c_cur, false, t, ig, fg, og, gg, c_new, h_new);
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

   void TrainBatch(const FX6AIHyperParams &hp)
   {
      int len = m_batch_size;
      if(len <= 0) return;

      m_step += len;
      m_adam_t++;

      double base_lr0 = FX6_Clamp(hp.lr, 0.0001, 0.5000);
      double warm = FX6_Clamp((double)m_step / 256.0, 0.10, 1.00);
      double decay = 1.0 / MathSqrt(1.0 + 0.0010 * MathMax(0.0, (double)m_step - 256.0));
      double base_lr = FX6_Clamp(base_lr0 * warm * decay, 0.00002, 0.05000);
      double l2 = FX6_Clamp(hp.l2, 0.0000, 0.2000);

      double h_prev[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double c_prev[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double ig[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double fg[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double og[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double gg[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double c_t[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];
      double h_t[FX6_LSTM_TBPTT][FX6_AI_MLP_HIDDEN];

      double logits[FX6_LSTM_TBPTT][FX6_LSTM_CLASS_COUNT];
      double probs[FX6_LSTM_TBPTT][FX6_LSTM_CLASS_COUNT];
      double mu[FX6_LSTM_TBPTT];
      double logv[FX6_LSTM_TBPTT];
      double q25[FX6_LSTM_TBPTT];
      double q75[FX6_LSTM_TBPTT];

      double h_cur[FX6_AI_MLP_HIDDEN];
      double c_cur[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         h_cur[h] = m_batch_h0[h];
         c_cur[h] = m_batch_c0[h];
      }

      for(int t=0; t<len; t++)
      {
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
         ForwardOne(xloc, h_cur, c_cur, true, t, ig_loc, fg_loc, og_loc, gg_loc, c_new, h_new);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            ig[t][h] = ig_loc[h];
            fg[t][h] = fg_loc[h];
            og[t][h] = og_loc[h];
            gg[t][h] = gg_loc[h];
            c_t[t][h] = c_new[h];
            h_t[t][h] = h_new[h];
            h_cur[h] = h_new[h];
            c_cur[h] = c_new[h];
         }

         double logits_loc[FX6_LSTM_CLASS_COUNT];
         double probs_loc[FX6_LSTM_CLASS_COUNT];
         double mu_loc, logv_loc, q25_loc, q75_loc;
         ComputeHeads(h_new, logits_loc, probs_loc, mu_loc, logv_loc, q25_loc, q75_loc);

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
         {
            logits[t][c] = logits_loc[c];
            probs[t][c] = probs_loc[c];
         }
         mu[t] = mu_loc;
         logv[t] = logv_loc;
         q25[t] = q25_loc;
         q75[t] = q75_loc;
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = h_cur[h];
         m_c[h] = c_cur[h];
      }

      double g_wi_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wf_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wo_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wg_x[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_wi_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_wf_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_wo_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_wg_h[FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_bi[FX6_AI_MLP_HIDDEN];
      double g_bf[FX6_AI_MLP_HIDDEN];
      double g_bo[FX6_AI_MLP_HIDDEN];
      double g_bg[FX6_AI_MLP_HIDDEN];

      double g_w_cls[FX6_LSTM_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
      double g_b_cls[FX6_LSTM_CLASS_COUNT];

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
         g_bi[h] = 0.0;
         g_bf[h] = 0.0;
         g_bo[h] = 0.0;
         g_bg[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
            g_w_cls[c][h] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            g_wi_x[h][i] = 0.0;
            g_wf_x[h][i] = 0.0;
            g_wo_x[h][i] = 0.0;
            g_wg_x[h][i] = 0.0;
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            g_wi_h[h][j] = 0.0;
            g_wf_h[h][j] = 0.0;
            g_wo_h[h][j] = 0.0;
            g_wg_h[h][j] = 0.0;
         }
      }
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
         g_b_cls[c] = 0.0;

      double dh_next[FX6_AI_MLP_HIDDEN];
      double dc_next[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         dh_next[h] = 0.0;
         dc_next[h] = 0.0;
      }

      for(int t=len - 1; t>=0; t--)
      {
         int cls = m_batch_y[t];
         if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
            cls = (int)FX6_LABEL_SKIP;

         double sw = FX6_Clamp(m_batch_w[t], 0.20, 6.00);

         double err_cls[FX6_LSTM_CLASS_COUNT];
         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
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

         double dh_prev[FX6_AI_MLP_HIDDEN];
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++) dh_prev[j] = 0.0;

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
               g_w_cls[c][h] += err_cls[c] * h_t[t][h];

            g_w_mu[h] += move_scale * err_mu * h_t[t][h];
            g_w_logv[h] += move_scale * err_lv * h_t[t][h];
            g_w_q25[h] += move_scale * err_q25 * h_t[t][h];
            g_w_q75[h] += move_scale * err_q75 * h_t[t][h];

            double tanh_c = FX6_Tanh(c_t[t][h]);
            double dh = dh_next[h];
            for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
               dh += err_cls[c] * m_w_cls[c][h];
            dh += move_scale * (err_mu * m_w_mu[h] +
                                0.30 * err_lv * m_w_logv[h] +
                                0.18 * err_q25 * m_w_q25[h] +
                                0.18 * err_q75 * m_w_q75[h]);

            double doo = (dh * tanh_c) * og[t][h] * (1.0 - og[t][h]);
            double dc = dc_next[h] + dh * og[t][h] * (1.0 - tanh_c * tanh_c);
            double di = (dc * gg[t][h]) * ig[t][h] * (1.0 - ig[t][h]);
            double df = (dc * c_prev[t][h]) * fg[t][h] * (1.0 - fg[t][h]);
            double dg = (dc * ig[t][h]) * (1.0 - gg[t][h] * gg[t][h]);
            double dc_prev = dc * fg[t][h];

            di = FX6_ClipSym(di, 4.0);
            df = FX6_ClipSym(df, 4.0);
            doo = FX6_ClipSym(doo, 4.0);
            dg = FX6_ClipSym(dg, 4.0);

            g_bi[h] += di;
            g_bf[h] += df;
            g_bo[h] += doo;
            g_bg[h] += dg;

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               double xv = m_batch_x[t][i];
               g_wi_x[h][i] += di * xv;
               g_wf_x[h][i] += df * xv;
               g_wo_x[h][i] += doo * xv;
               g_wg_x[h][i] += dg * xv;
            }

            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            {
               double hpv = h_prev[t][j];
               g_wi_h[h][j] += di * hpv;
               g_wf_h[h][j] += df * hpv;
               g_wo_h[h][j] += doo * hpv;
               g_wg_h[h][j] += dg * hpv;

               dh_prev[j] += m_wi_h[h][j] * di +
                             m_wf_h[h][j] * df +
                             m_wo_h[h][j] * doo +
                             m_wg_h[h][j] * dg;
            }

            dc_next[h] = dc_prev;
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            dh_next[j] = dh_prev[j];
      }

      double gnorm2 = g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
         gnorm2 += g_b_cls[c] * g_b_cls[c];

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         gnorm2 += g_bi[h] * g_bi[h] + g_bf[h] * g_bf[h] + g_bo[h] * g_bo[h] + g_bg[h] * g_bg[h];
         gnorm2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] +
                   g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
            gnorm2 += g_w_cls[c][h] * g_w_cls[c][h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            gnorm2 += g_wi_x[h][i] * g_wi_x[h][i] + g_wf_x[h][i] * g_wf_x[h][i] +
                      g_wo_x[h][i] * g_wo_x[h][i] + g_wg_x[h][i] * g_wg_x[h][i];
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            gnorm2 += g_wi_h[h][j] * g_wi_h[h][j] + g_wf_h[h][j] * g_wf_h[h][j] +
                      g_wo_h[h][j] * g_wo_h[h][j] + g_wg_h[h][j] * g_wg_h[h][j];
         }
      }

      double gnorm = MathSqrt(gnorm2);
      double gscale = (gnorm > 3.0 ? (3.0 / MathMax(gnorm, 1e-9)) : 1.0);

      double gmag_gate = 0.0;
      double gmag_cls = 0.0;
      double gmag_move = 0.0;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         gmag_gate += MathAbs(g_bi[h]) + MathAbs(g_bf[h]) + MathAbs(g_bo[h]) + MathAbs(g_bg[h]);
         gmag_move += MathAbs(g_w_mu[h]) + MathAbs(g_w_logv[h]) + MathAbs(g_w_q25[h]) + MathAbs(g_w_q75[h]);

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
            gmag_cls += MathAbs(g_w_cls[c][h]);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            gmag_gate += MathAbs(g_wi_x[h][i]) + MathAbs(g_wf_x[h][i]) +
                         MathAbs(g_wo_x[h][i]) + MathAbs(g_wg_x[h][i]);
         }
         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            gmag_gate += MathAbs(g_wi_h[h][j]) + MathAbs(g_wf_h[h][j]) +
                         MathAbs(g_wo_h[h][j]) + MathAbs(g_wg_h[h][j]);
         }
      }

      double lr_gate = AdamGroupLR(0, gmag_gate, base_lr);
      double lr_cls = AdamGroupLR(1, gmag_cls, base_lr * 0.80);
      double lr_move = AdamGroupLR(2, gmag_move, base_lr * 0.70);

      double wd_gate = FX6_Clamp(l2 * 0.20, 0.0, 0.20);
      double wd_cls = FX6_Clamp(l2 * 0.10, 0.0, 0.15);
      double wd_move = FX6_Clamp(l2 * 0.08, 0.0, 0.12);

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_bi[h] += lr_gate * gscale * g_bi[h];
         m_bf[h] += lr_gate * gscale * g_bf[h];
         m_bo[h] += lr_gate * gscale * g_bo[h];
         m_bg[h] += lr_gate * gscale * g_bg[h];

         m_w_mu[h] = m_w_mu[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_mu[h];
         m_w_logv[h] = m_w_logv[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_logv[h];
         m_w_q25[h] = m_w_q25[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q25[h];
         m_w_q75[h] = m_w_q75[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q75[h];

         for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
            m_w_cls[c][h] = m_w_cls[c][h] * (1.0 - lr_cls * wd_cls) + lr_cls * gscale * g_w_cls[c][h];

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_wi_x[h][i] = m_wi_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wi_x[h][i];
            m_wf_x[h][i] = m_wf_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wf_x[h][i];
            m_wo_x[h][i] = m_wo_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wo_x[h][i];
            m_wg_x[h][i] = m_wg_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wg_x[h][i];
         }

         for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
         {
            m_wi_h[h][j] = m_wi_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wi_h[h][j];
            m_wf_h[h][j] = m_wf_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wf_h[h][j];
            m_wo_h[h][j] = m_wo_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wo_h[h][j];
            m_wg_h[h][j] = m_wg_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wg_h[h][j];
         }
      }

      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         m_b_cls[c] += lr_cls * gscale * g_b_cls[c];
         m_b_cls[c] = FX6_ClipSym(m_b_cls[c], 6.0);
      }

      m_b_mu += lr_move * gscale * g_b_mu;
      m_b_logv += lr_move * gscale * g_b_logv;
      m_b_q25 += lr_move * gscale * g_b_q25;
      m_b_q75 += lr_move * gscale * g_b_q75;

      for(int t=0; t<len; t++)
      {
         int cls = m_batch_y[t];
         if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
            cls = (int)FX6_LABEL_SKIP;

         double sw = FX6_Clamp(m_batch_w[t], 0.25, 6.00);
         double p_raw[FX6_LSTM_CLASS_COUNT];
         p_raw[0] = probs[t][0];
         p_raw[1] = probs[t][1];
         p_raw[2] = probs[t][2];

         UpdateCalibrator3(p_raw, cls, sw, base_lr);

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
   CFX6AILSTM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_LSTM; }
   virtual string AIName(void) const { return "lstm"; }
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

      for(int g=0; g<6; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FX6_LSTM_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FX6_LSTM_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      ResetState();
      ResetBatch();
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
   }

   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      if(ArraySize(class_probs) < FX6_LSTM_CLASS_COUNT)
         ArrayResize(class_probs, FX6_LSTM_CLASS_COUNT);

      double h_state[FX6_AI_MLP_HIDDEN];
      double c_state[FX6_AI_MLP_HIDDEN];
      BuildStateWithPending(h_state, c_state);

      double ig[FX6_AI_MLP_HIDDEN];
      double fg[FX6_AI_MLP_HIDDEN];
      double og[FX6_AI_MLP_HIDDEN];
      double gg[FX6_AI_MLP_HIDDEN];
      double c_new[FX6_AI_MLP_HIDDEN];
      double h_new[FX6_AI_MLP_HIDDEN];
      ForwardOne(x, h_state, c_state, false, 0, ig, fg, og, gg, c_new, h_new);

      double logits[FX6_LSTM_CLASS_COUNT];
      double probs_raw[FX6_LSTM_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_new, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FX6_LABEL_SKIP]);
      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);

      if(ev > 0.0 && base_ev > 0.0) expected_move_points = 0.70 * ev + 0.30 * base_ev;
      else if(ev > 0.0) expected_move_points = ev;
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

      MaybeResetForRegime(x);

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FX6_LABEL_SELL || cls > (int)FX6_LABEL_SKIP)
         cls = (int)FX6_LABEL_SKIP;

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(x, move_points);
      if(cls == (int)FX6_LABEL_SKIP) w *= 0.90;
      w = FX6_Clamp(w, 0.10, 6.00);

      AppendBatch(cls, x, move_points, w);

      if(m_batch_size >= FX6_LSTM_TBPTT ||
         (m_batch_size >= 4 && (m_seen_updates % 4) == 0))
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

#endif // __FX6_AI_LSTM_MQH__
