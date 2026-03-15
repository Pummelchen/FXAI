#ifndef __FXAI_AI_LSTM_MQH__
#define __FXAI_AI_LSTM_MQH__

#include "..\API\plugin_base.mqh"

#define FXAI_LSTM_TBPTT 24
#define FXAI_LSTM_CLASS_COUNT 3
#define FXAI_LSTM_CAL_BINS 12
#define FXAI_LSTM_DROP_RATE 0.08
#define FXAI_LSTM_ZONEOUT 0.05
#define FXAI_LSTM_LN_EPS 0.00001
#define FXAI_LSTM_REPLAY 384
#define FXAI_LSTM_ECE_BINS 12

class CFXAIAILSTM : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_step;
   int    m_seen_updates;
   int    m_adam_t;

   // Regime/session-aware state control.
   datetime m_last_update_time;
   int      m_last_session_bucket;
   datetime m_last_sample_time;
   double   m_pending_reset_flag;
   bool     m_vol_ready;
   double   m_vol_ema;

   double m_h[FXAI_AI_MLP_HIDDEN];
   double m_c[FXAI_AI_MLP_HIDDEN];

   double m_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];

   double m_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];

   double m_bi[FXAI_AI_MLP_HIDDEN];
   double m_bf[FXAI_AI_MLP_HIDDEN];
   double m_bo[FXAI_AI_MLP_HIDDEN];
   double m_bg[FXAI_AI_MLP_HIDDEN];

   // LayerNorm parameters for recurrent pre-activations.
   double m_ln_gi[FXAI_AI_MLP_HIDDEN];
   double m_ln_gf[FXAI_AI_MLP_HIDDEN];
   double m_ln_go[FXAI_AI_MLP_HIDDEN];
   double m_ln_gg[FXAI_AI_MLP_HIDDEN];
   double m_ln_bi[FXAI_AI_MLP_HIDDEN];
   double m_ln_bf[FXAI_AI_MLP_HIDDEN];
   double m_ln_bo[FXAI_AI_MLP_HIDDEN];
   double m_ln_bg[FXAI_AI_MLP_HIDDEN];

   // Native 3-class head.
   double m_w_cls[FXAI_LSTM_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_b_cls[FXAI_LSTM_CLASS_COUNT];

   // Distributional move head.
   double m_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_b_q75;
   CFXAINativeQualityHeads m_quality_heads;

   // Sequence batching buffer for truncated BPTT.
   int    m_batch_size;
   double m_batch_h0[FXAI_AI_MLP_HIDDEN];
   double m_batch_c0[FXAI_AI_MLP_HIDDEN];
   double m_batch_x[FXAI_LSTM_TBPTT][FXAI_AI_WEIGHTS];
   int    m_batch_y[FXAI_LSTM_TBPTT];
   double m_batch_move[FXAI_LSTM_TBPTT];
   double m_batch_w[FXAI_LSTM_TBPTT];
   double m_batch_reset[FXAI_LSTM_TBPTT];

   // Grouped AdamW moments.
   double m_opt_m[6];
   double m_opt_v[6];

   // EMA/SWA shadow weights for robust inference.
   bool   m_ema_ready;
   int    m_ema_steps;
   double m_ema_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_ema_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_ema_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_ema_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_ema_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ema_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ema_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ema_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
   double m_ema_bi[FXAI_AI_MLP_HIDDEN];
   double m_ema_bf[FXAI_AI_MLP_HIDDEN];
   double m_ema_bo[FXAI_AI_MLP_HIDDEN];
   double m_ema_bg[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_gi[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_gf[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_go[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_gg[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_bi[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_bf[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_bo[FXAI_AI_MLP_HIDDEN];
   double m_ema_ln_bg[FXAI_AI_MLP_HIDDEN];
   double m_ema_w_cls[FXAI_LSTM_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_ema_b_cls[FXAI_LSTM_CLASS_COUNT];
   double m_ema_w_mu[FXAI_AI_MLP_HIDDEN];
   double m_ema_b_mu;
   double m_ema_w_logv[FXAI_AI_MLP_HIDDEN];
   double m_ema_b_logv;
   double m_ema_w_q25[FXAI_AI_MLP_HIDDEN];
   double m_ema_b_q25;
   double m_ema_w_q75[FXAI_AI_MLP_HIDDEN];
   double m_ema_b_q75;

   // Drift-aware replay buffer.
   double   m_lstm_replay_x[FXAI_LSTM_REPLAY][FXAI_AI_WEIGHTS];
   int      m_replay_y[FXAI_LSTM_REPLAY];
   double   m_lstm_replay_move[FXAI_LSTM_REPLAY];
   double   m_lstm_replay_cost[FXAI_LSTM_REPLAY];
   double   m_replay_w[FXAI_LSTM_REPLAY];
   datetime m_lstm_replay_time[FXAI_LSTM_REPLAY];
   int      m_replay_session[FXAI_LSTM_REPLAY];
   int      m_lstm_replay_head;
   int      m_lstm_replay_size;

   // Online validation / quality gate.
   bool   m_val_ready;
   int    m_val_steps;
   double m_val_nll_fast;
   double m_val_nll_slow;
   double m_val_brier_fast;
   double m_val_brier_slow;
   double m_val_ece_fast;
   double m_val_ece_slow;
   double m_val_ev_fast;
   double m_val_ev_slow;
   double m_ece_mass[FXAI_LSTM_ECE_BINS];
   double m_ece_acc[FXAI_LSTM_ECE_BINS];
   double m_ece_conf[FXAI_LSTM_ECE_BINS];
   bool   m_quality_degraded;

   // Native multiclass calibration.
   double m_cal3_temp;
   double m_cal3_bias[FXAI_LSTM_CLASS_COUNT];
   double m_cal3_iso_pos[FXAI_LSTM_CLASS_COUNT][FXAI_LSTM_CAL_BINS];
   double m_cal3_iso_cnt[FXAI_LSTM_CLASS_COUNT][FXAI_LSTM_CAL_BINS];
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
      double keep = 1.0 - FXAI_LSTM_DROP_RATE;
      if(keep <= 0.05) keep = 0.05;
      double u = Hash01(m_step, salt, idx);
      if(u < FXAI_LSTM_DROP_RATE) return 0.0;
      return 1.0 / keep;
   }

   bool ZoneKeep(const int salt, const int idx) const
   {
      double u = Hash01(m_seen_updates + 17, salt, idx);
      return (u < FXAI_LSTM_ZONEOUT);
   }

   double BlendParam(const double base, const double ema, const bool use_ema) const
   {
      return (use_ema ? ema : base);
   }

   bool UseEMAInference(void) const
   {
      return (m_ema_ready && m_seen_updates >= 32);
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
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) mean += z_in[h];
      mean /= (double)FXAI_AI_MLP_HIDDEN;

      double var = 0.0;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double d = z_in[h] - mean;
         var += d * d;
      }
      var /= (double)FXAI_AI_MLP_HIDDEN;

      inv_std = 1.0 / MathSqrt(var + FXAI_LSTM_LN_EPS);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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
      double dzhat[FXAI_AI_MLP_HIDDEN];
      double sum1 = 0.0;
      double sum2 = 0.0;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double dy = dy_ln[h];
         g_gamma[h] += dy * z_hat[h];
         g_beta[h] += dy;

         dzhat[h] = dy * gamma[h];
         sum1 += dzhat[h];
         sum2 += dzhat[h] * z_hat[h];
      }

      double n = (double)FXAI_AI_MLP_HIDDEN;
      double k = inv_std / n;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double v = (n * dzhat[h]) - sum1 - z_hat[h] * sum2;
         dz_raw[h] = k * v;
      }
   }

   void SyncEMAWithParams(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
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

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            m_ema_w_cls[c][h] = m_w_cls[c][h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_ema_wi_x[h][i] = m_wi_x[h][i];
            m_ema_wf_x[h][i] = m_wf_x[h][i];
            m_ema_wo_x[h][i] = m_wo_x[h][i];
            m_ema_wg_x[h][i] = m_wg_x[h][i];
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_ema_wi_h[h][j] = m_wi_h[h][j];
            m_ema_wf_h[h][j] = m_wf_h[h][j];
            m_ema_wo_h[h][j] = m_wo_h[h][j];
            m_ema_wg_h[h][j] = m_wg_h[h][j];
         }
      }

      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
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
      double d = FXAI_Clamp(decay, 0.9000, 0.9999);
      double one = 1.0 - d;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
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

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            m_ema_w_cls[c][h] = d * m_ema_w_cls[c][h] + one * m_w_cls[c][h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_ema_wi_x[h][i] = d * m_ema_wi_x[h][i] + one * m_wi_x[h][i];
            m_ema_wf_x[h][i] = d * m_ema_wf_x[h][i] + one * m_wf_x[h][i];
            m_ema_wo_x[h][i] = d * m_ema_wo_x[h][i] + one * m_wo_x[h][i];
            m_ema_wg_x[h][i] = d * m_ema_wg_x[h][i] + one * m_wg_x[h][i];
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_ema_wi_h[h][j] = d * m_ema_wi_h[h][j] + one * m_wi_h[h][j];
            m_ema_wf_h[h][j] = d * m_ema_wf_h[h][j] + one * m_wf_h[h][j];
            m_ema_wo_h[h][j] = d * m_ema_wo_h[h][j] + one * m_wo_h[h][j];
            m_ema_wg_h[h][j] = d * m_ema_wg_h[h][j] + one * m_wg_h[h][j];
         }
      }

      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         m_ema_b_cls[c] = d * m_ema_b_cls[c] + one * m_b_cls[c];

      m_ema_b_mu = d * m_ema_b_mu + one * m_b_mu;
      m_ema_b_logv = d * m_ema_b_logv + one * m_b_logv;
      m_ema_b_q25 = d * m_ema_b_q25 + one * m_b_q25;
      m_ema_b_q75 = d * m_ema_b_q75 + one * m_b_q75;

      m_ema_steps++;
      if(m_ema_steps >= 8) m_ema_ready = true;
   }

   int ReplayPos(const int logical_idx) const
   {
      if(m_lstm_replay_size <= 0) return 0;
      int start = m_lstm_replay_head - m_lstm_replay_size;
      while(start < 0) start += FXAI_LSTM_REPLAY;
      int p = start + logical_idx;
      while(p >= FXAI_LSTM_REPLAY) p -= FXAI_LSTM_REPLAY;
      return p;
   }

   void PushReplay(const int cls,
                   const double &x[],
                   const double move_points,
                   const double cost_points,
                   const double sample_w,
                   const datetime t_sample,
                   const int sess)
   {
      int p = m_lstm_replay_head;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_lstm_replay_x[p][i] = x[i];
      m_replay_y[p] = cls;
      m_lstm_replay_move[p] = move_points;
      m_lstm_replay_cost[p] = cost_points;
      m_replay_w[p] = sample_w;
      m_lstm_replay_time[p] = t_sample;
      m_replay_session[p] = sess;

      m_lstm_replay_head++;
      if(m_lstm_replay_head >= FXAI_LSTM_REPLAY) m_lstm_replay_head = 0;
      if(m_lstm_replay_size < FXAI_LSTM_REPLAY) m_lstm_replay_size++;
   }

   double ReplayAgeWeight(const datetime t_sample,
                          const datetime t_now) const
   {
      if(t_sample <= 0 || t_now <= 0) return 1.0;
      double age_min = (double)MathMax(0, (int)(t_now - t_sample)) / 60.0;
      double half_life = 24.0 * 60.0;
      return MathExp(-0.69314718056 * age_min / half_life);
   }

   void UpdateValidationMetrics(const int cls,
                                const double &p_cal[],
                                const double ev_after_cost)
   {
      int y = cls;
      if(y < 0) y = 0;
      if(y >= FXAI_LSTM_CLASS_COUNT) y = (int)FXAI_LABEL_SKIP;

      double ce = -MathLog(FXAI_Clamp(p_cal[y], 1e-6, 1.0));
      double brier = 0.0;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double t = (c == y ? 1.0 : 0.0);
         double d = p_cal[c] - t;
         brier += d * d;
      }
      brier /= (double)FXAI_LSTM_CLASS_COUNT;

      double conf = p_cal[0];
      int pred = 0;
      for(int c=1; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         if(p_cal[c] > conf)
         {
            conf = p_cal[c];
            pred = c;
         }
      }
      double acc = (pred == y ? 1.0 : 0.0);

      int bi = (int)MathFloor(conf * (double)FXAI_LSTM_ECE_BINS);
      if(bi < 0) bi = 0;
      if(bi >= FXAI_LSTM_ECE_BINS) bi = FXAI_LSTM_ECE_BINS - 1;

      for(int b=0; b<FXAI_LSTM_ECE_BINS; b++)
      {
         m_ece_mass[b] *= 0.997;
         m_ece_acc[b] *= 0.997;
         m_ece_conf[b] *= 0.997;
      }
      m_ece_mass[bi] += 1.0;
      m_ece_acc[bi] += acc;
      m_ece_conf[bi] += conf;

      double ece_num = 0.0;
      double ece_den = 0.0;
      for(int b=0; b<FXAI_LSTM_ECE_BINS; b++)
      {
         if(m_ece_mass[b] <= 1e-9) continue;
         double ba = m_ece_acc[b] / m_ece_mass[b];
         double bc = m_ece_conf[b] / m_ece_mass[b];
         ece_num += m_ece_mass[b] * MathAbs(ba - bc);
         ece_den += m_ece_mass[b];
      }
      double ece = (ece_den > 0.0 ? ece_num / ece_den : 0.0);

      if(!m_val_ready)
      {
         m_val_nll_fast = m_val_nll_slow = ce;
         m_val_brier_fast = m_val_brier_slow = brier;
         m_val_ece_fast = m_val_ece_slow = ece;
         m_val_ev_fast = m_val_ev_slow = ev_after_cost;
         m_val_ready = true;
      }
      else
      {
         m_val_nll_fast = 0.92 * m_val_nll_fast + 0.08 * ce;
         m_val_nll_slow = 0.995 * m_val_nll_slow + 0.005 * ce;
         m_val_brier_fast = 0.92 * m_val_brier_fast + 0.08 * brier;
         m_val_brier_slow = 0.995 * m_val_brier_slow + 0.005 * brier;
         m_val_ece_fast = 0.92 * m_val_ece_fast + 0.08 * ece;
         m_val_ece_slow = 0.995 * m_val_ece_slow + 0.005 * ece;
         m_val_ev_fast = 0.92 * m_val_ev_fast + 0.08 * ev_after_cost;
         m_val_ev_slow = 0.995 * m_val_ev_slow + 0.005 * ev_after_cost;
      }

      m_val_steps++;
      m_quality_degraded = false;
      if(m_val_steps > 128)
      {
         if(m_val_nll_fast > 1.15 * MathMax(0.05, m_val_nll_slow)) m_quality_degraded = true;
         if(m_val_brier_fast > 1.20 * MathMax(0.03, m_val_brier_slow)) m_quality_degraded = true;
         if(m_val_ece_fast > 1.25 * MathMax(0.02, m_val_ece_slow)) m_quality_degraded = true;
         if(m_val_ev_fast < 0.85 * m_val_ev_slow) m_quality_degraded = true;
      }
   }

   void SanitizeParams(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_bi[h] = FXAI_ClipSym(m_bi[h], 6.0);
         m_bf[h] = FXAI_ClipSym(m_bf[h], 6.0);
         m_bo[h] = FXAI_ClipSym(m_bo[h], 6.0);
         m_bg[h] = FXAI_ClipSym(m_bg[h], 6.0);

         m_ln_gi[h] = FXAI_Clamp(m_ln_gi[h], 0.10, 4.00);
         m_ln_gf[h] = FXAI_Clamp(m_ln_gf[h], 0.10, 4.00);
         m_ln_go[h] = FXAI_Clamp(m_ln_go[h], 0.10, 4.00);
         m_ln_gg[h] = FXAI_Clamp(m_ln_gg[h], 0.10, 4.00);
         m_ln_bi[h] = FXAI_ClipSym(m_ln_bi[h], 4.0);
         m_ln_bf[h] = FXAI_ClipSym(m_ln_bf[h], 4.0);
         m_ln_bo[h] = FXAI_ClipSym(m_ln_bo[h], 4.0);
         m_ln_bg[h] = FXAI_ClipSym(m_ln_bg[h], 4.0);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_wi_x[h][i] = FXAI_ClipSym(m_wi_x[h][i], 4.0);
            m_wf_x[h][i] = FXAI_ClipSym(m_wf_x[h][i], 4.0);
            m_wo_x[h][i] = FXAI_ClipSym(m_wo_x[h][i], 4.0);
            m_wg_x[h][i] = FXAI_ClipSym(m_wg_x[h][i], 4.0);
         }
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_wi_h[h][j] = FXAI_ClipSym(m_wi_h[h][j], 4.0);
            m_wf_h[h][j] = FXAI_ClipSym(m_wf_h[h][j], 4.0);
            m_wo_h[h][j] = FXAI_ClipSym(m_wo_h[h][j], 4.0);
            m_wg_h[h][j] = FXAI_ClipSym(m_wg_h[h][j], 4.0);
         }
      }

      m_b_mu = FXAI_ClipSym(m_b_mu, 30.0);
      m_b_logv = FXAI_Clamp(m_b_logv, -6.0, 6.0);
      m_b_q25 = FXAI_ClipSym(m_b_q25, 30.0);
      m_b_q75 = FXAI_ClipSym(m_b_q75, 30.0);
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 8.0);
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
      return FXAI_Clamp(base_lr * (0.55 + 0.45 * scale), 0.000002, 0.060000);
   }

   void Softmax3(const double &logits[], double &probs[]) const
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

   void Calibrate3(const double &p_raw[],
                   double &p_cal[]) const
   {
      double inv_temp = 1.0 / FXAI_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FXAI_LSTM_CLASS_COUNT];
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double pr = FXAI_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }
      Softmax3(logits, p_cal);

      if(m_cal3_steps < 30) return;

      double p_iso[FXAI_LSTM_CLASS_COUNT];
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double total = 0.0;
         for(int b=0; b<FXAI_LSTM_CAL_BINS; b++)
            total += m_cal3_iso_cnt[c][b];

         if(total < 30.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[FXAI_LSTM_CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<FXAI_LSTM_CAL_BINS; b++)
         {
            double r = prev;
            if(m_cal3_iso_cnt[c][b] > 1e-9)
               r = m_cal3_iso_pos[c][b] / m_cal3_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_LSTM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_LSTM_CAL_BINS) bi = FXAI_LSTM_CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
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
      double inv_temp = 1.0 / FXAI_Clamp(m_cal3_temp, 0.50, 3.00);
      double logits[FXAI_LSTM_CLASS_COUNT];
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double pr = FXAI_Clamp(p_raw[c], 0.0005, 0.9990);
         logits[c] = MathLog(pr) * inv_temp + m_cal3_bias[c];
      }

      double p_cal[FXAI_LSTM_CLASS_COUNT];
      Softmax3(logits, p_cal);

      double w = FXAI_Clamp(sample_w, 0.25, 6.00);
      double cal_lr = FXAI_Clamp(0.20 * lr * w, 0.0002, 0.0200);

      double g_temp = 0.0;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_cal3_bias[c] = FXAI_ClipSym(m_cal3_bias[c] + cal_lr * e, 4.0);
         g_temp += e * MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

         int bi = (int)MathFloor(p_cal[c] * (double)FXAI_LSTM_CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= FXAI_LSTM_CAL_BINS) bi = FXAI_LSTM_CAL_BINS - 1;
         m_cal3_iso_cnt[c][bi] += w;
         m_cal3_iso_pos[c][bi] += w * target;
      }

      m_cal3_temp = FXAI_Clamp(m_cal3_temp - 0.02 * cal_lr * g_temp, 0.50, 3.00);
      m_cal3_steps++;
   }

   void MaybeResetForRegime(const double &x[])
   {
      datetime now = ResolveContextTime();
      if(now <= 0) now = TimeCurrent();

      int sess = SessionBucket(now);
      bool hard_reset = false;
      bool soft_reset = false;
      m_pending_reset_flag = 0.0;

      if(m_last_update_time > 0)
      {
         int gap_sec = (int)(now - m_last_update_time);
         if(gap_sec > 45 * 60)
            hard_reset = true;
      }

      if(m_last_sample_time > 0)
      {
         int sample_gap = (int)(now - m_last_sample_time);
         if(sample_gap > 20 * 60) hard_reset = true;
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
         soft_reset = true;
      }

      if(hard_reset)
      {
         ResetState();
         ResetBatch();
         m_pending_reset_flag = 1.0;
      }
      else if(soft_reset)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_h[h] *= 0.40;
            m_c[h] *= 0.40;
         }
         ResetBatch();
         m_pending_reset_flag = 1.0;
      }

      m_last_update_time = now;
      m_last_sample_time = now;
      m_last_session_bucket = sess;
   }

   void ResetState(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = 0.0;
         m_c[h] = 0.0;
      }
   }

   void ResetBatch(void)
   {
      m_batch_size = 0;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_batch_h0[h] = m_h[h];
         m_batch_c0[h] = m_c[h];
      }
      for(int t=0; t<FXAI_LSTM_TBPTT; t++)
         m_batch_reset[t] = 0.0;
   }

   void InitWeights(void)
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double seed = (double)((h + 1) * (i + 3));
            m_wi_x[h][i] = 0.03 * MathSin(seed * 1.03);
            m_wf_x[h][i] = 0.03 * MathCos(seed * 1.07);
            m_wo_x[h][i] = 0.03 * MathSin(seed * 1.11);
            m_wg_x[h][i] = 0.03 * MathCos(seed * 1.13);
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
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

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            m_w_cls[c][h] = 0.04 * MathSin((double)((c + 2) * (h + 1)) * 1.07);
      }

      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         m_b_cls[c] = 0.0;
      m_b_cls[(int)FXAI_LABEL_SKIP] = 0.15;

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FXAI_LSTM_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      for(int b=0; b<FXAI_LSTM_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }

      for(int g=0; g<6; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_lstm_replay_head = 0;
      m_lstm_replay_size = 0;
      for(int i=0; i<FXAI_LSTM_REPLAY; i++)
      {
         m_replay_y[i] = (int)FXAI_LABEL_SKIP;
         m_lstm_replay_move[i] = 0.0;
         m_lstm_replay_cost[i] = 0.0;
         m_replay_w[i] = 1.0;
         m_lstm_replay_time[i] = 0;
         m_replay_session[i] = -1;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_lstm_replay_x[i][k] = 0.0;
      }

      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      for(int b=0; b<FXAI_LSTM_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_quality_degraded = false;

      SyncEMAWithParams();

      m_initialized = true;
   }

   void ForwardOne(const double &x[],
                   const double &h_prev[],
                   const double &c_prev[],
                   const bool training,
                   const bool use_ema,
                   const int salt,
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
      double h_in[FXAI_AI_MLP_HIDDEN];
      for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
      {
         if(training) h_in[j] = h_prev[j] * DropScale(salt, j);
         else h_in[j] = h_prev[j];
      }

      double zi_raw[FXAI_AI_MLP_HIDDEN];
      double zf_raw[FXAI_AI_MLP_HIDDEN];
      double zo_raw[FXAI_AI_MLP_HIDDEN];
      double zg_raw[FXAI_AI_MLP_HIDDEN];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zi = BlendParam(m_bi[h], m_ema_bi[h], use_ema);
         double zf = BlendParam(m_bf[h], m_ema_bf[h], use_ema);
         double zo = BlendParam(m_bo[h], m_ema_bo[h], use_ema);
         double zg = BlendParam(m_bg[h], m_ema_bg[h], use_ema);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            zi += BlendParam(m_wi_x[h][i], m_ema_wi_x[h][i], use_ema) * x[i];
            zf += BlendParam(m_wf_x[h][i], m_ema_wf_x[h][i], use_ema) * x[i];
            zo += BlendParam(m_wo_x[h][i], m_ema_wo_x[h][i], use_ema) * x[i];
            zg += BlendParam(m_wg_x[h][i], m_ema_wg_x[h][i], use_ema) * x[i];
         }
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            zi += BlendParam(m_wi_h[h][j], m_ema_wi_h[h][j], use_ema) * h_in[j];
            zf += BlendParam(m_wf_h[h][j], m_ema_wf_h[h][j], use_ema) * h_in[j];
            zo += BlendParam(m_wo_h[h][j], m_ema_wo_h[h][j], use_ema) * h_in[j];
            zg += BlendParam(m_wg_h[h][j], m_ema_wg_h[h][j], use_ema) * h_in[j];
         }

         zi_raw[h] = zi;
         zf_raw[h] = zf;
         zo_raw[h] = zo;
         zg_raw[h] = zg;
      }

      double zi_ln[FXAI_AI_MLP_HIDDEN];
      double zf_ln[FXAI_AI_MLP_HIDDEN];
      double zo_ln[FXAI_AI_MLP_HIDDEN];
      double zg_ln[FXAI_AI_MLP_HIDDEN];
      double mean_i, mean_f, mean_o, mean_g;
      LayerNormForward(zi_raw, m_ln_gi, m_ln_bi, m_ema_ln_gi, m_ema_ln_bi, use_ema,
                       zi_hat, zi_ln, mean_i, inv_i);
      LayerNormForward(zf_raw, m_ln_gf, m_ln_bf, m_ema_ln_gf, m_ema_ln_bf, use_ema,
                       zf_hat, zf_ln, mean_f, inv_f);
      LayerNormForward(zo_raw, m_ln_go, m_ln_bo, m_ema_ln_go, m_ema_ln_bo, use_ema,
                       zo_hat, zo_ln, mean_o, inv_o);
      LayerNormForward(zg_raw, m_ln_gg, m_ln_bg, m_ema_ln_gg, m_ema_ln_bg, use_ema,
                       zg_hat, zg_ln, mean_g, inv_g);

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         ig[h] = FXAI_Sigmoid(FXAI_ClipSym(zi_ln[h], 10.0));
         fg[h] = FXAI_Sigmoid(FXAI_ClipSym(zf_ln[h], 10.0));
         og[h] = FXAI_Sigmoid(FXAI_ClipSym(zo_ln[h], 10.0));
         gg[h] = FXAI_Tanh(FXAI_ClipSym(zg_ln[h], 10.0));

         c_new[h] = fg[h] * c_prev[h] + ig[h] * gg[h];
         c_new[h] = FXAI_ClipSym(c_new[h], 10.0);
         h_new[h] = og[h] * FXAI_Tanh(c_new[h]);

         if(training && ZoneKeep(salt, h))
         {
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
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         double z = BlendParam(m_b_cls[c], m_ema_b_cls[c], use_ema);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            z += BlendParam(m_w_cls[c][h], m_ema_w_cls[c][h], use_ema) * hvec[h];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      mu = BlendParam(m_b_mu, m_ema_b_mu, use_ema);
      logv = BlendParam(m_b_logv, m_ema_b_logv, use_ema);
      q25 = BlendParam(m_b_q25, m_ema_b_q25, use_ema);
      q75 = BlendParam(m_b_q75, m_ema_b_q75, use_ema);
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         mu += BlendParam(m_w_mu[h], m_ema_w_mu[h], use_ema) * hvec[h];
         logv += BlendParam(m_w_logv[h], m_ema_w_logv[h], use_ema) * hvec[h];
         q25 += BlendParam(m_w_q25[h], m_ema_w_q25[h], use_ema) * hvec[h];
         q75 += BlendParam(m_w_q75[h], m_ema_w_q75[h], use_ema) * hvec[h];
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
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
      double sigma = MathExp(0.5 * FXAI_Clamp(logv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double ev = (0.68 * MathAbs(mu) + 0.22 * sigma + 0.10 * iqr) * FXAI_Clamp(1.0 - p_skip, 0.0, 1.0);
      return ev;
   }

   void AppendBatch(const int y_cls,
                    const double &x[],
                    const double move_points,
                    const double sample_w,
                    const double reset_flag)
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_batch_h0[h] = m_h[h];
            m_batch_c0[h] = m_c[h];
         }
      }

      if(m_batch_size < FXAI_LSTM_TBPTT)
      {
         int p = m_batch_size;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[p][i] = x[i];
         m_batch_y[p] = y_cls;
         m_batch_move[p] = move_points;
         m_batch_w[p] = sample_w;
         m_batch_reset[p] = reset_flag;
         m_batch_size++;
         return;
      }

      for(int t=1; t<FXAI_LSTM_TBPTT; t++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[t - 1][i] = m_batch_x[t][i];
         m_batch_y[t - 1] = m_batch_y[t];
         m_batch_move[t - 1] = m_batch_move[t];
         m_batch_w[t - 1] = m_batch_w[t];
         m_batch_reset[t - 1] = m_batch_reset[t];
      }

      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_batch_x[FXAI_LSTM_TBPTT - 1][i] = x[i];
      m_batch_y[FXAI_LSTM_TBPTT - 1] = y_cls;
      m_batch_move[FXAI_LSTM_TBPTT - 1] = move_points;
      m_batch_w[FXAI_LSTM_TBPTT - 1] = sample_w;
      m_batch_reset[FXAI_LSTM_TBPTT - 1] = reset_flag;
      m_batch_size = FXAI_LSTM_TBPTT;
   }

   void BuildStateWithPending(double &h_out[],
                              double &c_out[],
                              const bool use_ema) const
   {
      if(m_batch_size <= 0)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_out[h] = m_h[h];
            c_out[h] = m_c[h];
         }
         return;
      }

      double h_cur[FXAI_AI_MLP_HIDDEN];
      double c_cur[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         h_cur[h] = m_batch_h0[h];
         c_cur[h] = m_batch_c0[h];
      }

      for(int t=0; t<m_batch_size; t++)
      {
         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               h_cur[h] = 0.0;
               c_cur[h] = 0.0;
            }
         }

         double ig[FXAI_AI_MLP_HIDDEN];
         double fg[FXAI_AI_MLP_HIDDEN];
         double og[FXAI_AI_MLP_HIDDEN];
         double gg[FXAI_AI_MLP_HIDDEN];
         double zi_hat[FXAI_AI_MLP_HIDDEN];
         double zf_hat[FXAI_AI_MLP_HIDDEN];
         double zo_hat[FXAI_AI_MLP_HIDDEN];
         double zg_hat[FXAI_AI_MLP_HIDDEN];
         double inv_i, inv_f, inv_o, inv_g;
         double c_new[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];

         double xloc[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            xloc[i] = m_batch_x[t][i];

         ForwardOne(xloc, h_cur, c_cur, false, use_ema, t,
                    zi_hat, zf_hat, zo_hat, zg_hat,
                    inv_i, inv_f, inv_o, inv_g,
                    ig, fg, og, gg, c_new, h_new);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_cur[h] = h_new[h];
            c_cur[h] = c_new[h];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         h_out[h] = h_cur[h];
         c_out[h] = c_cur[h];
      }
   }

   void TrainBatch(const FXAIAIHyperParams &hp)
   {
      int len = m_batch_size;
      if(len <= 0) return;

      m_step += len;
      m_adam_t++;

      double base_lr0 = FXAI_Clamp(hp.lr, 0.00005, 0.30000);
      double warm = FXAI_Clamp((double)m_step / 512.0, 0.05, 1.00);
      double invs = 1.0 / MathSqrt(1.0 + 0.0008 * MathMax(0.0, (double)m_step - 256.0));
      double phase = MathMod((double)m_step, 4096.0) / 4096.0;
      double cosine = 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
      double step_drop = MathPow(0.85, (double)((int)(m_step / 8192)));
      double base_lr = FXAI_Clamp(base_lr0 * warm * invs * (0.65 + 0.35 * cosine) * step_drop, 0.00001, 0.05000);
      if(m_quality_degraded) base_lr *= 0.75;
      double l2 = FXAI_Clamp(hp.l2, 0.0000, 0.2000);

      double h_prev[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double c_prev[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double zi_hat[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double zf_hat[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double zo_hat[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double zg_hat[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double inv_i[FXAI_LSTM_TBPTT];
      double inv_f[FXAI_LSTM_TBPTT];
      double inv_o[FXAI_LSTM_TBPTT];
      double inv_g[FXAI_LSTM_TBPTT];
      double ig[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double fg[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double og[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double gg[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double c_t[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];
      double h_t[FXAI_LSTM_TBPTT][FXAI_AI_MLP_HIDDEN];

      double logits[FXAI_LSTM_TBPTT][FXAI_LSTM_CLASS_COUNT];
      double probs[FXAI_LSTM_TBPTT][FXAI_LSTM_CLASS_COUNT];
      double mu[FXAI_LSTM_TBPTT];
      double logv[FXAI_LSTM_TBPTT];
      double q25[FXAI_LSTM_TBPTT];
      double q75[FXAI_LSTM_TBPTT];

      double h_cur[FXAI_AI_MLP_HIDDEN];
      double c_cur[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         h_cur[h] = m_batch_h0[h];
         c_cur[h] = m_batch_c0[h];
      }

      for(int t=0; t<len; t++)
      {
         if(m_batch_reset[t] > 0.5)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               h_cur[h] = 0.0;
               c_cur[h] = 0.0;
            }
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_prev[t][h] = h_cur[h];
            c_prev[t][h] = c_cur[h];
         }

         double xloc[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];

         double ig_loc[FXAI_AI_MLP_HIDDEN];
         double fg_loc[FXAI_AI_MLP_HIDDEN];
         double og_loc[FXAI_AI_MLP_HIDDEN];
         double gg_loc[FXAI_AI_MLP_HIDDEN];
         double zi_hat_loc[FXAI_AI_MLP_HIDDEN];
         double zf_hat_loc[FXAI_AI_MLP_HIDDEN];
         double zo_hat_loc[FXAI_AI_MLP_HIDDEN];
         double zg_hat_loc[FXAI_AI_MLP_HIDDEN];
         double inv_i_loc, inv_f_loc, inv_o_loc, inv_g_loc;
         double c_new[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];
         ForwardOne(xloc, h_cur, c_cur, true, false, t,
                    zi_hat_loc, zf_hat_loc, zo_hat_loc, zg_hat_loc,
                    inv_i_loc, inv_f_loc, inv_o_loc, inv_g_loc,
                    ig_loc, fg_loc, og_loc, gg_loc, c_new, h_new);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            ig[t][h] = ig_loc[h];
            fg[t][h] = fg_loc[h];
            og[t][h] = og_loc[h];
            gg[t][h] = gg_loc[h];
            zi_hat[t][h] = zi_hat_loc[h];
            zf_hat[t][h] = zf_hat_loc[h];
            zo_hat[t][h] = zo_hat_loc[h];
            zg_hat[t][h] = zg_hat_loc[h];
            c_t[t][h] = c_new[h];
            h_t[t][h] = h_new[h];
            h_cur[h] = h_new[h];
            c_cur[h] = c_new[h];
         }
         inv_i[t] = inv_i_loc;
         inv_f[t] = inv_f_loc;
         inv_o[t] = inv_o_loc;
         inv_g[t] = inv_g_loc;

         double logits_loc[FXAI_LSTM_CLASS_COUNT];
         double probs_loc[FXAI_LSTM_CLASS_COUNT];
         double mu_loc, logv_loc, q25_loc, q75_loc;
         ComputeHeads(h_new, false, logits_loc, probs_loc, mu_loc, logv_loc, q25_loc, q75_loc);

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         {
            logits[t][c] = logits_loc[c];
            probs[t][c] = probs_loc[c];
         }
         mu[t] = mu_loc;
         logv[t] = logv_loc;
         q25[t] = q25_loc;
         q75[t] = q75_loc;
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_h[h] = h_cur[h];
         m_c[h] = c_cur[h];
      }

      double g_wi_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_wf_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_wo_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_wg_x[FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_wi_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_wf_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_wo_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_wg_h[FXAI_AI_MLP_HIDDEN][FXAI_AI_MLP_HIDDEN];
      double g_bi[FXAI_AI_MLP_HIDDEN];
      double g_bf[FXAI_AI_MLP_HIDDEN];
      double g_bo[FXAI_AI_MLP_HIDDEN];
      double g_bg[FXAI_AI_MLP_HIDDEN];
      double g_ln_gi[FXAI_AI_MLP_HIDDEN];
      double g_ln_gf[FXAI_AI_MLP_HIDDEN];
      double g_ln_go[FXAI_AI_MLP_HIDDEN];
      double g_ln_gg[FXAI_AI_MLP_HIDDEN];
      double g_ln_bi[FXAI_AI_MLP_HIDDEN];
      double g_ln_bf[FXAI_AI_MLP_HIDDEN];
      double g_ln_bo[FXAI_AI_MLP_HIDDEN];
      double g_ln_bg[FXAI_AI_MLP_HIDDEN];

      double g_w_cls[FXAI_LSTM_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
      double g_b_cls[FXAI_LSTM_CLASS_COUNT];

      double g_w_mu[FXAI_AI_MLP_HIDDEN];
      double g_w_logv[FXAI_AI_MLP_HIDDEN];
      double g_w_q25[FXAI_AI_MLP_HIDDEN];
      double g_w_q75[FXAI_AI_MLP_HIDDEN];
      double g_b_mu = 0.0;
      double g_b_logv = 0.0;
      double g_b_q25 = 0.0;
      double g_b_q75 = 0.0;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g_bi[h] = 0.0;
         g_bf[h] = 0.0;
         g_bo[h] = 0.0;
         g_bg[h] = 0.0;
         g_ln_gi[h] = 0.0;
         g_ln_gf[h] = 0.0;
         g_ln_go[h] = 0.0;
         g_ln_gg[h] = 0.0;
         g_ln_bi[h] = 0.0;
         g_ln_bf[h] = 0.0;
         g_ln_bo[h] = 0.0;
         g_ln_bg[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            g_w_cls[c][h] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            g_wi_x[h][i] = 0.0;
            g_wf_x[h][i] = 0.0;
            g_wo_x[h][i] = 0.0;
            g_wg_x[h][i] = 0.0;
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            g_wi_h[h][j] = 0.0;
            g_wf_h[h][j] = 0.0;
            g_wo_h[h][j] = 0.0;
            g_wg_h[h][j] = 0.0;
         }
      }
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         g_b_cls[c] = 0.0;

      double dh_next[FXAI_AI_MLP_HIDDEN];
      double dc_next[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         dh_next[h] = 0.0;
         dc_next[h] = 0.0;
      }

      for(int t=len - 1; t>=0; t--)
      {
         if(t < len - 1 && m_batch_reset[t + 1] > 0.5)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               dh_next[h] = 0.0;
               dc_next[h] = 0.0;
            }
         }

         int cls = m_batch_y[t];
         if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
            cls = (int)FXAI_LABEL_SKIP;

         double xloc_t[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xloc_t[i] = m_batch_x[t][i];
         double cost_t = MathMax(0.0, ResolveCostPoints(xloc_t));
         double edge_t = MathAbs(m_batch_move[t]) - cost_t;

         double sw = FXAI_Clamp(m_batch_w[t], 0.20, 6.00);
         if(cls == (int)FXAI_LABEL_SKIP)
         {
            if(edge_t <= 0.0) sw *= 1.20;
            else sw *= 0.85;
         }
         else
         {
            if(edge_t <= 0.0) sw *= 0.60;
            else sw *= (1.0 + 0.06 * MathMin(edge_t, 30.0));
         }
         sw = FXAI_Clamp(sw, 0.10, 8.00);

         double err_cls[FXAI_LSTM_CLASS_COUNT];
         double target_cls[FXAI_LSTM_CLASS_COUNT];
         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++) target_cls[c] = 0.0;
         target_cls[cls] = 1.0;
         if(cls != (int)FXAI_LABEL_SKIP && edge_t <= 0.0)
         {
            target_cls[cls] = 0.35;
            target_cls[(int)FXAI_LABEL_SKIP] = 0.65;
         }
         else if(cls == (int)FXAI_LABEL_SKIP && edge_t > 0.0)
         {
            int dir = (m_batch_move[t] >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
            target_cls[(int)FXAI_LABEL_SKIP] = 0.60;
            target_cls[dir] = 0.40;
         }
         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         {
            double target = target_cls[c];
            err_cls[c] = FXAI_ClipSym((target - probs[t][c]) * sw, 3.0);
            g_b_cls[c] += err_cls[c];
         }

         double target_move = MathAbs(m_batch_move[t]);
         double sigma = MathExp(0.5 * FXAI_Clamp(logv[t], -4.0, 4.0));
         sigma = FXAI_Clamp(sigma, 0.05, 30.0);
         double sig2 = sigma * sigma;

         double err_mu = FXAI_ClipSym((target_move - mu[t]) / MathMax(1.0, target_move), 4.0);
         double err_lv = FXAI_ClipSym(((target_move - mu[t]) * (target_move - mu[t]) / (sig2 + 1e-6)) - 1.0, 4.0);
         double err_q25 = (target_move >= q25[t] ? 0.25 : -0.75);
         double err_q75 = (target_move >= q75[t] ? 0.75 : -0.25);

         double move_scale = FXAI_Clamp(0.65 * sw * (edge_t > 0.0 ? 1.0 + 0.04 * MathMin(edge_t, 25.0) : 0.70), 0.10, 8.00);

         g_b_mu += move_scale * err_mu;
         g_b_logv += move_scale * err_lv;
         g_b_q25 += move_scale * err_q25;
         g_b_q75 += move_scale * err_q75;

         double dh_prev[FXAI_AI_MLP_HIDDEN];
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++) dh_prev[j] = 0.0;

         double di_ln[FXAI_AI_MLP_HIDDEN];
         double df_ln[FXAI_AI_MLP_HIDDEN];
         double do_ln[FXAI_AI_MLP_HIDDEN];
         double dg_ln[FXAI_AI_MLP_HIDDEN];
         double dc_prev_arr[FXAI_AI_MLP_HIDDEN];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
               g_w_cls[c][h] += err_cls[c] * h_t[t][h];

            g_w_mu[h] += move_scale * err_mu * h_t[t][h];
            g_w_logv[h] += move_scale * err_lv * h_t[t][h];
            g_w_q25[h] += move_scale * err_q25 * h_t[t][h];
            g_w_q75[h] += move_scale * err_q75 * h_t[t][h];

            double tanh_c = FXAI_Tanh(c_t[t][h]);
            double dh = dh_next[h];
            for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
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

            di_ln[h] = FXAI_ClipSym(di, 4.0);
            df_ln[h] = FXAI_ClipSym(df, 4.0);
            do_ln[h] = FXAI_ClipSym(doo, 4.0);
            dg_ln[h] = FXAI_ClipSym(dg, 4.0);
            dc_prev_arr[h] = dc_prev;
         }

         double di_raw[FXAI_AI_MLP_HIDDEN];
         double df_raw[FXAI_AI_MLP_HIDDEN];
         double do_raw[FXAI_AI_MLP_HIDDEN];
         double dg_raw[FXAI_AI_MLP_HIDDEN];

         double zi_hat_t[FXAI_AI_MLP_HIDDEN];
         double zf_hat_t[FXAI_AI_MLP_HIDDEN];
         double zo_hat_t[FXAI_AI_MLP_HIDDEN];
         double zg_hat_t[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            zi_hat_t[h] = zi_hat[t][h];
            zf_hat_t[h] = zf_hat[t][h];
            zo_hat_t[h] = zo_hat[t][h];
            zg_hat_t[h] = zg_hat[t][h];
         }

         LayerNormBackward(di_ln, zi_hat_t, inv_i[t], m_ln_gi, g_ln_gi, g_ln_bi, di_raw);
         LayerNormBackward(df_ln, zf_hat_t, inv_f[t], m_ln_gf, g_ln_gf, g_ln_bf, df_raw);
         LayerNormBackward(do_ln, zo_hat_t, inv_o[t], m_ln_go, g_ln_go, g_ln_bo, do_raw);
         LayerNormBackward(dg_ln, zg_hat_t, inv_g[t], m_ln_gg, g_ln_gg, g_ln_bg, dg_raw);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_bi[h] += di_raw[h];
            g_bf[h] += df_raw[h];
            g_bo[h] += do_raw[h];
            g_bg[h] += dg_raw[h];

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xv = m_batch_x[t][i];
               g_wi_x[h][i] += di_raw[h] * xv;
               g_wf_x[h][i] += df_raw[h] * xv;
               g_wo_x[h][i] += do_raw[h] * xv;
               g_wg_x[h][i] += dg_raw[h] * xv;
            }

            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
            {
               double hpv = h_prev[t][j];
               g_wi_h[h][j] += di_raw[h] * hpv;
               g_wf_h[h][j] += df_raw[h] * hpv;
               g_wo_h[h][j] += do_raw[h] * hpv;
               g_wg_h[h][j] += dg_raw[h] * hpv;

               dh_prev[j] += m_wi_h[h][j] * di_raw[h] +
                             m_wf_h[h][j] * df_raw[h] +
                             m_wo_h[h][j] * do_raw[h] +
                             m_wg_h[h][j] * dg_raw[h];
            }

            dc_next[h] = dc_prev_arr[h];
         }

         if(m_batch_reset[t] > 0.5)
         {
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
            {
               dh_next[j] = 0.0;
               dc_next[j] = 0.0;
            }
         }
         else
         {
            for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
               dh_next[j] = dh_prev[j];
         }
      }

      double gnorm2 = g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
         gnorm2 += g_b_cls[c] * g_b_cls[c];

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         gnorm2 += g_bi[h] * g_bi[h] + g_bf[h] * g_bf[h] + g_bo[h] * g_bo[h] + g_bg[h] * g_bg[h];
         gnorm2 += g_ln_gi[h] * g_ln_gi[h] + g_ln_gf[h] * g_ln_gf[h] +
                   g_ln_go[h] * g_ln_go[h] + g_ln_gg[h] * g_ln_gg[h];
         gnorm2 += g_ln_bi[h] * g_ln_bi[h] + g_ln_bf[h] * g_ln_bf[h] +
                   g_ln_bo[h] * g_ln_bo[h] + g_ln_bg[h] * g_ln_bg[h];
         gnorm2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] +
                   g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            gnorm2 += g_w_cls[c][h] * g_w_cls[c][h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            gnorm2 += g_wi_x[h][i] * g_wi_x[h][i] + g_wf_x[h][i] * g_wf_x[h][i] +
                      g_wo_x[h][i] * g_wo_x[h][i] + g_wg_x[h][i] * g_wg_x[h][i];
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            gnorm2 += g_wi_h[h][j] * g_wi_h[h][j] + g_wf_h[h][j] * g_wf_h[h][j] +
                      g_wo_h[h][j] * g_wo_h[h][j] + g_wg_h[h][j] * g_wg_h[h][j];
         }
      }

      double gnorm = MathSqrt(gnorm2);
      double clip_norm = 3.0;
      if(m_quality_degraded) clip_norm = 2.4;
      if(m_vol_ready && m_vol_ema > 0.0) clip_norm = FXAI_Clamp(clip_norm * (1.0 + 0.03 * m_vol_ema), 2.0, 4.0);
      double gscale = (gnorm > clip_norm ? (clip_norm / MathMax(gnorm, 1e-9)) : 1.0);

      double gmag_gate = 0.0;
      double gmag_cls = 0.0;
      double gmag_move = 0.0;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         gmag_gate += MathAbs(g_bi[h]) + MathAbs(g_bf[h]) + MathAbs(g_bo[h]) + MathAbs(g_bg[h]);
         gmag_gate += MathAbs(g_ln_gi[h]) + MathAbs(g_ln_gf[h]) + MathAbs(g_ln_go[h]) + MathAbs(g_ln_gg[h]);
         gmag_gate += MathAbs(g_ln_bi[h]) + MathAbs(g_ln_bf[h]) + MathAbs(g_ln_bo[h]) + MathAbs(g_ln_bg[h]);
         gmag_move += MathAbs(g_w_mu[h]) + MathAbs(g_w_logv[h]) + MathAbs(g_w_q25[h]) + MathAbs(g_w_q75[h]);

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            gmag_cls += MathAbs(g_w_cls[c][h]);

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            gmag_gate += MathAbs(g_wi_x[h][i]) + MathAbs(g_wf_x[h][i]) +
                         MathAbs(g_wo_x[h][i]) + MathAbs(g_wg_x[h][i]);
         }
         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            gmag_gate += MathAbs(g_wi_h[h][j]) + MathAbs(g_wf_h[h][j]) +
                         MathAbs(g_wo_h[h][j]) + MathAbs(g_wg_h[h][j]);
         }
      }

      double lr_gate = AdamGroupLR(0, gmag_gate, base_lr);
      double lr_cls = AdamGroupLR(1, gmag_cls, base_lr * 0.80);
      double lr_move = AdamGroupLR(2, gmag_move, base_lr * 0.70);

      double wd_gate = FXAI_Clamp(l2 * 0.20, 0.0, 0.20);
      double wd_cls = FXAI_Clamp(l2 * 0.10, 0.0, 0.15);
      double wd_move = FXAI_Clamp(l2 * 0.08, 0.0, 0.12);

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_bi[h] += lr_gate * gscale * g_bi[h];
         m_bf[h] += lr_gate * gscale * g_bf[h];
         m_bo[h] += lr_gate * gscale * g_bo[h];
         m_bg[h] += lr_gate * gscale * g_bg[h];

         m_ln_gi[h] += 0.35 * lr_gate * gscale * g_ln_gi[h];
         m_ln_gf[h] += 0.35 * lr_gate * gscale * g_ln_gf[h];
         m_ln_go[h] += 0.35 * lr_gate * gscale * g_ln_go[h];
         m_ln_gg[h] += 0.35 * lr_gate * gscale * g_ln_gg[h];
         m_ln_bi[h] += 0.35 * lr_gate * gscale * g_ln_bi[h];
         m_ln_bf[h] += 0.35 * lr_gate * gscale * g_ln_bf[h];
         m_ln_bo[h] += 0.35 * lr_gate * gscale * g_ln_bo[h];
         m_ln_bg[h] += 0.35 * lr_gate * gscale * g_ln_bg[h];

         m_w_mu[h] = m_w_mu[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_mu[h];
         m_w_logv[h] = m_w_logv[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_logv[h];
         m_w_q25[h] = m_w_q25[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q25[h];
         m_w_q75[h] = m_w_q75[h] * (1.0 - lr_move * wd_move) + lr_move * gscale * g_w_q75[h];

         for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
            m_w_cls[c][h] = m_w_cls[c][h] * (1.0 - lr_cls * wd_cls) + lr_cls * gscale * g_w_cls[c][h];

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_wi_x[h][i] = m_wi_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wi_x[h][i];
            m_wf_x[h][i] = m_wf_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wf_x[h][i];
            m_wo_x[h][i] = m_wo_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wo_x[h][i];
            m_wg_x[h][i] = m_wg_x[h][i] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wg_x[h][i];
         }

         for(int j=0; j<FXAI_AI_MLP_HIDDEN; j++)
         {
            m_wi_h[h][j] = m_wi_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wi_h[h][j];
            m_wf_h[h][j] = m_wf_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wf_h[h][j];
            m_wo_h[h][j] = m_wo_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wo_h[h][j];
            m_wg_h[h][j] = m_wg_h[h][j] * (1.0 - lr_gate * wd_gate) + lr_gate * gscale * g_wg_h[h][j];
         }
      }

      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         m_b_cls[c] += lr_cls * gscale * g_b_cls[c];
         m_b_cls[c] = FXAI_ClipSym(m_b_cls[c], 6.0);
      }

      m_b_mu += lr_move * gscale * g_b_mu;
      m_b_logv += lr_move * gscale * g_b_logv;
      m_b_q25 += lr_move * gscale * g_b_q25;
      m_b_q75 += lr_move * gscale * g_b_q75;

      SanitizeParams();
      UpdateEMAFromParams(0.995);

      for(int t=0; t<len; t++)
      {
         int cls = m_batch_y[t];
         if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
            cls = (int)FXAI_LABEL_SKIP;

         double sw = FXAI_Clamp(m_batch_w[t], 0.25, 6.00);
         double p_raw[FXAI_LSTM_CLASS_COUNT];
         p_raw[0] = probs[t][0];
         p_raw[1] = probs[t][1];
         p_raw[2] = probs[t][2];

         double cal_lr = base_lr;
         if(m_quality_degraded) cal_lr *= 0.75;
         UpdateCalibrator3(p_raw, cls, sw, cal_lr);

         double p_cal[FXAI_LSTM_CLASS_COUNT];
         Calibrate3(p_raw, p_cal);

         double den = p_cal[(int)FXAI_LABEL_BUY] + p_cal[(int)FXAI_LABEL_SELL];
         if(den < 1e-9) den = 1e-9;
         double p_dir = p_cal[(int)FXAI_LABEL_BUY] / den;
         if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir, 1, sw);
         else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir, 0, sw);

         FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, m_batch_move[t], 0.05);

         double xloc[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xloc[i] = m_batch_x[t][i];
         double ev_raw = ExpectedMoveFromHeads(mu[t], logv[t], q25[t], q75[t], p_cal[(int)FXAI_LABEL_SKIP]);
         double cost = MathMax(0.0, ResolveCostPoints(xloc));
         UpdateValidationMetrics(cls, p_cal, ev_raw - cost);
         UpdateMoveHead(xloc, m_batch_move[t], hp, sw);
      }

      ResetBatch();
   }

public:
   CFXAIAILSTM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_LSTM; }
   virtual string AIName(void) const { return "ai_lstm"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RECURRENT, caps, 16, 128);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_seen_updates = 0;
      m_adam_t = 0;

      m_last_update_time = 0;
      m_last_session_bucket = -1;
      m_last_sample_time = 0;
      m_pending_reset_flag = 0.0;
      m_vol_ready = false;
      m_vol_ema = 0.0;

      m_ema_ready = false;
      m_ema_steps = 0;
      m_lstm_replay_head = 0;
      m_lstm_replay_size = 0;
      m_val_ready = false;
      m_val_steps = 0;
      m_quality_degraded = false;
      m_quality_heads.Reset();

      for(int g=0; g<6; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }

      m_cal3_temp = 1.0;
      m_cal3_steps = 0;
      for(int c=0; c<FXAI_LSTM_CLASS_COUNT; c++)
      {
         m_cal3_bias[c] = 0.0;
         for(int b=0; b<FXAI_LSTM_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      ResetState();
      ResetBatch();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(64, 48);
   }

   void ForwardSequenceContext(const double &x[],
                               const bool use_ema,
                               double &h_out[],
                               double &c_out[]) const
   {
      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_RECURRENT, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildChronologicalSequenceTensorConfigured(x, seq_cfg, seq, seq_len);

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      BuildStateWithPending(h_state, c_state, use_ema);

      for(int t=0; t<seq_len; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         double ig[FXAI_AI_MLP_HIDDEN];
         double fg[FXAI_AI_MLP_HIDDEN];
         double og[FXAI_AI_MLP_HIDDEN];
         double gg[FXAI_AI_MLP_HIDDEN];
         double zi_hat[FXAI_AI_MLP_HIDDEN];
         double zf_hat[FXAI_AI_MLP_HIDDEN];
         double zo_hat[FXAI_AI_MLP_HIDDEN];
         double zg_hat[FXAI_AI_MLP_HIDDEN];
         double inv_i, inv_f, inv_o, inv_g;
         double c_new[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];
         ForwardOne(x_step, h_state, c_state, false, use_ema, 0,
                    zi_hat, zf_hat, zo_hat, zg_hat,
                    inv_i, inv_f, inv_o, inv_g,
                    ig, fg, og, gg, c_new, h_new);
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_state[h] = h_new[h];
            c_state[h] = c_new[h];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         h_out[h] = h_state[h];
         c_out[h] = c_state[h];
      }
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      if(ArraySize(class_probs) < FXAI_LSTM_CLASS_COUNT)
         ArrayResize(class_probs, FXAI_LSTM_CLASS_COUNT);
      bool use_ema = UseEMAInference();

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);

      double logits[FXAI_LSTM_CLASS_COUNT];
      double probs_raw[FXAI_LSTM_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FXAI_LABEL_SKIP]);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.70 * ev + 0.30 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = 0.0;
      if(y == (int)FXAI_LABEL_BUY) pseudo_move = 1.0;
      else if(y == (int)FXAI_LABEL_SELL) pseudo_move = -1.0;
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      bool use_ema = UseEMAInference();

      double h_state[FXAI_AI_MLP_HIDDEN];
      double c_state[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, use_ema, h_state, c_state);
      double logits[FXAI_LSTM_CLASS_COUNT], probs_raw[FXAI_LSTM_CLASS_COUNT];
      double mu, logv, q25, q75;
      ComputeHeads(h_state, use_ema, logits, probs_raw, mu, logv, q25, q75);
      Calibrate3(probs_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, out.class_probs[(int)FXAI_LABEL_SKIP]);
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * (1.0 - out.class_probs[(int)FXAI_LABEL_SKIP]) + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
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

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen_updates++;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      MaybeResetForRegime(xa);

      int cls = NormalizeClassLabel(y, xa, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(x, move_points);
      if(cls == (int)FXAI_LABEL_SKIP) w *= 0.90;
      w = FXAI_Clamp(w, 0.10, 6.00);
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

      datetime t_now = ResolveContextTime();
      if(t_now <= 0) t_now = TimeCurrent();
      double cost_points = MathMax(0.0, ResolveCostPoints(xa));
      int sess = SessionBucket(t_now);

      AppendBatch(cls, xa, move_points, w, m_pending_reset_flag);
      PushReplay(cls, xa, move_points, cost_points, w, t_now, sess);

      if(m_lstm_replay_size >= 64 && m_batch_size < FXAI_LSTM_TBPTT && (m_seen_updates % 6) == 0)
      {
         int add_n = (m_batch_size <= 6 ? 2 : 1);
         for(int a=0; a<add_n; a++)
         {
            int pick = -1;
            double best_score = -1.0;
            for(int tries=0; tries<12; tries++)
            {
               int li = PluginRandIndex(m_lstm_replay_size);
               int rp = ReplayPos(li);
               double rw = ReplayAgeWeight(m_lstm_replay_time[rp], t_now);
               if(m_replay_session[rp] == sess) rw *= 1.20;
               if(m_replay_y[rp] == (int)FXAI_LABEL_SKIP) rw *= 0.95;
               double edge_r = MathAbs(m_lstm_replay_move[rp]) - MathMax(0.0, m_lstm_replay_cost[rp]);
               if(edge_r > 0.0) rw *= (1.0 + 0.04 * MathMin(edge_r, 20.0));
               else rw *= 0.80;
               if(rw > best_score)
               {
                  best_score = rw;
                  pick = rp;
               }
            }
            if(pick >= 0)
            {
               double xr[FXAI_AI_WEIGHTS];
               for(int i=0; i<FXAI_AI_WEIGHTS; i++) xr[i] = m_lstm_replay_x[pick][i];
               double edge_r = MathAbs(m_lstm_replay_move[pick]) - MathMax(0.0, m_lstm_replay_cost[pick]);
               double rw = FXAI_Clamp(m_replay_w[pick] * ReplayAgeWeight(m_lstm_replay_time[pick], t_now), 0.10, 6.00);
               if(edge_r > 0.0) rw *= (1.0 + 0.03 * MathMin(edge_r, 20.0));
               AppendBatch(m_replay_y[pick], xr, m_lstm_replay_move[pick], rw, 1.0);
               if(m_batch_size >= FXAI_LSTM_TBPTT) break;
            }
         }
      }

      if(m_batch_size >= FXAI_LSTM_TBPTT ||
         (m_batch_size >= 4 && (m_seen_updates % (m_quality_degraded ? 2 : 4)) == 0))
      {
         TrainBatch(h);
      }
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return 0.5;

      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[3];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0)
         return ev;

      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};

#endif // __FXAI_AI_LSTM_MQH__
