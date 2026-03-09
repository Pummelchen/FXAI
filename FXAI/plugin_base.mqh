#ifndef __FXAI_PLUGIN_BASE_MQH__
#define __FXAI_PLUGIN_BASE_MQH__

#include <Object.mqh>
#include "shared.mqh"

class CFXAIAIPlugin : public CObject
{
protected:
   bool   m_move_ready;
   double m_move_ema_abs;
   bool   m_move_head_ready;
   int    m_move_head_steps;
   double m_move_w[FXAI_AI_WEIGHTS];

   bool   m_cal_ready;
   int    m_cal_steps;
   double m_cal_a;
   double m_cal_b;
   double m_iso_pos[12];
   double m_iso_cnt[12];

   // V2 context payload (set by TrainV2/PredictV2).
   bool     m_ctx_time_ready;
   datetime m_ctx_time;
   bool     m_ctx_cost_ready;
   double   m_ctx_cost_points;
   double   m_ctx_min_move_points;
   int      m_ctx_regime_id;
   int      m_ctx_horizon_minutes;

   double m_bank_class_mass[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS][3];
   double m_bank_total[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];

   int      m_v2_replay_head;
   int      m_v2_replay_size;
   double   m_v2_replay_x[FXAI_PLUGIN_REPLAY_CAPACITY][FXAI_AI_WEIGHTS];
   int      m_v2_replay_label[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_v2_replay_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_v2_replay_cost[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_v2_replay_min_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   datetime m_v2_replay_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_v2_replay_regime[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_v2_replay_horizon[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_v2_replay_priority[FXAI_PLUGIN_REPLAY_CAPACITY];

   int      m_native_predict_calls;
   int      m_native_predict_failures;
   int      m_expected_prior_calls;
   int      m_v2_replay_rehearsals;

   double InputCostProxyPoints(const double &x[]) const
   {
      if(m_ctx_cost_ready && m_ctx_cost_points >= 0.0)
         return m_ctx_cost_points;

      // Fallback for legacy callers where explicit cost is not provided.
      if(ArraySize(x) <= 7) return 0.0;
      return MathMax(0.0, MathAbs(x[7]));
   }

   double ResolveCostPoints(const double &x[]) const
   {
      return InputCostProxyPoints(x);
   }

   double ResolveMinMovePoints(void) const
   {
      if(m_ctx_min_move_points > 0.0) return m_ctx_min_move_points;
      return 0.0;
   }

   datetime ResolveContextTime(void) const
   {
      if(m_ctx_time_ready && m_ctx_time > 0) return m_ctx_time;
      return TimeCurrent();
   }

   void SetContext(const datetime sample_time,
                   const double cost_points,
                   const double min_move_points,
                   const int regime_id,
                   const int horizon_minutes)
   {
      m_ctx_time_ready = (sample_time > 0);
      m_ctx_time = (m_ctx_time_ready ? sample_time : 0);

      m_ctx_cost_ready = (MathIsValidNumber(cost_points) && cost_points >= 0.0);
      m_ctx_cost_points = (m_ctx_cost_ready ? cost_points : 0.0);

      if(MathIsValidNumber(min_move_points) && min_move_points > 0.0)
         m_ctx_min_move_points = min_move_points;
      else
         m_ctx_min_move_points = 0.0;

      if(regime_id >= 0 && regime_id < FXAI_PLUGIN_REGIME_BUCKETS)
         m_ctx_regime_id = regime_id;
      else
         m_ctx_regime_id = 0;

      m_ctx_horizon_minutes = (horizon_minutes > 0 ? horizon_minutes : 1);
   }

   int NormalizeClassLabel(const int y,
                           const double &x[],
                           const double move_points) const
   {
      if(y >= (int)FXAI_LABEL_SELL && y <= (int)FXAI_LABEL_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return (int)FXAI_LABEL_SKIP;

      if(y > 0) return (int)FXAI_LABEL_BUY;
      if(y == 0) return (int)FXAI_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
   }

   void ResetAuxState(void)
   {
      m_move_ready = false;
      m_move_ema_abs = 0.0;

      m_move_head_ready = false;
      m_move_head_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_move_w[i] = 0.0;

      m_cal_ready = false;
      m_cal_steps = 0;
      m_cal_a = 1.0;
      m_cal_b = 0.0;
      for(int i=0; i<12; i++)
      {
         m_iso_pos[i] = 0.0;
         m_iso_cnt[i] = 0.0;
      }

      m_ctx_time_ready = false;
      m_ctx_time = 0;
      m_ctx_cost_ready = false;
      m_ctx_cost_points = 0.0;
      m_ctx_min_move_points = 0.0;
      m_ctx_regime_id = 0;
      m_ctx_horizon_minutes = 1;

      for(int r=0; r<FXAI_PLUGIN_REGIME_BUCKETS; r++)
      {
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         {
            for(int h=0; h<FXAI_PLUGIN_HORIZON_BUCKETS; h++)
            {
               m_bank_total[r][s][h] = 0.0;
               m_bank_ev_scale[r][s][h] = 1.0;
               m_bank_ev_bias[r][s][h] = 0.0;
               m_bank_ev_g2_scale[r][s][h] = 0.0;
               m_bank_ev_g2_bias[r][s][h] = 0.0;
               for(int c=0; c<3; c++)
                  m_bank_class_mass[r][s][h][c] = (c == (int)FXAI_LABEL_SKIP ? 1.2 : 1.0);
            }
         }
      }

      m_v2_replay_head = 0;
      m_v2_replay_size = 0;
      for(int i=0; i<FXAI_PLUGIN_REPLAY_CAPACITY; i++)
      {
         m_v2_replay_label[i] = (int)FXAI_LABEL_SKIP;
         m_v2_replay_move[i] = 0.0;
         m_v2_replay_cost[i] = 0.0;
         m_v2_replay_min_move[i] = 0.0;
         m_v2_replay_time[i] = 0;
         m_v2_replay_regime[i] = 0;
         m_v2_replay_horizon[i] = 1;
         m_v2_replay_priority[i] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_v2_replay_x[i][k] = 0.0;
      }

      m_native_predict_calls = 0;
      m_native_predict_failures = 0;
      m_expected_prior_calls = 0;
      m_v2_replay_rehearsals = 0;
   }

   void UpdateMoveHead(const double &x[],
                       const double move_points,
                       const FXAIAIHyperParams &hp,
                       const double sample_w)
   {
      double tgt = MathAbs(move_points);
      if(!MathIsValidNumber(tgt)) return;

      double pred = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         pred += m_move_w[i] * x[i];
      if(pred < 0.0) pred = 0.0;

      double err = FXAI_ClipSym(tgt - pred, 20.0);
      double lr = FXAI_Clamp(0.08 * hp.lr, 0.00005, 0.02000);
      double l2 = FXAI_Clamp(0.25 * hp.l2, 0.0000, 0.1000);
      double w = FXAI_Clamp(sample_w, 0.25, 4.00);

      m_move_w[0] += lr * w * err;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         m_move_w[i] += lr * (w * FXAI_ClipSym(err * x[i], 6.0) - l2 * m_move_w[i]);

      m_move_head_steps++;
      if(m_move_head_steps >= 16) m_move_head_ready = true;
   }

   double PredictMoveHeadRaw(const double &x[]) const
   {
      double p = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         p += m_move_w[i] * x[i];
      if(p < 0.0) p = 0.0;
      return p;
   }

   void UpdateCalibration(const double prob_raw,
                          const int y,
                          const double sample_w = 1.0)
   {
      double pr = FXAI_Clamp(prob_raw, 0.001, 0.999);
      double z = FXAI_Logit(pr);
      double py = FXAI_Sigmoid((m_cal_a * z) + m_cal_b);
      double e = ((double)y - py);

      double w = FXAI_Clamp(sample_w, 0.25, 4.00);
      double lr = 0.015 * w;
      double reg = 0.0005;

      m_cal_a += lr * (e * z - reg * (m_cal_a - 1.0));
      m_cal_b += lr * (e);
      m_cal_a = FXAI_Clamp(m_cal_a, 0.20, 5.00);
      m_cal_b = FXAI_Clamp(m_cal_b, -4.0, 4.0);

      int bins = 12;
      int bi = (int)MathFloor(pr * (double)bins);
      if(bi < 0) bi = 0;
      if(bi >= bins) bi = bins - 1;
      m_iso_cnt[bi] += w;
      m_iso_pos[bi] += w * (double)y;

      m_cal_steps++;
      if(m_cal_steps >= 20) m_cal_ready = true;
   }

   double CalibrateProb(const double prob_raw) const
   {
      double pr = FXAI_Clamp(prob_raw, 0.001, 0.999);
      double p_platt = FXAI_Sigmoid((m_cal_a * FXAI_Logit(pr)) + m_cal_b);
      if(!m_cal_ready)
         return FXAI_Clamp(p_platt, 0.001, 0.999);

      double total = 0.0;
      for(int i=0; i<12; i++) total += m_iso_cnt[i];
      if(total < 30.0)
         return FXAI_Clamp(p_platt, 0.001, 0.999);

      int bins = 12;
      int bi = (int)MathFloor(pr * (double)bins);
      if(bi < 0) bi = 0;
      if(bi >= bins) bi = bins - 1;

      double mono[12];
      double prev = 0.5;
      for(int i=0; i<12; i++)
      {
         double r = prev;
         if(m_iso_cnt[i] > 1e-9)
            r = m_iso_pos[i] / m_iso_cnt[i];
         r = FXAI_Clamp(r, 0.01, 0.99);
         if(i > 0 && r < mono[i - 1]) r = mono[i - 1];
         mono[i] = r;
         prev = r;
      }

      double p_iso = mono[bi];
      double p = (0.70 * p_platt) + (0.30 * p_iso);
      return FXAI_Clamp(p, 0.001, 0.999);
   }

   FXAIAIHyperParams ScaleHyperParamsForMove(const FXAIAIHyperParams &hp,
                                            const double move_points) const
   {
      FXAIAIHyperParams h = hp;
      double w = FXAI_MoveWeight(move_points);

      h.lr *= w;
      h.ftrl_alpha *= w;
      h.xgb_lr *= w;
      h.mlp_lr *= w;
      h.quantile_lr *= w;
      h.enhash_lr *= w;

      h.lr = FXAI_Clamp(h.lr, 0.0001, 1.0000);
      h.ftrl_alpha = FXAI_Clamp(h.ftrl_alpha, 0.0001, 5.0000);
      h.xgb_lr = FXAI_Clamp(h.xgb_lr, 0.0001, 1.0000);
      h.mlp_lr = FXAI_Clamp(h.mlp_lr, 0.0001, 1.0000);
      h.quantile_lr = FXAI_Clamp(h.quantile_lr, 0.0001, 1.0000);
      h.enhash_lr = FXAI_Clamp(h.enhash_lr, 0.0001, 1.0000);
      return h;
   }

   double MoveSampleWeight(const double &x[],
                           const double move_points) const
   {
      double cost = ResolveCostPoints(x);
      return FXAI_MoveEdgeWeight(move_points, cost);
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

   int ContextSessionBucket(void) const
   {
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      int hour = dt.hour;
      if(hour < 0) hour = 0;
      if(hour > 23) hour = 23;
      int bucket = hour / 4;
      if(bucket < 0) bucket = 0;
      if(bucket >= FXAI_PLUGIN_SESSION_BUCKETS) bucket = FXAI_PLUGIN_SESSION_BUCKETS - 1;
      return bucket;
   }

   int ContextHorizonBucket(void) const
   {
      int h = (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1);
      if(h <= 1) return 0;
      if(h <= 3) return 1;
      if(h <= 5) return 2;
      if(h <= 8) return 3;
      if(h <= 13) return 4;
      if(h <= 21) return 5;
      if(h <= 34) return 6;
      return FXAI_PLUGIN_HORIZON_BUCKETS - 1;
   }

   void NormalizeClassDistribution(double &probs[]) const
   {
      for(int c=0; c<3; c++)
      {
         if(!MathIsValidNumber(probs[c])) probs[c] = 0.0;
         probs[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      }

      double s = probs[0] + probs[1] + probs[2];
      if(!MathIsValidNumber(s) || s <= 0.0)
      {
         probs[0] = 0.10;
         probs[1] = 0.10;
         probs[2] = 0.80;
         return;
      }

      for(int c=0; c<3; c++)
         probs[c] /= s;
   }

   double ExpectedMovePrior(const double &x[])
   {
      m_expected_prior_calls++;
      double head = (m_move_head_ready ? PredictMoveHeadRaw(x) : -1.0);
      if(head > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.60 * head + 0.40 * m_move_ema_abs;
      if(head > 0.0) return head;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return MathMax(ResolveMinMovePoints(), 0.10);
   }

   void ApplyContextCalibrationBank(double &probs[])
   {
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();

      double total = m_bank_total[r][s][h];
      if(total <= 0.0)
      {
         NormalizeClassDistribution(probs);
         return;
      }

      double prior[3];
      for(int c=0; c<3; c++)
         prior[c] = m_bank_class_mass[r][s][h][c] / MathMax(total, 1e-9);

      double mix = FXAI_Clamp(total / 120.0, 0.05, 0.35);
      for(int c=0; c<3; c++)
         probs[c] = (1.0 - mix) * probs[c] + mix * prior[c];

      NormalizeClassDistribution(probs);
   }

   double ApplyExpectedMoveCalibrationBank(const double expected_move_points)
   {
      double ev = (expected_move_points > 0.0 ? expected_move_points : MathMax(ResolveMinMovePoints(), 0.10));
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();

      ev = ev * m_bank_ev_scale[r][s][h] + m_bank_ev_bias[r][s][h];
      if(!MathIsValidNumber(ev) || ev <= 0.0)
         ev = MathMax(ResolveMinMovePoints(), 0.10);
      return ev;
   }

   void UpdateContextCalibrationBank(const int label_class,
                                     const double &probs[],
                                     const double expected_move_points,
                                     const double move_points,
                                     const double sample_w)
   {
      if(label_class < (int)FXAI_LABEL_SELL || label_class > (int)FXAI_LABEL_SKIP)
         return;

      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      double w = FXAI_Clamp(sample_w, 0.25, 4.00);

      m_bank_class_mass[r][s][h][label_class] += w;
      m_bank_total[r][s][h] += w;
      if(m_bank_total[r][s][h] > 30000.0)
      {
         for(int c=0; c<3; c++)
            m_bank_class_mass[r][s][h][c] *= 0.5;
         m_bank_total[r][s][h] *= 0.5;
      }

      double pred = MathMax(expected_move_points, MathMax(ResolveMinMovePoints(), 0.10));
      double tgt = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      double err = FXAI_ClipSym(tgt - pred, 30.0);

      double lr = 0.015 * w;
      double g_scale = err / MathMax(pred, 0.25);
      double g_bias = 0.30 * err;

      m_bank_ev_g2_scale[r][s][h] += g_scale * g_scale;
      m_bank_ev_g2_bias[r][s][h] += g_bias * g_bias;

      double lr_scale = lr / MathSqrt(m_bank_ev_g2_scale[r][s][h] + 1e-8);
      double lr_bias = lr / MathSqrt(m_bank_ev_g2_bias[r][s][h] + 1e-8);

      m_bank_ev_scale[r][s][h] += lr_scale * g_scale;
      m_bank_ev_bias[r][s][h] += lr_bias * g_bias;

      m_bank_ev_scale[r][s][h] = FXAI_Clamp(m_bank_ev_scale[r][s][h], 0.40, 2.50);
      m_bank_ev_bias[r][s][h] = FXAI_Clamp(m_bank_ev_bias[r][s][h], -20.0, 20.0);
   }

   double ComputeReplayPriority(const int label_class,
                                const double &probs[],
                                const double move_points,
                                const double cost_points,
                                const double min_move_points) const
   {
      int cls = label_class;
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);

      double p_true = (cls >= 0 && cls < 3 ? probs[cls] : 0.3333333);
      double edge = MathMax(MathAbs(move_points) - MathMax(cost_points, 0.0), 0.0);
      double mm = MathMax(min_move_points, 0.10);
      double pri = 0.50 + (1.0 - FXAI_Clamp(p_true, 0.0, 1.0));
      pri += 0.35 * FXAI_Clamp(edge / mm, 0.0, 4.0);
      if(cls == (int)FXAI_LABEL_SKIP)
         pri += 0.15;
      return FXAI_Clamp(pri, 0.10, 8.00);
   }

   void StoreReplaySample(const FXAIAISampleV2 &sample,
                          const double priority)
   {
      int slot = m_v2_replay_head;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         m_v2_replay_x[slot][k] = sample.x[k];
      m_v2_replay_label[slot] = sample.label_class;
      m_v2_replay_move[slot] = sample.move_points;
      m_v2_replay_cost[slot] = sample.cost_points;
      m_v2_replay_min_move[slot] = sample.min_move_points;
      m_v2_replay_time[slot] = sample.sample_time;
      m_v2_replay_regime[slot] = sample.regime_id;
      m_v2_replay_horizon[slot] = sample.horizon_minutes;
      m_v2_replay_priority[slot] = priority;

      m_v2_replay_head = (m_v2_replay_head + 1) % FXAI_PLUGIN_REPLAY_CAPACITY;
      if(m_v2_replay_size < FXAI_PLUGIN_REPLAY_CAPACITY)
         m_v2_replay_size++;
   }

   void RunReplayRehearsal(const FXAIAIHyperParams &hp,
                           const int regime_id,
                           const int horizon_minutes)
   {
      if(m_v2_replay_size <= 0) return;

      int best_idx[FXAI_PLUGIN_REPLAY_STEPS];
      double best_score[FXAI_PLUGIN_REPLAY_STEPS];
      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         best_idx[j] = -1;
         best_score[j] = -1e18;
      }

      for(int i=0; i<m_v2_replay_size; i++)
      {
         double score = m_v2_replay_priority[i];
         if(m_v2_replay_regime[i] == regime_id) score += 0.80;
         if(m_v2_replay_horizon[i] == horizon_minutes) score += 0.60;

         for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
         {
            if(score > best_score[j])
            {
               for(int k=FXAI_PLUGIN_REPLAY_STEPS - 1; k>j; k--)
               {
                  best_score[k] = best_score[k - 1];
                  best_idx[k] = best_idx[k - 1];
               }
               best_score[j] = score;
               best_idx[j] = i;
               break;
            }
         }
      }

      datetime keep_time = m_ctx_time;
      bool keep_time_ready = m_ctx_time_ready;
      double keep_cost = m_ctx_cost_points;
      bool keep_cost_ready = m_ctx_cost_ready;
      double keep_min_move = m_ctx_min_move_points;
      int keep_regime = m_ctx_regime_id;
      int keep_horizon = m_ctx_horizon_minutes;

      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         int idx = best_idx[j];
         if(idx < 0 || idx >= m_v2_replay_size) continue;
         SetContext(m_v2_replay_time[idx],
                    m_v2_replay_cost[idx],
                    m_v2_replay_min_move[idx],
                    m_v2_replay_regime[idx],
                    m_v2_replay_horizon[idx]);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            replay_x[k] = m_v2_replay_x[idx][k];
         UpdateWithMove(m_v2_replay_label[idx], replay_x, hp, m_v2_replay_move[idx]);
         m_v2_replay_rehearsals++;
      }

      m_ctx_time_ready = keep_time_ready;
      m_ctx_time = keep_time;
      m_ctx_cost_ready = keep_cost_ready;
      m_ctx_cost_points = keep_cost;
      m_ctx_min_move_points = keep_min_move;
      m_ctx_regime_id = keep_regime;
      m_ctx_horizon_minutes = keep_horizon;
   }

   int DefaultFamilyV3(void) const
   {
      switch(AIId())
      {
         case (int)AI_SGD_LOGIT:
         case (int)AI_FTRL_LOGIT:
         case (int)AI_PA_LINEAR:
         case (int)AI_ENHASH:
            return (int)FXAI_FAMILY_LINEAR;

         case (int)AI_XGB_FAST:
         case (int)AI_XGBOOST:
         case (int)AI_LIGHTGBM:
         case (int)AI_CATBOOST:
            return (int)FXAI_FAMILY_TREE;

         case (int)AI_LSTM:
         case (int)AI_LSTMG:
            return (int)FXAI_FAMILY_RECURRENT;

         case (int)AI_TCN:
            return (int)FXAI_FAMILY_CONVOLUTIONAL;

         case (int)AI_S4:
            return (int)FXAI_FAMILY_STATE_SPACE;

         case (int)AI_QUANTILE:
            return (int)FXAI_FAMILY_DISTRIBUTIONAL;

         case (int)AI_LOFFM:
         case (int)AI_MOE_CONFORMAL:
            return (int)FXAI_FAMILY_MIXTURE;

         case (int)AI_RETRDIFF:
            return (int)FXAI_FAMILY_RETRIEVAL;

         case (int)AI_CFX_WORLD:
         case (int)AI_GRAPHWM:
            return (int)FXAI_FAMILY_WORLD_MODEL;

         case (int)AI_M1SYNC:
            return (int)FXAI_FAMILY_RULE_BASED;

         case (int)AI_AUTOFORMER:
         case (int)AI_CHRONOS:
         case (int)AI_GEODESICATTENTION:
         case (int)AI_PATCHTST:
         case (int)AI_STMN:
         case (int)AI_TFT:
         case (int)AI_TIMESFM:
         case (int)AI_TRR:
         case (int)AI_TST:
            return (int)FXAI_FAMILY_TRANSFORMER;

         default:
            return (int)FXAI_FAMILY_OTHER;
      }
   }

   ulong DefaultFeatureGroupsMaskV3(void) const
   {
      ulong mask = 0;
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_PRICE);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_VOLATILITY);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_TIME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_CONTEXT);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_COST);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_FILTERS);
      return mask;
   }

   void FillDefaultManifestV3(FXAIAIManifestV3 &out) const
   {
      out.api_version = FXAI_API_VERSION_V3;
      out.ai_id = AIId();
      out.ai_name = AIName();
      out.family = DefaultFamilyV3();
      out.supports_native_3class = SupportsNativeClassProbs();
      out.supports_distributional_move = true;
      out.supports_online_learning = true;
      out.supports_replay = true;
      out.supports_state = (out.family == (int)FXAI_FAMILY_RECURRENT ||
                            out.family == (int)FXAI_FAMILY_CONVOLUTIONAL ||
                            out.family == (int)FXAI_FAMILY_TRANSFORMER ||
                            out.family == (int)FXAI_FAMILY_STATE_SPACE ||
                            out.family == (int)FXAI_FAMILY_WORLD_MODEL);
      out.supports_window_context = out.supports_state;
      out.supports_multi_horizon = true;
      out.feature_schema_id = 1;
      out.feature_groups_mask = DefaultFeatureGroupsMaskV3();
      out.min_horizon_minutes = 1;
      out.max_horizon_minutes = 720;
   }

   void FillPredictionV3(const FXAIAIPredictionV2 &src,
                         FXAIAIPredictionV3 &dst) const
   {
      for(int c=0; c<3; c++)
         dst.class_probs[c] = src.class_probs[c];

      double buy_p = dst.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = dst.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = dst.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double uncertainty = FXAI_Clamp(1.0 - directional_conf + 0.50 * skip_p, 0.10, 1.50);
      double mean_move = MathMax(src.expected_move_points, ResolveMinMovePoints());

      dst.move_mean_points = mean_move;
      dst.move_q25_points = MathMax(ResolveMinMovePoints(), mean_move * MathMax(0.25, 1.0 - 0.45 * uncertainty));
      dst.move_q75_points = MathMax(dst.move_q25_points, mean_move * (1.0 + 0.45 * uncertainty));
      dst.confidence = FXAI_Clamp(directional_conf, 0.0, 1.0);
      dst.calibration_confidence = FXAI_Clamp(1.0 - 0.50 * skip_p, 0.0, 1.0);
   }

public:
   CFXAIAIPlugin(void) { ResetAuxState(); }

   virtual int AIId(void) const = 0;
   virtual string AIName(void) const = 0;
   virtual int APIVersion(void) const { return FXAI_API_VERSION_V3; }

   virtual void Reset(void) { ResetAuxState(); }
    virtual void DescribeV3(FXAIAIManifestV3 &out) const { FillDefaultManifestV3(out); }
   virtual void ResetStateV3(const int reason, const datetime when)
   {
      Reset();
   }
   virtual bool SelfTestV3(void)
   {
      FXAIAIManifestV3 manifest;
      DescribeV3(manifest);
      return (manifest.api_version == FXAI_API_VERSION_V3 && manifest.supports_native_3class);
   }
   virtual void EnsureInitialized(const FXAIAIHyperParams &hp) {}
   virtual bool SupportsNativeClassProbs(void) const { return false; }
   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      return false;
   }

   int NativePredictFailures(void) const { return m_native_predict_failures; }
   int ExpectedPriorCalls(void) const { return m_expected_prior_calls; }
   int ReplayRehearsals(void) const { return m_v2_replay_rehearsals; }

   // V2 training API: time/cost-aware sample payload.
   void TrainV2(const FXAIAISampleV2 &sample, const FXAIAIHyperParams &hp)
   {
      if(!sample.valid) return;
      EnsureInitialized(hp);
      SetContext(sample.sample_time,
                 sample.cost_points,
                 sample.min_move_points,
                 sample.regime_id,
                 sample.horizon_minutes);

      double pre_probs[3];
      pre_probs[0] = 0.10;
      pre_probs[1] = 0.10;
      pre_probs[2] = 0.80;
      double pre_move = MathMax(sample.min_move_points, 0.10);
      bool have_pre = PredictNativeClassProbs(sample.x, hp, pre_probs, pre_move);
      if(!have_pre)
         m_native_predict_failures++;
      NormalizeClassDistribution(pre_probs);
      pre_move = MathMax(pre_move, MathMax(sample.min_move_points, 0.10));

      double sample_w = MoveSampleWeight(sample.x, sample.move_points);
      UpdateContextCalibrationBank(sample.label_class, pre_probs, pre_move, sample.move_points, sample_w);
      double replay_pri = ComputeReplayPriority(sample.label_class,
                                                pre_probs,
                                                sample.move_points,
                                                sample.cost_points,
                                                sample.min_move_points);
      StoreReplaySample(sample, replay_pri);

      UpdateWithMove(sample.label_class, sample.x, hp, sample.move_points);
      RunReplayRehearsal(hp, sample.regime_id, sample.horizon_minutes);
   }

   // V2 inference API: returns calibrated 3-class distribution.
   void PredictV2(const FXAIAIPredictV2 &req,
                  const FXAIAIHyperParams &hp,
      FXAIAIPredictionV2 &out)
   {
      EnsureInitialized(hp);
      SetContext(req.sample_time,
                 req.cost_points,
                 req.min_move_points,
                 req.regime_id,
                 req.horizon_minutes);

      double native_probs[3];
      native_probs[0] = 0.10;
      native_probs[1] = 0.10;
      native_probs[2] = 0.80;
      double native_move = MathMax(req.min_move_points, 0.10);
      m_native_predict_calls++;
      if(!PredictNativeClassProbs(req.x, hp, native_probs, native_move))
      {
         m_native_predict_failures++;
         out.class_probs[(int)FXAI_LABEL_SELL] = 0.05;
         out.class_probs[(int)FXAI_LABEL_BUY] = 0.05;
         out.class_probs[(int)FXAI_LABEL_SKIP] = 0.90;
         out.p_up = out.class_probs[(int)FXAI_LABEL_BUY];
         out.expected_move_points = MathMax(req.min_move_points, 0.10);
         return;
      }

      NormalizeClassDistribution(native_probs);
      ApplyContextCalibrationBank(native_probs);
      out.class_probs[0] = native_probs[0];
      out.class_probs[1] = native_probs[1];
      out.class_probs[2] = native_probs[2];
      out.p_up = out.class_probs[(int)FXAI_LABEL_BUY];
      out.expected_move_points = ApplyExpectedMoveCalibrationBank(native_move);
   }

   void TrainV3(const FXAIAITrainRequestV3 &req, const FXAIAIHyperParams &hp)
   {
      FXAIAISampleV2 sample;
      sample.valid = true;
      sample.label_class = req.label_class;
      sample.regime_id = req.ctx.regime_id;
      sample.horizon_minutes = req.ctx.horizon_minutes;
      sample.move_points = req.move_points;
      sample.min_move_points = req.ctx.min_move_points;
      sample.cost_points = req.ctx.cost_points;
      sample.sample_time = req.ctx.sample_time;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         sample.x[i] = req.x[i];
      TrainV2(sample, hp);
   }

   bool PredictV3(const FXAIAIPredictRequestV3 &req,
                  const FXAIAIHyperParams &hp,
                  FXAIAIPredictionV3 &out)
   {
      FXAIAIPredictV2 req_v2;
      req_v2.regime_id = req.ctx.regime_id;
      req_v2.horizon_minutes = req.ctx.horizon_minutes;
      req_v2.min_move_points = req.ctx.min_move_points;
      req_v2.cost_points = req.ctx.cost_points;
      req_v2.sample_time = req.ctx.sample_time;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         req_v2.x[i] = req.x[i];

      FXAIAIPredictionV2 out_v2;
      PredictV2(req_v2, hp, out_v2);
      FillPredictionV3(out_v2, out);
      return true;
   }

protected:
   // Legacy model hooks (v1 core), kept protected to retire external v1 usage.
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points) = 0;
   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp) = 0;
   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      return ExpectedMovePrior(x);
   }
};

#endif // __FXAI_PLUGIN_BASE_MQH__
