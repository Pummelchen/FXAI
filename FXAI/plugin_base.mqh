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

   // V4 context payload (set by Train/Predict).
   bool     m_ctx_time_ready;
   datetime m_ctx_time;
   bool     m_ctx_cost_ready;
   double   m_ctx_cost_points;
   double   m_ctx_min_move_points;
   int      m_ctx_regime_id;
   int      m_ctx_session_bucket;
   int      m_ctx_horizon_minutes;
   int      m_ctx_feature_schema_id;
   int      m_ctx_normalization_method_id;
   int      m_ctx_sequence_bars;
   double   m_ctx_point_value;

   double m_bank_class_mass[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS][3];
   double m_bank_total[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];

   int      m_replay_head;
   int      m_replay_size;
   double   m_replay_x[FXAI_PLUGIN_REPLAY_CAPACITY][FXAI_AI_WEIGHTS];
   int      m_replay_label[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_cost[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_min_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   datetime m_replay_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_regime[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_horizon[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_priority[FXAI_PLUGIN_REPLAY_CAPACITY];

   int      m_core_predict_calls;
   int      m_core_predict_failures;
   int      m_expected_prior_calls;
   int      m_replay_rehearsals;

   double InputCostProxyPoints(const double &x[]) const
   {
      if(m_ctx_cost_ready && m_ctx_cost_points >= 0.0)
         return m_ctx_cost_points;

      // Fallback when context has not been set explicitly before a dry-run call.
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

   double ResolvePointValue(void) const
   {
      if(MathIsValidNumber(m_ctx_point_value) && m_ctx_point_value > 0.0)
         return m_ctx_point_value;
      return (_Point > 0.0 ? _Point : 1.0);
   }

   datetime ResolveContextTime(void) const
   {
      if(m_ctx_time_ready && m_ctx_time > 0) return m_ctx_time;
      return TimeCurrent();
   }

   void SetContext(const FXAIAIContextV4 &ctx)
   {
      m_ctx_time_ready = (ctx.sample_time > 0);
      m_ctx_time = (m_ctx_time_ready ? ctx.sample_time : 0);

      m_ctx_cost_ready = (MathIsValidNumber(ctx.cost_points) && ctx.cost_points >= 0.0);
      m_ctx_cost_points = (m_ctx_cost_ready ? ctx.cost_points : 0.0);

      if(MathIsValidNumber(ctx.min_move_points) && ctx.min_move_points > 0.0)
         m_ctx_min_move_points = ctx.min_move_points;
      else
         m_ctx_min_move_points = 0.0;

      if(ctx.regime_id >= 0 && ctx.regime_id < FXAI_PLUGIN_REGIME_BUCKETS)
         m_ctx_regime_id = ctx.regime_id;
      else
         m_ctx_regime_id = 0;

      if(ctx.session_bucket >= 0 && ctx.session_bucket < FXAI_PLUGIN_SESSION_BUCKETS)
         m_ctx_session_bucket = ctx.session_bucket;
      else
         m_ctx_session_bucket = FXAI_DeriveSessionBucket(ctx.sample_time);

      m_ctx_horizon_minutes = (ctx.horizon_minutes > 0 ? ctx.horizon_minutes : 1);
      m_ctx_feature_schema_id = (ctx.feature_schema_id > 0 ? ctx.feature_schema_id : 1);
      m_ctx_normalization_method_id = ctx.normalization_method_id;
      m_ctx_sequence_bars = (ctx.sequence_bars > 0 ? ctx.sequence_bars : 1);
      m_ctx_point_value = (MathIsValidNumber(ctx.point_value) && ctx.point_value > 0.0
                           ? ctx.point_value : (_Point > 0.0 ? _Point : 1.0));
   }

   void SetContext(const datetime sample_time,
                   const double cost_points,
                   const double min_move_points,
                   const int regime_id,
                   const int horizon_minutes)
   {
      FXAIAIContextV4 ctx;
      ctx.api_version = FXAI_API_VERSION_V4;
      ctx.regime_id = regime_id;
      ctx.session_bucket = FXAI_DeriveSessionBucket(sample_time);
      ctx.horizon_minutes = horizon_minutes;
      ctx.feature_schema_id = m_ctx_feature_schema_id;
      ctx.normalization_method_id = m_ctx_normalization_method_id;
      ctx.sequence_bars = m_ctx_sequence_bars;
      ctx.cost_points = cost_points;
      ctx.min_move_points = min_move_points;
      ctx.point_value = m_ctx_point_value;
      ctx.sample_time = sample_time;
      SetContext(ctx);
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
      m_ctx_session_bucket = 0;
      m_ctx_horizon_minutes = 1;
      m_ctx_feature_schema_id = 1;
      m_ctx_normalization_method_id = 0;
      m_ctx_sequence_bars = 1;
      m_ctx_point_value = (_Point > 0.0 ? _Point : 1.0);

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

      m_replay_head = 0;
      m_replay_size = 0;
      for(int i=0; i<FXAI_PLUGIN_REPLAY_CAPACITY; i++)
      {
         m_replay_label[i] = (int)FXAI_LABEL_SKIP;
         m_replay_move[i] = 0.0;
         m_replay_cost[i] = 0.0;
         m_replay_min_move[i] = 0.0;
         m_replay_time[i] = 0;
         m_replay_regime[i] = 0;
         m_replay_horizon[i] = 1;
         m_replay_priority[i] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_x[i][k] = 0.0;
      }

      m_core_predict_calls = 0;
      m_core_predict_failures = 0;
      m_expected_prior_calls = 0;
      m_replay_rehearsals = 0;
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
      if(m_ctx_session_bucket >= 0 && m_ctx_session_bucket < FXAI_PLUGIN_SESSION_BUCKETS)
         return m_ctx_session_bucket;
      return FXAI_DeriveSessionBucket(ResolveContextTime());
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

   void BuildCurrentContext(FXAIAIContextV4 &ctx) const
   {
      ctx.api_version = FXAI_API_VERSION_V4;
      ctx.regime_id = m_ctx_regime_id;
      ctx.session_bucket = ContextSessionBucket();
      ctx.horizon_minutes = m_ctx_horizon_minutes;
      ctx.feature_schema_id = m_ctx_feature_schema_id;
      ctx.normalization_method_id = m_ctx_normalization_method_id;
      ctx.sequence_bars = m_ctx_sequence_bars;
      ctx.cost_points = m_ctx_cost_points;
      ctx.min_move_points = m_ctx_min_move_points;
      ctx.point_value = ResolvePointValue();
      ctx.sample_time = ResolveContextTime();
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

   void StoreReplaySample(const FXAIAITrainRequestV4 &sample,
                          const double priority)
   {
      int slot = m_replay_head;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         m_replay_x[slot][k] = sample.x[k];
      m_replay_label[slot] = sample.label_class;
      m_replay_move[slot] = sample.move_points;
      m_replay_cost[slot] = sample.ctx.cost_points;
      m_replay_min_move[slot] = sample.ctx.min_move_points;
      m_replay_time[slot] = sample.ctx.sample_time;
      m_replay_regime[slot] = sample.ctx.regime_id;
      m_replay_horizon[slot] = sample.ctx.horizon_minutes;
      m_replay_priority[slot] = priority;

      m_replay_head = (m_replay_head + 1) % FXAI_PLUGIN_REPLAY_CAPACITY;
      if(m_replay_size < FXAI_PLUGIN_REPLAY_CAPACITY)
         m_replay_size++;
   }

   void RunReplayRehearsal(const FXAIAIHyperParams &hp,
                           const int regime_id,
                           const int horizon_minutes)
   {
      if(m_replay_size <= 0) return;

      int best_idx[FXAI_PLUGIN_REPLAY_STEPS];
      double best_score[FXAI_PLUGIN_REPLAY_STEPS];
      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         best_idx[j] = -1;
         best_score[j] = -1e18;
      }

      for(int i=0; i<m_replay_size; i++)
      {
         double score = m_replay_priority[i];
         if(m_replay_regime[i] == regime_id) score += 0.80;
         if(m_replay_horizon[i] == horizon_minutes) score += 0.60;

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
      int keep_session = m_ctx_session_bucket;
      int keep_horizon = m_ctx_horizon_minutes;
      int keep_feature_schema = m_ctx_feature_schema_id;
      int keep_norm_method = m_ctx_normalization_method_id;
      int keep_sequence_bars = m_ctx_sequence_bars;
      double keep_point_value = m_ctx_point_value;

      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         int idx = best_idx[j];
         if(idx < 0 || idx >= m_replay_size) continue;
         SetContext(m_replay_time[idx],
                    m_replay_cost[idx],
                    m_replay_min_move[idx],
                    m_replay_regime[idx],
                    m_replay_horizon[idx]);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            replay_x[k] = m_replay_x[idx][k];
         TrainModelCore(m_replay_label[idx], replay_x, hp, m_replay_move[idx]);
         m_replay_rehearsals++;
      }

      m_ctx_time_ready = keep_time_ready;
      m_ctx_time = keep_time;
      m_ctx_cost_ready = keep_cost_ready;
      m_ctx_cost_points = keep_cost;
      m_ctx_min_move_points = keep_min_move;
      m_ctx_regime_id = keep_regime;
      m_ctx_session_bucket = keep_session;
      m_ctx_horizon_minutes = keep_horizon;
      m_ctx_feature_schema_id = keep_feature_schema;
      m_ctx_normalization_method_id = keep_norm_method;
      m_ctx_sequence_bars = keep_sequence_bars;
      m_ctx_point_value = keep_point_value;
   }

   ulong DefaultFeatureGroupsMask(void) const
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

   void FillManifest(FXAIAIManifestV4 &out,
                     const int family,
                     const ulong capability_mask,
                     const int min_sequence_bars,
                     const int max_sequence_bars,
                     const int min_horizon_minutes = 1,
                     const int max_horizon_minutes = 720,
                     const ulong feature_groups_mask = 0,
                     const int feature_schema_id = 1) const
   {
      out.api_version = FXAI_API_VERSION_V4;
      out.ai_id = AIId();
      out.ai_name = AIName();
      out.family = family;
      out.capability_mask = capability_mask;
      out.feature_schema_id = feature_schema_id;
      out.feature_groups_mask = (feature_groups_mask != 0 ? feature_groups_mask : DefaultFeatureGroupsMask());
      out.min_horizon_minutes = min_horizon_minutes;
      out.max_horizon_minutes = max_horizon_minutes;
      out.min_sequence_bars = min_sequence_bars;
      out.max_sequence_bars = max_sequence_bars;
   }

   void ResetModelOutput(FXAIAIModelOutputV4 &out) const
   {
      out.class_probs[0] = 0.10;
      out.class_probs[1] = 0.10;
      out.class_probs[2] = 0.80;
      out.move_mean_points = MathMax(ResolveMinMovePoints(), 0.10);
      out.move_q25_points = 0.0;
      out.move_q50_points = 0.0;
      out.move_q75_points = 0.0;
      out.confidence = 0.0;
      out.reliability = 0.0;
      out.has_quantiles = false;
      out.has_confidence = false;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double move_mean_points = out.move_mean_points;
      if(!PredictModelCore(x, hp, out.class_probs, move_mean_points))
         return false;
      out.move_mean_points = MathMax(move_mean_points, MathMax(ResolveMinMovePoints(), 0.10));
      return true;
   }

   void FillPredictionV4(const FXAIAIModelOutputV4 &model_out,
                         const double calibrated_move_mean_points,
                         FXAIAIPredictionV4 &dst) const
   {
      for(int c=0; c<3; c++)
         dst.class_probs[c] = model_out.class_probs[c];

      double buy_p = dst.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = dst.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = dst.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double uncertainty = FXAI_Clamp(1.0 - directional_conf + 0.50 * skip_p, 0.10, 1.50);
      double mean_move = MathMax(calibrated_move_mean_points, ResolveMinMovePoints());

      dst.move_mean_points = mean_move;
      double raw_mean = MathMax(model_out.move_mean_points, ResolveMinMovePoints());
      double scale = (raw_mean > 1e-9 ? mean_move / raw_mean : 1.0);

      if(model_out.has_quantiles)
      {
         dst.move_q25_points = MathMax(ResolveMinMovePoints(), model_out.move_q25_points * scale);
         dst.move_q50_points = MathMax(dst.move_q25_points, model_out.move_q50_points * scale);
         dst.move_q75_points = MathMax(dst.move_q50_points, model_out.move_q75_points * scale);
      }
      else
      {
         dst.move_q25_points = MathMax(ResolveMinMovePoints(), mean_move * MathMax(0.25, 1.0 - 0.45 * uncertainty));
         dst.move_q50_points = mean_move;
         dst.move_q75_points = MathMax(dst.move_q50_points, mean_move * (1.0 + 0.45 * uncertainty));
      }

      dst.confidence = FXAI_Clamp(model_out.has_confidence ? model_out.confidence : directional_conf, 0.0, 1.0);
      dst.reliability = FXAI_Clamp(model_out.has_confidence ? model_out.reliability : (1.0 - 0.50 * skip_p), 0.0, 1.0);
   }

public:
   CFXAIAIPlugin(void) { ResetAuxState(); }

   virtual int AIId(void) const = 0;
   virtual string AIName(void) const = 0;

   virtual void Reset(void) { ResetAuxState(); }
   virtual void Describe(FXAIAIManifestV4 &out) const = 0;
   virtual void ResetState(const int reason, const datetime when)
   {
      Reset();
   }
   virtual bool SelfTest(void)
   {
      FXAIAIManifestV4 manifest;
      Describe(manifest);
      return (manifest.api_version == FXAI_API_VERSION_V4 &&
              manifest.ai_id == AIId() &&
              StringLen(manifest.ai_name) > 0);
   }
   virtual void EnsureInitialized(const FXAIAIHyperParams &hp) {}

   int CorePredictFailures(void) const { return m_core_predict_failures; }
   int ExpectedPriorCalls(void) const { return m_expected_prior_calls; }
   int ReplayRehearsals(void) const { return m_replay_rehearsals; }

   void Train(const FXAIAITrainRequestV4 &req, const FXAIAIHyperParams &hp)
   {
      if(!req.valid) return;
      EnsureInitialized(hp);
      SetContext(req.ctx);

      FXAIAIManifestV4 manifest;
      Describe(manifest);
      bool can_learn = FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING);
      bool can_replay = can_learn && FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY);
      if(!can_learn)
         return;

      double pre_probs[3];
      pre_probs[0] = 0.10;
      pre_probs[1] = 0.10;
      pre_probs[2] = 0.80;
      double pre_move = MathMax(req.ctx.min_move_points, 0.10);
      bool have_pre = PredictModelCore(req.x, hp, pre_probs, pre_move);
      if(!have_pre)
         m_core_predict_failures++;
      NormalizeClassDistribution(pre_probs);
      pre_move = MathMax(pre_move, MathMax(req.ctx.min_move_points, 0.10));

      double sample_w = (req.sample_weight > 0.0 ? req.sample_weight : MoveSampleWeight(req.x, req.move_points));
      if(have_pre)
         UpdateContextCalibrationBank(req.label_class, pre_probs, pre_move, req.move_points, sample_w);
      double replay_pri = ComputeReplayPriority(req.label_class,
                                                pre_probs,
                                                req.move_points,
                                                req.ctx.cost_points,
                                                req.ctx.min_move_points);
      if(can_replay)
         StoreReplaySample(req, replay_pri);

      TrainModelCore(req.label_class, req.x, hp, req.move_points);
      if(can_replay)
         RunReplayRehearsal(hp, req.ctx.regime_id, req.ctx.horizon_minutes);
   }

   bool Predict(const FXAIAIPredictRequestV4 &req,
                const FXAIAIHyperParams &hp,
                FXAIAIPredictionV4 &out)
   {
      if(!req.valid) return false;
      EnsureInitialized(hp);
      SetContext(req.ctx);

      FXAIAIModelOutputV4 model_out;
      ResetModelOutput(model_out);
      m_core_predict_calls++;
      if(!PredictDistributionCore(req.x, hp, model_out))
      {
         m_core_predict_failures++;
         ResetModelOutput(model_out);
         model_out.class_probs[(int)FXAI_LABEL_SELL] = 0.05;
         model_out.class_probs[(int)FXAI_LABEL_BUY] = 0.05;
         model_out.class_probs[(int)FXAI_LABEL_SKIP] = 0.90;
         model_out.move_mean_points = MathMax(req.ctx.min_move_points, 0.10);
         FillPredictionV4(model_out, model_out.move_mean_points, out);
         return false;
      }

      NormalizeClassDistribution(model_out.class_probs);
      ApplyContextCalibrationBank(model_out.class_probs);
      double calibrated_move = ApplyExpectedMoveCalibrationBank(model_out.move_mean_points);
      FillPredictionV4(model_out, calibrated_move, out);
      return true;
   }

protected:
   virtual bool PredictModelCore(const double &x[],
                                 const FXAIAIHyperParams &hp,
                                 double &class_probs[],
                                 double &move_mean_points) = 0;

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points) = 0;
};

#endif // __FXAI_PLUGIN_BASE_MQH__
