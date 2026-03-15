#ifndef __FXAI_PLUGIN_BASE_MQH__
#define __FXAI_PLUGIN_BASE_MQH__

#include <Object.mqh>
#include "..\Engine\core.mqh"
#include "..\TensorCore\TensorCore.mqh"

class CFXAITernaryCalibrator
{
private:
   enum { CAL_BINS = 12 };

   int    m_steps;
   double m_w[3][3];
   double m_b[3];
   double m_iso_pos[3][CAL_BINS];
   double m_iso_cnt[3][CAL_BINS];

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double mx = logits[0];
      for(int c=1; c<3; c++)
         if(logits[c] > mx) mx = logits[c];

      double den = 0.0;
      for(int c=0; c<3; c++)
      {
         probs[c] = MathExp(FXAI_ClipSym(logits[c] - mx, 30.0));
         den += probs[c];
      }
      if(den <= 0.0) den = 1.0;
      for(int c=0; c<3; c++)
         probs[c] /= den;
   }

   void BuildCalLogits(const double &p_raw[], double &logits[]) const
   {
      double lraw[3];
      for(int c=0; c<3; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      for(int c=0; c<3; c++)
      {
         double z = m_b[c];
         for(int j=0; j<3; j++)
            z += m_w[c][j] * lraw[j];
         logits[c] = z;
      }
   }

public:
   CFXAITernaryCalibrator(void) { Reset(); }

   void Reset(void)
   {
      m_steps = 0;
      for(int c=0; c<3; c++)
      {
         m_b[c] = 0.0;
         for(int j=0; j<3; j++)
            m_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<CAL_BINS; b++)
         {
            m_iso_pos[c][b] = 0.0;
            m_iso_cnt[c][b] = 0.0;
         }
      }
   }

   void Calibrate(const double &p_raw[], double &p_cal[]) const
   {
      double logits[3];
      BuildCalLogits(p_raw, logits);
      Softmax3(logits, p_cal);

      if(m_steps < 30)
         return;

      double p_iso[3];
      for(int c=0; c<3; c++)
      {
         double total = 0.0;
         for(int b=0; b<CAL_BINS; b++)
            total += m_iso_cnt[c][b];
         if(total < 40.0)
         {
            p_iso[c] = p_cal[c];
            continue;
         }

         double mono[CAL_BINS];
         double prev = 0.01;
         for(int b=0; b<CAL_BINS; b++)
         {
            double r = prev;
            if(m_iso_cnt[c][b] > 1e-9)
               r = m_iso_pos[c][b] / m_iso_cnt[c][b];
            r = FXAI_Clamp(r, 0.001, 0.999);
            if(r < prev) r = prev;
            mono[b] = r;
            prev = r;
         }

         int bi = (int)MathFloor(p_cal[c] * (double)CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= CAL_BINS) bi = CAL_BINS - 1;
         p_iso[c] = mono[bi];
      }

      for(int c=0; c<3; c++)
         p_cal[c] = FXAI_Clamp(0.75 * p_cal[c] + 0.25 * p_iso[c], 0.0005, 0.9990);

      double s = p_cal[0] + p_cal[1] + p_cal[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++)
         p_cal[c] /= s;
   }

   void Update(const double &p_raw[],
               const int cls,
               const double sample_w,
               const double lr)
   {
      double logits[3];
      BuildCalLogits(p_raw, logits);

      double p_cal[3];
      Softmax3(logits, p_cal);

      double lraw[3];
      for(int c=0; c<3; c++)
         lraw[c] = MathLog(FXAI_Clamp(p_raw[c], 0.0005, 0.9990));

      double w = FXAI_Clamp(sample_w, 0.20, 8.00);
      double cal_lr = FXAI_Clamp(0.25 * lr * w, 0.0002, 0.0200);
      double reg_l2 = 0.0005;

      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double e = target - p_cal[c];

         m_b[c] = FXAI_ClipSym(m_b[c] + cal_lr * e, 4.0);
         for(int j=0; j<3; j++)
         {
            double target_w = (c == j ? 1.0 : 0.0);
            double grad = e * lraw[j] - reg_l2 * (m_w[c][j] - target_w);
            m_w[c][j] = FXAI_ClipSym(m_w[c][j] + cal_lr * grad, 4.0);
         }

         int bi = (int)MathFloor(p_cal[c] * (double)CAL_BINS);
         if(bi < 0) bi = 0;
         if(bi >= CAL_BINS) bi = CAL_BINS - 1;
         m_iso_cnt[c][bi] += w;
         m_iso_pos[c][bi] += w * target;
      }
      m_steps++;
   }

   int Steps(void) const { return m_steps; }
};

class CFXAINativeQualityHeads
{
private:
   bool   m_ready;
   int    m_steps;
   double m_w_mfe[FXAI_AI_WEIGHTS];
   double m_w_mae[FXAI_AI_WEIGHTS];
   double m_w_hit[FXAI_AI_WEIGHTS];
   double m_w_path[FXAI_AI_WEIGHTS];
   double m_w_fill[FXAI_AI_WEIGHTS];
   double m_b_mfe;
   double m_b_mae;
   double m_b_hit;
   double m_b_path;
   double m_b_fill;
   double m_w_mask[FXAI_AI_WEIGHTS];
   double m_w_vol[FXAI_AI_WEIGHTS];
   double m_w_shift[FXAI_AI_WEIGHTS];
   double m_w_ctx[FXAI_AI_WEIGHTS];
   double m_b_mask;
   double m_b_vol;
   double m_b_shift;
   double m_b_ctx;

   double Dot(const double &w[],
              const double &x[]) const
   {
      double z = w[0];
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         z += w[i] * x[i];
      return z;
   }

   void UpdateHead(double &w[],
                   double &bias,
                   const double &x[],
                   const double pred,
                   const double target,
                   const double lr,
                   const double l2)
   {
      double e = FXAI_Clamp(-0.5 * FXAI_LossMSEGrad(pred, target), -12.0, 12.0);
      bias = FXAI_ClipSym(bias + lr * e, 12.0);
      w[0] = bias;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         w[i] = FXAI_ClipSym(w[i] + lr * (e * x[i] - l2 * w[i]), 12.0);
   }

public:
   CFXAINativeQualityHeads(void) { Reset(); }

   void Reset(void)
   {
      m_ready = false;
      m_steps = 0;
      m_b_mfe = 1.0;
      m_b_mae = -1.5;
      m_b_hit = 0.0;
      m_b_path = -0.5;
      m_b_fill = -0.5;
      m_b_mask = 0.0;
      m_b_vol = -1.0;
      m_b_shift = -1.2;
      m_b_ctx = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_w_mfe[i] = 0.0;
         m_w_mae[i] = 0.0;
         m_w_hit[i] = 0.0;
         m_w_path[i] = 0.0;
         m_w_fill[i] = 0.0;
         m_w_mask[i] = 0.0;
         m_w_vol[i] = 0.0;
         m_w_shift[i] = 0.0;
         m_w_ctx[i] = 0.0;
      }
      m_w_mfe[0] = m_b_mfe;
      m_w_mae[0] = m_b_mae;
      m_w_hit[0] = m_b_hit;
      m_w_path[0] = m_b_path;
      m_w_fill[0] = m_b_fill;
      m_w_mask[0] = m_b_mask;
      m_w_vol[0] = m_b_vol;
      m_w_shift[0] = m_b_shift;
      m_w_ctx[0] = m_b_ctx;
   }

   bool Ready(void) const { return m_ready; }
   int Steps(void) const { return m_steps; }

   void Predict(const double &x[],
                const double move_scale,
                const double activity_gate,
                const double structural_quality,
                const double execution_quality,
                const double bank_mfe,
                const double bank_mae,
                const double bank_hit,
                const double bank_path,
                const double bank_fill,
                const double bank_trust,
                FXAIAIModelOutputV4 &out) const
   {
      double active = FXAI_Clamp(activity_gate, 0.0, 1.0);
      double structure = FXAI_Clamp(structural_quality, 0.0, 1.0);
      double exec_q = FXAI_Clamp(execution_quality, 0.0, 1.0);
      double scale = MathMax(move_scale, 0.10);
      double self_trust = FXAI_Clamp((double)m_steps / 120.0, 0.0, 0.75);
      double mix = MathMax(bank_trust, self_trust);

      double mask_self = FXAI_ClipSym(Dot(m_w_mask, x), 8.0);
      double vol_self = MathExp(FXAI_ClipSym(Dot(m_w_vol, x), 4.0));
      double shift_self = FXAI_Clamp(FXAI_Sigmoid(Dot(m_w_shift, x)), 0.0, 1.0);
      double ctx_self = FXAI_Clamp(FXAI_Sigmoid(Dot(m_w_ctx, x)), 0.0, 1.0);
      double aux_vol_ratio = FXAI_Clamp(vol_self / MathMax(scale, 0.10), 0.0, 2.0);
      structure = FXAI_Clamp(structure +
                             0.10 * (1.0 - shift_self) +
                             0.08 * ctx_self -
                             0.06 * aux_vol_ratio -
                             0.05 * FXAI_Clamp(MathAbs(mask_self) / MathMax(scale, 0.10), 0.0, 1.0),
                             0.0,
                             1.0);
      exec_q = FXAI_Clamp(exec_q +
                          0.08 * ctx_self -
                          0.10 * shift_self -
                          0.06 * aux_vol_ratio,
                          0.0,
                          1.0);

      double mfe_raw = MathLog(MathMax(scale, 0.10)) + 0.25 * active + 0.15 * structure + Dot(m_w_mfe, x);
      double mae_raw = -1.0 + 0.20 * (1.0 - structure) + 0.10 * (1.0 - exec_q) + Dot(m_w_mae, x);
      double hit_raw = 0.40 + 0.20 * active - 0.15 * structure + Dot(m_w_hit, x);
      double path_raw = -0.35 + 0.25 * (1.0 - structure) + 0.20 * (1.0 - exec_q) + Dot(m_w_path, x);
      double fill_raw = -0.50 + 0.30 * (1.0 - exec_q) + Dot(m_w_fill, x);

      double mfe_self = MathMax(out.move_q75_points, MathExp(FXAI_ClipSym(mfe_raw, 5.0)));
      double mae_self = MathMax(0.0, scale * FXAI_Sigmoid(mae_raw));
      double hit_self = FXAI_Clamp(FXAI_Sigmoid(hit_raw), 0.0, 1.0);
      double path_self = FXAI_Clamp(FXAI_Sigmoid(path_raw), 0.0, 1.0);
      double fill_self = FXAI_Clamp(FXAI_Sigmoid(fill_raw), 0.0, 1.0);

      out.mfe_mean_points = (1.0 - mix) * mfe_self + mix * MathMax(bank_mfe, out.move_q75_points);
      out.mae_mean_points = (1.0 - mix) * mae_self + mix * MathMax(0.0, bank_mae);
      out.hit_time_frac = (1.0 - mix) * hit_self + mix * FXAI_Clamp(bank_hit, 0.0, 1.0);
      out.path_risk = FXAI_Clamp((1.0 - mix) * path_self + mix * FXAI_Clamp(bank_path, 0.0, 1.0), 0.0, 1.0);
      out.fill_risk = FXAI_Clamp((1.0 - mix) * fill_self + mix * FXAI_Clamp(bank_fill, 0.0, 1.0), 0.0, 1.0);
      out.has_path_quality = true;
   }

   void Update(const double &x[],
               const double sample_w,
               const double target_mfe,
               const double target_mae,
               const double target_hit,
               const double target_path,
               const double target_fill,
               const double target_masked_step,
               const double target_next_vol,
               const double target_regime_shift,
               const double target_context_lead,
               const double lr,
               const double l2)
   {
      double w = FXAI_Clamp(sample_w, 0.10, 6.00);
      double step = FXAI_Clamp(lr * 0.35 * w, 0.0002, 0.0250);
      double reg = FXAI_Clamp(l2, 0.0, 0.05);

      double pred_mfe = MathExp(FXAI_ClipSym(MathLog(0.10) + Dot(m_w_mfe, x), 5.0));
      double pred_mae = FXAI_Sigmoid(Dot(m_w_mae, x));
      double pred_hit = FXAI_Sigmoid(Dot(m_w_hit, x));
      double pred_path = FXAI_Sigmoid(Dot(m_w_path, x));
      double pred_fill = FXAI_Sigmoid(Dot(m_w_fill, x));
      double pred_mask = FXAI_ClipSym(Dot(m_w_mask, x), 8.0);
      double pred_vol = MathExp(FXAI_ClipSym(Dot(m_w_vol, x), 4.0));
      double pred_shift = FXAI_Sigmoid(Dot(m_w_shift, x));
      double pred_ctx = FXAI_Sigmoid(Dot(m_w_ctx, x));

      double tgt_mfe = MathLog(MathMax(target_mfe, 0.10));
      double tgt_mae = FXAI_Clamp(target_mae, 0.0, 1.0);
      double tgt_hit = FXAI_Clamp(target_hit, 0.0, 1.0);
      double tgt_path = FXAI_Clamp(target_path, 0.0, 1.0);
      double tgt_fill = FXAI_Clamp(target_fill, 0.0, 1.0);
      double tgt_mask = FXAI_ClipSym(target_masked_step, 8.0);
      double tgt_vol = MathLog(MathMax(target_next_vol, 0.05));
      double tgt_shift = FXAI_Clamp(target_regime_shift, 0.0, 1.0);
      double tgt_ctx = FXAI_Clamp(target_context_lead, 0.0, 1.0);

      UpdateHead(m_w_mfe, m_b_mfe, x, MathLog(MathMax(pred_mfe, 0.10)), tgt_mfe, step, reg);
      UpdateHead(m_w_mae, m_b_mae, x, pred_mae, tgt_mae, step, reg);
      UpdateHead(m_w_hit, m_b_hit, x, pred_hit, tgt_hit, step, reg);
      UpdateHead(m_w_path, m_b_path, x, pred_path, tgt_path, step, reg);
      UpdateHead(m_w_fill, m_b_fill, x, pred_fill, tgt_fill, step, reg);
      UpdateHead(m_w_mask, m_b_mask, x, pred_mask, tgt_mask, 0.80 * step, reg);
      UpdateHead(m_w_vol, m_b_vol, x, MathLog(MathMax(pred_vol, 0.05)), tgt_vol, 0.70 * step, reg);
      UpdateHead(m_w_shift, m_b_shift, x, pred_shift, tgt_shift, 0.75 * step, reg);
      UpdateHead(m_w_ctx, m_b_ctx, x, pred_ctx, tgt_ctx, 0.75 * step, reg);
      m_ready = true;
      m_steps++;
   }
};

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
   int      m_ctx_window_size;
   double   m_ctx_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   bool     m_target_quality_ready;
   double   m_target_mfe_points;
   double   m_target_mae_points;
   double   m_target_hit_time_frac;
   int      m_target_path_flags;
   double   m_target_path_risk;
   double   m_target_fill_risk;
   double   m_target_masked_step;
   double   m_target_next_vol;
   double   m_target_regime_shift;
   double   m_target_context_lead;
   bool     m_quality_head_ready;
   double   m_quality_mfe_ema;
   double   m_quality_mae_ema;
   double   m_quality_hit_ema;
   double   m_quality_path_risk_ema;
   double   m_quality_fill_risk_ema;
   bool     m_quality_bank_ready[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_obs[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_mfe[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_mae[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_hit[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_path[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_fill[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];

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
   double   m_replay_mfe[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_mae[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_hit_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_path_flags[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_path_risk[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_fill_risk[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_masked_step[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_next_vol[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_regime_shift[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_context_lead[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_cost[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_min_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   datetime m_replay_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_regime[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_session_bucket[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_horizon[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_feature_schema[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_norm_method[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_sequence_bars[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_point_value[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_window_size[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_window[FXAI_PLUGIN_REPLAY_CAPACITY][FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   double   m_replay_priority[FXAI_PLUGIN_REPLAY_CAPACITY];

   int      m_core_predict_calls;
   int      m_core_predict_failures;
   int      m_replay_rehearsals;
   bool m_rng_seeded;
   uint m_rng_state;

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
      m_ctx_feature_schema_id = ctx.feature_schema_id;
      if(m_ctx_feature_schema_id < FXAI_SCHEMA_FULL || m_ctx_feature_schema_id > FXAI_SCHEMA_CONTEXTUAL)
         m_ctx_feature_schema_id = FXAI_SCHEMA_FULL;
      m_ctx_normalization_method_id = ctx.normalization_method_id;
      if(m_ctx_normalization_method_id < 0 || m_ctx_normalization_method_id >= FXAI_NORM_METHOD_COUNT)
         m_ctx_normalization_method_id = FXAI_NORM_EXISTING;
      m_ctx_sequence_bars = (ctx.sequence_bars > 0 ? ctx.sequence_bars : 1);
      if(m_ctx_sequence_bars > FXAI_MAX_SEQUENCE_BARS)
         m_ctx_sequence_bars = FXAI_MAX_SEQUENCE_BARS;
      m_ctx_point_value = (MathIsValidNumber(ctx.point_value) && ctx.point_value > 0.0
                           ? ctx.point_value : (_Point > 0.0 ? _Point : 1.0));
   }

   void SetWindowPayload(const int window_size,
                         const double &x_window[][FXAI_AI_WEIGHTS])
   {
      m_ctx_window_size = window_size;
      if(m_ctx_window_size < 0) m_ctx_window_size = 0;
      if(m_ctx_window_size > FXAI_MAX_SEQUENCE_BARS) m_ctx_window_size = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = (b < m_ctx_window_size ? x_window[b][k] : 0.0);
      }
   }

   void ClearWindowPayload(void)
   {
      m_ctx_window_size = 0;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = 0.0;
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
      m_ctx_window_size = 0;
      m_target_quality_ready = false;
      m_target_mfe_points = 0.0;
      m_target_mae_points = 0.0;
      m_target_hit_time_frac = 1.0;
      m_target_path_flags = 0;
      m_target_path_risk = 0.0;
      m_target_fill_risk = 0.0;
      m_target_masked_step = 0.0;
      m_target_next_vol = 0.0;
      m_target_regime_shift = 0.0;
      m_target_context_lead = 0.5;
      m_quality_head_ready = false;
      m_quality_mfe_ema = 0.0;
      m_quality_mae_ema = 0.0;
      m_quality_hit_ema = 1.0;
      m_quality_path_risk_ema = 0.5;
      m_quality_fill_risk_ema = 0.5;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = 0.0;

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
               m_quality_bank_ready[r][s][h] = false;
               m_quality_bank_obs[r][s][h] = 0.0;
               m_quality_bank_mfe[r][s][h] = 0.0;
               m_quality_bank_mae[r][s][h] = 0.0;
               m_quality_bank_hit[r][s][h] = 1.0;
               m_quality_bank_path[r][s][h] = 0.5;
               m_quality_bank_fill[r][s][h] = 0.5;
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
         m_replay_mfe[i] = 0.0;
         m_replay_mae[i] = 0.0;
         m_replay_hit_time[i] = 1.0;
         m_replay_path_flags[i] = 0;
         m_replay_path_risk[i] = 0.0;
         m_replay_fill_risk[i] = 0.0;
         m_replay_masked_step[i] = 0.0;
         m_replay_next_vol[i] = 0.0;
         m_replay_regime_shift[i] = 0.0;
         m_replay_context_lead[i] = 0.5;
         m_replay_cost[i] = 0.0;
         m_replay_min_move[i] = 0.0;
         m_replay_time[i] = 0;
         m_replay_regime[i] = 0;
         m_replay_session_bucket[i] = 0;
         m_replay_horizon[i] = 1;
         m_replay_feature_schema[i] = 1;
         m_replay_norm_method[i] = 0;
         m_replay_sequence_bars[i] = 1;
         m_replay_point_value[i] = (_Point > 0.0 ? _Point : 1.0);
         m_replay_window_size[i] = 0;
         m_replay_priority[i] = 0.0;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_replay_window[i][b][k] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_x[i][k] = 0.0;
      }

      m_core_predict_calls = 0;
      m_core_predict_failures = 0;
      m_replay_rehearsals = 0;
      m_rng_seeded = false;
      m_rng_state = 0u;
   }

   void EnsurePluginRNG(void)
   {
      if(m_rng_seeded) return;

      uint seed = (uint)(AIId() + 1);
      seed = (uint)(seed * 747796405u + 2891336453u);
      if(seed == 0u)
         seed = 2463534242u;

      m_rng_state = seed;
      m_rng_seeded = true;
   }

   double PluginRand01(void)
   {
      EnsurePluginRNG();
      m_rng_state = (uint)(1664525u * m_rng_state + 1013904223u);
      return FXAI_Clamp(((double)m_rng_state + 0.5) / 4294967296.0, 0.0, 1.0);
   }

   int PluginRandIndex(const int n)
   {
      if(n <= 0) return -1;
      int idx = (int)MathFloor(PluginRand01() * (double)n);
      if(idx < 0) idx = 0;
      if(idx >= n) idx = n - 1;
      return idx;
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

   void SetTrainingTargets(const FXAIAITrainRequestV4 &req)
   {
      m_target_quality_ready = true;
      m_target_mfe_points = MathMax(req.mfe_points, 0.0);
      m_target_mae_points = MathMax(req.mae_points, 0.0);
      m_target_hit_time_frac = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      m_target_path_flags = req.path_flags;
      m_target_path_risk = FXAI_Clamp(req.path_risk, 0.0, 1.0);
      m_target_fill_risk = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
      m_target_masked_step = req.masked_step_target;
      m_target_next_vol = MathMax(req.next_vol_target, 0.0);
      m_target_regime_shift = FXAI_Clamp(req.regime_shift_target, 0.0, 1.0);
      m_target_context_lead = FXAI_Clamp(req.context_lead_target, 0.0, 1.0);
   }

   void UpdateQualityHeads(const FXAIAITrainRequestV4 &req,
                           const double sample_w)
   {
      double alpha = FXAI_Clamp(0.06 * FXAI_Clamp(sample_w, 0.25, 4.0), 0.01, 0.20);
      if(!m_quality_head_ready)
      {
         m_quality_mfe_ema = MathMax(req.mfe_points, 0.0);
         m_quality_mae_ema = MathMax(req.mae_points, 0.0);
         m_quality_hit_ema = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
         m_quality_path_risk_ema = FXAI_Clamp(req.path_risk, 0.0, 1.0);
         m_quality_fill_risk_ema = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
         m_quality_head_ready = true;
         return;
      }

      m_quality_mfe_ema = (1.0 - alpha) * m_quality_mfe_ema + alpha * MathMax(req.mfe_points, 0.0);
      m_quality_mae_ema = (1.0 - alpha) * m_quality_mae_ema + alpha * MathMax(req.mae_points, 0.0);
      m_quality_hit_ema = (1.0 - alpha) * m_quality_hit_ema + alpha * FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      m_quality_path_risk_ema = (1.0 - alpha) * m_quality_path_risk_ema + alpha * FXAI_Clamp(req.path_risk, 0.0, 1.0);
      m_quality_fill_risk_ema = (1.0 - alpha) * m_quality_fill_risk_ema + alpha * FXAI_Clamp(req.fill_risk, 0.0, 1.0);

      int r = req.ctx.regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = req.ctx.session_bucket;
      if(s < 0) s = 0;
      if(s >= FXAI_PLUGIN_SESSION_BUCKETS) s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
      int h = ContextHorizonBucket();
      if(req.ctx.horizon_minutes > 0)
      {
         int hh = req.ctx.horizon_minutes;
         if(hh <= 1) h = 0;
         else if(hh <= 3) h = 1;
         else if(hh <= 5) h = 2;
         else if(hh <= 8) h = 3;
         else if(hh <= 13) h = 4;
         else if(hh <= 21) h = 5;
         else if(hh <= 34) h = 6;
         else h = FXAI_PLUGIN_HORIZON_BUCKETS - 1;
      }

      double obs = m_quality_bank_obs[r][s][h];
      double bank_alpha = FXAI_Clamp(0.12 * FXAI_Clamp(sample_w, 0.25, 4.0) / MathSqrt(1.0 + 0.02 * obs), 0.02, 0.25);
      double mfe_now = MathMax(req.mfe_points, 0.0);
      double mae_now = MathMax(req.mae_points, 0.0);
      double hit_now = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      double path_now = FXAI_Clamp(req.path_risk, 0.0, 1.0);
      double fill_now = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
      if(!m_quality_bank_ready[r][s][h])
      {
         m_quality_bank_mfe[r][s][h] = mfe_now;
         m_quality_bank_mae[r][s][h] = mae_now;
         m_quality_bank_hit[r][s][h] = hit_now;
         m_quality_bank_path[r][s][h] = path_now;
         m_quality_bank_fill[r][s][h] = fill_now;
         m_quality_bank_ready[r][s][h] = true;
      }
      else
      {
         m_quality_bank_mfe[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_mfe[r][s][h] + bank_alpha * mfe_now;
         m_quality_bank_mae[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_mae[r][s][h] + bank_alpha * mae_now;
         m_quality_bank_hit[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_hit[r][s][h] + bank_alpha * hit_now;
         m_quality_bank_path[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_path[r][s][h] + bank_alpha * path_now;
         m_quality_bank_fill[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_fill[r][s][h] + bank_alpha * fill_now;
      }
      m_quality_bank_obs[r][s][h] = MathMin(obs + FXAI_Clamp(sample_w, 0.25, 4.0), 50000.0);
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
      double w = FXAI_MoveEdgeWeight(move_points, cost);
      if(!m_target_quality_ready)
         return w;

      double move_scale = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      double mfe_bonus = FXAI_Clamp(m_target_mfe_points / MathMax(move_scale, 0.10), 0.0, 2.0);
      double mae_penalty = FXAI_Clamp(m_target_mae_points / MathMax(MathMax(m_target_mfe_points, move_scale), 0.10), 0.0, 1.5);
      double timing_bonus = 1.0 - FXAI_Clamp(m_target_hit_time_frac, 0.0, 1.0);
      double execution_drag = 0.60 * FXAI_Clamp(m_target_path_risk, 0.0, 1.0) +
                              0.40 * FXAI_Clamp(m_target_fill_risk, 0.0, 1.0);
      double q = 1.0 + 0.18 * mfe_bonus + 0.14 * timing_bonus - 0.16 * mae_penalty - 0.18 * execution_drag;
      return w * FXAI_Clamp(q, 0.45, 1.85);
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
      double prior_total = 0.0;
      for(int c=0; c<3; c++)
         prior_total += m_bank_class_mass[r][s][h][c];
      if(prior_total <= 1e-9)
      {
         NormalizeClassDistribution(probs);
         return;
      }
      for(int c=0; c<3; c++)
         prior[c] = m_bank_class_mass[r][s][h][c] / prior_total;

      double mix = FXAI_Clamp(total / 120.0, 0.05, 0.35);
      for(int c=0; c<3; c++)
         probs[c] = (1.0 - mix) * probs[c] + mix * prior[c];

      NormalizeClassDistribution(probs);
   }

   double ApplyExpectedMoveCalibrationBank(const double expected_move_points)
   {
      double ev = expected_move_points;
      if(!MathIsValidNumber(ev) || ev <= 0.0)
         return 0.0;
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();

      ev = ev * m_bank_ev_scale[r][s][h] + m_bank_ev_bias[r][s][h];
      if(!MathIsValidNumber(ev) || ev <= 0.0)
         return 0.0;
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
      m_replay_mfe[slot] = sample.mfe_points;
      m_replay_mae[slot] = sample.mae_points;
      m_replay_hit_time[slot] = sample.time_to_hit_frac;
      m_replay_path_flags[slot] = sample.path_flags;
      m_replay_path_risk[slot] = sample.path_risk;
      m_replay_fill_risk[slot] = sample.fill_risk;
      m_replay_masked_step[slot] = sample.masked_step_target;
      m_replay_next_vol[slot] = sample.next_vol_target;
      m_replay_regime_shift[slot] = sample.regime_shift_target;
      m_replay_context_lead[slot] = sample.context_lead_target;
      m_replay_cost[slot] = sample.ctx.cost_points;
      m_replay_min_move[slot] = sample.ctx.min_move_points;
      m_replay_time[slot] = sample.ctx.sample_time;
      m_replay_regime[slot] = sample.ctx.regime_id;
      m_replay_session_bucket[slot] = sample.ctx.session_bucket;
      m_replay_horizon[slot] = sample.ctx.horizon_minutes;
      m_replay_feature_schema[slot] = sample.ctx.feature_schema_id;
      m_replay_norm_method[slot] = sample.ctx.normalization_method_id;
      m_replay_sequence_bars[slot] = sample.ctx.sequence_bars;
      m_replay_point_value[slot] = sample.ctx.point_value;
      m_replay_window_size[slot] = sample.window_size;
      if(m_replay_window_size[slot] < 0) m_replay_window_size[slot] = 0;
      if(m_replay_window_size[slot] > FXAI_MAX_SEQUENCE_BARS) m_replay_window_size[slot] = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_window[slot][b][k] = (b < m_replay_window_size[slot] ? sample.x_window[b][k] : 0.0);
      }
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
      int keep_window_size = m_ctx_window_size;
      double keep_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            keep_window[b][k] = m_ctx_window[b][k];

      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         int idx = best_idx[j];
         if(idx < 0 || idx >= m_replay_size) continue;
         FXAIAIContextV4 replay_ctx;
         replay_ctx.api_version = FXAI_API_VERSION_V4;
         replay_ctx.regime_id = m_replay_regime[idx];
         replay_ctx.session_bucket = m_replay_session_bucket[idx];
         replay_ctx.horizon_minutes = m_replay_horizon[idx];
         replay_ctx.feature_schema_id = m_replay_feature_schema[idx];
         replay_ctx.normalization_method_id = m_replay_norm_method[idx];
         replay_ctx.sequence_bars = m_replay_sequence_bars[idx];
         replay_ctx.cost_points = m_replay_cost[idx];
         replay_ctx.min_move_points = m_replay_min_move[idx];
         replay_ctx.point_value = m_replay_point_value[idx];
         replay_ctx.sample_time = m_replay_time[idx];
         SetContext(replay_ctx);
         m_target_quality_ready = true;
         m_target_mfe_points = m_replay_mfe[idx];
         m_target_mae_points = m_replay_mae[idx];
         m_target_hit_time_frac = m_replay_hit_time[idx];
         m_target_path_flags = m_replay_path_flags[idx];
         m_target_path_risk = m_replay_path_risk[idx];
         m_target_fill_risk = m_replay_fill_risk[idx];
         m_target_masked_step = m_replay_masked_step[idx];
         m_target_next_vol = m_replay_next_vol[idx];
         m_target_regime_shift = m_replay_regime_shift[idx];
         m_target_context_lead = m_replay_context_lead[idx];
         m_ctx_window_size = m_replay_window_size[idx];
         if(m_ctx_window_size < 0) m_ctx_window_size = 0;
         if(m_ctx_window_size > FXAI_MAX_SEQUENCE_BARS) m_ctx_window_size = FXAI_MAX_SEQUENCE_BARS;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         {
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_ctx_window[b][k] = (b < m_ctx_window_size ? m_replay_window[idx][b][k] : 0.0);
         }
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
      m_target_quality_ready = false;
      m_target_mfe_points = 0.0;
      m_target_mae_points = 0.0;
      m_target_hit_time_frac = 1.0;
      m_target_path_flags = 0;
      m_target_path_risk = 0.0;
      m_target_fill_risk = 0.0;
      m_target_masked_step = 0.0;
      m_target_next_vol = 0.0;
      m_target_regime_shift = 0.0;
      m_target_context_lead = 0.5;
      m_ctx_window_size = keep_window_size;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = keep_window[b][k];
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
                     const int feature_schema_id = 0) const
   {
      int min_seq = min_sequence_bars;
      int max_seq = max_sequence_bars;
      if(min_seq < 1) min_seq = 1;
      if(min_seq > FXAI_MAX_SEQUENCE_BARS) min_seq = FXAI_MAX_SEQUENCE_BARS;
      if(max_seq < min_seq) max_seq = min_seq;
      if(max_seq > FXAI_MAX_SEQUENCE_BARS) max_seq = FXAI_MAX_SEQUENCE_BARS;

      out.api_version = FXAI_API_VERSION_V4;
      out.ai_id = AIId();
      out.ai_name = AIName();
      out.family = family;
      out.reference_tier = FXAI_DefaultReferenceTierForAI(out.ai_id);
      out.capability_mask = capability_mask;
      out.feature_schema_id = (feature_schema_id > 0 ? feature_schema_id : FXAI_DefaultFeatureSchemaForFamily(family));
      out.feature_groups_mask = (feature_groups_mask != 0 ? feature_groups_mask : FXAI_DefaultFeatureGroupsForFamily(family));
      out.min_horizon_minutes = min_horizon_minutes;
      out.max_horizon_minutes = max_horizon_minutes;
      out.min_sequence_bars = min_seq;
      out.max_sequence_bars = max_seq;
   }

   void ResetModelOutput(FXAIAIModelOutputV4 &out) const
   {
      out.class_probs[0] = 0.10;
      out.class_probs[1] = 0.10;
      out.class_probs[2] = 0.80;
      out.move_mean_points = 0.0;
      out.move_q25_points = 0.0;
      out.move_q50_points = 0.0;
      out.move_q75_points = 0.0;
      out.mfe_mean_points = 0.0;
      out.mae_mean_points = 0.0;
      out.hit_time_frac = 1.0;
      out.path_risk = 0.0;
      out.fill_risk = 0.0;
      out.confidence = 0.0;
      out.reliability = 0.0;
      out.has_quantiles = false;
      out.has_confidence = false;
      out.has_path_quality = false;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double move_mean_points = out.move_mean_points;
      if(!PredictModelCore(x, hp, out.class_probs, move_mean_points))
         return false;
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = (MathIsValidNumber(move_mean_points) && move_mean_points > 0.0 ? move_mean_points : 0.0);

      double buy_p = out.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = out.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = out.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double entropy = 0.0;
      for(int c=0; c<3; c++)
      {
         double p = MathMax(out.class_probs[c], 1e-9);
         entropy -= p * MathLog(p);
      }
      entropy /= MathLog(3.0);

      double move_scale = MathMax(ResolveMinMovePoints(), 0.10);
      if(m_move_ready && m_move_ema_abs > 0.0)
         move_scale = MathMax(move_scale, 0.60 * m_move_ema_abs);
      if(CurrentWindowSize() > 1)
      {
         double mean1 = CurrentWindowFeatureMean(0);
         double var1 = 0.0;
         for(int b=0; b<CurrentWindowSize(); b++)
         {
            double d = CurrentWindowValue(b, 1) - mean1;
            var1 += d * d;
         }
         var1 = MathSqrt(var1 / (double)CurrentWindowSize());
         move_scale = MathMax(move_scale, 0.35 * var1);
      }

      double sigma = MathMax(0.10, 0.35 * out.move_mean_points + 0.35 * move_scale + 0.20 * skip_p + 0.15 * entropy);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      double mfe_scale = 1.20 + 0.35 * directional_conf + 0.20 * (1.0 - skip_p);
      double mae_scale = 0.35 + 0.25 * skip_p + 0.20 * entropy;
      if(m_quality_head_ready)
      {
         double quality_base = MathMax(m_move_ema_abs, MathMax(move_scale, 0.10));
         mfe_scale = FXAI_Clamp(m_quality_mfe_ema / MathMax(quality_base, 0.10), 0.80, 3.00);
         mae_scale = FXAI_Clamp(m_quality_mae_ema / MathMax(MathMax(m_quality_mfe_ema, quality_base), 0.10), 0.05, 1.50);
      }
      out.mfe_mean_points = MathMax(out.move_q75_points, out.move_mean_points * mfe_scale);
      out.mae_mean_points = MathMax(0.0, out.move_mean_points * mae_scale);
      double hit_frac = 0.60 - 0.25 * directional_conf + 0.20 * skip_p + 0.15 * entropy;
      if(m_quality_head_ready)
         hit_frac = 0.55 * m_quality_hit_ema + 0.45 * hit_frac;
      out.hit_time_frac = FXAI_Clamp(hit_frac, 0.0, 1.0);
      out.confidence = FXAI_Clamp(0.60 * directional_conf + 0.20 * (1.0 - skip_p) + 0.20 * (1.0 - entropy), 0.0, 1.0);
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      double bank_mass = m_bank_total[r][s][h];
      double bank_rel = FXAI_Clamp(bank_mass / 120.0, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * bank_rel + 0.20 * (1.0 - entropy) + 0.10 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      double path_risk = 0.40 * FXAI_Clamp(out.mae_mean_points / MathMax(out.mfe_mean_points, move_scale), 0.0, 1.0) +
                         0.35 * out.hit_time_frac +
                         0.25 * skip_p;
      if(m_quality_head_ready)
         path_risk = 0.55 * path_risk + 0.45 * m_quality_path_risk_ema;
      out.path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);
      double fill_risk = FXAI_Clamp((ResolveCostPoints(x) + 0.50 * ResolveMinMovePoints()) / MathMax(out.move_mean_points + move_scale, 0.25), 0.0, 1.0);
      if(m_quality_head_ready)
         fill_risk = 0.50 * fill_risk + 0.50 * m_quality_fill_risk_ema;
      out.fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      out.has_path_quality = true;
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
      double mean_move = (MathIsValidNumber(calibrated_move_mean_points) && calibrated_move_mean_points > 0.0 ? calibrated_move_mean_points : 0.0);

      dst.move_mean_points = mean_move;
      double raw_mean = (MathIsValidNumber(model_out.move_mean_points) && model_out.move_mean_points > 0.0 ? model_out.move_mean_points : 0.0);
      double scale = (raw_mean > 1e-9 ? mean_move / raw_mean : 1.0);

      if(model_out.has_quantiles && mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, model_out.move_q25_points * scale);
         dst.move_q50_points = MathMax(dst.move_q25_points, model_out.move_q50_points * scale);
         dst.move_q75_points = MathMax(dst.move_q50_points, model_out.move_q75_points * scale);
      }
      else if(mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, mean_move * MathMax(0.25, 1.0 - 0.45 * uncertainty));
         dst.move_q50_points = mean_move;
         dst.move_q75_points = MathMax(dst.move_q50_points, mean_move * (1.0 + 0.45 * uncertainty));
      }
      else
      {
         dst.move_q25_points = 0.0;
         dst.move_q50_points = 0.0;
         dst.move_q75_points = 0.0;
      }

      if(model_out.has_path_quality)
      {
         dst.mfe_mean_points = MathMax(0.0, model_out.mfe_mean_points * scale);
         dst.mae_mean_points = MathMax(0.0, model_out.mae_mean_points * scale);
         dst.hit_time_frac = FXAI_Clamp(model_out.hit_time_frac, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(model_out.path_risk, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp(model_out.fill_risk, 0.0, 1.0);
      }
      else
      {
         dst.mfe_mean_points = MathMax(dst.move_q75_points, dst.move_mean_points);
         dst.mae_mean_points = MathMax(0.0, 0.35 * dst.move_mean_points);
         dst.hit_time_frac = FXAI_Clamp(0.60 - 0.20 * directional_conf + 0.20 * skip_p, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(0.40 * skip_p + 0.35 * dst.hit_time_frac, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp((m_ctx_cost_points + 0.25 * ResolveMinMovePoints()) / MathMax(dst.move_mean_points + ResolveMinMovePoints(), 0.25), 0.0, 1.0);
      }

      dst.confidence = FXAI_Clamp(model_out.has_confidence ? model_out.confidence : directional_conf, 0.0, 1.0);
      dst.reliability = FXAI_Clamp(model_out.has_confidence ? model_out.reliability : (1.0 - 0.50 * skip_p), 0.0, 1.0);
   }

   double CurrentWindowSliceMean(const int input_idx,
                                 const int start_bar,
                                 const int count) const
   {
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0 || count <= 0)
         return 0.0;

      int first = start_bar;
      if(first < 0) first = 0;
      if(first >= m_ctx_window_size) return 0.0;
      int last = first + count;
      if(last > m_ctx_window_size) last = m_ctx_window_size;
      if(last <= first) return 0.0;

      double sum = 0.0;
      int n = 0;
      for(int b=first; b<last; b++)
      {
         sum += m_ctx_window[b][input_idx];
         n++;
      }
      if(n <= 0) return 0.0;
      return sum / (double)n;
   }

   int CurrentWindowSize(void) const
   {
      return m_ctx_window_size;
   }

   double CurrentWindowValue(const int bar_idx, const int input_idx) const
   {
      if(bar_idx < 0 || bar_idx >= m_ctx_window_size) return 0.0;
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS) return 0.0;
      return m_ctx_window[bar_idx][input_idx];
   }

   double CurrentWindowFeatureMean(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0) return 0.0;
      double full = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      int half_n = MathMax(m_ctx_window_size / 2, 1);
      int quarter_n = MathMax(m_ctx_window_size / 4, 1);
      double half = CurrentWindowSliceMean(input_idx, m_ctx_window_size - half_n, half_n);
      double quarter = CurrentWindowSliceMean(input_idx, m_ctx_window_size - quarter_n, quarter_n);
      return 0.40 * full + 0.35 * half + 0.25 * quarter;
   }

   double CurrentWindowFeatureRecentMean(const int feature_idx,
                                         const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0) n = 1;
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      return CurrentWindowSliceMean(input_idx, 0, n);
   }

   double CurrentWindowFeatureStd(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double mean = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      double acc = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double d = m_ctx_window[b][input_idx] - mean;
         acc += d * d;
      }
      return MathSqrt(acc / (double)MathMax(m_ctx_window_size, 1));
   }

   double CurrentWindowFeatureRange(const int feature_idx,
                                    const int recent_bars = 0) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0 || n > m_ctx_window_size) n = m_ctx_window_size;
      double lo = CurrentWindowValue(0, input_idx);
      double hi = lo;
      for(int b=0; b<n; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         if(v < lo) lo = v;
         if(v > hi) hi = v;
      }
      return hi - lo;
   }

   double CurrentWindowFeatureSlope(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double first = m_ctx_window[0][input_idx];
      double last = m_ctx_window[m_ctx_window_size - 1][input_idx];
      return (last - first) / (double)MathMax(m_ctx_window_size - 1, 1);
   }

   double CurrentWindowFeatureRecentDelta(const int feature_idx,
                                          const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 1) n = MathMax(m_ctx_window_size / 4, 2);
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      int last_idx = n - 1;
      if(last_idx < 0) last_idx = 0;
      return CurrentWindowValue(0, input_idx) - CurrentWindowValue(last_idx, input_idx);
   }

   double CurrentWindowFeatureEMAMean(const int feature_idx,
                                      const double decay = 0.72) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      double a = FXAI_Clamp(decay, 0.05, 0.98);
      double w = 1.0;
      double sw = 0.0;
      double sum = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         sum += w * v;
         sw += w;
         w *= a;
      }
      if(sw <= 0.0) return 0.0;
      return sum / sw;
   }

   void BuildChronologicalSequenceTensor(const double &x[],
                                         double &seq[][FXAI_AI_WEIGHTS],
                                         int &seq_len) const
   {
      FXAISequenceBuffer buffer;
      FXAI_SequenceBufferLoadWindow(buffer,
                                    x,
                                    m_ctx_window,
                                    m_ctx_window_size,
                                    FXAI_MAX_SEQUENCE_BARS,
                                    false);
      FXAI_SequenceBufferExport(buffer, seq, seq_len);
   }

   int ContextSequenceCap(const int max_cap,
                          const int base_min = 8) const
   {
      return FXAI_ContextSequenceSpan(max_cap,
                                      (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                      _Symbol,
                                      base_min);
   }

   int ContextBatchCap(const int max_cap,
                       const int base_min = 4) const
   {
      return FXAI_ContextBatchSpan(max_cap,
                                   (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                   _Symbol,
                                   base_min);
   }

   void CopyCurrentInputClipped(const double &x[],
                                double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }
   }

   void BuildChronologicalSequenceTensorCapped(const double &x[],
                                               const int max_steps,
                                               double &seq[][FXAI_AI_WEIGHTS],
                                               int &seq_len,
                                               const bool normalize = true) const
   {
      int mask[];
      double pos_bias[];
      BuildPackedSequenceTensorCapped(x, max_steps, seq, seq_len, mask, pos_bias, normalize);
   }

   void BuildPackedSequenceTensorCapped(const double &x[],
                                        const int max_steps,
                                        double &seq[][FXAI_AI_WEIGHTS],
                                        int &seq_len,
                                        int &mask[],
                                        double &pos_bias[],
                                        const bool normalize = true) const
   {
      FXAISequenceBuffer buffer;
      FXAI_SequenceBufferLoadWindow(buffer,
                                    x,
                                    m_ctx_window,
                                    m_ctx_window_size,
                                    max_steps,
                                    normalize);
      FXAI_SequenceBufferPreparePacked(buffer, seq, seq_len, mask, pos_bias);
   }

   void BuildTensorEncodedInput(const double &x[],
                                const int style,
                                double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      BuildPackedSequenceTensorCapped(x, FXAI_MAX_SEQUENCE_BARS, seq, seq_len, seq_mask, seq_pos_bias, true);
      if(seq_len <= 1)
         return;

      double query[FXAI_AI_WEIGHTS];
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         query[k] = seq[seq_len - 1][k];

      double attn[];
      double conv_slow[];
      double conv_fast[];
      double kernel_slow[3] = {0.20, 0.60, 0.20};
      double kernel_fast[3] = {1.00, 0.00, -1.00};
      FXAI_ModuleMultiHeadAttentionSummary(seq, seq_len, query, seq_mask, seq_pos_bias, 2, attn);
      FXAI_ModuleConv1DSummary(seq, seq_len, kernel_slow, 3, conv_slow);
      FXAI_ModuleConv1DSummary(seq, seq_len, kernel_fast, 3, conv_fast);

      int last = seq_len - 1;
      int prev = MathMax(last - 1, 0);
      int mid = MathMax(seq_len / 2, 0);
      int root = 0;
      double w_cur = 0.40;
      double w_prev = 0.20;
      double w_mid = 0.10;
      double w_att = 0.15;
      double w_slow = 0.10;
      double w_fast = 0.05;

      switch(style)
      {
         case FXAI_SEQ_STYLE_RECURRENT:
            w_cur = 0.34; w_prev = 0.24; w_mid = 0.10; w_att = 0.14; w_slow = 0.10; w_fast = 0.08;
            break;
         case FXAI_SEQ_STYLE_CONVOLUTIONAL:
            w_cur = 0.25; w_prev = 0.18; w_mid = 0.10; w_att = 0.12; w_slow = 0.17; w_fast = 0.18;
            break;
         case FXAI_SEQ_STYLE_TRANSFORMER:
            w_cur = 0.24; w_prev = 0.12; w_mid = 0.14; w_att = 0.28; w_slow = 0.12; w_fast = 0.10;
            break;
         case FXAI_SEQ_STYLE_STATE_SPACE:
            w_cur = 0.28; w_prev = 0.18; w_mid = 0.14; w_att = 0.12; w_slow = 0.18; w_fast = 0.10;
            break;
         case FXAI_SEQ_STYLE_WORLD:
            w_cur = 0.22; w_prev = 0.14; w_mid = 0.18; w_att = 0.24; w_slow = 0.12; w_fast = 0.10;
            break;
         default:
            break;
      }

      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
      {
         double v = w_cur * seq[last][k] +
                    w_prev * seq[prev][k] +
                    w_mid * seq[mid][k] +
                    w_att * attn[k] +
                    w_slow * conv_slow[k] +
                    w_fast * conv_fast[k];
         v += 0.04 * seq[root][k];
         xa[k] = FXAI_ClipSym(v, 8.0);
      }
      xa[0] = 1.0;
   }

   void GetQualityBankPriors(double &mfe_out,
                             double &mae_out,
                             double &hit_out,
                             double &path_out,
                             double &fill_out,
                             double &trust_out) const
   {
      mfe_out = m_quality_mfe_ema;
      mae_out = m_quality_mae_ema;
      hit_out = m_quality_hit_ema;
      path_out = m_quality_path_risk_ema;
      fill_out = m_quality_fill_risk_ema;
      trust_out = (m_quality_head_ready ? 0.35 : 0.0);

      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      if(!m_quality_bank_ready[r][s][h])
         return;

      double bank_trust = FXAI_Clamp(m_quality_bank_obs[r][s][h] / 120.0, 0.10, 0.85);
      mfe_out = (1.0 - bank_trust) * mfe_out + bank_trust * m_quality_bank_mfe[r][s][h];
      mae_out = (1.0 - bank_trust) * mae_out + bank_trust * m_quality_bank_mae[r][s][h];
      hit_out = (1.0 - bank_trust) * hit_out + bank_trust * m_quality_bank_hit[r][s][h];
      path_out = (1.0 - bank_trust) * path_out + bank_trust * m_quality_bank_path[r][s][h];
      fill_out = (1.0 - bank_trust) * fill_out + bank_trust * m_quality_bank_fill[r][s][h];
      trust_out = FXAI_Clamp(trust_out + 0.65 * bank_trust, 0.0, 1.0);
   }

   void PopulatePathQualityHeads(FXAIAIModelOutputV4 &out,
                                 const double &x[],
                                 const double activity_gate,
                                 const double structural_quality,
                                 const double execution_quality = -1.0) const
   {
      FXAIAIManifestV4 manifest;
      Describe(manifest);
      double bank_mfe = 0.0;
      double bank_mae = 0.0;
      double bank_hit = 1.0;
      double bank_path = 0.5;
      double bank_fill = 0.5;
      double bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);

      double active = FXAI_Clamp(activity_gate, 0.0, 1.0);
      double structure = FXAI_Clamp(structural_quality, 0.0, 1.0);
      double exec_q = (execution_quality >= 0.0 ? FXAI_Clamp(execution_quality, 0.0, 1.0) : structure);
      double move_scale = MathMax(out.move_mean_points,
                          MathMax(out.move_q50_points,
                          MathMax(ResolveMinMovePoints(), 0.10)));
      double qspan = MathMax(0.0, out.move_q75_points - out.move_q25_points);
      double sigma = MathMax(0.10, 0.30 * move_scale + 0.45 * qspan);
      double directional = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY],
                                              out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      double skip = FXAI_Clamp(out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double cost_ratio = FXAI_Clamp(ResolveCostPoints(x) / MathMax(move_scale + 0.40 * sigma, 0.25), 0.0, 1.0);
      double trend = 0.0;
      double trend_fast = 0.0;
      double noise = 0.0;
      double ctx_shape = 0.0;
      if(CurrentWindowSize() > 1)
      {
         double slope = MathAbs(CurrentWindowFeatureSlope(0));
         double slope_fast = MathAbs(CurrentWindowFeatureRecentDelta(0, MathMax(CurrentWindowSize() / 4, 2)));
         double stdv = CurrentWindowFeatureStd(0);
         double level = MathAbs(CurrentWindowFeatureEMAMean(0));
         double local_range = CurrentWindowFeatureRange(0, MathMax(CurrentWindowSize() / 2, 2));
         double ctx_recent = MathAbs(CurrentWindowFeatureRecentMean(10, MathMax(CurrentWindowSize() / 4, 1)));
         double ctx_slow = MathAbs(CurrentWindowFeatureMean(10));
         trend = FXAI_Clamp(slope / MathMax(stdv + 0.20 * MathAbs(level), 0.10), 0.0, 1.25);
         trend_fast = FXAI_Clamp(slope_fast / MathMax(local_range + 0.10, 0.10), 0.0, 1.25);
         noise = FXAI_Clamp((0.65 * stdv + 0.35 * local_range) / MathMax(MathAbs(level) + 0.10, 0.10), 0.0, 1.25);
         ctx_shape = FXAI_Clamp((ctx_recent + 0.50 * ctx_slow) / MathMax(local_range + 0.10, 0.10), 0.0, 1.25);
      }

      double fam_trend = 1.0;
      double fam_ctx = 1.0;
      double fam_exec = 1.0;
      switch(manifest.family)
      {
         case FXAI_FAMILY_RECURRENT:
         case FXAI_FAMILY_CONVOLUTIONAL:
         case FXAI_FAMILY_TRANSFORMER:
         case FXAI_FAMILY_STATE_SPACE:
            fam_trend = 1.12;
            fam_ctx = 1.08;
            break;
         case FXAI_FAMILY_WORLD_MODEL:
         case FXAI_FAMILY_RETRIEVAL:
         case FXAI_FAMILY_MIXTURE:
            fam_ctx = 1.15;
            fam_exec = 0.92;
            break;
         case FXAI_FAMILY_TREE:
         case FXAI_FAMILY_LINEAR:
            fam_trend = 0.92;
            fam_exec = 1.06;
            break;
         case FXAI_FAMILY_RULE_BASED:
            fam_trend = 0.80;
            fam_ctx = 0.85;
            fam_exec = 1.18;
            break;
         default:
            break;
      }

      double mfe_scale = 1.05 + 0.30 * directional + 0.18 * active + 0.16 * structure +
                         0.16 * trend * fam_trend + 0.10 * trend_fast + 0.08 * ctx_shape * fam_ctx;
      double mae_scale = 0.14 + 0.24 * (1.0 - active) + 0.18 * (1.0 - structure) +
                         0.16 * cost_ratio * fam_exec + 0.12 * noise + 0.08 * skip;
      if(m_quality_head_ready || bank_trust > 0.0)
      {
         double quality_base = MathMax(move_scale, 0.10);
         double bank_mfe_scale = FXAI_Clamp(bank_mfe / quality_base, 0.80, 3.40);
         double bank_mae_scale = FXAI_Clamp(bank_mae / MathMax(MathMax(bank_mfe, quality_base), 0.10), 0.05, 1.70);
         mfe_scale = (1.0 - 0.55 * bank_trust) * mfe_scale + 0.55 * bank_trust * bank_mfe_scale;
         mae_scale = (1.0 - 0.55 * bank_trust) * mae_scale + 0.55 * bank_trust * bank_mae_scale;
      }
      out.mfe_mean_points = MathMax(out.move_q75_points, move_scale * FXAI_Clamp(mfe_scale, 0.80, 3.50));
      out.mae_mean_points = MathMax(0.0, move_scale * FXAI_Clamp(mae_scale, 0.05, 1.80));

      double hit_frac = 0.70 - 0.20 * active - 0.12 * structure - 0.08 * trend_fast -
                        0.06 * ctx_shape + 0.18 * noise + 0.16 * cost_ratio + 0.10 * skip;
      if(m_quality_head_ready || bank_trust > 0.0)
         hit_frac = (1.0 - 0.60 * bank_trust) * hit_frac + 0.60 * bank_trust * bank_hit;
      out.hit_time_frac = FXAI_Clamp(hit_frac, 0.0, 1.0);

      double path_risk = 0.34 * FXAI_Clamp(out.mae_mean_points / MathMax(out.mfe_mean_points, move_scale), 0.0, 1.0) +
                         0.22 * out.hit_time_frac +
                         0.18 * cost_ratio +
                         0.14 * (1.0 - structure) +
                         0.12 * noise +
                         0.08 * (1.0 - exec_q);
      if(m_quality_head_ready || bank_trust > 0.0)
         path_risk = (1.0 - 0.60 * bank_trust) * path_risk + 0.60 * bank_trust * bank_path;
      out.path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);

      double fill_risk = 0.46 * cost_ratio +
                         0.26 * (1.0 - exec_q) +
                         0.16 * skip +
                         0.12 * noise;
      if(m_quality_head_ready || bank_trust > 0.0)
         fill_risk = (1.0 - 0.60 * bank_trust) * fill_risk + 0.60 * bank_trust * bank_fill;
      out.fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      out.has_path_quality = true;
   }

   void PredictNativeQualityHeads(const double &x[],
                                  const double activity_gate,
                                  const double structural_quality,
                                  const double execution_quality,
                                  FXAIAIModelOutputV4 &out) const
   {
      double bank_mfe = 0.0;
      double bank_mae = 0.0;
      double bank_hit = 1.0;
      double bank_path = 0.5;
      double bank_fill = 0.5;
      double bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(x,
                              out.move_mean_points,
                              activity_gate,
                              structural_quality,
                              execution_quality,
                              bank_mfe,
                              bank_mae,
                              bank_hit,
                              bank_path,
                              bank_fill,
                              bank_trust,
                              out);
   }

   void UpdateNativeQualityHeads(const double &x[],
                                 const double sample_w,
                                 const double lr,
                                 const double l2)
   {
      m_quality_heads.Update(x,
                             sample_w,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             lr,
                             l2);
   }

   double TargetMFEPoints(void) const { return m_target_quality_ready ? m_target_mfe_points : 0.0; }
   double TargetMAEPoints(void) const { return m_target_quality_ready ? m_target_mae_points : 0.0; }
   double TargetHitTimeFrac(void) const { return m_target_quality_ready ? m_target_hit_time_frac : 1.0; }
   int TargetPathFlags(void) const { return m_target_quality_ready ? m_target_path_flags : 0; }
   double TargetPathRisk(void) const { return m_target_quality_ready ? m_target_path_risk : 0.0; }
   double TargetFillRisk(void) const { return m_target_quality_ready ? m_target_fill_risk : 0.0; }
   double TargetMaskedStep(void) const { return m_target_quality_ready ? m_target_masked_step : 0.0; }
   double TargetNextVol(void) const { return m_target_quality_ready ? m_target_next_vol : 0.0; }
   double TargetRegimeShift(void) const { return m_target_quality_ready ? m_target_regime_shift : 0.0; }
   double TargetContextLead(void) const { return m_target_quality_ready ? m_target_context_lead : 0.5; }

public:
   CFXAIAIPlugin(void) { ResetAuxState(); }

   virtual int AIId(void) const = 0;
   virtual string AIName(void) const = 0;

   virtual void Reset(void) { ResetAuxState(); }
   virtual void Describe(FXAIAIManifestV4 &out) const = 0;
   virtual bool SupportsSyntheticSeries(void) const { return false; }
   virtual bool SetSyntheticSeries(const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[])
   {
      return false;
   }
   virtual void ClearSyntheticSeries(void) {}
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
   int ReplayRehearsals(void) const { return m_replay_rehearsals; }

   void Train(const FXAIAITrainRequestV4 &req, const FXAIAIHyperParams &hp)
   {
      string reason = "";
      if(!FXAI_ValidateTrainRequestV4(req, reason))
         return;
      EnsureInitialized(hp);
      SetContext(req.ctx);
      SetWindowPayload(req.window_size, req.x_window);
      SetTrainingTargets(req);

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
      double pre_move = 0.0;
      bool have_pre = PredictModelCore(req.x, hp, pre_probs, pre_move);
      if(!have_pre)
         m_core_predict_failures++;
      NormalizeClassDistribution(pre_probs);
      if(!MathIsValidNumber(pre_move) || pre_move < 0.0)
         pre_move = 0.0;

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
      UpdateQualityHeads(req, sample_w);
      if(can_replay)
         RunReplayRehearsal(hp, req.ctx.regime_id, req.ctx.horizon_minutes);
   }

   bool Predict(const FXAIAIPredictRequestV4 &req,
                const FXAIAIHyperParams &hp,
                FXAIAIPredictionV4 &out)
   {
      string reason = "";
      if(!FXAI_ValidatePredictRequestV4(req, reason))
      {
         for(int c=0; c<3; c++)
            out.class_probs[c] = (c == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
         out.move_mean_points = 0.0;
         out.move_q25_points = 0.0;
         out.move_q50_points = 0.0;
         out.move_q75_points = 0.0;
         out.mfe_mean_points = 0.0;
         out.mae_mean_points = 0.0;
         out.hit_time_frac = 1.0;
         out.path_risk = 0.0;
         out.fill_risk = 0.0;
         out.confidence = 0.0;
         out.reliability = 0.0;
         return false;
      }
      EnsureInitialized(hp);
      SetContext(req.ctx);
      SetWindowPayload(req.window_size, req.x_window);

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
         model_out.move_mean_points = 0.0;
         FillPredictionV4(model_out, model_out.move_mean_points, out);
         return false;
      }

      NormalizeClassDistribution(model_out.class_probs);
      if(!model_out.has_path_quality)
      {
         double structural = (model_out.has_confidence ? model_out.reliability : 0.50);
         double execution = (model_out.has_confidence ? model_out.confidence : 0.50);
         PopulatePathQualityHeads(model_out, req.x, FXAI_Clamp(1.0 - model_out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), structural, execution);
      }
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
