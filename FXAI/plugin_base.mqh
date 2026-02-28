// FXAI v1
#ifndef __FX6_PLUGIN_BASE_MQH__
#define __FX6_PLUGIN_BASE_MQH__

#include <Object.mqh>
#include "shared.mqh"

class CFX6AIPlugin : public CObject
{
protected:
   bool   m_move_ready;
   double m_move_ema_abs;
   bool   m_move_head_ready;
   int    m_move_head_steps;
   double m_move_w[FX6_AI_WEIGHTS];

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

   // Plugin-local 3-class calibrator head (v2 native class output).
   double m_v2_cls_w[3][FX6_PLUGIN_CLASS_FEATURES];
   int    m_v2_cls_steps;

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
                   const double min_move_points)
   {
      m_ctx_time_ready = (sample_time > 0);
      m_ctx_time = (m_ctx_time_ready ? sample_time : 0);

      m_ctx_cost_ready = (MathIsValidNumber(cost_points) && cost_points >= 0.0);
      m_ctx_cost_points = (m_ctx_cost_ready ? cost_points : 0.0);

      if(MathIsValidNumber(min_move_points) && min_move_points > 0.0)
         m_ctx_min_move_points = min_move_points;
      else
         m_ctx_min_move_points = 0.0;
   }

   int NormalizeClassLabel(const int y,
                           const double &x[],
                           const double move_points) const
   {
      if(y >= (int)FX6_LABEL_SELL && y <= (int)FX6_LABEL_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return (int)FX6_LABEL_SKIP;

      if(y > 0) return (int)FX6_LABEL_BUY;
      if(y == 0) return (int)FX6_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FX6_LABEL_BUY : (int)FX6_LABEL_SELL);
   }

   void ResetAuxState(void)
   {
      m_move_ready = false;
      m_move_ema_abs = 0.0;

      m_move_head_ready = false;
      m_move_head_steps = 0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
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

      m_v2_cls_steps = 0;
      for(int c=0; c<3; c++)
      {
         for(int k=0; k<FX6_PLUGIN_CLASS_FEATURES; k++)
            m_v2_cls_w[c][k] = 0.0;
      }
      // Conservative prior to reduce early overtrading.
      m_v2_cls_w[(int)FX6_LABEL_SKIP][0] = 0.20;
   }

   void UpdateMoveHead(const double &x[],
                       const double move_points,
                       const FX6AIHyperParams &hp,
                       const double sample_w)
   {
      double tgt = MathAbs(move_points);
      if(!MathIsValidNumber(tgt)) return;

      double pred = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         pred += m_move_w[i] * x[i];
      if(pred < 0.0) pred = 0.0;

      double err = FX6_ClipSym(tgt - pred, 20.0);
      double lr = FX6_Clamp(0.08 * hp.lr, 0.00005, 0.02000);
      double l2 = FX6_Clamp(0.25 * hp.l2, 0.0000, 0.1000);
      double w = FX6_Clamp(sample_w, 0.25, 4.00);

      m_move_w[0] += lr * w * err;
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
         m_move_w[i] += lr * (w * FX6_ClipSym(err * x[i], 6.0) - l2 * m_move_w[i]);

      m_move_head_steps++;
      if(m_move_head_steps >= 16) m_move_head_ready = true;
   }

   double PredictMoveHeadRaw(const double &x[]) const
   {
      double p = 0.0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         p += m_move_w[i] * x[i];
      if(p < 0.0) p = 0.0;
      return p;
   }

   void UpdateCalibration(const double prob_raw,
                          const int y,
                          const double sample_w = 1.0)
   {
      double pr = FX6_Clamp(prob_raw, 0.001, 0.999);
      double z = FX6_Logit(pr);
      double py = FX6_Sigmoid((m_cal_a * z) + m_cal_b);
      double e = ((double)y - py);

      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double lr = 0.015 * w;
      double reg = 0.0005;

      m_cal_a += lr * (e * z - reg * (m_cal_a - 1.0));
      m_cal_b += lr * (e);
      m_cal_a = FX6_Clamp(m_cal_a, 0.20, 5.00);
      m_cal_b = FX6_Clamp(m_cal_b, -4.0, 4.0);

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
      double pr = FX6_Clamp(prob_raw, 0.001, 0.999);
      double p_platt = FX6_Sigmoid((m_cal_a * FX6_Logit(pr)) + m_cal_b);
      if(!m_cal_ready)
         return FX6_Clamp(p_platt, 0.001, 0.999);

      double total = 0.0;
      for(int i=0; i<12; i++) total += m_iso_cnt[i];
      if(total < 30.0)
         return FX6_Clamp(p_platt, 0.001, 0.999);

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
         r = FX6_Clamp(r, 0.01, 0.99);
         if(i > 0 && r < mono[i - 1]) r = mono[i - 1];
         mono[i] = r;
         prev = r;
      }

      double p_iso = mono[bi];
      double p = (0.70 * p_platt) + (0.30 * p_iso);
      return FX6_Clamp(p, 0.001, 0.999);
   }

   FX6AIHyperParams ScaleHyperParamsForMove(const FX6AIHyperParams &hp,
                                            const double move_points) const
   {
      FX6AIHyperParams h = hp;
      double w = FX6_MoveWeight(move_points);

      h.lr *= w;
      h.ftrl_alpha *= w;
      h.xgb_lr *= w;
      h.mlp_lr *= w;
      h.quantile_lr *= w;
      h.enhash_lr *= w;

      h.lr = FX6_Clamp(h.lr, 0.0001, 1.0000);
      h.ftrl_alpha = FX6_Clamp(h.ftrl_alpha, 0.0001, 5.0000);
      h.xgb_lr = FX6_Clamp(h.xgb_lr, 0.0001, 1.0000);
      h.mlp_lr = FX6_Clamp(h.mlp_lr, 0.0001, 1.0000);
      h.quantile_lr = FX6_Clamp(h.quantile_lr, 0.0001, 1.0000);
      h.enhash_lr = FX6_Clamp(h.enhash_lr, 0.0001, 1.0000);
      return h;
   }

   double MoveSampleWeight(const double &x[],
                           const double move_points) const
   {
      double cost = ResolveCostPoints(x);
      return FX6_MoveEdgeWeight(move_points, cost);
   }

   void BuildClassInput(const double p_up,
                        const double expected_move_points,
                        const double min_move_points,
                        const double cost_points,
                        double &xc[]) const
   {
      double p = FX6_Clamp(p_up, 0.001, 0.999);
      double em = (expected_move_points > 0.0 ? expected_move_points : 0.0);
      double mm = (min_move_points > 0.0 ? min_move_points : 0.10);
      double cp = (cost_points >= 0.0 ? cost_points : mm);
      double denom = MathMax(mm, 0.10);

      xc[0] = 1.0;
      xc[1] = FX6_Clamp((p - 0.5) * 2.0, -1.0, 1.0);
      xc[2] = FX6_Clamp((em - mm) / denom, -6.0, 6.0);
      xc[3] = FX6_Clamp(em / denom, 0.0, 12.0);
      xc[4] = FX6_Clamp(cp / denom, 0.0, 4.0);
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

   void PredictLocalClassProbs(const double &xc[], double &probs[]) const
   {
      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int k=0; k<FX6_PLUGIN_CLASS_FEATURES; k++)
            z += m_v2_cls_w[c][k] * xc[k];
         logits[c] = z;
      }
      Softmax3(logits, probs);
   }

   void UpdateLocalClassHead(const int label_class,
                             const double &xc[],
                             const FX6AIHyperParams &hp,
                             const double sample_w)
   {
      if(label_class < (int)FX6_LABEL_SELL || label_class > (int)FX6_LABEL_SKIP)
         return;

      double probs[3];
      PredictLocalClassProbs(xc, probs);

      double w = FX6_Clamp(sample_w, 0.25, 4.00);
      double lr = FX6_Clamp(hp.lr * 0.40 * w, 0.0003, 0.0500);
      double l2 = FX6_Clamp(hp.l2, 0.0, 0.1000);

      for(int c=0; c<3; c++)
      {
         double target = (c == label_class ? 1.0 : 0.0);
         double err = target - probs[c];
         for(int k=0; k<FX6_PLUGIN_CLASS_FEATURES; k++)
         {
            double reg = (k == 0 ? 0.0 : l2 * m_v2_cls_w[c][k]);
            m_v2_cls_w[c][k] += lr * (err * xc[k] - reg);
         }
      }

      m_v2_cls_steps++;
   }

public:
   CFX6AIPlugin(void) { ResetAuxState(); }

   virtual int AIId(void) const = 0;
   virtual string AIName(void) const = 0;

   virtual void Reset(void) { ResetAuxState(); }
   virtual void EnsureInitialized(const FX6AIHyperParams &hp) {}
   virtual bool SupportsNativeClassProbs(void) const { return false; }
   virtual bool PredictNativeClassProbs(const double &x[],
                                        const FX6AIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      return false;
   }

   // V2 training API: time/cost-aware sample payload.
   void TrainV2(const FX6AISampleV2 &sample, const FX6AIHyperParams &hp)
   {
      if(!sample.valid) return;
      EnsureInitialized(hp);
      SetContext(sample.sample_time, sample.cost_points, sample.min_move_points);

      // Core model update.
      UpdateWithMove(sample.label_class, sample.x, hp, sample.move_points);

      if(SupportsNativeClassProbs())
         return;

      // Fallback local multiclass objective for legacy plugins.
      double p_up = FX6_Clamp(PredictProb(sample.x, hp), 0.001, 0.999);
      double exp_move = PredictExpectedMovePoints(sample.x, hp);
      if(exp_move <= 0.0) exp_move = MathAbs(sample.move_points);
      if(exp_move <= 0.0) exp_move = ResolveMinMovePoints();

      double xc[FX6_PLUGIN_CLASS_FEATURES];
      double mm = (sample.min_move_points > 0.0 ? sample.min_move_points : ResolveMinMovePoints());
      double cp = (sample.cost_points >= 0.0 ? sample.cost_points : ResolveCostPoints(sample.x));
      BuildClassInput(p_up, exp_move, mm, cp, xc);

      double sw = MoveSampleWeight(sample.x, sample.move_points);
      if(sample.label_class == (int)FX6_LABEL_SKIP) sw *= 0.75;
      UpdateLocalClassHead(sample.label_class, xc, hp, sw);
   }

   // V2 inference API: returns calibrated 3-class distribution.
   void PredictV2(const FX6AIPredictV2 &req,
                  const FX6AIHyperParams &hp,
                  FX6AIPredictionV2 &out)
   {
      EnsureInitialized(hp);
      SetContext(req.sample_time, req.cost_points, req.min_move_points);

      if(SupportsNativeClassProbs())
      {
         double native_probs[3];
         double native_move = -1.0;
         if(PredictNativeClassProbs(req.x, hp, native_probs, native_move))
         {
            out.class_probs[0] = FX6_Clamp(native_probs[0], 0.0005, 0.9990);
            out.class_probs[1] = FX6_Clamp(native_probs[1], 0.0005, 0.9990);
            out.class_probs[2] = FX6_Clamp(native_probs[2], 0.0005, 0.9990);
            double s0 = out.class_probs[0] + out.class_probs[1] + out.class_probs[2];
            if(s0 <= 0.0) s0 = 1.0;
            out.class_probs[0] /= s0;
            out.class_probs[1] /= s0;
            out.class_probs[2] /= s0;
            out.p_up = out.class_probs[(int)FX6_LABEL_BUY];
            out.expected_move_points = (native_move > 0.0 ? native_move : MathMax(ResolveMinMovePoints(), 0.10));
            return;
         }
      }

      double p_up = FX6_Clamp(PredictProb(req.x, hp), 0.001, 0.999);
      double exp_move = PredictExpectedMovePoints(req.x, hp);
      if(exp_move <= 0.0) exp_move = ResolveMinMovePoints();
      if(exp_move <= 0.0) exp_move = 0.10;

      double mm = (req.min_move_points > 0.0 ? req.min_move_points : ResolveMinMovePoints());
      if(mm <= 0.0) mm = 0.10;
      double cp = (req.cost_points >= 0.0 ? req.cost_points : ResolveCostPoints(req.x));
      if(cp < 0.0) cp = 0.0;

      double xc[FX6_PLUGIN_CLASS_FEATURES];
      BuildClassInput(p_up, exp_move, mm, cp, xc);

      double probs_head[3];
      PredictLocalClassProbs(xc, probs_head);

      double active = FX6_Clamp((exp_move - mm) / MathMax(mm, 0.10), 0.0, 1.0);
      double probs_anchor[3];
      probs_anchor[(int)FX6_LABEL_BUY] = FX6_Clamp(p_up * active, 0.001, 0.999);
      probs_anchor[(int)FX6_LABEL_SELL] = FX6_Clamp((1.0 - p_up) * active, 0.001, 0.999);
      probs_anchor[(int)FX6_LABEL_SKIP] = FX6_Clamp(1.0 - active, 0.001, 0.999);
      double sa = probs_anchor[0] + probs_anchor[1] + probs_anchor[2];
      if(sa <= 0.0) sa = 1.0;
      for(int c=0; c<3; c++) probs_anchor[c] /= sa;

      double w_head = (m_v2_cls_steps >= 24 ? 0.70 : 0.35);
      double w_anchor = 1.0 - w_head;
      for(int c=0; c<3; c++)
         out.class_probs[c] = FX6_Clamp(w_head * probs_head[c] + w_anchor * probs_anchor[c], 0.0005, 0.9990);

      double s = out.class_probs[0] + out.class_probs[1] + out.class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++) out.class_probs[c] /= s;

      out.p_up = out.class_probs[(int)FX6_LABEL_BUY];
      out.expected_move_points = exp_move;
   }

protected:
   // Legacy model hooks (v1 core), kept protected to retire external v1 usage.
   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points) = 0;
   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp) = 0;
   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      double head = (m_move_head_ready ? PredictMoveHeadRaw(x) : -1.0);
      if(head > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.60 * head + 0.40 * m_move_ema_abs;
      if(head > 0.0) return head;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return -1.0;
   }
};

#endif // __FX6_PLUGIN_BASE_MQH__
