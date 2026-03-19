#ifndef __FXAI_AI_GRAPHWM_MQH__
#define __FXAI_AI_GRAPHWM_MQH__

#include "..\..\API\plugin_base.mqh"

class CFXAIAIGraphWM : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_feat_n;
   int    m_steps;
   double m_gate_w[FXAI_AI_WEIGHTS];
   double m_dir_w[FXAI_AI_WEIGHTS];
   double m_move_mu_w[FXAI_AI_WEIGHTS];
   double m_move_logv_w[FXAI_AI_WEIGHTS];
   double m_graph_bias;
   double m_last_probs[3];
   double m_reliability_ema;
   double m_err_ema;
   CFXAITernaryCalibrator m_cal3;
   CFXAINativeQualityHeads m_quality_heads;

   double SafeX(const double &x[], const int i) const
   {
      if(i < 0 || i >= ArraySize(x)) return 0.0;
      double v = x[i];
      if(!MathIsValidNumber(v)) return 0.0;
      return v;
   }

   double Dot(const double &w[], const double &f[]) const
   {
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      double z = w[0];
      for(int i=0; i<n; i++) z += w[i + 1] * f[i];
      return z;
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   void BuildGraphFeatures(const double &x[], double &f[]) const
   {
      int n = ArraySize(f);
      for(int i=0; i<n; i++) f[i] = 0.0;
      if(n <= 0) return;

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_WORLD, ContextSequenceCap(64, 40));
      dims.stride = MathMax(dims.stride, 2);
      dims.patch_size = MathMax(dims.patch_size, 2);
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      int seq_mask[];
      double seq_pos_bias[];
      BuildPackedSequenceTensorConfigured(x, seq_cfg, seq, seq_len, seq_mask, seq_pos_bias);
      int last = (seq_len > 0 ? seq_len - 1 : 0);
      int prev = (seq_len > 1 ? seq_len - 2 : last);
      int root = 0;

      double attn[FXAI_AI_WEIGHTS];
      double conv_fast[FXAI_AI_WEIGHTS];
      double conv_slow[FXAI_AI_WEIGHTS];
      double block[FXAI_AI_WEIGHTS];
      double k_fast[3] = {0.58, 0.27, 0.15};
      double k_slow[5] = {0.34, 0.24, 0.18, 0.14, 0.10};
      BuildSequenceBlockSummaries(x, dims, seq_cfg, k_fast, 3, k_slow, 5, attn, conv_fast, conv_slow, block);

      int base_n = MathMin(12, n);
      for(int i=0; i<base_n; i++)
         f[i] = FXAI_Clamp(seq[last][i], -10.0, 10.0);

      double r1  = seq[last][0];
      double r5  = seq[last][1];
      double r15 = seq[last][2];
      double r60 = seq[last][3];
      double vol = MathAbs(seq[last][4]) + 1e-6;
      double cost = MathAbs(seq[last][7]);
      double carry1 = r1 - r5;
      double carry2 = r5 - r15;
      double accel  = r1 - 2.0 * r5 + r15;
      double trend  = 0.40 * r1 + 0.30 * r5 + 0.20 * r15 + 0.10 * r60;
      double edge   = trend / vol;
      double graph1 = carry1 + carry2;
      double graph2 = accel / vol;
      double graph3 = (MathAbs(trend) - cost) / MathMax(vol, 1e-6);
      double lag_flow = seq[last][0] - seq[prev][0];
      double long_span = seq[last][0] - seq[root][0];
      double ctx1 = attn[13];
      double ctx2 = conv_fast[14];
      double ctx3 = conv_slow[15];
      double ctx4 = seq[last][16];
      double ctx_mean = (ctx1 + ctx2 + ctx3) / 3.0;
      double ctx_disp = (MathAbs(ctx1 - ctx_mean) + MathAbs(ctx2 - ctx_mean) + MathAbs(ctx3 - ctx_mean)) / 3.0;
      double rel_spread = trend - ctx_mean;
      double ctx_flow = 0.45 * (ctx1 - ctx2) + 0.30 * (ctx2 - ctx3) + 0.20 * ctx4;

      int k = base_n;
      if(k < n) f[k++] = FXAI_Clamp(lag_flow, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(long_span, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(carry1, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(carry2, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(accel, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(edge, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph1, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph2, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph3, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(rel_spread, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(ctx_disp, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(ctx_flow, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(attn[1] + 0.25 * block[1], -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(conv_fast[1] + 0.25 * block[2], -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(conv_slow[1] + 0.25 * block[3], -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(m_graph_bias, -10.0, 10.0);
   }

   void UpdateLinear(double &w[], const double &f[], const double target, const double pred, const double lr, const double l2)
   {
      double err = target - pred;
      w[0] += lr * (err - l2 * w[0]);
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++)
         w[i + 1] += lr * (err * f[i] - l2 * w[i + 1]);
   }

protected:
   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(xa, f);
      return FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(xa, f);
      double mu = MathMax(0.0, Dot(m_move_mu_w, f));
      double logv = FXAI_Clamp(Dot(m_move_logv_w, f), -4.0, 4.0);
      double sigma = MathSqrt(MathExp(logv));
      double ev = MathMax(mu + 0.25 * sigma, 0.0);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.70 * ev + 0.30 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }

   virtual void TrainModelCore(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(xa, f);

      double lr = FXAI_Clamp(hp.lr, 0.0001, 0.05);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.10);
      double p_trade = FXAI_Clamp(FXAI_Sigmoid(Dot(m_gate_w, f)), 0.001, 0.999);
      double p_up    = FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
      double mu_move = MathMax(0.0, Dot(m_move_mu_w, f));
      double logv_move = FXAI_Clamp(Dot(m_move_logv_w, f), -4.0, 4.0);
      double var_move = MathExp(logv_move);

      double t_trade = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      double t_up    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
      double move_t  = MathAbs(move_points);
      double predicted_side = (p_up >= 0.5 ? 1.0 : -1.0);
      double realized_side = (y == (int)FXAI_LABEL_BUY ? 1.0 : (y == (int)FXAI_LABEL_SELL ? -1.0 : 0.0));
      double hit = (realized_side == 0.0 ? (p_trade < 0.5 ? 1.0 : 0.0) : (predicted_side == realized_side ? 1.0 : 0.0));
      double p_buy = p_trade * p_up;
      double p_sell = p_trade * (1.0 - p_up);
      double p_skip = 1.0 - p_trade;
      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_SELL] = FXAI_Clamp(p_sell, 0.0005, 0.9990);
      p_raw3[(int)FXAI_LABEL_BUY]  = FXAI_Clamp(p_buy, 0.0005, 0.9990);
      p_raw3[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(p_skip, 0.0005, 0.9990);
      NormalizeClassDistribution(p_raw3);
      m_cal3.Update(p_raw3, NormalizeClassLabel(y, x, move_points), FXAI_Clamp(MoveSampleWeight(x, move_points), 0.20, 4.00), lr);

      UpdateLinear(m_gate_w, f, t_trade, p_trade, lr, l2);
      if(y != (int)FXAI_LABEL_SKIP)
         UpdateLinear(m_dir_w, f, t_up, p_up, lr, l2);

      double err_move = FXAI_Clamp(move_t - mu_move, -50.0, 50.0);
      m_move_mu_w[0] += lr * 0.25 * (err_move - l2 * m_move_mu_w[0]);
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++)
      {
         m_move_mu_w[i + 1] += lr * 0.25 * (err_move * f[i] - l2 * m_move_mu_w[i + 1]);
         double grad_lv = ((err_move * err_move) - var_move) * f[i];
         m_move_logv_w[i + 1] += lr * 0.02 * (grad_lv - l2 * m_move_logv_w[i + 1]);
      }
      m_move_logv_w[0] += lr * 0.02 * (((err_move * err_move) - var_move) - l2 * m_move_logv_w[0]);

      m_graph_bias = 0.995 * m_graph_bias + 0.005 * FXAI_Clamp(SafeX(x,0) - SafeX(x,1), -5.0, 5.0);
      m_reliability_ema = 0.985 * m_reliability_ema + 0.015 * hit;
      m_err_ema = 0.985 * m_err_ema + 0.015 * MathAbs(err_move) / MathMax(move_t + 1.0, 1.0);
      m_quality_heads.Update(xa,
                             FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 4.00),
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
      m_steps++;
   }

public:
   CFXAIAIGraphWM(void)
   {
      m_init = false;
      m_feat_n = 32;
      m_steps = 0;
      m_graph_bias = 0.0;
      ArrayInitialize(m_gate_w, 0.0);
      ArrayInitialize(m_dir_w, 0.0);
      ArrayInitialize(m_move_mu_w, 0.0);
      ArrayInitialize(m_move_logv_w, 0.0);
      m_last_probs[0] = m_last_probs[1] = m_last_probs[2] = 1.0/3.0;
      m_reliability_ema = 0.50;
      m_err_ema = 0.0;
      m_quality_heads.Reset();
      m_cal3.Reset();
   }

   virtual int AIId(void) const { return (int)AI_GRAPHWM; }
   virtual string AIName(void) const { return "wm_graph"; }
   virtual int PersistentStateVersion(void) const { return 9; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST|FXAI_CAP_WINDOW_CONTEXT);
      FillManifest(out, (int)FXAI_FAMILY_WORLD_MODEL, caps, 8, 64);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_graph_bias = 0.0;
      ArrayInitialize(m_gate_w, 0.0);
      ArrayInitialize(m_dir_w, 0.0);
      ArrayInitialize(m_move_mu_w, 0.0);
      ArrayInitialize(m_move_logv_w, 0.0);
      m_last_probs[0] = m_last_probs[1] = m_last_probs[2] = 1.0/3.0;
      m_reliability_ema = 0.50;
      m_err_ema = 0.0;
      m_quality_heads.Reset();
      m_cal3.Reset();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_init) return;
      m_feat_n = MathMin(48, MathMax(16, FXAI_AI_WEIGHTS - 1));
      m_init = true;
   }


   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(xa, f);

      double p_trade = FXAI_Clamp(FXAI_Sigmoid(Dot(m_gate_w, f)), 0.001, 0.999);
      double p_up    = FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
      double mu = MathMax(0.0, Dot(m_move_mu_w, f));
      double logv = FXAI_Clamp(Dot(m_move_logv_w, f), -4.0, 4.0);
      double sigma = MathSqrt(MathExp(logv));
      expected_move_points = MathMax(mu + 0.25 * sigma, 0.0);

      double p_buy  = p_trade * p_up;
      double p_sell = p_trade * (1.0 - p_up);
      double p_skip = 1.0 - p_trade;

      double move_gate = FXAI_Clamp(expected_move_points / MathMax(ResolveMinMovePoints(), 0.10), 0.0, 1.5);
      double reliability_gate = FXAI_Clamp(0.55 + 0.90 * (m_reliability_ema - 0.5) - 0.25 * m_err_ema, 0.20, 1.20);
      p_buy  *= MathMin(move_gate, 1.0) * reliability_gate;
      p_sell *= MathMin(move_gate, 1.0) * reliability_gate;
      p_skip  = MathMax(0.001, 1.0 - (p_buy + p_sell));

      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_SELL] = FXAI_Clamp(p_sell, 0.0005, 0.9990);
      p_raw3[(int)FXAI_LABEL_BUY]  = FXAI_Clamp(p_buy,  0.0005, 0.9990);
      p_raw3[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(p_skip, 0.0005, 0.9990);
      m_cal3.Calibrate(p_raw3, class_probs);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<3; i++)
      {
         class_probs[i] /= s;
         m_last_probs[i] = class_probs[i];
      }
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return false;

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(xa, f);
      double mu = MathMax(0.0, Dot(m_move_mu_w, f));
      double logv = FXAI_Clamp(Dot(m_move_logv_w, f), -4.0, 4.0);
      double sigma = MathSqrt(MathExp(logv));

      for(int c=0; c<3; c++)
         out.class_probs[c] = probs[c];
      out.move_mean_points = MathMax(ev, (m_move_ready ? m_move_ema_abs : mu));
      out.move_q25_points = MathMax(0.0, mu - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, mu);
      out.move_q75_points = MathMax(out.move_q50_points, mu + 0.55 * sigma);
      out.confidence = FXAI_Clamp(0.60 * MathMax(probs[(int)FXAI_LABEL_BUY], probs[(int)FXAI_LABEL_SELL]) + 0.20 * (1.0 - probs[(int)FXAI_LABEL_SKIP]) + 0.20 * m_reliability_ema, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.55 + 0.25 * m_reliability_ema + 0.20 * (1.0 - FXAI_Clamp(m_err_ema, 0.0, 1.0)), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
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

   virtual bool SaveModelState(const int handle) const
   {
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, (m_init ? 1 : 0));
      FileWriteInteger(handle, m_feat_n);
      FileWriteInteger(handle, m_steps);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         FileWriteDouble(handle, m_gate_w[k]);
         FileWriteDouble(handle, m_dir_w[k]);
         FileWriteDouble(handle, m_move_mu_w[k]);
         FileWriteDouble(handle, m_move_logv_w[k]);
      }
      FileWriteDouble(handle, m_graph_bias);
      for(int c=0; c<3; c++)
         FileWriteDouble(handle, m_last_probs[c]);
      FileWriteDouble(handle, m_reliability_ema);
      FileWriteDouble(handle, m_err_ema);
      if(!m_cal3.Save(handle))
         return false;
      return m_quality_heads.Save(handle);
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      if(handle == INVALID_HANDLE || version < 8)
         return false;

      m_init = (FileReadInteger(handle) != 0);
      m_feat_n = FileReadInteger(handle);
      m_steps = FileReadInteger(handle);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         m_gate_w[k] = FileReadDouble(handle);
         m_dir_w[k] = FileReadDouble(handle);
         m_move_mu_w[k] = FileReadDouble(handle);
         m_move_logv_w[k] = FileReadDouble(handle);
      }
      m_graph_bias = FileReadDouble(handle);
      for(int c=0; c<3; c++)
         m_last_probs[c] = FileReadDouble(handle);
      m_reliability_ema = FileReadDouble(handle);
      m_err_ema = FileReadDouble(handle);
      if(!m_cal3.Load(handle))
         return false;
      return m_quality_heads.Load(handle);
   }
};

#endif
