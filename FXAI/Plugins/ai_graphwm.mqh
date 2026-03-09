// FXAI v2
#ifndef __FXAI_AI_GRAPHWM_MQH__
#define __FXAI_AI_GRAPHWM_MQH__

#include "..\plugin_base.mqh"

class CFXAIAIGraphWM : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_feat_n;
   int    m_steps;
   double m_gate_w[FXAI_AI_WEIGHTS];
   double m_dir_w[FXAI_AI_WEIGHTS];
   double m_move_head_w[FXAI_AI_WEIGHTS];
   double m_graph_bias;
   double m_last_probs[3];

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

   void BuildGraphFeatures(const double &x[], double &f[]) const
   {
      int n = ArraySize(f);
      for(int i=0; i<n; i++) f[i] = 0.0;
      if(n <= 0) return;

      int xn = ArraySize(x);
      int base_n = MathMin(xn, n - 8);
      for(int i=0; i<base_n; i++)
         f[i] = FXAI_Clamp(SafeX(x, i), -10.0, 10.0);

      double r1  = SafeX(x, 0);
      double r5  = SafeX(x, 1);
      double r15 = SafeX(x, 2);
      double r60 = SafeX(x, 3);
      double vol = MathAbs(SafeX(x, 4)) + 1e-6;
      double cost = MathAbs(SafeX(x, 7));
      double carry1 = r1 - r5;
      double carry2 = r5 - r15;
      double accel  = r1 - 2.0 * r5 + r15;
      double trend  = 0.40 * r1 + 0.30 * r5 + 0.20 * r15 + 0.10 * r60;
      double edge   = trend / vol;
      double graph1 = carry1 + carry2;
      double graph2 = accel / vol;
      double graph3 = (MathAbs(trend) - cost) / MathMax(vol, 1e-6);

      int k = base_n;
      if(k < n) f[k++] = FXAI_Clamp(carry1, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(carry2, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(accel, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(edge, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph1, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph2, -10.0, 10.0);
      if(k < n) f[k++] = FXAI_Clamp(graph3, -10.0, 10.0);
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
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(x, f);
      return FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(x, f);
      double raw = MathAbs(Dot(m_move_head_w, f));
      return MathMax(raw, 0.0);
   }

   virtual void UpdateWithMove(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureInitialized(hp);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(x, f);

      double lr = FXAI_Clamp(hp.lr, 0.0001, 0.05);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.10);
      double p_trade = FXAI_Clamp(FXAI_Sigmoid(Dot(m_gate_w, f)), 0.001, 0.999);
      double p_up    = FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
      double p_move  = Dot(m_move_head_w, f);

      double t_trade = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      double t_up    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
      double move_t  = MathAbs(move_points);

      UpdateLinear(m_gate_w, f, t_trade, p_trade, lr, l2);
      if(y != (int)FXAI_LABEL_SKIP)
         UpdateLinear(m_dir_w, f, t_up, p_up, lr, l2);

      double err_move = FXAI_Clamp(move_t - p_move, -50.0, 50.0);
      m_move_head_w[0] += lr * 0.25 * (err_move - l2 * m_move_head_w[0]);
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++)
         m_move_head_w[i + 1] += lr * 0.25 * (err_move * f[i] - l2 * m_move_head_w[i + 1]);

      m_graph_bias = 0.995 * m_graph_bias + 0.005 * FXAI_Clamp(SafeX(x,0) - SafeX(x,1), -5.0, 5.0);
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
      ArrayInitialize(m_move_head_w, 0.0);
      m_last_probs[0] = m_last_probs[1] = m_last_probs[2] = 1.0/3.0;
   }

   virtual int AIId(void) const { return (int)AI_GRAPHWM; }
   virtual string AIName(void) const { return "graphwm"; }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_graph_bias = 0.0;
      ArrayInitialize(m_gate_w, 0.0);
      ArrayInitialize(m_dir_w, 0.0);
      ArrayInitialize(m_move_head_w, 0.0);
      m_last_probs[0] = m_last_probs[1] = m_last_probs[2] = 1.0/3.0;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_init) return;
      m_feat_n = MathMin(48, MathMax(16, FXAI_AI_WEIGHTS - 1));
      m_init = true;
   }

   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureInitialized(hp);
      double f[]; ArrayResize(f, m_feat_n);
      BuildGraphFeatures(x, f);

      double p_trade = FXAI_Clamp(FXAI_Sigmoid(Dot(m_gate_w, f)), 0.001, 0.999);
      double p_up    = FXAI_Clamp(FXAI_Sigmoid(Dot(m_dir_w, f)), 0.001, 0.999);
      expected_move_points = MathMax(MathAbs(Dot(m_move_head_w, f)), 0.0);

      double p_buy  = p_trade * p_up;
      double p_sell = p_trade * (1.0 - p_up);
      double p_skip = 1.0 - p_trade;

      double move_gate = FXAI_Clamp(expected_move_points / MathMax(ResolveMinMovePoints(), 0.10), 0.0, 1.5);
      p_buy  *= MathMin(move_gate, 1.0);
      p_sell *= MathMin(move_gate, 1.0);
      p_skip  = MathMax(0.001, 1.0 - (p_buy + p_sell));

      class_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(p_sell, 0.0005, 0.9990);
      class_probs[(int)FXAI_LABEL_BUY]  = FXAI_Clamp(p_buy,  0.0005, 0.9990);
      class_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(p_skip, 0.0005, 0.9990);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<3; i++)
      {
         class_probs[i] /= s;
         m_last_probs[i] = class_probs[i];
      }
      return true;
   }
};

#endif
