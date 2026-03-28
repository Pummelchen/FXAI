#ifndef __FXAI_AI_MOE_CONFORMAL_MQH__
#define __FXAI_AI_MOE_CONFORMAL_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_MOE_E 4
#define FXAI_MOE_BUCKETS 12
#define FXAI_MOE_BUCKET_DEPTH 64

class CFXAIAIMoEConformal : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_feat_n;
   int    m_steps;
   int    m_cal_n;
   double m_router[FXAI_MOE_E][12];
   double m_gate[FXAI_MOE_E][FXAI_AI_WEIGHTS];
   double m_dir[FXAI_MOE_E][FXAI_AI_WEIGHTS];
   double m_move[FXAI_MOE_E][FXAI_AI_WEIGHTS];
   double m_scores[128];
   double m_bucket_scores[FXAI_MOE_BUCKETS][FXAI_MOE_BUCKET_DEPTH];
   int    m_bucket_counts[FXAI_MOE_BUCKETS];
   double m_usage_ema[FXAI_MOE_E];
   double m_cal_w[3][5];

   double SX(const double &x[], int i) const
   {
      if(i < 0 || i >= ArraySize(x)) return 0.0;
      double v = x[i];
      if(!MathIsValidNumber(v)) return 0.0;
      return v;
   }

   void BuildRegime(const double &x[], double &r[]) const
   {
      ArrayResize(r, 11);
      double r1 = SX(x,0), r5 = SX(x,1), r15 = SX(x,2), r60 = SX(x,3), vol = MathAbs(SX(x,4));
      r[0] = 1.0;
      r[1] = FXAI_Clamp(r1, -10.0, 10.0);
      r[2] = FXAI_Clamp(r5, -10.0, 10.0);
      r[3] = FXAI_Clamp(r15, -10.0, 10.0);
      r[4] = FXAI_Clamp(r60, -10.0, 10.0);
      r[5] = FXAI_Clamp(vol, 0.0, 10.0);
      r[6] = FXAI_Clamp(r1 - r5, -10.0, 10.0);
      r[7] = FXAI_Clamp(r5 - r15, -10.0, 10.0);
      r[8] = FXAI_Clamp((r1 + r5 + r15) / MathMax(vol, 1e-6), -10.0, 10.0);
      r[9] = FXAI_Clamp(SX(x,5), -10.0, 10.0);
      r[10]= FXAI_Clamp(SX(x,6), -10.0, 10.0);
   }

   void BuildFeat(const double &x[], double &f[]) const
   {
      ArrayResize(f, m_feat_n);
      for(int i=0; i<m_feat_n; i++) f[i] = 0.0;
      int n = MathMin(ArraySize(x), m_feat_n);
      for(int i=0; i<n; i++) f[i] = FXAI_Clamp(SX(x,i), -10.0, 10.0);
   }

   int SessionBucket(void) const
   {
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      if(dt.hour < 6) return 0;
      if(dt.hour < 12) return 1;
      if(dt.hour < 17) return 2;
      return 3;
   }

   int RegimeBucket(const double &x[]) const
   {
      double vol = MathAbs(SX(x, 4));
      if(vol < 0.75) return 0;
      if(vol < 1.75) return 1;
      return 2;
   }

   int BucketIndex(const double &x[]) const
   {
      return 3 * SessionBucket() + RegimeBucket(x);
   }

   double DotRouter(const int expert, const double &r[]) const
   {
      double z = 0.0;
      for(int i=0; i<ArraySize(r); i++) z += m_router[expert][i] * r[i];
      return z;
   }

   double DotGate(const int expert, const double &f[]) const
   {
      double z = m_gate[expert][0];
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++) z += m_gate[expert][i + 1] * f[i];
      return z;
   }

   double DotDir(const int expert, const double &f[]) const
   {
      double z = m_dir[expert][0];
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++) z += m_dir[expert][i + 1] * f[i];
      return z;
   }

   double DotMove(const int expert, const double &f[]) const
   {
      double z = m_move[expert][0];
      int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
      for(int i=0; i<n; i++) z += m_move[expert][i + 1] * f[i];
      return z;
   }

   void RouterSoftmax(const double &r[], double &g[]) const
   {
      ArrayResize(g, FXAI_MOE_E);
      double z[FXAI_MOE_E];
      double m = -DBL_MAX;
      for(int e=0; e<FXAI_MOE_E; e++)
      {
         z[e] = DotRouter(e, r) - 0.35 * (m_usage_ema[e] - (1.0 / (double)FXAI_MOE_E));
         if(z[e] > m) m = z[e];
      }
      double s = 0.0;
      for(int e=0; e<FXAI_MOE_E; e++) { g[e] = MathExp(FXAI_Clamp(z[e] - m, -30.0, 30.0)); s += g[e]; }
      if(s <= 0.0) s = 1.0;
      for(int e=0; e<FXAI_MOE_E; e++) g[e] /= s;
   }

   double Quantile90(const int bucket) const
   {
      int n = m_bucket_counts[bucket];
      if(n <= 8) return 0.40;
      double tmp[]; ArrayResize(tmp, n);
      for(int i=0; i<n; i++) tmp[i] = m_bucket_scores[bucket][i];
      ArraySort(tmp);
      int q = (int)MathFloor(0.90 * (n - 1));
      if(q < 0) q = 0;
      if(q >= n) q = n - 1;
      return tmp[q];
   }

   void ApplyCalibrator(const double &base_probs[], const double expected_move_points, double &out_probs[]) const
   {
      double mm = MathMax(ResolveMinMovePoints(), 0.10);
      double cp = (m_ctx_cost_ready && m_ctx_cost_points >= 0.0 ? m_ctx_cost_points : 0.0);
      double xc[5];
      xc[0] = 1.0;
      xc[1] = FXAI_Clamp(base_probs[(int)FXAI_LABEL_BUY] - base_probs[(int)FXAI_LABEL_SELL], -1.0, 1.0);
      xc[2] = FXAI_Clamp(base_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      xc[3] = FXAI_Clamp(expected_move_points / mm, 0.0, 12.0);
      xc[4] = FXAI_Clamp(cp / mm, 0.0, 4.0);

      double logits[3];
      for(int c=0; c<3; c++)
      {
         logits[c] = 0.0;
         for(int k=0; k<5; k++) logits[c] += m_cal_w[c][k] * xc[k];
         logits[c] += MathLog(MathMax(base_probs[c], 1e-6));
      }
      Softmax3(logits, out_probs);
   }

   void UpdateCalibrator3(const int label, const double &base_probs[], const double expected_move_points, const FXAIAIHyperParams &hp)
   {
      double mm = MathMax(ResolveMinMovePoints(), 0.10);
      double cp = (m_ctx_cost_ready && m_ctx_cost_points >= 0.0 ? m_ctx_cost_points : 0.0);
      double xc[5];
      xc[0] = 1.0;
      xc[1] = FXAI_Clamp(base_probs[(int)FXAI_LABEL_BUY] - base_probs[(int)FXAI_LABEL_SELL], -1.0, 1.0);
      xc[2] = FXAI_Clamp(base_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      xc[3] = FXAI_Clamp(expected_move_points / mm, 0.0, 12.0);
      xc[4] = FXAI_Clamp(cp / mm, 0.0, 4.0);

      double probs[3];
      ApplyCalibrator(base_probs, expected_move_points, probs);
      double lr = FXAI_Clamp(0.25 * hp.lr, 0.0002, 0.02);
      for(int c=0; c<3; c++)
      {
         double tgt = (c == label ? 1.0 : 0.0);
         double err = tgt - probs[c];
         for(int k=0; k<5; k++)
            m_cal_w[c][k] += lr * (err * xc[k] - 0.002 * m_cal_w[c][k]);
      }
   }

protected:
   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3]; double em = 0.0;
      PredictModelCore(x, hp, probs, em);
      return probs[(int)FXAI_LABEL_BUY];
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3]; double em = 0.0;
      PredictModelCore(x, hp, probs, em);
      return em;
   }

   virtual void TrainModelCore(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureInitialized(hp);
      double r[]; BuildRegime(x, r);
      double f[]; BuildFeat(x, f);
      double g[]; RouterSoftmax(r, g);
      double lr = FXAI_Clamp(hp.lr, 0.0001, 0.03);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.10);
      double router_loss[FXAI_MOE_E];
      double router_total = 0.0;

      for(int e=0; e<FXAI_MOE_E; e++)
      {
         double ge = g[e];
         double p_trade = FXAI_Clamp(FXAI_Sigmoid(DotGate(e, f)), 0.001, 0.999);
         double p_up    = FXAI_Clamp(FXAI_Sigmoid(DotDir(e, f)), 0.001, 0.999);
         double p_move  = DotMove(e, f);
         double t_trade = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
         double t_up    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
         double err_trade = (t_trade - p_trade) * ge;
         double err_up    = (t_up - p_up) * ge;
         double err_move  = (MathAbs(move_points) - p_move) * ge;
         router_loss[e] = MathAbs(t_trade - p_trade) + (y == (int)FXAI_LABEL_SKIP ? 0.0 : MathAbs(t_up - p_up)) + 0.10 * MathAbs(err_move);
         router_total += 1.0 / MathMax(router_loss[e], 0.05);

         m_gate[e][0] += lr * (err_trade - l2 * m_gate[e][0]);
         m_dir[e][0]  += lr * (err_up    - l2 * m_dir[e][0]);
         m_move[e][0] += lr * 0.20 * (err_move - l2 * m_move[e][0]);
         int n = MathMin(ArraySize(f), FXAI_AI_WEIGHTS - 1);
         for(int i=0; i<n; i++)
         {
            m_gate[e][i + 1] += lr * (err_trade * f[i] - l2 * m_gate[e][i + 1]);
            if(y != (int)FXAI_LABEL_SKIP)
               m_dir[e][i + 1] += lr * (err_up * f[i] - l2 * m_dir[e][i + 1]);
            m_move[e][i + 1] += lr * 0.20 * (err_move * f[i] - l2 * m_move[e][i + 1]);
         }
      }

      for(int e=0; e<FXAI_MOE_E; e++)
      {
         double reward = (1.0 / MathMax(router_loss[e], 0.05)) / MathMax(router_total, 1e-6);
         m_usage_ema[e] = 0.985 * m_usage_ema[e] + 0.015 * g[e];
         for(int i=0; i<ArraySize(r); i++)
            m_router[e][i] += lr * 0.10 * ((reward - g[e]) * r[i] - 0.002 * m_router[e][i]);
      }

      double probs[3]; double em = 0.0;
      PredictModelCore(x, hp, probs, em);
      double p_true = probs[(y == (int)FXAI_LABEL_BUY ? (int)FXAI_LABEL_BUY : (y == (int)FXAI_LABEL_SELL ? (int)FXAI_LABEL_SELL : (int)FXAI_LABEL_SKIP))];
      double score = 1.0 - FXAI_Clamp(p_true, 0.0005, 0.9990);
      m_scores[m_cal_n % 128] = score;
      if(m_cal_n < 128) m_cal_n++;
      int bucket = BucketIndex(x);
      int slot = m_bucket_counts[bucket] % FXAI_MOE_BUCKET_DEPTH;
      m_bucket_scores[bucket][slot] = score;
      if(m_bucket_counts[bucket] < FXAI_MOE_BUCKET_DEPTH) m_bucket_counts[bucket]++;
      UpdateCalibrator3((y >= 0 && y <= 2 ? y : (int)FXAI_LABEL_SKIP), probs, em, hp);
      UpdateNativeQualityHeads(x, FXAI_Clamp(MoveSampleWeight(x, move_points), 0.20, 4.00), hp.lr, hp.l2);
      m_steps++;
   }

public:
   CFXAIAIMoEConformal(void)
   {
      m_init = false;
      m_feat_n = 24;
      m_steps = 0;
      m_cal_n = 0;
      ArrayInitialize(m_scores, 0.40);
      ArrayInitialize(m_bucket_scores, 0.40);
      ArrayInitialize(m_bucket_counts, 0);
      ArrayInitialize(m_usage_ema, 1.0 / (double)FXAI_MOE_E);
      ArrayInitialize(m_router, 0.0);
      ArrayInitialize(m_gate, 0.0);
      ArrayInitialize(m_dir, 0.0);
      ArrayInitialize(m_move, 0.0);
      ArrayInitialize(m_cal_w, 0.0);
   }

   virtual int AIId(void) const { return (int)AI_MOE_CONFORMAL; }
   virtual string AIName(void) const { return "mix_moe_conformal"; }
   virtual int PersistentStateVersion(void) const { return 10; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_MIXTURE, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_cal_n = 0;
      ArrayInitialize(m_scores, 0.40);
      ArrayInitialize(m_bucket_scores, 0.40);
      ArrayInitialize(m_bucket_counts, 0);
      ArrayInitialize(m_usage_ema, 1.0 / (double)FXAI_MOE_E);
      ArrayInitialize(m_router, 0.0);
      ArrayInitialize(m_gate, 0.0);
      ArrayInitialize(m_dir, 0.0);
      ArrayInitialize(m_move, 0.0);
      ArrayInitialize(m_cal_w, 0.0);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_init) return;
      m_feat_n = MathMin(32, MathMax(16, FXAI_AI_WEIGHTS - 1));
      for(int e=0; e<FXAI_MOE_E; e++) m_router[e][e + 1] = 0.10;
      m_init = true;
   }


   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureInitialized(hp);
      double r[]; BuildRegime(x, r);
      double f[]; BuildFeat(x, f);
      double g[]; RouterSoftmax(r, g);

      double p_trade = 0.0, p_up = 0.0, p_move = 0.0;
      for(int e=0; e<FXAI_MOE_E; e++)
      {
         double ge = g[e];
         p_trade += ge * FXAI_Clamp(FXAI_Sigmoid(DotGate(e, f)), 0.001, 0.999);
         p_up    += ge * FXAI_Clamp(FXAI_Sigmoid(DotDir(e, f)),  0.001, 0.999);
         p_move  += ge * MathAbs(DotMove(e, f));
      }
      expected_move_points = MathMax(0.0, p_move);

      int bucket = BucketIndex(x);
      double q = Quantile90(bucket);
      double p_buy = p_trade * p_up;
      double p_sell = p_trade * (1.0 - p_up);
      double p_skip = 1.0 - p_trade;

      bool allow_buy = (1.0 - p_buy) <= q;
      bool allow_sell = (1.0 - p_sell) <= q;
      if(allow_buy == allow_sell)
      {
         p_skip = MathMax(p_skip, 0.55);
         p_buy *= 0.50;
         p_sell *= 0.50;
      }

      double raw_probs[3];
      raw_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(p_sell, 0.0005, 0.9990);
      raw_probs[(int)FXAI_LABEL_BUY]  = FXAI_Clamp(p_buy,  0.0005, 0.9990);
      raw_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(p_skip, 0.0005, 0.9990);
      ApplyCalibrator(raw_probs, expected_move_points, class_probs);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<3; i++) class_probs[i] /= s;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double expected = 0.0;
      if(!PredictModelCore(x, hp, out.class_probs, expected))
         return false;
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = MathMax(0.0, expected);
      double sigma = MathMax(0.10, 0.30 * out.move_mean_points + 0.25 * (m_steps > 0 ? 1.0 : 0.0));
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * MathMin((double)m_steps / 64.0, 1.0) + 0.20 * (1.0 - out.class_probs[(int)FXAI_LABEL_SKIP]), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
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
      FileWriteInteger(handle, m_cal_n);
      for(int e=0; e<FXAI_MOE_E; e++)
      {
         for(int k=0; k<12; k++)
            FileWriteDouble(handle, m_router[e][k]);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            FileWriteDouble(handle, m_gate[e][k]);
            FileWriteDouble(handle, m_dir[e][k]);
            FileWriteDouble(handle, m_move[e][k]);
         }
         FileWriteDouble(handle, m_usage_ema[e]);
      }
      for(int i=0; i<128; i++)
         FileWriteDouble(handle, m_scores[i]);
      for(int b=0; b<FXAI_MOE_BUCKETS; b++)
      {
         FileWriteInteger(handle, m_bucket_counts[b]);
         for(int d=0; d<FXAI_MOE_BUCKET_DEPTH; d++)
            FileWriteDouble(handle, m_bucket_scores[b][d]);
      }
      for(int c=0; c<3; c++)
         for(int k=0; k<5; k++)
            FileWriteDouble(handle, m_cal_w[c][k]);
      return true;
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      if(handle == INVALID_HANDLE || version < 8)
         return false;

      m_init = (FileReadInteger(handle) != 0);
      m_feat_n = FileReadInteger(handle);
      m_steps = FileReadInteger(handle);
      m_cal_n = FileReadInteger(handle);
      for(int e=0; e<FXAI_MOE_E; e++)
      {
         for(int k=0; k<12; k++)
            m_router[e][k] = FileReadDouble(handle);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            m_gate[e][k] = FileReadDouble(handle);
            m_dir[e][k] = FileReadDouble(handle);
            m_move[e][k] = FileReadDouble(handle);
         }
         m_usage_ema[e] = FileReadDouble(handle);
      }
      for(int i=0; i<128; i++)
         m_scores[i] = FileReadDouble(handle);
      for(int b=0; b<FXAI_MOE_BUCKETS; b++)
      {
         m_bucket_counts[b] = FileReadInteger(handle);
         for(int d=0; d<FXAI_MOE_BUCKET_DEPTH; d++)
            m_bucket_scores[b][d] = FileReadDouble(handle);
      }
      for(int c=0; c<3; c++)
         for(int k=0; k<5; k++)
            m_cal_w[c][k] = FileReadDouble(handle);
      return true;
   }
};

#endif
