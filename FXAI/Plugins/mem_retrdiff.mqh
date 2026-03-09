#ifndef __FXAI_AI_RETRDIFF_MQH__
#define __FXAI_AI_RETRDIFF_MQH__

#include "..\plugin_base.mqh"

#define FXAI_RETRDIFF_MEM 256
#define FXAI_RETRDIFF_EMB 16

class CFXAIAIRetrDiff : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_steps;
   int    m_head;
   int    m_count;
   double m_R[FXAI_RETRDIFF_EMB][8];
   double m_emb[FXAI_RETRDIFF_MEM][FXAI_RETRDIFF_EMB];
   double m_future_move[FXAI_RETRDIFF_MEM];
   double m_future_up[FXAI_RETRDIFF_MEM];
   double m_future_event[FXAI_RETRDIFF_MEM];
   datetime m_sample_time[FXAI_RETRDIFF_MEM];
   double m_regime_vol[FXAI_RETRDIFF_MEM];
   double m_regime_dir[FXAI_RETRDIFF_MEM];
   double m_proto_weight[FXAI_RETRDIFF_MEM];

   double SX(const double &x[], const int i) const
   {
      if(i < 0 || i >= ArraySize(x)) return 0.0;
      double v = x[i];
      if(!MathIsValidNumber(v)) return 0.0;
      return v;
   }

   void BuildBase(const double &x[], double &b[]) const
   {
      ArrayResize(b, 8);
      b[0] = SX(x,0);
      b[1] = SX(x,1);
      b[2] = SX(x,2);
      b[3] = SX(x,3);
      b[4] = SX(x,4);
      b[5] = SX(x,5);
      b[6] = SX(x,6);
      b[7] = SX(x,7);
   }

   void LearnProjection(const double &b[],
                        const int label,
                        const double move_points,
                        const double pred_up,
                        const double pred_event,
                        const double pred_move)
   {
      double dir_t = (label == (int)FXAI_LABEL_BUY ? 1.0 : (label == (int)FXAI_LABEL_SELL ? -1.0 : 0.0));
      double evt_t = (label == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      double mm = MathMax(ResolveMinMovePoints(), 0.10);
      double mag_t = FXAI_Clamp(MathAbs(move_points) / mm, 0.0, 8.0);
      double dir_err = dir_t - (2.0 * pred_up - 1.0);
      double evt_err = evt_t - pred_event;
      double mag_err = mag_t - FXAI_Clamp(pred_move / mm, 0.0, 8.0);
      double lr = 0.0030;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
      {
         double signal = (i % 3 == 0 ? dir_err : (i % 3 == 1 ? evt_err : 0.5 * mag_err));
         for(int j=0; j<8; j++)
            m_R[i][j] = FXAI_ClipSym(0.999 * m_R[i][j] + lr * signal * b[j], 2.0);
      }
   }

   void Embed(const double &x[], double &e[]) const
   {
      double b[]; BuildBase(x, b);
      ArrayResize(e, FXAI_RETRDIFF_EMB);
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
      {
         double z = 0.0;
         for(int j=0; j<8; j++) z += m_R[i][j] * b[j];
         e[i] = z;
      }
      double nrm = 0.0;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++) nrm += e[i] * e[i];
      nrm = MathSqrt(MathMax(nrm, 1e-9));
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++) e[i] /= nrm;
   }

   double DistMem(const double &a[], const int pos) const
   {
      double d = 0.0;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
      {
         double z = a[i] - m_emb[pos][i];
         d += z * z;
      }
      return d;
   }

   void Retrieve(const double &e[], int &idx1, int &idx2, int &idx3, double &d1, double &d2, double &d3) const
   {
      idx1 = idx2 = idx3 = -1;
      d1 = d2 = d3 = DBL_MAX;
      for(int i=0; i<m_count; i++)
      {
         int pos = (m_head - 1 - i + FXAI_RETRDIFF_MEM) % FXAI_RETRDIFF_MEM;
         double d = DistMem(e, pos);
         if(d < d1) { d3=d2; idx3=idx2; d2=d1; idx2=idx1; d1=d; idx1=pos; }
         else if(d < d2) { d3=d2; idx3=idx2; d2=d; idx2=pos; }
         else if(d < d3) { d3=d; idx3=pos; }
      }
   }

   bool TryMergePrototype(const double &e[],
                          const int label,
                          const double move_points,
                          const datetime sample_time,
                          const double regime_vol,
                          const double regime_dir)
   {
      int i1, i2, i3;
      double d1, d2, d3;
      Retrieve(e, i1, i2, i3, d1, d2, d3);
      if(i1 < 0 || d1 > 0.08) return false;

      double up_t = (label == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
      double ev_t = (label == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      if(MathAbs(m_future_event[i1] - ev_t) > 0.5) return false;
      if(ev_t > 0.5 && MathAbs(m_future_up[i1] - up_t) > 0.5) return false;

      double w = MathMin(16.0, m_proto_weight[i1] + 1.0);
      double alpha = 1.0 / w;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         m_emb[i1][i] = (1.0 - alpha) * m_emb[i1][i] + alpha * e[i];
      m_future_move[i1] = (1.0 - alpha) * m_future_move[i1] + alpha * MathAbs(move_points);
      m_future_up[i1] = (1.0 - alpha) * m_future_up[i1] + alpha * up_t;
      m_future_event[i1] = (1.0 - alpha) * m_future_event[i1] + alpha * ev_t;
      m_regime_vol[i1] = (1.0 - alpha) * m_regime_vol[i1] + alpha * regime_vol;
      m_regime_dir[i1] = (1.0 - alpha) * m_regime_dir[i1] + alpha * regime_dir;
      m_sample_time[i1] = sample_time;
      m_proto_weight[i1] = w;
      return true;
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
      double e[]; Embed(x, e);
      double b[]; BuildBase(x, b);
      double pred_probs[3];
      double pred_move = 0.0;
      PredictModelCore(x, hp, pred_probs, pred_move);
      LearnProjection(b, y, move_points, pred_probs[(int)FXAI_LABEL_BUY], 1.0 - pred_probs[(int)FXAI_LABEL_SKIP], pred_move);

      datetime sample_time = ResolveContextTime();
      double regime_vol = MathAbs(SX(x, 4));
      double regime_dir = SX(x, 0) - SX(x, 1);
      if(!TryMergePrototype(e, y, move_points, sample_time, regime_vol, regime_dir))
      {
         int pos = m_head;
         for(int i=0; i<FXAI_RETRDIFF_EMB; i++) m_emb[pos][i] = e[i];
         m_future_move[pos]  = MathAbs(move_points);
         m_future_up[pos]    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
         m_future_event[pos] = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
         m_sample_time[pos]  = sample_time;
         m_regime_vol[pos]   = regime_vol;
         m_regime_dir[pos]   = regime_dir;
         m_proto_weight[pos] = 1.0;
         m_head = (m_head + 1) % FXAI_RETRDIFF_MEM;
         if(m_count < FXAI_RETRDIFF_MEM) m_count++;
      }
      m_steps++;
   }

public:
   CFXAIAIRetrDiff(void)
   {
      m_init = false;
      m_steps = 0;
      m_head = 0;
      m_count = 0;
      ArrayInitialize(m_future_move, 0.0);
      ArrayInitialize(m_future_up, 0.0);
      ArrayInitialize(m_future_event, 0.0);
      ArrayInitialize(m_regime_vol, 0.0);
      ArrayInitialize(m_regime_dir, 0.0);
      ArrayInitialize(m_proto_weight, 1.0);
      ArrayInitialize(m_sample_time, 0);
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         for(int j=0; j<8; j++)
            m_R[i][j] = 0.0;
   }

   virtual int AIId(void) const { return (int)AI_RETRDIFF; }
   virtual string AIName(void) const { return "mem_retrdiff"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RETRIEVAL, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_head = 0;
      m_count = 0;
      ArrayInitialize(m_future_move, 0.0);
      ArrayInitialize(m_future_up, 0.0);
      ArrayInitialize(m_future_event, 0.0);
      ArrayInitialize(m_regime_vol, 0.0);
      ArrayInitialize(m_regime_dir, 0.0);
      ArrayInitialize(m_proto_weight, 1.0);
      ArrayInitialize(m_sample_time, 0);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_init) return;
      MathSrand(1337);
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         for(int j=0; j<8; j++)
            m_R[i][j] = ((double)(MathRand() % 2001) / 1000.0) - 1.0;
      m_init = true;
   }


   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      EnsureInitialized(hp);
      if(m_count < 8)
      {
         class_probs[(int)FXAI_LABEL_SELL] = 0.25;
         class_probs[(int)FXAI_LABEL_BUY]  = 0.25;
         class_probs[(int)FXAI_LABEL_SKIP] = 0.50;
         expected_move_points = MathMax(ResolveMinMovePoints(), 0.10);
         return true;
      }

      double e[]; Embed(x, e);
      int i1,i2,i3; double d1,d2,d3;
      Retrieve(e, i1,i2,i3, d1,d2,d3);
      if(i1 < 0)
      {
         class_probs[(int)FXAI_LABEL_SELL] = 0.25;
         class_probs[(int)FXAI_LABEL_BUY]  = 0.25;
         class_probs[(int)FXAI_LABEL_SKIP] = 0.50;
         expected_move_points = MathMax(ResolveMinMovePoints(), 0.10);
         return true;
      }

      double ids[3] = { (double)i1, (double)i2, (double)i3 };
      double ds[3]  = { d1, d2, d3 };
      double sw = 0.0, up = 0.0, ev = 0.0, mv = 0.0;
      datetime now = ResolveContextTime();
      double cur_vol = MathAbs(SX(x, 4));
      double cur_dir = SX(x, 0) - SX(x, 1);
      for(int k=0; k<3; k++)
      {
         int idx = (int)ids[k];
         if(idx < 0) continue;
         double w = 1.0 / MathMax(0.05, ds[k]);
         double regime_sim = 1.0 / (1.0 + MathAbs(m_regime_vol[idx] - cur_vol) + 0.5 * MathAbs(m_regime_dir[idx] - cur_dir));
         double recency = 1.0;
         if(m_sample_time[idx] > 0 && now > m_sample_time[idx])
         {
            double age_min = (double)(now - m_sample_time[idx]) / 60.0;
            recency = 1.0 / (1.0 + age_min / 720.0);
         }
         w *= regime_sim * recency * MathSqrt(MathMax(m_proto_weight[idx], 1.0));
         sw += w;
         up += w * m_future_up[idx];
         ev += w * m_future_event[idx];
         mv += w * m_future_move[idx];
      }
      if(sw <= 0.0) sw = 1.0;
      up /= sw; ev /= sw; mv /= sw;

      double p_buy = ev * up;
      double p_sell = ev * (1.0 - up);
      double p_skip = 1.0 - ev;
      expected_move_points = MathMax(mv, ResolveMinMovePoints());

      class_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(p_sell, 0.0005, 0.9990);
      class_probs[(int)FXAI_LABEL_BUY]  = FXAI_Clamp(p_buy,  0.0005, 0.9990);
      class_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(p_skip, 0.0005, 0.9990);
      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int i=0; i<3; i++) class_probs[i] /= s;
      return true;
   }
};

#endif
