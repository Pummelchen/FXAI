// FXAI v2
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

protected:
   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3]; double em = 0.0;
      PredictNativeClassProbs(x, hp, probs, em);
      return probs[(int)FXAI_LABEL_BUY];
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3]; double em = 0.0;
      PredictNativeClassProbs(x, hp, probs, em);
      return em;
   }

   virtual void UpdateWithMove(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureInitialized(hp);
      double e[]; Embed(x, e);
      int pos = m_head;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++) m_emb[pos][i] = e[i];
      m_future_move[pos]  = MathAbs(move_points);
      m_future_up[pos]    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
      m_future_event[pos] = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      m_head = (m_head + 1) % FXAI_RETRDIFF_MEM;
      if(m_count < FXAI_RETRDIFF_MEM) m_count++;
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
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         for(int j=0; j<8; j++)
            m_R[i][j] = 0.0;
   }

   virtual int AIId(void) const { return (int)AI_RETRDIFF; }
   virtual string AIName(void) const { return "retrdiff"; }

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

   virtual bool SupportsNativeClassProbs(void) const { return true; }

   virtual bool PredictNativeClassProbs(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
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
      for(int k=0; k<3; k++)
      {
         int idx = (int)ids[k];
         if(idx < 0) continue;
         double w = 1.0 / MathMax(0.05, ds[k]);
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
