#ifndef __FXAI_AI_RETRDIFF_MQH__
#define __FXAI_AI_RETRDIFF_MQH__

#include "..\API\plugin_base.mqh"

#define FXAI_RETRDIFF_MEM 256
#define FXAI_RETRDIFF_EMB 16
#define FXAI_RETRDIFF_BASE 16
#define FXAI_RETRDIFF_TOPK 5

class CFXAIAIRetrDiff : public CFXAIAIPlugin
{
private:
   bool   m_init;
   int    m_steps;
   int    m_head;
   int    m_count;
   double m_R[FXAI_RETRDIFF_EMB][FXAI_RETRDIFF_BASE];
   double m_emb[FXAI_RETRDIFF_MEM][FXAI_RETRDIFF_EMB];
   double m_future_move[FXAI_RETRDIFF_MEM];
   double m_move_var[FXAI_RETRDIFF_MEM];
   double m_future_up[FXAI_RETRDIFF_MEM];
   double m_future_event[FXAI_RETRDIFF_MEM];
   double m_label_mass[FXAI_RETRDIFF_MEM][3];
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
      ArrayResize(b, FXAI_RETRDIFF_BASE);
      b[0] = SX(x,0);
      b[1] = SX(x,1);
      b[2] = SX(x,2);
      b[3] = SX(x,3);
      b[4] = SX(x,4);
      b[5] = SX(x,5);
      b[6] = SX(x,6);
      b[7] = SX(x,7);

      int win_n = CurrentWindowSize();
      double mean1 = 0.0, mean2 = 0.0, mean4 = 0.0;
      double var1 = 0.0, var2 = 0.0;
      double trend1 = 0.0, trend2 = 0.0;
      if(win_n > 0)
      {
         double first1 = CurrentWindowValue(0, 1);
         double first2 = CurrentWindowValue(0, 2);
         double last1 = CurrentWindowValue(win_n - 1, 1);
         double last2 = CurrentWindowValue(win_n - 1, 2);
         for(int i=0; i<win_n; i++)
         {
            double v1 = CurrentWindowValue(i, 1);
            double v2 = CurrentWindowValue(i, 2);
            double v4 = CurrentWindowValue(i, 4);
            mean1 += v1;
            mean2 += v2;
            mean4 += v4;
         }
         mean1 /= (double)win_n;
         mean2 /= (double)win_n;
         mean4 /= (double)win_n;
         for(int i=0; i<win_n; i++)
         {
            double d1 = CurrentWindowValue(i, 1) - mean1;
            double d2 = CurrentWindowValue(i, 2) - mean2;
            var1 += d1 * d1;
            var2 += d2 * d2;
         }
         var1 = MathSqrt(var1 / (double)win_n);
         var2 = MathSqrt(var2 / (double)win_n);
         if(win_n > 1)
         {
            trend1 = (last1 - first1) / (double)(win_n - 1);
            trend2 = (last2 - first2) / (double)(win_n - 1);
         }
      }
      b[8] = mean1;
      b[9] = trend1;
      b[10] = var1;
      b[11] = mean2;
      b[12] = trend2;
      b[13] = var2;
      b[14] = mean4;
      b[15] = (double)ContextSessionBucket();
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
         for(int j=0; j<FXAI_RETRDIFF_BASE; j++)
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
         for(int j=0; j<FXAI_RETRDIFF_BASE; j++) z += m_R[i][j] * b[j];
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

   void RetrieveTopK(const double &e[], int &idxs[], double &ds[]) const
   {
      ArrayResize(idxs, FXAI_RETRDIFF_TOPK);
      ArrayResize(ds, FXAI_RETRDIFF_TOPK);
      for(int k=0; k<FXAI_RETRDIFF_TOPK; k++)
      {
         idxs[k] = -1;
         ds[k] = DBL_MAX;
      }
      for(int i=0; i<m_count; i++)
      {
         int pos = (m_head - 1 - i + FXAI_RETRDIFF_MEM) % FXAI_RETRDIFF_MEM;
         double d = DistMem(e, pos);
         for(int k=0; k<FXAI_RETRDIFF_TOPK; k++)
         {
            if(d < ds[k])
            {
               for(int j=FXAI_RETRDIFF_TOPK - 1; j>k; j--)
               {
                  ds[j] = ds[j - 1];
                  idxs[j] = idxs[j - 1];
               }
               ds[k] = d;
               idxs[k] = pos;
               break;
            }
         }
      }
   }

   bool TryMergePrototype(const double &e[],
                          const int label,
                          const double move_points,
                          const datetime sample_time,
                          const double regime_vol,
                          const double regime_dir)
   {
      int idxs[];
      double ds[];
      RetrieveTopK(e, idxs, ds);
      int i1 = (ArraySize(idxs) > 0 ? idxs[0] : -1);
      double d1 = (ArraySize(ds) > 0 ? ds[0] : DBL_MAX);
      if(i1 < 0 || d1 > 0.08) return false;

      double total_mass = m_label_mass[i1][0] + m_label_mass[i1][1] + m_label_mass[i1][2];
      if(total_mass <= 1e-9) total_mass = 1.0;
      int dom = 2;
      if(m_label_mass[i1][1] > m_label_mass[i1][dom]) dom = 1;
      if(m_label_mass[i1][0] > m_label_mass[i1][dom]) dom = 0;
      if(dom != label && (m_label_mass[i1][dom] / total_mass) > 0.70)
         return false;

      double up_t = (label == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
      double ev_t = (label == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
      double w = MathMin(16.0, m_proto_weight[i1] + 1.0);
      double alpha = 1.0 / w;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         m_emb[i1][i] = (1.0 - alpha) * m_emb[i1][i] + alpha * e[i];
      m_future_move[i1] = (1.0 - alpha) * m_future_move[i1] + alpha * MathAbs(move_points);
      double diff = MathAbs(move_points) - m_future_move[i1];
      m_move_var[i1] = (1.0 - alpha) * m_move_var[i1] + alpha * diff * diff;
      m_future_up[i1] = (1.0 - alpha) * m_future_up[i1] + alpha * up_t;
      m_future_event[i1] = (1.0 - alpha) * m_future_event[i1] + alpha * ev_t;
      for(int c=0; c<3; c++)
      {
         double target = (c == label ? 1.0 : 0.0);
         m_label_mass[i1][c] = (1.0 - alpha) * m_label_mass[i1][c] + alpha * target;
      }
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
         m_move_var[pos]     = 0.25 * MathAbs(move_points) * MathAbs(move_points);
         m_future_up[pos]    = (y == (int)FXAI_LABEL_BUY ? 1.0 : 0.0);
         m_future_event[pos] = (y == (int)FXAI_LABEL_SKIP ? 0.0 : 1.0);
         for(int c=0; c<3; c++)
            m_label_mass[pos][c] = (c == y ? 1.0 : 0.0);
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
      ArrayInitialize(m_move_var, 0.0);
      ArrayInitialize(m_future_up, 0.0);
      ArrayInitialize(m_future_event, 0.0);
      ArrayInitialize(m_regime_vol, 0.0);
      ArrayInitialize(m_regime_dir, 0.0);
      ArrayInitialize(m_proto_weight, 1.0);
      ArrayInitialize(m_sample_time, 0);
      for(int i=0; i<FXAI_RETRDIFF_MEM; i++)
         for(int c=0; c<3; c++)
            m_label_mass[i][c] = (c == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         for(int j=0; j<FXAI_RETRDIFF_BASE; j++)
            m_R[i][j] = 0.0;
   }

   virtual int AIId(void) const { return (int)AI_RETRDIFF; }
   virtual string AIName(void) const { return "mem_retrdiff"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_RETRIEVAL, caps, 8, 64);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_init = false;
      m_steps = 0;
      m_head = 0;
      m_count = 0;
      ArrayInitialize(m_future_move, 0.0);
      ArrayInitialize(m_move_var, 0.0);
      ArrayInitialize(m_future_up, 0.0);
      ArrayInitialize(m_future_event, 0.0);
      ArrayInitialize(m_regime_vol, 0.0);
      ArrayInitialize(m_regime_dir, 0.0);
      ArrayInitialize(m_proto_weight, 1.0);
      ArrayInitialize(m_sample_time, 0);
      for(int i=0; i<FXAI_RETRDIFF_MEM; i++)
         for(int c=0; c<3; c++)
            m_label_mass[i][c] = (c == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_init) return;
      for(int i=0; i<FXAI_RETRDIFF_EMB; i++)
         for(int j=0; j<FXAI_RETRDIFF_BASE; j++)
         {
            uint h = (uint)((i + 1) * 73856093) ^ (uint)((j + 3) * 19349663) ^ (uint)2654435769;
            h ^= (h >> 13);
            h *= (uint)1274126177;
            h ^= (h >> 16);
            double u = (double)(h & (uint)2147483647) / 2147483647.0;
            m_R[i][j] = 2.0 * u - 1.0;
         }
      m_init = true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);
      if(m_count < 8)
      {
         out.class_probs[(int)FXAI_LABEL_SELL] = 0.10;
         out.class_probs[(int)FXAI_LABEL_BUY]  = 0.10;
         out.class_probs[(int)FXAI_LABEL_SKIP] = 0.80;
         out.move_mean_points = 0.0;
         out.move_q25_points = 0.0;
         out.move_q50_points = 0.0;
         out.move_q75_points = 0.0;
         out.confidence = 0.0;
         out.reliability = 0.0;
         out.has_quantiles = true;
         out.has_confidence = true;
         PopulatePathQualityHeads(out, x, FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), out.reliability, out.confidence);
         return true;
      }

      double e[]; Embed(x, e);
      int idxs[];
      double ds[];
      RetrieveTopK(e, idxs, ds);
      if(ArraySize(idxs) <= 0 || idxs[0] < 0)
         return false;

      datetime now = ResolveContextTime();
      double cur_vol = MathAbs(SX(x, 4));
      double cur_dir = SX(x, 0) - SX(x, 1);
      double cls_mass[3] = {0.0, 0.0, 0.0};
      double sw = 0.0, mv = 0.0, mv2 = 0.0;
      double nearest = ds[0];
      double support = 0.0;
      for(int k=0; k<ArraySize(idxs); k++)
      {
         int idx = idxs[k];
         if(idx < 0) continue;
         double d = ds[k];
         double w = 1.0 / MathMax(0.03, d);
         double regime_sim = 1.0 / (1.0 + MathAbs(m_regime_vol[idx] - cur_vol) + 0.5 * MathAbs(m_regime_dir[idx] - cur_dir));
         double recency = 1.0;
         if(m_sample_time[idx] > 0 && now > m_sample_time[idx])
         {
            double age_min = (double)(now - m_sample_time[idx]) / 60.0;
            recency = 1.0 / (1.0 + age_min / 720.0);
         }
         double proto = MathSqrt(MathMax(m_proto_weight[idx], 1.0));
         w *= regime_sim * recency * proto;
         support += proto;
         for(int c=0; c<3; c++)
            cls_mass[c] += w * m_label_mass[idx][c];
         mv += w * m_future_move[idx];
         mv2 += w * (m_move_var[idx] + m_future_move[idx] * m_future_move[idx]);
         sw += w;
      }
      if(sw <= 0.0) return false;
      for(int c=0; c<3; c++)
         out.class_probs[c] = cls_mass[c] / sw;
      NormalizeClassDistribution(out.class_probs);
      mv /= sw;
      mv2 /= sw;
      double var = MathMax(0.0, mv2 - mv * mv);
      double sigma = MathSqrt(var + 1e-6);
      out.move_mean_points = MathMax(0.0, mv);
      out.move_q25_points = MathMax(0.0, mv - 0.674 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, mv);
      out.move_q75_points = MathMax(out.move_q50_points, mv + 0.674 * sigma);
      double entropy = 0.0;
      for(int c=0; c<3; c++)
      {
         double p = FXAI_Clamp(out.class_probs[c], 1e-9, 1.0);
         entropy += -p * MathLog(p);
      }
      out.confidence = FXAI_Clamp(1.0 - entropy / MathLog(3.0), 0.0, 1.0);
      out.reliability = FXAI_Clamp((1.0 / (1.0 + nearest)) * MathMin(1.0, support / 8.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PopulatePathQualityHeads(out, x, FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), out.reliability, out.confidence);
      return true;
   }

   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      FXAIAIModelOutputV4 out;
      if(!PredictDistributionCore(x, hp, out))
         return false;
      for(int i=0; i<3; i++)
         class_probs[i] = out.class_probs[i];
      expected_move_points = out.move_mean_points;
      return true;
   }
};

#endif
