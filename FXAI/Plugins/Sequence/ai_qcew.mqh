#ifndef __FXAI_AI_QCEW_MQH__
#define __FXAI_AI_QCEW_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_QCEW_DIM 6
#define FXAI_QCEW_CLASS_COUNT 3
#define FXAI_QCEW_REP 18

#define FXAI_QCEW_SELL 0
#define FXAI_QCEW_BUY  1
#define FXAI_QCEW_SKIP 2

class CFXAIAIQCEW : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_norm_ready;
   int    m_norm_steps;
   int    m_step;

   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   double m_w_re[FXAI_QCEW_DIM][FXAI_AI_WEIGHTS];
   double m_w_im[FXAI_QCEW_DIM][FXAI_AI_WEIGHTS];
   double m_b_re[FXAI_QCEW_DIM];
   double m_b_im[FXAI_QCEW_DIM];
   double m_weave_gate[FXAI_QCEW_DIM];
   double m_phase_bias[FXAI_QCEW_DIM];

   double m_mem_re[FXAI_QCEW_DIM];
   double m_mem_im[FXAI_QCEW_DIM];

   double m_w_cls[FXAI_QCEW_CLASS_COUNT][FXAI_QCEW_REP];
   double m_b_cls[FXAI_QCEW_CLASS_COUNT];
   double m_w_move[FXAI_QCEW_REP];
   double m_b_move;
   double m_w_logv[FXAI_QCEW_REP];
   double m_b_logv;

   int ResolveClass(const int y,
                    const double move_points) const
   {
      if(y >= FXAI_QCEW_SELL && y <= FXAI_QCEW_SKIP)
         return y;
      if(move_points > 0.0) return FXAI_QCEW_BUY;
      if(move_points < 0.0) return FXAI_QCEW_SELL;
      return FXAI_QCEW_SKIP;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double den = 0.0;
      for(int c=0; c<FXAI_QCEW_CLASS_COUNT; c++)
      {
         probs[c] = MathExp(FXAI_ClipSym(logits[c] - m, 24.0));
         den += probs[c];
      }
      if(den <= 0.0)
      {
         probs[0] = 0.3333333;
         probs[1] = 0.3333333;
         probs[2] = 0.3333333;
         return;
      }
      for(int c=0; c<FXAI_QCEW_CLASS_COUNT; c++)
         probs[c] /= den;
   }

   void ResetNorm(void)
   {
      m_norm_ready = false;
      m_norm_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      m_norm_steps++;
      double a = 1.0 / (double)MathMin(MathMax(m_norm_steps, 1), 256);
      if(m_norm_steps > 32) a = 0.015;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (MathIsValidNumber(x[i]) ? x[i] : 0.0);
         double d = v - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = d * d;
         m_x_var[i] = MathMax(1e-4, (1.0 - a) * m_x_var[i] + a * dv);
      }
      if(m_norm_steps >= 8)
         m_norm_ready = true;
   }

   void NormalizeInput(const double &x[],
                       double &xn[]) const
   {
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (MathIsValidNumber(x[i]) ? x[i] : 0.0);
         if(i == 0)
         {
            xn[i] = 1.0;
            continue;
         }
         if(!m_norm_ready)
         {
            xn[i] = FXAI_ClipSym(v, 8.0);
            continue;
         }
         double s = MathSqrt(MathMax(m_x_var[i], 1e-4));
         xn[i] = FXAI_ClipSym((v - m_x_mean[i]) / s, 8.0);
      }
   }

   void ResetMemory(void)
   {
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         m_mem_re[d] = 0.0;
         m_mem_im[d] = 0.0;
      }
   }

   void InitWeights(void)
   {
      double seed = 0.55 + 0.35 * FXAI_SymbolHash01(_Symbol);
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         m_b_re[d] = 0.0;
         m_b_im[d] = 0.0;
         m_weave_gate[d] = 0.20 * MathSin(seed * (double)(d + 1));
         m_phase_bias[d] = 0.18 * MathCos(seed * (double)(d + 2));
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double a = 0.031 * (double)(i + 1) * (double)(d + 1);
            m_w_re[d][i] = 0.080 * MathSin(seed + a);
            m_w_im[d][i] = 0.080 * MathCos(1.7 * seed + 1.3 * a);
         }
      }

      for(int c=0; c<FXAI_QCEW_CLASS_COUNT; c++)
      {
         m_b_cls[c] = (c == FXAI_QCEW_SKIP ? 0.15 : 0.0);
         for(int r=0; r<FXAI_QCEW_REP; r++)
         {
            double a = 0.057 * (double)(c + 1) * (double)(r + 1);
            m_w_cls[c][r] = 0.060 * MathSin(seed + a);
         }
      }

      for(int r=0; r<FXAI_QCEW_REP; r++)
      {
         double a = 0.041 * (double)(r + 1);
         m_w_move[r] = 0.045 * MathCos(seed + a);
         m_w_logv[r] = 0.030 * MathSin(seed + 0.7 * a);
      }
      m_b_move = 0.0;
      m_b_logv = -0.65;
      m_initialized = true;
   }

   void BuildWindowAwareInput(const double &x[],
                              double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
      if(CurrentWindowSize() <= 1)
         return;

      double trend = CurrentWindowFeatureSlope(0);
      double price_mean = CurrentWindowFeatureMean(0);
      double ret_mean = CurrentWindowFeatureMean(1);
      double ret_delta = CurrentWindowFeatureRecentDelta(1, MathMax(CurrentWindowSize() / 4, 2));
      double vol = CurrentWindowFeatureStd(0);
      double regime = CurrentWindowFeatureRange(6, MathMax(CurrentWindowSize() / 3, 4));
      double spread = CurrentWindowFeatureEMAMean(70);

      xa[1] = FXAI_ClipSym(0.56 * xa[1] + 0.16 * price_mean + 0.16 * trend + 0.12 * ret_delta, 8.0);
      xa[2] = FXAI_ClipSym(0.56 * xa[2] + 0.18 * ret_mean + 0.14 * ret_delta + 0.12 * regime, 8.0);
      xa[6] = FXAI_ClipSym(0.58 * xa[6] + 0.24 * vol + 0.10 * regime + 0.08 * spread, 8.0);
      xa[10] = FXAI_ClipSym(0.66 * xa[10] + 0.12 * trend + 0.12 * vol + 0.10 * regime, 8.0);
      xa[11] = FXAI_ClipSym(0.64 * xa[11] + 0.14 * ret_mean - 0.10 * trend + 0.12 * spread, 8.0);
   }

   void UpdateMemoryFromRep(const double &rep[])
   {
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         double vr = rep[2 * d];
         double vi = rep[2 * d + 1];
         m_mem_re[d] = 0.82 * m_mem_re[d] + 0.18 * vr;
         m_mem_im[d] = 0.82 * m_mem_im[d] + 0.18 * vi;
      }
   }

   void ForwardModel(const double &x[],
                     const bool adapt_norm,
                     const bool update_memory,
                     double &xa[],
                     double &rep[],
                     double &probs[],
                     double &move_mean_points,
                     double &q25_points,
                     double &q75_points,
                     double &confidence,
                     double &reliability,
                     double &entangle_score,
                     double &fidelity_score)
   {
      BuildWindowAwareInput(x, xa);
      if(adapt_norm)
         UpdateInputStats(xa);

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      double phase_drive = 0.28 * xn[1] + 0.17 * xn[2] + 0.11 * CurrentWindowFeatureSlope(0) + 0.09 * CurrentWindowFeatureStd(0);
      double re[FXAI_QCEW_DIM];
      double im[FXAI_QCEW_DIM];
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         double zr = m_b_re[d];
         double zi = m_b_im[d];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            zr += m_w_re[d][i] * xn[i];
            zi += m_w_im[d][i] * xn[i];
         }
         zr += 0.06 * phase_drive * MathSin(m_phase_bias[d]);
         zi += 0.06 * phase_drive * MathCos(m_phase_bias[d]);
         re[d] = FXAI_ClipSym(zr, 10.0);
         im[d] = FXAI_ClipSym(zi, 10.0);
      }

      entangle_score = 0.0;
      double fidelity_gap = 0.0;
      double amp_mean = 0.0;
      double phase_mean = 0.0;
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         int nxt = (d + 1) % FXAI_QCEW_DIM;
         int nxt2 = (d + 2) % FXAI_QCEW_DIM;
         double gate = FXAI_Sigmoid(m_weave_gate[d] + 0.35 * re[nxt] - 0.30 * im[nxt2]);
         double wr = re[d] + gate * (0.42 * re[nxt] - 0.24 * im[nxt2]) + 0.16 * m_mem_re[d] - 0.08 * m_mem_im[nxt];
         double wi = im[d] + gate * (0.42 * im[nxt] + 0.24 * re[nxt2]) + 0.16 * m_mem_im[d] + 0.08 * m_mem_re[nxt2];
         rep[2 * d] = FXAI_ClipSym(wr, 8.0);
         rep[2 * d + 1] = FXAI_ClipSym(wi, 8.0);

         double amp = MathSqrt(wr * wr + wi * wi + 1e-6);
         double mem_amp = MathSqrt(m_mem_re[d] * m_mem_re[d] + m_mem_im[d] * m_mem_im[d] + 1e-6);
         entangle_score += MathAbs((wr * wi) / (1.0 + amp));
         fidelity_gap += MathAbs(amp - mem_amp);
         amp_mean += amp;
         phase_mean += MathAbs(wr - wi);
      }

      entangle_score /= (double)FXAI_QCEW_DIM;
      amp_mean /= (double)FXAI_QCEW_DIM;
      phase_mean /= (double)FXAI_QCEW_DIM;
      fidelity_score = 1.0 / (1.0 + fidelity_gap / (double)FXAI_QCEW_DIM);

      rep[12] = FXAI_ClipSym(entangle_score, 4.0);
      rep[13] = FXAI_ClipSym(fidelity_score, 4.0);
      rep[14] = FXAI_ClipSym(CurrentWindowFeatureStd(0), 6.0);
      rep[15] = FXAI_ClipSym(CurrentWindowFeatureSlope(0), 6.0);
      rep[16] = FXAI_ClipSym(amp_mean, 6.0);
      rep[17] = FXAI_ClipSym(phase_mean, 6.0);

      double logits[FXAI_QCEW_CLASS_COUNT];
      for(int c=0; c<FXAI_QCEW_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int r=0; r<FXAI_QCEW_REP; r++)
            z += m_w_cls[c][r] * rep[r];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      double active = FXAI_Clamp(1.0 - probs[FXAI_QCEW_SKIP], 0.0, 1.0);
      double move_raw = m_b_move;
      double logv_raw = m_b_logv;
      for(int r=0; r<FXAI_QCEW_REP; r++)
      {
         move_raw += m_w_move[r] * rep[r];
         logv_raw += m_w_logv[r] * rep[r];
      }
      double sigma = FXAI_Clamp(MathExp(0.35 * FXAI_ClipSym(logv_raw, 6.0)), 0.05, 24.0);
      move_mean_points = MathMax(0.0,
                                 (0.62 * MathAbs(move_raw) +
                                  0.20 * amp_mean +
                                  0.12 * entangle_score +
                                  0.08 * (1.0 - fidelity_score)) * active);
      if(move_mean_points <= 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         move_mean_points = 0.55 * m_move_ema_abs * active;

      q25_points = MathMax(0.0, move_mean_points - 0.50 * sigma);
      q75_points = MathMax(q25_points, move_mean_points + 0.50 * sigma);
      confidence = FXAI_Clamp(0.55 * MathMax(probs[FXAI_QCEW_BUY], probs[FXAI_QCEW_SELL]) +
                              0.18 * entangle_score +
                              0.15 * fidelity_score +
                              0.12 * active,
                              0.0, 1.0);
      reliability = FXAI_Clamp(0.38 +
                               0.28 * fidelity_score +
                               0.18 * active +
                               0.16 * (m_move_ready ? 1.0 : 0.0),
                               0.0, 1.0);

      if(update_memory)
         UpdateMemoryFromRep(rep);
   }

public:
   CFXAIAIQCEW(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_QCEW; }
   virtual string AIName(void) const { return "ai_qcew"; }

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);
      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 160);
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      ResetNorm();
      ResetMemory();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights();
   }

   virtual bool PredictModelCore(const double &x[],
                                 const FXAIAIHyperParams &hp,
                                 double &class_probs[],
                                 double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      double rep[FXAI_QCEW_REP];
      double q25 = 0.0, q75 = 0.0;
      double conf = 0.0, rel = 0.0;
      double entangle = 0.0, fidelity = 0.0;
      ForwardModel(x, false, false, xa, rep, class_probs, expected_move_points, q25, q75, conf, rel, entangle, fidelity);
      NormalizeClassDistribution(class_probs);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      double rep[FXAI_QCEW_REP];
      double q25 = 0.0, q75 = 0.0;
      double entangle = 0.0, fidelity = 0.0;
      ForwardModel(x,
                   false,
                   false,
                   xa,
                   rep,
                   out.class_probs,
                   out.move_mean_points,
                   q25,
                   q75,
                   out.confidence,
                   out.reliability,
                   entangle,
                   fidelity);
      NormalizeClassDistribution(out.class_probs);
      out.move_q25_points = q25;
      out.move_q50_points = MathMax(q25, out.move_mean_points);
      out.move_q75_points = q75;
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double lr = FXAI_Clamp(h.lr, 0.0004, 0.0300);
      double reg = FXAI_Clamp(h.l2, 0.0, 0.0300);

      double xa[FXAI_AI_WEIGHTS];
      double rep[FXAI_QCEW_REP];
      double probs[FXAI_QCEW_CLASS_COUNT];
      double move_pred = 0.0, q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, entangle = 0.0, fidelity = 0.0;
      ForwardModel(x, true, false, xa, rep, probs, move_pred, q25, q75, conf, rel, entangle, fidelity);

      int cls = ResolveClass(y, move_points);
      double sample_w = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 6.00);
      double lr_eff = lr * sample_w;

      double target[FXAI_QCEW_CLASS_COUNT];
      target[0] = (cls == FXAI_QCEW_SELL ? 1.0 : 0.0);
      target[1] = (cls == FXAI_QCEW_BUY ? 1.0 : 0.0);
      target[2] = (cls == FXAI_QCEW_SKIP ? 1.0 : 0.0);
      for(int c=0; c<FXAI_QCEW_CLASS_COUNT; c++)
      {
         double err = probs[c] - target[c];
         for(int r=0; r<FXAI_QCEW_REP; r++)
            m_w_cls[c][r] -= lr_eff * (err * rep[r] + reg * m_w_cls[c][r]);
         m_b_cls[c] -= lr_eff * err;
      }

      double target_move = (cls == FXAI_QCEW_SKIP ? 0.0 : MathMax(MathAbs(move_points), ResolveMinMovePoints()));
      double err_move = move_pred - target_move;
      for(int r=0; r<FXAI_QCEW_REP; r++)
      {
         m_w_move[r] -= lr_eff * (0.28 * err_move * rep[r] + reg * m_w_move[r]);
         double sigma_target = MathMax(MathAbs(err_move), ResolveMinMovePoints() + 0.05);
         double logv_err = FXAI_ClipSym(m_w_logv[r] * rep[r], 4.0) - MathLog(MathMax(sigma_target, 0.05));
         m_w_logv[r] -= lr_eff * (0.06 * logv_err * rep[r] + 0.5 * reg * m_w_logv[r]);
      }
      m_b_move -= lr_eff * 0.28 * err_move;
      m_b_logv -= lr_eff * 0.06 * (MathExp(FXAI_ClipSym(m_b_logv, 4.0)) - MathMax(MathAbs(err_move), 0.10));

      double dir_target = (cls == FXAI_QCEW_BUY ? 1.0 : (cls == FXAI_QCEW_SELL ? -1.0 : 0.0));
      double dir_pred = probs[FXAI_QCEW_BUY] - probs[FXAI_QCEW_SELL];
      double core_err = dir_target - dir_pred;
      double fidelity_pull = (cls == FXAI_QCEW_SKIP ? 0.72 : 0.92) - fidelity;
      double entangle_pull = (cls == FXAI_QCEW_SKIP ? 0.30 : 0.62) - entangle;
      for(int d=0; d<FXAI_QCEW_DIM; d++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double xi = xa[i];
            double grad_re = 0.055 * core_err * xi + 0.015 * fidelity_pull * xi;
            double grad_im = 0.040 * core_err * xi + 0.018 * entangle_pull * xi * (double)((d % 2 == 0) ? 1 : -1);
            m_w_re[d][i] = (1.0 - lr_eff * reg) * m_w_re[d][i] + lr_eff * grad_re;
            m_w_im[d][i] = (1.0 - lr_eff * reg) * m_w_im[d][i] + lr_eff * grad_im;
         }
         m_weave_gate[d] = FXAI_ClipSym(m_weave_gate[d] + lr_eff * (0.20 * entangle_pull - 0.12 * fidelity_pull), 6.0);
         m_phase_bias[d] = FXAI_ClipSym(m_phase_bias[d] + lr_eff * 0.08 * core_err * (double)(d + 1) / (double)FXAI_QCEW_DIM, 3.14159);
      }

      double rep_live[FXAI_QCEW_REP];
      double probs_live[FXAI_QCEW_CLASS_COUNT];
      double move_live = 0.0, q25_live = 0.0, q75_live = 0.0, conf_live = 0.0, rel_live = 0.0, ent_live = 0.0, fid_live = 0.0;
      ForwardModel(x, false, true, xa, rep_live, probs_live, move_live, q25_live, q75_live, conf_live, rel_live, ent_live, fid_live);

      UpdateNativeQualityHeads(xa, sample_w, lr, reg);
      m_step++;
   }
};

#endif // __FXAI_AI_QCEW_MQH__
