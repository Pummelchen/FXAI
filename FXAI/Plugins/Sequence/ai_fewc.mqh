#ifndef __FXAI_AI_FEWC_MQH__
#define __FXAI_AI_FEWC_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_FEWC_HIDDEN 8
#define FXAI_FEWC_CHAMBERS 3
#define FXAI_FEWC_CLASS_COUNT 3
#define FXAI_FEWC_REP 14

#define FXAI_FEWC_SELL 0
#define FXAI_FEWC_BUY  1
#define FXAI_FEWC_SKIP 2

class CFXAIAIFEWC : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_norm_ready;
   int    m_norm_steps;
   int    m_step;

   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   double m_w_in[FXAI_FEWC_HIDDEN][FXAI_AI_WEIGHTS];
   double m_b_in[FXAI_FEWC_HIDDEN];
   double m_decay[FXAI_FEWC_CHAMBERS];
   double m_stretch[FXAI_FEWC_CHAMBERS];
   double m_amp[FXAI_FEWC_CHAMBERS][FXAI_FEWC_HIDDEN];
   double m_echo[FXAI_FEWC_CHAMBERS][FXAI_FEWC_HIDDEN];

   double m_w_cls[FXAI_FEWC_CLASS_COUNT][FXAI_FEWC_REP];
   double m_b_cls[FXAI_FEWC_CLASS_COUNT];
   double m_w_move[FXAI_FEWC_REP];
   double m_b_move;
   double m_w_logv[FXAI_FEWC_REP];
   double m_b_logv;

   int ResolveClass(const int y,
                    const double move_points) const
   {
      if(y >= FXAI_FEWC_SELL && y <= FXAI_FEWC_SKIP)
         return y;
      if(move_points > 0.0) return FXAI_FEWC_BUY;
      if(move_points < 0.0) return FXAI_FEWC_SELL;
      return FXAI_FEWC_SKIP;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];
      double den = 0.0;
      for(int c=0; c<FXAI_FEWC_CLASS_COUNT; c++)
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
      for(int c=0; c<FXAI_FEWC_CLASS_COUNT; c++)
         probs[c] /= den;
   }

   double TanhSafe(const double z) const
   {
      double c = FXAI_ClipSym(z, 10.0);
      double e2 = MathExp(2.0 * c);
      return (e2 - 1.0) / (e2 + 1.0);
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
         m_x_var[i] = MathMax(1e-4, (1.0 - a) * m_x_var[i] + a * d * d);
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

   void ResetEcho(void)
   {
      for(int c=0; c<FXAI_FEWC_CHAMBERS; c++)
         for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
            m_echo[c][h] = 0.0;
   }

   void InitWeights(void)
   {
      double seed = 0.42 + 0.33 * FXAI_SymbolHash01(_Symbol);
      for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
      {
         m_b_in[h] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double a = 0.027 * (double)(h + 1) * (double)(i + 1);
            m_w_in[h][i] = 0.090 * MathSin(seed + a);
         }
      }

      for(int c=0; c<FXAI_FEWC_CHAMBERS; c++)
      {
         m_decay[c] = 0.58 + 0.11 * (double)c;
         m_stretch[c] = 0.82 + 0.22 * (double)c;
         for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
         {
            double a = 0.061 * (double)(c + 1) * (double)(h + 1);
            m_amp[c][h] = 0.20 * MathCos(seed + a);
         }
      }

      for(int c=0; c<FXAI_FEWC_CLASS_COUNT; c++)
      {
         m_b_cls[c] = (c == FXAI_FEWC_SKIP ? 0.10 : 0.0);
         for(int r=0; r<FXAI_FEWC_REP; r++)
         {
            double a = 0.055 * (double)(c + 1) * (double)(r + 1);
            m_w_cls[c][r] = 0.055 * MathSin(seed + a);
         }
      }

      for(int r=0; r<FXAI_FEWC_REP; r++)
      {
         double a = 0.049 * (double)(r + 1);
         m_w_move[r] = 0.042 * MathCos(seed + a);
         m_w_logv[r] = 0.026 * MathSin(seed + 0.8 * a);
      }
      m_b_move = 0.0;
      m_b_logv = -0.55;
      m_initialized = true;
   }

   void BuildWindowAwareInput(const double &x[],
                              double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
      if(CurrentWindowSize() <= 1)
         return;

      double cyc = CurrentWindowFeatureEMAMean(1) - CurrentWindowFeatureRecentMean(1, MathMax(CurrentWindowSize() / 5, 2));
      double vol = CurrentWindowFeatureStd(0);
      double vol_cluster = CurrentWindowFeatureRecentMean(6, MathMax(CurrentWindowSize() / 5, 3)) - CurrentWindowFeatureMean(6);
      double rng = CurrentWindowFeatureRange(0, MathMax(CurrentWindowSize() / 4, 4));
      double trend = CurrentWindowFeatureSlope(0);

      xa[1] = FXAI_ClipSym(0.52 * xa[1] + 0.18 * cyc + 0.14 * trend + 0.10 * rng, 8.0);
      xa[2] = FXAI_ClipSym(0.52 * xa[2] + 0.20 * cyc + 0.10 * vol_cluster + 0.10 * rng, 8.0);
      xa[6] = FXAI_ClipSym(0.56 * xa[6] + 0.20 * vol + 0.16 * MathAbs(vol_cluster) + 0.08 * rng, 8.0);
      xa[10] = FXAI_ClipSym(0.66 * xa[10] + 0.12 * trend + 0.12 * cyc + 0.10 * vol_cluster, 8.0);
      xa[11] = FXAI_ClipSym(0.64 * xa[11] + 0.10 * rng + 0.12 * vol_cluster + 0.10 * cyc, 8.0);
   }

   double EstimateFractalDimension(void) const
   {
      if(CurrentWindowSize() <= 8)
         return 1.20;
      double r4 = MathMax(CurrentWindowFeatureRange(0, 4), 1e-4);
      double r8 = MathMax(CurrentWindowFeatureRange(0, 8), 1e-4);
      double r16 = MathMax(CurrentWindowFeatureRange(0, 16), 1e-4);
      double d1 = MathLog(r8 / r4) / MathLog(2.0);
      double d2 = MathLog(r16 / r8) / MathLog(2.0);
      double fd = 1.0 + 0.5 * (MathAbs(d1) + MathAbs(d2));
      return FXAI_Clamp(fd, 0.80, 1.95);
   }

   void ForwardModel(const double &x[],
                     const bool adapt_norm,
                     const bool update_echo,
                     double &xa[],
                     double &rep[],
                     double &probs[],
                     double &move_mean_points,
                     double &q25_points,
                     double &q75_points,
                     double &confidence,
                     double &reliability,
                     double &fractal_dim,
                     double &echo_energy)
   {
      BuildWindowAwareInput(x, xa);
      if(adapt_norm)
         UpdateInputStats(xa);

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      fractal_dim = EstimateFractalDimension();
      double base[FXAI_FEWC_HIDDEN];
      for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
      {
         double z = m_b_in[h];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            z += m_w_in[h][i] * xn[i];
         base[h] = TanhSafe(z);
      }

      double chamber[FXAI_FEWC_CHAMBERS][FXAI_FEWC_HIDDEN];
      echo_energy = 0.0;
      for(int c=0; c<FXAI_FEWC_CHAMBERS; c++)
      {
         double decay = FXAI_Clamp(m_decay[c] + 0.05 * (fractal_dim - 1.20), 0.35, 0.97);
         double stretch = FXAI_Clamp(m_stretch[c] + 0.12 * (fractal_dim - 1.10), 0.60, 2.10);
         for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
         {
            double src = (c == 0 ? base[h] : chamber[c - 1][h]);
            double coupled = src + decay * m_echo[c][h] + (m_amp[c][h] * base[(h + c + 1) % FXAI_FEWC_HIDDEN]) / stretch;
            chamber[c][h] = TanhSafe(coupled);
            echo_energy += MathAbs(chamber[c][h]);
         }
      }
      echo_energy /= (double)(FXAI_FEWC_CHAMBERS * FXAI_FEWC_HIDDEN);

      for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
         rep[h] = FXAI_ClipSym(0.44 * chamber[0][h] + 0.34 * chamber[1][h] + 0.22 * chamber[2][h], 6.0);
      rep[8]  = FXAI_ClipSym(fractal_dim, 4.0);
      rep[9]  = FXAI_ClipSym(echo_energy, 4.0);
      rep[10] = FXAI_ClipSym(CurrentWindowFeatureStd(0), 6.0);
      rep[11] = FXAI_ClipSym(CurrentWindowFeatureSlope(0), 6.0);
      rep[12] = FXAI_ClipSym(CurrentWindowFeatureRecentDelta(1, MathMax(CurrentWindowSize() / 4, 2)), 6.0);
      rep[13] = FXAI_ClipSym(CurrentWindowFeatureRecentMean(6, MathMax(CurrentWindowSize() / 5, 3)) - CurrentWindowFeatureMean(6), 6.0);

      double logits[FXAI_FEWC_CLASS_COUNT];
      for(int c=0; c<FXAI_FEWC_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int r=0; r<FXAI_FEWC_REP; r++)
            z += m_w_cls[c][r] * rep[r];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      double active = FXAI_Clamp(1.0 - probs[FXAI_FEWC_SKIP], 0.0, 1.0);
      double move_raw = m_b_move;
      double logv_raw = m_b_logv;
      for(int r=0; r<FXAI_FEWC_REP; r++)
      {
         move_raw += m_w_move[r] * rep[r];
         logv_raw += m_w_logv[r] * rep[r];
      }
      double sigma = FXAI_Clamp(MathExp(0.35 * FXAI_ClipSym(logv_raw, 6.0)), 0.05, 20.0);
      move_mean_points = MathMax(0.0,
                                 (0.58 * MathAbs(move_raw) +
                                  0.18 * echo_energy +
                                  0.14 * MathAbs(rep[13]) +
                                  0.10 * MathAbs(fractal_dim - 1.0)) * active);
      if(move_mean_points <= 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         move_mean_points = 0.50 * m_move_ema_abs * active;

      q25_points = MathMax(0.0, move_mean_points - 0.48 * sigma);
      q75_points = MathMax(q25_points, move_mean_points + 0.48 * sigma);
      confidence = FXAI_Clamp(0.58 * MathMax(probs[FXAI_FEWC_BUY], probs[FXAI_FEWC_SELL]) +
                              0.18 * echo_energy +
                              0.12 * FXAI_Clamp(2.0 - MathAbs(fractal_dim - 1.35), 0.0, 1.0) +
                              0.12 * active,
                              0.0, 1.0);
      reliability = FXAI_Clamp(0.36 +
                               0.26 * FXAI_Clamp(2.0 - MathAbs(fractal_dim - 1.35), 0.0, 1.0) +
                               0.22 * active +
                               0.16 * (m_move_ready ? 1.0 : 0.0),
                               0.0, 1.0);

      if(update_echo)
      {
         for(int c=0; c<FXAI_FEWC_CHAMBERS; c++)
            for(int h=0; h<FXAI_FEWC_HIDDEN; h++)
               m_echo[c][h] = 0.78 * m_echo[c][h] + 0.22 * chamber[c][h];
      }
   }

public:
   CFXAIAIFEWC(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_FEWC; }
   virtual string AIName(void) const { return "ai_fewc"; }

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);
      FillManifest(out, (int)FXAI_FAMILY_CONVOLUTIONAL, caps, 16, 144);
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      ResetNorm();
      ResetEcho();
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
      double rep[FXAI_FEWC_REP];
      double q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, fractal = 0.0, echo = 0.0;
      ForwardModel(x, false, false, xa, rep, class_probs, expected_move_points, q25, q75, conf, rel, fractal, echo);
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
      double rep[FXAI_FEWC_REP];
      double fractal = 0.0, echo = 0.0;
      ForwardModel(x,
                   false,
                   false,
                   xa,
                   rep,
                   out.class_probs,
                   out.move_mean_points,
                   out.move_q25_points,
                   out.move_q75_points,
                   out.confidence,
                   out.reliability,
                   fractal,
                   echo);
      NormalizeClassDistribution(out.class_probs);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
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
      double lr = FXAI_Clamp(h.lr, 0.0004, 0.0400);
      double reg = FXAI_Clamp(h.l2, 0.0, 0.0250);

      double xa[FXAI_AI_WEIGHTS];
      double rep[FXAI_FEWC_REP];
      double probs[FXAI_FEWC_CLASS_COUNT];
      double move_pred = 0.0, q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, fractal = 0.0, echo = 0.0;
      ForwardModel(x, true, false, xa, rep, probs, move_pred, q25, q75, conf, rel, fractal, echo);

      int cls = ResolveClass(y, move_points);
      double sample_w = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 6.00);
      double lr_eff = lr * sample_w;

      double target[FXAI_FEWC_CLASS_COUNT];
      target[0] = (cls == FXAI_FEWC_SELL ? 1.0 : 0.0);
      target[1] = (cls == FXAI_FEWC_BUY ? 1.0 : 0.0);
      target[2] = (cls == FXAI_FEWC_SKIP ? 1.0 : 0.0);
      for(int c=0; c<FXAI_FEWC_CLASS_COUNT; c++)
      {
         double err = probs[c] - target[c];
         for(int r=0; r<FXAI_FEWC_REP; r++)
            m_w_cls[c][r] -= lr_eff * (err * rep[r] + reg * m_w_cls[c][r]);
         m_b_cls[c] -= lr_eff * err;
      }

      double target_move = (cls == FXAI_FEWC_SKIP ? 0.0 : MathMax(MathAbs(move_points), ResolveMinMovePoints()));
      double err_move = move_pred - target_move;
      for(int r=0; r<FXAI_FEWC_REP; r++)
      {
         m_w_move[r] -= lr_eff * (0.26 * err_move * rep[r] + reg * m_w_move[r]);
         double logv_err = FXAI_ClipSym(m_w_logv[r] * rep[r], 4.0) - MathLog(MathMax(MathAbs(err_move), 0.08));
         m_w_logv[r] -= lr_eff * (0.06 * logv_err * rep[r] + 0.5 * reg * m_w_logv[r]);
      }
      m_b_move -= lr_eff * 0.26 * err_move;
      m_b_logv -= lr_eff * 0.05 * (MathExp(FXAI_ClipSym(m_b_logv, 4.0)) - MathMax(MathAbs(err_move), 0.08));

      double dir_target = (cls == FXAI_FEWC_BUY ? 1.0 : (cls == FXAI_FEWC_SELL ? -1.0 : 0.0));
      double dir_pred = probs[FXAI_FEWC_BUY] - probs[FXAI_FEWC_SELL];
      double core_err = dir_target - dir_pred;
      double fractal_pull = (cls == FXAI_FEWC_SKIP ? 1.10 : 1.35) - fractal;
      double echo_pull = (cls == FXAI_FEWC_SKIP ? 0.28 : 0.52) - echo;
      for(int hidx=0; hidx<FXAI_FEWC_HIDDEN; hidx++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double xi = xa[i];
            double grad = 0.05 * core_err * xi + 0.02 * fractal_pull * xi;
            m_w_in[hidx][i] = (1.0 - lr_eff * reg) * m_w_in[hidx][i] + lr_eff * grad;
         }
      }

      for(int c=0; c<FXAI_FEWC_CHAMBERS; c++)
      {
         m_decay[c] = FXAI_Clamp(m_decay[c] + lr_eff * (0.08 * echo_pull - 0.04 * fractal_pull), 0.30, 0.98);
         m_stretch[c] = FXAI_Clamp(m_stretch[c] + lr_eff * (0.06 * fractal_pull), 0.50, 2.20);
         for(int hidx=0; hidx<FXAI_FEWC_HIDDEN; hidx++)
            m_amp[c][hidx] = (1.0 - lr_eff * reg) * m_amp[c][hidx] + lr_eff * (0.04 * core_err + 0.03 * echo_pull);
      }

      double rep_live[FXAI_FEWC_REP];
      double probs_live[FXAI_FEWC_CLASS_COUNT];
      double move_live = 0.0, q25_live = 0.0, q75_live = 0.0, conf_live = 0.0, rel_live = 0.0, fractal_live = 0.0, echo_live = 0.0;
      ForwardModel(x, false, true, xa, rep_live, probs_live, move_live, q25_live, q75_live, conf_live, rel_live, fractal_live, echo_live);

      UpdateNativeQualityHeads(xa, sample_w, lr, reg);
      m_step++;
   }
};

#endif // __FXAI_AI_FEWC_MQH__
