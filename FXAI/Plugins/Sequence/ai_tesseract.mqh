#ifndef __FXAI_AI_TESSERACT_MQH__
#define __FXAI_AI_TESSERACT_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_TESS_AXES 4
#define FXAI_TESS_SHADOWS 4
#define FXAI_TESS_CLASS_COUNT 3
#define FXAI_TESS_REP 16

#define FXAI_TESS_SELL 0
#define FXAI_TESS_BUY  1
#define FXAI_TESS_SKIP 2

class CFXAIAITesseract : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_norm_ready;
   int    m_norm_steps;
   int    m_step;

   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   double m_w_axis[FXAI_TESS_AXES][FXAI_AI_WEIGHTS];
   double m_b_axis[FXAI_TESS_AXES];
   double m_theta[6];
   double m_proj_bias[FXAI_TESS_SHADOWS];
   double m_proj_mix[FXAI_TESS_SHADOWS];
   double m_shadow_u[FXAI_TESS_SHADOWS];
   double m_shadow_v[FXAI_TESS_SHADOWS];
   double m_latent_mem[FXAI_TESS_AXES];

   double m_w_cls[FXAI_TESS_CLASS_COUNT][FXAI_TESS_REP];
   double m_b_cls[FXAI_TESS_CLASS_COUNT];
   double m_w_move[FXAI_TESS_REP];
   double m_b_move;
   double m_w_logv[FXAI_TESS_REP];
   double m_b_logv;

   int ResolveClass(const int y,
                    const double move_points) const
   {
      if(y >= FXAI_TESS_SELL && y <= FXAI_TESS_SKIP)
         return y;
      if(move_points > 0.0) return FXAI_TESS_BUY;
      if(move_points < 0.0) return FXAI_TESS_SELL;
      return FXAI_TESS_SKIP;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];
      double den = 0.0;
      for(int c=0; c<FXAI_TESS_CLASS_COUNT; c++)
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
      for(int c=0; c<FXAI_TESS_CLASS_COUNT; c++)
         probs[c] /= den;
   }

   void SoftmaxN(const double &score[],
                 const int n,
                 double &weight[]) const
   {
      double m = score[0];
      for(int i=1; i<n; i++)
         if(score[i] > m) m = score[i];
      double den = 0.0;
      for(int i=0; i<n; i++)
      {
         weight[i] = MathExp(FXAI_ClipSym(score[i] - m, 24.0));
         den += weight[i];
      }
      if(den <= 0.0)
      {
         double inv = 1.0 / (double)MathMax(n, 1);
         for(int i=0; i<n; i++) weight[i] = inv;
         return;
      }
      for(int i=0; i<n; i++)
         weight[i] /= den;
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

   void ResetShadowState(void)
   {
      for(int p=0; p<FXAI_TESS_SHADOWS; p++)
      {
         m_shadow_u[p] = 0.0;
         m_shadow_v[p] = 0.0;
      }
      for(int a=0; a<FXAI_TESS_AXES; a++)
         m_latent_mem[a] = 0.0;
   }

   void InitWeights(void)
   {
      double seed = 0.68 + 0.23 * FXAI_SymbolHash01(_Symbol);
      for(int a=0; a<FXAI_TESS_AXES; a++)
      {
         m_b_axis[a] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double k = 0.023 * (double)(a + 1) * (double)(i + 1);
            m_w_axis[a][i] = 0.070 * MathSin(seed + k);
         }
      }
      for(int i=0; i<6; i++)
         m_theta[i] = 0.12 * MathCos(seed * (double)(i + 1));
      for(int p=0; p<FXAI_TESS_SHADOWS; p++)
      {
         m_proj_bias[p] = 0.06 * MathSin(seed * (double)(p + 1));
         m_proj_mix[p] = 0.18 * MathCos(seed * (double)(p + 2));
      }
      for(int c=0; c<FXAI_TESS_CLASS_COUNT; c++)
      {
         m_b_cls[c] = (c == FXAI_TESS_SKIP ? 0.14 : 0.0);
         for(int r=0; r<FXAI_TESS_REP; r++)
         {
            double a = 0.053 * (double)(c + 1) * (double)(r + 1);
            m_w_cls[c][r] = 0.052 * MathSin(seed + a);
         }
      }
      for(int r=0; r<FXAI_TESS_REP; r++)
      {
         double a = 0.046 * (double)(r + 1);
         m_w_move[r] = 0.040 * MathCos(seed + a);
         m_w_logv[r] = 0.024 * MathSin(seed + 0.8 * a);
      }
      m_b_move = 0.0;
      m_b_logv = -0.58;
      m_initialized = true;
   }

   void BuildWindowAwareInput(const double &x[],
                              double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
      if(CurrentWindowSize() <= 1)
         return;

      double trend = CurrentWindowFeatureSlope(0);
      double vol = CurrentWindowFeatureStd(0);
      double time_bias = CurrentWindowFeatureMean(74);
      double cost = CurrentWindowFeatureRecentMean(70, MathMax(CurrentWindowSize() / 5, 2));
      double latent_vol = CurrentWindowFeatureEMAMean(6);

      xa[1] = FXAI_ClipSym(0.56 * xa[1] + 0.16 * trend + 0.10 * latent_vol + 0.08 * time_bias, 8.0);
      xa[2] = FXAI_ClipSym(0.56 * xa[2] + 0.18 * vol - 0.10 * cost + 0.08 * time_bias, 8.0);
      xa[6] = FXAI_ClipSym(0.58 * xa[6] + 0.20 * vol + 0.12 * latent_vol + 0.08 * cost, 8.0);
      xa[10] = FXAI_ClipSym(0.66 * xa[10] + 0.12 * trend + 0.10 * time_bias + 0.08 * vol, 8.0);
      xa[11] = FXAI_ClipSym(0.64 * xa[11] + 0.10 * cost + 0.10 * latent_vol + 0.08 * trend, 8.0);
   }

   void RotatePair(double &a,
                   double &b,
                   const double theta) const
   {
      double c = MathCos(theta);
      double s = MathSin(theta);
      double na = c * a - s * b;
      double nb = s * a + c * b;
      a = na;
      b = nb;
   }

   void ForwardModel(const double &x[],
                     const bool adapt_norm,
                     const bool update_state,
                     double &xa[],
                     double &rep[],
                     double &probs[],
                     double &move_mean_points,
                     double &q25_points,
                     double &q75_points,
                     double &confidence,
                     double &reliability,
                     double &projection_diversity,
                     double &shadow_drift)
   {
      BuildWindowAwareInput(x, xa);
      if(adapt_norm)
         UpdateInputStats(xa);

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      double latent[FXAI_TESS_AXES];
      for(int a=0; a<FXAI_TESS_AXES; a++)
      {
         double z = m_b_axis[a];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            z += m_w_axis[a][i] * xn[i];
         z += 0.10 * m_latent_mem[a];
         latent[a] = FXAI_ClipSym(z, 8.0);
      }

      double rot[FXAI_TESS_AXES];
      for(int a=0; a<FXAI_TESS_AXES; a++)
         rot[a] = latent[a];
      RotatePair(rot[0], rot[1], m_theta[0]);
      RotatePair(rot[0], rot[2], m_theta[1]);
      RotatePair(rot[0], rot[3], m_theta[2]);
      RotatePair(rot[1], rot[2], m_theta[3]);
      RotatePair(rot[1], rot[3], m_theta[4]);
      RotatePair(rot[2], rot[3], m_theta[5]);

      double u[FXAI_TESS_SHADOWS];
      double v[FXAI_TESS_SHADOWS];
      u[0] = rot[0]; v[0] = rot[1];
      u[1] = rot[0]; v[1] = rot[2];
      u[2] = rot[1]; v[2] = rot[3];
      u[3] = rot[2]; v[3] = rot[3];

      double score[FXAI_TESS_SHADOWS];
      double alpha[FXAI_TESS_SHADOWS];
      projection_diversity = 0.0;
      shadow_drift = 0.0;
      for(int p=0; p<FXAI_TESS_SHADOWS; p++)
      {
         double radius = MathSqrt(u[p] * u[p] + v[p] * v[p] + 1e-6);
         score[p] = m_proj_bias[p] + radius + 0.18 * m_proj_mix[p] * (u[p] - v[p]);
         projection_diversity += MathAbs(u[p] - v[p]);
         shadow_drift += MathAbs(u[p] - m_shadow_u[p]) + MathAbs(v[p] - m_shadow_v[p]);
      }
      projection_diversity /= (double)FXAI_TESS_SHADOWS;
      shadow_drift /= (double)(2 * FXAI_TESS_SHADOWS);
      SoftmaxN(score, FXAI_TESS_SHADOWS, alpha);

      double w_u = 0.0;
      double w_v = 0.0;
      double w_r = 0.0;
      double proj_radius[FXAI_TESS_SHADOWS];
      for(int p=0; p<FXAI_TESS_SHADOWS; p++)
      {
         proj_radius[p] = MathSqrt(u[p] * u[p] + v[p] * v[p] + 1e-6);
         w_u += alpha[p] * u[p];
         w_v += alpha[p] * v[p];
         w_r += alpha[p] * proj_radius[p];
      }

      rep[0] = FXAI_ClipSym(w_u, 6.0);
      rep[1] = FXAI_ClipSym(w_v, 6.0);
      rep[2] = FXAI_ClipSym(w_r, 6.0);
      rep[3] = FXAI_ClipSym(rot[0], 6.0);
      rep[4] = FXAI_ClipSym(rot[1], 6.0);
      rep[5] = FXAI_ClipSym(rot[2], 6.0);
      rep[6] = FXAI_ClipSym(rot[3], 6.0);
      rep[7] = FXAI_ClipSym(proj_radius[0], 6.0);
      rep[8] = FXAI_ClipSym(proj_radius[1], 6.0);
      rep[9] = FXAI_ClipSym(proj_radius[2], 6.0);
      rep[10] = FXAI_ClipSym(proj_radius[3], 6.0);
      rep[11] = FXAI_ClipSym(projection_diversity, 6.0);
      rep[12] = FXAI_ClipSym(shadow_drift, 6.0);
      rep[13] = FXAI_ClipSym(CurrentWindowFeatureStd(0), 6.0);
      rep[14] = FXAI_ClipSym(CurrentWindowFeatureMean(74), 6.0);
      rep[15] = FXAI_ClipSym(CurrentWindowFeatureRecentMean(6, MathMax(CurrentWindowSize() / 5, 3)), 6.0);

      double logits[FXAI_TESS_CLASS_COUNT];
      for(int c=0; c<FXAI_TESS_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int r=0; r<FXAI_TESS_REP; r++)
            z += m_w_cls[c][r] * rep[r];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      double active = FXAI_Clamp(1.0 - probs[FXAI_TESS_SKIP], 0.0, 1.0);
      double move_raw = m_b_move;
      double logv_raw = m_b_logv;
      for(int r=0; r<FXAI_TESS_REP; r++)
      {
         move_raw += m_w_move[r] * rep[r];
         logv_raw += m_w_logv[r] * rep[r];
      }
      double sigma = FXAI_Clamp(MathExp(0.35 * FXAI_ClipSym(logv_raw, 6.0)), 0.05, 24.0);
      move_mean_points = MathMax(0.0,
                                 (0.56 * MathAbs(move_raw) +
                                  0.18 * projection_diversity +
                                  0.14 * shadow_drift +
                                  0.12 * w_r) * active);
      if(move_mean_points <= 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         move_mean_points = 0.50 * m_move_ema_abs * active;

      q25_points = MathMax(0.0, move_mean_points - 0.48 * sigma);
      q75_points = MathMax(q25_points, move_mean_points + 0.48 * sigma);
      confidence = FXAI_Clamp(0.56 * MathMax(probs[FXAI_TESS_BUY], probs[FXAI_TESS_SELL]) +
                              0.18 * projection_diversity +
                              0.14 * FXAI_Clamp(1.0 - shadow_drift / 3.0, 0.0, 1.0) +
                              0.12 * active,
                              0.0, 1.0);
      reliability = FXAI_Clamp(0.34 +
                               0.26 * FXAI_Clamp(1.0 - shadow_drift / 3.0, 0.0, 1.0) +
                               0.22 * active +
                               0.18 * (m_move_ready ? 1.0 : 0.0),
                               0.0, 1.0);

      if(update_state)
      {
         for(int p=0; p<FXAI_TESS_SHADOWS; p++)
         {
            m_shadow_u[p] = 0.80 * m_shadow_u[p] + 0.20 * u[p];
            m_shadow_v[p] = 0.80 * m_shadow_v[p] + 0.20 * v[p];
         }
         for(int a=0; a<FXAI_TESS_AXES; a++)
            m_latent_mem[a] = 0.82 * m_latent_mem[a] + 0.18 * rot[a];
      }
   }

public:
   CFXAIAITesseract(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TESSERACT; }
   virtual string AIName(void) const { return "ai_tesseract"; }

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);
      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 192);
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      ResetNorm();
      ResetShadowState();
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
      double rep[FXAI_TESS_REP];
      double q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, diversity = 0.0, drift = 0.0;
      ForwardModel(x, false, false, xa, rep, class_probs, expected_move_points, q25, q75, conf, rel, diversity, drift);
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
      double rep[FXAI_TESS_REP];
      double diversity = 0.0, drift = 0.0;
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
                   diversity,
                   drift);
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
      double lr = FXAI_Clamp(h.lr, 0.0004, 0.0200);
      double reg = FXAI_Clamp(h.l2, 0.0, 0.0250);

      double xa[FXAI_AI_WEIGHTS];
      double rep[FXAI_TESS_REP];
      double probs[FXAI_TESS_CLASS_COUNT];
      double move_pred = 0.0, q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, diversity = 0.0, drift = 0.0;
      ForwardModel(x, true, false, xa, rep, probs, move_pred, q25, q75, conf, rel, diversity, drift);

      int cls = ResolveClass(y, move_points);
      double sample_w = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 6.00);
      double lr_eff = lr * sample_w;

      double target[FXAI_TESS_CLASS_COUNT];
      target[0] = (cls == FXAI_TESS_SELL ? 1.0 : 0.0);
      target[1] = (cls == FXAI_TESS_BUY ? 1.0 : 0.0);
      target[2] = (cls == FXAI_TESS_SKIP ? 1.0 : 0.0);
      for(int c=0; c<FXAI_TESS_CLASS_COUNT; c++)
      {
         double err = probs[c] - target[c];
         for(int r=0; r<FXAI_TESS_REP; r++)
            m_w_cls[c][r] -= lr_eff * (err * rep[r] + reg * m_w_cls[c][r]);
         m_b_cls[c] -= lr_eff * err;
      }

      double target_move = (cls == FXAI_TESS_SKIP ? 0.0 : MathMax(MathAbs(move_points), ResolveMinMovePoints()));
      double err_move = move_pred - target_move;
      for(int r=0; r<FXAI_TESS_REP; r++)
      {
         m_w_move[r] -= lr_eff * (0.25 * err_move * rep[r] + reg * m_w_move[r]);
         double logv_err = FXAI_ClipSym(m_w_logv[r] * rep[r], 4.0) - MathLog(MathMax(MathAbs(err_move), 0.08));
         m_w_logv[r] -= lr_eff * (0.06 * logv_err * rep[r] + 0.5 * reg * m_w_logv[r]);
      }
      m_b_move -= lr_eff * 0.25 * err_move;
      m_b_logv -= lr_eff * 0.05 * (MathExp(FXAI_ClipSym(m_b_logv, 4.0)) - MathMax(MathAbs(err_move), 0.08));

      double dir_target = (cls == FXAI_TESS_BUY ? 1.0 : (cls == FXAI_TESS_SELL ? -1.0 : 0.0));
      double dir_pred = probs[FXAI_TESS_BUY] - probs[FXAI_TESS_SELL];
      double core_err = dir_target - dir_pred;
      double diversity_target = (cls == FXAI_TESS_SKIP ? 0.35 : 0.80);
      double diversity_pull = diversity_target - diversity;
      double drift_target = (cls == FXAI_TESS_SKIP ? 0.30 : 0.16);
      double drift_pull = drift_target - drift;

      for(int a=0; a<FXAI_TESS_AXES; a++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double xi = xa[i];
            double grad = 0.042 * core_err * xi + 0.014 * diversity_pull * xi - 0.010 * drift_pull * xi;
            m_w_axis[a][i] = (1.0 - lr_eff * reg) * m_w_axis[a][i] + lr_eff * grad;
         }
      }

      for(int i=0; i<6; i++)
         m_theta[i] = FXAI_ClipSym(m_theta[i] + lr_eff * (0.05 * diversity_pull - 0.03 * drift_pull), 3.14159);
      for(int p=0; p<FXAI_TESS_SHADOWS; p++)
      {
         m_proj_bias[p] = FXAI_ClipSym(m_proj_bias[p] + lr_eff * (0.05 * diversity_pull - 0.03 * drift_pull), 6.0);
         m_proj_mix[p] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_proj_mix[p] + lr_eff * (0.04 * core_err + 0.02 * diversity_pull), 4.0);
      }

      double rep_live[FXAI_TESS_REP];
      double probs_live[FXAI_TESS_CLASS_COUNT];
      double move_live = 0.0, q25_live = 0.0, q75_live = 0.0, conf_live = 0.0, rel_live = 0.0, diversity_live = 0.0, drift_live = 0.0;
      ForwardModel(x, false, true, xa, rep_live, probs_live, move_live, q25_live, q75_live, conf_live, rel_live, diversity_live, drift_live);

      UpdateNativeQualityHeads(xa, sample_w, lr, reg);
      m_step++;
   }
};

#endif // __FXAI_AI_TESSERACT_MQH__
