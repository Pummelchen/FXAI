#ifndef __FXAI_AI_GHA_MQH__
#define __FXAI_AI_GHA_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_GHA_NODES 5
#define FXAI_GHA_DIM 4
#define FXAI_GHA_EDGES 4
#define FXAI_GHA_CLASS_COUNT 3
#define FXAI_GHA_REP 16

#define FXAI_GHA_SELL 0
#define FXAI_GHA_BUY  1
#define FXAI_GHA_SKIP 2

class CFXAIAIGHA : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_norm_ready;
   int    m_norm_steps;
   int    m_step;

   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   double m_w_node[FXAI_GHA_NODES][FXAI_GHA_DIM][FXAI_AI_WEIGHTS];
   double m_b_node[FXAI_GHA_NODES][FXAI_GHA_DIM];
   double m_edge_bias[FXAI_GHA_EDGES];
   double m_edge_mix[FXAI_GHA_EDGES];
   double m_curvature[FXAI_GHA_DIM];
   double m_smooth_beta;

   double m_node_mem[FXAI_GHA_NODES][FXAI_GHA_DIM];

   double m_w_cls[FXAI_GHA_CLASS_COUNT][FXAI_GHA_REP];
   double m_b_cls[FXAI_GHA_CLASS_COUNT];
   double m_w_move[FXAI_GHA_REP];
   double m_b_move;
   double m_w_logv[FXAI_GHA_REP];
   double m_b_logv;

   int ResolveClass(const int y,
                    const double move_points) const
   {
      if(y >= FXAI_GHA_SELL && y <= FXAI_GHA_SKIP)
         return y;
      if(move_points > 0.0) return FXAI_GHA_BUY;
      if(move_points < 0.0) return FXAI_GHA_SELL;
      return FXAI_GHA_SKIP;
   }

   void Softmax3(const double &logits[],
                 double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];
      double den = 0.0;
      for(int c=0; c<FXAI_GHA_CLASS_COUNT; c++)
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
      for(int c=0; c<FXAI_GHA_CLASS_COUNT; c++)
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

   void ResetNodeMemory(void)
   {
      for(int n=0; n<FXAI_GHA_NODES; n++)
         for(int d=0; d<FXAI_GHA_DIM; d++)
            m_node_mem[n][d] = 0.0;
   }

   void InitWeights(void)
   {
      double seed = 0.61 + 0.29 * FXAI_SymbolHash01(_Symbol);
      for(int n=0; n<FXAI_GHA_NODES; n++)
      {
         for(int d=0; d<FXAI_GHA_DIM; d++)
         {
            m_b_node[n][d] = 0.0;
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double a = 0.019 * (double)(n + 1) * (double)(d + 1) * (double)(i + 1);
               m_w_node[n][d][i] = 0.060 * MathSin(seed + a);
            }
         }
      }
      for(int e=0; e<FXAI_GHA_EDGES; e++)
      {
         m_edge_bias[e] = 0.10 * MathCos(seed * (double)(e + 1));
         m_edge_mix[e] = 0.18 * MathSin(seed * (double)(e + 2));
      }
      for(int d=0; d<FXAI_GHA_DIM; d++)
         m_curvature[d] = 0.10 + 0.03 * (double)(d + 1);
      m_smooth_beta = 0.24;

      for(int c=0; c<FXAI_GHA_CLASS_COUNT; c++)
      {
         m_b_cls[c] = (c == FXAI_GHA_SKIP ? 0.12 : 0.0);
         for(int r=0; r<FXAI_GHA_REP; r++)
         {
            double a = 0.057 * (double)(c + 1) * (double)(r + 1);
            m_w_cls[c][r] = 0.050 * MathSin(seed + a);
         }
      }
      for(int r=0; r<FXAI_GHA_REP; r++)
      {
         double a = 0.047 * (double)(r + 1);
         m_w_move[r] = 0.042 * MathCos(seed + a);
         m_w_logv[r] = 0.025 * MathSin(seed + 0.7 * a);
      }
      m_b_move = 0.0;
      m_b_logv = -0.60;
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
      double cost = CurrentWindowFeatureRecentMean(70, MathMax(CurrentWindowSize() / 5, 2));
      double ctx = CurrentWindowFeatureMean(55);
      double regime = CurrentWindowFeatureRange(6, MathMax(CurrentWindowSize() / 4, 4));

      xa[1] = FXAI_ClipSym(0.54 * xa[1] + 0.18 * trend + 0.10 * ctx + 0.08 * regime, 8.0);
      xa[2] = FXAI_ClipSym(0.54 * xa[2] + 0.16 * vol - 0.10 * cost + 0.10 * regime, 8.0);
      xa[6] = FXAI_ClipSym(0.58 * xa[6] + 0.22 * vol + 0.10 * ctx + 0.08 * cost, 8.0);
      xa[10] = FXAI_ClipSym(0.64 * xa[10] + 0.14 * trend + 0.12 * ctx - 0.08 * cost, 8.0);
      xa[11] = FXAI_ClipSym(0.62 * xa[11] + 0.12 * regime + 0.12 * ctx + 0.08 * vol, 8.0);
   }

   double DotNodePair(const double &node[][FXAI_GHA_DIM],
                      const int a,
                      const int b) const
   {
      double s = 0.0;
      for(int d=0; d<FXAI_GHA_DIM; d++)
         s += node[a][d] * node[b][d];
      return s;
   }

   double DistNodePair(const double &node[][FXAI_GHA_DIM],
                       const int a,
                       const int b) const
   {
      double s = 0.0;
      for(int d=0; d<FXAI_GHA_DIM; d++)
      {
         double dv = node[a][d] - node[b][d];
         s += dv * dv;
      }
      return MathSqrt(MathMax(s, 1e-6));
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
                     double &smoothness,
                     double &divergence)
   {
      BuildWindowAwareInput(x, xa);
      if(adapt_norm)
         UpdateInputStats(xa);

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      double node[FXAI_GHA_NODES][FXAI_GHA_DIM];
      for(int n=0; n<FXAI_GHA_NODES; n++)
      {
         for(int d=0; d<FXAI_GHA_DIM; d++)
         {
            double z = m_b_node[n][d];
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
               z += m_w_node[n][d][i] * xn[i];
            node[n][d] = TanhSafe(z + 0.12 * m_node_mem[n][d]);
         }
      }

      double edge[FXAI_GHA_EDGES];
      double vol = CurrentWindowFeatureStd(0);
      double cost = CurrentWindowFeatureRecentMean(70, MathMax(CurrentWindowSize() / 5, 2));
      double trend = CurrentWindowFeatureSlope(0);
      edge[0] = FXAI_Sigmoid(m_edge_bias[0] + (0.30 + 0.12 * m_edge_mix[0]) * DotNodePair(node, 0, 1) + 0.18 * vol);
      edge[1] = FXAI_Sigmoid(m_edge_bias[1] + (0.28 + 0.12 * m_edge_mix[1]) * DotNodePair(node, 0, 2) + 0.12 * trend);
      edge[2] = FXAI_Sigmoid(m_edge_bias[2] + (0.24 + 0.12 * m_edge_mix[2]) * DotNodePair(node, 1, 3) - 0.16 * cost);
      edge[3] = FXAI_Sigmoid(m_edge_bias[3] + (0.22 + 0.12 * m_edge_mix[3]) * DotNodePair(node, 2, 4) + 0.14 * CurrentWindowFeatureMean(55));

      double mapped[FXAI_GHA_NODES][FXAI_GHA_DIM];
      for(int n=0; n<FXAI_GHA_NODES; n++)
      {
         double msg[FXAI_GHA_DIM];
         for(int d=0; d<FXAI_GHA_DIM; d++)
            msg[d] = node[n][d];

         if(n == 0)
            for(int d=0; d<FXAI_GHA_DIM; d++)
               msg[d] += edge[0] * node[1][d] + edge[1] * node[2][d];
         else if(n == 1)
            for(int d=0; d<FXAI_GHA_DIM; d++)
               msg[d] += edge[0] * node[0][d] + edge[2] * node[3][d];
         else if(n == 2)
            for(int d=0; d<FXAI_GHA_DIM; d++)
               msg[d] += edge[1] * node[0][d] + edge[3] * node[4][d];
         else if(n == 3)
            for(int d=0; d<FXAI_GHA_DIM; d++)
               msg[d] += edge[2] * node[1][d] + 0.35 * node[4][d];
         else
            for(int d=0; d<FXAI_GHA_DIM; d++)
               msg[d] += edge[3] * node[2][d] + 0.35 * node[3][d];

         double norm = 0.0;
         for(int d=0; d<FXAI_GHA_DIM; d++)
            norm += msg[d] * msg[d];
         norm = MathSqrt(MathMax(norm, 1e-6));
         for(int d=0; d<FXAI_GHA_DIM; d++)
         {
            double curv = MathMax(m_curvature[d], 1e-4);
            mapped[n][d] = msg[d] / (1.0 + curv * norm + 0.20 * m_smooth_beta);
         }
      }

      smoothness = 0.0;
      divergence = 0.0;
      int pair_n = 0;
      for(int a=0; a<FXAI_GHA_NODES; a++)
      {
         for(int b=a + 1; b<FXAI_GHA_NODES; b++)
         {
            double dist = DistNodePair(mapped, a, b);
            smoothness += dist;
            pair_n++;
         }
      }
      if(pair_n > 0) smoothness /= (double)pair_n;
      divergence = 0.50 * DistNodePair(mapped, 0, 2) + 0.50 * DistNodePair(mapped, 1, 4);

      double pooled[FXAI_GHA_DIM];
      double contrast[FXAI_GHA_DIM];
      for(int d=0; d<FXAI_GHA_DIM; d++)
      {
         pooled[d] = 0.0;
         for(int n=0; n<FXAI_GHA_NODES; n++)
            pooled[d] += mapped[n][d];
         pooled[d] /= (double)FXAI_GHA_NODES;
         contrast[d] = mapped[0][d] - mapped[3][d];
      }

      for(int d=0; d<FXAI_GHA_DIM; d++)
      {
         rep[d] = FXAI_ClipSym(pooled[d], 6.0);
         rep[4 + d] = FXAI_ClipSym(contrast[d], 6.0);
      }
      rep[8] = edge[0];
      rep[9] = edge[1];
      rep[10] = edge[2];
      rep[11] = edge[3];
      rep[12] = FXAI_ClipSym(smoothness, 6.0);
      rep[13] = FXAI_ClipSym(divergence, 6.0);
      rep[14] = FXAI_ClipSym(vol, 6.0);
      rep[15] = FXAI_ClipSym(cost, 6.0);

      double logits[FXAI_GHA_CLASS_COUNT];
      for(int c=0; c<FXAI_GHA_CLASS_COUNT; c++)
      {
         double z = m_b_cls[c];
         for(int r=0; r<FXAI_GHA_REP; r++)
            z += m_w_cls[c][r] * rep[r];
         logits[c] = z;
      }
      Softmax3(logits, probs);

      double active = FXAI_Clamp(1.0 - probs[FXAI_GHA_SKIP], 0.0, 1.0);
      double move_raw = m_b_move;
      double logv_raw = m_b_logv;
      for(int r=0; r<FXAI_GHA_REP; r++)
      {
         move_raw += m_w_move[r] * rep[r];
         logv_raw += m_w_logv[r] * rep[r];
      }
      double sigma = FXAI_Clamp(MathExp(0.35 * FXAI_ClipSym(logv_raw, 6.0)), 0.05, 24.0);
      move_mean_points = MathMax(0.0,
                                 (0.56 * MathAbs(move_raw) +
                                  0.16 * divergence +
                                  0.14 * MathMax(0.0, 1.20 - smoothness) +
                                  0.14 * (edge[0] + edge[1] + edge[2] + edge[3]) * 0.25) * active);
      if(move_mean_points <= 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         move_mean_points = 0.52 * m_move_ema_abs * active;

      q25_points = MathMax(0.0, move_mean_points - 0.50 * sigma);
      q75_points = MathMax(q25_points, move_mean_points + 0.50 * sigma);
      confidence = FXAI_Clamp(0.56 * MathMax(probs[FXAI_GHA_BUY], probs[FXAI_GHA_SELL]) +
                              0.18 * divergence +
                              0.14 * (1.0 - FXAI_Clamp(smoothness / 2.0, 0.0, 1.0)) +
                              0.12 * active,
                              0.0, 1.0);
      reliability = FXAI_Clamp(0.34 +
                               0.24 * (1.0 - FXAI_Clamp(smoothness / 2.0, 0.0, 1.0)) +
                               0.22 * active +
                               0.20 * (m_move_ready ? 1.0 : 0.0),
                               0.0, 1.0);

      if(update_memory)
      {
         for(int n=0; n<FXAI_GHA_NODES; n++)
            for(int d=0; d<FXAI_GHA_DIM; d++)
               m_node_mem[n][d] = 0.80 * m_node_mem[n][d] + 0.20 * mapped[n][d];
      }
   }

public:
   CFXAIAIGHA(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_GHA; }
   virtual string AIName(void) const { return "ai_gha"; }

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
      ResetNodeMemory();
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
      double rep[FXAI_GHA_REP];
      double q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, smooth = 0.0, div = 0.0;
      ForwardModel(x, false, false, xa, rep, class_probs, expected_move_points, q25, q75, conf, rel, smooth, div);
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
      double rep[FXAI_GHA_REP];
      double smooth = 0.0, div = 0.0;
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
                   smooth,
                   div);
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
      double rep[FXAI_GHA_REP];
      double probs[FXAI_GHA_CLASS_COUNT];
      double move_pred = 0.0, q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, smooth = 0.0, div = 0.0;
      ForwardModel(x, true, false, xa, rep, probs, move_pred, q25, q75, conf, rel, smooth, div);

      int cls = ResolveClass(y, move_points);
      double sample_w = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 6.00);
      double lr_eff = lr * sample_w;

      double target[FXAI_GHA_CLASS_COUNT];
      target[0] = (cls == FXAI_GHA_SELL ? 1.0 : 0.0);
      target[1] = (cls == FXAI_GHA_BUY ? 1.0 : 0.0);
      target[2] = (cls == FXAI_GHA_SKIP ? 1.0 : 0.0);
      for(int c=0; c<FXAI_GHA_CLASS_COUNT; c++)
      {
         double err = probs[c] - target[c];
         for(int r=0; r<FXAI_GHA_REP; r++)
            m_w_cls[c][r] -= lr_eff * (err * rep[r] + reg * m_w_cls[c][r]);
         m_b_cls[c] -= lr_eff * err;
      }

      double target_move = (cls == FXAI_GHA_SKIP ? 0.0 : MathMax(MathAbs(move_points), ResolveMinMovePoints()));
      double err_move = move_pred - target_move;
      for(int r=0; r<FXAI_GHA_REP; r++)
      {
         m_w_move[r] -= lr_eff * (0.24 * err_move * rep[r] + reg * m_w_move[r]);
         double logv_err = FXAI_ClipSym(m_w_logv[r] * rep[r], 4.0) - MathLog(MathMax(MathAbs(err_move), 0.08));
         m_w_logv[r] -= lr_eff * (0.06 * logv_err * rep[r] + 0.5 * reg * m_w_logv[r]);
      }
      m_b_move -= lr_eff * 0.24 * err_move;
      m_b_logv -= lr_eff * 0.05 * (MathExp(FXAI_ClipSym(m_b_logv, 4.0)) - MathMax(MathAbs(err_move), 0.08));

      double dir_target = (cls == FXAI_GHA_BUY ? 1.0 : (cls == FXAI_GHA_SELL ? -1.0 : 0.0));
      double dir_pred = probs[FXAI_GHA_BUY] - probs[FXAI_GHA_SELL];
      double core_err = dir_target - dir_pred;
      double smooth_target = (cls == FXAI_GHA_SKIP ? 1.10 : 0.78);
      double smooth_pull = smooth_target - smooth;
      double div_target = (cls == FXAI_GHA_SKIP ? 0.30 : 0.85);
      double div_pull = div_target - div;

      for(int n=0; n<FXAI_GHA_NODES; n++)
      {
         for(int d=0; d<FXAI_GHA_DIM; d++)
         {
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            {
               double xi = xa[i];
               double grad = 0.045 * core_err * xi + 0.012 * smooth_pull * xi + 0.010 * div_pull * xi;
               m_w_node[n][d][i] = (1.0 - lr_eff * reg) * m_w_node[n][d][i] + lr_eff * grad;
            }
         }
      }

      for(int e=0; e<FXAI_GHA_EDGES; e++)
      {
         m_edge_bias[e] = FXAI_ClipSym(m_edge_bias[e] + lr_eff * (0.06 * div_pull - 0.05 * smooth_pull), 6.0);
         m_edge_mix[e] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_edge_mix[e] + lr_eff * (0.04 * core_err + 0.02 * div_pull), 4.0);
      }
      for(int d=0; d<FXAI_GHA_DIM; d++)
         m_curvature[d] = FXAI_Clamp(m_curvature[d] + lr_eff * (0.018 * smooth_pull - 0.010 * div_pull), 0.01, 0.80);
      m_smooth_beta = FXAI_Clamp(m_smooth_beta + lr_eff * (0.02 * smooth_pull), 0.05, 0.70);

      double rep_live[FXAI_GHA_REP];
      double probs_live[FXAI_GHA_CLASS_COUNT];
      double move_live = 0.0, q25_live = 0.0, q75_live = 0.0, conf_live = 0.0, rel_live = 0.0, smooth_live = 0.0, div_live = 0.0;
      ForwardModel(x, false, true, xa, rep_live, probs_live, move_live, q25_live, q75_live, conf_live, rel_live, smooth_live, div_live);

      UpdateNativeQualityHeads(xa, sample_w, lr, reg);
      m_step++;
   }
};

#endif // __FXAI_AI_GHA_MQH__
