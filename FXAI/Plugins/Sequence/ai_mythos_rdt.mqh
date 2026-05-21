#ifndef __FXAI_AI_MYTHOS_RDT_MQH__
#define __FXAI_AI_MYTHOS_RDT_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_MYTHOS_HIDDEN 12
#define FXAI_MYTHOS_EXPERTS 4
#define FXAI_MYTHOS_SHARED 2
#define FXAI_MYTHOS_REP 18
#define FXAI_MYTHOS_ROUTER_FEATS (FXAI_MYTHOS_HIDDEN + 5)
#define FXAI_MYTHOS_MAX_LOOPS 8
#define FXAI_MYTHOS_CLASS_COUNT 3

#define FXAI_MYTHOS_SELL 0
#define FXAI_MYTHOS_BUY  1
#define FXAI_MYTHOS_SKIP 2

class CFXAIAIMythosRDT : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   bool   m_norm_ready;
   int    m_norm_steps;
   int    m_step;

   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   double m_prelude_w[FXAI_MYTHOS_HIDDEN][FXAI_AI_WEIGHTS];
   double m_prelude_b[FXAI_MYTHOS_HIDDEN];
   double m_decay[FXAI_MYTHOS_HIDDEN];
   double m_inject[FXAI_MYTHOS_HIDDEN];
   double m_depth_lora[FXAI_MYTHOS_MAX_LOOPS][FXAI_MYTHOS_HIDDEN];
   double m_loop_memory[FXAI_MYTHOS_HIDDEN];

   double m_router[FXAI_MYTHOS_EXPERTS][FXAI_MYTHOS_ROUTER_FEATS];
   double m_expert_up[FXAI_MYTHOS_EXPERTS][FXAI_MYTHOS_HIDDEN];
   double m_expert_down[FXAI_MYTHOS_EXPERTS][FXAI_MYTHOS_HIDDEN];
   double m_expert_bias[FXAI_MYTHOS_EXPERTS];
   double m_shared_up[FXAI_MYTHOS_SHARED][FXAI_MYTHOS_HIDDEN];
   double m_shared_down[FXAI_MYTHOS_SHARED][FXAI_MYTHOS_HIDDEN];
   double m_shared_bias[FXAI_MYTHOS_SHARED];
   double m_usage_ema[FXAI_MYTHOS_EXPERTS];

   double m_coda_w[FXAI_MYTHOS_CLASS_COUNT][FXAI_MYTHOS_REP];
   double m_coda_b[FXAI_MYTHOS_CLASS_COUNT];
   double m_mythos_move_w[FXAI_MYTHOS_REP];
   double m_b_move;
   double m_logv_w[FXAI_MYTHOS_REP];
   double m_b_logv;

   double TanhSafe(const double z) const
   {
      double c = FXAI_ClipSym(z, 12.0);
      double e2 = MathExp(2.0 * c);
      return (e2 - 1.0) / (e2 + 1.0);
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
         for(int i=0; i<n; i++)
            weight[i] = inv;
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

   void ResetLoopMemory(void)
   {
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         m_loop_memory[d] = 0.0;
   }

   void UpdateInputStats(const double &x[])
   {
      m_norm_steps++;
      double a = 1.0 / (double)MathMin(MathMax(m_norm_steps, 1), 256);
      if(m_norm_steps > 32)
         a = 0.015;

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

   void NormalizeHidden(double &h[]) const
   {
      double rms = 0.0;
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         rms += h[d] * h[d];
      rms = MathSqrt(MathMax(rms / (double)FXAI_MYTHOS_HIDDEN, 1e-6));
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         h[d] = FXAI_ClipSym(h[d] / rms, 6.0);
   }

   void BuildWindowAwareInput(const double &x[],
                              double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
      if(CurrentWindowSize() <= 1)
         return;

      double trend = CurrentWindowFeatureSlope(0);
      double fast_delta = CurrentWindowFeatureRecentDelta(0, MathMax(CurrentWindowSize() / 4, 2));
      double vol = CurrentWindowFeatureStd(0);
      double range_short = CurrentWindowFeatureRange(0, MathMax(CurrentWindowSize() / 4, 4));
      double liquidity_cost = CurrentWindowFeatureRecentMean(70, MathMax(CurrentWindowSize() / 5, 2));
      double macro_pressure = CurrentWindowFeatureMean(55);

      xa[1] = FXAI_ClipSym(0.52 * xa[1] + 0.18 * trend + 0.14 * fast_delta + 0.08 * macro_pressure, 8.0);
      xa[2] = FXAI_ClipSym(0.54 * xa[2] + 0.18 * vol + 0.12 * range_short - 0.08 * liquidity_cost, 8.0);
      xa[6] = FXAI_ClipSym(0.58 * xa[6] + 0.22 * vol + 0.10 * macro_pressure + 0.08 * range_short, 8.0);
      xa[10] = FXAI_ClipSym(0.62 * xa[10] + 0.14 * trend + 0.12 * macro_pressure - 0.08 * liquidity_cost, 8.0);
      xa[11] = FXAI_ClipSym(0.60 * xa[11] + 0.14 * range_short + 0.12 * vol + 0.08 * fast_delta, 8.0);
   }

   void InitWeights(void)
   {
      double seed = 0.73 + 0.31 * FXAI_SymbolHash01(_Symbol);
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         m_prelude_b[d] = 0.0;
         m_decay[d] = 0.58 + 0.035 * (double)(d % 6);
         m_inject[d] = 0.18 + 0.018 * (double)((d + 3) % 5);
         m_loop_memory[d] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double a = 0.017 * (double)(d + 1) * (double)(i + 1);
            m_prelude_w[d][i] = 0.052 * MathSin(seed + a);
         }
      }

      for(int t=0; t<FXAI_MYTHOS_MAX_LOOPS; t++)
      {
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            double a = 0.23 * (double)(t + 1) + 0.071 * (double)(d + 1);
            m_depth_lora[t][d] = 0.030 * MathCos(seed + a);
         }
      }

      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         m_expert_bias[e] = 0.05 * MathCos(seed * (double)(e + 1));
         m_usage_ema[e] = 1.0 / (double)FXAI_MYTHOS_EXPERTS;
         for(int r=0; r<FXAI_MYTHOS_ROUTER_FEATS; r++)
            m_router[e][r] = 0.040 * MathSin(seed + 0.031 * (double)(e + 1) * (double)(r + 1));
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            m_expert_up[e][d] = 0.062 * MathSin(seed + 0.047 * (double)(e + 1) * (double)(d + 1));
            m_expert_down[e][d] = 0.055 * MathCos(seed + 0.053 * (double)(e + 2) * (double)(d + 1));
         }
      }

      for(int s=0; s<FXAI_MYTHOS_SHARED; s++)
      {
         m_shared_bias[s] = 0.04 * MathSin(seed * (double)(s + 2));
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            m_shared_up[s][d] = 0.045 * MathSin(seed + 0.041 * (double)(s + 1) * (double)(d + 2));
            m_shared_down[s][d] = 0.040 * MathCos(seed + 0.049 * (double)(s + 2) * (double)(d + 1));
         }
      }

      for(int c=0; c<FXAI_MYTHOS_CLASS_COUNT; c++)
      {
         m_coda_b[c] = (c == FXAI_MYTHOS_SKIP ? 0.16 : 0.0);
         for(int r=0; r<FXAI_MYTHOS_REP; r++)
         {
            double a = 0.051 * (double)(c + 1) * (double)(r + 1);
            m_coda_w[c][r] = 0.048 * MathSin(seed + a);
         }
      }
      for(int r=0; r<FXAI_MYTHOS_REP; r++)
      {
         m_mythos_move_w[r] = 0.040 * MathCos(seed + 0.043 * (double)(r + 1));
         m_logv_w[r] = 0.023 * MathSin(seed + 0.037 * (double)(r + 1));
      }
      m_b_move = 0.0;
      m_b_logv = -0.62;
      m_initialized = true;
   }

   int ResolveLoopBudget(void) const
   {
      int loops = 3;
      if(m_ctx_horizon_minutes >= 15) loops++;
      if(m_ctx_horizon_minutes >= 60) loops++;
      if(m_ctx_horizon_minutes >= 240) loops++;
      if(CurrentWindowSize() >= 48) loops++;
      if(MathAbs(CurrentWindowFeatureStd(0)) > 0.70 || MathAbs(CurrentWindowFeatureSlope(0)) > 0.45)
         loops++;
      if(loops < 2) loops = 2;
      if(loops > FXAI_MYTHOS_MAX_LOOPS) loops = FXAI_MYTHOS_MAX_LOOPS;
      return loops;
   }

   void EncodePrelude(const double &x[],
                      const bool adapt_norm,
                      double &xa[],
                      double &encoded[])
   {
      BuildWindowAwareInput(x, xa);
      if(adapt_norm)
         UpdateInputStats(xa);

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, xn);

      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         double z = m_prelude_b[d] + 0.08 * m_loop_memory[d];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            z += m_prelude_w[d][i] * xn[i];
         encoded[d] = TanhSafe(z);
      }
      NormalizeHidden(encoded);
   }

   void BuildRouterFeatures(const double &h[],
                            const int loop_t,
                            const int max_loops,
                            double &rf[]) const
   {
      rf[0] = 1.0;
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         rf[d + 1] = FXAI_ClipSym(h[d], 6.0);

      double depth = (double)(loop_t + 1) / (double)MathMax(max_loops, 1);
      rf[FXAI_MYTHOS_HIDDEN + 1] = FXAI_ClipSym(2.0 * depth - 1.0, 2.0);
      rf[FXAI_MYTHOS_HIDDEN + 2] = FXAI_ClipSym(CurrentWindowFeatureSlope(0), 4.0);
      rf[FXAI_MYTHOS_HIDDEN + 3] = FXAI_ClipSym(CurrentWindowFeatureStd(0), 4.0);
      rf[FXAI_MYTHOS_HIDDEN + 4] = FXAI_ClipSym(CurrentWindowFeatureRecentDelta(0, MathMax(CurrentWindowSize() / 4, 2)), 4.0);
   }

   void RouterSoftmax(const double &rf[],
                      double &gate[],
                      double &route_entropy) const
   {
      double score[FXAI_MYTHOS_EXPERTS];
      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         double z = 0.0;
         for(int r=0; r<FXAI_MYTHOS_ROUTER_FEATS; r++)
            z += m_router[e][r] * rf[r];
         z -= 0.40 * (m_usage_ema[e] - (1.0 / (double)FXAI_MYTHOS_EXPERTS));
         score[e] = z;
      }

      SoftmaxN(score, FXAI_MYTHOS_EXPERTS, gate);
      route_entropy = 0.0;
      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         double p = MathMax(gate[e], 1e-9);
         route_entropy -= p * MathLog(p);
      }
      route_entropy /= MathLog((double)FXAI_MYTHOS_EXPERTS);
      route_entropy = FXAI_Clamp(route_entropy, 0.0, 1.0);
   }

   void ExpertMixture(const double &h[],
                      const double &gate[],
                      double &expert_out[]) const
   {
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         expert_out[d] = 0.0;

      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         double u = m_expert_bias[e];
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            u += m_expert_up[e][d] * h[d];
         double act = TanhSafe(u / MathSqrt((double)FXAI_MYTHOS_HIDDEN));
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            expert_out[d] += gate[e] * act * m_expert_down[e][d];
      }

      for(int s=0; s<FXAI_MYTHOS_SHARED; s++)
      {
         double u = m_shared_bias[s];
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            u += m_shared_up[s][d] * h[d];
         double act = TanhSafe(u / MathSqrt((double)FXAI_MYTHOS_HIDDEN));
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            expert_out[d] += 0.35 * act * m_shared_down[s][d];
      }
   }

   void RunRecurrent(const double &encoded[],
                     const bool update_state,
                     double &hidden[],
                     double &gate_out[],
                     double &loop_stability,
                     double &route_entropy,
                     double &halt_signal)
   {
      int max_loops = ResolveLoopBudget();
      double h[FXAI_MYTHOS_HIDDEN];
      double accum[FXAI_MYTHOS_HIDDEN];
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         h[d] = FXAI_ClipSym(encoded[d] + 0.08 * m_loop_memory[d], 6.0);
         accum[d] = 0.0;
      }

      double cumulative = 0.0;
      double total_w = 0.0;
      loop_stability = 0.0;
      route_entropy = 0.0;
      halt_signal = 0.0;

      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
         gate_out[e] = 1.0 / (double)FXAI_MYTHOS_EXPERTS;

      for(int t=0; t<max_loops; t++)
      {
         double prev[FXAI_MYTHOS_HIDDEN];
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            prev[d] = h[d];

         double rf[FXAI_MYTHOS_ROUTER_FEATS];
         double gate[FXAI_MYTHOS_EXPERTS];
         double entropy = 0.0;
         BuildRouterFeatures(h, t, max_loops, rf);
         RouterSoftmax(rf, gate, entropy);

         double expert_out[FXAI_MYTHOS_HIDDEN];
         ExpertMixture(h, gate, expert_out);

         double change = 0.0;
         double energy = 0.0;
         double depth = (double)(t + 1) / (double)MathMax(max_loops, 1);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            double phase = MathSin((double)(t + 1) * 0.73 + (double)(d + 1) * 0.19);
            double adapted = expert_out[d] + m_depth_lora[t][d] * phase + 0.04 * depth * encoded[d];
            h[d] = m_decay[d] * h[d] + m_inject[d] * encoded[d] + TanhSafe(adapted);
            h[d] = FXAI_ClipSym(h[d], 8.0);
         }
         NormalizeHidden(h);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            change += MathAbs(h[d] - prev[d]);
            energy += MathAbs(h[d]);
         }
         change /= (double)FXAI_MYTHOS_HIDDEN;
         energy /= (double)FXAI_MYTHOS_HIDDEN;

         double p_halt = FXAI_Sigmoid(1.10 + 0.28 * (double)t - 1.70 * change - 0.10 * energy);
         double weight = MathMin(MathMax(1.0 - cumulative, 0.0), p_halt);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            accum[d] += weight * h[d];
         cumulative += p_halt * MathMax(1.0 - cumulative, 0.0);
         total_w += weight;

         loop_stability = FXAI_Clamp(1.0 / (1.0 + change + 0.12 * energy), 0.0, 1.0);
         route_entropy = entropy;
         halt_signal = FXAI_Clamp(cumulative, 0.0, 1.0);
         for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
            gate_out[e] = gate[e];

         if(t >= 1 && (cumulative >= 0.985 || change < 0.018))
            break;
      }

      if(total_w <= 1e-9)
      {
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            hidden[d] = h[d];
      }
      else
      {
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            hidden[d] = accum[d] / total_w;
      }
      NormalizeHidden(hidden);

      if(update_state)
      {
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            m_loop_memory[d] = 0.82 * m_loop_memory[d] + 0.18 * hidden[d];
         for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
            m_usage_ema[e] = 0.985 * m_usage_ema[e] + 0.015 * gate_out[e];
      }
   }

   void BuildRepresentation(const double &encoded[],
                            const double &hidden[],
                            const double loop_stability,
                            const double route_entropy,
                            const double halt_signal,
                            double &rep[]) const
   {
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         rep[d] = FXAI_ClipSym(hidden[d], 6.0);

      double align = 0.0;
      double delta = 0.0;
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         align += hidden[d] * encoded[d];
         delta += MathAbs(hidden[d] - encoded[d]);
      }
      align /= (double)FXAI_MYTHOS_HIDDEN;
      delta /= (double)FXAI_MYTHOS_HIDDEN;

      rep[12] = FXAI_ClipSym(align, 6.0);
      rep[13] = FXAI_ClipSym(delta, 6.0);
      rep[14] = FXAI_ClipSym(CurrentWindowFeatureSlope(0), 6.0);
      rep[15] = FXAI_ClipSym(CurrentWindowFeatureStd(0), 6.0);
      rep[16] = FXAI_ClipSym(loop_stability + 0.25 * halt_signal, 6.0);
      rep[17] = FXAI_ClipSym(route_entropy, 6.0);
   }

   void DecodeOutput(const double &rep[],
                     double &probs[],
                     double &move_mean_points,
                     double &q25_points,
                     double &q75_points,
                     double &confidence,
                     double &reliability,
                     const double loop_stability,
                     const double route_entropy,
                     const double halt_signal) const
   {
      double logits[FXAI_MYTHOS_CLASS_COUNT];
      for(int c=0; c<FXAI_MYTHOS_CLASS_COUNT; c++)
      {
         double z = m_coda_b[c];
         for(int r=0; r<FXAI_MYTHOS_REP; r++)
            z += m_coda_w[c][r] * rep[r];
         logits[c] = z;
      }

      double dir_score = 0.38 * rep[0] + 0.24 * rep[3] + 0.18 * rep[12] + 0.16 * rep[14] - 0.08 * rep[15];
      double uncertainty = FXAI_Clamp(0.45 * route_entropy + 0.35 * (1.0 - loop_stability) + 0.20 * (1.0 - halt_signal), 0.0, 1.0);
      logits[FXAI_MYTHOS_BUY] += 0.32 * dir_score;
      logits[FXAI_MYTHOS_SELL] -= 0.32 * dir_score;
      logits[FXAI_MYTHOS_SKIP] += 0.20 + 0.45 * uncertainty;
      Softmax3(logits, probs);

      double raw_move = m_b_move;
      double logv = m_b_logv;
      for(int r=0; r<FXAI_MYTHOS_REP; r++)
      {
         raw_move += m_mythos_move_w[r] * rep[r];
         logv += m_logv_w[r] * rep[r];
      }

      double min_move = MathMax(ResolveMinMovePoints(), 0.10);
      double cost = (m_ctx_cost_ready && m_ctx_cost_points >= 0.0 ? m_ctx_cost_points : 0.0);
      double edge = MathAbs(raw_move) + 0.20 * MathAbs(dir_score) + 0.18 * MathAbs(rep[14]) + 0.12 * rep[15];
      double active_gate = FXAI_Sigmoid((edge - cost) / MathMax(min_move, 0.10));
      active_gate *= FXAI_Clamp(0.45 + 0.35 * loop_stability + 0.20 * halt_signal, 0.0, 1.0);

      probs[FXAI_MYTHOS_BUY] *= active_gate;
      probs[FXAI_MYTHOS_SELL] *= active_gate;
      probs[FXAI_MYTHOS_SKIP] = MathMax(probs[FXAI_MYTHOS_SKIP], 1.0 - active_gate);
      NormalizeClassDistribution(probs);

      double active = FXAI_Clamp(1.0 - probs[FXAI_MYTHOS_SKIP], 0.0, 1.0);
      move_mean_points = MathMax(0.0, edge * active);
      if(move_mean_points <= 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         move_mean_points = 0.45 * m_move_ema_abs * active;

      double sigma = FXAI_Clamp(MathExp(0.35 * FXAI_ClipSym(logv, 6.0)), 0.05, 24.0);
      sigma = MathMax(sigma, 0.25 * min_move + 0.35 * move_mean_points + 0.25 * uncertainty);
      q25_points = MathMax(0.0, move_mean_points - 0.52 * sigma);
      q75_points = MathMax(q25_points, move_mean_points + 0.52 * sigma);
      confidence = FXAI_Clamp(0.52 * MathMax(probs[FXAI_MYTHOS_BUY], probs[FXAI_MYTHOS_SELL]) +
                              0.22 * loop_stability +
                              0.14 * halt_signal +
                              0.12 * active,
                              0.0, 1.0);
      reliability = FXAI_Clamp(0.30 +
                               0.24 * MathMin((double)m_step / 96.0, 1.0) +
                               0.24 * loop_stability +
                               0.14 * (1.0 - route_entropy) +
                               0.08 * (m_move_ready ? 1.0 : 0.0),
                               0.0, 1.0);
   }

   void ForwardModel(const double &x[],
                     const bool adapt_norm,
                     const bool update_state,
                     double &xa[],
                     double &encoded[],
                     double &hidden[],
                     double &gate[],
                     double &rep[],
                     double &probs[],
                     double &move_mean_points,
                     double &q25_points,
                     double &q75_points,
                     double &confidence,
                     double &reliability,
                     double &loop_stability,
                     double &route_entropy,
                     double &halt_signal)
   {
      EncodePrelude(x, adapt_norm, xa, encoded);
      RunRecurrent(encoded, update_state, hidden, gate, loop_stability, route_entropy, halt_signal);
      BuildRepresentation(encoded, hidden, loop_stability, route_entropy, halt_signal, rep);
      DecodeOutput(rep,
                   probs,
                   move_mean_points,
                   q25_points,
                   q75_points,
                   confidence,
                   reliability,
                   loop_stability,
                   route_entropy,
                   halt_signal);
   }

public:
   CFXAIAIMythosRDT(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_MYTHOS_RDT; }
   virtual string AIName(void) const { return "ai_mythos_rdt"; }
   virtual int PersistentStateVersion(void) const { return 12; }
   virtual bool SupportsNativeParameterSnapshot(void) const { return true; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING |
                                 FXAI_CAP_REPLAY |
                                 FXAI_CAP_STATEFUL |
                                 FXAI_CAP_WINDOW_CONTEXT |
                                 FXAI_CAP_MULTI_HORIZON |
                                 FXAI_CAP_NATIVE_DISTRIBUTION |
                                 FXAI_CAP_SELF_TEST);
      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, FXAI_MAX_SEQUENCE_BARS);
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      ResetNorm();
      ResetLoopMemory();
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
      double encoded[FXAI_MYTHOS_HIDDEN];
      double hidden[FXAI_MYTHOS_HIDDEN];
      double gate[FXAI_MYTHOS_EXPERTS];
      double rep[FXAI_MYTHOS_REP];
      double q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, stability = 0.0, entropy = 0.0, halt = 0.0;
      ForwardModel(x, false, false, xa, encoded, hidden, gate, rep, class_probs, expected_move_points, q25, q75, conf, rel, stability, entropy, halt);
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
      double encoded[FXAI_MYTHOS_HIDDEN];
      double hidden[FXAI_MYTHOS_HIDDEN];
      double gate[FXAI_MYTHOS_EXPERTS];
      double rep[FXAI_MYTHOS_REP];
      double stability = 0.0, entropy = 0.0, halt = 0.0;
      ForwardModel(x,
                   false,
                   false,
                   xa,
                   encoded,
                   hidden,
                   gate,
                   rep,
                   out.class_probs,
                   out.move_mean_points,
                   out.move_q25_points,
                   out.move_q75_points,
                   out.confidence,
                   out.reliability,
                   stability,
                   entropy,
                   halt);

      NormalizeClassDistribution(out.class_probs);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.mfe_mean_points = MathMax(out.move_q75_points, out.move_mean_points * (1.05 + 0.25 * out.confidence));
      out.mae_mean_points = MathMax(0.0, out.move_mean_points * (0.28 + 0.28 * out.class_probs[(int)FXAI_LABEL_SKIP] + 0.20 * (1.0 - stability)));
      out.hit_time_frac = FXAI_Clamp(0.68 - 0.30 * halt + 0.18 * out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      out.path_risk = FXAI_Clamp(0.30 * out.class_probs[(int)FXAI_LABEL_SKIP] +
                                 0.30 * (1.0 - stability) +
                                 0.25 * entropy +
                                 0.15 * out.mae_mean_points / MathMax(out.mfe_mean_points, 0.10),
                                 0.0, 1.0);
      out.fill_risk = FXAI_Clamp(ResolveCostPoints(x) / MathMax(out.move_mean_points + MathMax(ResolveMinMovePoints(), 0.10), 0.10), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      out.has_path_quality = true;
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
      double pseudo_move = 0.0;
      if(y == (int)FXAI_LABEL_BUY)
         pseudo_move = 1.0;
      else if(y == (int)FXAI_LABEL_SELL)
         pseudo_move = -1.0;
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FXAIAIHyperParams hps = ScaleHyperParamsForMove(hp, move_points);
      double lr = FXAI_Clamp(hps.lr, 0.0004, 0.0180);
      double reg = FXAI_Clamp(hps.l2, 0.0, 0.0300);

      double xa[FXAI_AI_WEIGHTS];
      double encoded[FXAI_MYTHOS_HIDDEN];
      double hidden[FXAI_MYTHOS_HIDDEN];
      double gate[FXAI_MYTHOS_EXPERTS];
      double rep[FXAI_MYTHOS_REP];
      double probs[FXAI_MYTHOS_CLASS_COUNT];
      double move_pred = 0.0, q25 = 0.0, q75 = 0.0, conf = 0.0, rel = 0.0, stability = 0.0, entropy = 0.0, halt = 0.0;
      ForwardModel(x, true, false, xa, encoded, hidden, gate, rep, probs, move_pred, q25, q75, conf, rel, stability, entropy, halt);

      int cls = NormalizeClassLabel(y, xa, move_points);
      double sample_w = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.20, 6.00);
      double lr_eff = lr * sample_w;

      double target[FXAI_MYTHOS_CLASS_COUNT];
      target[FXAI_MYTHOS_SELL] = (cls == FXAI_MYTHOS_SELL ? 1.0 : 0.0);
      target[FXAI_MYTHOS_BUY] = (cls == FXAI_MYTHOS_BUY ? 1.0 : 0.0);
      target[FXAI_MYTHOS_SKIP] = (cls == FXAI_MYTHOS_SKIP ? 1.0 : 0.0);

      for(int c=0; c<FXAI_MYTHOS_CLASS_COUNT; c++)
      {
         double err = probs[c] - target[c];
         for(int r=0; r<FXAI_MYTHOS_REP; r++)
            m_coda_w[c][r] -= lr_eff * (err * rep[r] + reg * m_coda_w[c][r]);
         m_coda_b[c] -= lr_eff * err;
      }

      double target_move = (cls == FXAI_MYTHOS_SKIP ? 0.0 : MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10)));
      double err_move = move_pred - target_move;
      for(int r=0; r<FXAI_MYTHOS_REP; r++)
      {
         m_mythos_move_w[r] -= lr_eff * (0.22 * err_move * rep[r] + reg * m_mythos_move_w[r]);
         double logv_target = MathLog(MathMax(MathAbs(err_move), 0.08));
         double logv_err = FXAI_ClipSym(m_logv_w[r] * rep[r], 4.0) - logv_target;
         m_logv_w[r] -= lr_eff * (0.05 * logv_err * rep[r] + 0.5 * reg * m_logv_w[r]);
      }
      m_b_move -= lr_eff * 0.22 * err_move;
      m_b_logv -= lr_eff * 0.04 * (MathExp(FXAI_ClipSym(m_b_logv, 4.0)) - MathMax(MathAbs(err_move), 0.08));

      double dir_target = (cls == FXAI_MYTHOS_BUY ? 1.0 : (cls == FXAI_MYTHOS_SELL ? -1.0 : 0.0));
      double dir_pred = probs[FXAI_MYTHOS_BUY] - probs[FXAI_MYTHOS_SELL];
      double core_err = dir_target - dir_pred;
      double stability_target = (cls == FXAI_MYTHOS_SKIP ? 0.58 : 0.84);
      double entropy_target = (cls == FXAI_MYTHOS_SKIP ? 0.82 : 0.46);
      double halt_target = (cls == FXAI_MYTHOS_SKIP ? 0.92 : 0.72);

      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            double grad = 0.018 * core_err * xa[i] + 0.006 * (stability_target - stability) * xa[i];
            m_prelude_w[d][i] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_prelude_w[d][i] + lr_eff * grad, 4.0);
         }
         m_prelude_b[d] = FXAI_ClipSym(m_prelude_b[d] + lr_eff * 0.020 * core_err, 4.0);
         m_decay[d] = FXAI_Clamp(m_decay[d] + lr_eff * 0.010 * (stability_target - stability) - lr_eff * 0.001 * (m_decay[d] - 0.68), 0.35, 0.965);
         m_inject[d] = FXAI_Clamp(m_inject[d] + lr_eff * 0.008 * (halt_target - halt) - lr_eff * 0.001 * (m_inject[d] - 0.22), 0.05, 0.55);
      }

      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         double route_reward = ((dir_target == 0.0 ? entropy_target : 1.0 - entropy_target) - gate[e]);
         for(int r=0; r<FXAI_MYTHOS_ROUTER_FEATS; r++)
         {
            double rf_proxy = (r == 0 ? 1.0 : (r <= FXAI_MYTHOS_HIDDEN ? hidden[r - 1] : rep[14]));
            m_router[e][r] = FXAI_ClipSym((1.0 - lr_eff * 0.20 * reg) * m_router[e][r] + lr_eff * 0.026 * route_reward * rf_proxy, 4.0);
         }
         double expert_signal_err = core_err * gate[e];
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            m_expert_up[e][d] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_expert_up[e][d] + lr_eff * 0.030 * expert_signal_err * hidden[d], 4.0);
            m_expert_down[e][d] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_expert_down[e][d] + lr_eff * 0.026 * expert_signal_err * encoded[d], 4.0);
         }
         m_usage_ema[e] = 0.985 * m_usage_ema[e] + 0.015 * gate[e];
      }

      for(int t=0; t<FXAI_MYTHOS_MAX_LOOPS; t++)
      {
         double depth_pull = (halt_target - halt) * (double)(t + 1) / (double)FXAI_MYTHOS_MAX_LOOPS;
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            m_depth_lora[t][d] = FXAI_ClipSym((1.0 - lr_eff * reg) * m_depth_lora[t][d] + lr_eff * 0.012 * depth_pull * hidden[d], 2.0);
      }

      double xa_live[FXAI_AI_WEIGHTS];
      double encoded_live[FXAI_MYTHOS_HIDDEN];
      double hidden_live[FXAI_MYTHOS_HIDDEN];
      double gate_live[FXAI_MYTHOS_EXPERTS];
      double rep_live[FXAI_MYTHOS_REP];
      double probs_live[FXAI_MYTHOS_CLASS_COUNT];
      double move_live = 0.0, q25_live = 0.0, q75_live = 0.0, conf_live = 0.0, rel_live = 0.0, stability_live = 0.0, entropy_live = 0.0, halt_live = 0.0;
      ForwardModel(x, false, true, xa_live, encoded_live, hidden_live, gate_live, rep_live, probs_live, move_live, q25_live, q75_live, conf_live, rel_live, stability_live, entropy_live, halt_live);

      UpdateNativeQualityHeads(xa, sample_w, lr, reg);
      m_step++;
   }

   virtual bool SaveModelState(const int handle) const
   {
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, (m_initialized ? 1 : 0));
      FileWriteInteger(handle, (m_norm_ready ? 1 : 0));
      FileWriteInteger(handle, m_norm_steps);
      FileWriteInteger(handle, m_step);

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         FileWriteDouble(handle, m_x_mean[i]);
         FileWriteDouble(handle, m_x_var[i]);
      }
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         FileWriteDouble(handle, m_prelude_b[d]);
         FileWriteDouble(handle, m_decay[d]);
         FileWriteDouble(handle, m_inject[d]);
         FileWriteDouble(handle, m_loop_memory[d]);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            FileWriteDouble(handle, m_prelude_w[d][i]);
      }
      for(int t=0; t<FXAI_MYTHOS_MAX_LOOPS; t++)
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            FileWriteDouble(handle, m_depth_lora[t][d]);
      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         FileWriteDouble(handle, m_expert_bias[e]);
         FileWriteDouble(handle, m_usage_ema[e]);
         for(int r=0; r<FXAI_MYTHOS_ROUTER_FEATS; r++)
            FileWriteDouble(handle, m_router[e][r]);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            FileWriteDouble(handle, m_expert_up[e][d]);
            FileWriteDouble(handle, m_expert_down[e][d]);
         }
      }
      for(int s=0; s<FXAI_MYTHOS_SHARED; s++)
      {
         FileWriteDouble(handle, m_shared_bias[s]);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            FileWriteDouble(handle, m_shared_up[s][d]);
            FileWriteDouble(handle, m_shared_down[s][d]);
         }
      }
      for(int c=0; c<FXAI_MYTHOS_CLASS_COUNT; c++)
      {
         FileWriteDouble(handle, m_coda_b[c]);
         for(int r=0; r<FXAI_MYTHOS_REP; r++)
            FileWriteDouble(handle, m_coda_w[c][r]);
      }
      FileWriteDouble(handle, m_b_move);
      FileWriteDouble(handle, m_b_logv);
      for(int r=0; r<FXAI_MYTHOS_REP; r++)
      {
         FileWriteDouble(handle, m_mythos_move_w[r]);
         FileWriteDouble(handle, m_logv_w[r]);
      }
      return true;
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      if(handle == INVALID_HANDLE || version < 1)
         return false;

      m_initialized = (FileReadInteger(handle) != 0);
      m_norm_ready = (FileReadInteger(handle) != 0);
      m_norm_steps = FileReadInteger(handle);
      m_step = FileReadInteger(handle);

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = FileReadDouble(handle);
         m_x_var[i] = MathMax(FileReadDouble(handle), 1e-4);
      }
      for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
      {
         m_prelude_b[d] = FileReadDouble(handle);
         m_decay[d] = FXAI_Clamp(FileReadDouble(handle), 0.35, 0.965);
         m_inject[d] = FXAI_Clamp(FileReadDouble(handle), 0.05, 0.55);
         m_loop_memory[d] = FileReadDouble(handle);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_prelude_w[d][i] = FileReadDouble(handle);
      }
      for(int t=0; t<FXAI_MYTHOS_MAX_LOOPS; t++)
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
            m_depth_lora[t][d] = FileReadDouble(handle);
      for(int e=0; e<FXAI_MYTHOS_EXPERTS; e++)
      {
         m_expert_bias[e] = FileReadDouble(handle);
         m_usage_ema[e] = FXAI_Clamp(FileReadDouble(handle), 0.0, 1.0);
         for(int r=0; r<FXAI_MYTHOS_ROUTER_FEATS; r++)
            m_router[e][r] = FileReadDouble(handle);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            m_expert_up[e][d] = FileReadDouble(handle);
            m_expert_down[e][d] = FileReadDouble(handle);
         }
      }
      for(int s=0; s<FXAI_MYTHOS_SHARED; s++)
      {
         m_shared_bias[s] = FileReadDouble(handle);
         for(int d=0; d<FXAI_MYTHOS_HIDDEN; d++)
         {
            m_shared_up[s][d] = FileReadDouble(handle);
            m_shared_down[s][d] = FileReadDouble(handle);
         }
      }
      for(int c=0; c<FXAI_MYTHOS_CLASS_COUNT; c++)
      {
         m_coda_b[c] = FileReadDouble(handle);
         for(int r=0; r<FXAI_MYTHOS_REP; r++)
            m_coda_w[c][r] = FileReadDouble(handle);
      }
      m_b_move = FileReadDouble(handle);
      m_b_logv = FileReadDouble(handle);
      for(int r=0; r<FXAI_MYTHOS_REP; r++)
      {
         m_mythos_move_w[r] = FileReadDouble(handle);
         m_logv_w[r] = FileReadDouble(handle);
      }
      return true;
   }
};

#endif // __FXAI_AI_MYTHOS_RDT_MQH__
