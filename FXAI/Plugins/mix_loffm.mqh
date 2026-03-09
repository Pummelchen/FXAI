#ifndef __FXAI_AI_LOFFM_MQH__
#define __FXAI_AI_LOFFM_MQH__

#include "..\plugin_base.mqh"

#define FXAI_LOFFM_EXPERTS 4
#define FXAI_LOFFM_DERIVED 10
#define FXAI_LOFFM_STATE 6
#define FXAI_LOFFM_LATENT 12
#define FXAI_LOFFM_MOVE_FEATS 8
#define FXAI_LOFFM_REPLAY 12

class CFXAIAILOFFM : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_steps;
   double m_gate_w[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_DERIVED];
   double m_dir_w[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_LATENT];
   double m_dir_g2[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_LATENT];
   double m_move_head_w[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_MOVE_FEATS];
   double m_move_g2[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_MOVE_FEATS];
   double m_skip_w[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_DERIVED];
   double m_skip_g2[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_DERIVED];
   double m_latent[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_STATE];
   double m_conf_ema[FXAI_LOFFM_EXPERTS];
   double m_edge_ema[FXAI_LOFFM_EXPERTS];
   double m_hit_ema[FXAI_LOFFM_EXPERTS];
   double m_expert_mass[FXAI_LOFFM_EXPERTS];
   double m_usage_ema[FXAI_LOFFM_EXPERTS];
   double m_global_state[FXAI_LOFFM_STATE];
   double m_global_edge_ema;
   double m_global_hit_ema;
   double m_replay_d[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_REPLAY][FXAI_LOFFM_DERIVED];
   double m_loffm_replay_move[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_REPLAY];
   int    m_loffm_replay_label[FXAI_LOFFM_EXPERTS][FXAI_LOFFM_REPLAY];
   int    m_loffm_replay_head[FXAI_LOFFM_EXPERTS];
   int    m_replay_count[FXAI_LOFFM_EXPERTS];

   double ClampProb(const double p) const
   {
      return FXAI_Clamp(p, 0.0005, 0.9995);
   }

   void ResetModel(void)
   {
      m_initialized = false;
      m_steps = 0;
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         for(int k=0; k<FXAI_LOFFM_DERIVED; k++) m_gate_w[e][k] = 0.0;
         for(int k=0; k<FXAI_LOFFM_LATENT;  k++) { m_dir_w[e][k] = 0.0; m_dir_g2[e][k] = 0.0; }
         for(int k=0; k<FXAI_LOFFM_MOVE_FEATS; k++) { m_move_head_w[e][k] = 0.0; m_move_g2[e][k] = 0.0; }
         for(int k=0; k<FXAI_LOFFM_DERIVED; k++) { m_skip_w[e][k] = 0.0; m_skip_g2[e][k] = 0.0; }
         for(int k=0; k<FXAI_LOFFM_STATE; k++) m_latent[e][k] = 0.0;
         m_conf_ema[e]   = 0.50;
         m_edge_ema[e]   = 0.0;
         m_hit_ema[e]    = 0.50;
         m_expert_mass[e]= 0.0;
         m_usage_ema[e]  = 1.0 / (double)FXAI_LOFFM_EXPERTS;
         m_loffm_replay_head[e] = 0;
         m_replay_count[e] = 0;
         for(int r=0; r<FXAI_LOFFM_REPLAY; r++)
         {
            for(int k=0; k<FXAI_LOFFM_DERIVED; k++) m_replay_d[e][r][k] = 0.0;
            m_loffm_replay_move[e][r] = 0.0;
            m_loffm_replay_label[e][r] = (int)FXAI_LABEL_SKIP;
         }
      }
      for(int k=0; k<FXAI_LOFFM_STATE; k++) m_global_state[k] = 0.0;
      m_global_edge_ema = 0.0;
      m_global_hit_ema  = 0.50;
   }

   void SeedWeights(void)
   {
      // Expert 0: trend-pressure
      m_gate_w[0][0] =  1.20; m_gate_w[0][1] =  0.80; m_gate_w[0][2] = -0.25;
      m_gate_w[0][3] =  0.40; m_gate_w[0][4] =  0.15; m_gate_w[0][5] =  0.30;
      m_gate_w[0][6] = -0.20; m_gate_w[0][7] =  0.25; m_gate_w[0][8] =  0.10; m_gate_w[0][9] = 0.10;
      // Expert 1: mean reversion
      m_gate_w[1][0] = -0.85; m_gate_w[1][1] =  0.25; m_gate_w[1][2] =  0.70;
      m_gate_w[1][3] =  0.10; m_gate_w[1][4] = -0.25; m_gate_w[1][5] =  0.05;
      m_gate_w[1][6] =  0.25; m_gate_w[1][7] = -0.10; m_gate_w[1][8] =  0.15; m_gate_w[1][9] = -0.10;
      // Expert 2: squeeze-breakout
      m_gate_w[2][0] =  0.45; m_gate_w[2][1] =  1.10; m_gate_w[2][2] = -0.20;
      m_gate_w[2][3] =  0.90; m_gate_w[2][4] =  0.50; m_gate_w[2][5] =  0.10;
      m_gate_w[2][6] = -0.35; m_gate_w[2][7] =  0.20; m_gate_w[2][8] =  0.05; m_gate_w[2][9] =  0.05;
      // Expert 3: noisy-neutral
      m_gate_w[3][0] = -0.20; m_gate_w[3][1] = -0.35; m_gate_w[3][2] =  0.25;
      m_gate_w[3][3] =  0.10; m_gate_w[3][4] =  0.15; m_gate_w[3][5] =  0.55;
      m_gate_w[3][6] =  0.80; m_gate_w[3][7] = -0.10; m_gate_w[3][8] =  0.20; m_gate_w[3][9] = 0.0;

      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         // Direction head is deliberately tiny and conservative at init.
         m_dir_w[e][0] = 0.05;
         m_dir_w[e][1] = (e == 0 ? 0.35 : (e == 1 ? -0.30 : (e == 2 ? 0.22 : 0.05)));
         m_dir_w[e][2] = (e == 2 ? 0.22 : 0.10);
         m_dir_w[e][3] = (e == 1 ? 0.18 : 0.08);
         m_dir_w[e][4] = (e == 3 ? -0.22 : -0.06);
         m_dir_w[e][5] = (e == 0 ? 0.16 : (e == 1 ? -0.16 : 0.08));
         m_dir_w[e][6] = (e == 0 ? 0.14 : (e == 2 ? 0.10 : -0.08));
         m_dir_w[e][7] = (e == 1 ? 0.12 : 0.04);
         m_dir_w[e][8] = (e == 3 ? -0.18 : -0.04);
         m_dir_w[e][9] = 0.06;
         m_dir_w[e][10]= 0.04;
         m_dir_w[e][11]= (e == 2 ? 0.18 : 0.06);

         m_skip_w[e][0] = (e == 3 ? 0.25 : -0.10);
         m_skip_w[e][5] = (e == 3 ? 0.18 : 0.05);
         m_skip_w[e][8] = (e == 3 ? 0.20 : 0.08);
      }
   }

   void EnsureBootstrapped(void)
   {
      if(m_initialized) return;
      ResetModel();
      SeedWeights();
      m_initialized = true;
   }

   double SafeX(const double &x[], const int idx) const
   {
      if(idx < 0 || idx >= ArraySize(x)) return 0.0;
      double v = x[idx];
      if(!MathIsValidNumber(v)) return 0.0;
      return FXAI_ClipSym(v, 8.0);
   }

   double AvgAbsRange(const double &x[], const int a, const int b) const
   {
      double s = 0.0;
      int n = 0;
      for(int i=a; i<=b; i++)
      {
         s += MathAbs(SafeX(x, i));
         n++;
      }
      if(n <= 0) return 0.0;
      return s / (double)n;
   }

   void BuildDerived(const double &x[], double &d[]) const
   {
      double f1 = SafeX(x, 1),  f2 = SafeX(x, 2),  f3 = SafeX(x, 3),  f4 = SafeX(x, 4);
      double f5 = SafeX(x, 5),  f6 = SafeX(x, 6),  f7 = SafeX(x, 7),  f8 = SafeX(x, 8);
      double f9 = SafeX(x, 9),  f10 = SafeX(x, 10), f11 = SafeX(x, 11), f12 = SafeX(x, 12);
      double g1 = AvgAbsRange(x, 13, 20);
      double g2 = AvgAbsRange(x, 21, 32);
      double g3 = AvgAbsRange(x, 33, 48);
      double g4 = AvgAbsRange(x, 49, 62);

      double directional_impulse = FXAI_ClipSym(0.48*f1 + 0.34*f2 + 0.20*f3 - 0.10*f4 + 0.08*f12, 6.0);
      double volatility_pressure = FXAI_ClipSym(0.70*MathAbs(f6) + 0.45*MathAbs(f7) + 0.25*g1, 6.0);
      double reversion_bias = FXAI_ClipSym(-0.45*f1 + 0.35*f5 - 0.20*f9 + 0.10*f10, 6.0);
      double breakout_potential = FXAI_ClipSym(0.55*MathAbs(f2 - f5) + 0.35*MathAbs(f3 - f4) + 0.15*g2, 6.0);
      double liquidity_stress = FXAI_ClipSym(0.90*MathAbs(f7) + 0.35*MathAbs(f8) + 0.10*g3, 6.0);
      double asymmetry_proxy = FXAI_ClipSym(0.45*f10 - 0.35*f11 + 0.25*f12, 6.0);
      double smooth_trend = FXAI_ClipSym(0.65*f1 + 0.25*f2 - 0.12*f5 + 0.08*g4, 6.0);
      double noise_proxy = FXAI_ClipSym(0.35*MathAbs(f3 - f2) + 0.35*MathAbs(f5 - f4) + 0.20*g1 + 0.10*g4, 6.0);
      double parity_strain = FXAI_ClipSym(0.40*f8 + 0.25*f11 + 0.12*g2 - 0.10*g3, 6.0);
      double session_bias = 0.0;
      MqlDateTime dt;
      TimeToStruct(ResolveContextTime(), dt);
      if(dt.hour >= 6 && dt.hour < 12) session_bias = 0.20;
      else if(dt.hour >= 12 && dt.hour < 17) session_bias = 0.35;
      else if(dt.hour >= 17 && dt.hour < 21) session_bias = 0.15;
      else session_bias = -0.10;

      d[0] = 1.0;
      d[1] = directional_impulse;
      d[2] = volatility_pressure;
      d[3] = reversion_bias;
      d[4] = breakout_potential;
      d[5] = liquidity_stress;
      d[6] = asymmetry_proxy;
      d[7] = smooth_trend;
      d[8] = noise_proxy;
      d[9] = parity_strain + session_bias;
   }

   void BuildLatentInput(const double &d[], const int expert, double &z[]) const
   {
      z[0] = 1.0;
      z[1] = d[1];
      z[2] = d[2];
      z[3] = d[3];
      z[4] = d[4];
      z[5] = d[5];
      z[6] = d[6];
      z[7] = d[7];
      z[8] = d[8];
      z[9] = m_latent[expert][0];
      z[10]= m_latent[expert][1];
      z[11]= m_latent[expert][2] + 0.50*m_global_state[0] - 0.30*m_global_state[4];
   }

   void BuildMoveInput(const double &d[], const int expert, double &z[]) const
   {
      z[0] = 1.0;
      z[1] = MathAbs(d[1]);
      z[2] = MathAbs(d[2]);
      z[3] = MathAbs(d[4]);
      z[4] = MathAbs(m_latent[expert][0]);
      z[5] = MathAbs(m_latent[expert][1]);
      z[6] = MathAbs(m_edge_ema[expert]);
      z[7] = MathAbs(m_global_edge_ema);
   }

   void SoftmaxExperts(const double &d[], double &g[]) const
   {
      double logits[FXAI_LOFFM_EXPERTS];
      double mx = -1e9;
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         double z = 0.0;
         for(int k=0; k<FXAI_LOFFM_DERIVED; k++)
            z += m_gate_w[e][k] * d[k];
         z -= 0.35 * (m_usage_ema[e] - (1.0 / (double)FXAI_LOFFM_EXPERTS));
         z += 0.15 * m_hit_ema[e] - 0.10 * m_conf_ema[e];
         logits[e] = FXAI_ClipSym(z, 20.0);
         if(logits[e] > mx) mx = logits[e];
      }
      double s = 0.0;
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         g[e] = MathExp(FXAI_ClipSym(logits[e] - mx, 30.0));
         s += g[e];
      }
      if(s <= 0.0)
      {
         for(int e=0; e<FXAI_LOFFM_EXPERTS; e++) g[e] = 1.0 / (double)FXAI_LOFFM_EXPERTS;
         return;
      }
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++) g[e] /= s;
   }

   double PredictExpertSkip(const int expert, const double &d[]) const
   {
      double z = 0.0;
      for(int k=0; k<FXAI_LOFFM_DERIVED; k++) z += m_skip_w[expert][k] * d[k];
      z += 0.22 * MathAbs(d[5]) + 0.14 * MathAbs(d[8]) - 0.18 * m_hit_ema[expert] + 0.10 * m_conf_ema[expert];
      return ClampProb(FXAI_Sigmoid(FXAI_ClipSym(z, 20.0)));
   }

   double PredictExpertUp(const int expert, const double &d[]) const
   {
      double z[FXAI_LOFFM_LATENT];
      BuildLatentInput(d, expert, z);
      double s = 0.0;
      for(int k=0; k<FXAI_LOFFM_LATENT; k++) s += m_dir_w[expert][k] * z[k];
      s += 0.10 * (m_hit_ema[expert] - 0.50) - 0.08 * m_conf_ema[expert];
      return ClampProb(FXAI_Sigmoid(FXAI_ClipSym(s, 20.0)));
   }

   double PredictExpertMove(const int expert, const double &d[]) const
   {
      double z[FXAI_LOFFM_MOVE_FEATS];
      BuildMoveInput(d, expert, z);
      double s = 0.0;
      for(int k=0; k<FXAI_LOFFM_MOVE_FEATS; k++) s += m_move_head_w[expert][k] * z[k];
      double base = 0.45 * MathAbs(d[1]) + 0.35 * MathAbs(d[4]) + 0.20 * MathAbs(m_latent[expert][0]);
      double mv = MathMax(0.0, base + s);
      return FXAI_Clamp(mv, 0.0, 5000.0);
   }

   void UpdateLatentState(const double &d[], const double &g[], const double move_points)
   {
      double dir = FXAI_ClipSym(move_points, 10.0);
      double mag = MathAbs(dir);
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         double a_fast = 0.08 + 0.04 * g[e];
         double a_slow = 0.02 + 0.02 * g[e];
         m_latent[e][0] = (1.0 - a_fast) * m_latent[e][0] + a_fast * d[1];
         m_latent[e][1] = (1.0 - a_slow) * m_latent[e][1] + a_slow * d[7];
         m_latent[e][2] = (1.0 - a_fast) * m_latent[e][2] + a_fast * d[3];
         m_latent[e][3] = (1.0 - a_slow) * m_latent[e][3] + a_slow * d[2];
         m_latent[e][4] = (1.0 - a_fast) * m_latent[e][4] + a_fast * d[5];
         m_latent[e][5] = (1.0 - a_slow) * m_latent[e][5] + a_slow * dir;
         m_conf_ema[e]  = 0.97 * m_conf_ema[e] + 0.03 * MathAbs(d[8]);
         m_edge_ema[e]  = 0.95 * m_edge_ema[e] + 0.05 * mag;
         m_expert_mass[e] = MathMin(1e6, m_expert_mass[e] + g[e]);
      }
      for(int k=0; k<FXAI_LOFFM_STATE; k++)
         m_global_state[k] = 0.97 * m_global_state[k];
      m_global_state[0] += 0.03 * d[1];
      m_global_state[1] += 0.03 * d[2];
      m_global_state[2] += 0.03 * d[3];
      m_global_state[3] += 0.03 * d[4];
      m_global_state[4] += 0.03 * d[5];
      m_global_state[5] += 0.03 * dir;
   }

   int TargetDirFromLabel(const int y, const double move_points) const
   {
      if(y == (int)FXAI_LABEL_BUY) return 1;
      if(y == (int)FXAI_LABEL_SELL) return -1;
      if(move_points > 0.0) return 1;
      if(move_points < 0.0) return -1;
      return 0;
   }

   void StoreReplay(const int expert, const double &d[], const int label, const double move_points)
   {
      int slot = m_loffm_replay_head[expert];
      for(int k=0; k<FXAI_LOFFM_DERIVED; k++) m_replay_d[expert][slot][k] = d[k];
      m_loffm_replay_move[expert][slot] = move_points;
      m_loffm_replay_label[expert][slot] = label;
      m_loffm_replay_head[expert] = (slot + 1) % FXAI_LOFFM_REPLAY;
      if(m_replay_count[expert] < FXAI_LOFFM_REPLAY) m_replay_count[expert]++;
   }

   void TrainExpertSample(const int expert,
                          const double &d[],
                          const int label,
                          const double move_points,
                          const double target_move,
                          const double sample_w,
                          const FXAIAIHyperParams &hp,
                          const double gate_weight,
                          const bool adapt_gate)
   {
      int dir_target = TargetDirFromLabel(label, move_points);
      double y01 = (dir_target > 0 ? 1.0 : 0.0);
      double pe = PredictExpertUp(expert, d);
      double me = PredictExpertMove(expert, d);
      double pskip = PredictExpertSkip(expert, d);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.05);
      double gw = FXAI_Clamp(gate_weight, 0.15, 1.50);

      if(dir_target != 0)
      {
         double err = y01 - pe;
         double z[FXAI_LOFFM_LATENT];
         BuildLatentInput(d, expert, z);
         double lr = FXAI_Clamp(0.25 * hp.lr * sample_w * gw, 0.0002, 0.05);
         for(int k=0; k<FXAI_LOFFM_LATENT; k++)
         {
            double grad = err * z[k] - l2 * m_dir_w[expert][k];
            m_dir_g2[expert][k] += grad * grad;
            double step = lr / MathSqrt(1.0 + m_dir_g2[expert][k]);
            m_dir_w[expert][k] = FXAI_ClipSym(m_dir_w[expert][k] + step * grad, 4.0);
         }
      }

      double target_skip = (label == (int)FXAI_LABEL_SKIP ? 1.0 : 0.0);
      double skip_err = target_skip - pskip;
      double skip_lr = FXAI_Clamp(0.18 * hp.lr * sample_w * (0.40 + gw), 0.0002, 0.03);
      for(int k=0; k<FXAI_LOFFM_DERIVED; k++)
      {
         double grad = skip_err * d[k] - 0.5 * l2 * m_skip_w[expert][k];
         m_skip_g2[expert][k] += grad * grad;
         double step = skip_lr / MathSqrt(1.0 + m_skip_g2[expert][k]);
         m_skip_w[expert][k] = FXAI_ClipSym(m_skip_w[expert][k] + step * grad, 3.0);
      }

      double mz[FXAI_LOFFM_MOVE_FEATS];
      BuildMoveInput(d, expert, mz);
      double mv_err = target_move - me;
      double mlr = FXAI_Clamp(0.20 * hp.lr * sample_w * (0.40 + gw), 0.0002, 0.03);
      for(int k=0; k<FXAI_LOFFM_MOVE_FEATS; k++)
      {
         double grad = mv_err * mz[k] - 0.25 * l2 * m_move_head_w[expert][k];
         m_move_g2[expert][k] += grad * grad;
         double step = mlr / MathSqrt(1.0 + m_move_g2[expert][k]);
         m_move_head_w[expert][k] = FXAI_ClipSym(m_move_head_w[expert][k] + step * grad, 8.0);
      }

      if(adapt_gate)
      {
         double align = 0.0;
         if(dir_target != 0)
            align = (dir_target > 0 ? (pe - 0.5) : (0.5 - pe));
         double overload = m_usage_ema[expert] - (1.0 / (double)FXAI_LOFFM_EXPERTS);
         double reward = FXAI_ClipSym(0.70 * align + 0.20 * (target_move > 0.0 ? 1.0 : -0.5) - 0.35 * overload - 0.20 * target_skip, 1.2);
         double gate_lr = FXAI_Clamp(0.05 * hp.lr * sample_w, 0.0001, 0.01);
         for(int k=0; k<FXAI_LOFFM_DERIVED; k++)
            m_gate_w[expert][k] = FXAI_ClipSym(m_gate_w[expert][k] + gate_lr * reward * d[k], 2.5);
      }

      if(dir_target != 0)
      {
         double hit = ((dir_target > 0 && pe >= 0.5) || (dir_target < 0 && pe < 0.5)) ? 1.0 : 0.0;
         m_hit_ema[expert] = 0.98 * m_hit_ema[expert] + 0.02 * hit;
      }
      else
      {
         m_hit_ema[expert] = 0.985 * m_hit_ema[expert] + 0.015 * 0.50;
      }
      m_edge_ema[expert] = 0.97 * m_edge_ema[expert] + 0.03 * target_move;
   }

   void ReplayExpert(const int expert, const FXAIAIHyperParams &hp)
   {
      if(m_replay_count[expert] <= 0) return;
      int slot = m_loffm_replay_head[expert] - 1;
      if(slot < 0) slot += FXAI_LOFFM_REPLAY;
      double rd[FXAI_LOFFM_DERIVED];
      for(int k=0; k<FXAI_LOFFM_DERIVED; k++) rd[k] = m_replay_d[expert][slot][k];
      double replay_target_move = MathAbs(m_loffm_replay_move[expert][slot]);
      TrainExpertSample(expert, rd, m_loffm_replay_label[expert][slot], m_loffm_replay_move[expert][slot], replay_target_move, 0.45, hp, 0.55, false);
   }

public:
   CFXAIAILOFFM(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return AI_LOFFM; }
   virtual string AIName(void) const { return "mix_loffm"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_MIXTURE, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      ResetModel();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      EnsureBootstrapped();
   }

   virtual bool SupportsCorePrediction(void) const
   {
      return true;
   }

   virtual bool PredictModelCore(const double &x[], const FXAIAIHyperParams &hp, double &class_probs[], double &expected_move_points)
   {
      if(!m_initialized) return false;

      double d[FXAI_LOFFM_DERIVED];
      double g[FXAI_LOFFM_EXPERTS];
      BuildDerived(x, d);
      SoftmaxExperts(d, g);

      double p_buy = 0.0;
      double p_sell = 0.0;
      double exp_move = 0.0;
      double p_skip = 0.0;
      double p_mean = 0.0;
      double disagreement = 0.0;

      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         double pe = PredictExpertUp(e, d);
         double me = PredictExpertMove(e, d);
         double ps = PredictExpertSkip(e, d);
         p_buy += g[e] * (1.0 - ps) * pe;
         p_sell += g[e] * (1.0 - ps) * (1.0 - pe);
         p_skip += g[e] * ps;
         exp_move += g[e] * me;
         p_mean += g[e] * pe;
      }
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         double pe = PredictExpertUp(e, d);
         disagreement += g[e] * MathAbs(pe - p_mean);
      }

      double cost = ResolveCostPoints(x);
      if(cost < 0.0) cost = 0.0;
      double min_move = ResolveMinMovePoints();
      if(min_move <= 0.0) min_move = MathMax(0.10, cost);
      double tradable_edge = exp_move - cost;
      double tradable_ratio = (min_move > 0.0 ? tradable_edge / MathMax(min_move, 0.10) : tradable_edge);
      double stress = FXAI_Clamp(0.55*MathAbs(d[5]) + 0.45*MathAbs(d[8]), 0.0, 8.0);
      double conf_penalty = FXAI_Clamp(0.70*disagreement + 0.12*stress, 0.0, 0.95);
      double active = FXAI_Clamp((0.45 * FXAI_Sigmoid(1.2 * tradable_ratio) + 0.40 * (1.0 - conf_penalty) + 0.15 * (1.0 - p_skip)) * (1.0 - 0.35 * p_skip), 0.0, 1.0);

      double dir_total = p_buy + p_sell;
      if(dir_total <= 0.0) dir_total = 1.0;
      p_buy /= dir_total;
      p_sell /= dir_total;

      class_probs[(int)FXAI_LABEL_BUY]  = ClampProb(active * p_buy);
      class_probs[(int)FXAI_LABEL_SELL] = ClampProb(active * p_sell);
      class_probs[(int)FXAI_LABEL_SKIP] = ClampProb(MathMax(p_skip, 1.0 - active + 0.15 * disagreement));

      double s = class_probs[0] + class_probs[1] + class_probs[2];
      if(s <= 0.0) s = 1.0;
      for(int c=0; c<3; c++) class_probs[c] /= s;

      expected_move_points = MathMax(0.0, tradable_edge);
      if(expected_move_points <= 0.0)
         expected_move_points = FXAI_Clamp(0.35 * exp_move, 0.0, 5000.0);
      return true;
   }

protected:
   virtual void TrainModelCore(const int y, const double &x[], const FXAIAIHyperParams &hp, const double move_points)
   {
      EnsureBootstrapped();

      double d[FXAI_LOFFM_DERIVED];
      double g[FXAI_LOFFM_EXPERTS];
      BuildDerived(x, d);
      SoftmaxExperts(d, g);

      double cost = ResolveCostPoints(x);
      if(cost < 0.0) cost = 0.0;
      double edge = MathAbs(move_points) - cost;
      double target_move = MathMax(0.0, edge);
      double sample_w = FXAI_Clamp(MoveSampleWeight(x, move_points), 0.25, 4.0);
      int dir_target = TargetDirFromLabel(y, move_points);

      UpdateLatentState(d, g, move_points);
      FXAI_UpdateMoveEMA(m_global_edge_ema, m_move_ready, target_move, 0.03);
      m_global_hit_ema = 0.98 * m_global_hit_ema + 0.02 * ((dir_target == 0) ? 0.50 : 1.0);

      int best_e = 0;
      double best_g = g[0];
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
      {
         m_usage_ema[e] = 0.985 * m_usage_ema[e] + 0.015 * g[e];
         if(g[e] > best_g) { best_g = g[e]; best_e = e; }
      }

      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
         TrainExpertSample(e, d, y, move_points, target_move, sample_w, hp, g[e], true);

      StoreReplay(best_e, d, y, move_points);
      ReplayExpert(best_e, hp);

      m_steps++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[3];
      double exp_move = 0.0;
      if(PredictModelCore(x, hp, probs, exp_move))
         return probs[(int)FXAI_LABEL_BUY] / MathMax(probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL], 1e-6);
      return 0.50;
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) return ExpectedMovePrior(x);

      double d[FXAI_LOFFM_DERIVED];
      double g[FXAI_LOFFM_EXPERTS];
      BuildDerived(x, d);
      SoftmaxExperts(d, g);

      double exp_move = 0.0;
      for(int e=0; e<FXAI_LOFFM_EXPERTS; e++)
         exp_move += g[e] * PredictExpertMove(e, d);

      double base = ExpectedMovePrior(x);
      if(base > 0.0) return 0.70 * exp_move + 0.30 * base;
      return exp_move;
   }
};

#endif // __FXAI_AI_LOFFM_MQH__
