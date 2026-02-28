#ifndef __FX6_AI_STMN_MQH__
#define __FX6_AI_STMN_MQH__

#include "..\plugin_base.mqh"

#define FX6_STMN_NODES 6
#define FX6_STMN_CLASS_COUNT 3
#define FX6_STMN_SEQ 128
#define FX6_STMN_TBPTT 16

#define FX6_STMN_SELL 0
#define FX6_STMN_BUY  1
#define FX6_STMN_SKIP 2

class CFX6AISTMN : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   int    m_step;

   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FX6_AI_WEIGHTS];
   double m_x_var[FX6_AI_WEIGHTS];

   int    m_seq_ptr;
   int    m_seq_len;
   double m_hist_h[FX6_STMN_SEQ][FX6_AI_MLP_HIDDEN];
   double m_hist_g[FX6_STMN_SEQ][FX6_AI_MLP_HIDDEN];

   // Spatio-temporal node masks and parameters.
   double m_group_mask[FX6_STMN_NODES][FX6_AI_WEIGHTS];
   double m_w_node[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_b_node[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];

   double m_w_q[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_w_k[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_adj[FX6_STMN_NODES][FX6_STMN_NODES];

   double m_gate_logit[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_b_sp[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_pool_logit[FX6_STMN_NODES];

   // Temporal memory core (diagonal GRU + residual graph path).
   double m_wz_x[FX6_AI_MLP_HIDDEN], m_wz_h[FX6_AI_MLP_HIDDEN], m_bz[FX6_AI_MLP_HIDDEN];
   double m_wr_x[FX6_AI_MLP_HIDDEN], m_wr_h[FX6_AI_MLP_HIDDEN], m_br[FX6_AI_MLP_HIDDEN];
   double m_wh_x[FX6_AI_MLP_HIDDEN], m_wh_h[FX6_AI_MLP_HIDDEN], m_bh[FX6_AI_MLP_HIDDEN];
   double m_w_res[FX6_AI_MLP_HIDDEN];

   // 3-class head + move distribution head.
   double m_w_cls[FX6_STMN_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_STMN_CLASS_COUNT];

   double m_w_mu[FX6_AI_MLP_HIDDEN],   m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN], m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN],  m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN],  m_b_q75;

   // Training sequence buffer.
   int    m_train_len;
   double m_train_x[FX6_STMN_TBPTT][FX6_AI_WEIGHTS];
   int    m_train_cls[FX6_STMN_TBPTT];
   double m_train_move[FX6_STMN_TBPTT];
   double m_train_cost[FX6_STMN_TBPTT];
   double m_train_w[FX6_STMN_TBPTT];
   double m_train_hprev[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];

   // Forward caches for TBPTT.
   double m_cache_xn[FX6_STMN_TBPTT][FX6_AI_WEIGHTS];
   double m_cache_node_z[FX6_STMN_TBPTT][FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_cache_attn[FX6_STMN_TBPTT][FX6_STMN_NODES][FX6_STMN_NODES];
   double m_cache_msg[FX6_STMN_TBPTT][FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_cache_node_o[FX6_STMN_TBPTT][FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
   double m_cache_pool[FX6_STMN_TBPTT][FX6_STMN_NODES];
   double m_cache_graph[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];

   double m_cache_hprev[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_zgate[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_rgate[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_hcand[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_cache_hnew[FX6_STMN_TBPTT][FX6_AI_MLP_HIDDEN];

   double m_cache_logits[FX6_STMN_TBPTT][FX6_STMN_CLASS_COUNT];
   double m_cache_probs[FX6_STMN_TBPTT][FX6_STMN_CLASS_COUNT];
   double m_cache_mu[FX6_STMN_TBPTT];
   double m_cache_logv[FX6_STMN_TBPTT];
   double m_cache_q25[FX6_STMN_TBPTT];
   double m_cache_q75[FX6_STMN_TBPTT];

   double HuberGrad(const double err, const double delta) const
   {
      double d = (delta > 0.0 ? delta : 1.0);
      if(err > d) return d;
      if(err < -d) return -d;
      return err;
   }

   double PinballGrad(const double y, const double q, const double tau) const
   {
      if(y >= q) return -tau;
      return (1.0 - tau);
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FX6_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FX6_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FX6_Clamp(logits[2] - m, -30.0, 30.0));
      double s = e0 + e1 + e2;
      if(s <= 0.0)
      {
         probs[0] = 0.3333333;
         probs[1] = 0.3333333;
         probs[2] = 0.3333333;
         return;
      }
      probs[0] = e0 / s;
      probs[1] = e1 / s;
      probs[2] = e2 / s;
   }

   void ResetInputNorm(void)
   {
      m_x_norm_ready = false;
      m_x_norm_steps = 0;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 128 ? 0.05 : 0.015);
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         double d = x[i] - m_x_mean[i];
         m_x_mean[i] += a * d;
         double dv = x[i] - m_x_mean[i];
         m_x_var[i] = (1.0 - a) * m_x_var[i] + a * dv * dv;
         if(m_x_var[i] < 1e-6) m_x_var[i] = 1e-6;
      }

      m_x_norm_steps++;
      if(m_x_norm_steps >= 32) m_x_norm_ready = true;
   }

   void NormalizeInput(const double &x[], double &xn[]) const
   {
      xn[0] = 1.0;
      for(int i=1; i<FX6_AI_WEIGHTS; i++)
      {
         if(!m_x_norm_ready)
         {
            xn[i] = FX6_ClipSym(x[i], 8.0);
            continue;
         }

         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FX6_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FX6_STMN_SEQ; t++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_hist_h[t][h] = 0.0;
            m_hist_g[t][h] = 0.0;
         }
      }
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FX6_STMN_TBPTT; t++)
      {
         m_train_cls[t] = FX6_STMN_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;
         m_cache_mu[t] = 0.0;
         m_cache_logv[t] = 0.0;
         m_cache_q25[t] = 0.0;
         m_cache_q75[t] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_train_x[t][i] = 0.0;
            m_cache_xn[t][i] = 0.0;
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_train_hprev[t][h] = 0.0;
            m_cache_graph[t][h] = 0.0;
            m_cache_hprev[t][h] = 0.0;
            m_cache_zgate[t][h] = 0.0;
            m_cache_rgate[t][h] = 0.0;
            m_cache_hcand[t][h] = 0.0;
            m_cache_hnew[t][h] = 0.0;

            for(int n=0; n<FX6_STMN_NODES; n++)
            {
               m_cache_node_z[t][n][h] = 0.0;
               m_cache_msg[t][n][h] = 0.0;
               m_cache_node_o[t][n][h] = 0.0;
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            m_cache_pool[t][n] = 0.0;
            for(int m=0; m<FX6_STMN_NODES; m++)
               m_cache_attn[t][n][m] = 0.0;
         }

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = 0.0;
            m_cache_probs[t][c] = 1.0 / 3.0;
         }
      }
   }

   void GetLastHidden(double &h_prev[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) h_prev[h] = 0.0;
      if(m_seq_len <= 0 || m_seq_ptr < 0) return;

      int idx = m_seq_ptr;
      if(idx < 0 || idx >= FX6_STMN_SEQ) return;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         h_prev[h] = m_hist_h[idx][h];
   }

   void PushSequenceState(const double &graph[], const double &h_new[])
   {
      m_seq_ptr++;
      if(m_seq_ptr >= FX6_STMN_SEQ) m_seq_ptr = 0;
      if(m_seq_len < FX6_STMN_SEQ) m_seq_len++;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_hist_g[m_seq_ptr][h] = graph[h];
         m_hist_h[m_seq_ptr][h] = h_new[h];
      }
   }

   void BuildGroupMasks(void)
   {
      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_group_mask[n][i] = 0.0;
      }

      // Bias for every node.
      for(int n=0; n<FX6_STMN_NODES; n++)
         m_group_mask[n][0] = 1.0;

      // Node 0: short-horizon returns + slope.
      m_group_mask[0][1] = 1.0;
      m_group_mask[0][2] = 1.0;
      m_group_mask[0][3] = 1.0;
      m_group_mask[0][4] = 1.0;

      // Node 1: local dispersion/volatility.
      m_group_mask[1][5] = 1.0;
      m_group_mask[1][6] = 1.0;

      // Node 2: trading cost proxy.
      m_group_mask[2][7] = 1.0;

      // Node 3: multi-timeframe returns.
      m_group_mask[3][8] = 1.0;
      m_group_mask[3][9] = 1.0;
      m_group_mask[3][10] = 1.0;

      // Node 4: cross-symbol context.
      m_group_mask[4][11] = 1.0;
      m_group_mask[4][12] = 1.0;
      m_group_mask[4][13] = 1.0;

      // Node 5: higher timeframe slopes.
      m_group_mask[5][14] = 1.0;
      m_group_mask[5][15] = 1.0;
   }

   void InitWeights(void)
   {
      ResetInputNorm();
      ResetSequence();
      ResetTrainBuffer();
      BuildGroupMasks();
      m_step = 0;

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         m_pool_logit[n] = (n == 0 ? 0.25 : 0.0);

         for(int m=0; m<FX6_STMN_NODES; m++)
         {
            if(n == m) m_adj[n][m] = 0.20;
            else m_adj[n][m] = -0.05;
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double sh = (double)((n + 1) * (h + 2));
            m_b_node[n][h] = 0.0;
            m_w_q[n][h] = 0.06 * MathSin(0.61 * sh);
            m_w_k[n][h] = 0.06 * MathCos(0.67 * sh);
            m_gate_logit[n][h] = 0.0;
            m_b_sp[n][h] = 0.0;

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
            {
               double s = (double)((n + 1) * (h + 3) * (i + 2));
               double mask = m_group_mask[n][i];
               if(mask > 0.0)
                  m_w_node[n][h][i] = 0.05 * MathSin(0.73 * s);
               else
                  m_w_node[n][h][i] = 0.005 * MathCos(0.59 * s);
            }
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double s = (double)(h + 1);
         m_wz_x[h] = 0.10 * MathSin(0.51 * s);
         m_wz_h[h] = 0.10 * MathCos(0.53 * s);
         m_bz[h] = 0.0;

         m_wr_x[h] = 0.10 * MathCos(0.57 * s);
         m_wr_h[h] = 0.10 * MathSin(0.59 * s);
         m_br[h] = 0.0;

         m_wh_x[h] = 0.14 * MathSin(0.63 * s);
         m_wh_h[h] = 0.12 * MathCos(0.65 * s);
         m_bh[h] = 0.0;

         m_w_res[h] = 0.12;

         m_w_mu[h] = 0.05 * MathSin(0.71 * s);
         m_w_logv[h] = 0.04 * MathCos(0.73 * s);
         m_w_q25[h] = 0.04 * MathSin(0.79 * s);
         m_w_q75[h] = 0.04 * MathCos(0.83 * s);

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
            m_w_cls[c][h] = 0.05 * MathSin((double)((c + 2) * (h + 1)) * 0.69);
      }

      for(int c=0; c<FX6_STMN_CLASS_COUNT; c++) m_b_cls[c] = 0.0;
      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;
   }

   int ResolveClass(const int y,
                    const double &x[],
                    const double move_points) const
   {
      if(y == FX6_STMN_SELL || y == FX6_STMN_BUY || y == FX6_STMN_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FX6_STMN_SKIP;

      if(y > 0) return FX6_STMN_BUY;
      if(y == 0) return FX6_STMN_SELL;
      return (move_points >= 0.0 ? FX6_STMN_BUY : FX6_STMN_SELL);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost,
                      const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double base = FX6_Clamp(sample_w, 0.25, 4.00);

      if(cls == FX6_STMN_SKIP)
      {
         if(edge <= 0.0) return FX6_Clamp(base * 1.6, 0.25, 6.0);
         return FX6_Clamp(base * 0.7, 0.25, 6.0);
      }

      if(edge <= 0.0) return FX6_Clamp(base * 0.55, 0.25, 6.0);
      return FX6_Clamp(base * (1.0 + 0.06 * MathMin(edge, 20.0)), 0.25, 6.0);
   }

   double MoveWeight(const double move_points,
                     const double cost,
                     const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double ew = FX6_Clamp(0.5 + edge / denom, 0.25, 4.0);
      return FX6_Clamp(sample_w * ew, 0.25, 8.0);
   }

   double ScheduledLR(const FX6AIHyperParams &hp) const
   {
      double base = FX6_Clamp(hp.lr, 0.0002, 0.1200);
      double warm = 1.0;
      if(m_step < 200)
         warm = 0.20 + 0.80 * ((double)m_step / 200.0);
      double decay = 1.0 / MathSqrt(1.0 + 0.002 * MathMax(0, m_step - 200));
      double cyc = 0.95 + 0.05 * MathSin((double)m_step * 0.031);
      return FX6_Clamp(base * warm * decay * cyc, 0.00005, 0.0600);
   }

   void SpatialForward(const double &xn[],
                       double &node_z[][FX6_AI_MLP_HIDDEN],
                       double &attn[][FX6_STMN_NODES],
                       double &msg[][FX6_AI_MLP_HIDDEN],
                       double &node_o[][FX6_AI_MLP_HIDDEN],
                       double &pool_w[],
                       double &graph[]) const
   {
      double q_node[FX6_STMN_NODES];
      double k_node[FX6_STMN_NODES];

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         q_node[n] = 0.0;
         k_node[n] = 0.0;
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double z = m_b_node[n][h];
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
               z += (m_group_mask[n][i] * m_w_node[n][h][i]) * xn[i];

            node_z[n][h] = FX6_Tanh(FX6_ClipSym(z, 12.0));
            q_node[n] += m_w_q[n][h] * node_z[n][h];
            k_node[n] += m_w_k[n][h] * node_z[n][h];
         }
      }

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         double max_s = -1e100;
         for(int m=0; m<FX6_STMN_NODES; m++)
         {
            double s = m_adj[n][m] + q_node[n] + k_node[m];
            attn[n][m] = s;
            if(s > max_s) max_s = s;
         }

         double den = 0.0;
         for(int m=0; m<FX6_STMN_NODES; m++)
         {
            double e = MathExp(FX6_Clamp(attn[n][m] - max_s, -30.0, 30.0));
            attn[n][m] = e;
            den += e;
         }
         if(den <= 0.0) den = 1.0;
         for(int m=0; m<FX6_STMN_NODES; m++) attn[n][m] /= den;
      }

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double s = 0.0;
            for(int m=0; m<FX6_STMN_NODES; m++)
               s += attn[n][m] * node_z[m][h];
            msg[n][h] = s;

            double gate = FX6_Sigmoid(m_gate_logit[n][h]);
            double pre = gate * node_z[n][h] + (1.0 - gate) * msg[n][h] + m_b_sp[n][h];
            node_o[n][h] = FX6_Tanh(FX6_ClipSym(pre, 10.0));
         }
      }

      double maxp = m_pool_logit[0];
      for(int n=1; n<FX6_STMN_NODES; n++)
         if(m_pool_logit[n] > maxp) maxp = m_pool_logit[n];

      double denp = 0.0;
      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         pool_w[n] = MathExp(FX6_Clamp(m_pool_logit[n] - maxp, -30.0, 30.0));
         denp += pool_w[n];
      }
      if(denp <= 0.0) denp = 1.0;
      for(int n=0; n<FX6_STMN_NODES; n++) pool_w[n] /= denp;

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double g = 0.0;
         for(int n=0; n<FX6_STMN_NODES; n++)
            g += pool_w[n] * node_o[n][h];
         graph[h] = FX6_ClipSym(g, 8.0);
      }
   }

   void TemporalForward(const double &graph[],
                        const double &h_prev[],
                        double &z_gate[],
                        double &r_gate[],
                        double &h_cand[],
                        double &h_new[]) const
   {
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         double zpre = m_wz_x[h] * graph[h] + m_wz_h[h] * h_prev[h] + m_bz[h];
         double rpre = m_wr_x[h] * graph[h] + m_wr_h[h] * h_prev[h] + m_br[h];

         z_gate[h] = FX6_Sigmoid(FX6_ClipSym(zpre, 15.0));
         r_gate[h] = FX6_Sigmoid(FX6_ClipSym(rpre, 15.0));

         double cpre = m_wh_x[h] * graph[h] + m_wh_h[h] * (r_gate[h] * h_prev[h]) + m_bh[h];
         h_cand[h] = FX6_Tanh(FX6_ClipSym(cpre, 12.0));

         double hmix = (1.0 - z_gate[h]) * h_prev[h] + z_gate[h] * h_cand[h] + m_w_res[h] * graph[h];
         h_new[h] = FX6_ClipSym(hmix, 8.0);
      }
   }

   void HeadForward(const double &h_state[],
                    double &logits[],
                    double &probs[],
                    double &mu,
                    double &logv,
                    double &q25,
                    double &q75) const
   {
      for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
      {
         double s = m_b_cls[c];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            s += m_w_cls[c][h] * h_state[h];
         logits[c] = FX6_ClipSym(s, 20.0);
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         mu += m_w_mu[h] * h_state[h];
         logv += m_w_logv[h] * h_state[h];
         q25 += m_w_q25[h] * h_state[h];
         q75 += m_w_q75[h] * h_state[h];
      }

      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q75 < q25 + 1e-4) q75 = q25 + 1e-4;
   }

   void ForwardInference(const double &x[],
                         double &probs[],
                         double &mu,
                         double &logv,
                         double &q25,
                         double &q75) const
   {
      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double node_z[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double attn[FX6_STMN_NODES][FX6_STMN_NODES];
      double msg[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double node_o[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double pool[FX6_STMN_NODES];
      double graph[FX6_AI_MLP_HIDDEN];

      SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);

      double h_prev[FX6_AI_MLP_HIDDEN];
      double z_gate[FX6_AI_MLP_HIDDEN];
      double r_gate[FX6_AI_MLP_HIDDEN];
      double h_cand[FX6_AI_MLP_HIDDEN];
      double h_new[FX6_AI_MLP_HIDDEN];
      double logits[FX6_STMN_CLASS_COUNT];

      GetLastHidden(h_prev);
      TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);
      HeadForward(h_new, logits, probs, mu, logv, q25, q75);
   }

   void PushTrainSample(const int cls,
                        const double &x[],
                        const double move_points,
                        const double cost,
                        const double sample_w,
                        const double &h_prev[])
   {
      if(m_train_len < FX6_STMN_TBPTT)
      {
         int t = m_train_len;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_train_x[t][i] = x[i];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) m_train_hprev[t][h] = h_prev[h];
         m_train_cls[t] = cls;
         m_train_move[t] = move_points;
         m_train_cost[t] = cost;
         m_train_w[t] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FX6_STMN_TBPTT; t++)
      {
         int p = t - 1;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_train_x[p][i] = m_train_x[t][i];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) m_train_hprev[p][h] = m_train_hprev[t][h];
         m_train_cls[p] = m_train_cls[t];
         m_train_move[p] = m_train_move[t];
         m_train_cost[p] = m_train_cost[t];
         m_train_w[p] = m_train_w[t];
      }

      int last = FX6_STMN_TBPTT - 1;
      for(int i=0; i<FX6_AI_WEIGHTS; i++) m_train_x[last][i] = x[i];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) m_train_hprev[last][h] = h_prev[h];
      m_train_cls[last] = cls;
      m_train_move[last] = move_points;
      m_train_cost[last] = cost;
      m_train_w[last] = sample_w;
      m_train_len = FX6_STMN_TBPTT;
   }

   void TrainTBPTT(const FX6AIHyperParams &hp)
   {
      int T = m_train_len;
      if(T <= 0) return;

      // Forward pass over buffered sequence.
      for(int t=0; t<T; t++)
      {
         double xraw[FX6_AI_WEIGHTS];
         double xn[FX6_AI_WEIGHTS];
         for(int i=0; i<FX6_AI_WEIGHTS; i++) xraw[i] = m_train_x[t][i];
         NormalizeInput(xraw, xn);
         for(int i=0; i<FX6_AI_WEIGHTS; i++) m_cache_xn[t][i] = xn[i];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            m_cache_hprev[t][h] = (t == 0 ? m_train_hprev[0][h] : m_cache_hnew[t - 1][h]);

         double node_z[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double attn[FX6_STMN_NODES][FX6_STMN_NODES];
         double msg[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double node_o[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double pool[FX6_STMN_NODES];
         double graph[FX6_AI_MLP_HIDDEN];

         SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            m_cache_pool[t][n] = pool[n];
            for(int m=0; m<FX6_STMN_NODES; m++)
               m_cache_attn[t][n][m] = attn[n][m];
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               m_cache_node_z[t][n][h] = node_z[n][h];
               m_cache_msg[t][n][h] = msg[n][h];
               m_cache_node_o[t][n][h] = node_o[n][h];
            }
         }

         double h_prev[FX6_AI_MLP_HIDDEN];
         double z_gate[FX6_AI_MLP_HIDDEN];
         double r_gate[FX6_AI_MLP_HIDDEN];
         double h_cand[FX6_AI_MLP_HIDDEN];
         double h_new[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            h_prev[h] = m_cache_hprev[t][h];
            m_cache_graph[t][h] = graph[h];
         }

         TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_cache_zgate[t][h] = z_gate[h];
            m_cache_rgate[t][h] = r_gate[h];
            m_cache_hcand[t][h] = h_cand[h];
            m_cache_hnew[t][h] = h_new[h];
         }

         double logits[FX6_STMN_CLASS_COUNT];
         double probs[FX6_STMN_CLASS_COUNT];
         double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
         HeadForward(h_new, logits, probs, mu, logv, q25, q75);

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = logits[c];
            m_cache_probs[t][c] = probs[c];
         }
         m_cache_mu[t] = mu;
         m_cache_logv[t] = logv;
         m_cache_q25[t] = q25;
         m_cache_q75[t] = q75;
      }

      // Gradient buffers.
      double g_w_node[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_b_node[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double g_w_q[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double g_w_k[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double g_adj[FX6_STMN_NODES][FX6_STMN_NODES];
      double g_gate_logit[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double g_b_sp[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double g_pool_logit[FX6_STMN_NODES];

      double g_wz_x[FX6_AI_MLP_HIDDEN], g_wz_h[FX6_AI_MLP_HIDDEN], g_bz[FX6_AI_MLP_HIDDEN];
      double g_wr_x[FX6_AI_MLP_HIDDEN], g_wr_h[FX6_AI_MLP_HIDDEN], g_br[FX6_AI_MLP_HIDDEN];
      double g_wh_x[FX6_AI_MLP_HIDDEN], g_wh_h[FX6_AI_MLP_HIDDEN], g_bh[FX6_AI_MLP_HIDDEN];
      double g_w_res[FX6_AI_MLP_HIDDEN];

      double g_w_cls[FX6_STMN_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
      double g_b_cls[FX6_STMN_CLASS_COUNT];

      double g_w_mu[FX6_AI_MLP_HIDDEN], g_w_logv[FX6_AI_MLP_HIDDEN], g_w_q25[FX6_AI_MLP_HIDDEN], g_w_q75[FX6_AI_MLP_HIDDEN];
      double g_b_mu, g_b_logv, g_b_q25, g_b_q75;

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         g_pool_logit[n] = 0.0;
         for(int m=0; m<FX6_STMN_NODES; m++) g_adj[n][m] = 0.0;
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            g_b_node[n][h] = 0.0;
            g_w_q[n][h] = 0.0;
            g_w_k[n][h] = 0.0;
            g_gate_logit[n][h] = 0.0;
            g_b_sp[n][h] = 0.0;
            for(int i=0; i<FX6_AI_WEIGHTS; i++) g_w_node[n][h][i] = 0.0;
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         g_wz_x[h] = 0.0; g_wz_h[h] = 0.0; g_bz[h] = 0.0;
         g_wr_x[h] = 0.0; g_wr_h[h] = 0.0; g_br[h] = 0.0;
         g_wh_x[h] = 0.0; g_wh_h[h] = 0.0; g_bh[h] = 0.0;
         g_w_res[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++) g_w_cls[c][h] = 0.0;
      }
      for(int c=0; c<FX6_STMN_CLASS_COUNT; c++) g_b_cls[c] = 0.0;
      g_b_mu = 0.0;
      g_b_logv = 0.0;
      g_b_q25 = 0.0;
      g_b_q75 = 0.0;

      double dh_next[FX6_AI_MLP_HIDDEN];
      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) dh_next[h] = 0.0;

      for(int t=T-1; t>=0; t--)
      {
         int cls = m_train_cls[t];
         if(cls < FX6_STMN_SELL || cls > FX6_STMN_SKIP) cls = FX6_STMN_SKIP;

         double move = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = m_train_w[t];

         double cw = ClassWeight(cls, move, cost, sw);
         double mw = MoveWeight(move, cost, sw);

         double y_true[FX6_STMN_CLASS_COUNT];
         y_true[0] = (cls == FX6_STMN_SELL ? 1.0 : 0.0);
         y_true[1] = (cls == FX6_STMN_BUY ? 1.0 : 0.0);
         y_true[2] = (cls == FX6_STMN_SKIP ? 1.0 : 0.0);

         double dh[FX6_AI_MLP_HIDDEN];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++) dh[h] = dh_next[h];

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
         {
            double dlog = (m_cache_probs[t][c] - y_true[c]) * cw;
            g_b_cls[c] += dlog;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               g_w_cls[c][h] += dlog * m_cache_hnew[t][h];
               dh[h] += dlog * m_w_cls[c][h];
            }
         }

         double abs_move = MathAbs(move);
         double mu = m_cache_mu[t];
         double logv = m_cache_logv[t];
         double q25 = m_cache_q25[t];
         double q75 = m_cache_q75[t];
         if(q75 < q25 + 1e-4) q75 = q25 + 1e-4;

         double err_mu = mu - abs_move;
         double dmu = HuberGrad(err_mu, 6.0) * mw;

         double var = MathExp(logv);
         if(var < 1e-6) var = 1e-6;
         double dlogv = 0.5 * (1.0 - (err_mu * err_mu) / var) * mw;
         dlogv = FX6_ClipSym(dlogv, 10.0);

         double dq25 = PinballGrad(abs_move, q25, 0.25) * mw;
         double dq75 = PinballGrad(abs_move, q75, 0.75) * mw;
         if(q25 > q75)
         {
            double pen = FX6_ClipSym((q25 - q75), 5.0) * 0.25;
            dq25 += pen;
            dq75 -= pen;
         }

         g_b_mu += dmu;
         g_b_logv += dlogv;
         g_b_q25 += dq25;
         g_b_q75 += dq75;

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            g_w_mu[h] += dmu * m_cache_hnew[t][h];
            g_w_logv[h] += dlogv * m_cache_hnew[t][h];
            g_w_q25[h] += dq25 * m_cache_hnew[t][h];
            g_w_q75[h] += dq75 * m_cache_hnew[t][h];

            dh[h] += dmu * m_w_mu[h] + dlogv * m_w_logv[h] + dq25 * m_w_q25[h] + dq75 * m_w_q75[h];
            dh[h] = FX6_ClipSym(dh[h], 20.0);
         }

         double dgraph[FX6_AI_MLP_HIDDEN];
         double dh_prev[FX6_AI_MLP_HIDDEN];

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            double hpv = m_cache_hprev[t][h];
            double z = m_cache_zgate[t][h];
            double r = m_cache_rgate[t][h];
            double hc = m_cache_hcand[t][h];
            double gh = m_cache_graph[t][h];

            double dht = dh[h];

            g_w_res[h] += dht * gh;
            double dg = dht * m_w_res[h];
            double dhp = dht * (1.0 - z);

            double dz = dht * (hc - hpv);
            double dc = dht * z;

            double da_c = dc * (1.0 - hc * hc);
            g_wh_x[h] += da_c * gh;
            g_wh_h[h] += da_c * (r * hpv);
            g_bh[h] += da_c;
            dg += da_c * m_wh_x[h];
            double dr = da_c * m_wh_h[h] * hpv;
            dhp += da_c * m_wh_h[h] * r;

            double da_r = dr * r * (1.0 - r);
            g_wr_x[h] += da_r * gh;
            g_wr_h[h] += da_r * hpv;
            g_br[h] += da_r;
            dg += da_r * m_wr_x[h];
            dhp += da_r * m_wr_h[h];

            double da_z = dz * z * (1.0 - z);
            g_wz_x[h] += da_z * gh;
            g_wz_h[h] += da_z * hpv;
            g_bz[h] += da_z;
            dg += da_z * m_wz_x[h];
            dhp += da_z * m_wz_h[h];

            dgraph[h] = FX6_ClipSym(dg, 20.0);
            dh_prev[h] = FX6_ClipSym(dhp, 20.0);
         }

         // Spatial backprop at step t.
         double dnode_o[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double dnode_z[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double dmsg[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
         double dattn[FX6_STMN_NODES][FX6_STMN_NODES];
         double dscore[FX6_STMN_NODES][FX6_STMN_NODES];
         double dqn[FX6_STMN_NODES];
         double dkm[FX6_STMN_NODES];

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            dqn[n] = 0.0;
            dkm[n] = 0.0;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               dnode_o[n][h] = 0.0;
               dnode_z[n][h] = 0.0;
               dmsg[n][h] = 0.0;
            }
            for(int m=0; m<FX6_STMN_NODES; m++)
            {
               dattn[n][m] = 0.0;
               dscore[n][m] = 0.0;
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            double dot_pool = 0.0;
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               dnode_o[n][h] += dgraph[h] * m_cache_pool[t][n];
               dot_pool += dgraph[h] * (m_cache_node_o[t][n][h] - m_cache_graph[t][h]);
            }
            g_pool_logit[n] += m_cache_pool[t][n] * dot_pool;
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double o = m_cache_node_o[t][n][h];
               double da = dnode_o[n][h] * (1.0 - o * o);
               g_b_sp[n][h] += da;

               double gate = FX6_Sigmoid(m_gate_logit[n][h]);
               double nz = m_cache_node_z[t][n][h];
               double msgv = m_cache_msg[t][n][h];
               g_gate_logit[n][h] += da * (nz - msgv) * gate * (1.0 - gate);

               dnode_z[n][h] += da * gate;
               dmsg[n][h] += da * (1.0 - gate);
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            for(int m=0; m<FX6_STMN_NODES; m++)
            {
               double ds = 0.0;
               for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
               {
                  ds += dmsg[n][h] * m_cache_node_z[t][m][h];
                  dnode_z[m][h] += dmsg[n][h] * m_cache_attn[t][n][m];
               }
               dattn[n][m] += ds;
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            double dot = 0.0;
            for(int m=0; m<FX6_STMN_NODES; m++)
               dot += dattn[n][m] * m_cache_attn[t][n][m];

            for(int m=0; m<FX6_STMN_NODES; m++)
               dscore[n][m] = m_cache_attn[t][n][m] * (dattn[n][m] - dot);
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            for(int m=0; m<FX6_STMN_NODES; m++)
            {
               double ds = dscore[n][m];
               g_adj[n][m] += ds;
               dqn[n] += ds;
               dkm[m] += ds;
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               g_w_q[n][h] += dqn[n] * m_cache_node_z[t][n][h];
               dnode_z[n][h] += dqn[n] * m_w_q[n][h];

               g_w_k[n][h] += dkm[n] * m_cache_node_z[t][n][h];
               dnode_z[n][h] += dkm[n] * m_w_k[n][h];
            }
         }

         for(int n=0; n<FX6_STMN_NODES; n++)
         {
            for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            {
               double z = m_cache_node_z[t][n][h];
               double da = dnode_z[n][h] * (1.0 - z * z);
               g_b_node[n][h] += da;
               for(int i=0; i<FX6_AI_WEIGHTS; i++)
               {
                  g_w_node[n][h][i] += da * (m_group_mask[n][i] * m_cache_xn[t][i]);
               }
            }
         }

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
            dh_next[h] = dh_prev[h];
      }

      // Global gradient norm for clipping.
      double gn2 = 0.0;
      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         gn2 += g_pool_logit[n] * g_pool_logit[n];
         for(int m=0; m<FX6_STMN_NODES; m++) gn2 += g_adj[n][m] * g_adj[n][m];
         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            gn2 += g_b_node[n][h] * g_b_node[n][h];
            gn2 += g_w_q[n][h] * g_w_q[n][h];
            gn2 += g_w_k[n][h] * g_w_k[n][h];
            gn2 += g_gate_logit[n][h] * g_gate_logit[n][h];
            gn2 += g_b_sp[n][h] * g_b_sp[n][h];
            for(int i=0; i<FX6_AI_WEIGHTS; i++) gn2 += g_w_node[n][h][i] * g_w_node[n][h][i];
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         gn2 += g_wz_x[h] * g_wz_x[h] + g_wz_h[h] * g_wz_h[h] + g_bz[h] * g_bz[h];
         gn2 += g_wr_x[h] * g_wr_x[h] + g_wr_h[h] * g_wr_h[h] + g_br[h] * g_br[h];
         gn2 += g_wh_x[h] * g_wh_x[h] + g_wh_h[h] * g_wh_h[h] + g_bh[h] * g_bh[h];
         gn2 += g_w_res[h] * g_w_res[h];

         gn2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
            gn2 += g_w_cls[c][h] * g_w_cls[c][h];
      }

      for(int c=0; c<FX6_STMN_CLASS_COUNT; c++) gn2 += g_b_cls[c] * g_b_cls[c];
      gn2 += g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;

      double gn = MathSqrt(gn2 + 1e-12);
      double clip = 12.0;
      double gs = (gn > clip ? clip / gn : 1.0);

      double lr = ScheduledLR(hp);
      double l2 = FX6_Clamp(hp.l2, 0.0, 0.08);

      for(int n=0; n<FX6_STMN_NODES; n++)
      {
         m_pool_logit[n] -= lr * gs * g_pool_logit[n];
         m_pool_logit[n] = FX6_ClipSym(m_pool_logit[n], 6.0);

         for(int m=0; m<FX6_STMN_NODES; m++)
            m_adj[n][m] -= lr * gs * (g_adj[n][m] + 0.15 * l2 * m_adj[n][m]);

         for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
         {
            m_b_node[n][h] -= lr * gs * g_b_node[n][h];
            m_w_q[n][h] -= lr * gs * (g_w_q[n][h] + l2 * m_w_q[n][h]);
            m_w_k[n][h] -= lr * gs * (g_w_k[n][h] + l2 * m_w_k[n][h]);
            m_gate_logit[n][h] -= lr * gs * g_gate_logit[n][h];
            m_b_sp[n][h] -= lr * gs * g_b_sp[n][h];

            m_gate_logit[n][h] = FX6_ClipSym(m_gate_logit[n][h], 6.0);
            m_b_sp[n][h] = FX6_ClipSym(m_b_sp[n][h], 6.0);

            for(int i=0; i<FX6_AI_WEIGHTS; i++)
               m_w_node[n][h][i] -= lr * gs * (g_w_node[n][h][i] + l2 * m_w_node[n][h][i]);
         }
      }

      for(int h=0; h<FX6_AI_MLP_HIDDEN; h++)
      {
         m_wz_x[h] -= lr * gs * (g_wz_x[h] + l2 * m_wz_x[h]);
         m_wz_h[h] -= lr * gs * (g_wz_h[h] + l2 * m_wz_h[h]);
         m_bz[h]   -= lr * gs * g_bz[h];

         m_wr_x[h] -= lr * gs * (g_wr_x[h] + l2 * m_wr_x[h]);
         m_wr_h[h] -= lr * gs * (g_wr_h[h] + l2 * m_wr_h[h]);
         m_br[h]   -= lr * gs * g_br[h];

         m_wh_x[h] -= lr * gs * (g_wh_x[h] + l2 * m_wh_x[h]);
         m_wh_h[h] -= lr * gs * (g_wh_h[h] + l2 * m_wh_h[h]);
         m_bh[h]   -= lr * gs * g_bh[h];

         m_w_res[h] -= lr * gs * (g_w_res[h] + 0.5 * l2 * m_w_res[h]);

         m_w_mu[h]   -= lr * gs * (g_w_mu[h] + l2 * m_w_mu[h]);
         m_w_logv[h] -= lr * gs * (g_w_logv[h] + l2 * m_w_logv[h]);
         m_w_q25[h]  -= lr * gs * (g_w_q25[h] + l2 * m_w_q25[h]);
         m_w_q75[h]  -= lr * gs * (g_w_q75[h] + l2 * m_w_q75[h]);

         for(int c=0; c<FX6_STMN_CLASS_COUNT; c++)
            m_w_cls[c][h] -= lr * gs * (g_w_cls[c][h] + l2 * m_w_cls[c][h]);
      }

      for(int c=0; c<FX6_STMN_CLASS_COUNT; c++) m_b_cls[c] -= lr * gs * g_b_cls[c];
      m_b_mu -= lr * gs * g_b_mu;
      m_b_logv -= lr * gs * g_b_logv;
      m_b_q25 -= lr * gs * g_b_q25;
      m_b_q75 -= lr * gs * g_b_q75;

      m_b_logv = FX6_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;
   }

public:
   CFX6AISTMN(void)
   {
      m_initialized = false;
      InitWeights();
   }

   virtual int AIId(void) const { return (int)AI_TYPE_STMN; }
   virtual string AIName(void) const { return "stmn"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      InitWeights();
      m_initialized = true;
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
      m_initialized = true;
   }

   virtual void Update(const int y, const double &x[], const FX6AIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FX6AIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      FX6AIHyperParams hs = ScaleHyperParamsForMove(hp, move_points);
      double sw = FX6_Clamp(MoveSampleWeight(x, move_points), 0.25, 8.00);

      UpdateInputStats(x);

      int cls = ResolveClass(y, x, move_points);
      double cost = InputCostProxyPoints(x);

      double h_prev[FX6_AI_MLP_HIDDEN];
      GetLastHidden(h_prev);

      PushTrainSample(cls, x, move_points, cost, sw, h_prev);
      if(m_train_len >= 4)
         TrainTBPTT(hs);

      double probs[FX6_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;

      double xn[FX6_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double node_z[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double attn[FX6_STMN_NODES][FX6_STMN_NODES];
      double msg[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double node_o[FX6_STMN_NODES][FX6_AI_MLP_HIDDEN];
      double pool[FX6_STMN_NODES];
      double graph[FX6_AI_MLP_HIDDEN];

      double z_gate[FX6_AI_MLP_HIDDEN];
      double r_gate[FX6_AI_MLP_HIDDEN];
      double h_cand[FX6_AI_MLP_HIDDEN];
      double h_new[FX6_AI_MLP_HIDDEN];
      double logits[FX6_STMN_CLASS_COUNT];

      SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);
      TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);
      HeadForward(h_new, logits, probs, mu, logv, q25, q75);

      PushSequenceState(graph, h_new);

      double den = probs[FX6_STMN_BUY] + probs[FX6_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir = probs[FX6_STMN_BUY] / den;

      double cw = ClassWeight(cls, move_points, cost, sw);
      if(cls == FX6_STMN_BUY)
         UpdateCalibration(p_dir, 1, cw);
      else if(cls == FX6_STMN_SELL)
         UpdateCalibration(p_dir, 0, cw);
      else
         UpdateCalibration(p_dir, (move_points >= 0.0 ? 1 : 0), 0.25 * cw);

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hs, sw);

      m_step++;
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FX6_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double den = probs[FX6_STMN_BUY] + probs[FX6_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FX6_STMN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double p_up = p_dir_cal * FX6_Clamp(1.0 - probs[FX6_STMN_SKIP], 0.0, 1.0);
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FX6_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double spread_amp = MathMax(0.0, q75 - q25);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * spread_amp + 0.20 * sigma);

      double active = FX6_Clamp(1.0 - probs[FX6_STMN_SKIP], 0.0, 1.0);
      double ev = amp * active;

      double base_ev = CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
      if(ev > 0.0 && base_ev > 0.0) return 0.65 * ev + 0.35 * base_ev;
      if(ev > 0.0) return ev;
      return base_ev;
   }
};

#endif // __FX6_AI_STMN_MQH__
