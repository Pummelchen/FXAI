// FXAI v1
#ifndef __FXAI_AI_STMN_MQH__
#define __FXAI_AI_STMN_MQH__

#include "..\plugin_base.mqh"

#define FXAI_STMN_NODES 6
#define FXAI_STMN_CLASS_COUNT 3
#define FXAI_STMN_SEQ 128
#define FXAI_STMN_TBPTT 16

#define FXAI_STMN_SELL 0
#define FXAI_STMN_BUY  1
#define FXAI_STMN_SKIP 2

class CFXAIAISTMN : public CFXAIAIPlugin
{
private:
   bool   m_initialized;
   int    m_step;

   bool   m_x_norm_ready;
   int    m_x_norm_steps;
   double m_x_mean[FXAI_AI_WEIGHTS];
   double m_x_var[FXAI_AI_WEIGHTS];

   int    m_seq_ptr;
   int    m_seq_len;
   double m_hist_h[FXAI_STMN_SEQ][FXAI_AI_MLP_HIDDEN];
   double m_hist_g[FXAI_STMN_SEQ][FXAI_AI_MLP_HIDDEN];

   // Spatio-temporal node masks and parameters.
   double m_group_mask[FXAI_STMN_NODES][FXAI_AI_WEIGHTS];
   double m_w_node[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
   double m_b_node[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];

   double m_w_q[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_w_k[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_adj[FXAI_STMN_NODES][FXAI_STMN_NODES];

   double m_gate_logit[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_b_sp[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_pool_logit[FXAI_STMN_NODES];

   // Temporal memory core (diagonal GRU + residual graph path).
   double m_wz_x[FXAI_AI_MLP_HIDDEN], m_wz_h[FXAI_AI_MLP_HIDDEN], m_bz[FXAI_AI_MLP_HIDDEN];
   double m_wr_x[FXAI_AI_MLP_HIDDEN], m_wr_h[FXAI_AI_MLP_HIDDEN], m_br[FXAI_AI_MLP_HIDDEN];
   double m_wh_x[FXAI_AI_MLP_HIDDEN], m_wh_h[FXAI_AI_MLP_HIDDEN], m_bh[FXAI_AI_MLP_HIDDEN];
   double m_w_res[FXAI_AI_MLP_HIDDEN];

   // 3-class head + move distribution head.
   double m_w_cls[FXAI_STMN_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
   double m_b_cls[FXAI_STMN_CLASS_COUNT];

   double m_w_mu[FXAI_AI_MLP_HIDDEN],   m_b_mu;
   double m_w_logv[FXAI_AI_MLP_HIDDEN], m_b_logv;
   double m_w_q25[FXAI_AI_MLP_HIDDEN],  m_b_q25;
   double m_w_q75[FXAI_AI_MLP_HIDDEN],  m_b_q75;

   // Training sequence buffer.
   int    m_train_len;
   double m_train_x[FXAI_STMN_TBPTT][FXAI_AI_WEIGHTS];
   int    m_train_cls[FXAI_STMN_TBPTT];
   double m_train_move[FXAI_STMN_TBPTT];
   double m_train_cost[FXAI_STMN_TBPTT];
   double m_train_w[FXAI_STMN_TBPTT];
   double m_train_hprev[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];

   // Forward caches for TBPTT.
   double m_cache_xn[FXAI_STMN_TBPTT][FXAI_AI_WEIGHTS];
   double m_cache_node_z[FXAI_STMN_TBPTT][FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_cache_attn[FXAI_STMN_TBPTT][FXAI_STMN_NODES][FXAI_STMN_NODES];
   double m_cache_msg[FXAI_STMN_TBPTT][FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_cache_node_o[FXAI_STMN_TBPTT][FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
   double m_cache_pool[FXAI_STMN_TBPTT][FXAI_STMN_NODES];
   double m_cache_graph[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];

   double m_cache_hprev[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_zgate[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_rgate[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_hcand[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];
   double m_cache_hnew[FXAI_STMN_TBPTT][FXAI_AI_MLP_HIDDEN];

   double m_cache_logits[FXAI_STMN_TBPTT][FXAI_STMN_CLASS_COUNT];
   double m_cache_probs[FXAI_STMN_TBPTT][FXAI_STMN_CLASS_COUNT];
   double m_cache_mu[FXAI_STMN_TBPTT];
   double m_cache_logv[FXAI_STMN_TBPTT];
   double m_cache_q25[FXAI_STMN_TBPTT];
   double m_cache_q75[FXAI_STMN_TBPTT];

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

      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -30.0, 30.0));
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
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_x_mean[i] = 0.0;
         m_x_var[i] = 1.0;
      }
   }

   void UpdateInputStats(const double &x[])
   {
      double a = (m_x_norm_steps < 128 ? 0.05 : 0.015);
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
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
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
      {
         if(!m_x_norm_ready)
         {
            xn[i] = FXAI_ClipSym(x[i], 8.0);
            continue;
         }

         double inv = 1.0 / MathSqrt(m_x_var[i] + 1e-6);
         xn[i] = FXAI_ClipSym((x[i] - m_x_mean[i]) * inv, 8.0);
      }
   }

   void ResetSequence(void)
   {
      m_seq_ptr = -1;
      m_seq_len = 0;
      for(int t=0; t<FXAI_STMN_SEQ; t++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_hist_h[t][h] = 0.0;
            m_hist_g[t][h] = 0.0;
         }
      }
   }

   void ResetTrainBuffer(void)
   {
      m_train_len = 0;
      for(int t=0; t<FXAI_STMN_TBPTT; t++)
      {
         m_train_cls[t] = FXAI_STMN_SKIP;
         m_train_move[t] = 0.0;
         m_train_cost[t] = 0.0;
         m_train_w[t] = 1.0;
         m_cache_mu[t] = 0.0;
         m_cache_logv[t] = 0.0;
         m_cache_q25[t] = 0.0;
         m_cache_q75[t] = 0.0;

         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_train_x[t][i] = 0.0;
            m_cache_xn[t][i] = 0.0;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_train_hprev[t][h] = 0.0;
            m_cache_graph[t][h] = 0.0;
            m_cache_hprev[t][h] = 0.0;
            m_cache_zgate[t][h] = 0.0;
            m_cache_rgate[t][h] = 0.0;
            m_cache_hcand[t][h] = 0.0;
            m_cache_hnew[t][h] = 0.0;

            for(int n=0; n<FXAI_STMN_NODES; n++)
            {
               m_cache_node_z[t][n][h] = 0.0;
               m_cache_msg[t][n][h] = 0.0;
               m_cache_node_o[t][n][h] = 0.0;
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            m_cache_pool[t][n] = 0.0;
            for(int m=0; m<FXAI_STMN_NODES; m++)
               m_cache_attn[t][n][m] = 0.0;
         }

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
         {
            m_cache_logits[t][c] = 0.0;
            m_cache_probs[t][c] = 1.0 / 3.0;
         }
      }
   }

   void GetLastHidden(double &h_prev[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) h_prev[h] = 0.0;
      if(m_seq_len <= 0 || m_seq_ptr < 0) return;

      int idx = m_seq_ptr;
      if(idx < 0 || idx >= FXAI_STMN_SEQ) return;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         h_prev[h] = m_hist_h[idx][h];
   }

   void PushSequenceState(const double &graph[], const double &h_new[])
   {
      m_seq_ptr++;
      if(m_seq_ptr >= FXAI_STMN_SEQ) m_seq_ptr = 0;
      if(m_seq_len < FXAI_STMN_SEQ) m_seq_len++;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         m_hist_g[m_seq_ptr][h] = graph[h];
         m_hist_h[m_seq_ptr][h] = h_new[h];
      }
   }

   void BuildGroupMasks(void)
   {
      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_group_mask[n][i] = 0.0;
      }

      // Bias for every node.
      for(int n=0; n<FXAI_STMN_NODES; n++)
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

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         m_pool_logit[n] = (n == 0 ? 0.25 : 0.0);

         for(int m=0; m<FXAI_STMN_NODES; m++)
         {
            if(n == m) m_adj[n][m] = 0.20;
            else m_adj[n][m] = -0.05;
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double sh = (double)((n + 1) * (h + 2));
            m_b_node[n][h] = 0.0;
            m_w_q[n][h] = 0.06 * MathSin(0.61 * sh);
            m_w_k[n][h] = 0.06 * MathCos(0.67 * sh);
            m_gate_logit[n][h] = 0.0;
            m_b_sp[n][h] = 0.0;

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
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

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
            m_w_cls[c][h] = 0.05 * MathSin((double)((c + 2) * (h + 1)) * 0.69);
      }

      for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++) m_b_cls[c] = 0.0;
      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = 0.0;
      m_b_q75 = 0.0;
   }

   int ResolveClass(const int y,
                    const double &x[],
                    const double move_points) const
   {
      if(y == FXAI_STMN_SELL || y == FXAI_STMN_BUY || y == FXAI_STMN_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return FXAI_STMN_SKIP;

      if(y > 0) return FXAI_STMN_BUY;
      if(y == 0) return FXAI_STMN_SELL;
      return (move_points >= 0.0 ? FXAI_STMN_BUY : FXAI_STMN_SELL);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost,
                      const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double base = FXAI_Clamp(sample_w, 0.25, 4.00);

      if(cls == FXAI_STMN_SKIP)
      {
         if(edge <= 0.0) return FXAI_Clamp(base * 1.6, 0.25, 6.0);
         return FXAI_Clamp(base * 0.7, 0.25, 6.0);
      }

      if(edge <= 0.0) return FXAI_Clamp(base * 0.55, 0.25, 6.0);
      return FXAI_Clamp(base * (1.0 + 0.06 * MathMin(edge, 20.0)), 0.25, 6.0);
   }

   double MoveWeight(const double move_points,
                     const double cost,
                     const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double denom = MathMax(cost, 1.0);
      double ew = FXAI_Clamp(0.5 + edge / denom, 0.25, 4.0);
      return FXAI_Clamp(sample_w * ew, 0.25, 8.0);
   }

   double ScheduledLR(const FXAIAIHyperParams &hp) const
   {
      double base = FXAI_Clamp(hp.lr, 0.0002, 0.1200);
      double warm = 1.0;
      if(m_step < 200)
         warm = 0.20 + 0.80 * ((double)m_step / 200.0);
      double decay = 1.0 / MathSqrt(1.0 + 0.002 * MathMax(0, m_step - 200));
      double cyc = 0.95 + 0.05 * MathSin((double)m_step * 0.031);
      return FXAI_Clamp(base * warm * decay * cyc, 0.00005, 0.0600);
   }

   void SpatialForward(const double &xn[],
                       double &node_z[][FXAI_AI_MLP_HIDDEN],
                       double &attn[][FXAI_STMN_NODES],
                       double &msg[][FXAI_AI_MLP_HIDDEN],
                       double &node_o[][FXAI_AI_MLP_HIDDEN],
                       double &pool_w[],
                       double &graph[]) const
   {
      double q_node[FXAI_STMN_NODES];
      double k_node[FXAI_STMN_NODES];

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         q_node[n] = 0.0;
         k_node[n] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double z = m_b_node[n][h];
            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
               z += (m_group_mask[n][i] * m_w_node[n][h][i]) * xn[i];

            node_z[n][h] = FXAI_Tanh(FXAI_ClipSym(z, 12.0));
            q_node[n] += m_w_q[n][h] * node_z[n][h];
            k_node[n] += m_w_k[n][h] * node_z[n][h];
         }
      }

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         double max_s = -1e100;
         for(int m=0; m<FXAI_STMN_NODES; m++)
         {
            double s = m_adj[n][m] + q_node[n] + k_node[m];
            attn[n][m] = s;
            if(s > max_s) max_s = s;
         }

         double den = 0.0;
         for(int m=0; m<FXAI_STMN_NODES; m++)
         {
            double e = MathExp(FXAI_Clamp(attn[n][m] - max_s, -30.0, 30.0));
            attn[n][m] = e;
            den += e;
         }
         if(den <= 0.0) den = 1.0;
         for(int m=0; m<FXAI_STMN_NODES; m++) attn[n][m] /= den;
      }

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            double s = 0.0;
            for(int m=0; m<FXAI_STMN_NODES; m++)
               s += attn[n][m] * node_z[m][h];
            msg[n][h] = s;

            double gate = FXAI_Sigmoid(m_gate_logit[n][h]);
            double pre = gate * node_z[n][h] + (1.0 - gate) * msg[n][h] + m_b_sp[n][h];
            node_o[n][h] = FXAI_Tanh(FXAI_ClipSym(pre, 10.0));
         }
      }

      double maxp = m_pool_logit[0];
      for(int n=1; n<FXAI_STMN_NODES; n++)
         if(m_pool_logit[n] > maxp) maxp = m_pool_logit[n];

      double denp = 0.0;
      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         pool_w[n] = MathExp(FXAI_Clamp(m_pool_logit[n] - maxp, -30.0, 30.0));
         denp += pool_w[n];
      }
      if(denp <= 0.0) denp = 1.0;
      for(int n=0; n<FXAI_STMN_NODES; n++) pool_w[n] /= denp;

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double g = 0.0;
         for(int n=0; n<FXAI_STMN_NODES; n++)
            g += pool_w[n] * node_o[n][h];
         graph[h] = FXAI_ClipSym(g, 8.0);
      }
   }

   void TemporalForward(const double &graph[],
                        const double &h_prev[],
                        double &z_gate[],
                        double &r_gate[],
                        double &h_cand[],
                        double &h_new[]) const
   {
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         double zpre = m_wz_x[h] * graph[h] + m_wz_h[h] * h_prev[h] + m_bz[h];
         double rpre = m_wr_x[h] * graph[h] + m_wr_h[h] * h_prev[h] + m_br[h];

         z_gate[h] = FXAI_Sigmoid(FXAI_ClipSym(zpre, 15.0));
         r_gate[h] = FXAI_Sigmoid(FXAI_ClipSym(rpre, 15.0));

         double cpre = m_wh_x[h] * graph[h] + m_wh_h[h] * (r_gate[h] * h_prev[h]) + m_bh[h];
         h_cand[h] = FXAI_Tanh(FXAI_ClipSym(cpre, 12.0));

         double hmix = (1.0 - z_gate[h]) * h_prev[h] + z_gate[h] * h_cand[h] + m_w_res[h] * graph[h];
         h_new[h] = FXAI_ClipSym(hmix, 8.0);
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
      for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
      {
         double s = m_b_cls[c];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            s += m_w_cls[c][h] * h_state[h];
         logits[c] = FXAI_ClipSym(s, 20.0);
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         mu += m_w_mu[h] * h_state[h];
         logv += m_w_logv[h] * h_state[h];
         q25 += m_w_q25[h] * h_state[h];
         q75 += m_w_q75[h] * h_state[h];
      }

      logv = FXAI_Clamp(logv, -4.0, 4.0);
      if(q75 < q25 + 1e-4) q75 = q25 + 1e-4;
   }

   void ForwardInference(const double &x[],
                         double &probs[],
                         double &mu,
                         double &logv,
                         double &q25,
                         double &q75) const
   {
      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double node_z[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double attn[FXAI_STMN_NODES][FXAI_STMN_NODES];
      double msg[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double node_o[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double pool[FXAI_STMN_NODES];
      double graph[FXAI_AI_MLP_HIDDEN];

      SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);

      double h_prev[FXAI_AI_MLP_HIDDEN];
      double z_gate[FXAI_AI_MLP_HIDDEN];
      double r_gate[FXAI_AI_MLP_HIDDEN];
      double h_cand[FXAI_AI_MLP_HIDDEN];
      double h_new[FXAI_AI_MLP_HIDDEN];
      double logits[FXAI_STMN_CLASS_COUNT];

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
      if(m_train_len < FXAI_STMN_TBPTT)
      {
         int t = m_train_len;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[t][i] = x[i];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) m_train_hprev[t][h] = h_prev[h];
         m_train_cls[t] = cls;
         m_train_move[t] = move_points;
         m_train_cost[t] = cost;
         m_train_w[t] = sample_w;
         m_train_len++;
         return;
      }

      for(int t=1; t<FXAI_STMN_TBPTT; t++)
      {
         int p = t - 1;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[p][i] = m_train_x[t][i];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) m_train_hprev[p][h] = m_train_hprev[t][h];
         m_train_cls[p] = m_train_cls[t];
         m_train_move[p] = m_train_move[t];
         m_train_cost[p] = m_train_cost[t];
         m_train_w[p] = m_train_w[t];
      }

      int last = FXAI_STMN_TBPTT - 1;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_train_x[last][i] = x[i];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) m_train_hprev[last][h] = h_prev[h];
      m_train_cls[last] = cls;
      m_train_move[last] = move_points;
      m_train_cost[last] = cost;
      m_train_w[last] = sample_w;
      m_train_len = FXAI_STMN_TBPTT;
   }

   void TrainTBPTT(const FXAIAIHyperParams &hp)
   {
      int T = m_train_len;
      if(T <= 0) return;

      // Forward pass over buffered sequence.
      for(int t=0; t<T; t++)
      {
         double xraw[FXAI_AI_WEIGHTS];
         double xn[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) xraw[i] = m_train_x[t][i];
         NormalizeInput(xraw, xn);
         for(int i=0; i<FXAI_AI_WEIGHTS; i++) m_cache_xn[t][i] = xn[i];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            m_cache_hprev[t][h] = (t == 0 ? m_train_hprev[0][h] : m_cache_hnew[t - 1][h]);

         double node_z[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double attn[FXAI_STMN_NODES][FXAI_STMN_NODES];
         double msg[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double node_o[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double pool[FXAI_STMN_NODES];
         double graph[FXAI_AI_MLP_HIDDEN];

         SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            m_cache_pool[t][n] = pool[n];
            for(int m=0; m<FXAI_STMN_NODES; m++)
               m_cache_attn[t][n][m] = attn[n][m];
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               m_cache_node_z[t][n][h] = node_z[n][h];
               m_cache_msg[t][n][h] = msg[n][h];
               m_cache_node_o[t][n][h] = node_o[n][h];
            }
         }

         double h_prev[FXAI_AI_MLP_HIDDEN];
         double z_gate[FXAI_AI_MLP_HIDDEN];
         double r_gate[FXAI_AI_MLP_HIDDEN];
         double h_cand[FXAI_AI_MLP_HIDDEN];
         double h_new[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            h_prev[h] = m_cache_hprev[t][h];
            m_cache_graph[t][h] = graph[h];
         }

         TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_cache_zgate[t][h] = z_gate[h];
            m_cache_rgate[t][h] = r_gate[h];
            m_cache_hcand[t][h] = h_cand[h];
            m_cache_hnew[t][h] = h_new[h];
         }

         double logits[FXAI_STMN_CLASS_COUNT];
         double probs[FXAI_STMN_CLASS_COUNT];
         double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
         HeadForward(h_new, logits, probs, mu, logv, q25, q75);

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
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
      double g_w_node[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN][FXAI_AI_WEIGHTS];
      double g_b_node[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double g_w_q[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double g_w_k[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double g_adj[FXAI_STMN_NODES][FXAI_STMN_NODES];
      double g_gate_logit[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double g_b_sp[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double g_pool_logit[FXAI_STMN_NODES];

      double g_wz_x[FXAI_AI_MLP_HIDDEN], g_wz_h[FXAI_AI_MLP_HIDDEN], g_bz[FXAI_AI_MLP_HIDDEN];
      double g_wr_x[FXAI_AI_MLP_HIDDEN], g_wr_h[FXAI_AI_MLP_HIDDEN], g_br[FXAI_AI_MLP_HIDDEN];
      double g_wh_x[FXAI_AI_MLP_HIDDEN], g_wh_h[FXAI_AI_MLP_HIDDEN], g_bh[FXAI_AI_MLP_HIDDEN];
      double g_w_res[FXAI_AI_MLP_HIDDEN];

      double g_w_cls[FXAI_STMN_CLASS_COUNT][FXAI_AI_MLP_HIDDEN];
      double g_b_cls[FXAI_STMN_CLASS_COUNT];

      double g_w_mu[FXAI_AI_MLP_HIDDEN], g_w_logv[FXAI_AI_MLP_HIDDEN], g_w_q25[FXAI_AI_MLP_HIDDEN], g_w_q75[FXAI_AI_MLP_HIDDEN];
      double g_b_mu, g_b_logv, g_b_q25, g_b_q75;

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         g_pool_logit[n] = 0.0;
         for(int m=0; m<FXAI_STMN_NODES; m++) g_adj[n][m] = 0.0;
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_b_node[n][h] = 0.0;
            g_w_q[n][h] = 0.0;
            g_w_k[n][h] = 0.0;
            g_gate_logit[n][h] = 0.0;
            g_b_sp[n][h] = 0.0;
            for(int i=0; i<FXAI_AI_WEIGHTS; i++) g_w_node[n][h][i] = 0.0;
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         g_wz_x[h] = 0.0; g_wz_h[h] = 0.0; g_bz[h] = 0.0;
         g_wr_x[h] = 0.0; g_wr_h[h] = 0.0; g_br[h] = 0.0;
         g_wh_x[h] = 0.0; g_wh_h[h] = 0.0; g_bh[h] = 0.0;
         g_w_res[h] = 0.0;

         g_w_mu[h] = 0.0;
         g_w_logv[h] = 0.0;
         g_w_q25[h] = 0.0;
         g_w_q75[h] = 0.0;

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++) g_w_cls[c][h] = 0.0;
      }
      for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++) g_b_cls[c] = 0.0;
      g_b_mu = 0.0;
      g_b_logv = 0.0;
      g_b_q25 = 0.0;
      g_b_q75 = 0.0;

      double dh_next[FXAI_AI_MLP_HIDDEN];
      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) dh_next[h] = 0.0;

      for(int t=T-1; t>=0; t--)
      {
         int cls = m_train_cls[t];
         if(cls < FXAI_STMN_SELL || cls > FXAI_STMN_SKIP) cls = FXAI_STMN_SKIP;

         double move = m_train_move[t];
         double cost = m_train_cost[t];
         double sw = m_train_w[t];

         double cw = ClassWeight(cls, move, cost, sw);
         double mw = MoveWeight(move, cost, sw);

         double y_true[FXAI_STMN_CLASS_COUNT];
         y_true[0] = (cls == FXAI_STMN_SELL ? 1.0 : 0.0);
         y_true[1] = (cls == FXAI_STMN_BUY ? 1.0 : 0.0);
         y_true[2] = (cls == FXAI_STMN_SKIP ? 1.0 : 0.0);

         double dh[FXAI_AI_MLP_HIDDEN];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++) dh[h] = dh_next[h];

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
         {
            double dlog = (m_cache_probs[t][c] - y_true[c]) * cw;
            g_b_cls[c] += dlog;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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
         dlogv = FXAI_ClipSym(dlogv, 10.0);

         double dq25 = PinballGrad(abs_move, q25, 0.25) * mw;
         double dq75 = PinballGrad(abs_move, q75, 0.75) * mw;
         if(q25 > q75)
         {
            double pen = FXAI_ClipSym((q25 - q75), 5.0) * 0.25;
            dq25 += pen;
            dq75 -= pen;
         }

         g_b_mu += dmu;
         g_b_logv += dlogv;
         g_b_q25 += dq25;
         g_b_q75 += dq75;

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            g_w_mu[h] += dmu * m_cache_hnew[t][h];
            g_w_logv[h] += dlogv * m_cache_hnew[t][h];
            g_w_q25[h] += dq25 * m_cache_hnew[t][h];
            g_w_q75[h] += dq75 * m_cache_hnew[t][h];

            dh[h] += dmu * m_w_mu[h] + dlogv * m_w_logv[h] + dq25 * m_w_q25[h] + dq75 * m_w_q75[h];
            dh[h] = FXAI_ClipSym(dh[h], 20.0);
         }

         double dgraph[FXAI_AI_MLP_HIDDEN];
         double dh_prev[FXAI_AI_MLP_HIDDEN];

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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

            dgraph[h] = FXAI_ClipSym(dg, 20.0);
            dh_prev[h] = FXAI_ClipSym(dhp, 20.0);
         }

         // Spatial backprop at step t.
         double dnode_o[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double dnode_z[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double dmsg[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
         double dattn[FXAI_STMN_NODES][FXAI_STMN_NODES];
         double dscore[FXAI_STMN_NODES][FXAI_STMN_NODES];
         double dqn[FXAI_STMN_NODES];
         double dkm[FXAI_STMN_NODES];

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            dqn[n] = 0.0;
            dkm[n] = 0.0;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               dnode_o[n][h] = 0.0;
               dnode_z[n][h] = 0.0;
               dmsg[n][h] = 0.0;
            }
            for(int m=0; m<FXAI_STMN_NODES; m++)
            {
               dattn[n][m] = 0.0;
               dscore[n][m] = 0.0;
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            double dot_pool = 0.0;
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               dnode_o[n][h] += dgraph[h] * m_cache_pool[t][n];
               dot_pool += dgraph[h] * (m_cache_node_o[t][n][h] - m_cache_graph[t][h]);
            }
            g_pool_logit[n] += m_cache_pool[t][n] * dot_pool;
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double o = m_cache_node_o[t][n][h];
               double da = dnode_o[n][h] * (1.0 - o * o);
               g_b_sp[n][h] += da;

               double gate = FXAI_Sigmoid(m_gate_logit[n][h]);
               double nz = m_cache_node_z[t][n][h];
               double msgv = m_cache_msg[t][n][h];
               g_gate_logit[n][h] += da * (nz - msgv) * gate * (1.0 - gate);

               dnode_z[n][h] += da * gate;
               dmsg[n][h] += da * (1.0 - gate);
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            for(int m=0; m<FXAI_STMN_NODES; m++)
            {
               double ds = 0.0;
               for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
               {
                  ds += dmsg[n][h] * m_cache_node_z[t][m][h];
                  dnode_z[m][h] += dmsg[n][h] * m_cache_attn[t][n][m];
               }
               dattn[n][m] += ds;
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            double dot = 0.0;
            for(int m=0; m<FXAI_STMN_NODES; m++)
               dot += dattn[n][m] * m_cache_attn[t][n][m];

            for(int m=0; m<FXAI_STMN_NODES; m++)
               dscore[n][m] = m_cache_attn[t][n][m] * (dattn[n][m] - dot);
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            for(int m=0; m<FXAI_STMN_NODES; m++)
            {
               double ds = dscore[n][m];
               g_adj[n][m] += ds;
               dqn[n] += ds;
               dkm[m] += ds;
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               g_w_q[n][h] += dqn[n] * m_cache_node_z[t][n][h];
               dnode_z[n][h] += dqn[n] * m_w_q[n][h];

               g_w_k[n][h] += dkm[n] * m_cache_node_z[t][n][h];
               dnode_z[n][h] += dkm[n] * m_w_k[n][h];
            }
         }

         for(int n=0; n<FXAI_STMN_NODES; n++)
         {
            for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            {
               double z = m_cache_node_z[t][n][h];
               double da = dnode_z[n][h] * (1.0 - z * z);
               g_b_node[n][h] += da;
               for(int i=0; i<FXAI_AI_WEIGHTS; i++)
               {
                  g_w_node[n][h][i] += da * (m_group_mask[n][i] * m_cache_xn[t][i]);
               }
            }
         }

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
            dh_next[h] = dh_prev[h];
      }

      // Global gradient norm for clipping.
      double gn2 = 0.0;
      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         gn2 += g_pool_logit[n] * g_pool_logit[n];
         for(int m=0; m<FXAI_STMN_NODES; m++) gn2 += g_adj[n][m] * g_adj[n][m];
         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            gn2 += g_b_node[n][h] * g_b_node[n][h];
            gn2 += g_w_q[n][h] * g_w_q[n][h];
            gn2 += g_w_k[n][h] * g_w_k[n][h];
            gn2 += g_gate_logit[n][h] * g_gate_logit[n][h];
            gn2 += g_b_sp[n][h] * g_b_sp[n][h];
            for(int i=0; i<FXAI_AI_WEIGHTS; i++) gn2 += g_w_node[n][h][i] * g_w_node[n][h][i];
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
      {
         gn2 += g_wz_x[h] * g_wz_x[h] + g_wz_h[h] * g_wz_h[h] + g_bz[h] * g_bz[h];
         gn2 += g_wr_x[h] * g_wr_x[h] + g_wr_h[h] * g_wr_h[h] + g_br[h] * g_br[h];
         gn2 += g_wh_x[h] * g_wh_x[h] + g_wh_h[h] * g_wh_h[h] + g_bh[h] * g_bh[h];
         gn2 += g_w_res[h] * g_w_res[h];

         gn2 += g_w_mu[h] * g_w_mu[h] + g_w_logv[h] * g_w_logv[h] + g_w_q25[h] * g_w_q25[h] + g_w_q75[h] * g_w_q75[h];

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
            gn2 += g_w_cls[c][h] * g_w_cls[c][h];
      }

      for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++) gn2 += g_b_cls[c] * g_b_cls[c];
      gn2 += g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;

      double gn = MathSqrt(gn2 + 1e-12);
      double clip = 12.0;
      double gs = (gn > clip ? clip / gn : 1.0);

      double lr = ScheduledLR(hp);
      double l2 = FXAI_Clamp(hp.l2, 0.0, 0.08);

      for(int n=0; n<FXAI_STMN_NODES; n++)
      {
         m_pool_logit[n] -= lr * gs * g_pool_logit[n];
         m_pool_logit[n] = FXAI_ClipSym(m_pool_logit[n], 6.0);

         for(int m=0; m<FXAI_STMN_NODES; m++)
            m_adj[n][m] -= lr * gs * (g_adj[n][m] + 0.15 * l2 * m_adj[n][m]);

         for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
         {
            m_b_node[n][h] -= lr * gs * g_b_node[n][h];
            m_w_q[n][h] -= lr * gs * (g_w_q[n][h] + l2 * m_w_q[n][h]);
            m_w_k[n][h] -= lr * gs * (g_w_k[n][h] + l2 * m_w_k[n][h]);
            m_gate_logit[n][h] -= lr * gs * g_gate_logit[n][h];
            m_b_sp[n][h] -= lr * gs * g_b_sp[n][h];

            m_gate_logit[n][h] = FXAI_ClipSym(m_gate_logit[n][h], 6.0);
            m_b_sp[n][h] = FXAI_ClipSym(m_b_sp[n][h], 6.0);

            for(int i=0; i<FXAI_AI_WEIGHTS; i++)
               m_w_node[n][h][i] -= lr * gs * (g_w_node[n][h][i] + l2 * m_w_node[n][h][i]);
         }
      }

      for(int h=0; h<FXAI_AI_MLP_HIDDEN; h++)
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

         for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++)
            m_w_cls[c][h] -= lr * gs * (g_w_cls[c][h] + l2 * m_w_cls[c][h]);
      }

      for(int c=0; c<FXAI_STMN_CLASS_COUNT; c++) m_b_cls[c] -= lr * gs * g_b_cls[c];
      m_b_mu -= lr * gs * g_b_mu;
      m_b_logv -= lr * gs * g_b_logv;
      m_b_q25 -= lr * gs * g_b_q25;
      m_b_q75 -= lr * gs * g_b_q75;

      m_b_logv = FXAI_Clamp(m_b_logv, -4.0, 4.0);
      if(m_b_q75 < m_b_q25 + 1e-4) m_b_q75 = m_b_q25 + 1e-4;
   }

public:
   CFXAIAISTMN(void)
   {
      m_initialized = false;
      InitWeights();
   }

   virtual int AIId(void) const { return (int)AI_STMN; }
   virtual string AIName(void) const { return "stmn"; }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      InitWeights();
      m_initialized = true;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
      m_initialized = true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      UpdateWithMove(y, x, hp, pseudo_move);
   }

   virtual void UpdateWithMove(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      FXAIAIHyperParams hs = ScaleHyperParamsForMove(hp, move_points);
      double sw = FXAI_Clamp(MoveSampleWeight(x, move_points), 0.25, 8.00);

      UpdateInputStats(x);

      int cls = ResolveClass(y, x, move_points);
      double cost = InputCostProxyPoints(x);

      double h_prev[FXAI_AI_MLP_HIDDEN];
      GetLastHidden(h_prev);

      PushTrainSample(cls, x, move_points, cost, sw, h_prev);
      if(m_train_len >= 4)
         TrainTBPTT(hs);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;

      double xn[FXAI_AI_WEIGHTS];
      NormalizeInput(x, xn);

      double node_z[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double attn[FXAI_STMN_NODES][FXAI_STMN_NODES];
      double msg[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double node_o[FXAI_STMN_NODES][FXAI_AI_MLP_HIDDEN];
      double pool[FXAI_STMN_NODES];
      double graph[FXAI_AI_MLP_HIDDEN];

      double z_gate[FXAI_AI_MLP_HIDDEN];
      double r_gate[FXAI_AI_MLP_HIDDEN];
      double h_cand[FXAI_AI_MLP_HIDDEN];
      double h_new[FXAI_AI_MLP_HIDDEN];
      double logits[FXAI_STMN_CLASS_COUNT];

      SpatialForward(xn, node_z, attn, msg, node_o, pool, graph);
      TemporalForward(graph, h_prev, z_gate, r_gate, h_cand, h_new);
      HeadForward(h_new, logits, probs, mu, logv, q25, q75);

      PushSequenceState(graph, h_new);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir = probs[FXAI_STMN_BUY] / den;

      double cw = ClassWeight(cls, move_points, cost, sw);
      if(cls == FXAI_STMN_BUY)
         UpdateCalibration(p_dir, 1, cw);
      else if(cls == FXAI_STMN_SELL)
         UpdateCalibration(p_dir, 0, cw);
      else
         UpdateCalibration(p_dir, (move_points >= 0.0 ? 1 : 0), 0.25 * cw);

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hs, sw);

      m_step++;
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double den = probs[FXAI_STMN_BUY] + probs[FXAI_STMN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_STMN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double p_up = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double probs[FXAI_STMN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardInference(x, probs, mu, logv, q25, q75);

      double spread_amp = MathMax(0.0, q75 - q25);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * spread_amp + 0.20 * sigma);

      double active = FXAI_Clamp(1.0 - probs[FXAI_STMN_SKIP], 0.0, 1.0);
      double ev = amp * active;

      double base_ev = CFXAIAIPlugin::PredictExpectedMovePoints(x, hp);
      if(ev > 0.0 && base_ev > 0.0) return 0.65 * ev + 0.35 * base_ev;
      if(ev > 0.0) return ev;
      return base_ev;
   }
};

#endif // __FXAI_AI_STMN_MQH__
