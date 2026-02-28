#ifndef __FX6_AI_TCN_MQH__
#define __FX6_AI_TCN_MQH__

#include "..\plugin_base.mqh"

#define FX6_TCN_MAX_LAYERS 8
#define FX6_TCN_MAX_KERNEL 5
#define FX6_TCN_HIST 256
#define FX6_TCN_TBPTT 16
#define FX6_TCN_CLASS_COUNT 3

#define FX6_TCN_SELL 0
#define FX6_TCN_BUY  1
#define FX6_TCN_SKIP 2

class CFX6AITCN : public CFX6AIPlugin
{
private:
   bool   m_initialized;
   bool   m_cfg_ready;
   int    m_step;
   int    m_adam_t;

   int    m_ptr;
   int    m_hist_len;

   int    m_layers;
   int    m_kernel;
   int    m_dbase;

   double m_drop_rate;

   // Runtime optimizer config.
   double m_lr;
   double m_l2;
   double m_wd;
   double m_beta1;
   double m_beta2;
   double m_rms_decay;

   // Input projection.
   double m_w_in[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_b_in[FX6_AI_MLP_HIDDEN];

   // Residual TCN block parameters.
   // Conv1/Conv2 are full channel-mix causal convolutions.
   double m_w_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
   double m_b_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_w_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
   double m_b_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   // 1x1 residual projection + gating.
   double m_w_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_b_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_w_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_b_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   // Learnable normalization affine params.
   double m_g1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_bn1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_g2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_bn2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   // Output heads: 3-class + distributional move.
   double m_w_cls[FX6_TCN_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_b_cls[FX6_TCN_CLASS_COUNT];

   double m_w_mu[FX6_AI_MLP_HIDDEN];
   double m_b_mu;
   double m_w_logv[FX6_AI_MLP_HIDDEN];
   double m_b_logv;
   double m_w_q25[FX6_AI_MLP_HIDDEN];
   double m_b_q25;
   double m_w_q75[FX6_AI_MLP_HIDDEN];
   double m_b_q75;

   // History streams.
   // stream[0] = projected input, stream[l+1] = output of block l.
   double m_hist_stream[FX6_TCN_MAX_LAYERS + 1][FX6_TCN_HIST][FX6_AI_MLP_HIDDEN];
   // Mid activations after conv1/tanh/dropout for conv2 temporal taps.
   double m_hist_mid[FX6_TCN_MAX_LAYERS][FX6_TCN_HIST][FX6_AI_MLP_HIDDEN];

   // TBPTT sequence buffers.
   int    m_seq_len;
   double m_seq_x[FX6_TCN_TBPTT][FX6_AI_WEIGHTS];
   int    m_seq_cls[FX6_TCN_TBPTT];
   double m_seq_move[FX6_TCN_TBPTT];
   double m_seq_cost[FX6_TCN_TBPTT];
   double m_seq_w[FX6_TCN_TBPTT];

   // Forward caches per sequence step.
   double m_seq_in[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_n1[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_a1[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_drop[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_n2[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_pre[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_gate[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_seq_out[FX6_TCN_TBPTT][FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_seq_final[FX6_TCN_TBPTT][FX6_AI_MLP_HIDDEN];
   double m_seq_logits[FX6_TCN_TBPTT][FX6_TCN_CLASS_COUNT];
   double m_seq_probs[FX6_TCN_TBPTT][FX6_TCN_CLASS_COUNT];
   double m_seq_mu[FX6_TCN_TBPTT];
   double m_seq_logv[FX6_TCN_TBPTT];
   double m_seq_q25[FX6_TCN_TBPTT];
   double m_seq_q75[FX6_TCN_TBPTT];

   // RMSProp states (conv kernels/biases).
   double m_r_w_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
   double m_r_b_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_r_w_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
   double m_r_b_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   // AdamW states (all other params).
   double m_m_w_in[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS], m_v_w_in[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
   double m_m_b_in[FX6_AI_MLP_HIDDEN], m_v_b_in[FX6_AI_MLP_HIDDEN];

   double m_m_w_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_v_w_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_b_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_b_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_m_w_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_v_w_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
   double m_m_b_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_b_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_m_g1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_g1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_m_bn1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_bn1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_m_g2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_g2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
   double m_m_bn2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN], m_v_bn2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

   double m_m_w_cls[FX6_TCN_CLASS_COUNT][FX6_AI_MLP_HIDDEN], m_v_w_cls[FX6_TCN_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
   double m_m_b_cls[FX6_TCN_CLASS_COUNT], m_v_b_cls[FX6_TCN_CLASS_COUNT];

   double m_m_w_mu[FX6_AI_MLP_HIDDEN], m_v_w_mu[FX6_AI_MLP_HIDDEN];
   double m_m_w_logv[FX6_AI_MLP_HIDDEN], m_v_w_logv[FX6_AI_MLP_HIDDEN];
   double m_m_w_q25[FX6_AI_MLP_HIDDEN], m_v_w_q25[FX6_AI_MLP_HIDDEN];
   double m_m_w_q75[FX6_AI_MLP_HIDDEN], m_v_w_q75[FX6_AI_MLP_HIDDEN];

   double m_m_b_mu, m_v_b_mu;
   double m_m_b_logv, m_v_b_logv;
   double m_m_b_q25, m_v_b_q25;
   double m_m_b_q75, m_v_b_q75;

   int HistIdx(const int base, const int back) const
   {
      int p = base - back;
      while(p < 0) p += FX6_TCN_HIST;
      while(p >= FX6_TCN_HIST) p -= FX6_TCN_HIST;
      return p;
   }

   int ClampI(const int v, const int lo, const int hi) const
   {
      if(v < lo) return lo;
      if(v > hi) return hi;
      return v;
   }

   int CfgI(const double v, const int defv, const int lo, const int hi) const
   {
      if(!MathIsValidNumber(v)) return defv;
      return ClampI((int)MathRound(v), lo, hi);
   }

   int DilationAtLayer(const int l) const
   {
      int d = 1;
      if(m_dbase <= 1 || l <= 0) return 1;
      for(int i=0; i<l; i++)
      {
         if(d > FX6_TCN_HIST / m_dbase) return FX6_TCN_HIST;
         d *= m_dbase;
      }
      return d;
   }

   double DropMask(const int t,
                   const int l,
                   const int c,
                   const double drop_rate) const
   {
      if(drop_rate <= 1e-9) return 1.0;
      uint h = (uint)(m_step * 2654435761U);
      h ^= (uint)((t + 1) * 2246822519U);
      h ^= (uint)((l + 3) * 3266489917U);
      h ^= (uint)((c + 7) * 668265263U);
      double r = (double)(h & 0xFFFF) / 65535.0;
      if(r < drop_rate) return 0.0;
      return 1.0 / (1.0 - drop_rate);
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

   void RMSNormAffine(const double &v[],
                      const double &g[],
                      const double &b[],
                      double &out[]) const
   {
      double ss = 0.0;
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++) ss += v[i] * v[i];
      double inv = 1.0 / MathSqrt(ss / (double)FX6_AI_MLP_HIDDEN + 1e-6);
      for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
      {
         double n = v[i] * inv;
         out[i] = FX6_ClipSym(g[i] * n + b[i], 8.0);
      }
   }

   void ResetHistory(void)
   {
      m_ptr = 0;
      m_hist_len = 0;
      for(int s=0; s<=FX6_TCN_MAX_LAYERS; s++)
      {
         for(int t=0; t<FX6_TCN_HIST; t++)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
               m_hist_stream[s][t][c] = 0.0;
         }
      }
      for(int l=0; l<FX6_TCN_MAX_LAYERS; l++)
      {
         for(int t=0; t<FX6_TCN_HIST; t++)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
               m_hist_mid[l][t][c] = 0.0;
         }
      }
   }

   void ResetSequence(void)
   {
      m_seq_len = 0;
      for(int t=0; t<FX6_TCN_TBPTT; t++)
      {
         m_seq_cls[t] = FX6_TCN_SKIP;
         m_seq_move[t] = 0.0;
         m_seq_cost[t] = 0.0;
         m_seq_w[t] = 1.0;
         m_seq_mu[t] = 0.0;
         m_seq_logv[t] = MathLog(1.0);
         m_seq_q25[t] = 0.0;
         m_seq_q75[t] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            m_seq_x[t][i] = 0.0;

         for(int c=0; c<FX6_TCN_CLASS_COUNT; c++)
         {
            m_seq_logits[t][c] = 0.0;
            m_seq_probs[t][c] = 1.0 / 3.0;
         }

         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            m_seq_final[t][c] = 0.0;

         for(int l=0; l<FX6_TCN_MAX_LAYERS; l++)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            {
               m_seq_in[t][l][c] = 0.0;
               m_seq_n1[t][l][c] = 0.0;
               m_seq_a1[t][l][c] = 0.0;
               m_seq_drop[t][l][c] = 1.0;
               m_seq_n2[t][l][c] = 0.0;
               m_seq_pre[t][l][c] = 0.0;
               m_seq_gate[t][l][c] = 0.0;
               m_seq_out[t][l][c] = 0.0;
            }
         }
      }
   }

   void ResetOptimizers(void)
   {
      m_adam_t = 0;

      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         m_m_b_in[c] = 0.0; m_v_b_in[c] = 0.0;
         m_m_w_mu[c] = 0.0; m_v_w_mu[c] = 0.0;
         m_m_w_logv[c] = 0.0; m_v_w_logv[c] = 0.0;
         m_m_w_q25[c] = 0.0; m_v_w_q25[c] = 0.0;
         m_m_w_q75[c] = 0.0; m_v_w_q75[c] = 0.0;

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            m_m_w_in[c][i] = 0.0;
            m_v_w_in[c][i] = 0.0;
         }

         for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
         {
            m_m_w_cls[cls][c] = 0.0;
            m_v_w_cls[cls][c] = 0.0;
         }
      }

      m_m_b_mu = 0.0; m_v_b_mu = 0.0;
      m_m_b_logv = 0.0; m_v_b_logv = 0.0;
      m_m_b_q25 = 0.0; m_v_b_q25 = 0.0;
      m_m_b_q75 = 0.0; m_v_b_q75 = 0.0;

      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
      {
         m_m_b_cls[cls] = 0.0;
         m_v_b_cls[cls] = 0.0;
      }

      for(int l=0; l<FX6_TCN_MAX_LAYERS; l++)
      {
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            m_r_b_conv1[l][c] = 0.0;
            m_r_b_conv2[l][c] = 0.0;

            m_m_b_res[l][c] = 0.0; m_v_b_res[l][c] = 0.0;
            m_m_b_gate[l][c] = 0.0; m_v_b_gate[l][c] = 0.0;
            m_m_g1[l][c] = 0.0; m_v_g1[l][c] = 0.0;
            m_m_bn1[l][c] = 0.0; m_v_bn1[l][c] = 0.0;
            m_m_g2[l][c] = 0.0; m_v_g2[l][c] = 0.0;
            m_m_bn2[l][c] = 0.0; m_v_bn2[l][c] = 0.0;

            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            {
               m_m_w_res[l][c][j] = 0.0; m_v_w_res[l][c][j] = 0.0;
               m_m_w_gate[l][c][j] = 0.0; m_v_w_gate[l][c][j] = 0.0;

               for(int k=0; k<FX6_TCN_MAX_KERNEL; k++)
               {
                  m_r_w_conv1[l][c][j][k] = 0.0;
                  m_r_w_conv2[l][c][j][k] = 0.0;
               }
            }
         }
      }
   }

   void ApplyConfig(const FX6AIHyperParams &hp)
   {
      int layers = CfgI(hp.tcn_layers, 4, 2, FX6_TCN_MAX_LAYERS);
      int kernel = CfgI(hp.tcn_kernel, 3, 2, FX6_TCN_MAX_KERNEL);
      int dbase = CfgI(hp.tcn_dilation_base, 2, 1, 3);

      // Keep receptive field bounded by history.
      int use_layers = layers;
      while(use_layers > 2)
      {
         int receptive = 1;
         int d = 1;
         for(int l=0; l<use_layers; l++)
         {
            receptive += (kernel - 1) * d;
            if(l + 1 < use_layers)
            {
               if(d > FX6_TCN_HIST / MathMax(dbase, 1)) d = FX6_TCN_HIST;
               else d *= dbase;
            }
         }
         if(receptive <= FX6_TCN_HIST) break;
         use_layers--;
      }

      bool changed = (!m_cfg_ready || m_layers != use_layers || m_kernel != kernel || m_dbase != dbase);
      m_layers = use_layers;
      m_kernel = kernel;
      m_dbase = dbase;

      // Recommendation #4 + #8: robust optimizer defaults with light schedules.
      m_lr = FX6_Clamp(hp.lr, 0.00005, 0.25000);
      m_l2 = FX6_Clamp(hp.l2, 0.00000, 0.20000);
      m_wd = FX6_Clamp(0.25 * m_l2, 0.0, 0.05);
      m_beta1 = 0.90;
      m_beta2 = 0.999;
      m_rms_decay = 0.99;
      m_drop_rate = FX6_Clamp(0.02 + 0.15 * m_l2, 0.0, 0.20);

      m_cfg_ready = true;
      if(changed) ResetHistory();
   }

   void RMSPropUpdate(double &p,
                      double &r,
                      const double g,
                      const double lr,
                      const double decay,
                      const double wd)
   {
      double gc = FX6_ClipSym(g, 10.0);
      r = decay * r + (1.0 - decay) * gc * gc;
      p -= lr * (gc / MathSqrt(r + 1e-8));
      p -= lr * wd * p;
   }

   void AdamWUpdate(double &p,
                    double &m,
                    double &v,
                    const double g,
                    const double lr,
                    const double b1,
                    const double b2,
                    const double wd,
                    const int t)
   {
      double gc = FX6_ClipSym(g, 10.0);
      m = b1 * m + (1.0 - b1) * gc;
      v = b2 * v + (1.0 - b2) * gc * gc;

      double b1t = 1.0 - MathPow(b1, (double)t);
      double b2t = 1.0 - MathPow(b2, (double)t);
      if(b1t < 1e-9) b1t = 1e-9;
      if(b2t < 1e-9) b2t = 1e-9;

      double mh = m / b1t;
      double vh = v / b2t;
      p -= lr * (mh / (MathSqrt(vh) + 1e-8));
      p -= lr * wd * p;
   }

   int MapClass(const int y,
                const double &x[],
                const double move_points) const
   {
      if(y == FX6_TCN_SELL || y == FX6_TCN_BUY || y == FX6_TCN_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * cost;
      if(edge <= skip_band) return FX6_TCN_SKIP;

      if(y > 0) return FX6_TCN_BUY;
      if(y == 0) return FX6_TCN_SELL;
      return (move_points >= 0.0 ? FX6_TCN_BUY : FX6_TCN_SELL);
   }

   double ClassWeight(const int cls,
                      const double move_points,
                      const double cost,
                      const double sample_w) const
   {
      double edge = MathAbs(move_points) - cost;
      double base = FX6_Clamp(sample_w, 0.25, 4.00);

      if(cls == FX6_TCN_SKIP)
      {
         if(edge <= 0.0) return FX6_Clamp(base * 1.6, 0.25, 6.0);
         return FX6_Clamp(base * 0.75, 0.25, 6.0);
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

   void InitWeights(void)
   {
      m_step = 0;
      ResetHistory();
      ResetSequence();
      ResetOptimizers();

      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            double s = (double)((c + 1) * (i + 3));
            m_w_in[c][i] = 0.025 * MathSin(0.91 * s);
         }
         m_b_in[c] = 0.0;
      }

      for(int l=0; l<FX6_TCN_MAX_LAYERS; l++)
      {
         for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
         {
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double s1 = (double)((l + 1) * (oc + 2) * (ic + 3));
               double s2 = (double)((l + 2) * (oc + 3) * (ic + 1));

               m_w_res[l][oc][ic] = 0.015 * MathCos(0.73 * s1);
               m_w_gate[l][oc][ic] = 0.015 * MathSin(0.77 * s2);

               for(int k=0; k<FX6_TCN_MAX_KERNEL; k++)
               {
                  double a = (double)((l + 1) * (oc + 1) * (ic + 2) * (k + 1));
                  double b = (double)((l + 2) * (oc + 3) * (ic + 1) * (k + 1));
                  m_w_conv1[l][oc][ic][k] = 0.020 * MathSin(0.53 * a);
                  m_w_conv2[l][oc][ic][k] = 0.020 * MathCos(0.59 * b);
               }
            }

            m_b_conv1[l][oc] = 0.0;
            m_b_conv2[l][oc] = 0.0;
            m_b_res[l][oc] = 0.0;
            m_b_gate[l][oc] = 0.0;
            m_g1[l][oc] = 1.0;
            m_bn1[l][oc] = 0.0;
            m_g2[l][oc] = 1.0;
            m_bn2[l][oc] = 0.0;
         }
      }

      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
      {
         m_b_cls[cls] = 0.0;
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            double s = (double)((cls + 1) * (c + 2));
            m_w_cls[cls][c] = 0.020 * MathSin(0.83 * s);
         }
      }

      m_b_mu = 0.0;
      m_b_logv = MathLog(1.0);
      m_b_q25 = -0.5;
      m_b_q75 = 0.5;
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         m_w_mu[c] = 0.020 * MathCos((double)(c + 1) * 0.97);
         m_w_logv[c] = 0.015 * MathSin((double)(c + 1) * 1.07);
         m_w_q25[c] = 0.020 * MathSin((double)(c + 1) * 1.11);
         m_w_q75[c] = 0.020 * MathCos((double)(c + 1) * 1.13);
      }

      m_initialized = true;
      m_cfg_ready = false;
   }

   void ComputeHeads(const double &h[],
                     double &logits[],
                     double &probs[],
                     double &mu,
                     double &logv,
                     double &q25,
                     double &q75) const
   {
      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
      {
         double z = m_b_cls[cls];
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            z += m_w_cls[cls][c] * h[c];
         logits[cls] = z;
      }
      Softmax3(logits, probs);

      mu = m_b_mu;
      logv = m_b_logv;
      q25 = m_b_q25;
      q75 = m_b_q75;
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         mu += m_w_mu[c] * h[c];
         logv += m_w_logv[c] * h[c];
         q25 += m_w_q25[c] * h[c];
         q75 += m_w_q75[c] * h[c];
      }
      logv = FX6_Clamp(logv, -4.0, 4.0);
      if(q25 > q75)
      {
         double tmp = q25;
         q25 = q75;
         q75 = tmp;
      }
   }

   // Forward one step. If commit=true, advances history. If seq_t>=0, stores TBPTT caches.
   void ForwardStep(const double &x[],
                    const bool commit,
                    const bool training,
                    const int seq_t,
                    double &h_last[])
   {
      int ptr_new = (m_ptr + 1) % FX6_TCN_HIST;

      double curr[FX6_AI_MLP_HIDDEN];
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         double s = m_b_in[c];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            s += m_w_in[c][i] * x[i];
         curr[c] = FX6_Tanh(FX6_ClipSym(s, 8.0));
      }

      if(commit)
      {
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            m_hist_stream[0][ptr_new][c] = curr[c];
      }

      double skip_acc[FX6_AI_MLP_HIDDEN];
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++) skip_acc[c] = 0.0;

      for(int l=0; l<m_layers; l++)
      {
         double in_vec[FX6_AI_MLP_HIDDEN];
         double n1[FX6_AI_MLP_HIDDEN];
         double a1[FX6_AI_MLP_HIDDEN];
         double drop_mask[FX6_AI_MLP_HIDDEN];
         double n2[FX6_AI_MLP_HIDDEN];
         double gate[FX6_AI_MLP_HIDDEN];
         double pre[FX6_AI_MLP_HIDDEN];
         double out[FX6_AI_MLP_HIDDEN];

         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            in_vec[c] = curr[c];

         double g1v[FX6_AI_MLP_HIDDEN], b1v[FX6_AI_MLP_HIDDEN];
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            g1v[c] = m_g1[l][c];
            b1v[c] = m_bn1[l][c];
         }
         RMSNormAffine(in_vec, g1v, b1v, n1);

         int d = DilationAtLayer(l);

         // Conv1.
         for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
         {
            double s = m_b_conv1[l][oc];
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               s += m_w_conv1[l][oc][ic][0] * n1[ic];
               for(int k=1; k<m_kernel; k++)
               {
                  int back = k * d;
                  if(back > m_hist_len) continue;
                  int idx = HistIdx(ptr_new, back);
                  double src = (l == 0 ? m_hist_stream[0][idx][ic] : m_hist_stream[l][idx][ic]);
                  s += m_w_conv1[l][oc][ic][k] * src;
               }
            }
            a1[oc] = FX6_Tanh(FX6_ClipSym(s, 8.0));

            double dm = 1.0;
            if(training && commit)
               dm = DropMask(seq_t, l, oc, m_drop_rate);
            drop_mask[oc] = dm;
            a1[oc] *= dm;
         }

         double g2v[FX6_AI_MLP_HIDDEN], b2v[FX6_AI_MLP_HIDDEN];
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            g2v[c] = m_g2[l][c];
            b2v[c] = m_bn2[l][c];
         }
         RMSNormAffine(a1, g2v, b2v, n2);

         // Conv2 + residual/gate.
         for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
         {
            double c2 = m_b_conv2[l][oc];
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               c2 += m_w_conv2[l][oc][ic][0] * n2[ic];
               for(int k=1; k<m_kernel; k++)
               {
                  int back = k * d;
                  if(back > m_hist_len) continue;
                  int idx = HistIdx(ptr_new, back);
                  c2 += m_w_conv2[l][oc][ic][k] * m_hist_mid[l][idx][ic];
               }
            }

            double r = m_b_res[l][oc];
            double g = m_b_gate[l][oc];
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               r += m_w_res[l][oc][ic] * in_vec[ic];
               g += m_w_gate[l][oc][ic] * in_vec[ic];
            }
            g = FX6_Sigmoid(g);

            pre[oc] = r + c2;
            gate[oc] = g;
            out[oc] = FX6_Tanh(FX6_ClipSym(in_vec[oc] + g * pre[oc], 8.0));
            skip_acc[oc] += out[oc];
         }

         if(commit)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            {
               m_hist_mid[l][ptr_new][c] = a1[c];
               m_hist_stream[l + 1][ptr_new][c] = out[c];
            }
         }

         if(seq_t >= 0 && seq_t < FX6_TCN_TBPTT)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            {
               m_seq_in[seq_t][l][c] = in_vec[c];
               m_seq_n1[seq_t][l][c] = n1[c];
               m_seq_a1[seq_t][l][c] = a1[c];
               m_seq_drop[seq_t][l][c] = drop_mask[c];
               m_seq_n2[seq_t][l][c] = n2[c];
               m_seq_pre[seq_t][l][c] = pre[c];
               m_seq_gate[seq_t][l][c] = gate[c];
               m_seq_out[seq_t][l][c] = out[c];
            }
         }

         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            curr[c] = out[c];
      }

      // Aggregate deep skip output for richer representation.
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         double z = curr[c] + (0.20 / (double)MathMax(1, m_layers)) * skip_acc[c];
         h_last[c] = FX6_ClipSym(z, 8.0);
      }

      if(commit)
      {
         m_ptr = ptr_new;
         if(m_hist_len < FX6_TCN_HIST) m_hist_len++;
      }
   }

   void AppendSequenceSample(const int cls,
                             const double &x[],
                             const double move_points,
                             const double cost,
                             const double sample_w,
                             const FX6AIHyperParams &hp)
   {
      if(m_seq_len >= FX6_TCN_TBPTT) return;

      int t = m_seq_len;
      for(int i=0; i<FX6_AI_WEIGHTS; i++)
         m_seq_x[t][i] = x[i];
      m_seq_cls[t] = cls;
      m_seq_move[t] = move_points;
      m_seq_cost[t] = cost;
      m_seq_w[t] = sample_w;

      double h_last[FX6_AI_MLP_HIDDEN];
      ForwardStep(x, true, true, t, h_last);

      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         m_seq_final[t][c] = h_last[c];

      double logits_t[FX6_TCN_CLASS_COUNT];
      double probs_t[FX6_TCN_CLASS_COUNT];
      double mu_t = 0.0, logv_t = 0.0, q25_t = 0.0, q75_t = 0.0;
      ComputeHeads(h_last, logits_t, probs_t, mu_t, logv_t, q25_t, q75_t);
      for(int c=0; c<FX6_TCN_CLASS_COUNT; c++)
      {
         m_seq_logits[t][c] = logits_t[c];
         m_seq_probs[t][c] = probs_t[c];
      }
      m_seq_mu[t] = mu_t;
      m_seq_logv[t] = logv_t;
      m_seq_q25[t] = q25_t;
      m_seq_q75[t] = q75_t;

      // Online plugin-level calibration uses directional probability only.
      double den = m_seq_probs[t][FX6_TCN_BUY] + m_seq_probs[t][FX6_TCN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir = m_seq_probs[t][FX6_TCN_BUY] / den;
      if(cls != FX6_TCN_SKIP)
      {
         int y_dir = (cls == FX6_TCN_BUY ? 1 : 0);
         UpdateCalibration(p_dir, y_dir, sample_w);
      }

      FX6_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, hp, sample_w);

      m_seq_len++;
   }

   void TrainTBPTT(const FX6AIHyperParams &hp)
   {
      int T = m_seq_len;
      if(T <= 0) return;

      // Learning-rate schedule.
      double lr = FX6_Clamp(m_lr, 0.00005, 0.25000) / MathSqrt(1.0 + 0.0005 * (double)m_step);
      double lr_head = FX6_Clamp(1.05 * lr, 0.00002, 0.20000);
      double lr_core = FX6_Clamp(0.80 * lr, 0.00002, 0.15000);

      // Gradient accumulators.
      double g_w_in[FX6_AI_MLP_HIDDEN][FX6_AI_WEIGHTS];
      double g_b_in[FX6_AI_MLP_HIDDEN];

      double g_w_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
      double g_b_conv1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
      double g_w_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN][FX6_TCN_MAX_KERNEL];
      double g_b_conv2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

      double g_w_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_b_res[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
      double g_w_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN][FX6_AI_MLP_HIDDEN];
      double g_b_gate[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

      double g_g1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
      double g_bn1[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
      double g_g2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];
      double g_bn2[FX6_TCN_MAX_LAYERS][FX6_AI_MLP_HIDDEN];

      double g_w_cls[FX6_TCN_CLASS_COUNT][FX6_AI_MLP_HIDDEN];
      double g_b_cls[FX6_TCN_CLASS_COUNT];

      double g_w_mu[FX6_AI_MLP_HIDDEN], g_w_logv[FX6_AI_MLP_HIDDEN], g_w_q25[FX6_AI_MLP_HIDDEN], g_w_q75[FX6_AI_MLP_HIDDEN];
      double g_b_mu, g_b_logv, g_b_q25, g_b_q75;

      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         g_b_in[c] = 0.0;
         g_w_mu[c] = 0.0;
         g_w_logv[c] = 0.0;
         g_w_q25[c] = 0.0;
         g_w_q75[c] = 0.0;
         for(int i=0; i<FX6_AI_WEIGHTS; i++) g_w_in[c][i] = 0.0;
         for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++) g_w_cls[cls][c] = 0.0;
      }
      g_b_mu = 0.0; g_b_logv = 0.0; g_b_q25 = 0.0; g_b_q75 = 0.0;
      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++) g_b_cls[cls] = 0.0;

      for(int l=0; l<FX6_TCN_MAX_LAYERS; l++)
      {
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            g_b_conv1[l][c] = 0.0;
            g_b_conv2[l][c] = 0.0;
            g_b_res[l][c] = 0.0;
            g_b_gate[l][c] = 0.0;
            g_g1[l][c] = 0.0;
            g_bn1[l][c] = 0.0;
            g_g2[l][c] = 0.0;
            g_bn2[l][c] = 0.0;

            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            {
               g_w_res[l][c][j] = 0.0;
               g_w_gate[l][c][j] = 0.0;
               for(int k=0; k<FX6_TCN_MAX_KERNEL; k++)
               {
                  g_w_conv1[l][c][j][k] = 0.0;
                  g_w_conv2[l][c][j][k] = 0.0;
               }
            }
         }
      }

      for(int t=T - 1; t>=0; t--)
      {
         int cls = m_seq_cls[t];
         double move = m_seq_move[t];
         double cost = m_seq_cost[t];
         double sw = m_seq_w[t];

         double w_cls = ClassWeight(cls, move, cost, sw);
         double w_mv = MoveWeight(move, cost, sw);

         double dlogit[FX6_TCN_CLASS_COUNT];
         for(int k=0; k<FX6_TCN_CLASS_COUNT; k++)
         {
            double yk = (k == cls ? 1.0 : 0.0);
            dlogit[k] = (m_seq_probs[t][k] - yk) * w_cls;
            g_b_cls[k] += dlogit[k];
         }

         double dh[FX6_AI_MLP_HIDDEN];
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            dh[c] = 0.0;
            for(int k=0; k<FX6_TCN_CLASS_COUNT; k++)
            {
               g_w_cls[k][c] += dlogit[k] * m_seq_final[t][c];
               dh[c] += dlogit[k] * m_w_cls[k][c];
            }
         }

         // Distributional move heads: mean + log-variance + quantile bands.
         double mu = m_seq_mu[t];
         double logv = m_seq_logv[t];
         double q25 = m_seq_q25[t];
         double q75 = m_seq_q75[t];

         double target = move;
         double inv_var = MathExp(-logv);
         inv_var = FX6_Clamp(inv_var, 0.02, 50.0);
         double err = mu - target;

         double dmu = w_mv * err * inv_var;
         double dlogv = 0.5 * w_mv * (1.0 - err * err * inv_var);

         double tau25 = 0.25;
         double tau75 = 0.75;
         double d_q25 = (target >= q25 ? -tau25 : (1.0 - tau25)) * 0.25 * w_mv;
         double d_q75 = (target >= q75 ? -tau75 : (1.0 - tau75)) * 0.25 * w_mv;

         g_b_mu += dmu;
         g_b_logv += dlogv;
         g_b_q25 += d_q25;
         g_b_q75 += d_q75;

         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            double h = m_seq_final[t][c];
            g_w_mu[c] += dmu * h;
            g_w_logv[c] += dlogv * h;
            g_w_q25[c] += d_q25 * h;
            g_w_q75[c] += d_q75 * h;

            dh[c] += dmu * m_w_mu[c] + dlogv * m_w_logv[c] + d_q25 * m_w_q25[c] + d_q75 * m_w_q75[c];
         }

         // Backprop through residual TCN blocks (truncated within batch window).
         for(int l=m_layers - 1; l>=0; l--)
         {
            int d = DilationAtLayer(l);

            double d_in[FX6_AI_MLP_HIDDEN];
            double d_n1[FX6_AI_MLP_HIDDEN];
            double d_a1[FX6_AI_MLP_HIDDEN];
            double d_n2[FX6_AI_MLP_HIDDEN];
            for(int i=0; i<FX6_AI_MLP_HIDDEN; i++)
            {
               d_in[i] = 0.0;
               d_n1[i] = 0.0;
               d_a1[i] = 0.0;
               d_n2[i] = 0.0;
            }

            for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
            {
               double outv = m_seq_out[t][l][oc];
               double gate = m_seq_gate[t][l][oc];
               double pre = m_seq_pre[t][l][oc];

               double d_pre2 = dh[oc] * (1.0 - outv * outv);
               double d_pre = d_pre2 * gate;
               double d_gate = d_pre2 * pre * gate * (1.0 - gate);

               d_in[oc] += d_pre2;

               g_b_conv2[l][oc] += d_pre;
               g_b_res[l][oc] += d_pre;
               g_b_gate[l][oc] += d_gate;

               for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
               {
                  double src_n2 = m_seq_n2[t][l][ic];
                  g_w_conv2[l][oc][ic][0] += d_pre * src_n2;
                  d_n2[ic] += d_pre * m_w_conv2[l][oc][ic][0];

                  for(int k=1; k<m_kernel; k++)
                  {
                     int tp = t - k * d;
                     if(tp < 0) continue;
                     double src = m_seq_a1[tp][l][ic];
                     g_w_conv2[l][oc][ic][k] += d_pre * src;
                  }

                  double src_in = m_seq_in[t][l][ic];
                  g_w_res[l][oc][ic] += d_pre * src_in;
                  g_w_gate[l][oc][ic] += d_gate * src_in;

                  d_in[ic] += d_pre * m_w_res[l][oc][ic] + d_gate * m_w_gate[l][oc][ic];
               }
            }

            // Norm2 exact RMSNorm+affine backward (with output clip derivative).
            double ss2 = 0.0;
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double xv = m_seq_a1[t][l][ic];
               ss2 += xv * xv;
            }
            double inv2 = 1.0 / MathSqrt(ss2 / (double)FX6_AI_MLP_HIDDEN + 1e-6);
            double inv2_cub = inv2 * inv2 * inv2;

            double dzn2[FX6_AI_MLP_HIDDEN];
            double dot2 = 0.0;
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double clip_grad = (MathAbs(m_seq_n2[t][l][ic]) < 8.0 ? 1.0 : 0.0);
               double dout = d_n2[ic] * clip_grad;
               double xv = m_seq_a1[t][l][ic];
               double xhat = xv * inv2;

               g_g2[l][ic] += dout * xhat;
               g_bn2[l][ic] += dout;

               dzn2[ic] = dout * m_g2[l][ic];
               dot2 += dzn2[ic] * xv;
            }

            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double xv = m_seq_a1[t][l][ic];
               d_a1[ic] += dzn2[ic] * inv2 - xv * dot2 * (inv2_cub / (double)FX6_AI_MLP_HIDDEN);
            }

            // a1 = tanh(conv1) with dropout mask.
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               d_a1[ic] *= m_seq_drop[t][l][ic];
               d_a1[ic] *= (1.0 - m_seq_a1[t][l][ic] * m_seq_a1[t][l][ic]);
            }

            for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
            {
               double d1 = d_a1[oc];
               g_b_conv1[l][oc] += d1;
               for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
               {
                  double src_n1 = m_seq_n1[t][l][ic];
                  g_w_conv1[l][oc][ic][0] += d1 * src_n1;
                  d_n1[ic] += d1 * m_w_conv1[l][oc][ic][0];

                  for(int k=1; k<m_kernel; k++)
                  {
                     int tp = t - k * d;
                     if(tp < 0) continue;
                     double src = (l == 0 ? m_seq_in[tp][0][ic] : m_seq_out[tp][l - 1][ic]);
                     g_w_conv1[l][oc][ic][k] += d1 * src;
                  }
               }
            }

            // Norm1 exact RMSNorm+affine backward (with output clip derivative).
            double ss1 = 0.0;
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double xv = m_seq_in[t][l][ic];
               ss1 += xv * xv;
            }
            double inv1 = 1.0 / MathSqrt(ss1 / (double)FX6_AI_MLP_HIDDEN + 1e-6);
            double inv1_cub = inv1 * inv1 * inv1;

            double dzn1[FX6_AI_MLP_HIDDEN];
            double dot1 = 0.0;
            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double clip_grad = (MathAbs(m_seq_n1[t][l][ic]) < 8.0 ? 1.0 : 0.0);
               double dout = d_n1[ic] * clip_grad;
               double xv = m_seq_in[t][l][ic];
               double xhat = xv * inv1;

               g_g1[l][ic] += dout * xhat;
               g_bn1[l][ic] += dout;

               dzn1[ic] = dout * m_g1[l][ic];
               dot1 += dzn1[ic] * xv;
            }

            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               double xv = m_seq_in[t][l][ic];
               d_in[ic] += dzn1[ic] * inv1 - xv * dot1 * (inv1_cub / (double)FX6_AI_MLP_HIDDEN);
            }

            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
               dh[c] = d_in[c];
         }

         // Input projection grads.
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            double di = 0.20 * dh[c];
            g_b_in[c] += di;
            for(int i=0; i<FX6_AI_WEIGHTS; i++)
               g_w_in[c][i] += di * m_seq_x[t][i];
         }
      }

      // Global grad norm clip.
      double g2 = 0.0;
      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         g2 += g_b_in[c] * g_b_in[c] +
               g_w_mu[c] * g_w_mu[c] + g_w_logv[c] * g_w_logv[c] +
               g_w_q25[c] * g_w_q25[c] + g_w_q75[c] * g_w_q75[c];
         for(int i=0; i<FX6_AI_WEIGHTS; i++)
            g2 += g_w_in[c][i] * g_w_in[c][i];

         for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
            g2 += g_w_cls[cls][c] * g_w_cls[cls][c];
      }

      g2 += g_b_mu * g_b_mu + g_b_logv * g_b_logv + g_b_q25 * g_b_q25 + g_b_q75 * g_b_q75;
      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++) g2 += g_b_cls[cls] * g_b_cls[cls];

      for(int l=0; l<m_layers; l++)
      {
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            g2 += g_b_conv1[l][c] * g_b_conv1[l][c] + g_b_conv2[l][c] * g_b_conv2[l][c];
            g2 += g_b_res[l][c] * g_b_res[l][c] + g_b_gate[l][c] * g_b_gate[l][c];
            g2 += g_g1[l][c] * g_g1[l][c] + g_bn1[l][c] * g_bn1[l][c] + g_g2[l][c] * g_g2[l][c] + g_bn2[l][c] * g_bn2[l][c];
            for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
            {
               g2 += g_w_res[l][c][j] * g_w_res[l][c][j] + g_w_gate[l][c][j] * g_w_gate[l][c][j];
               for(int k=0; k<m_kernel; k++)
               {
                  g2 += g_w_conv1[l][c][j][k] * g_w_conv1[l][c][j][k] +
                        g_w_conv2[l][c][j][k] * g_w_conv2[l][c][j][k];
               }
            }
         }
      }

      double gnorm = MathSqrt(g2);
      double gscale = (gnorm > 12.0 ? (12.0 / gnorm) : 1.0);
      if(gscale < 1.0)
      {
         for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
         {
            g_b_in[c] *= gscale;
            g_w_mu[c] *= gscale;
            g_w_logv[c] *= gscale;
            g_w_q25[c] *= gscale;
            g_w_q75[c] *= gscale;
            for(int i=0; i<FX6_AI_WEIGHTS; i++) g_w_in[c][i] *= gscale;
            for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++) g_w_cls[cls][c] *= gscale;
         }

         g_b_mu *= gscale;
         g_b_logv *= gscale;
         g_b_q25 *= gscale;
         g_b_q75 *= gscale;
         for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++) g_b_cls[cls] *= gscale;

         for(int l=0; l<m_layers; l++)
         {
            for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
            {
               g_b_conv1[l][c] *= gscale;
               g_b_conv2[l][c] *= gscale;
               g_b_res[l][c] *= gscale;
               g_b_gate[l][c] *= gscale;
               g_g1[l][c] *= gscale;
               g_bn1[l][c] *= gscale;
               g_g2[l][c] *= gscale;
               g_bn2[l][c] *= gscale;

               for(int j=0; j<FX6_AI_MLP_HIDDEN; j++)
               {
                  g_w_res[l][c][j] *= gscale;
                  g_w_gate[l][c][j] *= gscale;
                  for(int k=0; k<m_kernel; k++)
                  {
                     g_w_conv1[l][c][j][k] *= gscale;
                     g_w_conv2[l][c][j][k] *= gscale;
                  }
               }
            }
         }
      }

      // Apply optimizer steps.
      m_adam_t++;
      int tstep = MathMax(1, m_adam_t);

      for(int l=0; l<m_layers; l++)
      {
         for(int oc=0; oc<FX6_AI_MLP_HIDDEN; oc++)
         {
            RMSPropUpdate(m_b_conv1[l][oc], m_r_b_conv1[l][oc], g_b_conv1[l][oc], lr_core, m_rms_decay, 0.5 * m_wd);
            RMSPropUpdate(m_b_conv2[l][oc], m_r_b_conv2[l][oc], g_b_conv2[l][oc], lr_core, m_rms_decay, 0.5 * m_wd);

            for(int ic=0; ic<FX6_AI_MLP_HIDDEN; ic++)
            {
               for(int k=0; k<m_kernel; k++)
               {
                  RMSPropUpdate(m_w_conv1[l][oc][ic][k], m_r_w_conv1[l][oc][ic][k], g_w_conv1[l][oc][ic][k], lr_core, m_rms_decay, m_wd);
                  RMSPropUpdate(m_w_conv2[l][oc][ic][k], m_r_w_conv2[l][oc][ic][k], g_w_conv2[l][oc][ic][k], lr_core, m_rms_decay, m_wd);
               }

               AdamWUpdate(m_w_res[l][oc][ic], m_m_w_res[l][oc][ic], m_v_w_res[l][oc][ic], g_w_res[l][oc][ic], lr_core, m_beta1, m_beta2, m_wd, tstep);
               AdamWUpdate(m_w_gate[l][oc][ic], m_m_w_gate[l][oc][ic], m_v_w_gate[l][oc][ic], g_w_gate[l][oc][ic], lr_core, m_beta1, m_beta2, m_wd, tstep);

               m_w_res[l][oc][ic] = FX6_ClipSym(m_w_res[l][oc][ic], 5.0);
               m_w_gate[l][oc][ic] = FX6_ClipSym(m_w_gate[l][oc][ic], 5.0);
            }

            AdamWUpdate(m_b_res[l][oc], m_m_b_res[l][oc], m_v_b_res[l][oc], g_b_res[l][oc], lr_core, m_beta1, m_beta2, 0.0, tstep);
            AdamWUpdate(m_b_gate[l][oc], m_m_b_gate[l][oc], m_v_b_gate[l][oc], g_b_gate[l][oc], lr_core, m_beta1, m_beta2, 0.0, tstep);

            AdamWUpdate(m_g1[l][oc], m_m_g1[l][oc], m_v_g1[l][oc], g_g1[l][oc], 0.5 * lr_core, m_beta1, m_beta2, 0.0, tstep);
            AdamWUpdate(m_bn1[l][oc], m_m_bn1[l][oc], m_v_bn1[l][oc], g_bn1[l][oc], 0.5 * lr_core, m_beta1, m_beta2, 0.0, tstep);
            AdamWUpdate(m_g2[l][oc], m_m_g2[l][oc], m_v_g2[l][oc], g_g2[l][oc], 0.5 * lr_core, m_beta1, m_beta2, 0.0, tstep);
            AdamWUpdate(m_bn2[l][oc], m_m_bn2[l][oc], m_v_bn2[l][oc], g_bn2[l][oc], 0.5 * lr_core, m_beta1, m_beta2, 0.0, tstep);

            m_g1[l][oc] = FX6_Clamp(m_g1[l][oc], 0.10, 6.0);
            m_g2[l][oc] = FX6_Clamp(m_g2[l][oc], 0.10, 6.0);
            m_bn1[l][oc] = FX6_ClipSym(m_bn1[l][oc], 4.0);
            m_bn2[l][oc] = FX6_ClipSym(m_bn2[l][oc], 4.0);
         }
      }

      for(int c=0; c<FX6_AI_MLP_HIDDEN; c++)
      {
         AdamWUpdate(m_b_in[c], m_m_b_in[c], m_v_b_in[c], g_b_in[c], 0.7 * lr_core, m_beta1, m_beta2, 0.0, tstep);

         for(int i=0; i<FX6_AI_WEIGHTS; i++)
         {
            AdamWUpdate(m_w_in[c][i], m_m_w_in[c][i], m_v_w_in[c][i], g_w_in[c][i], lr_core, m_beta1, m_beta2, m_wd, tstep);
            m_w_in[c][i] = FX6_ClipSym(m_w_in[c][i], 5.0);
         }

         for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
         {
            AdamWUpdate(m_w_cls[cls][c], m_m_w_cls[cls][c], m_v_w_cls[cls][c], g_w_cls[cls][c], lr_head, m_beta1, m_beta2, m_wd, tstep);
            m_w_cls[cls][c] = FX6_ClipSym(m_w_cls[cls][c], 6.0);
         }

         AdamWUpdate(m_w_mu[c], m_m_w_mu[c], m_v_w_mu[c], g_w_mu[c], lr_head, m_beta1, m_beta2, m_wd, tstep);
         AdamWUpdate(m_w_logv[c], m_m_w_logv[c], m_v_w_logv[c], g_w_logv[c], lr_head, m_beta1, m_beta2, m_wd, tstep);
         AdamWUpdate(m_w_q25[c], m_m_w_q25[c], m_v_w_q25[c], g_w_q25[c], lr_head, m_beta1, m_beta2, m_wd, tstep);
         AdamWUpdate(m_w_q75[c], m_m_w_q75[c], m_v_w_q75[c], g_w_q75[c], lr_head, m_beta1, m_beta2, m_wd, tstep);

         m_w_mu[c] = FX6_ClipSym(m_w_mu[c], 6.0);
         m_w_logv[c] = FX6_ClipSym(m_w_logv[c], 6.0);
         m_w_q25[c] = FX6_ClipSym(m_w_q25[c], 6.0);
         m_w_q75[c] = FX6_ClipSym(m_w_q75[c], 6.0);
      }

      for(int cls=0; cls<FX6_TCN_CLASS_COUNT; cls++)
      {
         AdamWUpdate(m_b_cls[cls], m_m_b_cls[cls], m_v_b_cls[cls], g_b_cls[cls], lr_head, m_beta1, m_beta2, 0.0, tstep);
         m_b_cls[cls] = FX6_ClipSym(m_b_cls[cls], 6.0);
      }

      AdamWUpdate(m_b_mu, m_m_b_mu, m_v_b_mu, g_b_mu, lr_head, m_beta1, m_beta2, 0.0, tstep);
      AdamWUpdate(m_b_logv, m_m_b_logv, m_v_b_logv, g_b_logv, lr_head, m_beta1, m_beta2, 0.0, tstep);
      AdamWUpdate(m_b_q25, m_m_b_q25, m_v_b_q25, g_b_q25, lr_head, m_beta1, m_beta2, 0.0, tstep);
      AdamWUpdate(m_b_q75, m_m_b_q75, m_v_b_q75, g_b_q75, lr_head, m_beta1, m_beta2, 0.0, tstep);

      m_b_mu = FX6_ClipSym(m_b_mu, 10.0);
      m_b_logv = FX6_Clamp(m_b_logv, -4.0, 4.0);
      m_b_q25 = FX6_ClipSym(m_b_q25, 10.0);
      m_b_q75 = FX6_ClipSym(m_b_q75, 10.0);

      ResetSequence();
   }

public:
   CFX6AITCN(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TYPE_TCN; }
   virtual string AIName(void) const { return "tcn"; }

   virtual void Reset(void)
   {
      CFX6AIPlugin::Reset();
      m_initialized = false;
      m_cfg_ready = false;
      m_step = 0;
      m_layers = 4;
      m_kernel = 3;
      m_dbase = 2;
      m_drop_rate = 0.05;
      m_lr = 0.01;
      m_l2 = 0.001;
      m_wd = 0.0005;
      m_beta1 = 0.90;
      m_beta2 = 0.999;
      m_rms_decay = 0.99;
      ResetHistory();
      ResetSequence();
      ResetOptimizers();
   }

   virtual void EnsureInitialized(const FX6AIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      ApplyConfig(hp);
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
      m_step++;

      FX6AIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      ApplyConfig(h);

      double sw = MoveSampleWeight(x, move_points);
      sw = FX6_Clamp(sw, 0.25, 4.00);
      double cost = InputCostProxyPoints(x);
      int cls = MapClass(y, x, move_points);

      AppendSequenceSample(cls, x, move_points, cost, sw, h);

      if(m_seq_len >= FX6_TCN_TBPTT)
         TrainTBPTT(h);
   }

   virtual double PredictProb(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double h_last[FX6_AI_MLP_HIDDEN];
      ForwardStep(x, false, false, -1, h_last);

      double logits[FX6_TCN_CLASS_COUNT];
      double probs[FX6_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);

      double den = probs[FX6_TCN_BUY] + probs[FX6_TCN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FX6_TCN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);

      double p_up = p_dir_cal * FX6_Clamp(1.0 - probs[FX6_TCN_SKIP], 0.0, 1.0);
      return FX6_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FX6AIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double h_last[FX6_AI_MLP_HIDDEN];
      ForwardStep(x, false, false, -1, h_last);

      double logits[FX6_TCN_CLASS_COUNT];
      double probs[FX6_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);

      double sigma = MathExp(0.5 * logv);
      sigma = FX6_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);

      // Distribution-aware expected move (absolute edge scale).
      double ev = MathAbs(mu) + 0.25 * sigma + 0.10 * iqr;
      ev *= FX6_Clamp(1.0 - probs[FX6_TCN_SKIP], 0.0, 1.0);

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return CFX6AIPlugin::PredictExpectedMovePoints(x, hp);
   }
};

#endif // __FX6_AI_TCN_MQH__
