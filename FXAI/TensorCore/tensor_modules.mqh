#ifndef __FXAI_NN_MODULES_MQH__
#define __FXAI_NN_MODULES_MQH__

#include "tensor_sequence.mqh"
#include "tensor_losses.mqh"

#ifndef FXAI_HIDDEN_HEADS_MAX
#define FXAI_HIDDEN_HEADS_MAX 4
#endif

double FXAI_ModuleReLU(const double x)
{
   return (x > 0.0 ? x : 0.0);
}

double FXAI_ModuleGELU(const double x)
{
   double x3 = x * x * x;
   double t = 0.7978845608 * (x + 0.044715 * x3);
   return 0.5 * x * (1.0 + FXAI_Tanh(t));
}

double FXAI_ModuleSwish(const double x)
{
   return x * FXAI_Sigmoid(x);
}

void FXAI_ModuleLinearInputHidden(const double &x[],
                                  const double &w[][FXAI_AI_MLP_HIDDEN],
                                  const double &b[],
                                  double &out[],
                                  const int out_dim = FXAI_AI_MLP_HIDDEN)
{
   int od = out_dim;
   if(od < 1) od = 1;
   if(od > FXAI_AI_MLP_HIDDEN) od = FXAI_AI_MLP_HIDDEN;
   ArrayResize(out, od);
   FXAI_TensorGemVInputHidden(x, w, b, out, od);
}

void FXAI_ModuleLinearHidden(const double &x[],
                             const double &w[][FXAI_AI_MLP_HIDDEN],
                             const double &b[],
                             double &out[],
                             const int out_dim = FXAI_AI_MLP_HIDDEN)
{
   int od = out_dim;
   if(od < 1) od = 1;
   if(od > FXAI_AI_MLP_HIDDEN) od = FXAI_AI_MLP_HIDDEN;
   ArrayResize(out, od);
   FXAI_TensorGemVHiddenHidden(x, w, b, out, od);
}

void FXAI_ModuleLayerNormAffine(double &v[],
                                const int n,
                                const double &g[],
                                const double &b[])
{
   FXAI_TensorLayerNorm(v, n);
   for(int i=0; i<n && i<ArraySize(v); i++)
   {
      double gi = (ArraySize(g) > i ? g[i] : 1.0);
      double bi = (ArraySize(b) > i ? b[i] : 0.0);
      v[i] = FXAI_ClipSym(gi * v[i] + bi, 8.0);
   }
}

void FXAI_ModuleRMSNormAffine(double &v[],
                              const int n,
                              const double &g[],
                              const double &b[])
{
   if(n <= 0) return;
   double rms = 0.0;
   for(int i=0; i<n && i<ArraySize(v); i++)
      rms += v[i] * v[i];
   rms = MathSqrt(rms / (double)MathMax(n, 1) + 1e-6);
   if(rms <= 0.0) rms = 1.0;
   for(int i=0; i<n && i<ArraySize(v); i++)
   {
      double gi = (ArraySize(g) > i ? g[i] : 1.0);
      double bi = (ArraySize(b) > i ? b[i] : 0.0);
      v[i] = FXAI_ClipSym(gi * (v[i] / rms) + bi, 8.0);
   }
}

void FXAI_ModuleConv1DSummary(const double &seq[][FXAI_AI_WEIGHTS],
                              const int seq_len,
                              const double &kernel[],
                              const int kernel_size,
                              double &summary[])
{
   FXAI_TensorConv1DSummary(seq, seq_len, kernel, kernel_size, summary);
}

void FXAI_ModuleMultiHeadAttentionSummary(const double &seq[][FXAI_AI_WEIGHTS],
                                          const int seq_len,
                                          const double &query[],
                                          const int &mask[],
                                          const double &pos_bias[],
                                          const int num_heads,
                                          double &summary[])
{
   ArrayResize(summary, FXAI_AI_WEIGHTS);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      summary[k] = 0.0;
   if(seq_len <= 0)
      return;

   int heads = num_heads;
   if(heads < 1) heads = 1;
   if(heads > FXAI_HIDDEN_HEADS_MAX) heads = FXAI_HIDDEN_HEADS_MAX;
   int feat_n = FXAI_AI_WEIGHTS - 1;
   int head_dim = MathMax(feat_n / heads, 1);

   double scores[FXAI_MAX_SEQUENCE_BARS];
   double smax = -1e100;
   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      if(ArraySize(mask) > t && mask[t] <= 0)
      {
         scores[t] = -1e100;
         continue;
      }
      double z = 0.0;
      int active_heads = 0;
      for(int h=0; h<heads; h++)
      {
         int start = 1 + h * head_dim;
         int end = (h == heads - 1 ? FXAI_AI_WEIGHTS - 1 : MathMin(FXAI_AI_WEIGHTS - 1, start + head_dim - 1));
         if(start > FXAI_AI_WEIGHTS - 1) break;
         double dot = 0.0;
         int hd = 0;
         for(int k=start; k<=end; k++)
         {
            dot += query[k] * seq[t][k];
            hd++;
         }
         if(hd > 0)
         {
            z += dot / MathSqrt((double)hd);
            active_heads++;
         }
      }
      if(active_heads <= 0) active_heads = 1;
      z /= (double)active_heads;
      if(ArraySize(pos_bias) > t)
         z += pos_bias[t];
      scores[t] = z;
      if(z > smax) smax = z;
   }

   double den = 0.0;
   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      if(scores[t] <= -1e90)
      {
         scores[t] = 0.0;
         continue;
      }
      scores[t] = MathExp(FXAI_ClipSym(scores[t] - smax, 30.0));
      den += scores[t];
   }
   if(den <= 0.0) den = 1.0;

   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      double wt = scores[t] / den;
      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
         summary[k] += wt * seq[t][k];
   }
   summary[0] = 1.0;
}

void FXAI_ModuleLSTMCellForward(const double &i_gate[],
                                const double &f_gate[],
                                const double &o_gate[],
                                const double &g_gate[],
                                const double &c_prev[],
                                double &c_next[],
                                double &h_next[],
                                const int hidden_dim = FXAI_AI_MLP_HIDDEN)
{
   int hd = hidden_dim;
   if(hd < 1) hd = 1;
   if(hd > FXAI_AI_MLP_HIDDEN) hd = FXAI_AI_MLP_HIDDEN;
   for(int h=0; h<hd; h++)
   {
      double i = FXAI_Clamp(i_gate[h], 0.0, 1.0);
      double f = FXAI_Clamp(f_gate[h], 0.0, 1.0);
      double o = FXAI_Clamp(o_gate[h], 0.0, 1.0);
      double g = FXAI_ClipSym(g_gate[h], 4.0);
      c_next[h] = FXAI_ClipSym(f * c_prev[h] + i * g, 8.0);
      h_next[h] = FXAI_ClipSym(o * FXAI_Tanh(c_next[h]), 8.0);
   }
}

void FXAI_ModuleGRUCellForward(const double &z_gate[],
                               const double &r_gate[],
                               const double &n_gate[],
                               const double &h_prev[],
                               double &h_next[],
                               const int hidden_dim = FXAI_AI_MLP_HIDDEN)
{
   int hd = hidden_dim;
   if(hd < 1) hd = 1;
   if(hd > FXAI_AI_MLP_HIDDEN) hd = FXAI_AI_MLP_HIDDEN;
   for(int h=0; h<hd; h++)
   {
      double z = FXAI_Clamp(z_gate[h], 0.0, 1.0);
      double r = FXAI_Clamp(r_gate[h], 0.0, 1.0);
      double n = FXAI_ClipSym(n_gate[h] + 0.10 * r * h_prev[h], 4.0);
      h_next[h] = FXAI_ClipSym((1.0 - z) * n + z * h_prev[h], 8.0);
   }
}

void FXAI_ModuleStateSpaceBlockForward(const double &x_in[],
                                       const double &prev_state[],
                                       const double &decay[],
                                       const double &input_mix[][FXAI_AI_WEIGHTS],
                                       const double &skip[],
                                       double &next_state[],
                                       double &out[],
                                       const int hidden_dim = FXAI_AI_MLP_HIDDEN)
{
   int hd = hidden_dim;
   if(hd < 1) hd = 1;
   if(hd > FXAI_AI_MLP_HIDDEN) hd = FXAI_AI_MLP_HIDDEN;
   ArrayResize(out, hd);
   for(int h=0; h<hd; h++)
   {
      double drive = 0.0;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         drive += input_mix[h][k] * x_in[k];
      double a = FXAI_Clamp(decay[h], 0.0, 1.0);
      next_state[h] = FXAI_ClipSym(a * prev_state[h] + (1.0 - a) * drive, 8.0);
      double s = (ArraySize(skip) > h ? skip[h] * x_in[MathMin(h + 1, FXAI_AI_WEIGHTS - 1)] : 0.0);
      out[h] = FXAI_ClipSym(next_state[h] + s, 8.0);
   }
}

void FXAI_ModuleGateBlend(const double &base[],
                          const double &alt[],
                          const double gate,
                          double &out[],
                          const int n)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   ArrayResize(out, use_n);
   double g = FXAI_Clamp(gate, 0.0, 1.0);
   for(int i=0; i<use_n; i++)
   {
      double a = (ArraySize(base) > i ? base[i] : 0.0);
      double b = (ArraySize(alt) > i ? alt[i] : 0.0);
      out[i] = FXAI_ClipSym((1.0 - g) * a + g * b, 8.0);
   }
}

void FXAI_ModuleMixtureFuse3(const double &a[],
                             const double &b[],
                             const double &c[],
                             const double &weights[],
                             double &out[],
                             const int n)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   ArrayResize(out, use_n);
   double w0 = (ArraySize(weights) > 0 ? MathMax(weights[0], 0.0) : 1.0);
   double w1 = (ArraySize(weights) > 1 ? MathMax(weights[1], 0.0) : 0.0);
   double w2 = (ArraySize(weights) > 2 ? MathMax(weights[2], 0.0) : 0.0);
   double den = w0 + w1 + w2;
   if(den <= 0.0) den = 1.0;
   w0 /= den;
   w1 /= den;
   w2 /= den;
   for(int i=0; i<use_n; i++)
   {
      double va = (ArraySize(a) > i ? a[i] : 0.0);
      double vb = (ArraySize(b) > i ? b[i] : 0.0);
      double vc = (ArraySize(c) > i ? c[i] : 0.0);
      out[i] = FXAI_ClipSym(w0 * va + w1 * vb + w2 * vc, 8.0);
   }
}

#endif // __FXAI_NN_MODULES_MQH__
