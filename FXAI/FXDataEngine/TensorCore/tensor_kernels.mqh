#ifndef __FXAI_TENSOR_KERNELS_MQH__
#define __FXAI_TENSOR_KERNELS_MQH__

#include "..\Engine\core.mqh"

struct FXAITensorDims
{
   int model_dim;
   int hidden_dim;
   int head_count;
   int head_dim;
   int seq_cap;
   int stride;
   int patch_size;
   int dilation;
   double pos_step_penalty;
};

FXAITensorDims FXAI_TensorMakeDims(const int model_dim,
                                   const int hidden_dim,
                                   const int head_count,
                                   const int seq_cap,
                                   const int stride = 1,
                                   const int patch_size = 1,
                                   const int dilation = 1,
                                   const double pos_step_penalty = 0.06)
{
   FXAITensorDims cfg;
   cfg.model_dim = MathMax(MathMin(model_dim, FXAI_AI_FEATURES), 1);
   cfg.hidden_dim = MathMax(MathMin(hidden_dim, FXAI_AI_MLP_HIDDEN), 1);
   cfg.head_count = MathMax(MathMin(head_count, FXAI_AI_FEATURES), 1);
   cfg.head_dim = MathMax(cfg.model_dim / cfg.head_count, 1);
   cfg.seq_cap = MathMax(MathMin(seq_cap, FXAI_MAX_SEQUENCE_BARS), 1);
   cfg.stride = MathMax(stride, 1);
   cfg.patch_size = MathMax(patch_size, 1);
   cfg.dilation = MathMax(dilation, 1);
   cfg.pos_step_penalty = FXAI_Clamp(pos_step_penalty, 0.0, 2.0);
   return cfg;
}

int FXAI_TensorFeatureDim(const FXAITensorDims &cfg)
{
   int d = cfg.model_dim;
   if(d < 1) d = 1;
   if(d > FXAI_AI_FEATURES) d = FXAI_AI_FEATURES;
   return d;
}

int FXAI_TensorResolvedSeqCap(const FXAITensorDims &cfg)
{
   int cap = cfg.seq_cap;
   if(cap < 1) cap = 1;
   if(cap > FXAI_MAX_SEQUENCE_BARS) cap = FXAI_MAX_SEQUENCE_BARS;
   return cap;
}

void FXAI_TensorClearSequence(double &seq[][FXAI_AI_WEIGHTS],
                              int &seq_len)
{
   seq_len = 0;
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         seq[t][k] = 0.0;
}

void FXAI_TensorBuildChronologicalSequence(const double &current_x[],
                                           const double &window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           double &seq[][FXAI_AI_WEIGHTS],
                                           int &seq_len)
{
   FXAI_TensorClearSequence(seq, seq_len);

   int usable_window = window_size;
   if(usable_window < 0) usable_window = 0;
   if(usable_window > FXAI_MAX_SEQUENCE_BARS - 1)
      usable_window = FXAI_MAX_SEQUENCE_BARS - 1;

   for(int src=usable_window - 1; src>=0; src--)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         double v = window[src][k];
         seq[seq_len][k] = (MathIsValidNumber(v) ? FXAI_ClipSym(v, 8.0) : 0.0);
      }
      seq[seq_len][0] = 1.0;
      seq_len++;
   }

   if(seq_len >= FXAI_MAX_SEQUENCE_BARS)
      return;

   int xn = ArraySize(current_x);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      double v = (k < xn ? current_x[k] : 0.0);
      seq[seq_len][k] = (MathIsValidNumber(v) ? FXAI_ClipSym(v, 8.0) : 0.0);
   }
   seq[seq_len][0] = 1.0;
   seq_len++;
}

void FXAI_TensorNormalizeSequence(double &seq[][FXAI_AI_WEIGHTS],
                                  const int seq_len,
                                  const int first_feature = 1,
                                  const int last_feature = FXAI_AI_WEIGHTS - 1)
{
   if(seq_len <= 1)
      return;

   int lo = first_feature;
   int hi = last_feature;
   if(lo < 1) lo = 1;
   if(hi >= FXAI_AI_WEIGHTS) hi = FXAI_AI_WEIGHTS - 1;
   if(hi < lo) return;

   for(int k=lo; k<=hi; k++)
   {
      double mean = 0.0;
      for(int t=0; t<seq_len; t++)
         mean += seq[t][k];
      mean /= (double)seq_len;

      double var = 0.0;
      for(int t=0; t<seq_len; t++)
      {
         double d = seq[t][k] - mean;
         var += d * d;
      }
      double inv = 1.0 / MathSqrt(var / (double)seq_len + 1e-6);
      for(int t=0; t<seq_len; t++)
         seq[t][k] = FXAI_ClipSym((seq[t][k] - mean) * inv, 8.0);
   }
}

void FXAI_TensorLayerNorm(double &v[],
                          const int n)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   if(use_n > ArraySize(v)) use_n = ArraySize(v);
   if(use_n <= 0) return;
   double mean = 0.0;
   for(int i=0; i<use_n; i++)
      mean += v[i];
   mean /= (double)use_n;

   double var = 0.0;
   for(int i=0; i<use_n; i++)
   {
      double d = v[i] - mean;
      var += d * d;
   }
   double inv = 1.0 / MathSqrt(var / (double)use_n + 1e-6);
   for(int i=0; i<use_n; i++)
      v[i] = FXAI_ClipSym((v[i] - mean) * inv, 8.0);
}

void FXAI_TensorGemVInputHidden(const double &x[],
                                const double &w[][FXAI_AI_MLP_HIDDEN],
                                const double &b[],
                                double &out[],
                                const int out_dim = FXAI_AI_MLP_HIDDEN)
{
   int od = out_dim;
   if(od < 1) od = 1;
   if(od > FXAI_AI_MLP_HIDDEN) od = FXAI_AI_MLP_HIDDEN;
   ArrayResize(out, od);
   int xn = ArraySize(x);
   for(int o=0; o<od; o++)
   {
      double z = (ArraySize(b) > o ? b[o] : 0.0);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         double xv = (k < xn ? x[k] : 0.0);
         z += xv * w[k][o];
      }
      out[o] = FXAI_ClipSym(z, 16.0);
   }
}

void FXAI_TensorGemVHiddenHidden(const double &x[],
                                 const double &w[][FXAI_AI_MLP_HIDDEN],
                                 const double &b[],
                                 double &out[],
                                 const int out_dim = FXAI_AI_MLP_HIDDEN)
{
   int od = out_dim;
   if(od < 1) od = 1;
   if(od > FXAI_AI_MLP_HIDDEN) od = FXAI_AI_MLP_HIDDEN;
   ArrayResize(out, od);
   int xn = ArraySize(x);
   for(int o=0; o<od; o++)
   {
      double z = (ArraySize(b) > o ? b[o] : 0.0);
      for(int k=0; k<FXAI_AI_MLP_HIDDEN; k++)
      {
         double xv = (k < xn ? x[k] : 0.0);
         z += xv * w[k][o];
      }
      out[o] = FXAI_ClipSym(z, 16.0);
   }
}

void FXAI_TensorBatchedMatMul(const double &src[][FXAI_AI_WEIGHTS],
                              const int seq_len,
                              const double &w[][FXAI_AI_MLP_HIDDEN],
                              const double &b[],
                              double &dst[][FXAI_AI_MLP_HIDDEN],
                              const int out_dim = FXAI_AI_MLP_HIDDEN)
{
   int od = out_dim;
   if(od < 1) od = 1;
   if(od > FXAI_AI_MLP_HIDDEN) od = FXAI_AI_MLP_HIDDEN;

   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      for(int o=0; o<FXAI_AI_MLP_HIDDEN; o++)
         dst[t][o] = 0.0;

   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      for(int o=0; o<od; o++)
      {
         double z = (ArraySize(b) > o ? b[o] : 0.0);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            z += src[t][k] * w[k][o];
         dst[t][o] = FXAI_ClipSym(z, 16.0);
      }
   }
}

void FXAI_TensorConv1DSummary(const double &seq[][FXAI_AI_WEIGHTS],
                              const int seq_len,
                              const double &kernel[],
                              const int kernel_size,
                              double &summary[])
{
   ArrayResize(summary, FXAI_AI_WEIGHTS);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      summary[k] = 0.0;

   if(seq_len <= 0 || kernel_size <= 0 || ArraySize(kernel) < kernel_size)
      return;

   int eff = MathMin(seq_len, FXAI_MAX_SEQUENCE_BARS);
   for(int feat=1; feat<FXAI_AI_WEIGHTS; feat++)
   {
      double acc = 0.0;
      double den = 0.0;
      for(int t=0; t<eff; t++)
      {
         double y = 0.0;
         double kw = 0.0;
         for(int j=0; j<kernel_size; j++)
         {
            int idx = t - j;
            if(idx < 0) break;
            double wj = kernel[j];
            y += wj * seq[idx][feat];
            kw += MathAbs(wj);
         }
         if(kw <= 0.0) kw = 1.0;
         acc += y / kw;
         den += 1.0;
      }
      if(den <= 0.0) den = 1.0;
      summary[feat] = FXAI_ClipSym(acc / den, 8.0);
   }
   summary[0] = 1.0;
}

void FXAI_TensorAttentionSummary(const double &seq[][FXAI_AI_WEIGHTS],
                                 const int seq_len,
                                 const double &query[],
                                 double &summary[])
{
   ArrayResize(summary, FXAI_AI_WEIGHTS);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      summary[k] = 0.0;
   if(seq_len <= 0)
      return;

   double scores[FXAI_MAX_SEQUENCE_BARS];
   double smax = -1e100;
   int qn = ArraySize(query);
   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      double z = 0.0;
      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
      {
         double qv = (k < qn ? query[k] : 0.0);
         z += qv * seq[t][k];
      }
      z /= MathSqrt((double)MathMax(FXAI_AI_FEATURES, 1));
      scores[t] = z;
      if(z > smax) smax = z;
   }

   double den = 0.0;
   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
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

void FXAI_TensorAttentionSummaryMasked(const double &seq[][FXAI_AI_WEIGHTS],
                                       const int seq_len,
                                       const double &query[],
                                       const int &mask[],
                                       const double &pos_bias[],
                                       double &summary[])
{
   ArrayResize(summary, FXAI_AI_WEIGHTS);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      summary[k] = 0.0;
   if(seq_len <= 0)
      return;

   double scores[FXAI_MAX_SEQUENCE_BARS];
   double smax = -1e100;
   int qn = ArraySize(query);
   for(int t=0; t<seq_len && t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      if(ArraySize(mask) > t && mask[t] <= 0)
      {
         scores[t] = -1e100;
         continue;
      }
      double z = 0.0;
      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
      {
         double qv = (k < qn ? query[k] : 0.0);
         z += qv * seq[t][k];
      }
      z /= MathSqrt((double)MathMax(FXAI_AI_FEATURES, 1));
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

int FXAI_TensorBuildTrailingUnroll(const double &seq[][FXAI_AI_WEIGHTS],
                                   const int seq_len,
                                   const int max_steps,
                                   double &out[][FXAI_AI_WEIGHTS])
{
   int use_n = max_steps;
   if(use_n < 1) use_n = 1;
   if(use_n > FXAI_MAX_SEQUENCE_BARS) use_n = FXAI_MAX_SEQUENCE_BARS;
   if(use_n > seq_len) use_n = seq_len;
   if(use_n < 0) use_n = 0;

   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         out[t][k] = 0.0;

   int start = seq_len - use_n;
   if(start < 0) start = 0;
   for(int t=0; t<use_n; t++)
   {
      int src = start + t;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         out[t][k] = seq[src][k];
   }
   return use_n;
}

void FXAI_TensorCausalConv1DSequence(const double &seq[][FXAI_AI_WEIGHTS],
                                     const int seq_len,
                                     const double &kernel[],
                                     const int kernel_size,
                                     const FXAITensorDims &cfg,
                                     double &out[][FXAI_AI_WEIGHTS])
{
   int cap = FXAI_TensorResolvedSeqCap(cfg);
   int feat_n = FXAI_TensorFeatureDim(cfg);
   int dil = MathMax(cfg.dilation, 1);
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      out[t][0] = 1.0;
      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
         out[t][k] = 0.0;
   }
   if(seq_len <= 0 || kernel_size <= 0 || ArraySize(kernel) < kernel_size)
      return;

   int eff = MathMin(seq_len, cap);
   for(int t=0; t<eff; t++)
   {
      out[t][0] = 1.0;
      for(int feat=1; feat<=feat_n; feat++)
      {
         double acc = 0.0;
         double den = 0.0;
         for(int j=0; j<kernel_size; j++)
         {
            int idx = t - j * dil;
            if(idx < 0) break;
            double wj = kernel[j];
            acc += wj * seq[idx][feat];
            den += MathAbs(wj);
         }
         if(den <= 0.0) den = 1.0;
         out[t][feat] = FXAI_ClipSym(acc / den, 8.0);
      }
      for(int feat=feat_n + 1; feat<FXAI_AI_WEIGHTS; feat++)
         out[t][feat] = seq[t][feat];
   }
}

void FXAI_TensorCausalSelfAttentionSequence(const double &seq[][FXAI_AI_WEIGHTS],
                                            const int seq_len,
                                            const int &mask[],
                                            const double &pos_bias[],
                                            const FXAITensorDims &cfg,
                                            double &out[][FXAI_AI_WEIGHTS])
{
   int cap = FXAI_TensorResolvedSeqCap(cfg);
   int feat_n = FXAI_TensorFeatureDim(cfg);
   int heads = MathMax(MathMin(cfg.head_count, feat_n), 1);
   int head_dim = MathMax(cfg.head_dim, 1);
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      out[t][0] = 1.0;
      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
         out[t][k] = 0.0;
   }
   if(seq_len <= 0)
      return;

   int eff = MathMin(seq_len, cap);
   double scores[FXAI_MAX_SEQUENCE_BARS];
   for(int t=0; t<eff; t++)
   {
      out[t][0] = 1.0;
      double smax = -1e100;
      for(int s=0; s<=t; s++)
      {
         if(ArraySize(mask) > s && mask[s] <= 0)
         {
            scores[s] = -1e100;
            continue;
         }
         double z = 0.0;
         int active_heads = 0;
         for(int h=0; h<heads; h++)
         {
            int start = 1 + h * head_dim;
            if(start > feat_n) break;
            int end = MathMin(feat_n, start + head_dim - 1);
            double dot = 0.0;
            int hd = 0;
            for(int k=start; k<=end; k++)
            {
               dot += seq[t][k] * seq[s][k];
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
         if(ArraySize(pos_bias) > s)
            z += pos_bias[s];
         scores[s] = z;
         if(z > smax) smax = z;
      }

      double den = 0.0;
      for(int s=0; s<=t; s++)
      {
         if(scores[s] <= -1e90)
         {
            scores[s] = 0.0;
            continue;
         }
         scores[s] = MathExp(FXAI_ClipSym(scores[s] - smax, 30.0));
         den += scores[s];
      }
      if(den <= 0.0) den = 1.0;

      for(int feat=1; feat<=feat_n; feat++)
      {
         double acc = 0.0;
         for(int s=0; s<=t; s++)
            acc += (scores[s] / den) * seq[s][feat];
         out[t][feat] = FXAI_ClipSym(acc, 8.0);
      }
      for(int feat=feat_n + 1; feat<FXAI_AI_WEIGHTS; feat++)
         out[t][feat] = seq[t][feat];
   }
}

void FXAI_TensorSequencePool(const double &seq[][FXAI_AI_WEIGHTS],
                             const int seq_len,
                             const FXAITensorDims &cfg,
                             double &summary[])
{
   ArrayResize(summary, FXAI_AI_WEIGHTS);
   ArrayInitialize(summary, 0.0);
   if(seq_len <= 0)
      return;

   int cap = FXAI_TensorResolvedSeqCap(cfg);
   int feat_n = FXAI_TensorFeatureDim(cfg);
   int eff = MathMin(seq_len, cap);
   int last = eff - 1;
   int prev = MathMax(last - 1, 0);
   int mid = MathMax(eff / 2, 0);
   int root = 0;
   double inv = 1.0 / (double)MathMax(eff, 1);
   summary[0] = 1.0;
   for(int feat=1; feat<=feat_n; feat++)
   {
      double mean = 0.0;
      for(int t=0; t<eff; t++)
         mean += seq[t][feat];
      mean *= inv;
      summary[feat] = FXAI_ClipSym(0.40 * seq[last][feat] +
                                   0.18 * seq[prev][feat] +
                                   0.14 * seq[mid][feat] +
                                   0.12 * seq[root][feat] +
                                   0.16 * mean,
                                   8.0);
   }
}

#endif // __FXAI_TENSOR_KERNELS_MQH__
