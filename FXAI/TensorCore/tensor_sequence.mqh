#ifndef __FXAI_NN_SEQUENCE_MQH__
#define __FXAI_NN_SEQUENCE_MQH__

#include "tensor_kernels.mqh"

struct FXAISequenceRuntimeConfig
{
   int max_steps;
   int stride;
   int patch_size;
   bool normalize;
   bool include_current;
   double pos_step_penalty;
};

struct FXAISequenceBuffer
{
   int len;
   double data[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int mask[FXAI_MAX_SEQUENCE_BARS];
   double pos_bias[FXAI_MAX_SEQUENCE_BARS];
};

struct FXAISequenceRuntimeState
{
   FXAISequenceRuntimeConfig cfg;
   FXAISequenceBuffer buffer;
   int steps_seen;
};

FXAISequenceRuntimeConfig FXAI_SequenceRuntimeMakeConfig(const int max_steps,
                                                         const int stride = 1,
                                                         const int patch_size = 1,
                                                         const bool normalize = true,
                                                         const bool include_current = true,
                                                         const double pos_step_penalty = 0.06)
{
   FXAISequenceRuntimeConfig cfg;
   cfg.max_steps = MathMax(MathMin(max_steps, FXAI_MAX_SEQUENCE_BARS), 1);
   cfg.stride = MathMax(stride, 1);
   cfg.patch_size = MathMax(patch_size, 1);
   cfg.normalize = normalize;
   cfg.include_current = include_current;
   cfg.pos_step_penalty = FXAI_Clamp(pos_step_penalty, 0.0, 2.0);
   return cfg;
}

void FXAI_SequenceBufferReset(FXAISequenceBuffer &buffer)
{
   buffer.len = 0;
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      buffer.mask[t] = 0;
      buffer.pos_bias[t] = 0.0;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         buffer.data[t][k] = 0.0;
   }
}

void FXAI_SequenceBufferCopy(const FXAISequenceBuffer &src,
                             FXAISequenceBuffer &dst)
{
   dst.len = src.len;
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      dst.mask[t] = src.mask[t];
      dst.pos_bias[t] = src.pos_bias[t];
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dst.data[t][k] = src.data[t][k];
   }
}

void FXAI_SequenceBufferBuildMask(FXAISequenceBuffer &buffer)
{
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      buffer.mask[t] = (t < buffer.len ? 1 : 0);
}

void FXAI_SequenceBufferBuildPositionalBias(FXAISequenceBuffer &buffer,
                                            const double step_penalty = 0.06)
{
   double penalty = FXAI_Clamp(step_penalty, 0.0, 2.0);
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      buffer.pos_bias[t] = 0.0;
   for(int t=0; t<buffer.len; t++)
   {
      int lag = buffer.len - 1 - t;
      buffer.pos_bias[t] = -penalty * (double)lag;
   }
}

void FXAI_SequenceRuntimeReset(FXAISequenceRuntimeState &state,
                               const FXAISequenceRuntimeConfig &cfg)
{
   state.cfg = cfg;
   FXAI_SequenceBufferReset(state.buffer);
   state.steps_seen = 0;
}

void FXAI_SequenceRuntimeCopy(const FXAISequenceRuntimeState &src,
                              FXAISequenceRuntimeState &dst)
{
   dst.cfg = src.cfg;
   FXAI_SequenceBufferCopy(src.buffer, dst.buffer);
   dst.steps_seen = src.steps_seen;
}

bool FXAI_SequenceBufferPush(FXAISequenceBuffer &buffer,
                             const double &x[])
{
   if(buffer.len >= FXAI_MAX_SEQUENCE_BARS)
   {
      for(int t=1; t<FXAI_MAX_SEQUENCE_BARS; t++)
      {
         buffer.mask[t - 1] = buffer.mask[t];
         buffer.pos_bias[t - 1] = buffer.pos_bias[t];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            buffer.data[t - 1][k] = buffer.data[t][k];
      }
      buffer.len = FXAI_MAX_SEQUENCE_BARS - 1;
   }

   int idx = buffer.len;
   int xn = ArraySize(x);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      double v = (k < xn && MathIsValidNumber(x[k]) ? x[k] : 0.0);
      buffer.data[idx][k] = (k == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
   }
   buffer.len++;
   FXAI_SequenceBufferBuildMask(buffer);
   FXAI_SequenceBufferBuildPositionalBias(buffer);
   return true;
}

int FXAI_SequenceResampleChronological(const double &src[][FXAI_AI_WEIGHTS],
                                       const int src_len,
                                       const FXAISequenceRuntimeConfig &cfg,
                                       double &dst[][FXAI_AI_WEIGHTS])
{
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dst[t][k] = 0.0;
   if(src_len <= 0)
      return 0;

   int patch = MathMax(cfg.patch_size, 1);
   int stride = MathMax(cfg.stride, 1);
   int cap = MathMax(MathMin(cfg.max_steps, FXAI_MAX_SEQUENCE_BARS), 1);
   double patched[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int patched_len = 0;
   for(int start=0; start<src_len && patched_len < FXAI_MAX_SEQUENCE_BARS; start += patch)
   {
      int stop = MathMin(start + patch, src_len);
      int n = MathMax(stop - start, 1);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         double acc = 0.0;
         for(int t=start; t<stop; t++)
            acc += src[t][k];
         patched[patched_len][k] = (k == 0 ? 1.0 : FXAI_ClipSym(acc / (double)n, 8.0));
      }
      patched_len++;
   }

   int out_len = 0;
   for(int t=0; t<patched_len && out_len < FXAI_MAX_SEQUENCE_BARS; t++)
   {
      bool keep = (((patched_len - 1 - t) % stride) == 0);
      if(t == patched_len - 1) keep = true;
      if(!keep) continue;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dst[out_len][k] = patched[t][k];
      out_len++;
   }

   if(out_len > cap)
   {
      int offset = out_len - cap;
      for(int t=0; t<cap; t++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            dst[t][k] = dst[offset + t][k];
      out_len = cap;
      for(int t=out_len; t<FXAI_MAX_SEQUENCE_BARS; t++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            dst[t][k] = 0.0;
   }
   return out_len;
}

void FXAI_SequenceCopyRow(const double &src[][FXAI_AI_WEIGHTS],
                          const int row,
                          double &dst[])
{
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      dst[k] = src[row][k];
}

bool FXAI_SequenceBufferPushRow(FXAISequenceBuffer &buffer,
                                const double &src[][FXAI_AI_WEIGHTS],
                                const int row)
{
   double row_buf[FXAI_AI_WEIGHTS];
   FXAI_SequenceCopyRow(src, row, row_buf);
   return FXAI_SequenceBufferPush(buffer, row_buf);
}

void FXAI_SequenceBufferLoadWindow(FXAISequenceBuffer &buffer,
                                   const double &current_x[],
                                   const double &window[][FXAI_AI_WEIGHTS],
                                   const int window_size,
                                   const int max_steps,
                                   const bool normalize = true)
{
   FXAISequenceRuntimeConfig cfg = FXAI_SequenceRuntimeMakeConfig(max_steps, 1, 1, normalize, true, 0.06);
   FXAI_SequenceBufferLoadWindowConfig(buffer, current_x, window, window_size, cfg);
}

void FXAI_SequenceBufferLoadWindowConfig(FXAISequenceBuffer &buffer,
                                         const double &current_x[],
                                         const double &window[][FXAI_AI_WEIGHTS],
                                         const int window_size,
                                         const FXAISequenceRuntimeConfig &cfg)
{
   FXAI_SequenceBufferReset(buffer);
   double raw[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int raw_len = 0;
   FXAI_TensorBuildChronologicalSequence(current_x, window, window_size, raw, raw_len);
   if(!cfg.include_current && raw_len > 0)
      raw_len--;

   double resampled[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int seq_len = FXAI_SequenceResampleChronological(raw, raw_len, cfg, resampled);
   for(int t=0; t<seq_len; t++)
      FXAI_SequenceBufferPushRow(buffer, resampled, t);

   if(cfg.normalize && buffer.len > 1)
      FXAI_TensorNormalizeSequence(buffer.data, buffer.len);
   FXAI_SequenceBufferBuildMask(buffer);
   FXAI_SequenceBufferBuildPositionalBias(buffer, cfg.pos_step_penalty);
}

bool FXAI_SequenceRuntimePush(FXAISequenceRuntimeState &state,
                              const double &x[])
{
   state.steps_seen++;
   bool ok = FXAI_SequenceBufferPush(state.buffer, x);
   FXAI_SequenceBufferBuildPositionalBias(state.buffer, state.cfg.pos_step_penalty);
   return ok;
}

void FXAI_SequenceRuntimeLoadWindow(FXAISequenceRuntimeState &state,
                                    const double &current_x[],
                                    const double &window[][FXAI_AI_WEIGHTS],
                                    const int window_size)
{
   FXAI_SequenceBufferLoadWindowConfig(state.buffer, current_x, window, window_size, state.cfg);
   state.steps_seen = state.buffer.len;
}

void FXAI_SequenceRuntimePreparePacked(const FXAISequenceRuntimeState &state,
                                       double &seq[][FXAI_AI_WEIGHTS],
                                       int &seq_len,
                                       int &mask[],
                                       double &pos_bias[])
{
   FXAI_SequenceBufferPreparePacked(state.buffer, seq, seq_len, mask, pos_bias);
}

void FXAI_SequenceBufferExport(const FXAISequenceBuffer &buffer,
                               double &seq[][FXAI_AI_WEIGHTS],
                               int &seq_len)
{
   FXAI_TensorClearSequence(seq, seq_len);
   seq_len = buffer.len;
   if(seq_len < 0) seq_len = 0;
   if(seq_len > FXAI_MAX_SEQUENCE_BARS) seq_len = FXAI_MAX_SEQUENCE_BARS;
   for(int t=0; t<seq_len; t++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         seq[t][k] = buffer.data[t][k];
}

void FXAI_SequenceBufferPreparePacked(const FXAISequenceBuffer &buffer,
                                      double &seq[][FXAI_AI_WEIGHTS],
                                      int &seq_len,
                                      int &mask[],
                                      double &pos_bias[])
{
   FXAI_SequenceBufferExport(buffer, seq, seq_len);
   ArrayResize(mask, FXAI_MAX_SEQUENCE_BARS);
   ArrayResize(pos_bias, FXAI_MAX_SEQUENCE_BARS);
   for(int t=0; t<FXAI_MAX_SEQUENCE_BARS; t++)
   {
      mask[t] = buffer.mask[t];
      pos_bias[t] = buffer.pos_bias[t];
   }
}

#endif // __FXAI_NN_SEQUENCE_MQH__
