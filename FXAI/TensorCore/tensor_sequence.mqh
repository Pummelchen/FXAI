#ifndef __FXAI_NN_SEQUENCE_MQH__
#define __FXAI_NN_SEQUENCE_MQH__

#include "tensor_kernels.mqh"

struct FXAISequenceBuffer
{
   int len;
   double data[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int mask[FXAI_MAX_SEQUENCE_BARS];
   double pos_bias[FXAI_MAX_SEQUENCE_BARS];
};

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
   FXAI_SequenceBufferReset(buffer);

   int cap = max_steps;
   if(cap <= 0 || cap > FXAI_MAX_SEQUENCE_BARS)
      cap = FXAI_MAX_SEQUENCE_BARS;

   int usable_window = window_size;
   if(usable_window < 0) usable_window = 0;
   if(usable_window > cap - 1)
      usable_window = cap - 1;

   for(int src=usable_window - 1; src>=0; src--)
      FXAI_SequenceBufferPushRow(buffer, window, src);
   FXAI_SequenceBufferPush(buffer, current_x);

   if(normalize && buffer.len > 1)
      FXAI_TensorNormalizeSequence(buffer.data, buffer.len);
   FXAI_SequenceBufferBuildMask(buffer);
   FXAI_SequenceBufferBuildPositionalBias(buffer);
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
