   void CopyCurrentInputClipped(const double &x[],
                                double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }
   }

   FXAITensorDims TensorContextDims(const int style,
                                    const int max_steps = FXAI_MAX_SEQUENCE_BARS) const
   {
      int cap = MathMax(MathMin(max_steps, FXAI_MAX_SEQUENCE_BARS), 4);
      int model_dim = 16;
      int hidden_dim = FXAI_AI_MLP_HIDDEN;
      int heads = 2;
      int stride = 1;
      int patch = 1;
      int dilation = 1;
      double pos_penalty = 0.06;
      switch(style)
      {
         case FXAI_SEQ_STYLE_RECURRENT:
            model_dim = MathMin(18, FXAI_AI_FEATURES);
            heads = 2;
            break;
         case FXAI_SEQ_STYLE_CONVOLUTIONAL:
            model_dim = MathMin(18, FXAI_AI_FEATURES);
            patch = 2;
            break;
         case FXAI_SEQ_STYLE_TRANSFORMER:
            model_dim = MathMin(24, FXAI_AI_FEATURES);
            heads = 4;
            patch = (cap >= 24 ? 2 : 1);
            break;
         case FXAI_SEQ_STYLE_STATE_SPACE:
            model_dim = MathMin(20, FXAI_AI_FEATURES);
            dilation = 2;
            break;
         case FXAI_SEQ_STYLE_WORLD:
            model_dim = MathMin(22, FXAI_AI_FEATURES);
            heads = 4;
            stride = (cap >= 32 ? 2 : 1);
            pos_penalty = 0.04;
            break;
         default:
            model_dim = MathMin(16, FXAI_AI_FEATURES);
            break;
      }
      if(m_ctx_horizon_minutes >= 60)
         stride = MathMax(stride, 2);
      return FXAI_TensorMakeDims(model_dim, hidden_dim, heads, cap, stride, patch, dilation, pos_penalty);
   }

   FXAISequenceRuntimeConfig TensorSequenceRuntimeConfig(const FXAITensorDims &dims,
                                                         const bool normalize = true,
                                                         const bool include_current = true) const
   {
      return FXAI_SequenceRuntimeMakeConfig(dims.seq_cap,
                                            dims.stride,
                                            dims.patch_size,
                                            normalize,
                                            include_current,
                                            dims.pos_step_penalty);
   }

   void BuildChronologicalSequenceTensorConfigured(const double &x[],
                                                   const FXAISequenceRuntimeConfig &cfg,
                                                   double &seq[][FXAI_AI_WEIGHTS],
                                                   int &seq_len) const
   {
      FXAISequenceBuffer buffer;
      FXAI_SequenceBufferLoadWindowConfig(buffer,
                                          x,
                                          m_ctx_window,
                                          m_ctx_window_size,
                                          cfg);
      FXAI_SequenceBufferExport(buffer, seq, seq_len);
   }

   void BuildChronologicalSequenceTensorCapped(const double &x[],
                                               const int max_steps,
                                               double &seq[][FXAI_AI_WEIGHTS],
                                               int &seq_len,
                                               const bool normalize = true) const
   {
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_GENERIC, max_steps);
      FXAISequenceRuntimeConfig cfg = TensorSequenceRuntimeConfig(dims, normalize, true);
      BuildChronologicalSequenceTensorConfigured(x, cfg, seq, seq_len);
   }

   void BuildPackedSequenceTensorCapped(const double &x[],
                                        const int max_steps,
                                        double &seq[][FXAI_AI_WEIGHTS],
                                        int &seq_len,
                                        int &mask[],
                                        double &pos_bias[],
                                        const bool normalize = true) const
   {
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_GENERIC, max_steps);
      FXAISequenceRuntimeConfig cfg = TensorSequenceRuntimeConfig(dims, normalize, true);
      BuildPackedSequenceTensorConfigured(x, cfg, seq, seq_len, mask, pos_bias);
   }

   void BuildPackedSequenceTensorConfigured(const double &x[],
                                            const FXAISequenceRuntimeConfig &cfg,
                                            double &seq[][FXAI_AI_WEIGHTS],
                                            int &seq_len,
                                            int &mask[],
                                            double &pos_bias[]) const
   {
      FXAISequenceBuffer buffer;
      FXAI_SequenceBufferLoadWindowConfig(buffer,
                                          x,
                                          m_ctx_window,
                                          m_ctx_window_size,
                                          cfg);
      FXAI_SequenceBufferPreparePacked(buffer, seq, seq_len, mask, pos_bias);
   }

   void BuildSequenceBlockSummaries(const double &x[],
                                    const FXAITensorDims &dims,
                                    const FXAISequenceRuntimeConfig &cfg,
                                    const double &kernel_fast[],
                                    const int kernel_fast_size,
                                    const double &kernel_slow[],
                                    const int kernel_slow_size,
                                    double &attn_summary[],
                                    double &conv_fast_summary[],
                                    double &conv_slow_summary[],
                                    double &block_summary[]) const
   {
      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double attn_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double conv_fast_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double conv_slow_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double block_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      BuildPackedSequenceTensorConfigured(x, cfg, seq, seq_len, seq_mask, seq_pos_bias);
      ArrayResize(attn_summary, FXAI_AI_WEIGHTS);
      ArrayResize(conv_fast_summary, FXAI_AI_WEIGHTS);
      ArrayResize(conv_slow_summary, FXAI_AI_WEIGHTS);
      ArrayResize(block_summary, FXAI_AI_WEIGHTS);
      ArrayInitialize(attn_summary, 0.0);
      ArrayInitialize(conv_fast_summary, 0.0);
      ArrayInitialize(conv_slow_summary, 0.0);
      ArrayInitialize(block_summary, 0.0);
      if(seq_len <= 1)
         return;

      FXAI_ModuleSequenceAttentionBlock(seq, seq_len, seq_mask, seq_pos_bias, dims, attn_seq);
      FXAI_ModuleSequenceConvBlock(seq, seq_len, kernel_fast, kernel_fast_size, dims, conv_fast_seq);
      FXAI_ModuleSequenceConvBlock(seq, seq_len, kernel_slow, kernel_slow_size, dims, conv_slow_seq);
      FXAI_ModuleSequenceResidualNormFFN(attn_seq, seq_len, dims, block_seq);
      FXAI_ModuleSequencePool(attn_seq, seq_len, dims, attn_summary);
      FXAI_ModuleSequencePool(conv_fast_seq, seq_len, dims, conv_fast_summary);
      FXAI_ModuleSequencePool(conv_slow_seq, seq_len, dims, conv_slow_summary);
      FXAI_ModuleSequencePool(block_seq, seq_len, dims, block_summary);
   }

   void BuildTensorEncodedInput(const double &x[],
                                const int style,
                                double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      FXAITensorDims dims = TensorContextDims(style, FXAI_MAX_SEQUENCE_BARS);
      FXAISequenceRuntimeConfig cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildPackedSequenceTensorConfigured(x, cfg, seq, seq_len, seq_mask, seq_pos_bias);
      if(seq_len <= 1)
         return;

      double attn_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double conv_slow_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double conv_fast_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double block_seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      double attn[];
      double conv_slow[];
      double conv_fast[];
      double block[];
      double kernel_slow[3] = {0.20, 0.60, 0.20};
      double kernel_fast[3] = {1.00, 0.00, -1.00};
      FXAI_ModuleSequenceAttentionBlock(seq, seq_len, seq_mask, seq_pos_bias, dims, attn_seq);
      FXAI_ModuleSequenceConvBlock(seq, seq_len, kernel_slow, 3, dims, conv_slow_seq);
      FXAI_ModuleSequenceConvBlock(seq, seq_len, kernel_fast, 3, dims, conv_fast_seq);
      FXAI_ModuleSequenceResidualNormFFN(attn_seq, seq_len, dims, block_seq);
      FXAI_ModuleSequencePool(attn_seq, seq_len, dims, attn);
      FXAI_ModuleSequencePool(conv_slow_seq, seq_len, dims, conv_slow);
      FXAI_ModuleSequencePool(conv_fast_seq, seq_len, dims, conv_fast);
      FXAI_ModuleSequencePool(block_seq, seq_len, dims, block);

      int last = seq_len - 1;
      int prev = MathMax(last - 1, 0);
      int mid = MathMax(seq_len / 2, 0);
      int root = 0;
      double w_cur = 0.40;
      double w_prev = 0.20;
      double w_mid = 0.10;
      double w_att = 0.15;
      double w_slow = 0.10;
      double w_fast = 0.05;

      switch(style)
      {
         case FXAI_SEQ_STYLE_RECURRENT:
            w_cur = 0.34; w_prev = 0.24; w_mid = 0.10; w_att = 0.14; w_slow = 0.10; w_fast = 0.08;
            break;
         case FXAI_SEQ_STYLE_CONVOLUTIONAL:
            w_cur = 0.25; w_prev = 0.18; w_mid = 0.10; w_att = 0.12; w_slow = 0.17; w_fast = 0.18;
            break;
         case FXAI_SEQ_STYLE_TRANSFORMER:
            w_cur = 0.24; w_prev = 0.12; w_mid = 0.14; w_att = 0.28; w_slow = 0.12; w_fast = 0.10;
            break;
         case FXAI_SEQ_STYLE_STATE_SPACE:
            w_cur = 0.28; w_prev = 0.18; w_mid = 0.14; w_att = 0.12; w_slow = 0.18; w_fast = 0.10;
            break;
         case FXAI_SEQ_STYLE_WORLD:
            w_cur = 0.22; w_prev = 0.14; w_mid = 0.18; w_att = 0.24; w_slow = 0.12; w_fast = 0.10;
            break;
         default:
            break;
      }

      for(int k=1; k<FXAI_AI_WEIGHTS; k++)
      {
         double v = w_cur * seq[last][k] +
                    w_prev * seq[prev][k] +
                    w_mid * seq[mid][k] +
                    w_att * attn[k] +
                    w_slow * conv_slow[k] +
                    w_fast * conv_fast[k];
         v += 0.04 * seq[root][k] + 0.10 * block[k];
         xa[k] = FXAI_ClipSym(v, 8.0);
      }
      xa[0] = 1.0;
   }

