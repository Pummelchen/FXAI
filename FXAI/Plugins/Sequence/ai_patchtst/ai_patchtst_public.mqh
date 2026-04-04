   CFXAIAIPatchTST(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_PATCHTST; }
   virtual string AIName(void) const { return "ai_patchtst"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 256);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_adam_t = 0;
      ResetSequence();
      ResetInputNorm();
      for(int c=0; c<FXAI_PTST_CLASS_COUNT; c++)
         m_cls_ema[c] = 1.0;
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      int xn = ArraySize(x);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         double v = (i < xn && MathIsValidNumber(x[i]) ? x[i] : 0.0);
         xa[i] = (i == 0 ? 1.0 : FXAI_ClipSym(v, 8.0));
      }
      int win_n = CurrentWindowSize();
      if(win_n <= 1) return;
      double mean1 = CurrentWindowFeatureMean(0);
      double mean2 = CurrentWindowFeatureMean(1);
      double mean6 = CurrentWindowFeatureMean(5);
      double first1 = CurrentWindowValue(0, 1);
      double last1  = CurrentWindowValue(win_n - 1, 1);
      double first2 = CurrentWindowValue(0, 2);
      double last2  = CurrentWindowValue(win_n - 1, 2);
      double vol1 = 0.0;
      for(int i=0; i<win_n; i++)
      {
         double d = CurrentWindowValue(i, 1) - mean1;
         vol1 += d * d;
      }
      vol1 = MathSqrt(vol1 / (double)win_n);

      double attn[];
      double conv_fast[];
      double conv_slow[];
      double block[];
      double k_fast[3] = {0.58, 0.27, 0.15};
      double k_slow[5] = {0.34, 0.24, 0.18, 0.14, 0.10};
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_TRANSFORMER, SequenceContextSpan());
      dims.patch_size = MathMax(dims.patch_size, 4);
      dims.stride = MathMax(dims.stride, 2);
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildSequenceBlockSummaries(x, dims, seq_cfg, k_fast, 3, k_slow, 5, attn, conv_fast, conv_slow, block);

      xa[1] = FXAI_ClipSym(0.46 * xa[1] + 0.18 * mean1 + 0.10 * (first1 - last1) + 0.10 * attn[1] + 0.08 * conv_fast[1] + 0.08 * block[1], 8.0);
      xa[2] = FXAI_ClipSym(0.46 * xa[2] + 0.18 * mean2 + 0.10 * (first2 - last2) + 0.10 * attn[2] + 0.08 * conv_fast[2] + 0.08 * block[2], 8.0);
      xa[6] = FXAI_ClipSym(0.50 * xa[6] + 0.25 * vol1 + 0.08 * MathAbs(attn[6]) + 0.07 * MathAbs(conv_slow[6]) + 0.10 * MathAbs(block[6]), 8.0);
      xa[7] = FXAI_ClipSym(0.44 * xa[7] + 0.18 * mean6 + 0.14 * attn[7] + 0.12 * conv_slow[7] + 0.12 * block[7], 8.0);
      xa[10] = FXAI_ClipSym(0.68 * xa[10] + 0.12 * attn[10] + 0.08 * conv_fast[10] + 0.12 * block[10], 8.0);
      xa[11] = FXAI_ClipSym(0.68 * xa[11] + 0.12 * attn[11] + 0.08 * conv_slow[11] + 0.12 * block[11], 8.0);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(FXAI_PTST_SEQ, 72);
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double rep[FXAI_PTST_D_MODEL];
      double p_raw[FXAI_PTST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double patch_stat[FXAI_AI_FEATURES][FXAI_PTST_PATCH_LEN];
      double layer_in_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_MODEL];
      double layer_ctx_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_MODEL];
      double layer_ff_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_FF];
      int token_count = 0;

      ForwardPass(xa,
                  false,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  patch_stat,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  token_count);

      Calibrate3(p_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, class_probs[(int)FXAI_LABEL_SKIP]);
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         expected_move_points = 0.70 * ev + 0.30 * m_move_ema_abs;
      else if(ev > 0.0)
         expected_move_points = ev;
      else
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      if(expected_move_points < 0.0)
         expected_move_points = 0.0;

      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double rep[FXAI_PTST_D_MODEL];
      double p_raw[FXAI_PTST_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double patch_stat[FXAI_AI_FEATURES][FXAI_PTST_PATCH_LEN];
      double layer_in_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_MODEL];
      double layer_ctx_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_MODEL];
      double layer_ff_mean[FXAI_PTST_LAYERS][FXAI_PTST_D_FF];
      int token_count = 0;
      ForwardPass(xa, false, rep, p_raw, mu, logv, q25, q75,
                  patch_stat, layer_in_mean, layer_ctx_mean, layer_ff_mean, token_count);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, out.class_probs[(int)FXAI_LABEL_SKIP]);
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * (1.0 - out.class_probs[(int)FXAI_LABEL_SKIP]) + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

