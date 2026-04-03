   CFXAIAITimesFM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TIMESFM; }
   virtual string AIName(void) const { return "ai_timesfm"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 32, 256);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_step = 0;
      m_adam_t = 0;
      m_obs_step = 0;
      m_last_m1_train_bar = 0;
      ResetSequence();
      ResetInputNorm();
      ResetFeatureStats();
      for(int c=0; c<FXAI_TFM_CLASS_COUNT; c++)
         m_cls_ema[c] = 1.0;
      for(int g=0; g<8; g++)
      {
         m_opt_m[g] = 0.0;
         m_opt_v[g] = 0.0;
      }
      m_mem_ptr = 0;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      InitWeights();
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double rep[FXAI_TFM_D_MODEL];
      double p_raw[FXAI_TFM_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TFM_HORIZONS];
      double patch_stat[FXAI_AI_FEATURES][FXAI_TFM_PATCH_LEN];
      double token_hist[FXAI_TFM_CODEBOOK];
      int token_target = 0;
      double layer_in_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ctx_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ff_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_FF];
      double mem_attn[FXAI_TFM_MEMORY];
      int token_count = 0;

      ForwardPass(xa,
                  false,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  mu_h,
                  patch_stat,
                  token_hist,
                  token_target,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  mem_attn,
                  token_count);

      Calibrate3(p_raw, class_probs);

      double ev = ExpectedMoveFromHeads(mu, logv, q25, q75, mu_h, class_probs[(int)FXAI_LABEL_SKIP]);
      double h_mean = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
         h_mean += MathAbs(mu_h[h]);
      h_mean /= (double)FXAI_TFM_HORIZONS;
      double h_disp = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
         h_disp += MathAbs(MathAbs(mu_h[h]) - h_mean);
      h_disp /= (double)FXAI_TFM_HORIZONS;

      double tok_prob[FXAI_TFM_CODEBOOK];
      int tok_top = 0;
      TokenHead(rep, tok_prob, tok_top);
      double tok_entropy = 0.0;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         double pt = FXAI_Clamp(tok_prob[t], 1e-9, 1.0);
         tok_entropy += -pt * MathLog(pt);
      }
      double tok_conf = 1.0 - FXAI_Clamp(tok_entropy / MathLog((double)FXAI_TFM_CODEBOOK), 0.0, 1.0);
      double horizon_cons = 1.0 - FXAI_Clamp(h_disp / MathMax(h_mean + 0.25, 0.25), 0.0, 1.0);
      ev *= (0.78 + 0.12 * tok_conf + 0.10 * horizon_cons);
      expected_move_points = ev;
      if(expected_move_points <= 0.0)
         expected_move_points = MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);

      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double rep[FXAI_TFM_D_MODEL];
      double p_raw[FXAI_TFM_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      double mu_h[FXAI_TFM_HORIZONS];
      double patch_stat[FXAI_AI_FEATURES][FXAI_TFM_PATCH_LEN];
      double token_hist[FXAI_TFM_CODEBOOK];
      int token_target = 0;
      double layer_in_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ctx_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_MODEL];
      double layer_ff_mean[FXAI_TFM_LAYERS][FXAI_TFM_D_FF];
      double mem_attn[FXAI_TFM_MEMORY];
      int token_count = 0;
      ForwardPass(xa,
                  false,
                  rep,
                  p_raw,
                  mu,
                  logv,
                  q25,
                  q75,
                  mu_h,
                  patch_stat,
                  token_hist,
                  token_target,
                  layer_in_mean,
                  layer_ctx_mean,
                  layer_ff_mean,
                  mem_attn,
                  token_count);

      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = ExpectedMoveFromHeads(mu, logv, q25, q75, mu_h, out.class_probs[(int)FXAI_LABEL_SKIP]);
      out.move_q25_points = MathMax(0.0, q25);
      out.move_q50_points = MathMax(out.move_q25_points, MathAbs(mu));
      out.move_q75_points = MathMax(out.move_q50_points, q75);

      double tok_prob[FXAI_TFM_CODEBOOK];
      int tok_top = 0;
      TokenHead(rep, tok_prob, tok_top);
      double tok_entropy = 0.0;
      for(int t=0; t<FXAI_TFM_CODEBOOK; t++)
      {
         double pt = FXAI_Clamp(tok_prob[t], 1e-9, 1.0);
         tok_entropy += -pt * MathLog(pt);
      }
      double tok_conf = 1.0 - FXAI_Clamp(tok_entropy / MathLog((double)FXAI_TFM_CODEBOOK), 0.0, 1.0);
      double h_mean = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
         h_mean += MathAbs(mu_h[h]);
      h_mean /= (double)FXAI_TFM_HORIZONS;
      double h_disp = 0.0;
      for(int h=0; h<FXAI_TFM_HORIZONS; h++)
         h_disp += MathAbs(MathAbs(mu_h[h]) - h_mean);
      h_disp /= (double)FXAI_TFM_HORIZONS;
      double horizon_cons = 1.0 - FXAI_Clamp(h_disp / MathMax(h_mean + 0.25, 0.25), 0.0, 1.0);
      out.confidence = FXAI_Clamp(0.55 * tok_conf + 0.45 * (1.0 - out.class_probs[(int)FXAI_LABEL_SKIP]), 0.0, 1.0);
      double mem_peak = 0.0;
      for(int m=0; m<FXAI_TFM_MEMORY; m++)
         if(mem_attn[m] > mem_peak) mem_peak = mem_attn[m];
      out.reliability = FXAI_Clamp(0.40 * mem_peak + 0.30 * tok_conf + 0.30 * horizon_cons, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      if(out.move_mean_points <= 0.0)
         out.move_mean_points = MathMax(PredictMoveHeadRaw(xa), m_move_ema_abs);
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                0.55 * horizon_cons + 0.45 * tok_conf,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

