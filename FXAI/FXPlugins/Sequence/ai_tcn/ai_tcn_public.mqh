   CFXAIAITCN(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TCN; }
   virtual string AIName(void) const { return "ai_tcn"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_CONVOLUTIONAL, caps, 16, 128);

   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(FXAI_TCN_HIST, 72);
   }

   void ForwardSequenceContext(const double &x[],
                               double &h_last[])
   {
      int saved_ptr = m_ptr;
      int saved_hist_len = m_hist_len;
      double saved_stream[FXAI_TCN_MAX_LAYERS + 1][FXAI_TCN_HIST][FXAI_AI_MLP_HIDDEN];
      double saved_mid[FXAI_TCN_MAX_LAYERS][FXAI_TCN_HIST][FXAI_AI_MLP_HIDDEN];

      for(int s=0; s<=FXAI_TCN_MAX_LAYERS; s++)
         for(int t=0; t<FXAI_TCN_HIST; t++)
            for(int c=0; c<FXAI_AI_MLP_HIDDEN; c++)
               saved_stream[s][t][c] = m_hist_stream[s][t][c];
      for(int l=0; l<FXAI_TCN_MAX_LAYERS; l++)
         for(int t=0; t<FXAI_TCN_HIST; t++)
            for(int c=0; c<FXAI_AI_MLP_HIDDEN; c++)
               saved_mid[l][t][c] = m_hist_mid[l][t][c];

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_CONVOLUTIONAL, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      BuildChronologicalSequenceTensorConfigured(x, seq_cfg, seq, seq_len);

      ResetHistory();
      for(int t=0; t<seq_len; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         ForwardStep(x_step, true, false, -1, h_last);
      }

      m_ptr = saved_ptr;
      m_hist_len = saved_hist_len;
      for(int s=0; s<=FXAI_TCN_MAX_LAYERS; s++)
         for(int t=0; t<FXAI_TCN_HIST; t++)
            for(int c=0; c<FXAI_AI_MLP_HIDDEN; c++)
               m_hist_stream[s][t][c] = saved_stream[s][t][c];
      for(int l=0; l<FXAI_TCN_MAX_LAYERS; l++)
         for(int t=0; t<FXAI_TCN_HIST; t++)
            for(int c=0; c<FXAI_AI_MLP_HIDDEN; c++)
               m_hist_mid[l][t][c] = saved_mid[l][t][c];
   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double h_last[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, h_last);

      double logits[FXAI_TCN_CLASS_COUNT];
      double probs[FXAI_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);

      double den = probs[FXAI_TCN_BUY] + probs[FXAI_TCN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TCN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_TCN_SKIP], 0.0, 1.0);
      class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;

      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      expected_move_points = (MathAbs(mu) + 0.25 * sigma + 0.10 * iqr) * active;
      if(expected_move_points <= 0.0)
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double h_last[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, h_last);
      double logits[FXAI_TCN_CLASS_COUNT], probs[FXAI_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);
      double den = probs[FXAI_TCN_BUY] + probs[FXAI_TCN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TCN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double active = FXAI_Clamp(1.0 - probs[FXAI_TCN_SKIP], 0.0, 1.0);
      out.class_probs[(int)FXAI_LABEL_BUY] = p_dir_cal * active;
      out.class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_dir_cal) * active;
      out.class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      NormalizeClassDistribution(out.class_probs);
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 30.0);
      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * MathAbs(q75 - q25)) * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * active);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * active);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(xa,
                              out.move_mean_points,
                              FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                              out.reliability,
                              out.confidence,
                              bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust,
                              out);
      return true;
   }


   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
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
      m_quality_heads.Reset();
      ResetHistory();
      ResetSequence();
      ResetOptimizers();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      ApplyConfig(hp);
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      ApplyConfig(h);

      double sw = MoveSampleWeight(xa, move_points);
      sw = FXAI_Clamp(sw, 0.25, 4.00);
      m_quality_heads.Update(xa,
                             sw,
                             TargetMFEPoints(),
                             FXAI_Clamp(TargetMAEPoints() / MathMax(TargetMFEPoints() + 0.10, 0.10), 0.0, 1.0),
                             TargetHitTimeFrac(),
                             TargetPathRisk(),
                             TargetFillRisk(),
                             TargetMaskedStep(),
                             TargetNextVol(),
                             TargetRegimeShift(),
                             TargetContextLead(),
                             h.lr,
                             h.l2);
      double cost = InputCostProxyPoints(xa);
      int cls = MapClass(y, xa, move_points);

      AppendSequenceSample(cls, xa, move_points, cost, sw, h);

      if(m_seq_len >= FXAI_TCN_TBPTT)
         TrainTBPTT(h);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double h_last[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, h_last);

      double logits[FXAI_TCN_CLASS_COUNT];
      double probs[FXAI_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);

      double den = probs[FXAI_TCN_BUY] + probs[FXAI_TCN_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_TCN_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);

      double p_up = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_TCN_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double h_last[FXAI_AI_MLP_HIDDEN];
      ForwardSequenceContext(x, h_last);

      double logits[FXAI_TCN_CLASS_COUNT];
      double probs[FXAI_TCN_CLASS_COUNT];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      ComputeHeads(h_last, logits, probs, mu, logv, q25, q75);

      double sigma = MathExp(0.5 * logv);
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);

      // Distribution-aware expected move (absolute edge scale).
      double ev = MathAbs(mu) + 0.25 * sigma + 0.10 * iqr;
      ev *= FXAI_Clamp(1.0 - probs[FXAI_TCN_SKIP], 0.0, 1.0);

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
