   CFXAIAITFT(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_TFT; }
   virtual string AIName(void) const { return "ai_tft"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 16, 128);

   }

   void BuildWindowAwareInput(const double &x[], double &xa[]) const
   {
      CopyCurrentInputClipped(x, xa);
   }

   int SequenceContextSpan(void) const
   {
      return ContextSequenceCap(FXAI_TFT_SEQ, 72);
   }

   void ForwardSequenceContext(const double &x[],
                               const bool use_shadow,
                               double &p_sell,
                               double &p_buy,
                               double &p_skip,
                               double &mu,
                               double &logv,
                               double &q25,
                               double &q75)
   {
      int saved_hist_len = m_hist_len;
      int saved_hist_ptr = m_hist_ptr;
      double saved_hist[FXAI_TFT_SEQ][FXAI_AI_WEIGHTS];
      for(int t=0; t<FXAI_TFT_SEQ; t++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            saved_hist[t][k] = m_hist_x[t][k];

      double seq[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int seq_len = 0;
      int seq_mask[];
      double seq_pos_bias[];
      FXAITensorDims dims = TensorContextDims(FXAI_SEQ_STYLE_TRANSFORMER, SequenceContextSpan());
      FXAISequenceRuntimeConfig seq_cfg = TensorSequenceRuntimeConfig(dims, true, true);
      double k_fast[3] = {1.00, 0.00, -1.00};
      double k_slow[5] = {0.10, 0.20, 0.40, 0.20, 0.10};
      BuildSequenceBlockSequence(x, dims, seq_cfg, k_fast, 3, k_slow, 5, seq, seq_len, seq_mask, seq_pos_bias);

      ResetHistory();
      for(int t=0; t<seq_len - 1; t++)
      {
         double x_step[FXAI_AI_WEIGHTS];
         FXAI_SequenceCopyRow(seq, t, x_step);
         PushHistory(x_step);
      }

      double x_cur[FXAI_AI_WEIGHTS];
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_cur[k] = (seq_len > 0 ? seq[seq_len - 1][k] : 0.0);
      ForwardInference(x_cur, use_shadow, p_sell, p_buy, p_skip, mu, logv, q25, q75);

      m_hist_len = saved_hist_len;
      m_hist_ptr = saved_hist_ptr;
      for(int t=0; t<FXAI_TFT_SEQ; t++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_hist_x[t][k] = saved_hist[t][k];
   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double ps = 0.3333, pb = 0.3333, pk = 0.3333;
      double mu = 0.0, logv = MathLog(1.0), q25 = 0.0, q75 = 0.0;

      ForwardSequenceContext(x, m_shadow_ready, ps, pb, pk, mu, logv, q25, q75);
      double den = pb + ps;
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = pb / den;

      double p_cal = CalibrateProb(p_dir_raw);
      int sess = SessionBucket(ResolveContextTime());
      p_cal = ApplySessionCalibration(sess, p_cal);

      double active = FXAI_Clamp(1.0 - pk, 0.0, 1.0);
      if(pk > m_thr_skip) active *= 0.25;
      else if(p_cal < m_thr_buy && p_cal > m_thr_sell) active *= 0.35;

      class_probs[(int)FXAI_LABEL_BUY] = p_cal * active;
      class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_cal) * active;
      class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);
      expected_move_points = MathMax(0.0, (0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr) * active);
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
      double ps = 0.3333, pb = 0.3333, pk = 0.3333;
      double mu = 0.0, logv = MathLog(1.0), q25 = 0.0, q75 = 0.0;
      ForwardSequenceContext(x, m_shadow_ready, ps, pb, pk, mu, logv, q25, q75);
      double den = pb + ps;
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = pb / den;
      double p_cal = ApplySessionCalibration(SessionBucket(ResolveContextTime()), CalibrateProb(p_dir_raw));
      double active = FXAI_Clamp(1.0 - pk, 0.0, 1.0);
      if(pk > m_thr_skip) active *= 0.25;
      else if(p_cal < m_thr_buy && p_cal > m_thr_sell) active *= 0.35;
      out.class_probs[(int)FXAI_LABEL_BUY] = p_cal * active;
      out.class_probs[(int)FXAI_LABEL_SELL] = (1.0 - p_cal) * active;
      out.class_probs[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      NormalizeClassDistribution(out.class_probs);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double ev = MathMax(0.0, (0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * MathAbs(q75 - q25)) * active);
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
      m_shadow_ready = false;
      m_step = 0;
      m_seen = 0;
      m_adam_t = 0;
      m_symbol_hash = SymbolHash(_Symbol);
      m_quality_heads.Reset();

      ResetNorm();
      ResetHistory();
      ResetTrainBuffer();
      ResetWalkForward();
      ResetSessionCalibration();

      m_thr_buy = 0.62;
      m_thr_sell = 0.38;
      m_thr_skip = 0.58;
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitParams();
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }
