   CFXAIAIS4(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_S4; }
   virtual string AIName(void) const { return "ai_s4"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_STATE_SPACE, caps, 16, 256);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);

      double mean_combo = 0.50 * mu1 + 0.30 * mu3 + 0.20 * mu5;
      if(mean_combo < 0.0) mean_combo = 0.0;
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 20.0);
      expected_move_points = mean_combo + 0.30 * sigma;
      if(expected_move_points <= 0.0)
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      double min_move = MathMax(ResolveMinMovePoints(), 0.10);
      double cost = MathMax(ResolveCostPoints(x), 0.0);
      double active = FXAI_Clamp((expected_move_points - 0.35 * cost) / MathMax(min_move, 0.10), 0.0, 1.0);
      double p_dir = FXAI_Clamp(p_local, 0.001, 0.999);
      double p_raw3[3];
      p_raw3[(int)FXAI_LABEL_BUY] = p_dir * active;
      p_raw3[(int)FXAI_LABEL_SELL] = (1.0 - p_dir) * active;
      p_raw3[(int)FXAI_LABEL_SKIP] = 1.0 - active;
      m_cal3.Calibrate(p_raw3, class_probs);
      NormalizeClassDistribution(class_probs);
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      EnsureInitialized(hp);

      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);

      double probs[3];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return false;
      for(int c=0; c<3; c++)
         out.class_probs[c] = probs[c];

      double mean_combo = MathMax(0.0, 0.50 * mu1 + 0.30 * mu3 + 0.20 * mu5);
      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 20.0);
      out.move_mean_points = MathMax(ev, (m_move_ready ? m_move_ema_abs : mean_combo));
      out.move_q25_points = MathMax(0.0, mean_combo - 0.60 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, mean_combo);
      out.move_q75_points = MathMax(out.move_q50_points, mean_combo + 0.60 * sigma);
      out.confidence = FXAI_Clamp(0.60 * MathMax(probs[(int)FXAI_LABEL_BUY], probs[(int)FXAI_LABEL_SELL]) + 0.20 * (1.0 - probs[(int)FXAI_LABEL_SKIP]) + 0.20 * (1.0 - FXAI_Clamp(m_margin_ema / 2.0, 0.0, 1.0)), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.55 + 0.20 * (m_margin_ready ? 1.0 - MathAbs(m_prob_bias) / 2.0 : 0.0) + 0.15 * (m_move_ready ? 1.0 : 0.0) + 0.10 * (1.0 - FXAI_Clamp(MathAbs(m_prob_scale - 1.0), 0.0, 1.0)), 0.0, 1.0);
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
      m_train_steps = 0;
      m_seen_updates = 0;
      ResetNormStats();
      ResetState();
      ResetBatch();
      ResetOptimizers();

      m_margin_ready = false;
      m_margin_ema = 1.0;
      m_prob_scale = 1.0;
      m_prob_bias = 0.0;
      m_quality_heads.Reset();
      m_cal3.Reset();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights();
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
      m_seen_updates++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls == (int)FXAI_LABEL_SKIP) return;
      int y_dir = (cls == (int)FXAI_LABEL_BUY ? 1 : 0);
      double cls_w = 1.0;

      // Controlled reset policy to prevent state bleed in long runs and shock bars.
      if((m_seen_updates % 512) == 0 ||
         MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
      {
         ResetState();
         ResetBatch();
      }

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = FXAI_Clamp(MoveSampleWeight(xa, move_points) * cls_w, 0.10, 4.00);
      m_quality_heads.Update(xa,
                             w,
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
      double pre_probs[3];
      double pre_ev = 0.0;
      if(PredictModelCore(x, h, pre_probs, pre_ev))
         m_cal3.Update(pre_probs, cls, w, h.lr);
      AppendBatch(y_dir, xa, move_points, w);

      if(m_batch_size >= FXAI_S4_TBPTT)
         TrainBatch(h);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double st_re[FXAI_AI_MLP_HIDDEN], st_im[FXAI_AI_MLP_HIDDEN];
      double z = 0.0, p_local = 0.5;
      double mu1 = 0.0, mu3 = 0.0, mu5 = 0.0, logv = 0.0;
      ForwardSequenceContext(x, st_re, st_im, z, p_local, mu1, mu3, mu5, logv);
      return CalibrateProb(p_local);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0)
         return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
