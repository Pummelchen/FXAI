   CFXAIAIAutoformer(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_AUTOFORMER; }
   virtual string AIName(void) const { return "ai_autoformer"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 256);

   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, false, false, -1, fin, p_raw, mu, lv, q25, q75);

      int sess = SessionBucket(ResolveContextTime());
      Calibrate3(sess, p_raw, class_probs);

      double sigma = MathExp(0.5 * FXAI_Clamp(lv, -4.0, 4.0));
      sigma = FXAI_Clamp(sigma, 0.05, 30.0);
      double iqr = MathAbs(q75 - q25);
      double active = FXAI_Clamp(1.0 - class_probs[FXAI_AF_SKIP], 0.0, 1.0);

      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * iqr) * active;
      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0)
         expected_move_points = 0.65 * ev + 0.35 * m_move_ema_abs;
      else if(ev > 0.0)
         expected_move_points = ev;
      else
         expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, false, false, -1, fin, p_raw, mu, lv, q25, q75);

      int sess = SessionBucket(ResolveContextTime());
      Calibrate3(sess, p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);

      double sigma = FXAI_Clamp(MathExp(0.5 * FXAI_Clamp(lv, -4.0, 4.0)), 0.05, 30.0);
      double active = FXAI_Clamp(1.0 - out.class_probs[FXAI_AF_SKIP], 0.0, 1.0);
      double ev = (MathAbs(mu) + 0.25 * sigma + 0.10 * MathAbs(q75 - q25)) * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;

      double ql = MathAbs(q25) * active;
      double qh = MathAbs(q75) * active;
      if(ql > qh) { double t = ql; ql = qh; qh = t; }

      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, ql);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, qh);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[FXAI_AF_BUY], out.class_probs[FXAI_AF_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(xa,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_shadow_ready = false;
      m_printed_reco = false;
      m_step = 0;
      m_adam_t = 0;

      ResetNorm();
      ResetSequence();
      ResetTrainBuffer();
      ResetCalibrator3();
      ZeroMoments();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
      PrintRecommendationOnce();
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

