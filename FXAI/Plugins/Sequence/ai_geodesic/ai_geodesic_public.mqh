   CFXAIAIGeodesicAttention(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_GEODESICATTENTION; }
   virtual string AIName(void) const { return "ai_geodesic"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_STATEFUL|FXAI_CAP_WINDOW_CONTEXT|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TRANSFORMER, caps, 24, 128);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, false, emb, loc, fin, grp, ctx, gcons, bin, bout);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0;
      double q_all[FXAI_GA_QUANTILES];
      ComputeHeads(fin, logits, probs, mu, logv, q_all);

      Calibrate3(probs, class_probs);

      double geo_mean = 0.0;
      int geo_n = 0;
      for(int b=0; b<FXAI_GA_BLOCKS; b++)
         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            geo_mean += gcons[b][hd];
            geo_n++;
         }
      if(geo_n <= 0) geo_n = 1;
      geo_mean /= (double)geo_n;
      double ev = ExpectedMoveFromHeads(mu, logv, q_all, class_probs[FXAI_GA_SKIP]);
      ev *= (0.82 + 0.18 * (1.0 - FXAI_Clamp(geo_mean / 2.0, 0.0, 1.0)));
      expected_move_points = (ev > 0.0 ? ev : (m_move_ready ? m_move_ema_abs : 0.0));
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
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, false, emb, loc, fin, grp, ctx, gcons, bin, bout);

      double logits[FXAI_GA_CLASS_COUNT];
      double probs[FXAI_GA_CLASS_COUNT];
      double mu = 0.0, logv = 0.0;
      double q_all[FXAI_GA_QUANTILES];
      ComputeHeads(fin, logits, probs, mu, logv, q_all);
      Calibrate3(probs, out.class_probs);

      double sigma = FXAI_Clamp(MathExp(0.5 * logv), 0.05, 50.0);
      double geo_mean = 0.0;
      int geo_n = 0;
      for(int b=0; b<FXAI_GA_BLOCKS; b++)
         for(int hd=0; hd<FXAI_GA_HEADS; hd++)
         {
            geo_mean += gcons[b][hd];
            geo_n++;
         }
      if(geo_n <= 0) geo_n = 1;
      geo_mean /= (double)geo_n;
      double geo_cons = 1.0 - FXAI_Clamp(geo_mean / 2.0, 0.0, 1.0);
      double ev = ExpectedMoveFromHeads(mu, logv, q_all, out.class_probs[FXAI_GA_SKIP]);
      out.move_mean_points = (ev > 0.0 ? ev * (0.82 + 0.18 * geo_cons) : (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, q_all[1]);
      out.move_q50_points = MathMax(out.move_q25_points, q_all[3]);
      out.move_q75_points = MathMax(out.move_q50_points, q_all[5]);
      if(out.move_q75_points <= 0.0)
      {
         out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
         out.move_q50_points = out.move_mean_points;
         out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      }

      double dir_conf = MathMax(out.class_probs[FXAI_GA_BUY], out.class_probs[FXAI_GA_SELL]);
      out.confidence = FXAI_Clamp(0.50 * dir_conf + 0.15 * (1.0 - out.class_probs[FXAI_GA_SKIP]) + 0.20 * (1.0 - FXAI_Clamp(m_val_ece_slow, 0.0, 1.0)) + 0.15 * geo_cons, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.18 * (1.0 - FXAI_Clamp(m_val_ece_fast, 0.0, 1.0)) + 0.12 * (1.0 - FXAI_Clamp(m_val_nll_fast / 2.5, 0.0, 1.0)) + 0.10 * (m_quality_degraded ? -0.5 : 0.5) + 0.15 * geo_cons, 0.0, 1.0);
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
      m_step = 0;
      ResetSequence();
      ResetTrainBuffer();
      ResetInputNorm();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) InitWeights();
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? FXAI_GA_BUY : FXAI_GA_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);

      // Controlled reset policy to avoid state bleed in non-stationary jumps.
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);

      if((m_step % 768) == 0) ResetSequence();
      if(MathAbs(xa[1]) > 8.0 || MathAbs(xa[2]) > 8.0) ResetSequence();

      UpdateInputStats(xa);

      int cls = MapClass(y, xa, move_points);
      double sw = MoveSampleWeight(xa, move_points);
      sw = FXAI_Clamp(sw, 0.20, 8.00);
      double cost = InputCostProxyPoints(x);

      AppendTrainSample(cls, xa, move_points, cost, sw);
      ReplayPush(cls, xa, move_points, cost, sw);

      TrainTBPTT(h);
      TrainReplayMiniBatch(h);

      // Update live state with most recent bar after isolated TBPTT step.
      double emb[FXAI_GA_D_MODEL];
      double loc[FXAI_GA_D_MODEL];
      double fin[FXAI_GA_D_MODEL];
      double grp[3];
      double ctx[FXAI_GA_BLOCKS][FXAI_GA_HEADS][FXAI_GA_D_HEAD];
      double gcons[FXAI_GA_BLOCKS][FXAI_GA_HEADS];
      double bin[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      double bout[FXAI_GA_BLOCKS][FXAI_GA_D_MODEL];
      ForwardStep(xa, true, emb, loc, fin, grp, ctx, gcons, bin, bout);
      UpdateNativeQualityHeads(xa, sw, h.lr, h.l2);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double expected_move = 0.0;
      if(!PredictModelCore(x, hp, probs, expected_move)) return 0.5;
      double den = probs[FXAI_GA_BUY] + probs[FXAI_GA_SELL];
      if(den < 1e-9) den = 1e-9;
      return FXAI_Clamp(probs[FXAI_GA_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      FXAIAIModelOutputV4 out;
      if(PredictDistributionCore(x, hp, out) && out.move_mean_points > 0.0) return out.move_mean_points;
      if(m_move_ready && m_move_ema_abs > 0.0) return m_move_ema_abs;
      return 0.0;
   }
