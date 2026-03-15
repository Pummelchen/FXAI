   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);

      if((m_step % 1024) == 0 || MathAbs(x[1]) > 9.0 || MathAbs(x[2]) > 9.0)
         ResetSequence();

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      UpdateNormStats(xa);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      int cls = ResolveClass(y, xa, move_points);
      double sw = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.25, 6.0);
      double cost = InputCostProxyPoints(xa);

      AppendTrainSample(cls, xa, move_points, cost, sw);
      TrainTBPTT(h);

      // Commit the newest sample to live state.
      double fin[FXAI_AI_MLP_HIDDEN];
      double p_raw[FXAI_AF_CLASS_COUNT];
      double mu = 0.0, lv = 0.0, q25 = 0.0, q75 = 0.0;
      ForwardStep(xa, true, false, -1, fin, p_raw, mu, lv, q25, q75);
      UpdateNativeQualityHeads(xa, sw, h.lr, h.l2);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_AF_CLASS_COUNT];
      double ev = 0.0;
      if(!PredictModelCore(x, hp, probs, ev))
         return 0.5;

      double den = probs[FXAI_AF_BUY] + probs[FXAI_AF_SELL];
      if(den < 1e-9) den = 1e-9;
      double p_dir_raw = probs[FXAI_AF_BUY] / den;
      double p_dir_cal = CalibrateProb(p_dir_raw);
      double p = p_dir_cal * FXAI_Clamp(1.0 - probs[FXAI_AF_SKIP], 0.0, 1.0);
      return FXAI_Clamp(p, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_AF_CLASS_COUNT];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0)
         return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
};
