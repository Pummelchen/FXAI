   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_seen++;

      // Per-symbol lifecycle guard.
      uint sh = SymbolHash(_Symbol);
      if(sh != m_symbol_hash)
      {
         m_symbol_hash = sh;
         ResetHistory();
         ResetTrainBuffer();
         ResetWalkForward();
         ResetSessionCalibration();
      }

      // Controlled reset policy for state-bleed / regime shocks.
      if((m_seen % 1024) == 0 || MathAbs(x[1]) > 8.0 || MathAbs(x[2]) > 8.0)
      {
         ResetHistory();
         ResetTrainBuffer();
      }

      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      UpdateNorm(xa);

      int cls = MapClass(y, xa, move_points);
      double sw = FXAI_Clamp(MoveSampleWeight(xa, move_points), 0.25, 4.00);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
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

      AppendTrainSample(cls, xa, move_points, cost, sw);
      TrainTBPTT(h);

      PushHistory(xa);
   }

   virtual double PredictProb(const double &x[],
                              const FXAIAIHyperParams &hp)
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

      double p_up = p_cal * active;
      return FXAI_Clamp(p_up, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[],
                                            const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double ps = 0.3333, pb = 0.3333, pk = 0.3333;
      double mu = 0.0, logv = MathLog(1.0), q25 = 0.0, q75 = 0.0;

      ForwardSequenceContext(x, m_shadow_ready, ps, pb, pk, mu, logv, q25, q75);

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);

      double active = FXAI_Clamp(1.0 - pk, 0.0, 1.0);
      double ev = MathMax(0.0, (0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr) * active);

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) return 0.65 * ev + 0.35 * m_move_ema_abs;
      if(ev > 0.0) return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
