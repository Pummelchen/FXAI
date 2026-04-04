   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      int cls = NormalizeClassLabel(y, x, move_points);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost_proxy = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double excess = MathMax(0.0, abs_move - cost_proxy);

      double edge_ratio = excess / MathMax(cost_proxy, 0.50);

      // Explicit skip EV prior.
      if(cls == (int)FXAI_LABEL_SKIP)
      {
         double skip_prior = FXAI_Clamp(1.35 - 0.30 * edge_ratio, 0.25, 1.50);
         if(edge_ratio > 1.8 && abs_move > 1.2 * MathMax(cost_proxy, 0.5))
            cls = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
         else
            edge_ratio *= skip_prior;
      }
      else
      {
         if(edge_ratio < 0.05) cls = (int)FXAI_LABEL_SKIP;
      }

      double ev_w = FXAI_Clamp(0.35 + edge_ratio, 0.10, 6.00);
      if(cls == (int)FXAI_LABEL_SKIP)
         ev_w *= FXAI_Clamp(1.20 - 0.20 * edge_ratio, 0.25, 1.40);
      else
         ev_w *= FXAI_Clamp(0.65 + 0.40 * edge_ratio, 0.40, 2.50);

      double move_scale = FXAI_Clamp(1.0 + 0.10 * excess, 0.70, 3.50);
      double margin_scale = FXAI_Clamp(move_scale * (1.0 + 0.25 * (ev_w - 1.0)), 0.60, 4.00);
      // Regime/session-aware margin schedule.
      datetime t = ResolveContextTime();
      if(t <= 0) t = TimeCurrent();
      int sess = SessionBucket(t);
      double sess_scale = 1.0;
      if(sess == 0) sess_scale = 1.10;
      else if(sess == 1) sess_scale = 0.95;
      else if(sess == 2) sess_scale = 0.92;
      else sess_scale = 1.05;

      double vol_proxy = MathAbs(x[1]) + 0.7 * MathAbs(x[2]) + 0.5 * MathAbs(x[3]);
      double regime_scale = 1.0;
      if(vol_proxy < 0.20) regime_scale = 1.10;
      else if(vol_proxy > 2.00) regime_scale = 0.90;

      double spread_scale = FXAI_Clamp(1.0 + 0.08 * (cost_proxy - 1.0), 0.80, 1.30);
      margin_scale = FXAI_Clamp(margin_scale * sess_scale * regime_scale * spread_scale, 0.50, 5.00);
      UpdateNativeQualityHeads(x, ev_w, h.lr, h.l2);

      UpdateWeighted(cls, x, h, margin_scale, ev_w, move_points, false);

      // Hard-example replay.
      int replay_n = 0;
      if(m_rep_size >= 96) replay_n = 2;
      else if(m_rep_size >= 24) replay_n = 1;
      for(int r=0; r<replay_n; r++)
      {
         int idx = PickHardReplay();
         if(idx < 0) break;
         double rw = FXAI_Clamp(0.75 * m_rep_w[idx], 0.10, 4.00);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            replay_x[i] = m_rep_x[idx][i];
         UpdateWeighted(m_rep_cls[idx], replay_x, h, margin_scale, rw, m_rep_move[idx], true);
      }
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      double probs[FXAI_PA_CLASS_COUNT];
      double expected = -1.0;
      if(!PredictModelCore(x, hp, probs, expected))
         return 0.5;

      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      double head = PredictMoveRaw(x);

      if(m_mv_steps >= 24 && m_move_ready && m_move_ema_abs > 0.0)
         head = 0.70 * head + 0.30 * m_move_ema_abs;

      if(head > 0.0) return head;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
