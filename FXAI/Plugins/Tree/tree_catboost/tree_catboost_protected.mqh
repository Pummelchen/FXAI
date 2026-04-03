   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (int)FXAI_LABEL_SKIP;

      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
         m_cls_ema[c] = 0.997 * m_cls_ema[c] + (c == cls ? 0.003 : 0.0);
      double mean_cls = (m_cls_ema[0] + m_cls_ema[1] + m_cls_ema[2]) / 3.0;
      double cls_bal = FXAI_Clamp(mean_cls / MathMax(m_cls_ema[cls], 0.005), 0.60, 2.50);

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double cost = InputCostProxyPoints(x);
      double abs_move = MathAbs(move_points);
      double edge = MathMax(0.0, abs_move - cost);
      double ev_w = FXAI_Clamp(0.35 + (edge / MathMax(cost, 0.50)), 0.10, 6.00);
      if(cls == (int)FXAI_LABEL_SKIP) ev_w *= 0.85;
      double w = FXAI_Clamp(ev_w * cls_bal, 0.10, 6.00);
      UpdateNativeQualityHeads(x, w, h.lr, h.l2);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      double p_cal[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, p_cal);

      double ce = -MathLog(FXAI_Clamp(p_cal[cls], 1e-6, 1.0));
      UpdateLossDrift(ce);
      UpdateValidationHarness(cls, x, p_cal, w);

      double cal_lr = FXAI_Clamp(0.01 + 0.12 * FXAI_Clamp(h.xgb_lr, 0.0005, 0.3000), 0.0005, 0.0300);
      UpdateCalibrator3(p_raw, cls, w, cal_lr);

      // Keep legacy binary calibrator aligned for compatibility paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, w);

      PushSample(cls, x, move_points, w);

      int build_every = FXAI_CAT_BUILD_EVERY;
      if(m_drift_cooldown > 0) build_every = FXAI_CAT_BUILD_EVERY / 2;
      if(m_quality_alarm > 8) build_every = MathMax(24, build_every / 2);
      if(build_every < 16) build_every = 16;
      if(m_buf_size >= FXAI_CAT_MIN_BUFFER && (m_step % build_every) == 0)
         BuildOneTree(h);

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, w);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      double p_cal[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, p_cal);

      double den = p_cal[(int)FXAI_LABEL_BUY] + p_cal[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) return 0.5;
      return FXAI_Clamp(p_cal[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double sum = 0.0;
      double wsum = 0.0;
      for(int t=0; t<m_tree_count; t++)
      {
         int leaf = TraverseLeafIndex(m_trees[t], x_ext);
         if(leaf < 0 || leaf >= FXAI_CAT_MAX_LEAVES) continue;

         double mv = m_trees[t].leaf_move_mean[leaf];
         if(mv <= 0.0) continue;

         double conf = MathAbs(m_trees[t].leaf_value[leaf][(int)FXAI_LABEL_BUY] -
                               m_trees[t].leaf_value[leaf][(int)FXAI_LABEL_SELL]) + 0.15;
         sum += conf * mv;
         wsum += conf;
      }

      double tree_est = (wsum > 0.0 ? sum / wsum : -1.0);
      if(tree_est > 0.0 && m_move_ready && m_move_ema_abs > 0.0) return 0.70 * tree_est + 0.30 * m_move_ema_abs;
      if(tree_est > 0.0) return tree_est;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }
