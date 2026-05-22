   CFXAIAILightGBM(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_LIGHTGBM; }
   virtual string AIName(void) const { return "tree_lgbm"; }
   virtual int PersistentStateVersion(void) const { return 10; }
   virtual bool SupportsNativeParameterSnapshot(void) const { return true; }
   virtual string PersistentStateCoverageTag(void) const { return "native_model"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TREE, caps, 1, 1);

   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, class_probs);

      double ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      double ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev = class_probs[(int)FXAI_LABEL_BUY] * ev_buy + class_probs[(int)FXAI_LABEL_SELL] * ev_sell;

      double cost = ResolveCostPoints(x);
      if(cost < 0.0) cost = 0.0;
      ev = MathMax(0.0, ev - 0.35 * cost);

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.75 * ev + 0.25 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);

      if(expected_move_points < 0.0) expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double ev_buy = 0.0, q10_buy = 0.0, q50_buy = 0.0, q90_buy = 0.0, sup_buy = 0.0;
      double ev_sell = 0.0, q10_sell = 0.0, q50_sell = 0.0, q90_sell = 0.0, sup_sell = 0.0;
      ClassMoveStats((int)FXAI_LABEL_BUY, x, ev_buy, q10_buy, q50_buy, q90_buy, sup_buy);
      ClassMoveStats((int)FXAI_LABEL_SELL, x, ev_sell, q10_sell, q50_sell, q90_sell, sup_sell);
      if(ev_buy <= 0.0) ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      if(ev_sell <= 0.0) ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev = out.class_probs[(int)FXAI_LABEL_BUY] * ev_buy + out.class_probs[(int)FXAI_LABEL_SELL] * ev_sell;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      double mix_q10 = out.class_probs[(int)FXAI_LABEL_BUY] * q10_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q10_sell;
      double mix_q50 = out.class_probs[(int)FXAI_LABEL_BUY] * q50_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q50_sell;
      double mix_q90 = out.class_probs[(int)FXAI_LABEL_BUY] * q90_buy + out.class_probs[(int)FXAI_LABEL_SELL] * q90_sell;
      double sigma = MathMax(0.10, 0.25 * out.move_mean_points + 0.20 * (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, mix_q10 > 0.0 ? mix_q10 : (out.move_mean_points - 0.55 * sigma));
      out.move_q50_points = MathMax(out.move_q25_points, mix_q50 > 0.0 ? mix_q50 : out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, mix_q90 > 0.0 ? mix_q90 : (out.move_mean_points + 0.55 * sigma));
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      double support_rel = FXAI_Clamp((sup_buy + sup_sell) / 240.0, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.40 + 0.20 * (m_move_ready ? 1.0 : 0.0) + 0.20 * MathMin((double)m_tree_count[(int)FXAI_LABEL_BUY] / 32.0, 1.0) + 0.20 * support_rel, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      double bank_mfe = 0.0, bank_mae = 0.0, bank_hit = 1.0, bank_path = 0.5, bank_fill = 0.5, bank_trust = 0.0;
      GetQualityBankPriors(bank_mfe, bank_mae, bank_hit, bank_path, bank_fill, bank_trust);
      m_quality_heads.Predict(x,
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
      m_step = 0;
      m_buf_head = 0;
      m_buf_size = 0;
      m_quality_heads.Reset();

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_tree_count[c] = 0;
         for(int t=0; t<FXAI_LGB_MAX_TREES; t++) InitTree(m_trees[c][t]);
      }

      for(int i=0; i<FXAI_LGB_BUFFER; i++)
      {
         m_buf_cls[i] = (int)FXAI_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_cost[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++) m_buf_x[i][k] = 0.0;
      }

      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++) m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      m_cal3_steps = 0;

      m_val_ready = false;
      m_val_steps = 0;
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;
      m_quality_degraded = false;
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized) m_initialized = true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      EnsureInitialized(hp);
      m_step++;

      int cls = NormalizeClassLabel(y, x, move_points);
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP) cls = (int)FXAI_LABEL_SKIP;

      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double sample_w = MoveSampleWeight(x, move_points);
      sample_w = FXAI_Clamp(sample_w, 0.10, 6.00);
      m_quality_heads.Update(x,
                             sample_w,
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
      double cost = InputCostProxyPoints(x);

      // Online pre-update for calibration/metrics before structure update.
      double logits[FXAI_LGB_CLASS_COUNT];
      ModelRawLogits(x, logits);
      double p_raw[FXAI_LGB_CLASS_COUNT];
      Softmax3(logits, p_raw);
      double p_cal[FXAI_LGB_CLASS_COUNT];
      Calibrate3(p_raw, p_cal);

      double ev_buy = ClassExpectedMove((int)FXAI_LABEL_BUY, x);
      double ev_sell = ClassExpectedMove((int)FXAI_LABEL_SELL, x);
      double ev_now = p_cal[(int)FXAI_LABEL_BUY] * ev_buy + p_cal[(int)FXAI_LABEL_SELL] * ev_sell;
      UpdateValidationMetrics(cls, p_cal, ev_now, cost);

      double lr_cal = FXAI_Clamp(h.xgb_lr * 0.30, 0.0002, 0.0200);
      if(m_quality_degraded) lr_cal *= 0.80;
      UpdateCalibrator3(p_raw, cls, sample_w, lr_cal);

      // Legacy binary calibrator stays aligned for compatibility paths.
      double den_dir = p_raw[(int)FXAI_LABEL_BUY] + p_raw[(int)FXAI_LABEL_SELL];
      if(den_dir < 1e-9) den_dir = 1e-9;
      double p_dir_raw = p_raw[(int)FXAI_LABEL_BUY] / den_dir;
      if(cls == (int)FXAI_LABEL_BUY) UpdateCalibration(p_dir_raw, 1, sample_w);
      else if(cls == (int)FXAI_LABEL_SELL) UpdateCalibration(p_dir_raw, 0, sample_w);

      PushSample(cls, x, move_points, cost, sample_w);

      if(m_buf_size >= FXAI_LGB_MIN_BUFFER && (m_step % FXAI_LGB_BUILD_EVERY) == 0)
      {
         BuildOneTreeClass((int)FXAI_LABEL_SELL, h);
         BuildOneTreeClass((int)FXAI_LABEL_BUY, h);
         BuildOneTreeClass((int)FXAI_LABEL_SKIP, h);
      }

      FXAI_UpdateMoveEMA(m_move_ema_abs, m_move_ready, move_points, 0.05);
      UpdateMoveHead(x, move_points, h, sample_w);
   }

   virtual double PredictProb(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double expected_move = 0.0;
      if(!PredictModelCore(x, hp, probs, expected_move)) return 0.5;
      double den = probs[(int)FXAI_LABEL_BUY] + probs[(int)FXAI_LABEL_SELL];
      if(den < 1e-9) den = 1e-9;
      return FXAI_Clamp(probs[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[], const FXAIAIHyperParams &hp)
   {
      EnsureInitialized(hp);
      double probs[3];
      double ev = -1.0;
      if(PredictModelCore(x, hp, probs, ev) && ev > 0.0) return ev;
      return (m_move_ready ? m_move_ema_abs : 0.0);
   }

   virtual bool SaveModelState(const int handle) const
   {
      if(handle == INVALID_HANDLE)
         return false;

      FileWriteInteger(handle, (m_initialized ? 1 : 0));
      FileWriteInteger(handle, m_step);
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         FileWriteDouble(handle, m_bias[c]);
         FileWriteInteger(handle, m_tree_count[c]);
         for(int t=0; t<FXAI_LGB_MAX_TREES; t++)
         {
            FileWriteInteger(handle, m_trees[c][t].node_count);
            for(int n=0; n<FXAI_LGB_MAX_NODES; n++)
            {
               FXAILGBNode node = m_trees[c][t].nodes[n];
               FileWriteInteger(handle, (node.is_leaf ? 1 : 0));
               FileWriteInteger(handle, node.feature);
               FileWriteDouble(handle, node.threshold);
               FileWriteInteger(handle, (node.default_left ? 1 : 0));
               FileWriteInteger(handle, node.left);
               FileWriteInteger(handle, node.right);
               FileWriteInteger(handle, node.depth);
               FileWriteDouble(handle, node.leaf_value);
               FileWriteDouble(handle, node.move_mean);
               FileWriteDouble(handle, node.move_var);
               FileWriteDouble(handle, node.move_q10);
               FileWriteDouble(handle, node.move_q50);
               FileWriteDouble(handle, node.move_q90);
               FileWriteInteger(handle, node.sample_count);
            }
         }
      }
      FileWriteInteger(handle, m_buf_head);
      FileWriteInteger(handle, m_buf_size);
      for(int i=0; i<FXAI_LGB_BUFFER; i++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            FileWriteDouble(handle, m_buf_x[i][k]);
         FileWriteInteger(handle, m_buf_cls[i]);
         FileWriteDouble(handle, m_buf_move[i]);
         FileWriteDouble(handle, m_buf_cost[i]);
         FileWriteDouble(handle, m_buf_w[i]);
      }
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++)
            FileWriteDouble(handle, m_cal_vs_w[c][j]);
         FileWriteDouble(handle, m_cal_vs_b[c]);
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++)
         {
            FileWriteDouble(handle, m_cal_iso_pos[c][b]);
            FileWriteDouble(handle, m_cal_iso_cnt[c][b]);
         }
      }
      FileWriteInteger(handle, m_cal3_steps);
      FileWriteInteger(handle, (m_val_ready ? 1 : 0));
      FileWriteInteger(handle, m_val_steps);
      FileWriteDouble(handle, m_val_nll_fast);
      FileWriteDouble(handle, m_val_nll_slow);
      FileWriteDouble(handle, m_val_brier_fast);
      FileWriteDouble(handle, m_val_brier_slow);
      FileWriteDouble(handle, m_val_ece_fast);
      FileWriteDouble(handle, m_val_ece_slow);
      FileWriteDouble(handle, m_val_ev_fast);
      FileWriteDouble(handle, m_val_ev_slow);
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
      {
         FileWriteDouble(handle, m_ece_mass[b]);
         FileWriteDouble(handle, m_ece_acc[b]);
         FileWriteDouble(handle, m_ece_conf[b]);
      }
      FileWriteInteger(handle, (m_quality_degraded ? 1 : 0));
      return m_quality_heads.Save(handle);
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      if(handle == INVALID_HANDLE || version < 8)
         return false;

      m_initialized = (FileReadInteger(handle) != 0);
      m_step = FileReadInteger(handle);
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         m_bias[c] = FileReadDouble(handle);
         m_tree_count[c] = FileReadInteger(handle);
         for(int t=0; t<FXAI_LGB_MAX_TREES; t++)
         {
            m_trees[c][t].node_count = FileReadInteger(handle);
            for(int n=0; n<FXAI_LGB_MAX_NODES; n++)
            {
               m_trees[c][t].nodes[n].is_leaf = (FileReadInteger(handle) != 0);
               m_trees[c][t].nodes[n].feature = FileReadInteger(handle);
               m_trees[c][t].nodes[n].threshold = FileReadDouble(handle);
               m_trees[c][t].nodes[n].default_left = (FileReadInteger(handle) != 0);
               m_trees[c][t].nodes[n].left = FileReadInteger(handle);
               m_trees[c][t].nodes[n].right = FileReadInteger(handle);
               m_trees[c][t].nodes[n].depth = FileReadInteger(handle);
               m_trees[c][t].nodes[n].leaf_value = FileReadDouble(handle);
               m_trees[c][t].nodes[n].move_mean = FileReadDouble(handle);
               m_trees[c][t].nodes[n].move_var = FileReadDouble(handle);
               m_trees[c][t].nodes[n].move_q10 = FileReadDouble(handle);
               m_trees[c][t].nodes[n].move_q50 = FileReadDouble(handle);
               m_trees[c][t].nodes[n].move_q90 = FileReadDouble(handle);
               m_trees[c][t].nodes[n].sample_count = FileReadInteger(handle);
            }
         }
      }
      m_buf_head = FileReadInteger(handle);
      m_buf_size = FileReadInteger(handle);
      for(int i=0; i<FXAI_LGB_BUFFER; i++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_buf_x[i][k] = FileReadDouble(handle);
         m_buf_cls[i] = FileReadInteger(handle);
         m_buf_move[i] = FileReadDouble(handle);
         m_buf_cost[i] = FileReadDouble(handle);
         m_buf_w[i] = FileReadDouble(handle);
      }
      for(int c=0; c<FXAI_LGB_CLASS_COUNT; c++)
      {
         for(int j=0; j<FXAI_LGB_CLASS_COUNT; j++)
            m_cal_vs_w[c][j] = FileReadDouble(handle);
         m_cal_vs_b[c] = FileReadDouble(handle);
         for(int b=0; b<FXAI_LGB_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = FileReadDouble(handle);
            m_cal_iso_cnt[c][b] = FileReadDouble(handle);
         }
      }
      m_cal3_steps = FileReadInteger(handle);
      m_val_ready = (FileReadInteger(handle) != 0);
      m_val_steps = FileReadInteger(handle);
      m_val_nll_fast = FileReadDouble(handle);
      m_val_nll_slow = FileReadDouble(handle);
      m_val_brier_fast = FileReadDouble(handle);
      m_val_brier_slow = FileReadDouble(handle);
      m_val_ece_fast = FileReadDouble(handle);
      m_val_ece_slow = FileReadDouble(handle);
      m_val_ev_fast = FileReadDouble(handle);
      m_val_ev_slow = FileReadDouble(handle);
      for(int b=0; b<FXAI_LGB_ECE_BINS; b++)
      {
         m_ece_mass[b] = FileReadDouble(handle);
         m_ece_acc[b] = FileReadDouble(handle);
         m_ece_conf[b] = FileReadDouble(handle);
      }
      m_quality_degraded = (FileReadInteger(handle) != 0);
      return m_quality_heads.Load(handle);
   }
