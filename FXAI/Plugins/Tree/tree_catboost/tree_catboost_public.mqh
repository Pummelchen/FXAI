   CFXAIAICatBoost(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_CATBOOST; }
   virtual string AIName(void) const { return "tree_catboost"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_TREE, caps, 1, 1);

   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
      m_initialized = false;
      m_quality_heads.Reset();
      m_step = 0;
      m_tree_count = 0;
      m_buf_head = 0;
      m_buf_size = 0;

      m_loss_ready = false;
      m_loss_fast = 0.0;
      m_loss_slow = 0.0;
      m_drift_cooldown = 0;

      m_cal3_steps = 0;
      for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
      {
         m_bias[c] = 0.0;
         m_cls_ema[c] = 1.0;
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_CAT_CLASS_COUNT; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);

         for(int b=0; b<FXAI_CAT_CAL_BINS; b++)
         {
            m_cal3_iso_pos[c][b] = 0.0;
            m_cal3_iso_cnt[c][b] = 0.0;
         }
      }

      for(int s=0; s<4; s++)
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            m_cal_session_b[s][c] = 0.0;
      for(int r=0; r<2; r++)
         for(int c=0; c<FXAI_CAT_CLASS_COUNT; c++)
            m_cal_regime_b[r][c] = 0.0;

      m_val_ready = false;
      m_val_steps = 0;
      m_val_ce_fast = 0.0;
      m_val_ce_slow = 0.0;
      m_val_brier_fast = 0.0;
      m_val_brier_slow = 0.0;
      m_val_ece_fast = 0.0;
      m_val_ece_slow = 0.0;
      for(int b=0; b<FXAI_CAT_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_feat_stats_ready = false;
      m_feat_drift_score = 0.0;
      m_quality_alarm = 0;
      for(int k=0; k<FXAI_CAT_TRACK_FEATS; k++)
      {
         m_feat_ref_mean[k] = 0.0;
         m_feat_ref_var[k] = 1.0;
         m_feat_cur_mean[k] = 0.0;
         m_feat_cur_var[k] = 1.0;
      }

      for(int i=0; i<FXAI_CAT_BUFFER; i++)
      {
         m_buf_y[i] = (int)FXAI_LABEL_SKIP;
         m_buf_move[i] = 0.0;
         m_buf_w[i] = 1.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_buf_x[i][k] = 0.0;
      }

      InitCTRLayout();
      ResetCTRState();
      ResetSplitState();

      for(int t=0; t<FXAI_CAT_MAX_TREES; t++)
         InitTree(m_trees[t]);
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(m_initialized) return;
      m_initialized = true;
      // Slight skip prior reduces early over-trading before first trees are built.
      m_bias[(int)FXAI_LABEL_SKIP] = 0.10;
   }

   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);

      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);

      double margins[FXAI_CAT_CLASS_COUNT];
      double p_raw[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, class_probs);

      expected_move_points = PredictExpectedMovePoints(x, hp);
      if(expected_move_points < 0.0)
         expected_move_points = 0.0;
      return true;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      EnsureInitialized(hp);
      ResetModelOutput(out);
      double x_ext[FXAI_CAT_EXT_WEIGHTS];
      BuildInferenceExtended(x, x_ext);
      double margins[FXAI_CAT_CLASS_COUNT], p_raw[FXAI_CAT_CLASS_COUNT];
      ModelMarginsExt(x_ext, margins);
      Softmax3(margins, p_raw);
      Calibrate3(p_raw, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double pred = PredictExpectedMovePoints(x, hp);
      out.move_mean_points = MathMax(0.0, pred);
      double sigma = MathMax(0.10, 0.30 * out.move_mean_points + 0.25 * (m_move_ready ? m_move_ema_abs : 0.0));
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * (m_move_ready ? 1.0 : 0.0) + 0.30 * MathMin((double)m_tree_count / 32.0, 1.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      PredictNativeQualityHeads(x,
                                FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0),
                                out.reliability,
                                out.confidence,
                                out);
      return true;
   }

   virtual void Update(const int y, const double &x[], const FXAIAIHyperParams &hp)
   {
      int cls = (y > 0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double pseudo_move = (y > 0 ? 1.0 : -1.0);
      TrainModelCore(cls, x, hp, pseudo_move);
   }

