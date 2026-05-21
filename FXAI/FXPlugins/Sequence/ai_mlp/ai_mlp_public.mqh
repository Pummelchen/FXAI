   CFXAIAIMLPTiny(void) { Reset(); }

   virtual int AIId(void) const { return (int)AI_MLP_TINY; }
   virtual string AIName(void) const { return "ai_mlp"; }


   virtual void Describe(FXAIAIManifestV4 &out) const

   {

      const ulong caps = (ulong)(FXAI_CAP_ONLINE_LEARNING|FXAI_CAP_REPLAY|FXAI_CAP_MULTI_HORIZON|FXAI_CAP_SELF_TEST);

      FillManifest(out, (int)FXAI_FAMILY_OTHER, caps, 1, 1);

   }


   virtual bool PredictModelCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        double &class_probs[],
                                        double &expected_move_points)
   {
      EnsureInitialized(hp);
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double x_norm[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, x_norm);

      double ctx_raw[FXAI_MLP_CTX], ctx_act[FXAI_MLP_CTX], ctx[FXAI_MLP_CTX];
      BuildTemporalContext(xa, ctx_raw);
      AdaptContext(ctx_raw, ctx, ctx_act);

      double h1_raw[FXAI_MLP_H1], h1[FXAI_MLP_H1], h2[FXAI_MLP_H2], h2_norm[FXAI_MLP_H2];
      double h3_tanh[FXAI_MLP_H2], h3_gate[FXAI_MLP_H2], h3[FXAI_MLP_H2];
      double ln_inv_std = 1.0, drop1[FXAI_MLP_H1];
      double logits[FXAI_MLP_CLASSES], probs[FXAI_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x_norm, ctx, m_shadow_ready, false, h1_raw, h1, h2, h2_norm, h3_tanh, h3_gate, h3, ln_inv_std, drop1, logits, probs, mu, logv, q25, q75);

      double p_raw[FXAI_MLP_CLASSES];
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         p_raw[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      Calibrate3(p_raw, class_probs);

      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double iqr = MathAbs(q75 - q25);
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * iqr);
      double active = FXAI_Clamp(1.0 - class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double ev = amp * active;

      if(ev > 0.0 && m_move_ready && m_move_ema_abs > 0.0) expected_move_points = 0.65 * ev + 0.35 * m_move_ema_abs;
      else if(ev > 0.0) expected_move_points = ev;
      else expected_move_points = (m_move_ready ? m_move_ema_abs : 0.0);
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
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      double x_norm[FXAI_AI_WEIGHTS];
      NormalizeInput(xa, x_norm);
      double ctx_raw[FXAI_MLP_CTX], ctx_act[FXAI_MLP_CTX], ctx[FXAI_MLP_CTX];
      BuildTemporalContext(xa, ctx_raw);
      AdaptContext(ctx_raw, ctx, ctx_act);
      double h1_raw[FXAI_MLP_H1], h1[FXAI_MLP_H1], h2[FXAI_MLP_H2], h2_norm[FXAI_MLP_H2];
      double h3_tanh[FXAI_MLP_H2], h3_gate[FXAI_MLP_H2], h3[FXAI_MLP_H2];
      double ln_inv_std = 1.0, drop1[FXAI_MLP_H1];
      double logits[FXAI_MLP_CLASSES], probs[FXAI_MLP_CLASSES];
      double mu = 0.0, logv = 0.0, q25 = 0.0, q75 = 0.0;
      Forward(x_norm, ctx, m_shadow_ready, false, h1_raw, h1, h2, h2_norm, h3_tanh, h3_gate, h3, ln_inv_std, drop1, logits, probs, mu, logv, q25, q75);
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         out.class_probs[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      Calibrate3(out.class_probs, out.class_probs);
      NormalizeClassDistribution(out.class_probs);
      double sigma = MathSqrt(MathMax(MathExp(logv), 1e-6));
      double amp = MathMax(0.0, 0.55 * MathAbs(mu) + 0.25 * sigma + 0.20 * MathAbs(q75 - q25));
      double active = FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0);
      double ev = amp * active;
      if(ev <= 0.0 && m_move_ready) ev = m_move_ema_abs;
      out.move_mean_points = MathMax(0.0, ev);
      out.move_q25_points = MathMax(0.0, MathAbs(q25) * active);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, MathAbs(q75) * active);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY], out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.35 * active + 0.20 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
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
      m_adam_step = 0;
      m_shadow_decay = 0.995;
      m_symbol_hash = 0.0;
      m_norm_steps = 0;
      m_mlp_replay_head = 0;
      m_mlp_replay_size = 0;
      m_swa_ready = false;
      m_swa_count = 0;
      m_cal3_steps = 0;
      m_val_ready = false;
      m_val_steps = 0;
      m_quality_degraded = false;
      m_quality_heads.Reset();

      ResetHistory();

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
         m_class_ema[c] = 1.0;

      for(int h=0; h<FXAI_MLP_H1; h++)
      {
         m_b1[h] = 0.0;
         m_sb1[h] = 0.0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         {
            m_w1[h][i] = 0.0;
            m_sw1[h][i] = 0.0;
         }
         for(int c=0; c<FXAI_MLP_CTX; c++)
         {
            m_w1c[h][c] = 0.0;
            m_sw1c[h][c] = 0.0;
            m_swa_w1c[h][c] = 0.0;
         }
         m_swa_b1[h] = 0.0;
      }

      for(int h=0; h<FXAI_MLP_H2; h++)
      {
         m_b2[h] = 0.0;
         m_sb2[h] = 0.0;
         m_b3[h] = 0.0;
         m_sb3[h] = 0.0;
         m_b3g[h] = 0.0;
         m_sb3g[h] = 0.0;
         m_swa_b2[h] = 0.0;
         m_swa_b3[h] = 0.0;
         m_swa_b3g[h] = 0.0;

         m_w_mu[h] = 0.0; m_sw_mu[h] = 0.0;
         m_w_logv[h] = 0.0; m_sw_logv[h] = 0.0;
         m_w_q25[h] = 0.0; m_sw_q25[h] = 0.0;
         m_w_q75[h] = 0.0; m_sw_q75[h] = 0.0;
         m_swa_w_mu[h] = 0.0;
         m_swa_w_logv[h] = 0.0;
         m_swa_w_q25[h] = 0.0;
         m_swa_w_q75[h] = 0.0;

         for(int j=0; j<FXAI_MLP_H1; j++)
         {
            m_w2[h][j] = 0.0;
            m_sw2[h][j] = 0.0;
            m_swa_w2[h][j] = 0.0;
         }
         for(int j=0; j<FXAI_MLP_H2; j++)
         {
            m_w3[h][j] = 0.0;
            m_sw3[h][j] = 0.0;
            m_swa_w3[h][j] = 0.0;
            m_w3g[h][j] = 0.0;
            m_sw3g[h][j] = 0.0;
            m_swa_w3g[h][j] = 0.0;
         }
      }

      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_b_cls[c] = 0.0;
         m_sb_cls[c] = 0.0;
         m_swa_b_cls[c] = 0.0;
         for(int h=0; h<FXAI_MLP_H2; h++)
         {
            m_w_cls[c][h] = 0.0;
            m_sw_cls[c][h] = 0.0;
            m_swa_w_cls[c][h] = 0.0;
         }
      }

      m_b_mu = 0.0; m_sb_mu = 0.0;
      m_b_logv = 0.0; m_sb_logv = 0.0;
      m_b_q25 = 0.0; m_sb_q25 = 0.0;
      m_b_q75 = 0.0; m_sb_q75 = 0.0;
      m_swa_b_mu = 0.0;
      m_swa_b_logv = 0.0;
      m_swa_b_q25 = 0.0;
      m_swa_b_q75 = 0.0;

      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      {
         m_norm_mean[i] = 0.0;
         m_norm_var[i] = 1.0;
         m_norm_median[i] = 0.0;
         m_norm_iqr[i] = 1.0;
         m_norm_gain[i] = 1.0;
         m_norm_bias[i] = 0.0;
      }
      for(int c=0; c<FXAI_MLP_CTX; c++)
      {
         m_ctx_b[c] = 0.0;
         for(int j=0; j<FXAI_MLP_CTX; j++)
            m_ctx_w[c][j] = (c == j ? 1.0 : 0.0);
      }
      for(int c=0; c<FXAI_MLP_CLASSES; c++)
      {
         m_cal_vs_b[c] = 0.0;
         for(int j=0; j<FXAI_MLP_CLASSES; j++)
            m_cal_vs_w[c][j] = (c == j ? 1.0 : 0.0);
         for(int b=0; b<FXAI_MLP_CAL_BINS; b++)
         {
            m_cal_iso_pos[c][b] = 0.0;
            m_cal_iso_cnt[c][b] = 0.0;
         }
      }
      for(int b=0; b<FXAI_MLP_ECE_BINS; b++)
      {
         m_ece_mass[b] = 0.0;
         m_ece_acc[b] = 0.0;
         m_ece_conf[b] = 0.0;
      }
      m_val_nll_fast = m_val_nll_slow = 0.0;
      m_val_brier_fast = m_val_brier_slow = 0.0;
      m_val_ece_fast = m_val_ece_slow = 0.0;
      m_val_ev_fast = m_val_ev_slow = 0.0;

      for(int r=0; r<FXAI_MLP_REPLAY; r++)
      {
         m_replay_y[r] = (int)FXAI_LABEL_SKIP;
         m_mlp_replay_move[r] = 0.0;
         m_mlp_replay_cost[r] = 0.0;
         m_mlp_replay_min_move[r] = 0.0;
         m_replay_w[r] = 1.0;
         m_mlp_replay_time[r] = 0;
         m_mlp_replay_regime[r] = 0;
         m_mlp_replay_horizon[r] = 1;
         m_replay_session[r] = -1;
         m_mlp_replay_feature_schema[r] = FXAI_SCHEMA_FULL;
         m_mlp_replay_norm_method[r] = FXAI_NORM_EXISTING;
         m_mlp_replay_sequence_bars[r] = 1;
         m_mlp_replay_point_value[r] = (_Point > 0.0 ? _Point : 1.0);
         m_mlp_replay_window_size[r] = 0;
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            m_mlp_replay_x[r][i] = 0.0;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_mlp_replay_window[r][b][k] = 0.0;
      }

      InitMoments();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp)
   {
      if(!m_initialized)
         InitWeights(hp.mlp_init);
   }

   virtual void Update(const int y,
                       const double &x[],
                       const FXAIAIHyperParams &hp)
   {
      double pseudo_move = (y == 1 ? 1.0 : -1.0);
      TrainModelCore(y, x, hp, pseudo_move);
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
      double xa[FXAI_AI_WEIGHTS];
      BuildWindowAwareInput(x, xa);
      FXAIAIHyperParams h = ScaleHyperParamsForMove(hp, move_points);
      double w = MoveSampleWeight(xa, move_points);
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
      datetime cur_t = ResolveContextTime();
      if(cur_t <= 0) cur_t = TimeCurrent();
      double cur_cost = ResolveCostPoints(xa);
      double cur_min = ResolveMinMovePoints();
      int cur_regime = m_ctx_regime_id;
      int cur_session = m_ctx_session_bucket;
      int cur_horizon = m_ctx_horizon_minutes;
      int cur_feature_schema = m_ctx_feature_schema_id;
      int cur_norm_method = m_ctx_normalization_method_id;
      int cur_sequence_bars = m_ctx_sequence_bars;
      double cur_point_value = m_ctx_point_value;
      int cur_window_size = m_ctx_window_size;
      double cur_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            cur_window[b][k] = m_ctx_window[b][k];
      int cur_sess = SessionBucket(cur_t);

      UpdateWeighted(y, xa, h, w, move_points, false);

      int replay_n = 0;
      if(m_mlp_replay_size >= 96) replay_n = 3;
      else if(m_mlp_replay_size >= 32) replay_n = 2;
      else if(m_mlp_replay_size >= 12) replay_n = 1;

      for(int r=0; r<replay_n; r++)
      {
         if(m_mlp_replay_size <= 0) break;
         int li = PluginRandIndex(m_mlp_replay_size);
         int p = ReplayPos(li);
         double rw = FXAI_Clamp(m_replay_w[p], 0.20, 4.00);
         rw *= ReplayAgeWeight(m_mlp_replay_time[p], cur_t);
         if(m_replay_session[p] >= 0 && m_replay_session[p] != cur_sess) rw *= 0.85;
         FXAIAIContextV4 replay_ctx;
         FXAI_ClearContextV4(replay_ctx);
         replay_ctx.sample_time = m_mlp_replay_time[p];
         replay_ctx.cost_points = m_mlp_replay_cost[p];
         replay_ctx.min_move_points = m_mlp_replay_min_move[p];
         replay_ctx.regime_id = m_mlp_replay_regime[p];
         replay_ctx.session_bucket = (m_replay_session[p] >= 0 ? m_replay_session[p] : cur_sess);
         replay_ctx.horizon_minutes = m_mlp_replay_horizon[p];
         replay_ctx.feature_schema_id = m_mlp_replay_feature_schema[p];
         replay_ctx.normalization_method_id = m_mlp_replay_norm_method[p];
         replay_ctx.sequence_bars = m_mlp_replay_sequence_bars[p];
         replay_ctx.point_value = m_mlp_replay_point_value[p];
         SetContext(replay_ctx);
         double replay_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               replay_window[b][k] = m_mlp_replay_window[p][b][k];
         SetWindowPayload(m_mlp_replay_window_size[p], replay_window);
         double replay_x[FXAI_AI_WEIGHTS];
         for(int i=0; i<FXAI_AI_WEIGHTS; i++)
            replay_x[i] = m_mlp_replay_x[p][i];
         UpdateWeighted(m_replay_y[p], replay_x, h, rw, m_mlp_replay_move[p], true);
      }

      FXAIAIContextV4 cur_ctx;
      FXAI_ClearContextV4(cur_ctx);
      cur_ctx.sample_time = cur_t;
      cur_ctx.cost_points = cur_cost;
      cur_ctx.min_move_points = cur_min;
      cur_ctx.regime_id = cur_regime;
      cur_ctx.session_bucket = cur_session;
      cur_ctx.horizon_minutes = cur_horizon;
      cur_ctx.feature_schema_id = cur_feature_schema;
      cur_ctx.normalization_method_id = cur_norm_method;
      cur_ctx.sequence_bars = cur_sequence_bars;
      cur_ctx.point_value = cur_point_value;
      SetContext(cur_ctx);
      m_ctx_window_size = cur_window_size;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = cur_window[b][k];
   }

   virtual double PredictProb(const double &x[],
                              const FXAIAIHyperParams &hp)
   {
      double cp[FXAI_MLP_CLASSES];
      double em = 0.0;
      PredictModelCore(x, hp, cp, em);
      double den = cp[(int)FXAI_LABEL_BUY] + cp[(int)FXAI_LABEL_SELL];
      if(den <= 1e-9) return 0.5;
      return FXAI_Clamp(cp[(int)FXAI_LABEL_BUY] / den, 0.001, 0.999);
   }

   virtual double PredictExpectedMovePoints(const double &x[],
                                            const FXAIAIHyperParams &hp)
   {
      double cp[FXAI_MLP_CLASSES];
      double em = 0.0;
      PredictModelCore(x, hp, cp, em);
      return em;
   }
