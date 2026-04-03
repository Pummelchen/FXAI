   void SetContext(const datetime sample_time,
                   const double cost_points,
                   const double min_move_points,
                   const int regime_id,
                   const int horizon_minutes)
   {
      FXAIAIContextV4 ctx;
      FXAI_ClearContextV4(ctx);
      ctx.api_version = FXAI_API_VERSION_V4;
      ctx.regime_id = regime_id;
      ctx.session_bucket = FXAI_DeriveSessionBucket(sample_time);
      ctx.horizon_minutes = horizon_minutes;
      ctx.feature_schema_id = m_ctx_feature_schema_id;
      ctx.normalization_method_id = m_ctx_normalization_method_id;
      ctx.sequence_bars = m_ctx_sequence_bars;
      ctx.cost_points = cost_points;
      ctx.min_move_points = min_move_points;
      ctx.point_value = m_ctx_point_value;
      ctx.domain_hash = m_ctx_domain_hash;
      ctx.sample_time = sample_time;
      SetContext(ctx);
   }

   int NormalizeClassLabel(const int y,
                           const double &x[],
                           const double move_points) const
   {
      if(y >= (int)FXAI_LABEL_SELL && y <= (int)FXAI_LABEL_SKIP)
         return y;

      double cost = InputCostProxyPoints(x);
      double edge = MathAbs(move_points) - cost;
      double skip_band = 0.10 + 0.25 * MathMax(cost, 0.0);
      if(edge <= skip_band) return (int)FXAI_LABEL_SKIP;

      if(y > 0) return (int)FXAI_LABEL_BUY;
      if(y == 0) return (int)FXAI_LABEL_SELL;
      return (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
   }

   void ResetAuxState(void)
   {
      m_move_ready = false;
      m_move_ema_abs = 0.0;

      m_move_head_ready = false;
      m_move_head_steps = 0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_move_w[i] = 0.0;

      m_cal_ready = false;
      m_cal_steps = 0;
      m_cal_a = 1.0;
      m_cal_b = 0.0;
      for(int i=0; i<12; i++)
      {
         m_iso_pos[i] = 0.0;
         m_iso_cnt[i] = 0.0;
      }
      m_shared_adapter_ready = false;
      m_shared_adapter_steps = 0;
      m_shared_backbone_ready = false;
      m_shared_backbone_steps = 0;
      for(int c=0; c<3; c++)
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_cls_w[c][i] = 0.0;
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         m_shared_move_w[i] = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         m_shared_backbone_b[j] = 0.0;
         m_shared_backbone_move[j] = 0.01 * (double)(((j * 5) % 9) - 4);
         for(int c=0; c<3; c++)
            m_shared_backbone_cls[c][j] = 0.01 * (double)((((c + 1) * (j + 3)) % 7) - 3);
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_backbone_w[j][i] = 0.004 * (double)((((j + 1) * (i + 5)) % 11) - 5);
         for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
            m_shared_backbone_seq_w[j][t] = 0.003 * (double)((((j + 3) * (t + 7)) % 13) - 6);
         for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
         {
            m_shared_backbone_time_w[j][c] = 0.0045 * (double)((((j + 5) * (c + 3)) % 15) - 7);
            m_shared_backbone_time_gate_w[j][c] = 0.0025 * (double)((((j + 7) * (c + 4)) % 13) - 6);
            m_shared_backbone_state_w[j][c] = 0.0035 * (double)((((j + 9) * (c + 2)) % 15) - 7);
         }
         m_shared_backbone_state_rec_w[j] = 0.16 + 0.02 * (double)((j % 5) - 2);
         m_shared_backbone_state_b[j] = 0.0;
      }
      for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            m_shared_domain_emb[d][j] = 0.003 * (double)((((d + 1) * (j + 2)) % 9) - 4);
      for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            m_shared_horizon_emb[h][j] = 0.003 * (double)((((h + 2) * (j + 1)) % 9) - 4);
      for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            m_shared_session_emb[s][j] = 0.002 * (double)((((s + 3) * (j + 1)) % 7) - 3);
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         m_transfer_slot_obs[slot] = 0.0;
         m_transfer_slot_align[slot] = 0.0;
         m_transfer_slot_lead[slot] = 0.5;
         m_transfer_slot_move[slot] = 1.0;
      }

      m_ctx_time_ready = false;
      m_ctx_time = 0;
      m_ctx_cost_ready = false;
      m_ctx_cost_points = 0.0;
      m_ctx_min_move_points = 0.0;
      m_ctx_regime_id = 0;
      m_ctx_session_bucket = 0;
      m_ctx_horizon_minutes = 1;
      m_ctx_feature_schema_id = 1;
      m_ctx_normalization_method_id = 0;
      m_ctx_sequence_bars = 1;
      m_ctx_point_value = (_Point > 0.0 ? _Point : 1.0);
      m_ctx_domain_hash = FXAI_SymbolHash01(_Symbol);
      m_ctx_window_size = 0;
      m_target_quality_ready = false;
      m_target_mfe_points = 0.0;
      m_target_mae_points = 0.0;
      m_target_hit_time_frac = 1.0;
      m_target_path_flags = 0;
      m_target_path_risk = 0.0;
      m_target_fill_risk = 0.0;
      m_target_masked_step = 0.0;
      m_target_next_vol = 0.0;
      m_target_regime_shift = 0.0;
      m_target_context_lead = 0.5;
      m_quality_head_ready = false;
      m_quality_mfe_ema = 0.0;
      m_quality_mae_ema = 0.0;
      m_quality_hit_ema = 1.0;
      m_quality_path_risk_ema = 0.5;
      m_quality_fill_risk_ema = 0.5;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = 0.0;

      for(int r=0; r<FXAI_PLUGIN_REGIME_BUCKETS; r++)
      {
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         {
            for(int h=0; h<FXAI_PLUGIN_HORIZON_BUCKETS; h++)
            {
               m_bank_total[r][s][h] = 0.0;
               m_bank_ev_scale[r][s][h] = 1.0;
               m_bank_ev_bias[r][s][h] = 0.0;
               m_bank_ev_g2_scale[r][s][h] = 0.0;
               m_bank_ev_g2_bias[r][s][h] = 0.0;
               m_quality_bank_ready[r][s][h] = false;
               m_quality_bank_obs[r][s][h] = 0.0;
               m_quality_bank_mfe[r][s][h] = 0.0;
               m_quality_bank_mae[r][s][h] = 0.0;
               m_quality_bank_hit[r][s][h] = 1.0;
               m_quality_bank_path[r][s][h] = 0.5;
               m_quality_bank_fill[r][s][h] = 0.5;
               for(int c=0; c<3; c++)
                  m_bank_class_mass[r][s][h][c] = (c == (int)FXAI_LABEL_SKIP ? 1.2 : 1.0);
            }
         }
      }

      m_replay_head = 0;
      m_replay_size = 0;
      for(int i=0; i<FXAI_PLUGIN_REPLAY_CAPACITY; i++)
      {
         m_replay_label[i] = (int)FXAI_LABEL_SKIP;
         m_replay_move[i] = 0.0;
         m_replay_mfe[i] = 0.0;
         m_replay_mae[i] = 0.0;
         m_replay_hit_time[i] = 1.0;
         m_replay_path_flags[i] = 0;
         m_replay_path_risk[i] = 0.0;
         m_replay_fill_risk[i] = 0.0;
         m_replay_masked_step[i] = 0.0;
         m_replay_next_vol[i] = 0.0;
         m_replay_regime_shift[i] = 0.0;
         m_replay_context_lead[i] = 0.5;
         m_replay_cost[i] = 0.0;
         m_replay_min_move[i] = 0.0;
         m_replay_time[i] = 0;
         m_replay_regime[i] = 0;
         m_replay_session_bucket[i] = 0;
         m_replay_horizon[i] = 1;
         m_replay_feature_schema[i] = 1;
         m_replay_norm_method[i] = 0;
         m_replay_sequence_bars[i] = 1;
         m_replay_point_value[i] = (_Point > 0.0 ? _Point : 1.0);
         m_replay_domain_hash[i] = FXAI_SymbolHash01(_Symbol);
         m_replay_window_size[i] = 0;
         m_replay_priority[i] = 0.0;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_replay_window[i][b][k] = 0.0;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_x[i][k] = 0.0;
      }

      m_core_predict_calls = 0;
      m_core_predict_failures = 0;
      m_replay_rehearsals = 0;
      m_rng_seeded = false;
      m_rng_state = 0u;
      m_persist_hp_ready = false;
      m_persist_train_events = 0;
      m_persist_hp.lr = 0.0;
      m_persist_hp.l2 = 0.0;
      m_persist_hp.ftrl_alpha = 0.0;
      m_persist_hp.ftrl_beta = 0.0;
      m_persist_hp.ftrl_l1 = 0.0;
      m_persist_hp.ftrl_l2 = 0.0;
      m_persist_hp.pa_c = 0.0;
      m_persist_hp.pa_margin = 0.0;
      m_persist_hp.xgb_lr = 0.0;
      m_persist_hp.xgb_l2 = 0.0;
      m_persist_hp.xgb_split = 0.0;
      m_persist_hp.mlp_lr = 0.0;
      m_persist_hp.mlp_l2 = 0.0;
      m_persist_hp.mlp_init = 0.0;
      m_persist_hp.quantile_lr = 0.0;
      m_persist_hp.quantile_l2 = 0.0;
      m_persist_hp.enhash_lr = 0.0;
      m_persist_hp.enhash_l1 = 0.0;
      m_persist_hp.enhash_l2 = 0.0;
      m_persist_hp.tcn_layers = 0.0;
      m_persist_hp.tcn_kernel = 0.0;
      m_persist_hp.tcn_dilation_base = 0.0;
   }

   void EnsurePluginRNG(void)
   {
      if(m_rng_seeded) return;

      uint seed = (uint)(AIId() + 1);
      seed = (uint)(seed * 747796405u + 2891336453u);
      if(seed == 0u)
         seed = 2463534242u;

      m_rng_state = seed;
      m_rng_seeded = true;
   }

   double PluginRand01(void)
   {
      EnsurePluginRNG();
      m_rng_state = (uint)(1664525u * m_rng_state + 1013904223u);
      return FXAI_Clamp(((double)m_rng_state + 0.5) / 4294967296.0, 0.0, 1.0);
   }

   int PluginRandIndex(const int n)
   {
      if(n <= 0) return -1;
      int idx = (int)MathFloor(PluginRand01() * (double)n);
      if(idx < 0) idx = 0;
      if(idx >= n) idx = n - 1;
      return idx;
   }

   void UpdateMoveHead(const double &x[],
                       const double move_points,
                       const FXAIAIHyperParams &hp,
                       const double sample_w)
   {
      double tgt = MathAbs(move_points);
      if(!MathIsValidNumber(tgt)) return;

      double pred = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         pred += m_move_w[i] * x[i];
      if(pred < 0.0) pred = 0.0;

      double err = FXAI_ClipSym(tgt - pred, 20.0);
      double lr = FXAI_Clamp(0.08 * hp.lr, 0.00005, 0.02000);
      double l2 = FXAI_Clamp(0.25 * hp.l2, 0.0000, 0.1000);
      double w = FXAI_Clamp(sample_w, 0.25, 4.00);

      m_move_w[0] += lr * w * err;
      for(int i=1; i<FXAI_AI_WEIGHTS; i++)
         m_move_w[i] += lr * (w * FXAI_ClipSym(err * x[i], 6.0) - l2 * m_move_w[i]);

      m_move_head_steps++;
      if(m_move_head_steps >= 16) m_move_head_ready = true;
   }

   void SetTrainingTargetsRaw(const double mfe_points,
                              const double mae_points,
                              const double hit_time_frac,
                              const int path_flags,
                              const double path_risk,
                              const double fill_risk,
                              const double masked_step,
                              const double next_vol,
                              const double regime_shift,
                              const double context_lead)
   {
      m_target_quality_ready = true;
      m_target_mfe_points = MathMax(mfe_points, 0.0);
      m_target_mae_points = MathMax(mae_points, 0.0);
      m_target_hit_time_frac = FXAI_Clamp(hit_time_frac, 0.0, 1.0);
      m_target_path_flags = path_flags;
      m_target_path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);
      m_target_fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      m_target_masked_step = masked_step;
      m_target_next_vol = MathMax(next_vol, 0.0);
      m_target_regime_shift = FXAI_Clamp(regime_shift, 0.0, 1.0);
      m_target_context_lead = FXAI_Clamp(context_lead, 0.0, 1.0);
   }

   void SetTrainingTargets(const FXAIAITrainRequestV4 &req)
   {
      SetTrainingTargetsRaw(req.mfe_points,
                            req.mae_points,
                            req.time_to_hit_frac,
                            req.path_flags,
                            req.path_risk,
                            req.fill_risk,
                            req.masked_step_target,
                            req.next_vol_target,
                            req.regime_shift_target,
                            req.context_lead_target);
   }

   void UpdateQualityHeads(const FXAIAITrainRequestV4 &req,
                           const double sample_w)
   {
      double alpha = FXAI_Clamp(0.06 * FXAI_Clamp(sample_w, 0.25, 4.0), 0.01, 0.20);
      if(!m_quality_head_ready)
      {
         m_quality_mfe_ema = MathMax(req.mfe_points, 0.0);
         m_quality_mae_ema = MathMax(req.mae_points, 0.0);
         m_quality_hit_ema = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
         m_quality_path_risk_ema = FXAI_Clamp(req.path_risk, 0.0, 1.0);
         m_quality_fill_risk_ema = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
         m_quality_head_ready = true;
         return;
      }

      m_quality_mfe_ema = (1.0 - alpha) * m_quality_mfe_ema + alpha * MathMax(req.mfe_points, 0.0);
      m_quality_mae_ema = (1.0 - alpha) * m_quality_mae_ema + alpha * MathMax(req.mae_points, 0.0);
      m_quality_hit_ema = (1.0 - alpha) * m_quality_hit_ema + alpha * FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      m_quality_path_risk_ema = (1.0 - alpha) * m_quality_path_risk_ema + alpha * FXAI_Clamp(req.path_risk, 0.0, 1.0);
      m_quality_fill_risk_ema = (1.0 - alpha) * m_quality_fill_risk_ema + alpha * FXAI_Clamp(req.fill_risk, 0.0, 1.0);

      int r = req.ctx.regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = req.ctx.session_bucket;
      if(s < 0) s = 0;
      if(s >= FXAI_PLUGIN_SESSION_BUCKETS) s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
      int h = ContextHorizonBucket();
      if(req.ctx.horizon_minutes > 0)
      {
         int hh = req.ctx.horizon_minutes;
         if(hh <= 1) h = 0;
         else if(hh <= 3) h = 1;
         else if(hh <= 5) h = 2;
         else if(hh <= 8) h = 3;
         else if(hh <= 13) h = 4;
         else if(hh <= 21) h = 5;
         else if(hh <= 34) h = 6;
         else h = FXAI_PLUGIN_HORIZON_BUCKETS - 1;
      }

      double obs = m_quality_bank_obs[r][s][h];
      double bank_alpha = FXAI_Clamp(0.12 * FXAI_Clamp(sample_w, 0.25, 4.0) / MathSqrt(1.0 + 0.02 * obs), 0.02, 0.25);
      double mfe_now = MathMax(req.mfe_points, 0.0);
      double mae_now = MathMax(req.mae_points, 0.0);
      double hit_now = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      double path_now = FXAI_Clamp(req.path_risk, 0.0, 1.0);
      double fill_now = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
      if(!m_quality_bank_ready[r][s][h])
      {
         m_quality_bank_mfe[r][s][h] = mfe_now;
         m_quality_bank_mae[r][s][h] = mae_now;
         m_quality_bank_hit[r][s][h] = hit_now;
         m_quality_bank_path[r][s][h] = path_now;
         m_quality_bank_fill[r][s][h] = fill_now;
         m_quality_bank_ready[r][s][h] = true;
      }
      else
      {
         m_quality_bank_mfe[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_mfe[r][s][h] + bank_alpha * mfe_now;
         m_quality_bank_mae[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_mae[r][s][h] + bank_alpha * mae_now;
         m_quality_bank_hit[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_hit[r][s][h] + bank_alpha * hit_now;
         m_quality_bank_path[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_path[r][s][h] + bank_alpha * path_now;
         m_quality_bank_fill[r][s][h] = (1.0 - bank_alpha) * m_quality_bank_fill[r][s][h] + bank_alpha * fill_now;
      }
      m_quality_bank_obs[r][s][h] = MathMin(obs + FXAI_Clamp(sample_w, 0.25, 4.0), 50000.0);
   }

   double PredictMoveHeadRaw(const double &x[]) const
   {
      double p = 0.0;
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         p += m_move_w[i] * x[i];
      if(p < 0.0) p = 0.0;
      return p;
   }

   void UpdateCalibration(const double prob_raw,
                          const int y,
                          const double sample_w = 1.0)
   {
      double pr = FXAI_Clamp(prob_raw, 0.001, 0.999);
      double z = FXAI_Logit(pr);
      double py = FXAI_Sigmoid((m_cal_a * z) + m_cal_b);
      double e = ((double)y - py);

      double w = FXAI_Clamp(sample_w, 0.25, 4.00);
      double lr = 0.015 * w;
      double reg = 0.0005;

      m_cal_a += lr * (e * z - reg * (m_cal_a - 1.0));
      m_cal_b += lr * (e);
      m_cal_a = FXAI_Clamp(m_cal_a, 0.20, 5.00);
      m_cal_b = FXAI_Clamp(m_cal_b, -4.0, 4.0);

      int bins = 12;
      int bi = (int)MathFloor(pr * (double)bins);
      if(bi < 0) bi = 0;
      if(bi >= bins) bi = bins - 1;
      m_iso_cnt[bi] += w;
      m_iso_pos[bi] += w * (double)y;

      m_cal_steps++;
      if(m_cal_steps >= 20) m_cal_ready = true;
   }

   double CalibrateProb(const double prob_raw) const
   {
      double pr = FXAI_Clamp(prob_raw, 0.001, 0.999);
      double p_platt = FXAI_Sigmoid((m_cal_a * FXAI_Logit(pr)) + m_cal_b);
      if(!m_cal_ready)
         return FXAI_Clamp(p_platt, 0.001, 0.999);

      double total = 0.0;
      for(int i=0; i<12; i++) total += m_iso_cnt[i];
      if(total < 30.0)
         return FXAI_Clamp(p_platt, 0.001, 0.999);

      int bins = 12;
      int bi = (int)MathFloor(pr * (double)bins);
      if(bi < 0) bi = 0;
      if(bi >= bins) bi = bins - 1;

      double mono[12];
      double prev = 0.5;
      for(int i=0; i<12; i++)
      {
         double r = prev;
         if(m_iso_cnt[i] > 1e-9)
            r = m_iso_pos[i] / m_iso_cnt[i];
         r = FXAI_Clamp(r, 0.01, 0.99);
         if(i > 0 && r < mono[i - 1]) r = mono[i - 1];
         mono[i] = r;
         prev = r;
      }

      double p_iso = mono[bi];
      double p = (0.70 * p_platt) + (0.30 * p_iso);
      return FXAI_Clamp(p, 0.001, 0.999);
   }

   FXAIAIHyperParams ScaleHyperParamsForMove(const FXAIAIHyperParams &hp,
                                            const double move_points) const
   {
      FXAIAIHyperParams h = hp;
      double w = FXAI_MoveWeight(move_points);

      h.lr *= w;
      h.ftrl_alpha *= w;
      h.xgb_lr *= w;
      h.mlp_lr *= w;
      h.quantile_lr *= w;
      h.enhash_lr *= w;

      h.lr = FXAI_Clamp(h.lr, 0.0001, 1.0000);
      h.ftrl_alpha = FXAI_Clamp(h.ftrl_alpha, 0.0001, 5.0000);
      h.xgb_lr = FXAI_Clamp(h.xgb_lr, 0.0001, 1.0000);
      h.mlp_lr = FXAI_Clamp(h.mlp_lr, 0.0001, 1.0000);
      h.quantile_lr = FXAI_Clamp(h.quantile_lr, 0.0001, 1.0000);
      h.enhash_lr = FXAI_Clamp(h.enhash_lr, 0.0001, 1.0000);
      return h;
   }

   double MoveSampleWeight(const double &x[],
                           const double move_points) const
   {
      double cost = ResolveCostPoints(x);
      double w = FXAI_MoveEdgeWeight(move_points, cost);
      if(!m_target_quality_ready)
         return w;

      double move_scale = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      double mfe_bonus = FXAI_Clamp(m_target_mfe_points / MathMax(move_scale, 0.10), 0.0, 2.0);
      double mae_penalty = FXAI_Clamp(m_target_mae_points / MathMax(MathMax(m_target_mfe_points, move_scale), 0.10), 0.0, 1.5);
      double timing_bonus = 1.0 - FXAI_Clamp(m_target_hit_time_frac, 0.0, 1.0);
      double execution_drag = 0.60 * FXAI_Clamp(m_target_path_risk, 0.0, 1.0) +
                              0.40 * FXAI_Clamp(m_target_fill_risk, 0.0, 1.0);
      double q = 1.0 + 0.18 * mfe_bonus + 0.14 * timing_bonus - 0.16 * mae_penalty - 0.18 * execution_drag;
      return w * FXAI_Clamp(q, 0.45, 1.85);
   }

   void Softmax3(const double &logits[], double &probs[]) const
   {
      double m = logits[0];
      if(logits[1] > m) m = logits[1];
      if(logits[2] > m) m = logits[2];

      double e0 = MathExp(FXAI_Clamp(logits[0] - m, -30.0, 30.0));
      double e1 = MathExp(FXAI_Clamp(logits[1] - m, -30.0, 30.0));
      double e2 = MathExp(FXAI_Clamp(logits[2] - m, -30.0, 30.0));
      double s = e0 + e1 + e2;
      if(s <= 0.0)
      {
         probs[0] = 0.3333333;
         probs[1] = 0.3333333;
         probs[2] = 0.3333333;
         return;
      }

      probs[0] = e0 / s;
      probs[1] = e1 / s;
      probs[2] = e2 / s;
   }

   int ContextSessionBucket(void) const
   {
      if(m_ctx_session_bucket >= 0 && m_ctx_session_bucket < FXAI_PLUGIN_SESSION_BUCKETS)
         return m_ctx_session_bucket;
      return FXAI_DeriveSessionBucket(ResolveContextTime());
   }

   int ContextHorizonBucket(void) const
   {
      int h = (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1);
      if(h <= 1) return 0;
      if(h <= 3) return 1;
      if(h <= 5) return 2;
      if(h <= 8) return 3;
      if(h <= 13) return 4;
      if(h <= 21) return 5;
      if(h <= 34) return 6;
      return FXAI_PLUGIN_HORIZON_BUCKETS - 1;
   }

   void BuildCurrentContext(FXAIAIContextV4 &ctx) const
   {
      ctx.api_version = FXAI_API_VERSION_V4;
      ctx.regime_id = m_ctx_regime_id;
      ctx.session_bucket = ContextSessionBucket();
      ctx.horizon_minutes = m_ctx_horizon_minutes;
      ctx.feature_schema_id = m_ctx_feature_schema_id;
      ctx.normalization_method_id = m_ctx_normalization_method_id;
      ctx.sequence_bars = m_ctx_sequence_bars;
      ctx.cost_points = m_ctx_cost_points;
      ctx.min_move_points = m_ctx_min_move_points;
      ctx.point_value = ResolvePointValue();
      ctx.domain_hash = FXAI_Clamp(m_ctx_domain_hash, 0.0, 1.0);
      ctx.sample_time = ResolveContextTime();
   }

   void NormalizeClassDistribution(double &probs[]) const
   {
      for(int c=0; c<3; c++)
      {
         if(!MathIsValidNumber(probs[c])) probs[c] = 0.0;
         probs[c] = FXAI_Clamp(probs[c], 0.0005, 0.9990);
      }

      double s = probs[0] + probs[1] + probs[2];
      if(!MathIsValidNumber(s) || s <= 0.0)
      {
         probs[0] = 0.10;
         probs[1] = 0.10;
         probs[2] = 0.80;
         return;
      }

      for(int c=0; c<3; c++)
         probs[c] /= s;
   }

   void ApplyContextCalibrationBank(double &probs[])
   {
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();

      double total = m_bank_total[r][s][h];
      if(total <= 0.0)
      {
         NormalizeClassDistribution(probs);
         return;
      }

      double prior[3];
      double prior_total = 0.0;
      for(int c=0; c<3; c++)
         prior_total += m_bank_class_mass[r][s][h][c];
      if(prior_total <= 1e-9)
      {
         NormalizeClassDistribution(probs);
         return;
      }
      for(int c=0; c<3; c++)
         prior[c] = m_bank_class_mass[r][s][h][c] / prior_total;

      double mix = FXAI_Clamp(total / 120.0, 0.05, 0.35);
      for(int c=0; c<3; c++)
         probs[c] = (1.0 - mix) * probs[c] + mix * prior[c];

      NormalizeClassDistribution(probs);
   }

   double ApplyExpectedMoveCalibrationBank(const double expected_move_points)
   {
      double ev = expected_move_points;
      if(!MathIsValidNumber(ev) || ev <= 0.0)
         return 0.0;
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();

      ev = ev * m_bank_ev_scale[r][s][h] + m_bank_ev_bias[r][s][h];
      if(!MathIsValidNumber(ev) || ev <= 0.0)
         return 0.0;
      return ev;
   }

   void UpdateContextCalibrationBank(const int label_class,
                                     const double &probs[],
                                     const double expected_move_points,
                                     const double move_points,
                                     const double sample_w)
   {
      if(label_class < (int)FXAI_LABEL_SELL || label_class > (int)FXAI_LABEL_SKIP)
         return;

      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      double w = FXAI_Clamp(sample_w, 0.25, 4.00);

      m_bank_class_mass[r][s][h][label_class] += w;
      m_bank_total[r][s][h] += w;
      if(m_bank_total[r][s][h] > 30000.0)
      {
         for(int c=0; c<3; c++)
            m_bank_class_mass[r][s][h][c] *= 0.5;
         m_bank_total[r][s][h] *= 0.5;
      }

      double pred = MathMax(expected_move_points, MathMax(ResolveMinMovePoints(), 0.10));
      double tgt = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      double err = FXAI_ClipSym(tgt - pred, 30.0);

      double lr = 0.015 * w;
      double g_scale = err / MathMax(pred, 0.25);
      double g_bias = 0.30 * err;

      m_bank_ev_g2_scale[r][s][h] += g_scale * g_scale;
      m_bank_ev_g2_bias[r][s][h] += g_bias * g_bias;

      double lr_scale = lr / MathSqrt(m_bank_ev_g2_scale[r][s][h] + 1e-8);
      double lr_bias = lr / MathSqrt(m_bank_ev_g2_bias[r][s][h] + 1e-8);

      m_bank_ev_scale[r][s][h] += lr_scale * g_scale;
      m_bank_ev_bias[r][s][h] += lr_bias * g_bias;

      m_bank_ev_scale[r][s][h] = FXAI_Clamp(m_bank_ev_scale[r][s][h], 0.40, 2.50);
      m_bank_ev_bias[r][s][h] = FXAI_Clamp(m_bank_ev_bias[r][s][h], -20.0, 20.0);
   }

   double ComputeReplayPriority(const int label_class,
                                const double &probs[],
                                const double move_points,
                                const double cost_points,
                                const double min_move_points) const
   {
      int cls = label_class;
      if(cls < (int)FXAI_LABEL_SELL || cls > (int)FXAI_LABEL_SKIP)
         cls = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);

      double p_true = (cls >= 0 && cls < 3 ? probs[cls] : 0.3333333);
      double edge = MathMax(MathAbs(move_points) - MathMax(cost_points, 0.0), 0.0);
      double mm = MathMax(min_move_points, 0.10);
      double pri = 0.50 + (1.0 - FXAI_Clamp(p_true, 0.0, 1.0));
      pri += 0.35 * FXAI_Clamp(edge / mm, 0.0, 4.0);
      if(cls == (int)FXAI_LABEL_SKIP)
         pri += 0.15;
      return FXAI_Clamp(pri, 0.10, 8.00);
   }

   void StoreReplaySample(const FXAIAITrainRequestV4 &sample,
                          const double priority)
   {
      int slot = m_replay_head;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         m_replay_x[slot][k] = sample.x[k];
      m_replay_label[slot] = sample.label_class;
      m_replay_move[slot] = sample.move_points;
      m_replay_mfe[slot] = sample.mfe_points;
      m_replay_mae[slot] = sample.mae_points;
      m_replay_hit_time[slot] = sample.time_to_hit_frac;
      m_replay_path_flags[slot] = sample.path_flags;
      m_replay_path_risk[slot] = sample.path_risk;
      m_replay_fill_risk[slot] = sample.fill_risk;
      m_replay_masked_step[slot] = sample.masked_step_target;
      m_replay_next_vol[slot] = sample.next_vol_target;
      m_replay_regime_shift[slot] = sample.regime_shift_target;
      m_replay_context_lead[slot] = sample.context_lead_target;
      m_replay_cost[slot] = sample.ctx.cost_points;
      m_replay_min_move[slot] = sample.ctx.min_move_points;
      m_replay_time[slot] = sample.ctx.sample_time;
      m_replay_regime[slot] = sample.ctx.regime_id;
      m_replay_session_bucket[slot] = sample.ctx.session_bucket;
      m_replay_horizon[slot] = sample.ctx.horizon_minutes;
      m_replay_feature_schema[slot] = sample.ctx.feature_schema_id;
      m_replay_norm_method[slot] = sample.ctx.normalization_method_id;
      m_replay_sequence_bars[slot] = sample.ctx.sequence_bars;
      m_replay_point_value[slot] = sample.ctx.point_value;
      m_replay_domain_hash[slot] = sample.ctx.domain_hash;
      m_replay_window_size[slot] = sample.window_size;
      if(m_replay_window_size[slot] < 0) m_replay_window_size[slot] = 0;
      if(m_replay_window_size[slot] > FXAI_MAX_SEQUENCE_BARS) m_replay_window_size[slot] = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_window[slot][b][k] = (b < m_replay_window_size[slot] ? sample.x_window[b][k] : 0.0);
      }
      m_replay_priority[slot] = priority;

      m_replay_head = (m_replay_head + 1) % FXAI_PLUGIN_REPLAY_CAPACITY;
      if(m_replay_size < FXAI_PLUGIN_REPLAY_CAPACITY)
         m_replay_size++;
   }

