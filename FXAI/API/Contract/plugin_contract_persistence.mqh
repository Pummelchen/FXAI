   bool SaveBasePersistentState(const int handle) const
   {
      FileWriteInteger(handle, (m_move_ready ? 1 : 0));
      FileWriteDouble(handle, m_move_ema_abs);
      FileWriteInteger(handle, (m_move_head_ready ? 1 : 0));
      FileWriteInteger(handle, m_move_head_steps);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         FileWriteDouble(handle, m_move_w[i]);

      FileWriteInteger(handle, (m_cal_ready ? 1 : 0));
      FileWriteInteger(handle, m_cal_steps);
      FileWriteDouble(handle, m_cal_a);
      FileWriteDouble(handle, m_cal_b);
      for(int i=0; i<12; i++)
      {
         FileWriteDouble(handle, m_iso_pos[i]);
         FileWriteDouble(handle, m_iso_cnt[i]);
      }

      FileWriteInteger(handle, (m_shared_adapter_ready ? 1 : 0));
      FileWriteInteger(handle, m_shared_adapter_steps);
      FileWriteInteger(handle, (m_shared_backbone_ready ? 1 : 0));
      FileWriteInteger(handle, m_shared_backbone_steps);
      for(int c=0; c<3; c++)
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            FileWriteDouble(handle, m_shared_cls_w[c][i]);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         FileWriteDouble(handle, m_shared_move_w[i]);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         FileWriteDouble(handle, m_shared_backbone_b[j]);
         FileWriteDouble(handle, m_shared_backbone_move[j]);
         for(int c=0; c<3; c++)
            FileWriteDouble(handle, m_shared_backbone_cls[c][j]);
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            FileWriteDouble(handle, m_shared_backbone_w[j][i]);
         for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
            FileWriteDouble(handle, m_shared_backbone_seq_w[j][t]);
         for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
            FileWriteDouble(handle, m_shared_backbone_time_w[j][c]);
         for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
            FileWriteDouble(handle, m_shared_backbone_time_gate_w[j][c]);
         for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
            FileWriteDouble(handle, m_shared_backbone_state_w[j][c]);
         FileWriteDouble(handle, m_shared_backbone_state_rec_w[j]);
         FileWriteDouble(handle, m_shared_backbone_state_b[j]);
      }
      for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            FileWriteDouble(handle, m_shared_domain_emb[d][j]);
      for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            FileWriteDouble(handle, m_shared_horizon_emb[h][j]);
      for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            FileWriteDouble(handle, m_shared_session_emb[s][j]);
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         FileWriteDouble(handle, m_transfer_slot_obs[slot]);
         FileWriteDouble(handle, m_transfer_slot_align[slot]);
         FileWriteDouble(handle, m_transfer_slot_lead[slot]);
         FileWriteDouble(handle, m_transfer_slot_move[slot]);
      }

      FileWriteInteger(handle, (m_quality_head_ready ? 1 : 0));
      FileWriteDouble(handle, m_quality_mfe_ema);
      FileWriteDouble(handle, m_quality_mae_ema);
      FileWriteDouble(handle, m_quality_hit_ema);
      FileWriteDouble(handle, m_quality_path_risk_ema);
      FileWriteDouble(handle, m_quality_fill_risk_ema);
      for(int r=0; r<FXAI_PLUGIN_REGIME_BUCKETS; r++)
      {
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         {
            for(int h=0; h<FXAI_PLUGIN_HORIZON_BUCKETS; h++)
            {
               FileWriteInteger(handle, (m_quality_bank_ready[r][s][h] ? 1 : 0));
               FileWriteDouble(handle, m_quality_bank_obs[r][s][h]);
               FileWriteDouble(handle, m_quality_bank_mfe[r][s][h]);
               FileWriteDouble(handle, m_quality_bank_mae[r][s][h]);
               FileWriteDouble(handle, m_quality_bank_hit[r][s][h]);
               FileWriteDouble(handle, m_quality_bank_path[r][s][h]);
               FileWriteDouble(handle, m_quality_bank_fill[r][s][h]);
               FileWriteDouble(handle, m_bank_total[r][s][h]);
               FileWriteDouble(handle, m_bank_ev_scale[r][s][h]);
               FileWriteDouble(handle, m_bank_ev_bias[r][s][h]);
               FileWriteDouble(handle, m_bank_ev_g2_scale[r][s][h]);
               FileWriteDouble(handle, m_bank_ev_g2_bias[r][s][h]);
               for(int c=0; c<3; c++)
                  FileWriteDouble(handle, m_bank_class_mass[r][s][h][c]);
            }
         }
      }

      if(!m_native_quality_heads.Save(handle))
         return false;

      FileWriteInteger(handle, m_replay_head);
      FileWriteInteger(handle, m_replay_size);
      for(int i=0; i<FXAI_PLUGIN_REPLAY_CAPACITY; i++)
      {
         FileWriteInteger(handle, m_replay_label[i]);
         FileWriteDouble(handle, m_replay_move[i]);
         FileWriteDouble(handle, m_replay_mfe[i]);
         FileWriteDouble(handle, m_replay_mae[i]);
         FileWriteDouble(handle, m_replay_hit_time[i]);
         FileWriteInteger(handle, m_replay_path_flags[i]);
         FileWriteDouble(handle, m_replay_path_risk[i]);
         FileWriteDouble(handle, m_replay_fill_risk[i]);
         FileWriteDouble(handle, m_replay_masked_step[i]);
         FileWriteDouble(handle, m_replay_next_vol[i]);
         FileWriteDouble(handle, m_replay_regime_shift[i]);
         FileWriteDouble(handle, m_replay_context_lead[i]);
         FileWriteDouble(handle, m_replay_cost[i]);
         FileWriteDouble(handle, m_replay_min_move[i]);
         FileWriteLong(handle, (long)m_replay_time[i]);
         FileWriteInteger(handle, m_replay_regime[i]);
         FileWriteInteger(handle, m_replay_session_bucket[i]);
         FileWriteInteger(handle, m_replay_horizon[i]);
         FileWriteInteger(handle, m_replay_feature_schema[i]);
         FileWriteInteger(handle, m_replay_norm_method[i]);
         FileWriteInteger(handle, m_replay_sequence_bars[i]);
         FileWriteDouble(handle, m_replay_point_value[i]);
         FileWriteDouble(handle, m_replay_domain_hash[i]);
         FileWriteInteger(handle, m_replay_window_size[i]);
         FileWriteDouble(handle, m_replay_priority[i]);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            FileWriteDouble(handle, m_replay_x[i][k]);
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               FileWriteDouble(handle, m_replay_window[i][b][k]);
      }

      FileWriteInteger(handle, m_core_predict_calls);
      FileWriteInteger(handle, m_core_predict_failures);
      FileWriteInteger(handle, m_replay_rehearsals);
      FileWriteInteger(handle, (m_rng_seeded ? 1 : 0));
      FileWriteInteger(handle, (int)m_rng_state);
      FileWriteInteger(handle, (m_persist_hp_ready ? 1 : 0));
      FileWriteInteger(handle, m_persist_train_events);
      FileWriteDouble(handle, m_persist_hp.lr);
      FileWriteDouble(handle, m_persist_hp.l2);
      FileWriteDouble(handle, m_persist_hp.ftrl_alpha);
      FileWriteDouble(handle, m_persist_hp.ftrl_beta);
      FileWriteDouble(handle, m_persist_hp.ftrl_l1);
      FileWriteDouble(handle, m_persist_hp.ftrl_l2);
      FileWriteDouble(handle, m_persist_hp.pa_c);
      FileWriteDouble(handle, m_persist_hp.pa_margin);
      FileWriteDouble(handle, m_persist_hp.xgb_lr);
      FileWriteDouble(handle, m_persist_hp.xgb_l2);
      FileWriteDouble(handle, m_persist_hp.xgb_split);
      FileWriteDouble(handle, m_persist_hp.mlp_lr);
      FileWriteDouble(handle, m_persist_hp.mlp_l2);
      FileWriteDouble(handle, m_persist_hp.mlp_init);
      FileWriteDouble(handle, m_persist_hp.quantile_lr);
      FileWriteDouble(handle, m_persist_hp.quantile_l2);
      FileWriteDouble(handle, m_persist_hp.enhash_lr);
      FileWriteDouble(handle, m_persist_hp.enhash_l1);
      FileWriteDouble(handle, m_persist_hp.enhash_l2);
      FileWriteDouble(handle, m_persist_hp.tcn_layers);
      FileWriteDouble(handle, m_persist_hp.tcn_kernel);
      FileWriteDouble(handle, m_persist_hp.tcn_dilation_base);
      return true;
   }

   bool LoadBasePersistentState(const int handle,
                                const int version)
   {
      m_move_ready = (FileReadInteger(handle) != 0);
      m_move_ema_abs = FileReadDouble(handle);
      m_move_head_ready = (FileReadInteger(handle) != 0);
      m_move_head_steps = FileReadInteger(handle);
      for(int i=0; i<FXAI_AI_WEIGHTS; i++)
         m_move_w[i] = FileReadDouble(handle);

      m_cal_ready = (FileReadInteger(handle) != 0);
      m_cal_steps = FileReadInteger(handle);
      m_cal_a = FileReadDouble(handle);
      m_cal_b = FileReadDouble(handle);
      for(int i=0; i<12; i++)
      {
         m_iso_pos[i] = FileReadDouble(handle);
         m_iso_cnt[i] = FileReadDouble(handle);
      }

      m_shared_adapter_ready = (FileReadInteger(handle) != 0);
      m_shared_adapter_steps = FileReadInteger(handle);
      if(version >= 3)
      {
         m_shared_backbone_ready = (FileReadInteger(handle) != 0);
         m_shared_backbone_steps = FileReadInteger(handle);
      }
      else
      {
         m_shared_backbone_ready = false;
         m_shared_backbone_steps = 0;
      }
      for(int c=0; c<3; c++)
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_cls_w[c][i] = FileReadDouble(handle);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         m_shared_move_w[i] = FileReadDouble(handle);
      if(version >= 3)
      {
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         {
            m_shared_backbone_b[j] = FileReadDouble(handle);
            m_shared_backbone_move[j] = FileReadDouble(handle);
            for(int c=0; c<3; c++)
               m_shared_backbone_cls[c][j] = FileReadDouble(handle);
            for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
               m_shared_backbone_w[j][i] = FileReadDouble(handle);
            if(version >= 9)
            {
               for(int t=0; t<FXAI_SHARED_TRANSFER_SEQUENCE_TOKENS; t++)
                  m_shared_backbone_seq_w[j][t] = FileReadDouble(handle);
            }
            if(version >= 10)
            {
               for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
                  m_shared_backbone_time_w[j][c] = FileReadDouble(handle);
               for(int c=0; c<FXAI_SHARED_TRANSFER_BAR_FEATURES; c++)
                  m_shared_backbone_time_gate_w[j][c] = FileReadDouble(handle);
            }
            if(version >= 11)
            {
               for(int c=0; c<FXAI_SHARED_TRANSFER_STATE_FEATURES; c++)
                  m_shared_backbone_state_w[j][c] = FileReadDouble(handle);
               m_shared_backbone_state_rec_w[j] = FileReadDouble(handle);
               m_shared_backbone_state_b[j] = FileReadDouble(handle);
            }
         }
         for(int d=0; d<FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS; d++)
            for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               m_shared_domain_emb[d][j] = FileReadDouble(handle);
         for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
            for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               m_shared_horizon_emb[h][j] = FileReadDouble(handle);
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
            for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
               m_shared_session_emb[s][j] = FileReadDouble(handle);
      }
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         m_transfer_slot_obs[slot] = FileReadDouble(handle);
         m_transfer_slot_align[slot] = FileReadDouble(handle);
         m_transfer_slot_lead[slot] = FileReadDouble(handle);
         m_transfer_slot_move[slot] = FileReadDouble(handle);
      }

      m_quality_head_ready = (FileReadInteger(handle) != 0);
      m_quality_mfe_ema = FileReadDouble(handle);
      m_quality_mae_ema = FileReadDouble(handle);
      m_quality_hit_ema = FileReadDouble(handle);
      m_quality_path_risk_ema = FileReadDouble(handle);
      m_quality_fill_risk_ema = FileReadDouble(handle);
      for(int r=0; r<FXAI_PLUGIN_REGIME_BUCKETS; r++)
      {
         for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
         {
            for(int h=0; h<FXAI_PLUGIN_HORIZON_BUCKETS; h++)
            {
               m_quality_bank_ready[r][s][h] = (FileReadInteger(handle) != 0);
               m_quality_bank_obs[r][s][h] = FileReadDouble(handle);
               m_quality_bank_mfe[r][s][h] = FileReadDouble(handle);
               m_quality_bank_mae[r][s][h] = FileReadDouble(handle);
               m_quality_bank_hit[r][s][h] = FileReadDouble(handle);
               m_quality_bank_path[r][s][h] = FileReadDouble(handle);
               m_quality_bank_fill[r][s][h] = FileReadDouble(handle);
               m_bank_total[r][s][h] = FileReadDouble(handle);
               m_bank_ev_scale[r][s][h] = FileReadDouble(handle);
               m_bank_ev_bias[r][s][h] = FileReadDouble(handle);
               m_bank_ev_g2_scale[r][s][h] = FileReadDouble(handle);
               m_bank_ev_g2_bias[r][s][h] = FileReadDouble(handle);
               for(int c=0; c<3; c++)
                  m_bank_class_mass[r][s][h][c] = FileReadDouble(handle);
            }
         }
      }

      if(!m_native_quality_heads.Load(handle))
         return false;

      m_replay_head = FileReadInteger(handle);
      m_replay_size = FileReadInteger(handle);
      for(int i=0; i<FXAI_PLUGIN_REPLAY_CAPACITY; i++)
      {
         m_replay_label[i] = FileReadInteger(handle);
         m_replay_move[i] = FileReadDouble(handle);
         m_replay_mfe[i] = FileReadDouble(handle);
         m_replay_mae[i] = FileReadDouble(handle);
         m_replay_hit_time[i] = FileReadDouble(handle);
         m_replay_path_flags[i] = FileReadInteger(handle);
         m_replay_path_risk[i] = FileReadDouble(handle);
         m_replay_fill_risk[i] = FileReadDouble(handle);
         m_replay_masked_step[i] = FileReadDouble(handle);
         m_replay_next_vol[i] = FileReadDouble(handle);
         m_replay_regime_shift[i] = FileReadDouble(handle);
         m_replay_context_lead[i] = FileReadDouble(handle);
         m_replay_cost[i] = FileReadDouble(handle);
         m_replay_min_move[i] = FileReadDouble(handle);
         m_replay_time[i] = (datetime)FileReadLong(handle);
         m_replay_regime[i] = FileReadInteger(handle);
         m_replay_session_bucket[i] = FileReadInteger(handle);
         m_replay_horizon[i] = FileReadInteger(handle);
         m_replay_feature_schema[i] = FileReadInteger(handle);
         m_replay_norm_method[i] = FileReadInteger(handle);
         m_replay_sequence_bars[i] = FileReadInteger(handle);
         m_replay_point_value[i] = FileReadDouble(handle);
         m_replay_domain_hash[i] = FileReadDouble(handle);
         m_replay_window_size[i] = FileReadInteger(handle);
         m_replay_priority[i] = FileReadDouble(handle);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_replay_x[i][k] = FileReadDouble(handle);
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_replay_window[i][b][k] = FileReadDouble(handle);
      }

      m_core_predict_calls = FileReadInteger(handle);
      m_core_predict_failures = FileReadInteger(handle);
      m_replay_rehearsals = FileReadInteger(handle);
      m_rng_seeded = (FileReadInteger(handle) != 0);
      m_rng_state = (uint)FileReadInteger(handle);
      if(version >= 6)
      {
         m_persist_hp_ready = (FileReadInteger(handle) != 0);
         m_persist_train_events = FileReadInteger(handle);
         m_persist_hp.lr = FileReadDouble(handle);
         m_persist_hp.l2 = FileReadDouble(handle);
         m_persist_hp.ftrl_alpha = FileReadDouble(handle);
         m_persist_hp.ftrl_beta = FileReadDouble(handle);
         m_persist_hp.ftrl_l1 = FileReadDouble(handle);
         m_persist_hp.ftrl_l2 = FileReadDouble(handle);
         m_persist_hp.pa_c = FileReadDouble(handle);
         m_persist_hp.pa_margin = FileReadDouble(handle);
         m_persist_hp.xgb_lr = FileReadDouble(handle);
         m_persist_hp.xgb_l2 = FileReadDouble(handle);
         m_persist_hp.xgb_split = FileReadDouble(handle);
         m_persist_hp.mlp_lr = FileReadDouble(handle);
         m_persist_hp.mlp_l2 = FileReadDouble(handle);
         m_persist_hp.mlp_init = FileReadDouble(handle);
         m_persist_hp.quantile_lr = FileReadDouble(handle);
         m_persist_hp.quantile_l2 = FileReadDouble(handle);
         m_persist_hp.enhash_lr = FileReadDouble(handle);
         m_persist_hp.enhash_l1 = FileReadDouble(handle);
         m_persist_hp.enhash_l2 = FileReadDouble(handle);
         m_persist_hp.tcn_layers = FileReadDouble(handle);
         m_persist_hp.tcn_kernel = FileReadDouble(handle);
         m_persist_hp.tcn_dilation_base = FileReadDouble(handle);
      }
      else
      {
         m_persist_hp_ready = false;
         m_persist_train_events = 0;
      }
      return true;
   }

   bool RebuildModelStateFromReplay(const FXAIAIHyperParams &hp)
   {
      if(m_replay_size <= 0)
         return true;

      EnsureInitialized(hp);

      int start = (m_replay_size < FXAI_PLUGIN_REPLAY_CAPACITY ? 0 : m_replay_head);
      for(int n=0; n<m_replay_size; n++)
      {
         int idx = (start + n) % FXAI_PLUGIN_REPLAY_CAPACITY;
         FXAIAIContextV4 replay_ctx;
         FXAI_ClearContextV4(replay_ctx);
         replay_ctx.api_version = FXAI_API_VERSION_V4;
         replay_ctx.regime_id = m_replay_regime[idx];
         replay_ctx.session_bucket = m_replay_session_bucket[idx];
         replay_ctx.horizon_minutes = m_replay_horizon[idx];
         replay_ctx.feature_schema_id = m_replay_feature_schema[idx];
         replay_ctx.normalization_method_id = m_replay_norm_method[idx];
         replay_ctx.sequence_bars = m_replay_sequence_bars[idx];
         replay_ctx.cost_points = m_replay_cost[idx];
         replay_ctx.min_move_points = m_replay_min_move[idx];
         replay_ctx.point_value = m_replay_point_value[idx];
         replay_ctx.domain_hash = m_replay_domain_hash[idx];
         replay_ctx.sample_time = m_replay_time[idx];
         SetContext(replay_ctx);
         SetTrainingTargetsRaw(m_replay_mfe[idx],
                               m_replay_mae[idx],
                               m_replay_hit_time[idx],
                               m_replay_path_flags[idx],
                               m_replay_path_risk[idx],
                               m_replay_fill_risk[idx],
                               m_replay_masked_step[idx],
                               m_replay_next_vol[idx],
                               m_replay_regime_shift[idx],
                               m_replay_context_lead[idx]);
         m_ctx_window_size = m_replay_window_size[idx];
         if(m_ctx_window_size < 0) m_ctx_window_size = 0;
         if(m_ctx_window_size > FXAI_MAX_SEQUENCE_BARS) m_ctx_window_size = FXAI_MAX_SEQUENCE_BARS;
         for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         {
            for(int k=0; k<FXAI_AI_WEIGHTS; k++)
               m_ctx_window[b][k] = (b < m_ctx_window_size ? m_replay_window[idx][b][k] : 0.0);
         }

         double replay_x[FXAI_AI_WEIGHTS];
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            replay_x[k] = m_replay_x[idx][k];
         TrainModelCore(m_replay_label[idx], replay_x, hp, m_replay_move[idx]);
      }
      return true;
   }

   double ReplayCheckpointChecksum(void) const
   {
      double acc = 0.0;
      int idx6 = (FXAI_AI_WEIGHTS - 1 < 6 ? FXAI_AI_WEIGHTS - 1 : 6);
      int idx24 = (FXAI_AI_WEIGHTS - 1 < 24 ? FXAI_AI_WEIGHTS - 1 : 24);
      int idx_macro = (FXAI_AI_WEIGHTS - 1 < FXAI_MACRO_EVENT_FEATURE_OFFSET + 2 ? FXAI_AI_WEIGHTS - 1 : FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
      for(int n=0; n<m_replay_size; n++)
      {
         int idx = (m_replay_head - 1 - n);
         while(idx < 0) idx += FXAI_PLUGIN_REPLAY_CAPACITY;
         idx %= FXAI_PLUGIN_REPLAY_CAPACITY;
         acc += 0.13 * (double)(n + 1) * (double)(m_replay_label[idx] + 2);
         acc += 0.07 * m_replay_move[idx];
         acc += 0.03 * m_replay_mfe[idx];
         acc += 0.03 * m_replay_mae[idx];
         acc += 0.02 * m_replay_priority[idx];
         acc += 0.01 * (double)m_replay_window_size[idx];
         acc += 0.005 * m_replay_domain_hash[idx];
         acc += 0.004 * (double)m_replay_horizon[idx];
         acc += 0.002 * (double)(m_replay_time[idx] % 1000003);
         acc += 0.011 * m_replay_x[idx][0];
         acc += 0.009 * m_replay_x[idx][idx6];
         acc += 0.007 * m_replay_x[idx][idx24];
         acc += 0.005 * m_replay_x[idx][idx_macro];
      }
      double frac = acc / 131071.0;
      return frac - MathFloor(frac);
   }

   double ReplayHyperParamsChecksum(void) const
   {
      double acc = 0.0;
      acc += 0.41 * m_persist_hp.lr;
      acc += 0.23 * m_persist_hp.l2;
      acc += 0.17 * m_persist_hp.ftrl_alpha;
      acc += 0.13 * m_persist_hp.ftrl_beta;
      acc += 0.11 * m_persist_hp.ftrl_l1;
      acc += 0.11 * m_persist_hp.ftrl_l2;
      acc += 0.09 * m_persist_hp.pa_c;
      acc += 0.08 * m_persist_hp.pa_margin;
      acc += 0.07 * m_persist_hp.xgb_lr;
      acc += 0.07 * m_persist_hp.xgb_l2;
      acc += 0.06 * m_persist_hp.mlp_lr;
      acc += 0.06 * m_persist_hp.mlp_l2;
      acc += 0.05 * m_persist_hp.quantile_lr;
      acc += 0.05 * m_persist_hp.quantile_l2;
      acc += 0.04 * m_persist_hp.enhash_lr;
      acc += 0.04 * m_persist_hp.enhash_l1;
      acc += 0.04 * m_persist_hp.enhash_l2;
      acc += 0.03 * m_persist_hp.tcn_layers;
      acc += 0.03 * m_persist_hp.tcn_kernel;
      acc += 0.03 * m_persist_hp.tcn_dilation_base;
      double frac = acc / 7919.0;
      return frac - MathFloor(frac);
   }

   virtual bool SaveModelState(const int handle) const
   {
      FileWriteInteger(handle, 0x46585250);
      FileWriteInteger(handle, m_replay_size);
      FileWriteDouble(handle, ReplayCheckpointChecksum());
      FileWriteDouble(handle, (m_persist_hp_ready ? ReplayHyperParamsChecksum() : 0.0));
      FileWriteInteger(handle, m_persist_train_events);
      return true;
   }

   virtual bool LoadModelState(const int handle, const int version)
   {
      int marker = FileReadInteger(handle);
      if(marker != 0x46585250)
         return false;
      if(version >= 10)
      {
         int expected_replay = FileReadInteger(handle);
         double expected_replay_ck = FileReadDouble(handle);
         double expected_hp_ck = FileReadDouble(handle);
         int expected_train_events = FileReadInteger(handle);
         if(expected_replay != m_replay_size)
            return false;
         if(MathAbs(expected_replay_ck - ReplayCheckpointChecksum()) > 1e-6)
            return false;
         if(m_persist_hp_ready && MathAbs(expected_hp_ck - ReplayHyperParamsChecksum()) > 1e-6)
            return false;
         if(expected_train_events != m_persist_train_events)
            return false;
      }
      if(!m_persist_hp_ready || m_replay_size <= 0)
         return true;
      return RebuildModelStateFromReplay(m_persist_hp);
   }

   virtual bool PredictModelCore(const double &x[],
                                 const FXAIAIHyperParams &hp,
                                 double &class_probs[],
                                 double &move_mean_points) = 0;

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points) = 0;
