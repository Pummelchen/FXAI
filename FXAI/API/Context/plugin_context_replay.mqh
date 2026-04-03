   void RunReplayRehearsal(const FXAIAIHyperParams &hp,
                           const int regime_id,
                           const int horizon_minutes)
   {
      if(m_replay_size <= 0) return;

      int best_idx[FXAI_PLUGIN_REPLAY_STEPS];
      double best_score[FXAI_PLUGIN_REPLAY_STEPS];
      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         best_idx[j] = -1;
         best_score[j] = -1e18;
      }

      for(int i=0; i<m_replay_size; i++)
      {
         double score = m_replay_priority[i];
         if(m_replay_regime[i] == regime_id) score += 0.80;
         if(m_replay_horizon[i] == horizon_minutes) score += 0.60;

         for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
         {
            if(score > best_score[j])
            {
               for(int k=FXAI_PLUGIN_REPLAY_STEPS - 1; k>j; k--)
               {
                  best_score[k] = best_score[k - 1];
                  best_idx[k] = best_idx[k - 1];
               }
               best_score[j] = score;
               best_idx[j] = i;
               break;
            }
         }
      }

      datetime keep_time = m_ctx_time;
      bool keep_time_ready = m_ctx_time_ready;
      double keep_cost = m_ctx_cost_points;
      bool keep_cost_ready = m_ctx_cost_ready;
      double keep_min_move = m_ctx_min_move_points;
      int keep_regime = m_ctx_regime_id;
      int keep_session = m_ctx_session_bucket;
      int keep_horizon = m_ctx_horizon_minutes;
      int keep_feature_schema = m_ctx_feature_schema_id;
      int keep_norm_method = m_ctx_normalization_method_id;
      int keep_sequence_bars = m_ctx_sequence_bars;
      double keep_point_value = m_ctx_point_value;
      double keep_domain_hash = m_ctx_domain_hash;
      int keep_window_size = m_ctx_window_size;
      double keep_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            keep_window[b][k] = m_ctx_window[b][k];

      for(int j=0; j<FXAI_PLUGIN_REPLAY_STEPS; j++)
      {
         int idx = best_idx[j];
         if(idx < 0 || idx >= m_replay_size) continue;
         FXAIAIContextV4 replay_ctx;
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
         m_target_quality_ready = true;
         m_target_mfe_points = m_replay_mfe[idx];
         m_target_mae_points = m_replay_mae[idx];
         m_target_hit_time_frac = m_replay_hit_time[idx];
         m_target_path_flags = m_replay_path_flags[idx];
         m_target_path_risk = m_replay_path_risk[idx];
         m_target_fill_risk = m_replay_fill_risk[idx];
         m_target_masked_step = m_replay_masked_step[idx];
         m_target_next_vol = m_replay_next_vol[idx];
         m_target_regime_shift = m_replay_regime_shift[idx];
         m_target_context_lead = m_replay_context_lead[idx];
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
         m_replay_rehearsals++;
      }

      m_ctx_time_ready = keep_time_ready;
      m_ctx_time = keep_time;
      m_ctx_cost_ready = keep_cost_ready;
      m_ctx_cost_points = keep_cost;
      m_ctx_min_move_points = keep_min_move;
      m_ctx_regime_id = keep_regime;
      m_ctx_session_bucket = keep_session;
      m_ctx_horizon_minutes = keep_horizon;
      m_ctx_feature_schema_id = keep_feature_schema;
      m_ctx_normalization_method_id = keep_norm_method;
      m_ctx_sequence_bars = keep_sequence_bars;
      m_ctx_point_value = keep_point_value;
      m_ctx_domain_hash = keep_domain_hash;
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
      m_ctx_window_size = keep_window_size;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = keep_window[b][k];
   }

