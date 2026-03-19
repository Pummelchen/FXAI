   bool   m_move_ready;
   double m_move_ema_abs;
   bool   m_move_head_ready;
   int    m_move_head_steps;
   double m_move_w[FXAI_AI_WEIGHTS];

   bool   m_cal_ready;
   int    m_cal_steps;
   double m_cal_a;
   double m_cal_b;
   double m_iso_pos[12];
   double m_iso_cnt[12];
   bool   m_shared_adapter_ready;
   int    m_shared_adapter_steps;
   bool   m_shared_backbone_ready;
   int    m_shared_backbone_steps;
   double m_shared_cls_w[3][FXAI_SHARED_TRANSFER_FEATURES];
   double m_shared_move_w[FXAI_SHARED_TRANSFER_FEATURES];
   double m_shared_backbone_w[FXAI_SHARED_TRANSFER_LATENT][FXAI_SHARED_TRANSFER_FEATURES];
   double m_shared_backbone_b[FXAI_SHARED_TRANSFER_LATENT];
   double m_shared_backbone_cls[3][FXAI_SHARED_TRANSFER_LATENT];
   double m_shared_backbone_move[FXAI_SHARED_TRANSFER_LATENT];
   double m_shared_domain_emb[FXAI_SHARED_TRANSFER_DOMAIN_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];
   double m_shared_horizon_emb[FXAI_SHARED_TRANSFER_HORIZON_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];
   double m_shared_session_emb[FXAI_PLUGIN_SESSION_BUCKETS][FXAI_SHARED_TRANSFER_LATENT];
   double m_transfer_slot_obs[FXAI_CONTEXT_TOP_SYMBOLS];
   double m_transfer_slot_align[FXAI_CONTEXT_TOP_SYMBOLS];
   double m_transfer_slot_lead[FXAI_CONTEXT_TOP_SYMBOLS];
   double m_transfer_slot_move[FXAI_CONTEXT_TOP_SYMBOLS];

   // V4 context payload (set by Train/Predict).
   bool     m_ctx_time_ready;
   datetime m_ctx_time;
   bool     m_ctx_cost_ready;
   double   m_ctx_cost_points;
   double   m_ctx_min_move_points;
   int      m_ctx_regime_id;
   int      m_ctx_session_bucket;
   int      m_ctx_horizon_minutes;
   int      m_ctx_feature_schema_id;
   int      m_ctx_normalization_method_id;
   int      m_ctx_sequence_bars;
   double   m_ctx_point_value;
   double   m_ctx_domain_hash;
   int      m_ctx_window_size;
   double   m_ctx_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   bool     m_target_quality_ready;
   double   m_target_mfe_points;
   double   m_target_mae_points;
   double   m_target_hit_time_frac;
   int      m_target_path_flags;
   double   m_target_path_risk;
   double   m_target_fill_risk;
   double   m_target_masked_step;
   double   m_target_next_vol;
   double   m_target_regime_shift;
   double   m_target_context_lead;
   bool     m_quality_head_ready;
   double   m_quality_mfe_ema;
   double   m_quality_mae_ema;
   double   m_quality_hit_ema;
   double   m_quality_path_risk_ema;
   double   m_quality_fill_risk_ema;
   bool     m_quality_bank_ready[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_obs[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_mfe[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_mae[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_hit[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_path[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double   m_quality_bank_fill[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   CFXAINativeQualityHeads m_native_quality_heads;

   double m_bank_class_mass[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS][3];
   double m_bank_total[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_scale[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];
   double m_bank_ev_g2_bias[FXAI_PLUGIN_REGIME_BUCKETS][FXAI_PLUGIN_SESSION_BUCKETS][FXAI_PLUGIN_HORIZON_BUCKETS];

   int      m_replay_head;
   int      m_replay_size;
   double   m_replay_x[FXAI_PLUGIN_REPLAY_CAPACITY][FXAI_AI_WEIGHTS];
   int      m_replay_label[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_mfe[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_mae[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_hit_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_path_flags[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_path_risk[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_fill_risk[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_masked_step[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_next_vol[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_regime_shift[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_context_lead[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_cost[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_min_move[FXAI_PLUGIN_REPLAY_CAPACITY];
   datetime m_replay_time[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_regime[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_session_bucket[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_horizon[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_feature_schema[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_norm_method[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_sequence_bars[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_point_value[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_domain_hash[FXAI_PLUGIN_REPLAY_CAPACITY];
   int      m_replay_window_size[FXAI_PLUGIN_REPLAY_CAPACITY];
   double   m_replay_window[FXAI_PLUGIN_REPLAY_CAPACITY][FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   double   m_replay_priority[FXAI_PLUGIN_REPLAY_CAPACITY];

   int      m_core_predict_calls;
   int      m_core_predict_failures;
   int      m_replay_rehearsals;
   bool m_rng_seeded;
   uint m_rng_state;

   double InputCostProxyPoints(const double &x[]) const
   {
      if(m_ctx_cost_ready && m_ctx_cost_points >= 0.0)
         return m_ctx_cost_points;

      // Fallback when context has not been set explicitly before a dry-run call.
      if(ArraySize(x) <= 7) return 0.0;
      return MathMax(0.0, MathAbs(x[7]));
   }

   double ResolveCostPoints(const double &x[]) const
   {
      return InputCostProxyPoints(x);
   }

   double ResolveMinMovePoints(void) const
   {
      if(m_ctx_min_move_points > 0.0) return m_ctx_min_move_points;
      return 0.0;
   }

   double ResolvePointValue(void) const
   {
      if(MathIsValidNumber(m_ctx_point_value) && m_ctx_point_value > 0.0)
         return m_ctx_point_value;
      return (_Point > 0.0 ? _Point : 1.0);
   }

   datetime ResolveContextTime(void) const
   {
      if(m_ctx_time_ready && m_ctx_time > 0) return m_ctx_time;
      return TimeCurrent();
   }

   void SetContext(const FXAIAIContextV4 &ctx)
   {
      m_ctx_time_ready = (ctx.sample_time > 0);
      m_ctx_time = (m_ctx_time_ready ? ctx.sample_time : 0);

      m_ctx_cost_ready = (MathIsValidNumber(ctx.cost_points) && ctx.cost_points >= 0.0);
      m_ctx_cost_points = (m_ctx_cost_ready ? ctx.cost_points : 0.0);

      if(MathIsValidNumber(ctx.min_move_points) && ctx.min_move_points > 0.0)
         m_ctx_min_move_points = ctx.min_move_points;
      else
         m_ctx_min_move_points = 0.0;

      if(ctx.regime_id >= 0 && ctx.regime_id < FXAI_PLUGIN_REGIME_BUCKETS)
         m_ctx_regime_id = ctx.regime_id;
      else
         m_ctx_regime_id = 0;

      if(ctx.session_bucket >= 0 && ctx.session_bucket < FXAI_PLUGIN_SESSION_BUCKETS)
         m_ctx_session_bucket = ctx.session_bucket;
      else
         m_ctx_session_bucket = FXAI_DeriveSessionBucket(ctx.sample_time);

      m_ctx_horizon_minutes = (ctx.horizon_minutes > 0 ? ctx.horizon_minutes : 1);
      m_ctx_feature_schema_id = ctx.feature_schema_id;
      if(m_ctx_feature_schema_id < FXAI_SCHEMA_FULL || m_ctx_feature_schema_id > FXAI_SCHEMA_CONTEXTUAL)
         m_ctx_feature_schema_id = FXAI_SCHEMA_FULL;
      m_ctx_normalization_method_id = ctx.normalization_method_id;
      if(m_ctx_normalization_method_id < 0 || m_ctx_normalization_method_id >= FXAI_NORM_METHOD_COUNT)
         m_ctx_normalization_method_id = FXAI_NORM_EXISTING;
      m_ctx_sequence_bars = (ctx.sequence_bars > 0 ? ctx.sequence_bars : 1);
      if(m_ctx_sequence_bars > FXAI_MAX_SEQUENCE_BARS)
         m_ctx_sequence_bars = FXAI_MAX_SEQUENCE_BARS;
      m_ctx_point_value = (MathIsValidNumber(ctx.point_value) && ctx.point_value > 0.0
                           ? ctx.point_value : (_Point > 0.0 ? _Point : 1.0));
      m_ctx_domain_hash = (MathIsValidNumber(ctx.domain_hash) && ctx.domain_hash >= 0.0 && ctx.domain_hash <= 1.0
                           ? ctx.domain_hash : FXAI_SymbolHash01(_Symbol));
   }

   void SetWindowPayload(const int window_size,
                         const double &x_window[][FXAI_AI_WEIGHTS])
   {
      m_ctx_window_size = window_size;
      if(m_ctx_window_size < 0) m_ctx_window_size = 0;
      if(m_ctx_window_size > FXAI_MAX_SEQUENCE_BARS) m_ctx_window_size = FXAI_MAX_SEQUENCE_BARS;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = (b < m_ctx_window_size ? x_window[b][k] : 0.0);
      }
   }

   void ClearWindowPayload(void)
   {
      m_ctx_window_size = 0;
      for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            m_ctx_window[b][k] = 0.0;
   }

   void BuildSharedAdapterInput(const double &x[],
                                double &out[]) const
   {
      ArrayResize(out, FXAI_SHARED_TRANSFER_FEATURES);
      out[0] = 1.0;
      out[1] = FXAI_GetInputFeature(x, 62);
      out[2] = FXAI_GetInputFeature(x, 63);
      out[3] = FXAI_GetInputFeature(x, 64);
      out[4] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);
      double ret_mix = 0.0;
      double lag_mix = 0.0;
      double rel_mix = 0.0;
      double corr_mix = 0.0;
      double weight_total = 0.0;
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         int base = 50 + slot * 4;
         double ctx_ret = FXAI_GetInputFeature(x, base + 0);
         double ctx_lag = FXAI_GetInputFeature(x, base + 1);
         double ctx_rel = FXAI_GetInputFeature(x, base + 2);
         double ctx_corr = FXAI_GetInputFeature(x, base + 3);
         double w = FXAI_Clamp((0.30 + 0.70 * MathAbs(ctx_corr)) *
                               (0.35 + 0.65 * out[4]) *
                               (0.35 + 0.25 * MathAbs(ctx_ret) + 0.25 * MathAbs(ctx_lag) + 0.15 * MathAbs(ctx_rel)),
                               0.0,
                               3.0);
         if(w <= 1e-6)
            continue;
         ret_mix += w * ctx_ret;
         lag_mix += w * ctx_lag;
         rel_mix += w * ctx_rel;
         corr_mix += w * ctx_corr;
         weight_total += w;
      }
      if(weight_total > 1e-6)
      {
         out[5] = ret_mix / weight_total;
         out[6] = lag_mix / weight_total;
         out[7] = rel_mix / weight_total;
         out[8] = corr_mix / weight_total;
      }
      else
      {
         out[5] = 0.0;
         out[6] = 0.0;
         out[7] = 0.0;
         out[8] = 0.0;
      }
      double domain = FXAI_Clamp(m_ctx_domain_hash, 0.0, 1.0);
      double horizon_scale = FXAI_Clamp(MathLog(1.0 + (double)MathMax(m_ctx_horizon_minutes, 1)) / MathLog(1.0 + 1440.0), 0.0, 1.0);
      double main_mtf_body = 0.0;
      double main_mtf_loc = 0.0;
      double main_mtf_range = 0.0;
      double main_mtf_spread = 0.0;
      for(int tf_slot=0; tf_slot<FXAI_MAIN_MTF_TF_COUNT; tf_slot++)
      {
         int base = FXAI_MainMTFFeatureIndex(tf_slot, 0);
         if(base < 0)
            continue;
         main_mtf_body += FXAI_GetInputFeature(x, base + 0);
         main_mtf_loc += FXAI_GetInputFeature(x, base + 1);
         main_mtf_range += FXAI_GetInputFeature(x, base + 2);
         main_mtf_spread += FXAI_GetInputFeature(x, base + 3);
      }
      main_mtf_body /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
      main_mtf_loc /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
      main_mtf_range /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
      main_mtf_spread /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);

      double ctx_mtf_body = 0.0;
      double ctx_mtf_loc = 0.0;
      double ctx_mtf_range = 0.0;
      double ctx_mtf_spread = 0.0;
      double ctx_mtf_weight = 0.0;
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         double slot_corr = MathAbs(FXAI_GetInputFeature(x, 50 + slot * 4 + 3));
         double slot_weight = 0.35 + 0.65 * slot_corr;
         double slot_body = 0.0;
         double slot_loc = 0.0;
         double slot_range = 0.0;
         double slot_spread = 0.0;
         int slot_used = 0;
         for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
         {
            int base = FXAI_ContextMTFFeatureIndex(slot, tf_slot, 0);
            if(base < 0)
               continue;
            slot_body += FXAI_GetInputFeature(x, base + 0);
            slot_loc += FXAI_GetInputFeature(x, base + 1);
            slot_range += FXAI_GetInputFeature(x, base + 2);
            slot_spread += FXAI_GetInputFeature(x, base + 3);
            slot_used++;
         }
         if(slot_used <= 0)
            continue;
         slot_body /= (double)slot_used;
         slot_loc /= (double)slot_used;
         slot_range /= (double)slot_used;
         slot_spread /= (double)slot_used;
         ctx_mtf_body += slot_weight * slot_body;
         ctx_mtf_loc += slot_weight * slot_loc;
         ctx_mtf_range += slot_weight * slot_range;
         ctx_mtf_spread += slot_weight * slot_spread;
         ctx_mtf_weight += slot_weight;
      }
      if(ctx_mtf_weight > 1e-6)
      {
         ctx_mtf_body /= ctx_mtf_weight;
         ctx_mtf_loc /= ctx_mtf_weight;
         ctx_mtf_range /= ctx_mtf_weight;
         ctx_mtf_spread /= ctx_mtf_weight;
      }
      out[9] = 2.0 * domain - 1.0;
      out[10] = 2.0 * horizon_scale - 1.0;
      out[11] = FXAI_Clamp(0.70 * FXAI_GetInputFeature(x, 72) +
                           0.15 * main_mtf_loc +
                           0.15 * ctx_mtf_loc,
                           -1.0,
                           1.0);
      out[12] = FXAI_Clamp(0.35 * FXAI_GetInputFeature(x, 74) +
                           0.20 * FXAI_GetInputFeature(x, 75) +
                           0.20 * FXAI_GetInputFeature(x, 78) +
                           0.10 * FXAI_GetInputFeature(x, 73) +
                           0.08 * main_mtf_body +
                           0.07 * ctx_mtf_body,
                           -1.0,
                           1.0);
      out[13] = FXAI_Clamp(0.18 * FXAI_GetInputFeature(x, 76) -
                           0.18 * FXAI_GetInputFeature(x, 77) -
                           0.12 * FXAI_GetInputFeature(x, 79) +
                           0.08 * FXAI_GetInputFeature(x, 6) +
                           0.08 * FXAI_GetInputFeature(x, 81) -
                           0.06 * main_mtf_spread -
                           0.06 * FXAI_GetInputFeature(x, 82) +
                           0.04 * ctx_mtf_spread,
                           -4.0,
                           4.0);
      out[14] = FXAI_Clamp(0.45 * FXAI_GetInputFeature(x, 18) +
                           0.20 * FXAI_GetInputFeature(x, 19) -
                           0.20 * FXAI_GetInputFeature(x, 20) +
                           0.15 * FXAI_GetInputFeature(x, 21) +
                           0.10 * main_mtf_body +
                           0.10 * main_mtf_loc,
                           -4.0,
                           4.0);
      out[15] = FXAI_Clamp(0.18 * FXAI_GetInputFeature(x, 66) +
                           0.14 * FXAI_GetInputFeature(x, 67) +
                           0.14 * FXAI_GetInputFeature(x, 68) +
                           0.12 * FXAI_GetInputFeature(x, 69) +
                           0.20 * FXAI_GetInputFeature(x, 71) +
                           0.10 * FXAI_GetInputFeature(x, 81) +
                           0.08 * FXAI_GetInputFeature(x, 82) +
                           0.04 * FXAI_GetInputFeature(x, 83) +
                           0.08 * main_mtf_range +
                           0.06 * ctx_mtf_range,
                           -6.0,
                           6.0);
      out[16] = FXAI_Clamp(0.60 * FXAI_GetInputFeature(x, 68) +
                           0.20 * FXAI_GetInputFeature(x, 81) +
                           0.10 * FXAI_GetInputFeature(x, 80) +
                           0.10 * FXAI_GetInputFeature(x, 82) +
                           0.10 * main_mtf_spread,
                           -4.0,
                           8.0);
      out[17] = FXAI_Clamp(0.65 * FXAI_GetInputFeature(x, 70) +
                           0.20 * FXAI_GetInputFeature(x, 82) +
                           0.10 * main_mtf_range +
                           0.05 * MathAbs(FXAI_GetInputFeature(x, 83)),
                           0.0,
                           8.0);
      out[18] = FXAI_Clamp(FXAI_GetInputFeature(x, 78), -6.0, 6.0);
      out[19] = FXAI_Clamp(FXAI_GetInputFeature(x, 79), 0.0, 6.0);
   }

   bool HasSharedAdapterSignal(const double &a[]) const
   {
      if(ArraySize(a) < FXAI_SHARED_TRANSFER_FEATURES) return false;
      double mag = 0.0;
      for(int i=1; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         mag += MathAbs(a[i]);
      return (mag > 1e-6);
   }

   double SharedAdapterSignalStrength(const double &a[]) const
   {
      if(ArraySize(a) < FXAI_SHARED_TRANSFER_FEATURES)
         return 0.0;
      return FXAI_Clamp(0.18 +
                        0.08 * a[4] +
                        0.05 * MathAbs(a[8]) +
                        0.05 * MathAbs(a[11]) +
                        0.05 * MathAbs(a[12]) +
                        0.05 * MathAbs(a[15]) +
                        0.04 * MathAbs(a[18]) -
                        0.03 * MathAbs(a[19]),
                        0.0,
                        0.52);
   }

   void EncodeSharedTransferBackbone(const double &a[],
                                     double &latent[]) const
   {
      FXAI_SharedTransferEncode(a,
                                FXAI_SharedTransferDomainBucket(m_ctx_domain_hash),
                                FXAI_SharedTransferHorizonBucket(m_ctx_horizon_minutes),
                                m_ctx_session_bucket,
                                m_shared_backbone_w,
                                m_shared_backbone_b,
                                m_shared_domain_emb,
                                m_shared_horizon_emb,
                                m_shared_session_emb,
                                latent);
   }

   void PredictSharedTransferBackbone(const double &a[],
                                      double &probs[],
                                      double &move_adj) const
   {
      double latent[];
      EncodeSharedTransferBackbone(a, latent);

      double logits[3];
      for(int c=0; c<3; c++)
      {
         logits[c] = 0.0;
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            logits[c] += m_shared_backbone_cls[c][j] * latent[j];
      }
      FXAI_SharedTransferSoftmax(logits, probs);

      move_adj = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         move_adj += m_shared_backbone_move[j] * latent[j];
   }

   double BuildTransferSlotSignal(const double &x[],
                                  const int slot) const
   {
      if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
         return 0.0;
      int base = 50 + slot * 4;
      double ctx_ret = FXAI_GetInputFeature(x, base + 0);
      double ctx_lag = FXAI_GetInputFeature(x, base + 1);
      double ctx_rel = FXAI_GetInputFeature(x, base + 2);
      double signal = 0.30 * ctx_ret + 0.50 * ctx_lag + 0.20 * ctx_rel;
      return FXAI_ClipSym(signal, 4.0);
   }

   void BlendTransferSlotPriors(const double &x[],
                                double &probs[],
                                double &move_scale_mult,
                                double &reliability_boost) const
   {
      double coverage = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);
      double domain_buy = 0.0;
      double domain_sell = 0.0;
      double domain_skip = 0.0;
      double domain_move = 0.0;
      double domain_rel = 0.0;
      double domain_weight = 0.0;
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         if(m_transfer_slot_obs[slot] <= 1e-6)
            continue;

         int base = 50 + slot * 4;
         double ctx_corr = FXAI_GetInputFeature(x, base + 3);
         double signal = BuildTransferSlotSignal(x, slot);
         double obs_trust = FXAI_Clamp(m_transfer_slot_obs[slot] / 24.0, 0.0, 1.0);
         double w = FXAI_Clamp(obs_trust *
                               (0.25 + 0.75 * MathAbs(ctx_corr)) *
                               (0.20 + 0.80 * coverage) *
                               (0.20 + 0.80 * MathAbs(signal)),
                               0.0,
                               2.0);
         if(w <= 1e-6)
            continue;

         double align = FXAI_Clamp(m_transfer_slot_align[slot], -1.0, 1.0);
         double lead = FXAI_Clamp(m_transfer_slot_lead[slot], 0.0, 1.0);
         double move_scale = FXAI_Clamp(m_transfer_slot_move[slot], 0.50, 2.50);
         double buy_prior = 0.10;
         double sell_prior = 0.10;
         double skip_prior = FXAI_Clamp(0.55 - 0.10 * MathAbs(signal), 0.05, 0.80);
         if(signal > 0.0)
         {
            buy_prior = FXAI_Clamp(0.45 + 0.22 * align + 0.15 * lead + 0.08 * MathAbs(signal), 0.05, 0.95);
            sell_prior = FXAI_Clamp(0.20 - 0.12 * align, 0.02, 0.60);
         }
         else if(signal < 0.0)
         {
            sell_prior = FXAI_Clamp(0.45 - 0.22 * align + 0.15 * lead + 0.08 * MathAbs(signal), 0.05, 0.95);
            buy_prior = FXAI_Clamp(0.20 + 0.12 * align, 0.02, 0.60);
         }

         double ps = buy_prior + sell_prior + skip_prior;
         if(ps <= 0.0) ps = 1.0;
         buy_prior /= ps;
         sell_prior /= ps;
         skip_prior /= ps;

         domain_buy += w * buy_prior;
         domain_sell += w * sell_prior;
         domain_skip += w * skip_prior;
         domain_move += w * move_scale;
         domain_rel += w * (0.50 + 0.25 * MathAbs(align) + 0.25 * lead);
         domain_weight += w;
      }

      if(domain_weight <= 1e-6)
      {
         move_scale_mult = 1.0;
         reliability_boost = 0.0;
         return;
      }

      probs[0] = domain_sell / domain_weight;
      probs[1] = domain_buy / domain_weight;
      probs[2] = domain_skip / domain_weight;
      move_scale_mult = FXAI_Clamp(domain_move / domain_weight, 0.70, 1.50);
      reliability_boost = FXAI_Clamp((domain_rel / domain_weight) - 0.50, -0.15, 0.20);
   }

   void ApplySharedContextAdapter(FXAIAIModelOutputV4 &out,
                                  const double &x[]) const
   {
      double a[];
      BuildSharedAdapterInput(x, a);
      if(!HasSharedAdapterSignal(a))
         return;

      double shallow_trust = FXAI_Clamp((double)m_shared_adapter_steps / 96.0, 0.0, 1.0);
      shallow_trust *= SharedAdapterSignalStrength(a);
      double backbone_trust = FXAI_Clamp((double)m_shared_backbone_steps / 144.0, 0.0, 1.0);
      backbone_trust *= FXAI_Clamp(0.10 + 1.15 * SharedAdapterSignalStrength(a), 0.0, 0.55);
      double trust = MathMax(shallow_trust, backbone_trust);
      if(trust <= 1e-6)
         return;

      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            z += m_shared_cls_w[c][i] * a[i];
         logits[c] = z;
      }
      double probs[3];
      FXAI_SharedTransferSoftmax(logits, probs);

      double bb_probs[];
      double bb_move_adj = 0.0;
      PredictSharedTransferBackbone(a, bb_probs, bb_move_adj);
      if(ArraySize(bb_probs) == 3 && backbone_trust > 1e-6)
      {
         double blend = FXAI_Clamp(0.35 + 0.65 * backbone_trust, 0.0, 0.85);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - blend) * probs[c] + blend * bb_probs[c], 0.0005, 0.9990);
      }

      double transfer_probs[3];
      transfer_probs[0] = 0.0;
      transfer_probs[1] = 0.0;
      transfer_probs[2] = 0.0;
      double transfer_move_mult = 1.0;
      double transfer_rel_boost = 0.0;
      BlendTransferSlotPriors(x, transfer_probs, transfer_move_mult, transfer_rel_boost);
      double transfer_mass = transfer_probs[0] + transfer_probs[1] + transfer_probs[2];
      if(transfer_mass > 0.0)
      {
         for(int c=0; c<3; c++)
            transfer_probs[c] /= transfer_mass;
         double transfer_trust = FXAI_Clamp(0.10 + 0.35 * a[4], 0.0, 0.28);
         for(int c=0; c<3; c++)
            probs[c] = FXAI_Clamp((1.0 - transfer_trust) * probs[c] + transfer_trust * transfer_probs[c], 0.0005, 0.9990);
      }

      for(int c=0; c<3; c++)
         out.class_probs[c] = FXAI_Clamp((1.0 - trust) * out.class_probs[c] + trust * probs[c], 0.0005, 0.9990);
      NormalizeClassDistribution(out.class_probs);

      double move_adj = 0.0;
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         move_adj += m_shared_move_w[i] * a[i];
      move_adj = 0.70 * move_adj + 0.30 * bb_move_adj;
      double scale = FXAI_Clamp((1.0 + 0.16 * trust * FXAI_ClipSym(move_adj, 1.5)) * transfer_move_mult, 0.75, 1.45);
      out.move_mean_points = MathMax(0.0, out.move_mean_points * scale);
      out.move_q25_points = MathMax(0.0, out.move_q25_points * scale);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_q50_points * scale);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_q75_points * scale);
      out.confidence = FXAI_Clamp(MathMax(out.class_probs[(int)FXAI_LABEL_BUY],
                                          out.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
      out.reliability = FXAI_Clamp(out.reliability * (1.0 - 0.12 * trust) +
                                   trust * FXAI_Clamp(0.48 +
                                                      0.14 * a[2] +
                                                      0.10 * a[4] +
                                                      0.05 * MathAbs(a[11]) +
                                                      0.05 * MathAbs(a[15]) +
                                                      0.05 * (1.0 - MathAbs(a[19]) / 6.0) +
                                                      transfer_rel_boost,
                                                      0.0,
                                                      1.0),
                                   0.0,
                                   1.0);
   }

   void UpdateCrossSymbolTransferBank(const double &x[],
                                      const double move_points,
                                      const double sample_w)
   {
      double move_sign = FXAI_Sign(move_points);
      if(MathAbs(move_sign) <= 1e-9)
         return;

      double coverage = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);
      double move_scale = MathMax(MathAbs(move_points), MathMax(ResolveMinMovePoints(), 0.10));
      for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
      {
         int base = 50 + slot * 4;
         double ctx_corr = FXAI_GetInputFeature(x, base + 3);
         double signal = BuildTransferSlotSignal(x, slot);
         if(MathAbs(signal) <= 1e-6)
            continue;

         double trust = FXAI_Clamp((0.30 + 0.70 * MathAbs(ctx_corr)) *
                                   (0.20 + 0.80 * coverage) *
                                   FXAI_Clamp(sample_w, 0.25, 4.0),
                                   0.02,
                                   2.50);
         double alpha = FXAI_Clamp(0.05 * trust / MathSqrt(1.0 + 0.02 * m_transfer_slot_obs[slot]), 0.01, 0.20);
         double align_target = FXAI_Clamp(FXAI_Sign(signal) * move_sign, -1.0, 1.0);
         double lead_target = FXAI_Clamp(0.5 + 0.5 * FXAI_Sign(FXAI_GetInputFeature(x, base + 1)) * move_sign, 0.0, 1.0);
         double move_target = FXAI_Clamp(move_scale / MathMax(MathAbs(signal), 0.10), 0.50, 2.50);

         if(m_transfer_slot_obs[slot] <= 1e-6)
         {
            m_transfer_slot_align[slot] = align_target;
            m_transfer_slot_lead[slot] = lead_target;
            m_transfer_slot_move[slot] = move_target;
         }
         else
         {
            m_transfer_slot_align[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_align[slot] + alpha * align_target, -1.0, 1.0);
            m_transfer_slot_lead[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_lead[slot] + alpha * lead_target, 0.0, 1.0);
            m_transfer_slot_move[slot] = FXAI_Clamp((1.0 - alpha) * m_transfer_slot_move[slot] + alpha * move_target, 0.50, 2.50);
         }
         m_transfer_slot_obs[slot] = MathMin(m_transfer_slot_obs[slot] + trust, 5000.0);
      }
   }

   void UpdateSharedContextAdapter(const double &x[],
                                   const int y,
                                   const double move_points,
                                   const double sample_w,
                                   const double lr)
   {
      double a[];
      BuildSharedAdapterInput(x, a);
      if(!HasSharedAdapterSignal(a))
         return;

      int cls = NormalizeClassLabel(y, x, move_points);
      double logits[3];
      for(int c=0; c<3; c++)
      {
         double z = 0.0;
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            z += m_shared_cls_w[c][i] * a[i];
         logits[c] = z;
      }

      double probs[3];
      FXAI_SharedTransferSoftmax(logits, probs);

      double step = FXAI_Clamp(0.18 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0002, 0.0200);
      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double err = target - probs[c];
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_cls_w[c][i] = FXAI_ClipSym(m_shared_cls_w[c][i] + step * err * a[i], 3.0);
      }

      double move_pred = 0.0;
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         move_pred += m_shared_move_w[i] * a[i];
      double move_target = FXAI_Clamp(MathLog(1.0 + MathAbs(move_points)), 0.0, 4.0);
      double move_err = FXAI_ClipSym(move_target - move_pred, 3.0);
      for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
         m_shared_move_w[i] = FXAI_ClipSym(m_shared_move_w[i] + 0.80 * step * move_err * a[i], 3.0);

      double latent[];
      EncodeSharedTransferBackbone(a, latent);
      double bb_logits[3];
      for(int c=0; c<3; c++)
      {
         bb_logits[c] = 0.0;
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
            bb_logits[c] += m_shared_backbone_cls[c][j] * latent[j];
      }
      double bb_probs[];
      FXAI_SharedTransferSoftmax(bb_logits, bb_probs);

      double latent_grad[FXAI_SHARED_TRANSFER_LATENT];
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         latent_grad[j] = 0.0;

      double bb_step = FXAI_Clamp(0.12 * lr * FXAI_Clamp(sample_w, 0.25, 4.0), 0.0001, 0.0120);
      for(int c=0; c<3; c++)
      {
         double target = (c == cls ? 1.0 : 0.0);
         double err = target - (ArraySize(bb_probs) == 3 ? bb_probs[c] : 0.3333333);
         for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         {
            latent_grad[j] += err * m_shared_backbone_cls[c][j];
            m_shared_backbone_cls[c][j] = FXAI_ClipSym(m_shared_backbone_cls[c][j] + bb_step * err * latent[j], 3.0);
         }
      }

      double bb_move_pred = 0.0;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
         bb_move_pred += m_shared_backbone_move[j] * latent[j];
      double bb_move_err = FXAI_ClipSym(move_target - bb_move_pred, 3.0);
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         latent_grad[j] += 0.30 * bb_move_err * m_shared_backbone_move[j];
         m_shared_backbone_move[j] = FXAI_ClipSym(m_shared_backbone_move[j] + 0.65 * bb_step * bb_move_err * latent[j], 3.0);
      }

      int domain_bucket = FXAI_SharedTransferDomainBucket(m_ctx_domain_hash);
      int horizon_bucket = FXAI_SharedTransferHorizonBucket(m_ctx_horizon_minutes);
      int session_bucket = m_ctx_session_bucket;
      if(session_bucket < 0) session_bucket = 0;
      if(session_bucket >= FXAI_PLUGIN_SESSION_BUCKETS) session_bucket = FXAI_PLUGIN_SESSION_BUCKETS - 1;
      for(int j=0; j<FXAI_SHARED_TRANSFER_LATENT; j++)
      {
         double g = FXAI_ClipSym(latent_grad[j] * (1.0 - latent[j] * latent[j]), 2.5);
         m_shared_backbone_b[j] = FXAI_ClipSym(m_shared_backbone_b[j] + bb_step * g, 3.0);
         for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
            m_shared_backbone_w[j][i] = FXAI_ClipSym(m_shared_backbone_w[j][i] + bb_step * g * a[i], 3.0);
         m_shared_domain_emb[domain_bucket][j] = FXAI_ClipSym(m_shared_domain_emb[domain_bucket][j] + 0.40 * bb_step * g, 3.0);
         m_shared_horizon_emb[horizon_bucket][j] = FXAI_ClipSym(m_shared_horizon_emb[horizon_bucket][j] + 0.40 * bb_step * g, 3.0);
         m_shared_session_emb[session_bucket][j] = FXAI_ClipSym(m_shared_session_emb[session_bucket][j] + 0.30 * bb_step * g, 3.0);
      }

      m_shared_adapter_steps++;
      if(m_shared_adapter_steps >= 24)
         m_shared_adapter_ready = true;
      m_shared_backbone_steps++;
      if(m_shared_backbone_steps >= 36)
         m_shared_backbone_ready = true;
   }

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

   void SetTrainingTargets(const FXAIAITrainRequestV4 &req)
   {
      m_target_quality_ready = true;
      m_target_mfe_points = MathMax(req.mfe_points, 0.0);
      m_target_mae_points = MathMax(req.mae_points, 0.0);
      m_target_hit_time_frac = FXAI_Clamp(req.time_to_hit_frac, 0.0, 1.0);
      m_target_path_flags = req.path_flags;
      m_target_path_risk = FXAI_Clamp(req.path_risk, 0.0, 1.0);
      m_target_fill_risk = FXAI_Clamp(req.fill_risk, 0.0, 1.0);
      m_target_masked_step = req.masked_step_target;
      m_target_next_vol = MathMax(req.next_vol_target, 0.0);
      m_target_regime_shift = FXAI_Clamp(req.regime_shift_target, 0.0, 1.0);
      m_target_context_lead = FXAI_Clamp(req.context_lead_target, 0.0, 1.0);
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

   ulong DefaultFeatureGroupsMask(void) const
   {
      ulong mask = 0;
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_PRICE);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_VOLATILITY);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_TIME);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_CONTEXT);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_COST);
      mask |= ((ulong)1 << (int)FXAI_FEAT_GROUP_FILTERS);
      return mask;
   }

   void FillManifest(FXAIAIManifestV4 &out,
                     const int family,
                     const ulong capability_mask,
                     const int min_sequence_bars,
                     const int max_sequence_bars,
                     const int min_horizon_minutes = 1,
                     const int max_horizon_minutes = 720,
                     const ulong feature_groups_mask = 0,
                     const int feature_schema_id = 0) const
   {
      int min_seq = min_sequence_bars;
      int max_seq = max_sequence_bars;
      if(min_seq < 1) min_seq = 1;
      if(min_seq > FXAI_MAX_SEQUENCE_BARS) min_seq = FXAI_MAX_SEQUENCE_BARS;
      if(max_seq < min_seq) max_seq = min_seq;
      if(max_seq > FXAI_MAX_SEQUENCE_BARS) max_seq = FXAI_MAX_SEQUENCE_BARS;

      out.api_version = FXAI_API_VERSION_V4;
      out.ai_id = AIId();
      out.ai_name = AIName();
      out.family = family;
      out.reference_tier = FXAI_DefaultReferenceTierForAI(out.ai_id);
      out.capability_mask = capability_mask;
      out.feature_schema_id = (feature_schema_id > 0 ? feature_schema_id : FXAI_DefaultFeatureSchemaForFamily(family));
      out.feature_groups_mask = (feature_groups_mask != 0 ? feature_groups_mask : FXAI_DefaultFeatureGroupsForFamily(family));
      out.min_horizon_minutes = min_horizon_minutes;
      out.max_horizon_minutes = max_horizon_minutes;
      out.min_sequence_bars = min_seq;
      out.max_sequence_bars = max_seq;
   }

   void ResetModelOutput(FXAIAIModelOutputV4 &out) const
   {
      out.class_probs[0] = 0.10;
      out.class_probs[1] = 0.10;
      out.class_probs[2] = 0.80;
      out.move_mean_points = 0.0;
      out.move_q25_points = 0.0;
      out.move_q50_points = 0.0;
      out.move_q75_points = 0.0;
      out.mfe_mean_points = 0.0;
      out.mae_mean_points = 0.0;
      out.hit_time_frac = 1.0;
      out.path_risk = 0.0;
      out.fill_risk = 0.0;
      out.confidence = 0.0;
      out.reliability = 0.0;
      out.has_quantiles = false;
      out.has_confidence = false;
      out.has_path_quality = false;
   }

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      double move_mean_points = out.move_mean_points;
      if(!PredictModelCore(x, hp, out.class_probs, move_mean_points))
         return false;
      NormalizeClassDistribution(out.class_probs);
      out.move_mean_points = (MathIsValidNumber(move_mean_points) && move_mean_points > 0.0 ? move_mean_points : 0.0);

      double buy_p = out.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = out.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = out.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double entropy = 0.0;
      for(int c=0; c<3; c++)
      {
         double p = MathMax(out.class_probs[c], 1e-9);
         entropy -= p * MathLog(p);
      }
      entropy /= MathLog(3.0);

      double move_scale = MathMax(ResolveMinMovePoints(), 0.10);
      if(m_move_ready && m_move_ema_abs > 0.0)
         move_scale = MathMax(move_scale, 0.60 * m_move_ema_abs);
      if(CurrentWindowSize() > 1)
      {
         double mean1 = CurrentWindowFeatureMean(0);
         double var1 = 0.0;
         for(int b=0; b<CurrentWindowSize(); b++)
         {
            double d = CurrentWindowValue(b, 1) - mean1;
            var1 += d * d;
         }
         var1 = MathSqrt(var1 / (double)CurrentWindowSize());
         move_scale = MathMax(move_scale, 0.35 * var1);
      }

      double sigma = MathMax(0.10, 0.35 * out.move_mean_points + 0.35 * move_scale + 0.20 * skip_p + 0.15 * entropy);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.55 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.55 * sigma);
      double mfe_scale = 1.20 + 0.35 * directional_conf + 0.20 * (1.0 - skip_p);
      double mae_scale = 0.35 + 0.25 * skip_p + 0.20 * entropy;
      if(m_quality_head_ready)
      {
         double quality_base = MathMax(m_move_ema_abs, MathMax(move_scale, 0.10));
         mfe_scale = FXAI_Clamp(m_quality_mfe_ema / MathMax(quality_base, 0.10), 0.80, 3.00);
         mae_scale = FXAI_Clamp(m_quality_mae_ema / MathMax(MathMax(m_quality_mfe_ema, quality_base), 0.10), 0.05, 1.50);
      }
      out.mfe_mean_points = MathMax(out.move_q75_points, out.move_mean_points * mfe_scale);
      out.mae_mean_points = MathMax(0.0, out.move_mean_points * mae_scale);
      double hit_frac = 0.60 - 0.25 * directional_conf + 0.20 * skip_p + 0.15 * entropy;
      if(m_quality_head_ready)
         hit_frac = 0.55 * m_quality_hit_ema + 0.45 * hit_frac;
      out.hit_time_frac = FXAI_Clamp(hit_frac, 0.0, 1.0);
      out.confidence = FXAI_Clamp(0.60 * directional_conf + 0.20 * (1.0 - skip_p) + 0.20 * (1.0 - entropy), 0.0, 1.0);
      int r = m_ctx_regime_id;
      if(r < 0) r = 0;
      if(r >= FXAI_PLUGIN_REGIME_BUCKETS) r = FXAI_PLUGIN_REGIME_BUCKETS - 1;
      int s = ContextSessionBucket();
      int h = ContextHorizonBucket();
      double bank_mass = m_bank_total[r][s][h];
      double bank_rel = FXAI_Clamp(bank_mass / 120.0, 0.0, 1.0);
      out.reliability = FXAI_Clamp(0.45 + 0.25 * bank_rel + 0.20 * (1.0 - entropy) + 0.10 * (m_move_ready ? 1.0 : 0.0), 0.0, 1.0);
      double path_risk = 0.40 * FXAI_Clamp(out.mae_mean_points / MathMax(out.mfe_mean_points, move_scale), 0.0, 1.0) +
                         0.35 * out.hit_time_frac +
                         0.25 * skip_p;
      if(m_quality_head_ready)
         path_risk = 0.55 * path_risk + 0.45 * m_quality_path_risk_ema;
      out.path_risk = FXAI_Clamp(path_risk, 0.0, 1.0);
      double fill_risk = FXAI_Clamp((ResolveCostPoints(x) + 0.50 * ResolveMinMovePoints()) / MathMax(out.move_mean_points + move_scale, 0.25), 0.0, 1.0);
      if(m_quality_head_ready)
         fill_risk = 0.50 * fill_risk + 0.50 * m_quality_fill_risk_ema;
      out.fill_risk = FXAI_Clamp(fill_risk, 0.0, 1.0);
      out.has_quantiles = true;
      out.has_confidence = true;
      out.has_path_quality = true;
      return true;
   }

   void FillPredictionV4(const FXAIAIModelOutputV4 &model_out,
                         const double calibrated_move_mean_points,
                         FXAIAIPredictionV4 &dst) const
   {
      for(int c=0; c<3; c++)
         dst.class_probs[c] = model_out.class_probs[c];

      double buy_p = dst.class_probs[(int)FXAI_LABEL_BUY];
      double sell_p = dst.class_probs[(int)FXAI_LABEL_SELL];
      double skip_p = dst.class_probs[(int)FXAI_LABEL_SKIP];
      double directional_conf = MathMax(buy_p, sell_p);
      double uncertainty = FXAI_Clamp(1.0 - directional_conf + 0.50 * skip_p, 0.10, 1.50);
      double mean_move = (MathIsValidNumber(calibrated_move_mean_points) && calibrated_move_mean_points > 0.0 ? calibrated_move_mean_points : 0.0);

      dst.move_mean_points = mean_move;
      double raw_mean = (MathIsValidNumber(model_out.move_mean_points) && model_out.move_mean_points > 0.0 ? model_out.move_mean_points : 0.0);
      double scale = (raw_mean > 1e-9 ? mean_move / raw_mean : 1.0);

      if(model_out.has_quantiles && mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, model_out.move_q25_points * scale);
         dst.move_q50_points = MathMax(dst.move_q25_points, model_out.move_q50_points * scale);
         dst.move_q75_points = MathMax(dst.move_q50_points, model_out.move_q75_points * scale);
      }
      else if(mean_move > 0.0)
      {
         dst.move_q25_points = MathMax(0.0, mean_move * MathMax(0.25, 1.0 - 0.45 * uncertainty));
         dst.move_q50_points = mean_move;
         dst.move_q75_points = MathMax(dst.move_q50_points, mean_move * (1.0 + 0.45 * uncertainty));
      }
      else
      {
         dst.move_q25_points = 0.0;
         dst.move_q50_points = 0.0;
         dst.move_q75_points = 0.0;
      }

      if(model_out.has_path_quality)
      {
         dst.mfe_mean_points = MathMax(0.0, model_out.mfe_mean_points * scale);
         dst.mae_mean_points = MathMax(0.0, model_out.mae_mean_points * scale);
         dst.hit_time_frac = FXAI_Clamp(model_out.hit_time_frac, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(model_out.path_risk, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp(model_out.fill_risk, 0.0, 1.0);
      }
      else
      {
         dst.mfe_mean_points = MathMax(dst.move_q75_points, dst.move_mean_points);
         dst.mae_mean_points = MathMax(0.0, 0.35 * dst.move_mean_points);
         dst.hit_time_frac = FXAI_Clamp(0.60 - 0.20 * directional_conf + 0.20 * skip_p, 0.0, 1.0);
         dst.path_risk = FXAI_Clamp(0.40 * skip_p + 0.35 * dst.hit_time_frac, 0.0, 1.0);
         dst.fill_risk = FXAI_Clamp((m_ctx_cost_points + 0.25 * ResolveMinMovePoints()) / MathMax(dst.move_mean_points + ResolveMinMovePoints(), 0.25), 0.0, 1.0);
      }

      dst.confidence = FXAI_Clamp(model_out.has_confidence ? model_out.confidence : directional_conf, 0.0, 1.0);
      dst.reliability = FXAI_Clamp(model_out.has_confidence ? model_out.reliability : (1.0 - 0.50 * skip_p), 0.0, 1.0);
   }

   double CurrentWindowSliceMean(const int input_idx,
                                 const int start_bar,
                                 const int count) const
   {
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0 || count <= 0)
         return 0.0;

      int first = start_bar;
      if(first < 0) first = 0;
      if(first >= m_ctx_window_size) return 0.0;
      int last = first + count;
      if(last > m_ctx_window_size) last = m_ctx_window_size;
      if(last <= first) return 0.0;

      double sum = 0.0;
      int n = 0;
      for(int b=first; b<last; b++)
      {
         sum += m_ctx_window[b][input_idx];
         n++;
      }
      if(n <= 0) return 0.0;
      return sum / (double)n;
   }

   int CurrentWindowSize(void) const
   {
      return m_ctx_window_size;
   }

   double CurrentWindowValue(const int bar_idx, const int input_idx) const
   {
      if(bar_idx < 0 || bar_idx >= m_ctx_window_size) return 0.0;
      if(input_idx < 0 || input_idx >= FXAI_AI_WEIGHTS) return 0.0;
      return m_ctx_window[bar_idx][input_idx];
   }

   double CurrentWindowFeatureMean(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0) return 0.0;
      double full = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      int half_n = MathMax(m_ctx_window_size / 2, 1);
      int quarter_n = MathMax(m_ctx_window_size / 4, 1);
      double half = CurrentWindowSliceMean(input_idx, m_ctx_window_size - half_n, half_n);
      double quarter = CurrentWindowSliceMean(input_idx, m_ctx_window_size - quarter_n, quarter_n);
      return 0.40 * full + 0.35 * half + 0.25 * quarter;
   }

   double CurrentWindowFeatureRecentMean(const int feature_idx,
                                         const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0) n = 1;
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      return CurrentWindowSliceMean(input_idx, 0, n);
   }

   double CurrentWindowFeatureStd(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double mean = CurrentWindowSliceMean(input_idx, 0, m_ctx_window_size);
      double acc = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double d = m_ctx_window[b][input_idx] - mean;
         acc += d * d;
      }
      return MathSqrt(acc / (double)MathMax(m_ctx_window_size, 1));
   }

   double CurrentWindowFeatureRange(const int feature_idx,
                                    const int recent_bars = 0) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 0 || n > m_ctx_window_size) n = m_ctx_window_size;
      double lo = CurrentWindowValue(0, input_idx);
      double hi = lo;
      for(int b=0; b<n; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         if(v < lo) lo = v;
         if(v > hi) hi = v;
      }
      return hi - lo;
   }

   double CurrentWindowFeatureSlope(const int feature_idx) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 1) return 0.0;
      double first = m_ctx_window[0][input_idx];
      double last = m_ctx_window[m_ctx_window_size - 1][input_idx];
      return (first - last) / (double)MathMax(m_ctx_window_size - 1, 1);
   }

   double CurrentWindowFeatureRecentDelta(const int feature_idx,
                                          const int recent_bars) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      int n = recent_bars;
      if(n <= 1) n = MathMax(m_ctx_window_size / 4, 2);
      if(n > m_ctx_window_size) n = m_ctx_window_size;
      int last_idx = n - 1;
      if(last_idx < 0) last_idx = 0;
      return CurrentWindowValue(0, input_idx) - CurrentWindowValue(last_idx, input_idx);
   }

   double CurrentWindowFeatureEMAMean(const int feature_idx,
                                      const double decay = 0.72) const
   {
      int input_idx = feature_idx + 1;
      if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS || m_ctx_window_size <= 0)
         return 0.0;
      double a = FXAI_Clamp(decay, 0.05, 0.98);
      double w = 1.0;
      double sw = 0.0;
      double sum = 0.0;
      for(int b=0; b<m_ctx_window_size; b++)
      {
         double v = CurrentWindowValue(b, input_idx);
         sum += w * v;
         sw += w;
         w *= a;
      }
      if(sw <= 0.0) return 0.0;
      return sum / sw;
   }

   void BuildChronologicalSequenceTensor(const double &x[],
                                         double &seq[][FXAI_AI_WEIGHTS],
                                         int &seq_len) const
   {
      FXAISequenceRuntimeConfig cfg = FXAI_SequenceRuntimeMakeConfig(FXAI_MAX_SEQUENCE_BARS, 1, 1, false, true, 0.06);
      BuildChronologicalSequenceTensorConfigured(x, cfg, seq, seq_len);
   }

   int ContextSequenceCap(const int max_cap,
                          const int base_min = 8) const
   {
      return FXAI_ContextSequenceSpan(max_cap,
                                      (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                      _Symbol,
                                      base_min);
   }

   int ContextBatchCap(const int max_cap,
                       const int base_min = 4) const
   {
      return FXAI_ContextBatchSpan(max_cap,
                                   (m_ctx_horizon_minutes > 0 ? m_ctx_horizon_minutes : 1),
                                   _Symbol,
                                   base_min);
   }
