   double current_raw_x[FXAI_AI_WEIGHTS];
   current_raw_x[0] = 1.0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      current_raw_x[f + 1] = feat_pred[f];
   double current_shared_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int current_shared_window_size = 0;
   FXAI_ClearInputWindow(current_shared_window, current_shared_window_size);
   int current_shared_span = FXAI_ContextSequenceSpan(24, H, snapshot.symbol, 8);
   if(current_shared_span < 1) current_shared_span = 1;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      current_shared_window[0][k] = current_raw_x[k];
   current_shared_window_size = 1;
   for(int idx=1; idx<current_shared_span && idx<ArraySize(samples) && current_shared_window_size < FXAI_MAX_SEQUENCE_BARS; idx++)
   {
      if(!samples[idx].valid)
         continue;
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         current_shared_window[current_shared_window_size][k] = samples[idx].x[k];
      current_shared_window_size++;
   }
   if(have_online_window && online_start >= 0 && online_start < ArraySize(samples) && samples[online_start].valid)
   {
      double online_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int online_window_size = 0;
      FXAI_BuildPreparedSampleWindow(samples, online_start, current_shared_span, online_window, online_window_size);
      double online_a[];
      FXAI_BuildSharedTransferInputGlobal(samples[online_start].x,
                                          online_window,
                                          online_window_size,
                                          samples[online_start].domain_hash,
                                          samples[online_start].horizon_minutes,
                                          online_a);
   FXAI_GlobalFoundationUpdate(online_a,
                               online_window,
                               online_window_size,
                               samples[online_start].domain_hash,
                               samples[online_start].horizon_minutes,
                               FXAI_DeriveSessionBucket(samples[online_start].sample_time),
                               samples[online_start].masked_step_target,
                               samples[online_start].next_vol_target,
                               samples[online_start].regime_shift_target,
                               samples[online_start].context_lead_target,
                               samples[online_start].sample_weight,
                               0.012 * FXAI_Clamp(0.55 + deploy_profile.foundation_weight,
                                                    0.35,
                                                    1.45) *
                               FXAI_Clamp(deploy_profile.foundation_quality_gain, 0.40, 1.80));
   }
   int runtime_session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
   ulong transfer_stage_t0 = GetMicrosecondCount();
   double current_transfer_a[];
   FXAI_BuildSharedTransferInputGlobal(current_raw_x,
                                       current_shared_window,
                                       current_shared_window_size,
                                       FXAI_SymbolHash01(snapshot.symbol),
                                       H,
                                       current_transfer_a);
   FXAIFoundationSignals current_foundation_sig;
   FXAI_GlobalFoundationPredict(current_transfer_a,
                                current_shared_window,
                                current_shared_window_size,
                                FXAI_SymbolHash01(snapshot.symbol),
                                H,
                                runtime_session_bucket,
                                current_foundation_sig);
   FXAIStudentSignals current_student_sig;
   FXAI_GlobalStudentPredict(current_transfer_a,
                             current_shared_window,
                             current_shared_window_size,
                             FXAI_SymbolHash01(snapshot.symbol),
                             H,
                             runtime_session_bucket,
                             current_student_sig);
   current_foundation_sig.trust = FXAI_Clamp(current_foundation_sig.trust *
                                             FXAI_Clamp(deploy_profile.foundation_quality_gain, 0.40, 1.80),
                                             0.0,
                                             1.0);
   current_foundation_sig.tradability = FXAI_Clamp(current_foundation_sig.tradability *
                                                   FXAI_Clamp(0.85 + 0.15 * deploy_profile.foundation_quality_gain, 0.40, 1.80),
                                                   0.0,
                                                   1.0);
   current_foundation_sig.move_ratio = FXAI_Clamp(current_foundation_sig.move_ratio *
                                                  FXAI_Clamp(deploy_profile.teacher_signal_gain, 0.40, 1.80),
                                                  0.0,
                                                  4.0);
   current_student_sig.trust = FXAI_Clamp(current_student_sig.trust *
                                          FXAI_Clamp(deploy_profile.student_signal_gain, 0.40, 1.80),
                                          0.0,
                                          1.0);
   current_student_sig.tradability = FXAI_Clamp(current_student_sig.tradability *
                                                FXAI_Clamp(0.85 + 0.15 * deploy_profile.policy_lifecycle_gain, 0.40, 1.80),
                                                0.0,
                                                1.0);
   FXAIAnalogMemoryQuery current_analog_q;
   FXAI_QueryAnalogMemory(current_raw_x,
                          regime_id,
                          runtime_session_bucket,
                          H,
                          FXAI_SymbolHash01(snapshot.symbol),
                          current_analog_q);
   g_ai_last_macro_state_quality = FXAI_Clamp(FXAI_GetInputFeature(current_raw_x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19) *
                                              FXAI_Clamp(deploy_profile.macro_state_gain, 0.40, 1.80),
                                              0.0,
                                              1.0);
   FXAIRegimeGraphQuery current_regime_q;
   FXAI_QueryRegimeGraph(regime_id,
                         g_ai_last_macro_state_quality,
                         current_regime_q);
   FXAIAdaptiveRegimeState adaptive_regime_state;
   FXAI_BuildAdaptiveRegimeState(symbol,
                                 snapshot,
                                 spread_pred,
                                 vol_proxy_abs,
                                 high_arr,
                                 low_arr,
                                 close_arr,
                                 min_move_pred,
                                 context_strength,
                                 context_quality,
                                 current_regime_q,
                                 adaptive_news_state,
                                 adaptive_cross_asset_state,
                                 adaptive_micro_state,
                                 adaptive_regime_state);
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_TRANSFER,
                             (double)(GetMicrosecondCount() - transfer_stage_t0) / 1000.0);
   double macro_profile_shortfall = (FXAI_MacroEventLeakageSafe()
                                     ? FXAI_Clamp(deploy_profile.macro_quality_floor -
                                                  g_ai_last_macro_state_quality,
                                                  0.0,
                                                  1.0)
                                     : 0.0);
   double regime_transition_penalty = FXAI_Clamp(deploy_profile.regime_transition_weight, 0.0, 1.0) *
                                      FXAI_Clamp(current_regime_q.instability, 0.0, 1.0);
   FXAIControlPlaneAggregate cp_buy;
   FXAIControlPlaneAggregate cp_sell;
   FXAI_ReadControlPlaneAggregate(symbol, 1, cp_buy);
   FXAI_ReadControlPlaneAggregate(symbol, 0, cp_sell);
   g_control_plane_last_buy_score = cp_buy.score;
   g_control_plane_last_sell_score = cp_sell.score;
