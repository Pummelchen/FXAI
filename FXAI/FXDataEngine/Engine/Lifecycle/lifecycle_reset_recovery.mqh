void ResetAIState(const string symbol)
{
   g_ai_last_symbol = symbol;
   g_ai_last_signal_bar = 0;
   g_ai_last_signal = -1;
   g_ai_last_signal_key = -1;
   g_ai_last_reason = "reset";
   g_ai_last_expected_move_points = 0.0;
   g_ai_last_trade_edge_points = 0.0;
   g_ai_last_confidence = 0.0;
   g_ai_last_reliability = 0.0;
   g_ai_last_path_risk = 1.0;
   g_ai_last_fill_risk = 1.0;
   g_ai_last_trade_gate = 0.0;
   g_ai_last_context_quality = 0.0;
   g_ai_last_context_strength = 0.0;
   g_ai_last_min_move_points = 0.0;
   g_ai_last_horizon_minutes = 0;
   g_ai_last_regime_id = 0;
   g_prob_calibration_last_ready = false;
   g_prob_calibration_last_fallback_used = false;
   g_prob_calibration_last_abstain = false;
   g_prob_calibration_last_calibration_stale = true;
   g_prob_calibration_last_input_stale = true;
   g_prob_calibration_last_generated_at = 0;
   g_prob_calibration_last_method = "LOGISTIC_AFFINE";
   g_prob_calibration_last_tier_kind = "GLOBAL";
   g_prob_calibration_last_tier_key = "GLOBAL|*|*|*";
   g_prob_calibration_last_support = 0;
   g_prob_calibration_last_quality = 0.0;
   g_prob_calibration_last_raw_score = 0.0;
   g_prob_calibration_last_raw_action = "SKIP";
   g_prob_calibration_last_raw_buy_prob = 0.0;
   g_prob_calibration_last_raw_sell_prob = 0.0;
   g_prob_calibration_last_raw_skip_prob = 1.0;
   g_prob_calibration_last_buy_prob = 0.0;
   g_prob_calibration_last_sell_prob = 0.0;
   g_prob_calibration_last_skip_prob = 1.0;
   g_prob_calibration_last_confidence = 0.0;
   g_prob_calibration_last_move_mean = 0.0;
   g_prob_calibration_last_move_q25 = 0.0;
   g_prob_calibration_last_move_q50 = 0.0;
   g_prob_calibration_last_move_q75 = 0.0;
   g_prob_calibration_last_spread_cost = 0.0;
   g_prob_calibration_last_slippage_cost = 0.0;
   g_prob_calibration_last_uncertainty_score = 0.0;
   g_prob_calibration_last_uncertainty_penalty = 0.0;
   g_prob_calibration_last_risk_penalty = 0.0;
   g_prob_calibration_last_gross_edge = 0.0;
   g_prob_calibration_last_edge_after_costs = 0.0;
   g_prob_calibration_last_final_action = "SKIP";
   g_prob_calibration_last_session = "UNKNOWN";
   g_prob_calibration_last_regime = "UNKNOWN";
   g_prob_calibration_last_reasons_csv = "";
   g_prob_calibration_last_primary_reason = "";
   g_execution_quality_last_ready = false;
   g_execution_quality_last_fallback_used = false;
   g_execution_quality_last_memory_stale = true;
   g_execution_quality_last_data_stale = true;
   g_execution_quality_last_support_usable = false;
   g_execution_quality_last_generated_at = 0;
   g_execution_quality_last_method = "SCORECARD_V1";
   g_execution_quality_last_tier_kind = "GLOBAL";
   g_execution_quality_last_tier_key = "GLOBAL|*|*|*";
   g_execution_quality_last_support = 0;
   g_execution_quality_last_quality = 0.0;
   g_execution_quality_last_spread_now = 0.0;
   g_execution_quality_last_spread_expected = 0.0;
   g_execution_quality_last_spread_widening_risk = 0.0;
   g_execution_quality_last_expected_slippage = 0.0;
   g_execution_quality_last_slippage_risk = 0.0;
   g_execution_quality_last_fill_quality = 0.0;
   g_execution_quality_last_latency_sensitivity = 0.0;
   g_execution_quality_last_liquidity_fragility = 0.0;
   g_execution_quality_last_quality_score = 0.0;
   g_execution_quality_last_allowed_deviation = 0.0;
   g_execution_quality_last_caution_lot_scale = 1.0;
   g_execution_quality_last_caution_enter_prob_buffer = 0.0;
   g_execution_quality_last_state = "UNKNOWN";
   g_execution_quality_last_reasons_csv = "";
   g_ai_warmup_done = (!AI_Warmup);
   FXAI_ParseHorizonList(AI_Horizons, PredictionTargetMinutes, g_horizon_minutes);
   FXAI_ResetModelHyperParams();
   FXAI_ResetReliabilityPending();
   FXAI_ResetHorizonPolicyPending();
   FXAI_ResetStackPending();
   FXAI_ResetPolicyState();
   FXAI_ResetConformalState();
   FXAI_ResetAdaptiveRoutingState();
   FXAI_ResetRegimeCalibration();
   FXAI_ResetReplayReservoir();
   FXAI_ResetFeatureDriftDiagnostics();
   FXAI_ResetGlobalSharedTransferBackbone();
   FXAI_ResetAnalogMemory();
   FXAI_ResetRegimeGraph();
   FXAI_ResetBrokerExecutionReplayStats();
   FXAI_ResetMacroEventStore();
   if(g_plugins_ready)
      g_plugins.ResetAll();
   for(int i=0; i<FXAI_AI_COUNT; i++)
      FXAI_ResetModelAuxState(i);
   FXAI_LoadMetaArtifacts(symbol);
   FXAI_LoadRuntimeArtifacts(symbol);
   for(int s=0; s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      g_context_symbol_utility[s] = 0.0;
      g_context_symbol_stability[s] = 0.0;
      g_context_symbol_lead[s] = 0.0;
      g_context_symbol_coverage[s] = 0.0;
      g_context_symbol_utility_ready[s] = false;
   }

   if(!g_norm_windows_ready)
   {
      int windows[];
      int default_w = FXAI_GetNormDefaultWindow();
      FXAI_BuildNormWindowsFromGroups(default_w, default_w, default_w, default_w, windows);
      FXAI_ApplyNormWindows(windows, default_w);
   }
   else
   {
      FXAI_ApplyNormWindows(g_norm_feature_windows, g_norm_default_window);
   }

   for(int i=0; i<FXAI_AI_COUNT; i++)
   {
      g_ai_trained[i] = false;
      g_ai_last_train_bar[i] = 0;
   }
}
