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

