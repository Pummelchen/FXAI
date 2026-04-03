#ifndef __FXAI_ENGINE_RUNTIME_MQH__
#define __FXAI_ENGINE_RUNTIME_MQH__

void FXAI_RuntimePublishIdleSnapshot(const string symbol)
{
   if(StringLen(symbol) <= 0)
      return;
   FXAI_WriteControlPlaneLocalSnapshot(symbol, -1, 0.0);
}

int SpecialDirectionAI(const string symbol)
{
   g_ai_last_reason = "start";
   double cached_expected_move_points = g_ai_last_expected_move_points;
   double cached_trade_edge_points = g_ai_last_trade_edge_points;
   double cached_confidence = g_ai_last_confidence;
   double cached_reliability = g_ai_last_reliability;
   double cached_path_risk = g_ai_last_path_risk;
   double cached_fill_risk = g_ai_last_fill_risk;
   double cached_trade_gate = g_ai_last_trade_gate;
   double cached_hierarchy_score = g_ai_last_hierarchy_score;
   double cached_hierarchy_consistency = g_ai_last_hierarchy_consistency;
   double cached_hierarchy_tradability = g_ai_last_hierarchy_tradability;
   double cached_hierarchy_execution = g_ai_last_hierarchy_execution;
   double cached_hierarchy_horizon_fit = g_ai_last_hierarchy_horizon_fit;
   double cached_macro_state_quality = g_ai_last_macro_state_quality;
   double cached_portfolio_pressure = g_ai_last_portfolio_pressure;
   double cached_context_quality = g_ai_last_context_quality;
   double cached_context_strength = g_ai_last_context_strength;
   double cached_min_move_points = g_ai_last_min_move_points;
   int cached_horizon_minutes = g_ai_last_horizon_minutes;
   int cached_regime_id = g_ai_last_regime_id;
   double cached_policy_trade_prob = g_policy_last_trade_prob;
   double cached_policy_no_trade_prob = g_policy_last_no_trade_prob;
   double cached_policy_enter_prob = g_policy_last_enter_prob;
   double cached_policy_exit_prob = g_policy_last_exit_prob;
   double cached_policy_direction_bias = g_policy_last_direction_bias;
   double cached_policy_size_mult = g_policy_last_size_mult;
   double cached_policy_hold_quality = g_policy_last_hold_quality;
   double cached_policy_expected_utility = g_policy_last_expected_utility;
   double cached_policy_confidence = g_policy_last_confidence;
   double cached_policy_portfolio_fit = g_policy_last_portfolio_fit;
   double cached_policy_capital_efficiency = g_policy_last_capital_efficiency;
   int cached_policy_action = g_policy_last_action;
   double cached_control_plane_score = g_control_plane_last_score;
   double cached_control_plane_buy_score = g_control_plane_last_buy_score;
   double cached_control_plane_sell_score = g_control_plane_last_sell_score;
   string cached_control_plane_symbol = g_control_plane_last_symbol;
   datetime cached_control_plane_bar_time = g_control_plane_last_bar_time;
   g_ai_last_expected_move_points = 0.0;
   g_ai_last_trade_edge_points = 0.0;
   g_ai_last_confidence = 0.0;
   g_ai_last_reliability = 0.0;
   g_ai_last_path_risk = 1.0;
   g_ai_last_fill_risk = 1.0;
   g_ai_last_trade_gate = 0.0;
   g_ai_last_hierarchy_score = 0.0;
   g_ai_last_hierarchy_consistency = 0.0;
   g_ai_last_hierarchy_tradability = 0.0;
   g_ai_last_hierarchy_execution = 0.0;
   g_ai_last_hierarchy_horizon_fit = 0.0;
   g_ai_last_macro_state_quality = 0.0;
   g_ai_last_portfolio_pressure = 0.0;
   g_ai_last_context_quality = 0.0;
   g_ai_last_context_strength = 0.0;
   g_ai_last_min_move_points = 0.0;
   g_ai_last_horizon_minutes = 0;
   g_ai_last_regime_id = 0;
   g_policy_last_trade_prob = 0.0;
   g_policy_last_no_trade_prob = 1.0;
   g_policy_last_enter_prob = 0.0;
   g_policy_last_exit_prob = 0.0;
   g_policy_last_direction_bias = 0.0;
   g_policy_last_size_mult = 1.0;
   g_policy_last_hold_quality = 0.0;
   g_policy_last_expected_utility = 0.0;
   g_policy_last_confidence = 0.0;
   g_policy_last_portfolio_fit = 0.0;
   g_policy_last_capital_efficiency = 0.0;
   g_policy_last_action = FXAI_POLICY_ACTION_NO_TRADE;
   g_control_plane_last_score = 0.0;
   g_control_plane_last_buy_score = 0.0;
   g_control_plane_last_sell_score = 0.0;
   g_control_plane_last_symbol = "";
   g_control_plane_last_bar_time = 0;
   if(!g_plugins_ready)
   {
      g_ai_last_reason = "plugins_not_ready";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);

   int base = AI_Window;
   if(base < 50) base = 50;
   if(base > 500) base = 500;

   int K = AI_OnlineSamples;
   if(K < 5) K = 5;
   if(K > 200) K = 200;

   int onlineEpochs = AI_OnlineEpochs;
   if(onlineEpochs < 1) onlineEpochs = 1;
   if(onlineEpochs > 5) onlineEpochs = 5;

   int trainEpochs = AI_Epochs;
   if(trainEpochs < 1) trainEpochs = 1;
   if(trainEpochs > 20) trainEpochs = 20;

   int aiType = (int)AI_Type;
   if(aiType < 0 || aiType >= FXAI_AI_COUNT)
      aiType = (int)AI_SGD_LOGIT;

   bool ensembleMode = (bool)AI_Ensemble;
   double agreePct = FXAI_Clamp(Ensemble_AgreePct, 50.0, 100.0);

   double buyThr = AI_BuyThreshold;
   double sellThr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(buyThr, sellThr);

   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int evLookback = AI_EVLookbackSamples;
   if(evLookback < 20) evLookback = 20;
   if(evLookback > 400) evLookback = 400;

   if(g_ai_last_symbol != symbol)
      ResetAIState(symbol);

   if(AI_Warmup && !g_ai_warmup_done)
   {
      if(!FXAI_WarmupTrainAndTune(symbol))
      {
         g_ai_last_reason = "warmup_pending";
         FXAI_RuntimePublishIdleSnapshot(symbol);
         return -1;
      }
   }

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar == 0)
   {
      g_ai_last_reason = "bar_time_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int pctKey = (int)MathRound(agreePct * 10.0);
   int decisionKey = (ensembleMode == 1 ? (100000 + pctKey) : aiType);
   if(g_ai_last_signal_bar == signal_bar && g_ai_last_signal_key == decisionKey)
   {
      g_ai_last_expected_move_points = cached_expected_move_points;
      g_ai_last_trade_edge_points = cached_trade_edge_points;
      g_ai_last_confidence = cached_confidence;
      g_ai_last_reliability = cached_reliability;
      g_ai_last_path_risk = cached_path_risk;
      g_ai_last_fill_risk = cached_fill_risk;
      g_ai_last_trade_gate = cached_trade_gate;
      g_ai_last_hierarchy_score = cached_hierarchy_score;
      g_ai_last_hierarchy_consistency = cached_hierarchy_consistency;
      g_ai_last_hierarchy_tradability = cached_hierarchy_tradability;
      g_ai_last_hierarchy_execution = cached_hierarchy_execution;
      g_ai_last_hierarchy_horizon_fit = cached_hierarchy_horizon_fit;
      g_ai_last_macro_state_quality = cached_macro_state_quality;
      g_ai_last_portfolio_pressure = cached_portfolio_pressure;
      g_ai_last_context_quality = cached_context_quality;
      g_ai_last_context_strength = cached_context_strength;
      g_ai_last_min_move_points = cached_min_move_points;
      g_ai_last_horizon_minutes = cached_horizon_minutes;
      g_ai_last_regime_id = cached_regime_id;
      g_policy_last_trade_prob = cached_policy_trade_prob;
      g_policy_last_no_trade_prob = cached_policy_no_trade_prob;
      g_policy_last_enter_prob = cached_policy_enter_prob;
      g_policy_last_exit_prob = cached_policy_exit_prob;
      g_policy_last_direction_bias = cached_policy_direction_bias;
      g_policy_last_size_mult = cached_policy_size_mult;
      g_policy_last_hold_quality = cached_policy_hold_quality;
      g_policy_last_expected_utility = cached_policy_expected_utility;
      g_policy_last_confidence = cached_policy_confidence;
      g_policy_last_portfolio_fit = cached_policy_portfolio_fit;
      g_policy_last_capital_efficiency = cached_policy_capital_efficiency;
      g_policy_last_action = cached_policy_action;
      g_control_plane_last_score = cached_control_plane_score;
      g_control_plane_last_buy_score = cached_control_plane_buy_score;
      g_control_plane_last_sell_score = cached_control_plane_sell_score;
      g_control_plane_last_symbol = cached_control_plane_symbol;
      g_control_plane_last_bar_time = cached_control_plane_bar_time;
      g_ai_last_reason = "signal_cache_hit";
      return g_ai_last_signal;
   }

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
   {
      g_ai_last_reason = "snapshot_export_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   // Keep cache/training keyed to the same closed bar anchor.
   snapshot.bar_time = signal_bar;
   FXAI_ResetFeatureNormalizationState();

   const int FEATURE_LB = 10;
   int horizon_load_max = FXAI_GetMaxConfiguredHorizon(base_h);
   int needed = (K > base ? K : base) + horizon_load_max + FEATURE_LB;
   if(needed < 128) needed = 128;
   int align_upto = needed - 1;

   static MqlRates rates_m1[];
   static MqlRates rates_m5[];
   static MqlRates rates_m15[];
   static MqlRates rates_m30[];
   static MqlRates rates_h1[];
   static string cache_symbol = "";
   static datetime last_bar_m1 = 0;
   static datetime last_bar_m5 = 0;
   static datetime last_bar_m15 = 0;
   static datetime last_bar_m30 = 0;
   static datetime last_bar_h1 = 0;

   static double open_arr[];
   static double high_arr[];
   static double low_arr[];
   static double close_arr[];
   static datetime time_arr[];
   static int spread_m1[];
   static FXAIContextSeries ctx_series[];
   static double ctx_mean_arr[];
   static double ctx_std_arr[];
   static double ctx_up_arr[];
   static double ctx_extra_arr[];

   if(cache_symbol != symbol)
   {
      cache_symbol = symbol;
      last_bar_m1 = 0;
      last_bar_m5 = 0;
      last_bar_m15 = 0;
      last_bar_m30 = 0;
      last_bar_h1 = 0;
      ArrayResize(rates_m1, 0);
      ArrayResize(rates_m5, 0);
      ArrayResize(rates_m15, 0);
      ArrayResize(rates_m30, 0);
      ArrayResize(rates_h1, 0);
      ArrayResize(ctx_series, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   int signal_seq = g_rel_clock_seq;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, last_bar_m1, rates_m1))
   {
      g_ai_last_reason = "m1_series_load_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   FXAI_ExtractRatesCloseTimeSpread(rates_m1, close_arr, time_arr, spread_m1);
   FXAI_ExtractRatesOHLC(rates_m1, open_arr, high_arr, low_arr, close_arr);
   if(ArraySize(close_arr) < needed || ArraySize(time_arr) < needed || ArraySize(spread_m1) < needed)
   {
      g_ai_last_reason = "m1_series_size_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }
   if(!FXAI_ValidateM1SeriesBundle(time_arr, open_arr, high_arr, low_arr, close_arr, spread_m1, needed))
   {
      g_ai_last_reason = "m1_series_integrity_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int needed_m5 = (needed / 5) + 80;
   int needed_m15 = (needed / 15) + 80;
   int needed_m30 = (needed / 30) + 80;
   int needed_h1 = (needed / 60) + 80;
   if(needed_m5 < 220) needed_m5 = 220;
   if(needed_m15 < 220) needed_m15 = 220;
   if(needed_m30 < 220) needed_m30 = 220;
   if(needed_h1 < 220) needed_h1 = 220;

   static double close_m5[];
   static datetime time_m5[];
   static double close_m15[];
   static datetime time_m15[];
   static double close_m30[];
   static datetime time_m30[];
   static double close_h1[];
   static datetime time_h1[];
   static int map_m5[];
   static int map_m15[];
   static int map_m30[];
   static int map_h1[];
   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M5, needed_m5, last_bar_m5, rates_m5))
      FXAI_ExtractRatesCloseTime(rates_m5, close_m5, time_m5);
   else
   {
      ArrayResize(close_m5, 0);
      ArrayResize(time_m5, 0);
      ArrayResize(map_m5, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M15, needed_m15, last_bar_m15, rates_m15))
      FXAI_ExtractRatesCloseTime(rates_m15, close_m15, time_m15);
   else
   {
      ArrayResize(close_m15, 0);
      ArrayResize(time_m15, 0);
      ArrayResize(map_m15, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_M30, needed_m30, last_bar_m30, rates_m30))
      FXAI_ExtractRatesCloseTime(rates_m30, close_m30, time_m30);
   else
   {
      ArrayResize(close_m30, 0);
      ArrayResize(time_m30, 0);
      ArrayResize(map_m30, 0);
   }

   if(FXAI_UpdateRatesRolling(symbol, PERIOD_H1, needed_h1, last_bar_h1, rates_h1))
      FXAI_ExtractRatesCloseTime(rates_h1, close_h1, time_h1);
   else
   {
      ArrayResize(close_h1, 0);
      ArrayResize(time_h1, 0);
      ArrayResize(map_h1, 0);
   }

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   FXAI_BuildAlignedIndexMapRange(time_arr, time_m5, lag_m5, align_upto, map_m5);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m15, lag_m15, align_upto, map_m15);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_m30, lag_m30, align_upto, map_m30);
   FXAI_BuildAlignedIndexMapRange(time_arr, time_h1, lag_h1, align_upto, map_h1);

   int ctx_count = ArraySize(g_context_symbols);
   if(ctx_count > FXAI_MAX_CONTEXT_SYMBOLS) ctx_count = FXAI_MAX_CONTEXT_SYMBOLS;
   if(ArraySize(ctx_series) != ctx_count)
   {
      ArrayResize(ctx_series, ctx_count);
      for(int s=0; s<ctx_count; s++)
      {
         ctx_series[s].loaded = false;
         ctx_series[s].symbol = "";
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }
   for(int s=0; s<ctx_count; s++)
   {
      string ctx_symbol = g_context_symbols[s];
      if(ctx_series[s].symbol != ctx_symbol)
      {
         ctx_series[s].symbol = ctx_symbol;
         ctx_series[s].last_bar_time = 0;
         ArrayResize(ctx_series[s].rates, 0);
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }

      ctx_series[s].loaded = FXAI_UpdateRatesRolling(ctx_symbol,
                                                    PERIOD_M1,
                                                    needed,
                                                    ctx_series[s].last_bar_time,
                                                    ctx_series[s].rates);
      if(ctx_series[s].loaded)
      {
         FXAI_ExtractRatesCloseTimeSpread(ctx_series[s].rates,
                                         ctx_series[s].close,
                                         ctx_series[s].time,
                                         ctx_series[s].spread);
         FXAI_ExtractRatesOHLC(ctx_series[s].rates,
                               ctx_series[s].open,
                               ctx_series[s].high,
                               ctx_series[s].low,
                               ctx_series[s].close);
         if(!FXAI_ValidateM1SeriesBundle(ctx_series[s].time,
                                         ctx_series[s].open,
                                         ctx_series[s].high,
                                         ctx_series[s].low,
                                         ctx_series[s].close,
                                         ctx_series[s].spread,
                                         needed))
         {
            ctx_series[s].loaded = false;
         }
      }
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].open, 0);
         ArrayResize(ctx_series[s].high, 0);
         ArrayResize(ctx_series[s].low, 0);
         ArrayResize(ctx_series[s].close, 0);
         ArrayResize(ctx_series[s].time, 0);
         ArrayResize(ctx_series[s].spread, 0);
         ArrayResize(ctx_series[s].aligned_idx, 0);
      }
   }

   FXAI_PrecomputeDynamicContextAggregates(time_arr,
                                           close_arr,
                                           ctx_series,
                                           ctx_count,
                                           align_upto,
                                           ctx_mean_arr,
                                           ctx_std_arr,
                                           ctx_up_arr,
                                           ctx_extra_arr);

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double profile_commission_points = FXAI_GetCommissionPointsRoundTripPerLot(symbol,
                                                                              exec_profile.commission_per_lot_side);
   if(profile_commission_points > commission_points)
      commission_points = profile_commission_points;
   snapshot.commission_points = commission_points;
   double spread_pred = FXAI_GetSpreadAtIndex(0, spread_m1, snapshot.spread_points);
   double min_move_pred = FXAI_ExecutionEntryCostPoints(spread_pred,
                                                        commission_points,
                                                        cost_buffer_points,
                                                        exec_profile);
   if(min_move_pred < 0.0) min_move_pred = 0.0;
   snapshot.min_move_points = min_move_pred;
   FXAILiveDeploymentProfile deploy_profile;
   FXAI_LoadLiveDeploymentProfile(symbol, deploy_profile, false);
   double vol_hint = MathAbs(FXAI_SafeReturn(close_arr, 0, 1));
   int regime_hint = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_hint);
   int ai_hint = (ensembleMode ? -1 : aiType);
   double ctx_util = 0.0, ctx_stability = 0.0, ctx_lead = 0.0, ctx_coverage = 0.0;
   FXAI_GetDynamicContextState(ctx_util, ctx_stability, ctx_lead, ctx_coverage);
   double context_strength = FXAI_Clamp(MathAbs(FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0)) +
                                        FXAI_GetArrayValue(ctx_std_arr, 0, 0.0) +
                                        MathAbs(FXAI_GetArrayValue(ctx_up_arr, 0, 0.5) - 0.5),
                                        0.0,
                                        4.0);
   double context_quality = FXAI_Clamp(0.45 * ctx_util +
                                       0.25 * ctx_stability +
                                       0.20 * ctx_lead +
                                       0.10 * ctx_coverage,
                                       -1.0,
                                       2.0);
   g_ai_last_context_strength = context_strength;
   g_ai_last_context_quality = context_quality;
   g_ai_last_min_move_points = min_move_pred;
   double model_reliability_hint = 0.50;
   if(ai_hint >= 0 && ai_hint < FXAI_AI_COUNT)
      model_reliability_hint = FXAI_Clamp(g_model_reliability[ai_hint], 0.0, 1.0);

   int H = FXAI_SelectRoutedHorizon(close_arr,
                                    snapshot,
                                    min_move_pred,
                                    evLookback,
                                    base_h,
                                    regime_hint,
                                    ai_hint,
                                    context_strength,
                                    context_quality,
                                    model_reliability_hint);
   int init_start = H;
   int init_end = H + base - 1;
   int online_start = H;
   int online_end = H + K - 1;
   int shadow_samples = Ensemble_ShadowSamples;
   if(shadow_samples < 8) shadow_samples = 8;
   if(shadow_samples > 200) shadow_samples = 200;
   int shadow_epochs = Ensemble_ShadowEpochs;
   if(shadow_epochs < 1) shadow_epochs = 1;
   if(shadow_epochs > 3) shadow_epochs = 3;
   int shadow_every = Ensemble_ShadowEveryBars;
   if(shadow_every < 1) shadow_every = 1;
   bool run_shadow = (ensembleMode && FXAI_IsShadowBar(shadow_every, signal_seq));
   int shadow_start = H;
   int shadow_end = H + shadow_samples - 1;

   int max_valid = needed - FEATURE_LB - 1;
   if(init_end > max_valid) init_end = max_valid;
   if(online_end > max_valid) online_end = max_valid;
   if(shadow_end > max_valid) shadow_end = max_valid;
   bool have_init_window = (init_end >= init_start);
   bool have_online_window = (online_end >= online_start);
   bool have_shadow_window = (shadow_end >= shadow_start);

   int precompute_end = -1;
   if(have_init_window) precompute_end = init_end;
   if(have_online_window && online_end > precompute_end) precompute_end = online_end;
   if(run_shadow && have_shadow_window && shadow_end > precompute_end) precompute_end = shadow_end;

   double ctx_mean_pred = FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FXAI_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FXAI_GetArrayValue(ctx_up_arr, 0, 0.5);

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_GetFeatureNormalizationMethod();
   double feat_pred[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(0,
                                _Symbol,
                                spread_pred,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                spread_m1,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_pred,
                                ctx_std_pred,
                                ctx_up_pred,
                                ctx_extra_arr,
                                norm_method,
                                feat_pred))
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "predict_features_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   double fallback_expected_move = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                                      H,
                                                                      evLookback,
                                                                      snapshot.point);
   if(fallback_expected_move <= 0.0)
      fallback_expected_move = 0.0;
   double vol_proxy_abs = FXAI_RollingReturnStd(close_arr, 0, 10);
   if(vol_proxy_abs < 1e-6)
      vol_proxy_abs = FXAI_RollingAbsReturn(close_arr, 0, 10);
   if(vol_proxy_abs < 1e-6)
      vol_proxy_abs = vol_hint;
   FXAI_UpdateRegimeEMAs(spread_pred, vol_proxy_abs);
   int regime_id = FXAI_GetRegimeId(snapshot.bar_time, spread_pred, vol_proxy_abs);
   g_ai_last_horizon_minutes = H;
   g_ai_last_regime_id = regime_id;
   double hpolicy_feat[FXAI_HPOL_FEATS];
   FXAI_BuildHorizonPolicyFeatures(H,
                                   base_h,
                                   fallback_expected_move,
                                   min_move_pred,
                                   snapshot,
                                   MathAbs(FXAI_SafeReturn(close_arr, 0, 1)),
                                   regime_id,
                                   ai_hint,
                                   context_strength,
                                   context_quality,
                                   model_reliability_hint,
                                   hpolicy_feat);
   FXAI_EnqueueHorizonPolicyPending(signal_seq, regime_id, H, min_move_pred, hpolicy_feat);

   static FXAIPreparedSample samples[];
   if(precompute_end >= 1)
   {
      // Start at 1 (not H) so rolling normalizers see the full recent past for
      // prediction-time feature scaling, even when horizon H is large.
      FXAI_PrecomputeTrainingSamples(1,
                                    precompute_end,
                                    H,
                                    commission_points,
                                    cost_buffer_points,
                                    evThresholdPoints,
                                    snapshot,
                                    spread_m1,
                                    time_arr,
                                    open_arr,
                                    high_arr,
                                    low_arr,
                                    close_arr,
                                    time_m5,
                                    close_m5,
                                    map_m5,
                                    time_m15,
                                    close_m15,
                                    map_m15,
                                    time_m30,
                                    close_m30,
                                    map_m30,
                                    time_h1,
                                    close_h1,
                                    map_h1,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    ctx_extra_arr,
                                    -1,
                                    samples);
   }

   if(have_online_window && online_start >= 0 && online_start < ArraySize(samples))
      FXAI_AddReplaySample(samples[online_start]);

   double current_raw_x[FXAI_AI_WEIGHTS];
   current_raw_x[0] = 1.0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      current_raw_x[f + 1] = feat_pred[f];
   double current_shared_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   int current_shared_window_size = 0;
   FXAI_ClearInputWindow(current_shared_window, current_shared_window_size);
   int current_shared_span = FXAI_ContextSequenceSpan(24, H, _Symbol, 8);
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
                                                     1.45));
   }
   int runtime_session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
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
   FXAIAnalogMemoryQuery current_analog_q;
   FXAI_QueryAnalogMemory(current_raw_x,
                          regime_id,
                          runtime_session_bucket,
                          H,
                          FXAI_SymbolHash01(snapshot.symbol),
                          current_analog_q);
   g_ai_last_macro_state_quality = FXAI_Clamp(FXAI_GetInputFeature(current_raw_x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19), 0.0, 1.0);
   FXAIRegimeGraphQuery current_regime_q;
   FXAI_QueryRegimeGraph(regime_id,
                         g_ai_last_macro_state_quality,
                         current_regime_q);
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

   int active_ai_ids[];
   ArrayResize(active_ai_ids, 0);
   if(ensembleMode == 0)
   {
      if(g_plugins.Get(aiType) != NULL)
      {
         ArrayResize(active_ai_ids, 1);
         active_ai_ids[0] = aiType;
      }
   }
   else
   {
      int cand_ai_ids[];
      double cand_scores[];
      ArrayResize(cand_ai_ids, 0);
      ArrayResize(cand_scores, 0);

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(g_plugins.Get(ai_idx) == NULL) continue;
         if(FXAI_IsModelPruned(ai_idx, regime_id)) continue;
         double score = FXAI_GetModelMetaScore(ai_idx, regime_id, runtime_session_bucket, H, min_move_pred);
         if(score <= 0.0) continue;
         int sz = ArraySize(cand_ai_ids);
         ArrayResize(cand_ai_ids, sz + 1);
         ArrayResize(cand_scores, sz + 1);
         cand_ai_ids[sz] = ai_idx;
         cand_scores[sz] = score;
      }

      int cand_n = ArraySize(cand_ai_ids);
      if(cand_n > 0)
      {
         int top_k = cand_n;
         if(top_k > 10) top_k = 10;
         if(top_k < 2 && cand_n >= 2) top_k = 2;

         bool used[];
         ArrayResize(used, cand_n);
         for(int j=0; j<cand_n; j++) used[j] = false;

         ArrayResize(active_ai_ids, top_k);
         int picked = 0;
         for(int pick=0; pick<top_k; pick++)
         {
            int best_j = -1;
            double best_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               if(cand_scores[j] > best_sc)
               {
                  best_sc = cand_scores[j];
                  best_j = j;
               }
            }
            if(best_j < 0) break;
            used[best_j] = true;
            active_ai_ids[picked] = cand_ai_ids[best_j];
            picked++;
         }
         if(picked < ArraySize(active_ai_ids))
            ArrayResize(active_ai_ids, picked);

         // Exploration arm: occasionally add one non-top model (cold-start biased)
         // to improve adaptation and avoid starvation under pruning.
         if(cand_n > picked && FXAI_ShouldSampleByPct(signal_bar, regime_id + 17, Ensemble_ExplorePct))
         {
            int explore_j = -1;
            double explore_sc = -1e18;
            for(int j=0; j<cand_n; j++)
            {
               if(used[j]) continue;
               int ai_id = cand_ai_ids[j];
               int obs = 0;
               if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
                  obs = g_model_regime_obs[ai_id][regime_id];
               double cold_bonus = 1.0 / MathSqrt(1.0 + (double)obs);
               double score = cand_scores[j] * (1.0 + (0.35 * cold_bonus));
               if(score > explore_sc)
               {
                  explore_sc = score;
                  explore_j = j;
               }
            }
            if(explore_j >= 0)
            {
               int add_id = cand_ai_ids[explore_j];
               if(!FXAI_IsModelInList(add_id, active_ai_ids))
               {
                  int sz = ArraySize(active_ai_ids);
                  ArrayResize(active_ai_ids, sz + 1);
                  active_ai_ids[sz] = add_id;
               }
            }
         }
      }
      else
      {
         // If all models are pruned for this regime, keep one conservative fallback.
         int fallback_id = -1;
         double fallback_rel = -1e9;
         for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
         {
            if(g_plugins.Get(ai_idx) == NULL) continue;
            double rel = FXAI_GetModelVoteWeight(ai_idx);
            if(rel > fallback_rel)
            {
               fallback_rel = rel;
               fallback_id = ai_idx;
            }
         }
         if(fallback_id >= 0)
         {
            ArrayResize(active_ai_ids, 1);
            active_ai_ids[0] = fallback_id;
         }
      }
   }

   if(ArraySize(active_ai_ids) <= 0)
   {
      g_ai_last_signal_bar = signal_bar;
      g_ai_last_signal_key = decisionKey;
      g_ai_last_signal = -1;
      g_ai_last_reason = "no_active_models";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   FXAINormSampleCache runtime_norm_caches[];
   ArrayResize(runtime_norm_caches, 0);
   FXAINormInputCache input_caches[];
   ArrayResize(input_caches, 0);

   for(int ai_pass=0; ai_pass<FXAI_AI_COUNT; ai_pass++)
   {
      bool needed_ai = false;
      if(FXAI_IsModelInList(ai_pass, active_ai_ids))
         needed_ai = true;
      else if(run_shadow && !FXAI_IsModelInList(ai_pass, active_ai_ids) && g_plugins.Get(ai_pass) != NULL)
         needed_ai = true;
      if(!needed_ai) continue;

      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_pass, regime_id, H);
      if(precompute_end >= 1)
      {
         FXAI_EnsureRoutedNormCachesForSamples(ai_pass,
                                               1,
                                               precompute_end,
                                               H,
                                               commission_points,
                                               cost_buffer_points,
                                               evThresholdPoints,
                                               snapshot,
                                               spread_m1,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_arr,
                                               ctx_std_arr,
                                               ctx_up_arr,
                                               ctx_extra_arr,
                                               samples,
                                               runtime_norm_caches);
      }

      if(FXAI_FindNormInputCache(method_id, input_caches) < 0)
      {
         FXAI_EnsureNormInputCache(method_id,
                                   spread_pred,
                                   spread_m1,
                                   snapshot,
                                   time_arr,
                                   open_arr,
                                   high_arr,
                                   low_arr,
                                   close_arr,
                                   time_m5,
                                   close_m5,
                                   map_m5,
                                   time_m15,
                                   close_m15,
                                   map_m15,
                                   time_m30,
                                   close_m30,
                                   map_m30,
                                   time_h1,
                                   close_h1,
                                   map_h1,
                                   ctx_mean_arr,
                                   ctx_std_arr,
                                   ctx_up_arr,
                                   ctx_extra_arr,
                                   input_caches);
      }
   }

   if(run_shadow && have_shadow_window)
   {
      int warm_epochs = trainEpochs;
      if(warm_epochs > 4) warm_epochs = 4;
      if(warm_epochs < 1) warm_epochs = 1;

      for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
      {
         if(FXAI_IsModelInList(ai_idx, active_ai_ids)) continue;
         CFXAIAIPlugin *plugin_shadow = g_plugins.Get(ai_idx);
         if(plugin_shadow == NULL) continue;

         FXAIAIHyperParams hp_shadow;
         FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_shadow);
         plugin_shadow.EnsureInitialized(hp_shadow);

         if(!g_ai_trained[ai_idx])
         {
            if(have_init_window)
               FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                         *plugin_shadow,
                                                         init_start,
                                                         init_end,
                                                         warm_epochs,
                                                         samples,
                                                         runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_trained[ai_idx] = true;
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }

         if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin_shadow,
                                                      shadow_start,
                                                      shadow_end,
                                                      shadow_epochs,
                                                      samples,
                                                      runtime_norm_caches);
            FXAI_TrainModelReplay(ai_idx, *plugin_shadow, regime_id, H, 1);
            g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
         }
      }
   }

   int singleSignal = -1;
   string singleNoTradeReason = "";
   double ensemble_buy_ev_sum = 0.0;
   double ensemble_sell_ev_sum = 0.0;
   double ensemble_buy_support = 0.0;
   double ensemble_sell_support = 0.0;
   double ensemble_skip_support = 0.0;
   double ensemble_meta_total = 0.0;
   double ensemble_expected_sum = 0.0;
   double ensemble_expected_sq_sum = 0.0;
   double ensemble_conf_sum = 0.0;
   double ensemble_rel_sum = 0.0;
   double ensemble_margin_sum = 0.0;
   double ensemble_hit_time_sum = 0.0;
   double ensemble_path_risk_sum = 0.0;
   double ensemble_fill_risk_sum = 0.0;
   double ensemble_mfe_ratio_sum = 0.0;
   double ensemble_mae_ratio_sum = 0.0;
   double ensemble_ctx_edge_sum = 0.0;
   double ensemble_ctx_regret_sum = 0.0;
   double ensemble_global_edge_sum = 0.0;
   double ensemble_port_edge_sum = 0.0;
   double ensemble_port_stability_sum = 0.0;
   double ensemble_port_corr_sum = 0.0;
   double ensemble_port_div_sum = 0.0;
   double ensemble_ctx_trust_sum = 0.0;
   double best_model_signal_edge = -1e12;
   double best_model_meta_w = 0.0;
   double best_buy_edge = -1e12;
   double best_sell_edge = -1e12;
   double best_buy_meta_w = 0.0;
   double best_sell_meta_w = 0.0;
   double family_support[FXAI_FAMILY_OTHER + 1];
   for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++) family_support[fam_i] = 0.0;
   double ensemble_probs[3];
   ensemble_probs[0] = 0.3333;
   ensemble_probs[1] = 0.3333;
   ensemble_probs[2] = 0.3334;
   double stack_feat[FXAI_STACK_FEATS];
   for(int sf=0; sf<FXAI_STACK_FEATS; sf++) stack_feat[sf] = 0.0;
   bool feature_drift_updated = false;
   double live_feature_drift = FXAI_GetFeatureDriftPenalty();
   double drift_norm = FXAI_Clamp(live_feature_drift / 4.0, 0.0, 1.0);

   for(int m=0; m<ArraySize(active_ai_ids); m++)
   {
      int ai_idx = active_ai_ids[m];

      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
         continue;

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);

      FXAIAIHyperParams hp_model;
      FXAI_GetModelHyperParamsRouted(ai_idx, regime_id, H, hp_model);
      plugin.EnsureInitialized(hp_model);

      if(!g_ai_trained[ai_idx])
      {
         if(have_init_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      init_start,
                                                      init_end,
                                                      trainEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_trained[ai_idx] = true;
         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }
      else if(snapshot.bar_time != g_ai_last_train_bar[ai_idx])
      {
         if(have_online_window)
         {
            FXAI_TrainModelWindowPreparedRoutedCached(ai_idx,
                                                      *plugin,
                                                      online_start,
                                                      online_end,
                                                      onlineEpochs,
                                                      samples,
                                                      runtime_norm_caches);
         }
         FXAI_TrainModelReplay(ai_idx, *plugin, regime_id, H, 1);

         g_ai_last_train_bar[ai_idx] = snapshot.bar_time;
      }

      FXAIAIPredictRequestV4 req;
      FXAI_ClearPredictRequest(req);
      req.valid = true;
      req.ctx.api_version = FXAI_API_VERSION_V4;
      req.ctx.regime_id = regime_id;
      req.ctx.session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
      req.ctx.horizon_minutes = H;
      req.ctx.feature_schema_id = manifest.feature_schema_id;
      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx, regime_id, H);
      req.ctx.normalization_method_id = method_id;
      req.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*plugin, H);
      req.ctx.min_move_points = min_move_pred;
      req.ctx.cost_points = min_move_pred;
      req.ctx.point_value = (snapshot.point > 0.0 ? snapshot.point : (_Point > 0.0 ? _Point : 1.0));
      req.ctx.domain_hash = FXAI_SymbolHash01(snapshot.symbol);
      req.ctx.sample_time = snapshot.bar_time;
      int input_idx = FXAI_FindNormInputCache(method_id, input_caches);
      if(input_idx < 0)
      {
         input_idx = FXAI_EnsureNormInputCache(method_id,
                                               spread_pred,
                                               spread_m1,
                                               snapshot,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_arr,
                                               ctx_std_arr,
                                               ctx_up_arr,
                                               ctx_extra_arr,
                                               input_caches);
      }
      if(input_idx < 0)
         continue;
      if(!feature_drift_updated)
      {
         FXAI_UpdateFeatureDriftLiveFromInput(input_caches[input_idx].x, snapshot.bar_time);
         FXAI_MarkRuntimeArtifactsDirty();
         live_feature_drift = FXAI_GetFeatureDriftPenalty();
         drift_norm = FXAI_Clamp(live_feature_drift / 4.0, 0.0, 1.0);
         feature_drift_updated = true;
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x[k] = input_caches[input_idx].x[k];
      FXAI_BuildPreparedSampleWindowCached(ai_idx, samples, 0, runtime_norm_caches, req.ctx.sequence_bars, req.x_window, req.window_size);
      FXAI_ApplyFeatureSchemaToPayloadEx(manifest.feature_schema_id,
                                       manifest.feature_groups_mask,
                                       req.ctx.sequence_bars,
                                       req.x_window,
                                       req.window_size,
                                       req.x);

      FXAIAIPredictionV4 pred;
      FXAI_PredictViaV4(*plugin, req, hp_model, pred);

      double class_probs_pred[3];
      class_probs_pred[0] = pred.class_probs[0];
      class_probs_pred[1] = pred.class_probs[1];
      class_probs_pred[2] = pred.class_probs[2];
      FXAI_ApplyRegimeCalibration(ai_idx, regime_id, class_probs_pred);

      double expected_move = pred.move_mean_points;
      if(expected_move <= 0.0)
         expected_move = FXAI_GetModelExpectedMove(ai_idx, 0.0);
      if(expected_move <= 0.0)
         expected_move = 0.0;

      double modelBuyThr = buyThr;
      double modelSellThr = sellThr;
      FXAI_GetModelThresholds(ai_idx, regime_id, H, buyThr, sellThr, modelBuyThr, modelSellThr);

      double buyMinProb = modelBuyThr;
      double sellMinProb = 1.0 - modelSellThr;
      double skipMinProb = 0.55;
      FXAI_DeriveAdaptiveThresholds(modelBuyThr,
                                   modelSellThr,
                                   min_move_pred,
                                   expected_move,
                                   feat_pred[5],
                                   buyMinProb,
                                   sellMinProb,
                                   skipMinProb);

      int signal = FXAI_ClassSignalFromEV(class_probs_pred,
                                         buyMinProb,
                                         sellMinProb,
                                         skipMinProb,
                                         expected_move,
                                         min_move_pred,
                                         evThresholdPoints);
      if(ensembleMode == 0 && ai_idx == (int)AI_M1SYNC && signal == -1)
      {
         double p_buy = class_probs_pred[(int)FXAI_LABEL_BUY];
         double p_sell = class_probs_pred[(int)FXAI_LABEL_SELL];
         double p_skip = class_probs_pred[(int)FXAI_LABEL_SKIP];
         double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move - min_move_pred;
         double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move - min_move_pred;
         bool have_chain = (expected_move > 0.0 &&
                            (p_buy >= 0.50 || p_sell >= 0.50) &&
                            p_skip < 0.80);

         if(!have_chain)
            singleNoTradeReason = "m1sync_no_chain";
         else if(p_skip >= skipMinProb)
            singleNoTradeReason = "m1sync_chain_skip_block";
         else if((p_buy >= buyMinProb && buy_ev < evThresholdPoints) ||
                 (p_sell >= sellMinProb && sell_ev < evThresholdPoints))
            singleNoTradeReason = "m1sync_chain_ev_blocked";
         else
            singleNoTradeReason = "m1sync_chain_threshold_block";
      }
      FXAI_EnqueueReliabilityPending(ai_idx,
                                     signal_seq,
                                     signal,
                                     regime_id,
                                     runtime_session_bucket,
                                     expected_move,
                                     H,
                                     class_probs_pred);
      FXAI_EnqueueConformalPending(ai_idx,
                                   signal_seq,
                                   regime_id,
                                   H,
                                   pred);

      if(ensembleMode == 0)
      {
         double buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         double single_trade_gate = FXAI_Clamp(0.35 +
                                               0.25 * FXAI_Clamp(pred.confidence, 0.0, 1.0) +
                                               0.20 * FXAI_Clamp(pred.reliability, 0.0, 1.0) +
                                               0.10 * (1.0 - FXAI_Clamp(pred.path_risk, 0.0, 1.0)) +
                                               0.10 * (1.0 - FXAI_Clamp(pred.fill_risk, 0.0, 1.0)) +
                                               0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) +
                                               0.07 * FXAI_Clamp(context_strength / 2.0, 0.0, 1.0) -
                                               0.08 * drift_norm -
                                               0.10 * class_probs_pred[(int)FXAI_LABEL_SKIP],
                                               0.0,
                                               1.0);
         double chosen_edge = MathMax(buy_ev, sell_ev);
         if(signal == 1) chosen_edge = buy_ev;
         else if(signal == 0) chosen_edge = sell_ev;
         g_ai_last_expected_move_points = MathMax(expected_move, 0.0);
         g_ai_last_trade_edge_points = chosen_edge;
         g_ai_last_confidence = FXAI_Clamp(pred.confidence, 0.0, 1.0);
         g_ai_last_reliability = FXAI_Clamp(pred.reliability * (1.0 - 0.15 * drift_norm), 0.0, 1.0);
         g_ai_last_path_risk = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         g_ai_last_fill_risk = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         g_ai_last_trade_gate = single_trade_gate;
         FXAIHierarchicalSignals single_hierarchy_sig;
         FXAI_BuildHierarchicalSignals(class_probs_pred,
                                       expected_move,
                                       min_move_pred,
                                       pred.confidence,
                                       pred.reliability,
                                       pred.path_risk,
                                       pred.fill_risk,
                                       pred.hit_time_frac,
                                       context_quality,
                                       H,
                                       current_foundation_sig,
                                       current_student_sig,
                                       current_analog_q,
                                       single_hierarchy_sig);
         g_ai_last_hierarchy_score = FXAI_Clamp(single_hierarchy_sig.score, 0.0, 1.0);
         g_ai_last_hierarchy_consistency = FXAI_Clamp(single_hierarchy_sig.consistency, 0.0, 1.0);
         g_ai_last_hierarchy_tradability = FXAI_Clamp(single_hierarchy_sig.tradability, 0.0, 1.0);
         g_ai_last_hierarchy_execution = FXAI_Clamp(single_hierarchy_sig.execution_viability, 0.0, 1.0);
         g_ai_last_hierarchy_horizon_fit = FXAI_Clamp(single_hierarchy_sig.horizon_fit, 0.0, 1.0);
         double single_stack_feat[];
         ArrayResize(single_stack_feat, FXAI_STACK_FEATS);
         for(int sf=0; sf<FXAI_STACK_FEATS; sf++)
            single_stack_feat[sf] = 0.0;
         single_stack_feat[1] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_BUY] - 1.0, -1.0, 1.0);
         single_stack_feat[2] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_SELL] - 1.0, -1.0, 1.0);
         single_stack_feat[3] = FXAI_Clamp(2.0 * class_probs_pred[(int)FXAI_LABEL_SKIP] - 1.0, -1.0, 1.0);
         single_stack_feat[20] = FXAI_Clamp(pred.confidence, 0.0, 1.0);
         single_stack_feat[21] = FXAI_Clamp(pred.reliability, 0.0, 1.0);
         single_stack_feat[49] = FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         single_stack_feat[50] = FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         single_stack_feat[80] = g_ai_last_hierarchy_score;
         single_stack_feat[81] = g_ai_last_hierarchy_consistency;
         single_stack_feat[82] = g_ai_last_hierarchy_tradability;
         single_stack_feat[83] = g_ai_last_hierarchy_execution;
         double single_policy_feat[];
         double single_policy_pressure = (class_probs_pred[(int)FXAI_LABEL_BUY] >= class_probs_pred[(int)FXAI_LABEL_SELL]
                                          ? cp_buy.score : cp_sell.score);
         FXAI_BuildPolicyFeatures(single_stack_feat,
                                  single_trade_gate,
                                  chosen_edge,
                                  expected_move,
                                  min_move_pred,
                                  g_ai_last_macro_state_quality,
                                  context_quality,
                                  context_strength,
                                  current_foundation_sig.trust,
                                  current_foundation_sig.direction_bias,
                                  current_student_sig.trust,
                                  current_analog_q.similarity,
                                  current_analog_q.quality,
                                  current_regime_q,
                                  deploy_profile,
                                  single_policy_pressure,
                                  single_policy_feat);
         FXAIPolicyDecision single_policy;
         FXAI_PolicyPredict(regime_id, single_policy_feat, deploy_profile, single_policy);
         g_policy_last_trade_prob = single_policy.trade_prob;
         g_policy_last_no_trade_prob = single_policy.no_trade_prob;
         g_policy_last_enter_prob = single_policy.enter_prob;
         g_policy_last_exit_prob = single_policy.exit_prob;
         g_policy_last_direction_bias = single_policy.direction_bias;
         g_policy_last_size_mult = single_policy.size_mult;
         g_policy_last_hold_quality = single_policy.hold_quality;
         g_policy_last_expected_utility = single_policy.expected_utility;
         g_policy_last_confidence = single_policy.confidence;
         g_policy_last_portfolio_fit = single_policy.portfolio_fit;
         g_policy_last_capital_efficiency = single_policy.capital_efficiency;
         g_policy_last_action = single_policy.action_code;
         double single_transition_guard = FXAI_Clamp(1.0 - 0.35 * regime_transition_penalty -
                                                     0.40 * macro_profile_shortfall,
                                                     0.20,
                                                     1.0);
         g_ai_last_trade_gate = FXAI_Clamp((0.40 * single_trade_gate +
                                            0.34 * single_policy.trade_prob +
                                            0.14 * single_policy.enter_prob +
                                            0.12 * single_policy.portfolio_fit) *
                                           single_transition_guard,
                                           0.0,
                                           1.0);
         if(single_policy.action_code == FXAI_POLICY_ACTION_NO_TRADE ||
            single_policy.no_trade_prob > FXAI_Clamp(deploy_profile.policy_no_trade_cap, 0.25, 0.95))
            signal = -1;
         else if(FXAI_MacroEventLeakageSafe() &&
                 g_ai_last_macro_state_quality < FXAI_Clamp(deploy_profile.macro_quality_floor, 0.0, 1.0))
            signal = -1;
         else
         {
            double single_enter_floor = MathMax(FXAI_Clamp(deploy_profile.policy_trade_floor, 0.20, 0.90),
                                                single_policy.no_trade_prob);
            single_enter_floor = MathMax(single_enter_floor, 0.42);
            if(single_policy.enter_prob < single_enter_floor)
               signal = -1;
            else if(single_policy.direction_bias > 0.12 && buy_ev >= sell_ev && buy_ev >= evThresholdPoints)
               signal = 1;
            else if(single_policy.direction_bias < -0.12 && sell_ev >= buy_ev && sell_ev >= evThresholdPoints)
               signal = 0;
         }
         FXAI_EnqueuePolicyPending(signal_seq, regime_id, H, min_move_pred, single_policy_feat);
         singleSignal = signal;
      }
      else
      {
         double meta_w = FXAI_GetModelMetaScore(ai_idx, regime_id, runtime_session_bucket, H, min_move_pred);
         if(meta_w <= 0.0) continue;

         double model_buy_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move - min_move_pred;
         double model_sell_ev = ((2.0 * class_probs_pred[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move - min_move_pred;
         model_buy_ev = FXAI_Clamp(model_buy_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         model_sell_ev = FXAI_Clamp(model_sell_ev, -10.0 * min_move_pred, 10.0 * min_move_pred);
         double mm_meta = MathMax(min_move_pred, 0.50);
         double ctx_edge_norm = FXAI_Clamp(FXAI_GetModelContextEdge(ai_idx, regime_id, H) / mm_meta, -4.0, 4.0) / 4.0;
         double ctx_regret = FXAI_Clamp(FXAI_GetModelContextRegret(ai_idx, regime_id, H), 0.0, 6.0) / 6.0;
         double global_edge_norm = FXAI_Clamp(FXAI_GetModelRegimeEdge(ai_idx, regime_id) / mm_meta, -4.0, 4.0) / 4.0;
         double port_edge_norm = FXAI_GetModelPortfolioEdgeNorm(ai_idx, mm_meta);
         double port_stability = FXAI_GetModelPortfolioStability(ai_idx);
         double port_corr = FXAI_GetModelPortfolioCorrPenalty(ai_idx);
         double port_div = FXAI_GetModelPortfolioDiversification(ai_idx);
         double ctx_trust = FXAI_GetModelContextTrust(ai_idx, regime_id, H);
         double model_best_edge = MathMax(model_buy_ev, model_sell_ev);

         ensemble_meta_total += meta_w;
         ensemble_buy_ev_sum += meta_w * model_buy_ev;
         ensemble_sell_ev_sum += meta_w * model_sell_ev;
         ensemble_expected_sum += meta_w * expected_move;
         ensemble_expected_sq_sum += meta_w * expected_move * expected_move;
         ensemble_conf_sum += meta_w * FXAI_Clamp(pred.confidence, 0.0, 1.0);
         ensemble_rel_sum += meta_w * FXAI_Clamp(pred.reliability, 0.0, 1.0);
         ensemble_margin_sum += meta_w * FXAI_Clamp(MathAbs(class_probs_pred[(int)FXAI_LABEL_BUY] - class_probs_pred[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
         ensemble_hit_time_sum += meta_w * FXAI_Clamp(pred.hit_time_frac, 0.0, 1.0);
         ensemble_path_risk_sum += meta_w * FXAI_Clamp(pred.path_risk, 0.0, 1.0);
         ensemble_fill_risk_sum += meta_w * FXAI_Clamp(pred.fill_risk, 0.0, 1.0);
         ensemble_mfe_ratio_sum += meta_w * FXAI_Clamp(pred.mfe_mean_points / MathMax(expected_move, min_move_pred), 0.0, 4.0);
         ensemble_mae_ratio_sum += meta_w * FXAI_Clamp(pred.mae_mean_points / MathMax(pred.mfe_mean_points, min_move_pred), 0.0, 2.0);
         ensemble_ctx_edge_sum += meta_w * ctx_edge_norm;
         ensemble_ctx_regret_sum += meta_w * ctx_regret;
         ensemble_global_edge_sum += meta_w * global_edge_norm;
         ensemble_port_edge_sum += meta_w * port_edge_norm;
         ensemble_port_stability_sum += meta_w * port_stability;
         ensemble_port_corr_sum += meta_w * port_corr;
         ensemble_port_div_sum += meta_w * port_div;
         ensemble_ctx_trust_sum += meta_w * ctx_trust;
         if(manifest.family >= 0 && manifest.family <= FXAI_FAMILY_OTHER)
            family_support[manifest.family] += meta_w;

         if(model_best_edge > best_model_signal_edge)
         {
            best_model_signal_edge = model_best_edge;
            best_model_meta_w = meta_w;
         }
         if(model_buy_ev > best_buy_edge)
         {
            best_buy_edge = model_buy_ev;
            best_buy_meta_w = meta_w;
         }
         if(model_sell_ev > best_sell_edge)
         {
            best_sell_edge = model_sell_ev;
            best_sell_meta_w = meta_w;
         }

         if(signal == 1) ensemble_buy_support += meta_w;
         else if(signal == 0) ensemble_sell_support += meta_w;
         else ensemble_skip_support += meta_w;
      }
   }

   int decision = -1;
   if(ensembleMode == 0)
   {
      decision = singleSignal;
   }
   else
   {
      if(ensemble_meta_total > 0.0)
      {
         double buyPct = 100.0 * (ensemble_buy_support / ensemble_meta_total);
         double sellPct = 100.0 * (ensemble_sell_support / ensemble_meta_total);
         double skipPct = 100.0 * (ensemble_skip_support / ensemble_meta_total);
         double avg_buy_ev = ensemble_buy_ev_sum / ensemble_meta_total;
         double avg_sell_ev = ensemble_sell_ev_sum / ensemble_meta_total;
         double avg_expected = ensemble_expected_sum / ensemble_meta_total;
         double avg_expected_sq = ensemble_expected_sq_sum / ensemble_meta_total;
         double avg_conf = ensemble_conf_sum / ensemble_meta_total;
         double avg_rel = ensemble_rel_sum / ensemble_meta_total;
         double avg_margin = ensemble_margin_sum / ensemble_meta_total;
         double avg_hit_time = ensemble_hit_time_sum / ensemble_meta_total;
         double avg_path_risk = ensemble_path_risk_sum / ensemble_meta_total;
         double avg_fill_risk = ensemble_fill_risk_sum / ensemble_meta_total;
         double avg_mfe_ratio = ensemble_mfe_ratio_sum / ensemble_meta_total;
         double avg_mae_ratio = ensemble_mae_ratio_sum / ensemble_meta_total;
         double avg_ctx_edge_norm = ensemble_ctx_edge_sum / ensemble_meta_total;
         double avg_ctx_regret = ensemble_ctx_regret_sum / ensemble_meta_total;
         double avg_global_edge_norm = ensemble_global_edge_sum / ensemble_meta_total;
         double avg_port_edge_norm = ensemble_port_edge_sum / ensemble_meta_total;
         double avg_port_stability = ensemble_port_stability_sum / ensemble_meta_total;
         double avg_port_corr = ensemble_port_corr_sum / ensemble_meta_total;
         double avg_port_div = ensemble_port_div_sum / ensemble_meta_total;
         double avg_ctx_trust = ensemble_ctx_trust_sum / ensemble_meta_total;
         double move_dispersion = MathSqrt(MathMax(avg_expected_sq - avg_expected * avg_expected, 0.0));
         int active_family_count = 0;
         double dominant_family_support = 0.0;
         for(int fam_i=0; fam_i<=FXAI_FAMILY_OTHER; fam_i++)
         {
            if(family_support[fam_i] > 0.0)
            {
               active_family_count++;
               if(family_support[fam_i] > dominant_family_support)
                  dominant_family_support = family_support[fam_i];
            }
         }
         double active_family_ratio = (double)active_family_count / (double)MathMax(FXAI_FAMILY_OTHER + 1, 1);
         double dominant_family_ratio = dominant_family_support / MathMax(ensemble_meta_total, 1e-6);
         double best_counterfactual_edge_norm = 0.0;
         if(best_model_signal_edge > -1e11)
            best_counterfactual_edge_norm = FXAI_Clamp(best_model_signal_edge / MathMax(min_move_pred, 0.10), -4.0, 4.0) / 4.0;
         double ensemble_vs_best_gap_norm = 0.0;
         if(best_model_signal_edge > -1e11)
            ensemble_vs_best_gap_norm = FXAI_Clamp((MathMax(best_model_signal_edge, 0.0) - MathMax(avg_buy_ev, avg_sell_ev)) / MathMax(min_move_pred, 0.10), 0.0, 4.0) / 4.0;
         double best_model_share = FXAI_Clamp(best_model_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
         double best_buy_share = FXAI_Clamp(best_buy_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
         double best_sell_share = FXAI_Clamp(best_sell_meta_w / MathMax(ensemble_meta_total, 1e-6), 0.0, 1.0);
         double vote_probs[3];
         vote_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(ensemble_sell_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(ensemble_buy_support / ensemble_meta_total, 0.0, 1.0);
         vote_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(ensemble_skip_support / ensemble_meta_total, 0.0, 1.0);
         double vs = vote_probs[0] + vote_probs[1] + vote_probs[2];
         if(vs <= 0.0) vs = 1.0;
         vote_probs[0] /= vs; vote_probs[1] /= vs; vote_probs[2] /= vs;
         FXAIHierarchicalSignals current_hierarchy_sig;
         FXAI_BuildHierarchicalSignals(vote_probs,
                                       avg_expected,
                                       min_move_pred,
                                       avg_conf,
                                       avg_rel,
                                       avg_path_risk,
                                       avg_fill_risk,
                                       avg_hit_time,
                                       context_quality,
                                       H,
                                       current_foundation_sig,
                                       current_student_sig,
                                       current_analog_q,
                                       current_hierarchy_sig);
         g_ai_last_hierarchy_score = FXAI_Clamp(current_hierarchy_sig.score, 0.0, 1.0);
         g_ai_last_hierarchy_consistency = FXAI_Clamp(current_hierarchy_sig.consistency, 0.0, 1.0);
         g_ai_last_hierarchy_tradability = FXAI_Clamp(current_hierarchy_sig.tradability, 0.0, 1.0);
         g_ai_last_hierarchy_execution = FXAI_Clamp(current_hierarchy_sig.execution_viability, 0.0, 1.0);
         g_ai_last_hierarchy_horizon_fit = FXAI_Clamp(current_hierarchy_sig.horizon_fit, 0.0, 1.0);

         FXAI_StackBuildFeatures(buyPct,
                                 sellPct,
                                 skipPct,
                                 avg_buy_ev,
                                 avg_sell_ev,
                                 min_move_pred,
                                 avg_expected,
                                 vol_proxy_abs,
                                 H,
                                 avg_conf,
                                 avg_rel,
                                 move_dispersion,
                                 avg_margin,
                                 active_family_ratio,
                                 dominant_family_ratio,
                                 context_strength,
                                 context_quality,
                                 avg_hit_time,
                                 avg_path_risk,
                                 avg_fill_risk,
                                 avg_mfe_ratio,
                                 avg_mae_ratio,
                                 avg_ctx_edge_norm,
                                 avg_ctx_regret,
                                 avg_global_edge_norm,
                                 best_counterfactual_edge_norm,
                                 ensemble_vs_best_gap_norm,
                                 avg_port_edge_norm,
                                 avg_port_stability,
                                 avg_port_corr,
                                 avg_port_div,
                                 best_model_share,
                                 best_buy_share,
                                 best_sell_share,
                                 avg_ctx_trust,
                                 current_foundation_sig.trust,
                                 current_foundation_sig.direction_bias,
                                 current_foundation_sig.move_ratio,
                                 current_student_sig.trust,
                                 current_student_sig.tradability,
                                 current_analog_q.similarity,
                                 current_analog_q.edge_norm,
                                 current_analog_q.quality,
                                 current_hierarchy_sig.consistency,
                                 current_hierarchy_sig.tradability,
                                 current_hierarchy_sig.execution_viability,
                                 current_hierarchy_sig.horizon_fit,
                                 stack_feat);
         double stack_probs_dyn[];
         ArrayResize(stack_probs_dyn, 3);
         FXAI_StackPredict(regime_id, H, stack_feat, stack_probs_dyn);
         double teacher_probs[3];
         for(int c=0; c<3; c++)
            teacher_probs[c] = FXAI_Clamp(0.58 * stack_probs_dyn[c] + 0.42 * vote_probs[c], 0.0005, 0.9990);
         double teacher_sum = teacher_probs[0] + teacher_probs[1] + teacher_probs[2];
         if(teacher_sum <= 0.0) teacher_sum = 1.0;
         teacher_probs[0] /= teacher_sum;
         teacher_probs[1] /= teacher_sum;
         teacher_probs[2] /= teacher_sum;
         FXAI_GlobalStudentUpdate(current_transfer_a,
                                  current_shared_window,
                                  current_shared_window_size,
                                  FXAI_SymbolHash01(snapshot.symbol),
                                  H,
                                  runtime_session_bucket,
                                  teacher_probs,
                                  FXAI_Clamp(avg_expected / MathMax(min_move_pred, 0.10), 0.05, 4.0),
                                  current_hierarchy_sig.tradability,
                                  current_hierarchy_sig.horizon_fit,
                                  1.0,
                                  0.010 * FXAI_Clamp(0.55 + deploy_profile.student_weight,
                                                     0.35,
                                                     1.45));
         double trade_gate_prob = FXAI_TradeGatePredict(regime_id, H, stack_feat);
         double trade_gate_floor = FXAI_Clamp(0.34 +
                                              0.18 * avg_conf +
                                              0.16 * avg_rel +
                                              0.10 * dominant_family_ratio +
                                              0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) +
                                              0.10 * (1.0 - avg_path_risk) +
                                              0.08 * (1.0 - avg_fill_risk) -
                                              0.08 * drift_norm -
                                              0.10 * vote_probs[(int)FXAI_LABEL_SKIP] +
                                              0.10 * current_hierarchy_sig.tradability +
                                              0.08 * current_hierarchy_sig.execution_viability +
                                              0.06 * current_hierarchy_sig.consistency,
                                              0.05,
                                              0.95);
         double trade_gate = FXAI_Clamp(0.65 * trade_gate_prob + 0.35 * trade_gate_floor, 0.0, 1.0);
         double trade_gate_thr = FXAI_Clamp(0.52 +
                                            0.06 * vote_probs[(int)FXAI_LABEL_SKIP] +
                                            0.05 * FXAI_Clamp(move_dispersion / MathMax(min_move_pred, 0.10), 0.0, 1.0) -
                                            0.05 * avg_conf -
                                            0.04 * avg_rel +
                                            0.03 * drift_norm -
                                            0.05 * current_hierarchy_sig.consistency -
                                            0.04 * current_hierarchy_sig.execution_viability,
                                            0.42,
                                            0.68);
         double stack_blend = FXAI_Clamp(0.40 + 0.20 * avg_conf + 0.18 * avg_rel + 0.12 * dominant_family_ratio + 0.08 * FXAI_Clamp(context_quality, 0.0, 1.5) - 0.06 * FXAI_Clamp(move_dispersion / MathMax(min_move_pred, 0.10), 0.0, 1.0),
                                         0.45,
                                         0.85);
         ensemble_probs[0] = FXAI_Clamp(stack_blend * stack_probs_dyn[0] + (1.0 - stack_blend) * vote_probs[0], 0.0005, 0.9990);
         ensemble_probs[1] = FXAI_Clamp(stack_blend * stack_probs_dyn[1] + (1.0 - stack_blend) * vote_probs[1], 0.0005, 0.9990);
         ensemble_probs[2] = FXAI_Clamp(stack_blend * stack_probs_dyn[2] + (1.0 - stack_blend) * vote_probs[2], 0.0005, 0.9990);
         double ps = ensemble_probs[0] + ensemble_probs[1] + ensemble_probs[2];
         if(ps <= 0.0) ps = 1.0;
         ensemble_probs[0] /= ps; ensemble_probs[1] /= ps; ensemble_probs[2] /= ps;

         double stack_move = MathMax(avg_expected, 0.0);
         double stack_buy_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_BUY]) - 1.0) * stack_move - min_move_pred;
         double stack_sell_ev = ((2.0 * ensemble_probs[(int)FXAI_LABEL_SELL]) - 1.0) * stack_move - min_move_pred;
         double chosen_edge = MathMax(stack_buy_ev, stack_sell_ev);
         double policy_pressure_hint = (ensemble_probs[(int)FXAI_LABEL_BUY] >= ensemble_probs[(int)FXAI_LABEL_SELL]
                                        ? cp_buy.score : cp_sell.score);
         double policy_feat[];
         FXAI_BuildPolicyFeatures(stack_feat,
                                  trade_gate,
                                  chosen_edge,
                                  stack_move,
                                  min_move_pred,
                                  g_ai_last_macro_state_quality,
                                  context_quality,
                                  context_strength,
                                  current_foundation_sig.trust,
                                  current_foundation_sig.direction_bias,
                                  current_student_sig.trust,
                                  current_analog_q.similarity,
                                  current_analog_q.quality,
                                  current_regime_q,
                                  deploy_profile,
                                  policy_pressure_hint,
                                  policy_feat);
         FXAIPolicyDecision policy_decision;
         FXAI_PolicyPredict(regime_id, policy_feat, deploy_profile, policy_decision);
         g_policy_last_trade_prob = policy_decision.trade_prob;
         g_policy_last_no_trade_prob = policy_decision.no_trade_prob;
         g_policy_last_enter_prob = policy_decision.enter_prob;
         g_policy_last_exit_prob = policy_decision.exit_prob;
         g_policy_last_direction_bias = policy_decision.direction_bias;
         g_policy_last_size_mult = policy_decision.size_mult;
         g_policy_last_hold_quality = policy_decision.hold_quality;
         g_policy_last_expected_utility = policy_decision.expected_utility;
         g_policy_last_confidence = policy_decision.confidence;
         g_policy_last_portfolio_fit = policy_decision.portfolio_fit;
         g_policy_last_capital_efficiency = policy_decision.capital_efficiency;
         g_policy_last_action = policy_decision.action_code;
         double policy_gate = FXAI_Clamp(0.40 * policy_decision.trade_prob +
                                         0.24 * policy_decision.enter_prob +
                                         0.18 * policy_decision.portfolio_fit +
                                         0.18 * trade_gate,
                                         0.0,
                                         1.0);
         double buy_policy_score = FXAI_Clamp(ensemble_probs[(int)FXAI_LABEL_BUY] +
                                              0.18 * MathMax(policy_decision.direction_bias, 0.0) +
                                              0.08 * policy_decision.expected_utility +
                                              0.10 * policy_decision.capital_efficiency +
                                              0.08 * policy_decision.portfolio_fit -
                                              0.10 * cp_buy.score,
                                              0.0,
                                              1.25);
         double sell_policy_score = FXAI_Clamp(ensemble_probs[(int)FXAI_LABEL_SELL] +
                                               0.18 * MathMax(-policy_decision.direction_bias, 0.0) +
                                               0.08 * policy_decision.expected_utility +
                                               0.10 * policy_decision.capital_efficiency +
                                               0.08 * policy_decision.portfolio_fit -
                                               0.10 * cp_sell.score,
                                               0.0,
                                               1.25);
         double regime_transition_guard = FXAI_Clamp(1.0 - 0.32 * regime_transition_penalty -
                                                     0.42 * macro_profile_shortfall,
                                                     0.20,
                                                     1.0);
         double analog_bonus = FXAI_Clamp(deploy_profile.analog_weight, 0.0, 0.80) *
                               FXAI_Clamp(current_analog_q.similarity * current_analog_q.quality, 0.0, 1.0);
         policy_gate *= regime_transition_guard;
         buy_policy_score = FXAI_Clamp(buy_policy_score +
                                       0.08 * analog_bonus +
                                       0.06 * FXAI_Clamp(current_regime_q.edge_bias, 0.0, 1.0) -
                                       0.10 * regime_transition_penalty,
                                       0.0,
                                       1.25);
         sell_policy_score = FXAI_Clamp(sell_policy_score +
                                        0.08 * analog_bonus +
                                        0.06 * MathMax(-current_regime_q.edge_bias, 0.0) -
                                        0.10 * regime_transition_penalty,
                                        0.0,
                                        1.25);
         g_ai_last_expected_move_points = stack_move;
         g_ai_last_confidence = FXAI_Clamp(avg_conf, 0.0, 1.0);
         g_ai_last_reliability = FXAI_Clamp(avg_rel * (1.0 - 0.15 * drift_norm), 0.0, 1.0);
         g_ai_last_path_risk = FXAI_Clamp(avg_path_risk, 0.0, 1.0);
         g_ai_last_fill_risk = FXAI_Clamp(avg_fill_risk, 0.0, 1.0);
         g_ai_last_trade_gate = FXAI_Clamp(policy_gate * (0.60 + 0.40 * current_hierarchy_sig.score), 0.0, 1.0);

         if(current_hierarchy_sig.consistency < 0.38 || current_hierarchy_sig.execution_viability < 0.32)
            decision = -1;
         else if(policy_decision.action_code == FXAI_POLICY_ACTION_NO_TRADE ||
                 policy_decision.no_trade_prob > FXAI_Clamp(deploy_profile.policy_no_trade_cap, 0.25, 0.95))
            decision = -1;
         else if(FXAI_MacroEventLeakageSafe() &&
                 g_ai_last_macro_state_quality < FXAI_Clamp(deploy_profile.macro_quality_floor, 0.0, 1.0))
            decision = -1;
         else if(ensemble_probs[(int)FXAI_LABEL_SKIP] >= 0.58 || skipPct >= 75.0)
            decision = -1;
         else
         {
            double policy_gate_floor = MathMax(trade_gate_thr,
                                               FXAI_Clamp(deploy_profile.policy_trade_floor, 0.20, 0.90));
            policy_gate_floor = MathMax(policy_gate_floor, policy_decision.enter_prob);
            if(policy_gate < policy_gate_floor)
               decision = -1;
            else if(buy_policy_score >= sell_policy_score &&
                    buyPct >= agreePct &&
                    stack_buy_ev >= evThresholdPoints &&
                    avg_buy_ev > avg_sell_ev)
               decision = 1;
            else if(sell_policy_score > buy_policy_score &&
                    sellPct >= agreePct &&
                    stack_sell_ev >= evThresholdPoints &&
                    avg_sell_ev > avg_buy_ev)
               decision = 0;
            else
            {
               // Conservative fallback if stack is uncertain.
               if(buyPct >= agreePct && avg_buy_ev >= evThresholdPoints && avg_buy_ev > avg_sell_ev)
                  decision = 1;
               else if(sellPct >= agreePct && avg_sell_ev >= evThresholdPoints && avg_sell_ev > avg_buy_ev)
                  decision = 0;
            }
         }
         FXAI_EnqueuePolicyPending(signal_seq, regime_id, H, min_move_pred, policy_feat);

         if(decision == 1) chosen_edge = stack_buy_ev;
         else if(decision == 0) chosen_edge = stack_sell_ev;
         g_ai_last_trade_edge_points = chosen_edge;
      }
   }

   if(ensembleMode != 0 && ensemble_meta_total > 0.0)
   {
      double ens_expected = MathMax(min_move_pred,
                                    (ensemble_expected_sum > 0.0 ? ensemble_expected_sum / ensemble_meta_total :
                                     (MathAbs(ensemble_buy_ev_sum) + MathAbs(ensemble_sell_ev_sum)) / MathMax(ensemble_meta_total, 1.0)));
      FXAI_EnqueueStackPending(signal_seq,
                               decision,
                               regime_id,
                               H,
                               ens_expected,
                               ensemble_probs,
                               stack_feat);
   }

   g_ai_last_signal_bar = signal_bar;
   g_ai_last_signal_key = decisionKey;
   g_ai_last_signal = decision;
   if(decision == 1) g_ai_last_reason = "buy";
   else if(decision == 0) g_ai_last_reason = "sell";
   else if(ensembleMode == 0 && aiType == (int)AI_M1SYNC && StringLen(singleNoTradeReason) > 0) g_ai_last_reason = singleNoTradeReason;
   else if(ensembleMode != 0 && ensemble_meta_total <= 0.0) g_ai_last_reason = "no_meta_weight";
   else g_ai_last_reason = "no_consensus_or_ev";

   double signal_intensity = FXAI_Clamp((0.55 * g_ai_last_trade_gate +
                                         0.25 * g_policy_last_trade_prob +
                                         0.20 * g_policy_last_confidence) *
                                        FXAI_Clamp(g_policy_last_size_mult, 0.25, 1.60) *
                                        FXAI_Clamp(1.0 - 0.35 * macro_profile_shortfall -
                                                   0.20 * regime_transition_penalty,
                                                   0.20,
                                                   1.0),
                                        0.0,
                                        4.0);
   if(decision < 0)
      signal_intensity = 0.0;
   FXAI_WriteControlPlaneLocalSnapshot(symbol, decision, signal_intensity);

   return decision;
}


#endif // __FXAI_ENGINE_RUNTIME_MQH__
