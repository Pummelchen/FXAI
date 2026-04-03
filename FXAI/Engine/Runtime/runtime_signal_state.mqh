#ifndef __FXAI_RUNTIME_SIGNAL_STATE_MQH__
#define __FXAI_RUNTIME_SIGNAL_STATE_MQH__

struct FXAIRuntimeSignalCache
{
   double expected_move_points;
   double trade_edge_points;
   double confidence;
   double reliability;
   double path_risk;
   double fill_risk;
   double trade_gate;
   double hierarchy_score;
   double hierarchy_consistency;
   double hierarchy_tradability;
   double hierarchy_execution;
   double hierarchy_horizon_fit;
   double macro_state_quality;
   double portfolio_pressure;
   double context_quality;
   double context_strength;
   double min_move_points;
   int horizon_minutes;
   int regime_id;
   double policy_trade_prob;
   double policy_no_trade_prob;
   double policy_enter_prob;
   double policy_exit_prob;
   double policy_direction_bias;
   double policy_size_mult;
   double policy_hold_quality;
   double policy_expected_utility;
   double policy_confidence;
   double policy_portfolio_fit;
   double policy_capital_efficiency;
   double policy_add_prob;
   double policy_reduce_prob;
   double policy_tighten_prob;
   double policy_timeout_prob;
   int policy_action;
   double control_plane_score;
   double control_plane_buy_score;
   double control_plane_sell_score;
   string control_plane_symbol;
   datetime control_plane_bar_time;
};

void FXAI_RuntimePublishIdleSnapshot(const string symbol)
{
   if(StringLen(symbol) <= 0)
      return;
   FXAI_WriteControlPlaneLocalSnapshot(symbol, -1, 0.0);
}

void FXAI_RuntimeCaptureSignalCache(FXAIRuntimeSignalCache &out)
{
   out.expected_move_points = g_ai_last_expected_move_points;
   out.trade_edge_points = g_ai_last_trade_edge_points;
   out.confidence = g_ai_last_confidence;
   out.reliability = g_ai_last_reliability;
   out.path_risk = g_ai_last_path_risk;
   out.fill_risk = g_ai_last_fill_risk;
   out.trade_gate = g_ai_last_trade_gate;
   out.hierarchy_score = g_ai_last_hierarchy_score;
   out.hierarchy_consistency = g_ai_last_hierarchy_consistency;
   out.hierarchy_tradability = g_ai_last_hierarchy_tradability;
   out.hierarchy_execution = g_ai_last_hierarchy_execution;
   out.hierarchy_horizon_fit = g_ai_last_hierarchy_horizon_fit;
   out.macro_state_quality = g_ai_last_macro_state_quality;
   out.portfolio_pressure = g_ai_last_portfolio_pressure;
   out.context_quality = g_ai_last_context_quality;
   out.context_strength = g_ai_last_context_strength;
   out.min_move_points = g_ai_last_min_move_points;
   out.horizon_minutes = g_ai_last_horizon_minutes;
   out.regime_id = g_ai_last_regime_id;
   out.policy_trade_prob = g_policy_last_trade_prob;
   out.policy_no_trade_prob = g_policy_last_no_trade_prob;
   out.policy_enter_prob = g_policy_last_enter_prob;
   out.policy_exit_prob = g_policy_last_exit_prob;
   out.policy_direction_bias = g_policy_last_direction_bias;
   out.policy_size_mult = g_policy_last_size_mult;
   out.policy_hold_quality = g_policy_last_hold_quality;
   out.policy_expected_utility = g_policy_last_expected_utility;
   out.policy_confidence = g_policy_last_confidence;
   out.policy_portfolio_fit = g_policy_last_portfolio_fit;
   out.policy_capital_efficiency = g_policy_last_capital_efficiency;
   out.policy_add_prob = g_policy_last_add_prob;
   out.policy_reduce_prob = g_policy_last_reduce_prob;
   out.policy_tighten_prob = g_policy_last_tighten_prob;
   out.policy_timeout_prob = g_policy_last_timeout_prob;
   out.policy_action = g_policy_last_action;
   out.control_plane_score = g_control_plane_last_score;
   out.control_plane_buy_score = g_control_plane_last_buy_score;
   out.control_plane_sell_score = g_control_plane_last_sell_score;
   out.control_plane_symbol = g_control_plane_last_symbol;
   out.control_plane_bar_time = g_control_plane_last_bar_time;
}

void FXAI_RuntimeResetSignalState(void)
{
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
   g_policy_last_add_prob = 0.0;
   g_policy_last_reduce_prob = 0.0;
   g_policy_last_tighten_prob = 0.0;
   g_policy_last_timeout_prob = 0.0;
   g_policy_last_action = FXAI_POLICY_ACTION_NO_TRADE;
   g_control_plane_last_score = 0.0;
   g_control_plane_last_buy_score = 0.0;
   g_control_plane_last_sell_score = 0.0;
   g_control_plane_last_symbol = "";
   g_control_plane_last_bar_time = 0;
}

void FXAI_RuntimeRestoreSignalCache(const FXAIRuntimeSignalCache &cache)
{
   g_ai_last_expected_move_points = cache.expected_move_points;
   g_ai_last_trade_edge_points = cache.trade_edge_points;
   g_ai_last_confidence = cache.confidence;
   g_ai_last_reliability = cache.reliability;
   g_ai_last_path_risk = cache.path_risk;
   g_ai_last_fill_risk = cache.fill_risk;
   g_ai_last_trade_gate = cache.trade_gate;
   g_ai_last_hierarchy_score = cache.hierarchy_score;
   g_ai_last_hierarchy_consistency = cache.hierarchy_consistency;
   g_ai_last_hierarchy_tradability = cache.hierarchy_tradability;
   g_ai_last_hierarchy_execution = cache.hierarchy_execution;
   g_ai_last_hierarchy_horizon_fit = cache.hierarchy_horizon_fit;
   g_ai_last_macro_state_quality = cache.macro_state_quality;
   g_ai_last_portfolio_pressure = cache.portfolio_pressure;
   g_ai_last_context_quality = cache.context_quality;
   g_ai_last_context_strength = cache.context_strength;
   g_ai_last_min_move_points = cache.min_move_points;
   g_ai_last_horizon_minutes = cache.horizon_minutes;
   g_ai_last_regime_id = cache.regime_id;
   g_policy_last_trade_prob = cache.policy_trade_prob;
   g_policy_last_no_trade_prob = cache.policy_no_trade_prob;
   g_policy_last_enter_prob = cache.policy_enter_prob;
   g_policy_last_exit_prob = cache.policy_exit_prob;
   g_policy_last_direction_bias = cache.policy_direction_bias;
   g_policy_last_size_mult = cache.policy_size_mult;
   g_policy_last_hold_quality = cache.policy_hold_quality;
   g_policy_last_expected_utility = cache.policy_expected_utility;
   g_policy_last_confidence = cache.policy_confidence;
   g_policy_last_portfolio_fit = cache.policy_portfolio_fit;
   g_policy_last_capital_efficiency = cache.policy_capital_efficiency;
   g_policy_last_add_prob = cache.policy_add_prob;
   g_policy_last_reduce_prob = cache.policy_reduce_prob;
   g_policy_last_tighten_prob = cache.policy_tighten_prob;
   g_policy_last_timeout_prob = cache.policy_timeout_prob;
   g_policy_last_action = cache.policy_action;
   g_control_plane_last_score = cache.control_plane_score;
   g_control_plane_last_buy_score = cache.control_plane_buy_score;
   g_control_plane_last_sell_score = cache.control_plane_sell_score;
   g_control_plane_last_symbol = cache.control_plane_symbol;
   g_control_plane_last_bar_time = cache.control_plane_bar_time;
}

#endif // __FXAI_RUNTIME_SIGNAL_STATE_MQH__
