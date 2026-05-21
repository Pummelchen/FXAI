#ifndef __FXAI_RUNTIME_CONTROL_PLANE_TYPES_MQH__
#define __FXAI_RUNTIME_CONTROL_PLANE_TYPES_MQH__
#define FXAI_CONTROL_PLANE_DIR "FXAI\\ControlPlane"
#define FXAI_CONTROL_PLANE_TTL_SEC 7200
#define FXAI_PORTFOLIO_SUPERVISOR_FILE "FXAI\\Offline\\Promotions\\fxai_portfolio_supervisor.tsv"
#define FXAI_SUPERVISOR_SERVICE_GLOBAL_FILE "FXAI\\Offline\\Promotions\\fxai_supervisor_service_global.tsv"
#define FXAI_SUPERVISOR_COMMAND_GLOBAL_FILE "FXAI\\Offline\\Promotions\\fxai_supervisor_command_global.tsv"
#define FXAI_ADAPTIVE_ROUTER_REGIME_COUNT 7
#define FXAI_ADAPTIVE_ROUTER_SESSION_COUNT 5
#define FXAI_ADAPTIVE_ROUTER_MAX_REASONS 6

#ifndef FXAI_POLICY_ACTION_NO_TRADE
#define FXAI_POLICY_ACTION_NO_TRADE 0
#endif
#ifndef FXAI_POLICY_ACTION_ENTER
#define FXAI_POLICY_ACTION_ENTER 1
#endif
#ifndef FXAI_POLICY_ACTION_HOLD
#define FXAI_POLICY_ACTION_HOLD 2
#endif
#ifndef FXAI_POLICY_ACTION_EXIT
#define FXAI_POLICY_ACTION_EXIT 3
#endif
#ifndef FXAI_POLICY_ACTION_ADD
#define FXAI_POLICY_ACTION_ADD 4
#endif
#ifndef FXAI_POLICY_ACTION_REDUCE
#define FXAI_POLICY_ACTION_REDUCE 5
#endif
#ifndef FXAI_POLICY_ACTION_TIGHTEN
#define FXAI_POLICY_ACTION_TIGHTEN 6
#endif
#ifndef FXAI_POLICY_ACTION_TIMEOUT
#define FXAI_POLICY_ACTION_TIMEOUT 7
#endif

struct FXAILiveDeploymentProfile
{
   bool   ready;
   string profile_name;
   string symbol;
   double teacher_weight;
   double student_weight;
   double analog_weight;
   double foundation_weight;
   double policy_trade_floor;
   double policy_size_bias;
   double portfolio_budget_bias;
   double challenger_promote_margin;
   double regime_transition_weight;
   double macro_quality_floor;
   double policy_no_trade_cap;
   double capital_efficiency_bias;
   double supervisor_blend;
   double teacher_signal_gain;
   double student_signal_gain;
   double foundation_quality_gain;
   double macro_state_gain;
   double policy_lifecycle_gain;
   double policy_hold_floor;
   double policy_exit_floor;
   double policy_add_floor;
   double policy_reduce_floor;
   double policy_timeout_floor;
   double max_add_fraction;
   double reduce_fraction;
   int    soft_timeout_bars;
   int    hard_timeout_bars;
   string runtime_mode;
   string telemetry_level;
   double performance_budget_ms;
   bool   shadow_enabled;
   string snapshot_detail;
   int    max_runtime_models;
   string promotion_tier;
   datetime loaded_at;
};

struct FXAIPortfolioSupervisorProfile
{
   bool   ready;
   string profile_name;
   double gross_budget_bias;
   double correlated_budget_bias;
   double directional_budget_bias;
   double capital_risk_cap_pct;
   double macro_overlap_cap;
   double concentration_cap;
   double supervisor_weight;
   double hard_block_score;
   double policy_enter_floor;
   double policy_no_trade_ceiling;
   datetime loaded_at;
};

struct FXAIControlPlaneSnapshot
{
   bool   valid;
   long   login;
   ulong  magic;
   long   chart_id;
   string symbol;
   datetime bar_time;
   int    direction;
   double signal_intensity;
   double confidence;
   double reliability;
   double trade_gate;
   double hierarchy_score;
   double macro_quality;
   double trade_edge_norm;
   double expected_move_norm;
   double policy_trade_prob;
   double policy_no_trade_prob;
   double policy_enter_prob;
   double policy_exit_prob;
   double policy_add_prob;
   double policy_reduce_prob;
   double policy_tighten_prob;
   double policy_timeout_prob;
   double policy_size_mult;
   double policy_portfolio_fit;
   double policy_capital_efficiency;
   int    policy_lifecycle_action;
   double gross_exposure_lots;
   double correlated_exposure_lots;
   double directional_cluster_lots;
   double capital_risk_pct;
   double portfolio_pressure;
};

struct FXAIControlPlaneAggregate
{
   int    peer_count;
   double gross_intensity;
   double correlated_intensity;
   double directional_intensity;
   double macro_overlap;
   double quality_overlap;
   double diversity_bonus;
   double concentration_penalty;
   double max_capital_risk_pct;
   double mean_trade_prob;
   double mean_no_trade_prob;
   double mean_capital_efficiency;
   double mean_portfolio_fit;
   double supervisor_score;
   double score;
};

struct FXAISupervisorServiceState
{
   bool   ready;
   string profile_name;
   string symbol;
   datetime generated_at;
   datetime expires_at;
   int    snapshot_count;
   double gross_pressure;
   double directional_long_pressure;
   double directional_short_pressure;
   double macro_pressure;
   double concentration_pressure;
   double freshness_penalty;
   double pressure_velocity;
   double gross_velocity;
   double long_entry_budget_mult;
   double short_entry_budget_mult;
   double budget_multiplier;
   double add_multiplier;
   double reduce_bias;
   double exit_bias;
   double entry_floor;
   double block_score;
   double supervisor_score;
   datetime loaded_at;
};

struct FXAIStudentRouterProfile
{
   bool   ready;
   string profile_name;
   string symbol;
   bool   champion_only;
   int    max_active_models;
   double min_meta_weight;
   string allow_plugins_csv;
   string plugin_weights_csv;
   double family_weight[FXAI_FAMILY_OTHER + 1];
   datetime loaded_at;
};

struct FXAIAdaptiveRouterProfile
{
   bool   ready;
   bool   enabled;
   string profile_name;
   string symbol;
   string router_mode;
   bool   fallback_to_student_router_only;
   string pair_tags_csv;
   double caution_threshold;
   double abstain_threshold;
   double block_threshold;
   double confidence_floor;
   double suppression_threshold;
   double downweight_threshold;
   double stale_news_abstain_bias;
   bool   stale_news_force_caution;
   double min_plugin_weight;
   double max_plugin_weight;
   double max_active_weight_share;
   string plugin_global_weights_csv;
   string plugin_news_compatibility_csv;
   string plugin_liquidity_robustness_csv;
   string plugin_regime_weights_csv[FXAI_ADAPTIVE_ROUTER_REGIME_COUNT];
   string plugin_session_weights_csv[FXAI_ADAPTIVE_ROUTER_SESSION_COUNT];
   datetime loaded_at;
};

struct FXAISupervisorCommandState
{
   bool   ready;
   string profile_name;
   string symbol;
   datetime generated_at;
   datetime expires_at;
   double entry_budget_mult;
   double long_entry_budget_mult;
   double short_entry_budget_mult;
   double hold_budget_mult;
   double add_cap_mult;
   double reduce_bias;
   double exit_bias;
   double tighten_bias;
   double timeout_bias;
   bool   long_block;
   bool   short_block;
   double block_score;
   int    max_active_models;
   bool   champion_only;
   datetime loaded_at;
};

double g_control_plane_last_score = 0.0;
double g_control_plane_last_buy_score = 0.0;
double g_control_plane_last_sell_score = 0.0;
string g_control_plane_last_symbol = "";
datetime g_control_plane_last_bar_time = 0;
double g_portfolio_supervisor_last_score = 0.0;
double g_portfolio_supervisor_last_capital_risk_pct = 0.0;
double g_supervisor_service_last_score = 0.0;

void FXAI_ParseSymbolLegs(const string symbol,
                         string &base_out,
                         string &quote_out);
double FXAI_CorrelationExposureWeight(const string anchor_symbol,
                                      const string other_symbol);
double FXAI_DirectionalClusterAlignment(const string anchor_symbol,
                                        const int anchor_direction,
                                        const string other_symbol,
                                        const int other_direction);
double FXAI_ManagedExposureLots(const string symbol);
double FXAI_ManagedCorrelatedExposureLots(const string symbol);
double FXAI_ManagedDirectionalClusterLots(const string symbol,
                                          const int direction);
double FXAI_EstimatedRiskPointsForDecision();
double FXAI_MoneyPerPointPerLot(const string symbol);

void FXAI_ResetLiveDeploymentProfile(FXAILiveDeploymentProfile &out)
{
   out.ready = false;
   out.profile_name = "";
   out.symbol = "";
   out.teacher_weight = 0.58;
   out.student_weight = 0.42;
   out.analog_weight = 0.18;
   out.foundation_weight = 0.24;
   out.policy_trade_floor = 0.52;
   out.policy_size_bias = 1.0;
   out.portfolio_budget_bias = 1.0;
   out.challenger_promote_margin = 1.0;
   out.regime_transition_weight = 0.35;
   out.macro_quality_floor = 0.24;
   out.policy_no_trade_cap = 0.62;
   out.capital_efficiency_bias = 1.0;
   out.supervisor_blend = 0.45;
   out.teacher_signal_gain = 1.0;
   out.student_signal_gain = 1.0;
   out.foundation_quality_gain = 1.0;
   out.macro_state_gain = 1.0;
   out.policy_lifecycle_gain = 1.0;
   out.policy_hold_floor = 0.48;
   out.policy_exit_floor = 0.58;
   out.policy_add_floor = 0.68;
   out.policy_reduce_floor = 0.56;
   out.policy_timeout_floor = 0.72;
   out.max_add_fraction = 0.50;
   out.reduce_fraction = 0.35;
   out.soft_timeout_bars = 8;
   out.hard_timeout_bars = 18;
   out.runtime_mode = "research";
   out.telemetry_level = "full";
   out.performance_budget_ms = 12.0;
   out.shadow_enabled = true;
   out.snapshot_detail = "full";
   out.max_runtime_models = 12;
   out.promotion_tier = "experimental";
   out.loaded_at = 0;
}

void FXAI_ResetPortfolioSupervisorProfile(FXAIPortfolioSupervisorProfile &out)
{
   out.ready = false;
   out.profile_name = "";
   out.gross_budget_bias = 1.0;
   out.correlated_budget_bias = 1.0;
   out.directional_budget_bias = 1.0;
   out.capital_risk_cap_pct = 1.20;
   out.macro_overlap_cap = 0.92;
   out.concentration_cap = 0.82;
   out.supervisor_weight = 0.45;
   out.hard_block_score = 1.08;
   out.policy_enter_floor = 0.42;
   out.policy_no_trade_ceiling = 0.74;
   out.loaded_at = 0;
}

void FXAI_ResetControlPlaneSnapshot(FXAIControlPlaneSnapshot &out)
{
   out.valid = false;
   out.login = 0;
   out.magic = 0;
   out.chart_id = 0;
   out.symbol = "";
   out.bar_time = 0;
   out.direction = -1;
   out.signal_intensity = 0.0;
   out.confidence = 0.0;
   out.reliability = 0.0;
   out.trade_gate = 0.0;
   out.hierarchy_score = 0.0;
   out.macro_quality = 0.0;
   out.trade_edge_norm = 0.0;
   out.expected_move_norm = 0.0;
   out.policy_trade_prob = 0.0;
   out.policy_no_trade_prob = 1.0;
   out.policy_enter_prob = 0.0;
   out.policy_exit_prob = 0.0;
   out.policy_add_prob = 0.0;
   out.policy_reduce_prob = 0.0;
   out.policy_tighten_prob = 0.0;
   out.policy_timeout_prob = 0.0;
   out.policy_size_mult = 0.0;
   out.policy_portfolio_fit = 0.0;
   out.policy_capital_efficiency = 0.0;
   out.policy_lifecycle_action = FXAI_POLICY_ACTION_NO_TRADE;
   out.gross_exposure_lots = 0.0;
   out.correlated_exposure_lots = 0.0;
   out.directional_cluster_lots = 0.0;
   out.capital_risk_pct = 0.0;
   out.portfolio_pressure = 0.0;
}

void FXAI_ResetControlPlaneAggregate(FXAIControlPlaneAggregate &out)
{
   out.peer_count = 0;
   out.gross_intensity = 0.0;
   out.correlated_intensity = 0.0;
   out.directional_intensity = 0.0;
   out.macro_overlap = 0.0;
   out.quality_overlap = 0.0;
   out.diversity_bonus = 0.0;
   out.concentration_penalty = 0.0;
   out.max_capital_risk_pct = 0.0;
   out.mean_trade_prob = 0.0;
   out.mean_no_trade_prob = 0.0;
   out.mean_capital_efficiency = 0.0;
   out.mean_portfolio_fit = 0.0;
   out.supervisor_score = 0.0;
   out.score = 0.0;
}

void FXAI_ResetSupervisorServiceState(FXAISupervisorServiceState &out)
{
   out.ready = false;
   out.profile_name = "";
   out.symbol = "";
   out.generated_at = 0;
   out.expires_at = 0;
   out.snapshot_count = 0;
   out.gross_pressure = 0.0;
   out.directional_long_pressure = 0.0;
   out.directional_short_pressure = 0.0;
   out.macro_pressure = 0.0;
   out.concentration_pressure = 0.0;
   out.freshness_penalty = 0.0;
   out.pressure_velocity = 0.0;
   out.gross_velocity = 0.0;
   out.long_entry_budget_mult = 1.0;
   out.short_entry_budget_mult = 1.0;
   out.budget_multiplier = 1.0;
   out.add_multiplier = 1.0;
   out.reduce_bias = 0.0;
   out.exit_bias = 0.0;
   out.entry_floor = 0.42;
   out.block_score = 1.10;
   out.supervisor_score = 0.0;
   out.loaded_at = 0;
}

void FXAI_ResetStudentRouterProfile(FXAIStudentRouterProfile &out)
{
   out.ready = false;
   out.profile_name = "";
   out.symbol = "";
   out.champion_only = false;
   out.max_active_models = 12;
   out.min_meta_weight = 0.0;
   out.allow_plugins_csv = "";
   out.plugin_weights_csv = "";
   for(int i=0; i<=FXAI_FAMILY_OTHER; i++)
      out.family_weight[i] = 1.0;
   out.loaded_at = 0;
}

void FXAI_ResetAdaptiveRouterProfile(FXAIAdaptiveRouterProfile &out)
{
   out.ready = false;
   out.enabled = false;
   out.profile_name = "";
   out.symbol = "";
   out.router_mode = "WEIGHTED_ENSEMBLE";
   out.fallback_to_student_router_only = true;
   out.pair_tags_csv = "";
   out.caution_threshold = 0.55;
   out.abstain_threshold = 0.35;
   out.block_threshold = 0.16;
   out.confidence_floor = 0.12;
   out.suppression_threshold = 0.34;
   out.downweight_threshold = 0.78;
   out.stale_news_abstain_bias = 0.24;
   out.stale_news_force_caution = true;
   out.min_plugin_weight = 0.05;
   out.max_plugin_weight = 1.80;
   out.max_active_weight_share = 0.72;
   out.plugin_global_weights_csv = "";
   out.plugin_news_compatibility_csv = "";
   out.plugin_liquidity_robustness_csv = "";
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
      out.plugin_regime_weights_csv[i] = "";
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_SESSION_COUNT; i++)
      out.plugin_session_weights_csv[i] = "";
   out.loaded_at = 0;
}

void FXAI_ResetSupervisorCommandState(FXAISupervisorCommandState &out)
{
   out.ready = false;
   out.profile_name = "";
   out.symbol = "";
   out.generated_at = 0;
   out.expires_at = 0;
   out.entry_budget_mult = 1.0;
   out.long_entry_budget_mult = 1.0;
   out.short_entry_budget_mult = 1.0;
   out.hold_budget_mult = 1.0;
   out.add_cap_mult = 1.0;
   out.reduce_bias = 0.0;
   out.exit_bias = 0.0;
   out.tighten_bias = 0.0;
   out.timeout_bias = 0.0;
   out.long_block = false;
   out.short_block = false;
   out.block_score = 1.10;
   out.max_active_models = 12;
   out.champion_only = false;
   out.loaded_at = 0;
}

#endif // __FXAI_RUNTIME_CONTROL_PLANE_TYPES_MQH__
