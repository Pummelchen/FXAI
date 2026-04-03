#ifndef __FXAI_AUDIT_SCENARIO_SPEC_MQH__
#define __FXAI_AUDIT_SCENARIO_SPEC_MQH__
void FXAI_AuditFillScenarioSpec(const int scenario_id,
                                FXAIAuditScenarioSpec &spec)
{
   spec.id = scenario_id;
   spec.name = "random_walk";
   spec.drift_per_bar = 0.0;
   spec.sigma_per_bar = 0.00018;
   spec.mean_revert_strength = 0.0;
   spec.vol_cluster = 0.0;
   spec.spike_prob = 0.0;
   spec.spike_scale = 0.0;
   spec.spread_points = 1.2;
   spec.macro_focus = 0.0;
   spec.world_sigma_scale = 1.0;
   spec.world_drift_bias = 0.0;
   spec.world_spread_scale = 1.0;
   spec.world_gap_prob = 0.0;
   spec.world_gap_scale = 0.0;
   spec.world_flip_prob = 0.0;
   spec.world_context_corr_bias = 0.0;
   spec.world_liquidity_stress = 0.0;
   spec.world_session_edge_focus = 0.0;
   spec.world_trend_persistence = 0.5;
   spec.world_shock_memory = 0.0;
   spec.world_recovery_bias = 0.0;
   spec.world_spread_shock_prob = 0.0;
   spec.world_spread_shock_scale = 1.0;
   spec.world_regime_transition_burst = 0.0;
   spec.world_transition_entropy = 0.0;
   spec.world_mean_revert_bias = 0.0;
   spec.world_vol_cluster_bias = 0.0;
   spec.world_shock_decay = 0.6;
   spec.world_asia_sigma_scale = 1.0;
   spec.world_london_sigma_scale = 1.0;
   spec.world_newyork_sigma_scale = 1.0;
   spec.world_asia_spread_scale = 1.0;
   spec.world_london_spread_scale = 1.0;
   spec.world_newyork_spread_scale = 1.0;

   switch(scenario_id)
   {
      case 1:
         spec.name = "drift_up";
         spec.drift_per_bar = 0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 2:
         spec.name = "drift_down";
         spec.drift_per_bar = -0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 3:
         spec.name = "mean_revert";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.mean_revert_strength = 0.22;
         spec.spread_points = 1.3;
         break;
      case 4:
         spec.name = "vol_cluster";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.vol_cluster = 0.85;
         spec.spike_prob = 0.01;
         spec.spike_scale = 4.0;
         spec.spread_points = 1.8;
         break;
      case 5:
         spec.name = "monotonic_up";
         spec.drift_per_bar = 0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 6:
         spec.name = "monotonic_down";
         spec.drift_per_bar = -0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 7:
         spec.name = "regime_shift";
         spec.drift_per_bar = 0.00008;
         spec.sigma_per_bar = 0.00015;
         spec.vol_cluster = 0.55;
         spec.spike_prob = 0.005;
         spec.spike_scale = 3.0;
         spec.spread_points = 1.5;
         break;
      case 8:
         spec.name = "market_recent";
         spec.spread_points = 1.2;
         break;
      case 9:
         spec.name = "market_trend";
         spec.spread_points = 1.2;
         break;
      case 10:
         spec.name = "market_chop";
         spec.spread_points = 1.4;
         break;
      case 11:
         spec.name = "market_session_edges";
         spec.spread_points = 1.6;
         break;
      case 12:
         spec.name = "market_spread_shock";
         spec.spread_points = 2.2;
         break;
      case 13:
         spec.name = "market_walkforward";
         spec.spread_points = 1.5;
         break;
      case 14:
         spec.name = "market_macro_event";
         spec.spread_points = 1.7;
         spec.macro_focus = 1.0;
         break;
      case 15:
         spec.name = "market_adversarial";
         spec.spread_points = 1.9;
         spec.macro_focus = 0.5;
         break;
      default:
         break;
   }

   if(spec.id >= 11)
      FXAI_AuditApplyWorldPlan(spec, _Symbol);
}

void FXAI_AuditResetMetrics(FXAIAuditScenarioMetrics &m,
                            const int ai_id,
                            const string ai_name,
                            const int family,
                            const string scenario,
                            const int bars_total)
{
   m.ai_id = ai_id;
   m.ai_name = ai_name;
   m.family = family;
   m.scenario = scenario;
   m.bars_total = bars_total;
   m.samples_total = 0;
   m.valid_preds = 0;
   m.invalid_preds = 0;
   m.buy_count = 0;
   m.sell_count = 0;
   m.skip_count = 0;
   m.true_buy_count = 0;
   m.true_sell_count = 0;
   m.true_skip_count = 0;
   m.exact_match_count = 0;
   m.directional_eval_count = 0;
   m.directional_correct_count = 0;
   m.trend_alignment_sum = 0.0;
   m.trend_alignment_count = 0;
   m.conf_sum = 0.0;
   m.rel_sum = 0.0;
   m.move_sum = 0.0;
   m.dir_conf_sum = 0.0;
   m.dir_hit_sum = 0.0;
   m.brier_sum = 0.0;
   m.calibration_abs_sum = 0.0;
   m.path_quality_abs_sum = 0.0;
   m.path_quality_count = 0;
   m.net_sum = 0.0;
   m.skip_ratio = 0.0;
   m.active_ratio = 0.0;
   m.bias_abs = 0.0;
   m.conf_drift = 0.0;
   m.brier_score = 0.0;
   m.calibration_error = 0.0;
   m.path_quality_error = 0.0;
   m.macro_event_rate = 0.0;
   m.macro_pre_rate = 0.0;
   m.macro_post_rate = 0.0;
   m.macro_importance_mean = 0.0;
   m.macro_surprise_abs_mean = 0.0;
   m.macro_data_coverage = 0.0;
   m.reset_delta = 0.0;
   m.sequence_delta = 0.0;
   m.wf_folds = 0;
   m.wf_train_samples = 0;
   m.wf_test_samples = 0;
   m.wf_train_score = 0.0;
   m.wf_test_score = 0.0;
   m.wf_test_score_std = 0.0;
   m.wf_gap = 0.0;
   m.wf_pbo = 0.0;
   m.wf_dsr = 0.0;
   m.wf_pass_rate = 0.0;
   m.score = 0.0;
   m.issue_flags = 0;
}
#endif // __FXAI_AUDIT_SCENARIO_SPEC_MQH__
