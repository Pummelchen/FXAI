   ulong policy_stage_t0 = GetMicrosecondCount();
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
                                                     1.45) *
                                  FXAI_Clamp(deploy_profile.student_signal_gain, 0.40, 1.80));
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
         if(dynamic_ensemble_applied && dynamic_ensemble_state.ready)
         {
            double dynamic_blend = FXAI_Clamp(0.30 + 0.50 * dynamic_ensemble_state.ensemble_quality, 0.25, 0.80);
            ensemble_probs[(int)FXAI_LABEL_SELL] = FXAI_Clamp(dynamic_blend * dynamic_ensemble_state.sell_prob +
                                                              (1.0 - dynamic_blend) * ensemble_probs[(int)FXAI_LABEL_SELL],
                                                              0.0005,
                                                              0.9990);
            ensemble_probs[(int)FXAI_LABEL_BUY] = FXAI_Clamp(dynamic_blend * dynamic_ensemble_state.buy_prob +
                                                             (1.0 - dynamic_blend) * ensemble_probs[(int)FXAI_LABEL_BUY],
                                                             0.0005,
                                                             0.9990);
            ensemble_probs[(int)FXAI_LABEL_SKIP] = FXAI_Clamp(dynamic_blend * dynamic_ensemble_state.skip_prob +
                                                              (1.0 - dynamic_blend) * ensemble_probs[(int)FXAI_LABEL_SKIP],
                                                              0.0005,
                                                              0.9990);
            ps = ensemble_probs[0] + ensemble_probs[1] + ensemble_probs[2];
            if(ps <= 0.0) ps = 1.0;
            ensemble_probs[0] /= ps; ensemble_probs[1] /= ps; ensemble_probs[2] /= ps;
         }

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
         g_policy_last_add_prob = policy_decision.add_prob;
         g_policy_last_reduce_prob = policy_decision.reduce_prob;
         g_policy_last_tighten_prob = policy_decision.tighten_prob;
         g_policy_last_timeout_prob = policy_decision.timeout_prob;
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
   if(dynamic_ensemble_applied && dynamic_ensemble_state.ready)
   {
      FXAI_DynamicEnsembleApplyPosture(dynamic_ensemble_state, decision);
      if(decision != -1)
      {
         if(decision == 1 && dynamic_ensemble_state.final_score < -0.08)
            decision = -1;
         else if(decision == 0 && dynamic_ensemble_state.final_score > 0.08)
            decision = -1;
      }
   }
   string adaptive_router_posture = "NORMAL";
   double adaptive_router_abstain_bias = 0.0;
   if(adaptive_router_active)
   {
      int adaptive_eligible_count = 0;
      double adaptive_best_suitability = 0.0;
      for(int ai_i=0; ai_i<FXAI_AI_COUNT; ai_i++)
      {
         if(adaptive_router_status[ai_i] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
            continue;
         if(adaptive_router_suitability[ai_i] > adaptive_best_suitability)
            adaptive_best_suitability = adaptive_router_suitability[ai_i];
         if(routed_meta_selected[ai_i] && routed_meta_weight[ai_i] > 0.0)
            adaptive_eligible_count++;
      }
      adaptive_router_posture = FXAI_AdaptiveRouterComputePosture(adaptive_router_profile,
                                                                  adaptive_regime_state,
                                                                  adaptive_best_suitability,
                                                                  adaptive_eligible_count);
      adaptive_router_abstain_bias = FXAI_AdaptiveRouterPostureAbstainBias(adaptive_router_profile,
                                                                           adaptive_regime_state,
                                                                           adaptive_router_posture);
      FXAI_AdaptiveRouterApplyPosture(adaptive_router_posture,
                                      adaptive_router_abstain_bias,
                                      decision);
   }
   FXAI_AdaptiveRouterWriteRuntimeArtifacts(symbol,
                                            adaptive_router_profile,
                                            adaptive_regime_state,
                                            adaptive_router_posture,
                                            adaptive_router_abstain_bias,
                                            adaptive_router_suitability,
                                            routed_meta_weight,
                                            routed_meta_selected,
                                            adaptive_router_status);
   FXAI_DynamicEnsembleWriteRuntimeArtifacts(symbol,
                                             dynamic_ensemble_state,
                                             dynamic_records,
                                             decision);
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_POLICY,
                             (double)(GetMicrosecondCount() - policy_stage_t0) / 1000.0);
