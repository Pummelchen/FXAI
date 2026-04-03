#ifndef __FXAI_RUNTIME_TRADE_RISK_MQH__
#define __FXAI_RUNTIME_TRADE_RISK_MQH__
bool FXAI_RegimeKillSwitchTriggered(string &reason)
{
   reason = "ok";
   if(g_ai_last_trade_gate <= FXAI_Clamp(RiskKillTradeGate, 0.0, 1.0))
   {
      reason = "kill_trade_gate";
      return true;
   }
   if(g_policy_last_no_trade_prob >= 0.92)
   {
      reason = "kill_policy_no_trade";
      return true;
   }
   if(g_policy_last_action == FXAI_POLICY_ACTION_EXIT &&
      g_policy_last_enter_prob < 0.25)
   {
      reason = "kill_policy_exit";
      return true;
   }
   if(g_ai_last_path_risk >= FXAI_Clamp(RiskKillPathRisk, 0.0, 1.0))
   {
      reason = "kill_path_risk";
      return true;
   }
   if(g_ai_last_fill_risk >= FXAI_Clamp(RiskKillFillRisk, 0.0, 1.0))
   {
      reason = "kill_fill_risk";
      return true;
   }
   if(g_policy_last_hold_quality < 0.18 &&
      (g_ai_last_trade_gate < 0.36 || g_ai_last_path_risk > 0.82))
   {
      reason = "kill_policy_hold";
      return true;
   }
   return false;
}

double FXAI_CalcRiskAwareLot(const string symbol,
                             const int direction,
                             string &reason)
{
   reason = "ok";
   if(direction != 0 && direction != 1)
   {
      reason = "invalid_direction";
      return 0.0;
   }

   FXAILiveDeploymentProfile deploy_profile;
   FXAI_LoadLiveDeploymentProfile(symbol, deploy_profile, false);

   if(g_ai_last_confidence < FXAI_Clamp(RiskMinConfidence, 0.0, 1.0))
   {
      reason = "risk_confidence_floor";
      return 0.0;
   }
   if(g_ai_last_reliability < FXAI_Clamp(RiskMinReliability, 0.0, 1.0))
   {
      reason = "risk_reliability_floor";
      return 0.0;
   }
   if(g_ai_last_path_risk > FXAI_Clamp(RiskMaxPathRisk, 0.0, 1.0))
   {
      reason = "risk_path_cap";
      return 0.0;
   }
   if(g_ai_last_fill_risk > FXAI_Clamp(RiskMaxFillRisk, 0.0, 1.0))
   {
      reason = "risk_fill_cap";
      return 0.0;
   }
   if(g_ai_last_trade_gate < FXAI_Clamp(RiskMinTradeGate, 0.0, 1.0))
   {
      reason = "risk_trade_gate_floor";
      return 0.0;
   }
   if(g_policy_last_enter_prob < 0.05)
   {
      reason = "risk_policy_enter_floor";
      return 0.0;
   }
   if(g_policy_last_no_trade_prob > FXAI_Clamp(deploy_profile.policy_no_trade_cap, 0.25, 0.95))
   {
      reason = "risk_policy_no_trade";
      return 0.0;
   }
   if(g_ai_last_hierarchy_score < FXAI_Clamp(RiskMinHierarchyScore, 0.0, 1.0))
   {
      reason = "risk_hierarchy_score_floor";
      return 0.0;
   }
   if(g_ai_last_hierarchy_consistency < FXAI_Clamp(RiskMinHierarchyConsistency, 0.0, 1.0))
   {
      reason = "risk_hierarchy_consistency_floor";
      return 0.0;
   }
   if(g_ai_last_hierarchy_tradability < FXAI_Clamp(RiskMinHierarchyTradability, 0.0, 1.0))
   {
      reason = "risk_hierarchy_tradability_floor";
      return 0.0;
   }
   if(g_ai_last_hierarchy_execution < FXAI_Clamp(RiskMinHierarchyExecution, 0.0, 1.0))
   {
      reason = "risk_hierarchy_execution_floor";
      return 0.0;
   }
   if(FXAI_MacroEventLeakageSafe() &&
      g_ai_last_macro_state_quality < MathMax(FXAI_Clamp(RiskMinMacroStateQuality, 0.0, 1.0),
                                              FXAI_Clamp(deploy_profile.macro_quality_floor, 0.0, 1.0)))
   {
      reason = "risk_macro_state_floor";
      return 0.0;
   }

   double requested_lot = Lot;
   double hard_cap_lot = 1000000.0;
   double edge_scale = FXAI_Clamp(g_ai_last_trade_edge_points / MathMax(g_ai_last_min_move_points, 0.25), -1.0, 4.0);
   double conviction = FXAI_Clamp(0.20 +
                                  0.22 * FXAI_Clamp(g_ai_last_confidence, 0.0, 1.0) +
                                  0.18 * FXAI_Clamp(g_ai_last_reliability, 0.0, 1.0) +
                                  0.16 * FXAI_Clamp(g_ai_last_trade_gate, 0.0, 1.0) +
                                  0.12 * FXAI_Clamp(g_ai_last_hierarchy_score, 0.0, 1.0) +
                                  0.08 * FXAI_Clamp(g_ai_last_hierarchy_consistency, 0.0, 1.0) +
                                  0.10 * FXAI_Clamp(g_ai_last_context_strength / 2.0, 0.0, 1.0) +
                                  0.10 * (1.0 - FXAI_Clamp(g_ai_last_path_risk, 0.0, 1.0)) +
                                  0.08 * (1.0 - FXAI_Clamp(g_ai_last_fill_risk, 0.0, 1.0)) +
                                  0.06 * FXAI_Clamp(g_ai_last_macro_state_quality, 0.0, 1.0) +
                                  0.10 * FXAI_Clamp(edge_scale / 2.0, 0.0, 1.0),
                                  0.20,
                                  1.60);
   conviction *= FXAI_Clamp(g_policy_last_size_mult, 0.25, 1.60);
   conviction *= FXAI_Clamp(deploy_profile.portfolio_budget_bias, 0.40, 1.60);
   conviction *= FXAI_Clamp(0.75 + 0.35 * g_policy_last_portfolio_fit, 0.25, 1.25);
   conviction *= FXAI_Clamp(deploy_profile.capital_efficiency_bias, 0.40, 1.80) *
                 FXAI_Clamp(0.70 + 0.45 * g_policy_last_capital_efficiency, 0.25, 1.40);
   conviction = FXAI_Clamp(conviction, 0.20, 2.20);

   if(AI_PositionSizing == FXAI_SIZE_CONVICTION)
      requested_lot *= conviction;

   double risk_budget_lot = 0.0;
   double risk_budget_pct = MathMax(RiskPerTradePct, 0.0);
   double money_per_point = FXAI_MoneyPerPointPerLot(symbol);
   double risk_points = FXAI_EstimatedRiskPointsForDecision();
   if(risk_budget_pct > 0.0 && money_per_point > 0.0 && risk_points > 0.0)
   {
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(equity > 0.0)
      {
         double risk_budget_money = equity * (risk_budget_pct / 100.0);
         risk_budget_lot = risk_budget_money / (money_per_point * risk_points);
         if(risk_budget_lot > 0.0 && risk_budget_lot < hard_cap_lot)
            hard_cap_lot = risk_budget_lot;
      }
   }

   if(AI_PositionSizing == FXAI_SIZE_VOL_TARGET)
   {
      if(risk_budget_lot > 0.0)
      {
         double vol_scale = FXAI_Clamp(0.55 +
                                       0.25 * conviction +
                                       0.10 * FXAI_Clamp(g_ai_last_expected_move_points / MathMax(risk_points, 0.25), 0.0, 1.5),
                                       0.35,
                                       1.00);
         requested_lot = risk_budget_lot * vol_scale;
      }
      else
      {
         requested_lot *= FXAI_Clamp(0.50 + 0.35 * conviction, 0.35, 1.10);
      }
   }
   else if(risk_budget_lot > 0.0 && requested_lot > risk_budget_lot)
   {
      requested_lot = risk_budget_lot;
   }

   if(g_ai_last_trade_edge_points < MathMax(AI_EVThresholdPoints * 0.50, 0.0))
   {
      reason = "risk_edge_floor";
      return 0.0;
   }

   double portfolio_pressure = FXAI_PortfolioPressureScore(symbol, direction);
   g_ai_last_portfolio_pressure = portfolio_pressure;
   if(portfolio_pressure > FXAI_Clamp(RiskMaxPortfolioPressure, 0.0, 1.5))
   {
      reason = "risk_portfolio_pressure";
      return 0.0;
   }
   FXAIControlPlaneAggregate cp;
   FXAI_ReadControlPlaneAggregate(symbol, direction, cp);
   FXAIPortfolioSupervisorProfile supervisor;
   double supervisor_score = FXAI_PortfolioSupervisorScore(symbol, direction, cp, supervisor);
   FXAISupervisorServiceState service_state;
   FXAI_LoadSupervisorServiceState(symbol, service_state, false);
   FXAISupervisorCommandState command_state;
   FXAI_LoadSupervisorCommandState(symbol, command_state, false);
   double service_score = FXAI_SupervisorServiceScore(direction, service_state);
   if(cp.max_capital_risk_pct > FXAI_Clamp(supervisor.capital_risk_cap_pct, 0.10, 10.0))
   {
      reason = "risk_supervisor_capital";
      return 0.0;
   }
   if(supervisor_score > FXAI_Clamp(supervisor.hard_block_score, 0.20, 3.0))
   {
      reason = "risk_supervisor_block";
      return 0.0;
   }
   if(service_state.ready)
   {
      if(service_score > FXAI_Clamp(service_state.block_score, 0.20, 3.0))
      {
         reason = "risk_supervisor_service_block";
         return 0.0;
      }
      if(g_policy_last_enter_prob < MathMax(FXAI_Clamp(service_state.entry_floor, 0.10, 0.95), 0.05))
      {
         reason = "risk_supervisor_service_entry_floor";
         return 0.0;
      }
   }
   if(command_state.ready)
   {
      if(FXAI_SupervisorCommandBlocksDirection(command_state, direction))
      {
         reason = "risk_supervisor_command_block";
         return 0.0;
      }
   }
   double control_plane_pressure = g_control_plane_last_score;
   if(direction == 1)
      control_plane_pressure = MathMax(control_plane_pressure, g_control_plane_last_buy_score);
   else if(direction == 0)
      control_plane_pressure = MathMax(control_plane_pressure, g_control_plane_last_sell_score);
   double service_entry_budget = FXAI_Clamp(service_state.ready
                                            ? (direction == 1
                                               ? service_state.long_entry_budget_mult
                                               : (direction == 0
                                                  ? service_state.short_entry_budget_mult
                                                  : service_state.budget_multiplier))
                                            : 1.0,
                                            0.10,
                                            1.20);
   requested_lot *= FXAI_Clamp((1.08 - 0.60 * portfolio_pressure) *
                               (1.02 - 0.25 * FXAI_Clamp(control_plane_pressure, 0.0, 1.5)) *
                               FXAI_Clamp(1.04 - 0.22 * supervisor_score, 0.25, 1.10) *
                               service_entry_budget *
                               FXAI_SupervisorCommandEntryBudgetMult(command_state, direction) *
                               FXAI_Clamp(g_policy_last_size_mult, 0.25, 1.60) *
                               FXAI_Clamp(0.70 + 0.30 * g_policy_last_enter_prob, 0.20, 1.10) *
                               FXAI_Clamp(0.72 + 0.28 * g_policy_last_capital_efficiency, 0.25, 1.15),
                               0.20,
                               1.10);

   double portfolio_cap = MathMax(MaxPortfolioExposureLots, 0.0);
   if(portfolio_cap > 0.0)
   {
      double available = portfolio_cap * MathMax(supervisor.gross_budget_bias, 0.40) - FXAI_ManagedExposureLots("");
      if(available <= 0.0)
      {
         reason = "risk_portfolio_cap";
         return 0.0;
      }
      if(requested_lot > available)
         requested_lot = available;
      if(available < hard_cap_lot)
         hard_cap_lot = available;
   }

   double corr_cap = MathMax(MaxCorrelatedExposureLots, 0.0);
   if(corr_cap > 0.0)
   {
      double available = corr_cap * MathMax(supervisor.correlated_budget_bias, 0.40) - FXAI_ManagedCorrelatedExposureLots(symbol);
      if(available <= 0.0)
      {
         reason = "risk_correlated_cap";
         return 0.0;
      }
      if(requested_lot > available)
         requested_lot = available;
      if(available < hard_cap_lot)
         hard_cap_lot = available;
   }

    double dir_cap = MathMax(MaxDirectionalClusterLots, 0.0);
    if(dir_cap > 0.0)
    {
       double available = dir_cap * MathMax(supervisor.directional_budget_bias, 0.40) - FXAI_ManagedDirectionalClusterLots(symbol, direction);
       if(available <= 0.0)
       {
          reason = "risk_directional_cluster_cap";
          return 0.0;
       }
       if(requested_lot > available)
          requested_lot = available;
       if(available < hard_cap_lot)
          hard_cap_lot = available;
    }

   requested_lot = FXAI_NormalizeLot(symbol, requested_lot);
   if(requested_lot > hard_cap_lot + 1e-9)
   {
      reason = "risk_min_volume_cap";
      return 0.0;
   }
   if(!MathIsValidNumber(requested_lot) || requested_lot <= 0.0)
   {
      reason = "risk_lot_invalid";
      return 0.0;
   }

   return requested_lot;
}

ENUM_ORDER_TYPE_FILLING FXAI_ResolveOrderFilling(const string symbol)
{
   long filling_mode = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((filling_mode & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;
   if((filling_mode & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
}

#endif // __FXAI_RUNTIME_TRADE_RISK_MQH__
