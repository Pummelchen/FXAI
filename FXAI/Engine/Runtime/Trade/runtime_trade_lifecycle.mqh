#ifndef __FXAI_RUNTIME_TRADE_LIFECYCLE_MQH__
#define __FXAI_RUNTIME_TRADE_LIFECYCLE_MQH__
bool FXAI_ApplyPolicyLifecycle(const string symbol,
                               const int signal_direction,
                               string &action_reason)
{
   action_reason = "lifecycle_idle";
   FXAIManagedPositionState state;
   if(!FXAI_ReadManagedPositionState(symbol, state))
      return false;

   FXAILiveDeploymentProfile deploy;
   FXAI_LoadLiveDeploymentProfile(symbol, deploy, false);
   FXAISupervisorServiceState service_state;
   FXAI_LoadSupervisorServiceState(symbol, service_state, false);
   FXAISupervisorCommandState command_state;
   FXAI_LoadSupervisorCommandState(symbol, command_state, false);
   FXAIControlPlaneAggregate cp;
   FXAI_ReadControlPlaneAggregate(symbol, state.direction, cp);
   FXAIPortfolioSupervisorProfile supervisor;
   double supervisor_score = FXAI_PortfolioSupervisorScore(symbol, state.direction, cp, supervisor);
   double service_score = FXAI_SupervisorServiceScore(state.direction, service_state);
   double lifecycle_gain = FXAI_Clamp(deploy.policy_lifecycle_gain, 0.40, 1.80);
   int held_bars = FXAI_ManagedPositionHeldBars(state);
   bool direction_match = (signal_direction == state.direction);
   bool opposite_signal = (signal_direction >= 0 && signal_direction != state.direction);
   double profit_norm = FXAI_Clamp(state.profit_points / MathMax(g_ai_last_min_move_points, 0.25), -4.0, 4.0) / 4.0;
   double cycle_profit_norm = FXAI_Clamp((state.open_profit + state.cycle_realized_profit) /
                                         MathMax(MathAbs(state.cycle_peak_profit) + 1e-6, 1.0),
                                         -1.0,
                                         1.0);
   double giveback = FXAI_Clamp(state.giveback_fraction, 0.0, 1.0);
   double time_soft = (deploy.soft_timeout_bars > 0
                       ? FXAI_Clamp((double)held_bars / (double)deploy.soft_timeout_bars, 0.0, 3.0)
                       : 0.0);
   double time_hard = (deploy.hard_timeout_bars > 0
                       ? FXAI_Clamp((double)held_bars / (double)deploy.hard_timeout_bars, 0.0, 3.0)
                       : 0.0);
   double control_pressure = FXAI_Clamp(g_control_plane_last_score / 2.0, 0.0, 1.5);
   double service_pressure = FXAI_Clamp(service_score / MathMax(service_state.block_score, 0.20), 0.0, 1.5);
   double hold_budget = FXAI_Clamp(command_state.ready ? command_state.hold_budget_mult : 1.0, 0.10, 1.20);
   double command_pressure = (command_state.ready
                              ? FXAI_Clamp(0.34 * command_state.reduce_bias +
                                           0.30 * command_state.exit_bias +
                                           0.20 * command_state.tighten_bias +
                                           0.16 * command_state.timeout_bias,
                                           0.0,
                                           1.0)
                              : 0.0);

   double add_prob = FXAI_Clamp(g_policy_last_add_prob +
                                (direction_match ? 0.16 : -0.18) +
                                0.10 * MathMax(profit_norm, 0.0) +
                                0.08 * MathMax(cycle_profit_norm, 0.0) -
                                0.14 * giveback +
                                0.10 * (lifecycle_gain - 1.0) +
                                0.08 * (hold_budget - 1.0) +
                                0.10 * g_policy_last_capital_efficiency +
                                0.08 * g_policy_last_portfolio_fit -
                                0.14 * FXAI_Clamp(g_ai_last_portfolio_pressure / 1.5, 0.0, 1.0) -
                                0.12 * control_pressure -
                                0.10 * service_pressure -
                                0.12 * command_pressure +
                                0.10 * (command_state.ready ? command_state.add_cap_mult - 1.0 : 0.0),
                                0.0,
                                1.0);
   double reduce_prob = FXAI_Clamp(g_policy_last_reduce_prob +
                                   0.16 * FXAI_Clamp(g_ai_last_portfolio_pressure / 1.5, 0.0, 1.0) +
                                   0.10 * control_pressure +
                                   0.12 * service_pressure +
                                   0.14 * (1.0 - hold_budget) +
                                   0.12 * giveback +
                                   0.08 * MathMax(1.0 - lifecycle_gain, 0.0) +
                                   0.10 * command_pressure +
                                   0.12 * MathMax(-profit_norm, 0.0) +
                                   (opposite_signal ? 0.22 : 0.0),
                                   0.0,
                                   1.0);
   double timeout_prob = FXAI_Clamp(g_policy_last_timeout_prob +
                                    0.20 * FXAI_Clamp(time_soft / 1.5, 0.0, 1.0) +
                                    0.18 * FXAI_Clamp(time_hard - 1.0, 0.0, 1.0) +
                                    0.10 * (1.0 - hold_budget) +
                                    0.10 * service_pressure +
                                    0.16 * giveback +
                                    0.08 * MathMax(1.0 - lifecycle_gain, 0.0) +
                                    0.14 * (command_state.ready ? command_state.timeout_bias : 0.0),
                                    0.0,
                                    1.0);
   double tighten_prob = FXAI_Clamp(g_policy_last_tighten_prob +
                                    0.18 * FXAI_Clamp(time_soft / 1.5, 0.0, 1.0) +
                                    0.12 * MathMax(profit_norm, 0.0) +
                                    0.10 * (1.0 - hold_budget) +
                                    0.18 * giveback +
                                    0.06 * (lifecycle_gain - 1.0) +
                                    0.10 * service_state.reduce_bias +
                                    0.14 * (command_state.ready ? command_state.tighten_bias : 0.0),
                                    0.0,
                                    1.0);
   double exit_prob = FXAI_Clamp(g_policy_last_exit_prob +
                                 0.18 * service_state.exit_bias +
                                 0.14 * FXAI_Clamp(time_hard - 1.0, 0.0, 1.0) +
                                 0.10 * (1.0 - hold_budget) +
                                 0.18 * giveback +
                                 0.08 * MathMax(1.0 - lifecycle_gain, 0.0) +
                                 0.12 * (command_state.ready ? command_state.exit_bias : 0.0) +
                                 0.12 * MathMax(-profit_norm, 0.0) +
                                 (opposite_signal ? 0.24 : 0.0),
                                 0.0,
                                 1.0);

   g_policy_last_add_prob = add_prob;
   g_policy_last_reduce_prob = reduce_prob;
   g_policy_last_timeout_prob = timeout_prob;
   g_policy_last_tighten_prob = tighten_prob;

   if(held_bars >= deploy.hard_timeout_bars &&
      deploy.hard_timeout_bars > 0 &&
      (timeout_prob >= FXAI_Clamp(deploy.policy_timeout_floor, 0.30, 0.99) || exit_prob >= FXAI_Clamp(deploy.policy_exit_floor, 0.20, 0.99)))
   {
      if(CloseAll())
      {
         ResetCycleState();
         Calc_TP();
         g_policy_last_action = FXAI_POLICY_ACTION_TIMEOUT;
         FXAI_RememberLifecycleAction(symbol, FXAI_POLICY_ACTION_TIMEOUT);
         action_reason = "lifecycle_timeout_exit";
         return true;
      }
   }

   if(exit_prob >= FXAI_Clamp(deploy.policy_exit_floor, 0.20, 0.99))
   {
      if(CloseAll())
      {
         ResetCycleState();
         Calc_TP();
         g_policy_last_action = FXAI_POLICY_ACTION_EXIT;
         FXAI_RememberLifecycleAction(symbol, FXAI_POLICY_ACTION_EXIT);
         action_reason = "lifecycle_exit";
         return true;
      }
   }

   if(reduce_prob >= FXAI_Clamp(deploy.policy_reduce_floor, 0.25, 0.99))
   {
      double reduce_fraction = FXAI_Clamp(deploy.reduce_fraction *
                                          (0.70 + 0.30 * reduce_prob) *
                                          (service_state.ready ? (0.85 + 0.35 * service_state.reduce_bias) : 1.0),
                                          0.05,
                                          0.95);
      if(!FXAI_LifecycleActionCooling(symbol, FXAI_POLICY_ACTION_REDUCE) &&
         FXAI_CloseManagedFraction(symbol, reduce_fraction))
      {
         Calc_TP();
         g_policy_last_action = FXAI_POLICY_ACTION_REDUCE;
         FXAI_RememberLifecycleAction(symbol, FXAI_POLICY_ACTION_REDUCE);
         action_reason = "lifecycle_reduce";
         return true;
      }
   }

   if(tighten_prob >= 0.52 ||
      (held_bars >= deploy.soft_timeout_bars && deploy.soft_timeout_bars > 0 && state.profit_points > 0.0))
   {
      if(!FXAI_LifecycleActionCooling(symbol, FXAI_POLICY_ACTION_TIGHTEN) &&
         FXAI_TightenManagedStops(symbol, state, tighten_prob))
      {
         g_policy_last_action = FXAI_POLICY_ACTION_TIGHTEN;
         FXAI_RememberLifecycleAction(symbol, FXAI_POLICY_ACTION_TIGHTEN);
         action_reason = "lifecycle_tighten";
         return true;
      }
   }

   if(direction_match &&
      state.pending_count <= 0 &&
      add_prob >= FXAI_Clamp(deploy.policy_add_floor, 0.30, 0.99))
   {
      string add_reason = "lifecycle_add";
      double base_lot = FXAI_CalcRiskAwareLot(symbol, state.direction, add_reason);
      if(base_lot > 0.0)
      {
         double add_lot = FXAI_NormalizeLot(symbol,
                                            base_lot *
                                            FXAI_Clamp(deploy.max_add_fraction, 0.05, 1.00) *
                                            (service_state.ready ? service_state.add_multiplier : 1.0) *
                                            (command_state.ready ? command_state.add_cap_mult : 1.0));
         if(add_lot > 0.0 &&
            add_lot <= FXAI_Clamp(state.total_volume * 1.25 + base_lot, add_lot, 1000.0) &&
            !FXAI_LifecycleActionCooling(symbol, FXAI_POLICY_ACTION_ADD) &&
            FXAI_SendLifecycleAddOrder(symbol, state.direction, add_lot, add_reason))
         {
            g_policy_last_action = FXAI_POLICY_ACTION_ADD;
            FXAI_RememberLifecycleAction(symbol, FXAI_POLICY_ACTION_ADD);
            action_reason = "lifecycle_add";
            return true;
         }
      }
   }

   if(g_policy_last_hold_quality >= FXAI_Clamp(deploy.policy_hold_floor, 0.20, 0.95))
      g_policy_last_action = FXAI_POLICY_ACTION_HOLD;
   else
      g_policy_last_action = FXAI_POLICY_ACTION_NO_TRADE;
   action_reason = "lifecycle_hold";
   return false;
}

//------------------------- CLOSE ALL --------------------------------
bool CloseAll(const string symbol = "")
{
   const int max_passes = 25;
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   int op_pause_ms = (in_tester ? 0 : 100);
   string target_symbol = symbol;

   for(int pass=0; pass<max_passes; pass++)
   {
      int pos_before = FXAI_ManagedPositionsTotal(target_symbol);
      int ord_before = FXAI_ManagedOrdersTotal(target_symbol);

      for(int i=PositionsTotal() - 1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
         if(StringLen(target_symbol) > 0 && PositionGetString(POSITION_SYMBOL) != target_symbol) continue;
         trade.PositionClose(ticket);
         if(op_pause_ms > 0) Sleep(op_pause_ms);
      }

      for(int i=OrdersTotal() - 1; i>=0; i--)
      {
         ulong ticket = OrderGetTicket(i);
         if(ticket == 0) continue;
         if(!OrderSelect(ticket)) continue;
         if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
         if(StringLen(target_symbol) > 0 && OrderGetString(ORDER_SYMBOL) != target_symbol) continue;

         long orderType = OrderGetInteger(ORDER_TYPE);
         if(orderType == ORDER_TYPE_BUY_LIMIT      || orderType == ORDER_TYPE_SELL_LIMIT ||
            orderType == ORDER_TYPE_BUY_STOP       || orderType == ORDER_TYPE_SELL_STOP  ||
            orderType == ORDER_TYPE_BUY_STOP_LIMIT || orderType == ORDER_TYPE_SELL_STOP_LIMIT)
         {
            trade.OrderDelete(ticket);
            if(op_pause_ms > 0) Sleep(op_pause_ms);
         }
      }

      int pos_after = FXAI_ManagedPositionsTotal(target_symbol);
      int ord_after = FXAI_ManagedOrdersTotal(target_symbol);
      if(pos_after == 0 && ord_after == 0)
         return true;

      // no progress; avoid hard lock in OnTick if broker rejects close/delete
      if(pos_after >= pos_before && ord_after >= ord_before)
         break;
   }

   Print("FXAI warning: CloseAll incomplete. Remaining managed positions=",
         FXAI_ManagedPositionsTotal(target_symbol),
         ", managed orders=", FXAI_ManagedOrdersTotal(target_symbol));
   return (FXAI_ManagedPositionsTotal(target_symbol) == 0 &&
           FXAI_ManagedOrdersTotal(target_symbol) == 0);
}

//---------------------- TRADE POSSIBLE ------------------------------
int TradePossible(const string symbol, string &reason)
{
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   reason = "ok";

   if(g_system_health_last_ready &&
      g_system_health_last_state.posture == "DEGRADED" &&
      g_system_health_last_state.health_score < 0.35)
   {
      reason = "system_health_degraded";
      return 0;
   }

   if(!SymbolSelect(symbol, true))
   {
      reason = "symbol_select_failed";
      return 0;
   }

   long tradeMode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
   if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
   {
      reason = "trade_mode_disabled";
      return 0;
   }

   if(!in_tester && !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      reason = "terminal_trade_not_allowed";
      return 0;
   }
   if(!in_tester && !TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      reason = "terminal_not_connected";
      return 0;
   }

   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   if(bid <= 0 || ask <= 0)
   {
      reason = "invalid_bid_ask";
      return 0;
   }

   MqlTick last_tick;
   if(!FXAI_MarketDataGetLatestTick(symbol, last_tick))
   {
      reason = "symbol_tick_failed";
      return 0;
   }

   datetime lastTickTime = last_tick.time;
   datetime currentTime  = FXAI_ServerNow();
   if(currentTime <= 0)
      currentTime = TimeCurrent();
   if(!in_tester && currentTime - lastTickTime > 10)
   {
      reason = "stale_tick";
      return 0;
   }

   if(SessionFilterEnabled)
   {
      if(!FXAI_IsInLiquidSession(symbol,
                                currentTime,
                                SessionMinAfterOpenMinutes,
                                SessionMinBeforeCloseMinutes))
      {
         reason = "session_filter_block";
         return 0;
      }
   }

   return 1;
}
#endif // __FXAI_RUNTIME_TRADE_LIFECYCLE_MQH__
