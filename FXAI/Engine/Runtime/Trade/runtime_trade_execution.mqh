#ifndef __FXAI_RUNTIME_TRADE_EXECUTION_MQH__
#define __FXAI_RUNTIME_TRADE_EXECUTION_MQH__

struct FXAIExecutionPlan
{
   bool            ready;
   bool            use_pending;
   ENUM_ORDER_TYPE order_type;
   double          entry_price;
   double          stop_loss;
   double          take_profit;
   datetime        expiry_time;
   string          mode;
};

void FXAI_ResetExecutionPlan(FXAIExecutionPlan &out)
{
   out.ready = false;
   out.use_pending = false;
   out.order_type = ORDER_TYPE_BUY;
   out.entry_price = 0.0;
   out.stop_loss = 0.0;
   out.take_profit = 0.0;
   out.expiry_time = 0;
   out.mode = "MARKET";
}

double FXAI_SymbolPointValue(const string symbol)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = (_Point > 0.0 ? _Point : 0.0001);
   return point;
}

double FXAI_NormalizeOrderPrice(const string symbol,
                                const double price)
{
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   if(digits < 0)
      digits = _Digits;
   return NormalizeDouble(price, digits);
}

double FXAI_TradeDistanceMinPoints(const string symbol)
{
   long stops_level = 0;
   long freeze_level = 0;
   SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL, stops_level);
   SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL, freeze_level);
   return (double)MathMax((long)2, MathMax(stops_level, freeze_level));
}

void FXAI_ApplyProtectiveLevels(const string symbol,
                                const int direction,
                                const double entry_price,
                                double &stop_loss,
                                double &take_profit)
{
   double point = FXAI_SymbolPointValue(symbol);
   double min_points = FXAI_TradeDistanceMinPoints(symbol);
   double risk_points = MathMax(FXAI_EstimatedRiskPointsForDecision(), min_points + 2.0);
   double reward_points = MathMax(1.35 * MathMax(g_ai_last_expected_move_points, risk_points), risk_points * 1.15);
   if(direction == 1)
   {
      stop_loss = entry_price - risk_points * point;
      take_profit = entry_price + reward_points * point;
   }
   else
   {
      stop_loss = entry_price + risk_points * point;
      take_profit = entry_price - reward_points * point;
   }
   stop_loss = FXAI_NormalizeOrderPrice(symbol, stop_loss);
   take_profit = FXAI_NormalizeOrderPrice(symbol, take_profit);
}

bool FXAI_BuildExecutionPlan(const string symbol,
                             const int direction,
                             FXAIExecutionPlan &plan)
{
   FXAI_ResetExecutionPlan(plan);
   if(direction != 0 && direction != 1)
      return false;

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
      return false;

   double point = FXAI_SymbolPointValue(symbol);
   double min_points = FXAI_TradeDistanceMinPoints(symbol);
   double setup_points = MathMax(MathMax(g_ai_last_min_move_points, 4.0), min_points + 1.0);
   double entry_offset_points = MathMax(0.30 * FXAI_EstimatedRiskPointsForDecision(),
                                        MathMax(min_points + 1.0, 0.35 * setup_points));
   bool prefer_pending = (g_system_health_last_ready &&
                          g_system_health_last_state.posture != "HEALTHY") ||
                         (g_execution_quality_last_state == "CAUTION") ||
                         (g_execution_quality_last_state == "STRESSED") ||
                         (g_newspulse_last_trade_gate == "CAUTION") ||
                         g_last_order_request_pending;

   double entry_price = 0.0;
   if(prefer_pending)
   {
      bool use_breakout = (g_ai_last_trade_gate >= 0.60 &&
                           g_ai_last_hierarchy_execution >= 0.55 &&
                           g_ai_last_path_risk <= 0.55);
      if(direction == 1)
      {
         if(use_breakout)
         {
            plan.order_type = ORDER_TYPE_BUY_STOP;
            entry_price = tick.ask + entry_offset_points * point;
            plan.mode = "BUY_STOP";
         }
         else
         {
            plan.order_type = ORDER_TYPE_BUY_LIMIT;
            entry_price = tick.bid - entry_offset_points * point;
            plan.mode = "BUY_LIMIT";
         }
      }
      else
      {
         if(use_breakout)
         {
            plan.order_type = ORDER_TYPE_SELL_STOP;
            entry_price = tick.bid - entry_offset_points * point;
            plan.mode = "SELL_STOP";
         }
         else
         {
            plan.order_type = ORDER_TYPE_SELL_LIMIT;
            entry_price = tick.ask + entry_offset_points * point;
            plan.mode = "SELL_LIMIT";
         }
      }

      plan.use_pending = true;
      plan.entry_price = FXAI_NormalizeOrderPrice(symbol, entry_price);
      plan.expiry_time = FXAI_ServerNow();
      if(plan.expiry_time <= 0)
         plan.expiry_time = TimeCurrent();
      if(plan.expiry_time <= 0)
         plan.expiry_time = tick.time;
      plan.expiry_time += 20 * 60;
   }
   else
   {
      plan.use_pending = false;
      plan.order_type = (direction == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
      plan.entry_price = (direction == 1 ? tick.ask : tick.bid);
      plan.mode = "MARKET";
   }

   if(plan.entry_price <= 0.0)
      return false;

   FXAI_ApplyProtectiveLevels(symbol, direction, plan.entry_price, plan.stop_loss, plan.take_profit);

   if(plan.use_pending)
   {
      double min_distance = min_points * point;
      if(plan.order_type == ORDER_TYPE_BUY_LIMIT && plan.entry_price >= tick.ask - min_distance)
         plan.entry_price = FXAI_NormalizeOrderPrice(symbol, tick.ask - (min_points + 1.0) * point);
      if(plan.order_type == ORDER_TYPE_SELL_LIMIT && plan.entry_price <= tick.bid + min_distance)
         plan.entry_price = FXAI_NormalizeOrderPrice(symbol, tick.bid + (min_points + 1.0) * point);
      if(plan.order_type == ORDER_TYPE_BUY_STOP && plan.entry_price <= tick.ask + min_distance)
         plan.entry_price = FXAI_NormalizeOrderPrice(symbol, tick.ask + (min_points + 1.0) * point);
      if(plan.order_type == ORDER_TYPE_SELL_STOP && plan.entry_price >= tick.bid - min_distance)
         plan.entry_price = FXAI_NormalizeOrderPrice(symbol, tick.bid - (min_points + 1.0) * point);
      FXAI_ApplyProtectiveLevels(symbol,
                                 direction,
                                 plan.entry_price,
                                 plan.stop_loss,
                                 plan.take_profit);
   }

   plan.ready = true;
   return true;
}

bool FXAI_SendExecutionPlanChecked(const string symbol,
                                   const int direction,
                                   const double trade_lot,
                                   const FXAIExecutionPlan &plan,
                                   string &reason,
                                   uint &retcode,
                                   string &ret_desc)
{
   reason = "ok";
   retcode = 0;
   ret_desc = "";
   if(!plan.ready)
   {
      reason = "execution_plan_invalid";
      return false;
   }

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
   {
      reason = "symbol_tick_failed";
      return false;
   }

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);

   MqlTradeRequest request;
   MqlTradeCheckResult check;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(check);
   ZeroMemory(result);

   request.action = (plan.use_pending ? TRADE_ACTION_PENDING : TRADE_ACTION_DEAL);
   request.magic = TradeMagic;
   request.symbol = symbol;
   request.volume = trade_lot;
   request.type = plan.order_type;
   request.type_filling = FXAI_ResolveOrderFilling(symbol);
   request.price = plan.entry_price;
   request.sl = plan.stop_loss;
   request.tp = plan.take_profit;
   if(plan.use_pending)
   {
      request.type_time = ORDER_TIME_SPECIFIED;
      request.expiration = plan.expiry_time;
   }

   double allowed_deviation = FXAI_ExecutionAllowedDeviationPoints(exec_profile,
                                                                   g_ai_last_path_risk,
                                                                   g_ai_last_fill_risk);
   if(ExecutionQualityEnabled &&
      g_execution_quality_last_ready &&
      !g_execution_quality_last_data_stale &&
      g_execution_quality_last_allowed_deviation > 0.0)
   {
      allowed_deviation = MathMax(allowed_deviation, g_execution_quality_last_allowed_deviation);
   }
   request.deviation = (ulong)MathRound(MathMax(allowed_deviation, 0.0));
   request.comment = StringFormat("FXAI|%s|%s|H%d|R%d",
                                  (direction == 1 ? "BUY" : "SELL"),
                                  plan.mode,
                                  g_ai_last_horizon_minutes,
                                  g_ai_last_regime_id);

   if(request.price <= 0.0)
   {
      reason = "invalid_order_price";
      return false;
   }

   ulong order_start_us = GetMicrosecondCount();
   datetime order_sample_time = FXAI_ServerNow();
   if(order_sample_time <= 0)
      order_sample_time = tick.time;

   if(!OrderCheck(request, check))
   {
      retcode = (uint)check.retcode;
      ret_desc = check.comment;
      reason = "order_check_failed";
      FXAI_ClearLastOrderRequestState();
      return false;
   }
   if(!FXAI_IsTradeRetcodeSuccess((uint)check.retcode))
   {
      retcode = (uint)check.retcode;
      ret_desc = check.comment;
      reason = "order_check_rejected";
      FXAI_ClearLastOrderRequestState();
      return false;
   }

   g_last_order_request_time = order_sample_time;
   g_last_order_request_us = order_start_us;
   g_last_order_request_price = request.price;
   g_last_order_request_volume = request.volume;
   g_last_order_request_filled_volume = 0.0;
   g_last_order_request_horizon = g_ai_last_horizon_minutes;
   g_last_order_request_side = (direction == 1 ? 1 : -1);
   g_last_order_request_type = FXAI_BrokerOrderTypeBucket(request.type);
   g_last_order_request_symbol = symbol;
   g_last_order_request_order_ticket = 0;
   g_last_order_request_uses_pending_order = plan.use_pending;
   g_last_order_request_pending = true;

   if(!OrderSend(request, result))
   {
      retcode = (uint)result.retcode;
      ret_desc = result.comment;
      reason = "order_send_failed";
      FXAI_ClearLastOrderRequestState();
      return false;
   }

   retcode = (uint)result.retcode;
   ret_desc = result.comment;
   if(!FXAI_IsTradeRetcodeSuccess(retcode))
   {
      reason = "order_send_rejected";
      FXAI_ClearLastOrderRequestState();
      return false;
   }

   g_last_order_request_order_ticket = result.order;
   if(plan.use_pending)
      FXAI_MarkRuntimeArtifactsDirty();
   return true;
}

int FXAI_BrokerOrderTypeBucket(const ENUM_ORDER_TYPE order_type)
{
   switch(order_type)
   {
      case ORDER_TYPE_BUY:
      case ORDER_TYPE_SELL:
         return 1;
      case ORDER_TYPE_BUY_LIMIT:
      case ORDER_TYPE_SELL_LIMIT:
         return 2;
      case ORDER_TYPE_BUY_STOP:
      case ORDER_TYPE_SELL_STOP:
         return 3;
      case ORDER_TYPE_BUY_STOP_LIMIT:
      case ORDER_TYPE_SELL_STOP_LIMIT:
         return 4;
      default:
         return 0;
   }
}

bool FXAI_SendMarketOrderChecked(const string symbol,
                                 const int direction,
                                 const double trade_lot,
                                 string &reason,
                                 uint &retcode,
                                 string &ret_desc)
{
   FXAIExecutionPlan plan;
   FXAI_ResetExecutionPlan(plan);
   plan.ready = true;
   plan.use_pending = false;
   plan.order_type = (direction == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
   {
      reason = "symbol_tick_failed";
      retcode = 0;
      ret_desc = "";
      return false;
   }
   plan.entry_price = (direction == 1 ? tick.ask : tick.bid);
   plan.mode = "MARKET";
   FXAI_ApplyProtectiveLevels(symbol, direction, plan.entry_price, plan.stop_loss, plan.take_profit);
   return FXAI_SendExecutionPlanChecked(symbol, direction, trade_lot, plan, reason, retcode, ret_desc);
}

bool FXAI_SendTradeIntentChecked(const string symbol,
                                 const int direction,
                                 const double trade_lot,
                                 string &reason,
                                 uint &retcode,
                                 string &ret_desc)
{
   FXAIExecutionPlan plan;
   if(!FXAI_BuildExecutionPlan(symbol, direction, plan))
   {
      reason = "execution_plan_build_failed";
      retcode = 0;
      ret_desc = "";
      return false;
   }
   return FXAI_SendExecutionPlanChecked(symbol, direction, trade_lot, plan, reason, retcode, ret_desc);
}

bool FXAI_CancelManagedPendingOrders(const string symbol,
                                     const bool cancel_all = false)
{
   datetime now_time = FXAI_ServerNow();
   bool any = false;
   for(int i=OrdersTotal() - 1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && OrderGetString(ORDER_SYMBOL) != symbol) continue;

      ENUM_ORDER_TYPE order_type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      bool is_pending = (order_type == ORDER_TYPE_BUY_LIMIT ||
                         order_type == ORDER_TYPE_SELL_LIMIT ||
                         order_type == ORDER_TYPE_BUY_STOP ||
                         order_type == ORDER_TYPE_SELL_STOP ||
                         order_type == ORDER_TYPE_BUY_STOP_LIMIT ||
                         order_type == ORDER_TYPE_SELL_STOP_LIMIT);
      if(!is_pending)
         continue;

      bool delete_now = cancel_all;
      datetime expiry = (datetime)OrderGetInteger(ORDER_TIME_EXPIRATION);
      if(!delete_now && expiry > 0 && now_time > 0 && now_time >= expiry)
         delete_now = true;
      if(!delete_now &&
         g_system_health_last_ready &&
         g_system_health_last_state.posture == "DEGRADED" &&
         g_system_health_last_state.health_score < 0.35)
         delete_now = true;

      if(delete_now && trade.OrderDelete(ticket))
         any = true;
   }
   return any;
}

bool FXAI_CloseManagedFraction(const string symbol,
                               const double fraction)
{
   double frac = FXAI_Clamp(fraction, 0.0, 1.0);
   if(frac <= 0.0)
      return false;

   double total_volume = FXAI_ManagedPositionLots(symbol);
   if(total_volume <= 0.0)
      return false;

   bool hedging = (AccountInfoInteger(ACCOUNT_MARGIN_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   double target_close = total_volume * frac;
   double closed = 0.0;
   bool any = false;

   for(int i=PositionsTotal() - 1; i>=0 && closed + 1e-9 < target_close; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;

      double volume = MathMax(PositionGetDouble(POSITION_VOLUME), 0.0);
      if(volume <= 0.0)
         continue;

      double remaining = MathMax(target_close - closed, 0.0);
      bool ok = false;
      if(hedging && remaining + 1e-9 < volume)
      {
         double min_volume = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
         double max_volume = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
         double volume_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
         if(min_volume <= 0.0)
            min_volume = 0.01;
         if(max_volume <= 0.0)
            max_volume = volume;
         if(volume_step <= 0.0)
            volume_step = min_volume;

         double partial_volume = remaining;
         if(volume_step > 0.0)
            partial_volume = MathFloor((partial_volume + 1e-12) / volume_step) * volume_step;
         partial_volume = MathMax(partial_volume, 0.0);
         partial_volume = MathMin(partial_volume, MathMin(volume, max_volume));
         partial_volume = NormalizeDouble(partial_volume, 8);

         if(partial_volume + 1e-9 >= volume)
         {
            ok = trade.PositionClose(ticket);
            if(ok)
               closed += volume;
         }
         else if(partial_volume + 1e-9 >= min_volume)
         {
            ok = trade.PositionClosePartial(ticket, partial_volume);
            if(ok)
               closed += partial_volume;
         }
      }
      else
      {
         ok = trade.PositionClose(ticket);
         if(ok)
            closed += volume;
      }
      if(ok)
         any = true;
   }
   return any;
}

bool FXAI_TightenManagedStops(const string symbol,
                              const FXAIManagedPositionState &state,
                              const double tighten_strength)
{
   if(!state.has_position)
      return false;

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
      return false;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = (_Point > 0.0 ? _Point : 1.0);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   if(digits <= 0)
      digits = _Digits;
   double strength = FXAI_Clamp(tighten_strength, 0.0, 1.0);
   double tighten_points = MathMax(g_ai_last_min_move_points,
                                   FXAI_EstimatedRiskPointsForDecision() *
                                   FXAI_Clamp(0.65 - 0.35 * strength, 0.15, 0.65));
   bool any = false;

   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;

      long pos_type = PositionGetInteger(POSITION_TYPE);
      double existing_sl = PositionGetDouble(POSITION_SL);
      double existing_tp = PositionGetDouble(POSITION_TP);
      double new_sl = existing_sl;
      if(pos_type == POSITION_TYPE_BUY)
      {
         new_sl = tick.bid - tighten_points * point;
         double max_sl = tick.bid - 2.0 * point;
         if(new_sl > max_sl)
            new_sl = max_sl;
         if(existing_sl > 0.0)
            new_sl = MathMax(existing_sl, new_sl);
      }
      else
      {
         new_sl = tick.ask + tighten_points * point;
         double min_sl = tick.ask + 2.0 * point;
         if(new_sl < min_sl)
            new_sl = min_sl;
         if(existing_sl > 0.0)
            new_sl = MathMin(existing_sl, new_sl);
      }
      if(new_sl <= 0.0)
         continue;
      new_sl = NormalizeDouble(new_sl, digits);
      if(existing_sl > 0.0 && MathAbs(new_sl - existing_sl) < point)
         continue;
      if(trade.PositionModify(ticket, new_sl, existing_tp))
         any = true;
   }
   return any;
}

bool FXAI_SendLifecycleAddOrder(const string symbol,
                                const int direction,
                                const double trade_lot,
                                string &reason)
{
   uint retcode = 0;
   string ret_desc = "";
   bool exec_ok = FXAI_SendTradeIntentChecked(symbol, direction, trade_lot, reason, retcode, ret_desc);
   if(!exec_ok)
      return false;

   if(!CycleActive && FXAI_ManagedPositionsTotal(symbol) > 0)
   {
      ResetCycleState();
      CycleActive = true;
      CycleEntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      CycleEntryManagedPnl = FXAI_ManagedOpenProfit(symbol);
      CycleEntryRealizedProfit = RealizedManagedProfit;
      CycleStartTime = FXAI_GetOldestPositionTime(symbol);
      if(CycleStartTime <= 0)
         CycleStartTime = FXAI_ServerNow();
      if(CycleStartTime <= 0)
         CycleStartTime = TimeCurrent();
      if(CycleStartTime <= 0)
         CycleStartTime = iTime(symbol, PERIOD_M1, 0);
   }
   return true;
}

#endif // __FXAI_RUNTIME_TRADE_EXECUTION_MQH__
