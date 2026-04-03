#ifndef __FXAI_RUNTIME_TRADE_EXECUTION_MQH__
#define __FXAI_RUNTIME_TRADE_EXECUTION_MQH__
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
   reason = "ok";
   retcode = 0;
   ret_desc = "";

   if(direction != 0 && direction != 1)
   {
      reason = "invalid_direction";
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

   request.action = TRADE_ACTION_DEAL;
   request.magic = TradeMagic;
   request.symbol = symbol;
   request.volume = trade_lot;
   request.type = (direction == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   request.type_filling = FXAI_ResolveOrderFilling(symbol);
   request.price = (direction == 1 ? tick.ask : tick.bid);
   request.deviation = (ulong)MathRound(FXAI_ExecutionAllowedDeviationPoints(exec_profile,
                                                                             g_ai_last_path_risk,
                                                                             g_ai_last_fill_risk));
   request.comment = StringFormat("FXAI|%s|H%d|R%d",
                                  (direction == 1 ? "BUY" : "SELL"),
                                  g_ai_last_horizon_minutes,
                                  g_ai_last_regime_id);

   if(request.price <= 0.0)
   {
      reason = "invalid_order_price";
      return false;
   }

   ulong order_start_us = GetMicrosecondCount();
   datetime order_sample_time = TimeCurrent();
   if(order_sample_time <= 0)
      order_sample_time = tick.time;
   if(order_sample_time <= 0)
      order_sample_time = TimeTradeServer();

   if(!OrderCheck(request, check))
   {
      retcode = (uint)check.retcode;
      ret_desc = check.comment;
      double elapsed_ms = (double)(GetMicrosecondCount() - order_start_us) / 1000.0;
      double latency_points = 0.05 * MathLog(1.0 + MathMax(elapsed_ms, 0.0));
      FXAI_RecordBrokerExecutionEventEx(order_sample_time,
                                        symbol,
                                        g_ai_last_horizon_minutes,
                                        (direction == 1 ? 1 : -1),
                                        FXAI_BrokerOrderTypeBucket(request.type),
                                        0,
                                        0.0,
                                        latency_points,
                                        true,
                                        false,
                                        0.0);
      FXAI_MarkRuntimeArtifactsDirty();
      reason = "order_check_failed";
      return false;
   }
   if(!FXAI_IsTradeRetcodeSuccess((uint)check.retcode))
   {
      retcode = (uint)check.retcode;
      ret_desc = check.comment;
      double elapsed_ms = (double)(GetMicrosecondCount() - order_start_us) / 1000.0;
      double latency_points = 0.05 * MathLog(1.0 + MathMax(elapsed_ms, 0.0));
      FXAI_RecordBrokerExecutionEventEx(order_sample_time,
                                        symbol,
                                        g_ai_last_horizon_minutes,
                                        (direction == 1 ? 1 : -1),
                                        FXAI_BrokerOrderTypeBucket(request.type),
                                        0,
                                        0.0,
                                        latency_points,
                                        true,
                                        false,
                                        0.0);
      FXAI_MarkRuntimeArtifactsDirty();
      reason = "order_check_rejected";
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
   g_last_order_request_pending = true;

   if(!OrderSend(request, result))
   {
      retcode = (uint)result.retcode;
      ret_desc = result.comment;
      double elapsed_ms = (double)(GetMicrosecondCount() - order_start_us) / 1000.0;
      double latency_points = 0.05 * MathLog(1.0 + MathMax(elapsed_ms, 0.0));
      FXAI_RecordBrokerExecutionEventEx(order_sample_time,
                                        symbol,
                                        g_ai_last_horizon_minutes,
                                        (direction == 1 ? 1 : -1),
                                        FXAI_BrokerOrderTypeBucket(request.type),
                                        0,
                                        0.0,
                                        latency_points,
                                        true,
                                        false,
                                        0.0);
      FXAI_MarkRuntimeArtifactsDirty();
      FXAI_ClearLastOrderRequestState();
      reason = "order_send_failed";
      return false;
   }

   retcode = (uint)result.retcode;
   ret_desc = result.comment;
   if(!FXAI_IsTradeRetcodeSuccess(retcode))
   {
      double elapsed_ms = (double)(GetMicrosecondCount() - order_start_us) / 1000.0;
      double latency_points = 0.05 * MathLog(1.0 + MathMax(elapsed_ms, 0.0));
      FXAI_RecordBrokerExecutionEventEx(order_sample_time,
                                        symbol,
                                        g_ai_last_horizon_minutes,
                                        (direction == 1 ? 1 : -1),
                                        FXAI_BrokerOrderTypeBucket(request.type),
                                        0,
                                        0.0,
                                        latency_points,
                                        true,
                                        false,
                                        0.0);
      FXAI_MarkRuntimeArtifactsDirty();
      FXAI_ClearLastOrderRequestState();
      reason = "order_send_rejected";
      return false;
   }

   return true;
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
         ok = trade.PositionClosePartial(ticket, remaining);
         if(ok)
            closed += remaining;
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
   bool exec_ok = FXAI_SendMarketOrderChecked(symbol, direction, trade_lot, reason, retcode, ret_desc);
   if(!exec_ok)
      return false;

   if(!CycleActive)
   {
      ResetCycleState();
      CycleActive = true;
      CycleEntryEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      CycleEntryManagedPnl = FXAI_ManagedOpenProfit(symbol);
      CycleEntryRealizedProfit = RealizedManagedProfit;
      CycleStartTime = TimeCurrent();
      if(CycleStartTime <= 0)
         CycleStartTime = TimeTradeServer();
      if(CycleStartTime <= 0)
         CycleStartTime = iTime(symbol, PERIOD_M1, 0);
   }
   return true;
}

#endif // __FXAI_RUNTIME_TRADE_EXECUTION_MQH__
