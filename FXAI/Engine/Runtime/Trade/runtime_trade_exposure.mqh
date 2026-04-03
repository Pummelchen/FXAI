#ifndef __FXAI_RUNTIME_TRADE_EXPOSURE_MQH__
#define __FXAI_RUNTIME_TRADE_EXPOSURE_MQH__
bool FXAI_IsTradeRetcodeSuccess(const uint retcode)
{
   return (retcode == TRADE_RETCODE_DONE ||
           retcode == TRADE_RETCODE_PLACED ||
           retcode == TRADE_RETCODE_DONE_PARTIAL);
}

datetime g_lifecycle_last_action_bar = 0;
int      g_lifecycle_last_action_code = -1;

datetime FXAI_CurrentLifecycleBarTime(const string symbol)
{
   datetime bar_time = iTime(symbol, PERIOD_M1, 0);
   if(bar_time > 0)
      return bar_time;
   bar_time = TimeCurrent();
   if(bar_time > 0)
      return bar_time;
   return TimeTradeServer();
}

bool FXAI_LifecycleActionCooling(const string symbol,
                                 const int action_code)
{
   datetime bar_time = FXAI_CurrentLifecycleBarTime(symbol);
   if(bar_time <= 0)
      return false;
   return (g_lifecycle_last_action_bar == bar_time &&
           g_lifecycle_last_action_code == action_code);
}

void FXAI_RememberLifecycleAction(const string symbol,
                                  const int action_code)
{
   g_lifecycle_last_action_bar = FXAI_CurrentLifecycleBarTime(symbol);
   g_lifecycle_last_action_code = action_code;
}

double FXAI_NormalizeLot(const string symbol, const double requested_lot)
{
   double vmin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double vmax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(vstep <= 0.0) vstep = 0.01;
   if(vmin <= 0.0) vmin = vstep;
   if(vmax < vmin) vmax = vmin;

   double lot = requested_lot;
   if(!MathIsValidNumber(lot) || lot <= 0.0)
      lot = vmin;

   lot = FXAI_Clamp(lot, vmin, vmax);
   lot = vmin + MathFloor(((lot - vmin) / vstep) + 1e-9) * vstep;

   if(lot < vmin) lot = vmin;
   if(lot > vmax) lot = vmax;

   return NormalizeDouble(lot, 8);
}

void FXAI_ParseSymbolLegs(const string symbol,
                         string &base_out,
                         string &quote_out)
{
   base_out = "";
   quote_out = "";

   base_out = SymbolInfoString(symbol, SYMBOL_CURRENCY_BASE);
   quote_out = SymbolInfoString(symbol, SYMBOL_CURRENCY_PROFIT);
   StringToUpper(base_out);
   StringToUpper(quote_out);
   if(StringLen(base_out) == 3 && StringLen(quote_out) == 3)
      return;

   string clean = symbol;
   StringToUpper(clean);
   string letters = "";
   int n = StringLen(clean);
   for(int i=0; i<n; i++)
   {
      string ch = StringSubstr(clean, i, 1);
      ushort code = (ushort)StringGetCharacter(ch, 0);
      if(code >= 'A' && code <= 'Z')
         letters += ch;
      if(StringLen(letters) >= 6)
         break;
   }

   if(StringLen(letters) >= 6)
   {
      base_out = StringSubstr(letters, 0, 3);
      quote_out = StringSubstr(letters, 3, 3);
   }
}

bool FXAI_SymbolsShareCurrency(const string lhs,
                               const string rhs)
{
   string base_l = "";
   string quote_l = "";
   string base_r = "";
   string quote_r = "";
   FXAI_ParseSymbolLegs(lhs, base_l, quote_l);
   FXAI_ParseSymbolLegs(rhs, base_r, quote_r);

   if(StringLen(base_l) != 3 || StringLen(quote_l) != 3 ||
      StringLen(base_r) != 3 || StringLen(quote_r) != 3)
      return false;

   return (base_l == base_r || base_l == quote_r ||
           quote_l == base_r || quote_l == quote_r);
}

double FXAI_CorrelationExposureWeight(const string anchor_symbol,
                                      const string other_symbol)
{
   if(anchor_symbol == other_symbol)
      return 1.0;

   string base_a = "";
   string quote_a = "";
   string base_b = "";
   string quote_b = "";
   FXAI_ParseSymbolLegs(anchor_symbol, base_a, quote_a);
   FXAI_ParseSymbolLegs(other_symbol, base_b, quote_b);
   if(StringLen(base_a) != 3 || StringLen(quote_a) != 3 ||
      StringLen(base_b) != 3 || StringLen(quote_b) != 3)
      return 0.0;

   if(base_a == quote_b && quote_a == base_b)
      return 1.0;
   if(base_a == base_b && quote_a == quote_b)
      return 1.0;
   if(base_a == base_b || quote_a == quote_b)
      return 0.85;
   if(base_a == quote_b || quote_a == base_b)
      return 0.70;
   if(FXAI_SymbolsShareCurrency(anchor_symbol, other_symbol))
      return 0.55;
   return 0.0;
}

double FXAI_ManagedPositionLots(const string symbol = "")
{
   double lots = 0.0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && PositionGetString(POSITION_SYMBOL) != symbol) continue;
      lots += MathMax(PositionGetDouble(POSITION_VOLUME), 0.0);
   }
   return lots;
}

double FXAI_ManagedOrderLots(const string symbol = "")
{
   double lots = 0.0;
   for(int i=OrdersTotal() - 1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      if(StringLen(symbol) > 0 && OrderGetString(ORDER_SYMBOL) != symbol) continue;
      lots += MathMax(OrderGetDouble(ORDER_VOLUME_CURRENT), 0.0);
   }
   return lots;
}

double FXAI_ManagedExposureLots(const string symbol = "")
{
   return FXAI_ManagedPositionLots(symbol) + FXAI_ManagedOrderLots(symbol);
}

double FXAI_ManagedCorrelatedExposureLots(const string symbol = "")
{
   string anchor = (StringLen(symbol) > 0 ? symbol : _Symbol);
   double lots = 0.0;

   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      string pos_symbol = PositionGetString(POSITION_SYMBOL);
      double weight = FXAI_CorrelationExposureWeight(anchor, pos_symbol);
      if(weight <= 0.0) continue;
      lots += weight * MathMax(PositionGetDouble(POSITION_VOLUME), 0.0);
   }

   for(int i=OrdersTotal() - 1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      string order_symbol = OrderGetString(ORDER_SYMBOL);
      double weight = FXAI_CorrelationExposureWeight(anchor, order_symbol);
      if(weight <= 0.0) continue;
      lots += weight * MathMax(OrderGetDouble(ORDER_VOLUME_CURRENT), 0.0);
   }

   return lots;
}

string FXAI_RuntimeBaseCurrency(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   if(StringLen(symbol) < 6)
      return "";
   return StringSubstr(symbol, 0, 3);
}

string FXAI_RuntimeQuoteCurrency(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   if(StringLen(symbol) < 6)
      return "";
   return StringSubstr(symbol, 3, 3);
}

int FXAI_DirectionalExposureSign(const string symbol,
                                 const int direction,
                                 const string currency)
{
   string base = FXAI_RuntimeBaseCurrency(symbol);
   string quote = FXAI_RuntimeQuoteCurrency(symbol);
   if(direction != 0 && direction != 1)
      return 0;
   int dir_sign = (direction == 1 ? 1 : -1);
   if(currency == base)
      return dir_sign;
   if(currency == quote)
      return -dir_sign;
   return 0;
}

double FXAI_DirectionalClusterAlignment(const string anchor_symbol,
                                        const int anchor_direction,
                                        const string other_symbol,
                                        const int other_direction)
{
   string anchor_base = FXAI_RuntimeBaseCurrency(anchor_symbol);
   string anchor_quote = FXAI_RuntimeQuoteCurrency(anchor_symbol);
   string other_base = FXAI_RuntimeBaseCurrency(other_symbol);
   string other_quote = FXAI_RuntimeQuoteCurrency(other_symbol);
   if(StringLen(anchor_base) != 3 || StringLen(anchor_quote) != 3 ||
      StringLen(other_base) != 3 || StringLen(other_quote) != 3)
      return 0.0;

   string currencies[2];
   currencies[0] = anchor_base;
   currencies[1] = anchor_quote;
   double align = 0.0;
   for(int i=0; i<2; i++)
   {
      string cur = currencies[i];
      int anchor_sign = FXAI_DirectionalExposureSign(anchor_symbol, anchor_direction, cur);
      int other_sign = FXAI_DirectionalExposureSign(other_symbol, other_direction, cur);
      if(anchor_sign == 0 || other_sign == 0)
         continue;
      if(anchor_sign == other_sign)
         align += 0.50;
      else
         align -= 0.25;
   }
   if(anchor_symbol == other_symbol && anchor_direction == other_direction)
      align = 1.0;
   else if(anchor_base == other_quote && anchor_quote == other_base && anchor_direction != other_direction)
      align = MathMax(align, 0.95);
   return FXAI_Clamp(align, 0.0, 1.0);
}

double FXAI_ManagedDirectionalClusterLots(const string symbol,
                                          const int direction)
{
   if(direction != 0 && direction != 1)
      return 0.0;

   double lots = 0.0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      string pos_symbol = PositionGetString(POSITION_SYMBOL);
      int pos_direction = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : 0);
      double weight = FXAI_DirectionalClusterAlignment(symbol, direction, pos_symbol, pos_direction);
      if(weight <= 0.0) continue;
      lots += weight * MathMax(PositionGetDouble(POSITION_VOLUME), 0.0);
   }

   for(int i=OrdersTotal() - 1; i>=0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;
      if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
      string order_symbol = OrderGetString(ORDER_SYMBOL);
      ENUM_ORDER_TYPE order_type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      int order_direction = -1;
      if(order_type == ORDER_TYPE_BUY || order_type == ORDER_TYPE_BUY_LIMIT || order_type == ORDER_TYPE_BUY_STOP || order_type == ORDER_TYPE_BUY_STOP_LIMIT)
         order_direction = 1;
      else if(order_type == ORDER_TYPE_SELL || order_type == ORDER_TYPE_SELL_LIMIT || order_type == ORDER_TYPE_SELL_STOP || order_type == ORDER_TYPE_SELL_STOP_LIMIT)
         order_direction = 0;
      if(order_direction < 0) continue;
      double weight = FXAI_DirectionalClusterAlignment(symbol, direction, order_symbol, order_direction);
      if(weight <= 0.0) continue;
      lots += weight * MathMax(OrderGetDouble(ORDER_VOLUME_CURRENT), 0.0);
   }

   return lots;
}

struct FXAIManagedPositionState
{
   bool has_position;
   int direction;
   double total_volume;
   double avg_price;
   double open_profit;
   double profit_points;
   double cycle_realized_profit;
   double cycle_peak_profit;
   double giveback_fraction;
   int position_count;
   int pending_count;
   datetime oldest_time;
};

void FXAI_ResetManagedPositionState(FXAIManagedPositionState &out)
{
   out.has_position = false;
   out.direction = -1;
   out.total_volume = 0.0;
   out.avg_price = 0.0;
   out.open_profit = 0.0;
   out.profit_points = 0.0;
   out.cycle_realized_profit = 0.0;
   out.cycle_peak_profit = 0.0;
   out.giveback_fraction = 0.0;
   out.position_count = 0;
   out.pending_count = 0;
   out.oldest_time = 0;
}

bool FXAI_ReadManagedPositionState(const string symbol,
                                   FXAIManagedPositionState &out)
{
   FXAI_ResetManagedPositionState(out);
   double signed_volume = 0.0;
   double weighted_price = 0.0;
   for(int i=PositionsTotal() - 1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;

      double volume = MathMax(PositionGetDouble(POSITION_VOLUME), 0.0);
      if(volume <= 0.0)
         continue;
      int dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : 0);
      double sign = (dir == 1 ? 1.0 : -1.0);
      signed_volume += sign * volume;
      weighted_price += volume * PositionGetDouble(POSITION_PRICE_OPEN);
      out.total_volume += volume;
      out.open_profit += PositionGetDouble(POSITION_PROFIT) +
                         PositionGetDouble(POSITION_SWAP);
      out.position_count++;
      datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(out.oldest_time <= 0 || (pos_time > 0 && pos_time < out.oldest_time))
         out.oldest_time = pos_time;
   }

   out.pending_count = FXAI_ManagedOrdersTotal(symbol);
   if(out.total_volume <= 0.0)
      return false;

   out.has_position = true;
   out.direction = (signed_volume >= 0.0 ? 1 : 0);
   out.avg_price = weighted_price / MathMax(out.total_volume, 1e-9);
   out.open_profit = FXAI_ManagedOpenProfit(symbol);
   out.cycle_realized_profit = (RealizedManagedProfit - CycleEntryRealizedProfit);
   out.cycle_peak_profit = MathMax(TrailPeakProfit, 0.0);
   double cycle_total = out.open_profit + out.cycle_realized_profit;
   if(out.cycle_peak_profit > 1e-9)
      out.giveback_fraction = FXAI_Clamp((out.cycle_peak_profit - MathMax(cycle_total, 0.0)) /
                                         out.cycle_peak_profit,
                                         0.0,
                                         1.0);

   MqlTick tick;
   if(SymbolInfoTick(symbol, tick))
   {
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(point <= 0.0)
         point = (_Point > 0.0 ? _Point : 1.0);
      double current_price = (out.direction == 1 ? tick.bid : tick.ask);
      if(current_price > 0.0 && point > 0.0)
      {
         if(out.direction == 1)
            out.profit_points = (current_price - out.avg_price) / point;
         else
            out.profit_points = (out.avg_price - current_price) / point;
      }
   }
   return true;
}

int FXAI_ManagedPositionHeldBars(const FXAIManagedPositionState &state)
{
   if(!state.has_position || state.oldest_time <= 0)
      return 0;
   datetime now_t = TimeCurrent();
   if(now_t <= 0)
      now_t = TimeTradeServer();
   if(now_t <= 0 || now_t <= state.oldest_time)
      return 0;
   int bars = (int)((now_t - state.oldest_time) / 60);
   if(bars < 0)
      bars = 0;
   return bars;
}

double FXAI_PortfolioPressureScore(const string symbol,
                                   const int direction)
{
   FXAILiveDeploymentProfile deploy_profile;
   FXAI_LoadLiveDeploymentProfile(symbol, deploy_profile, false);
   double portfolio_cap = MathMax(MaxPortfolioExposureLots, 0.0);
   double corr_cap = MathMax(MaxCorrelatedExposureLots, 0.0);
   double dir_cap = MathMax(MaxDirectionalClusterLots, 0.0);
   double gross_ratio = (portfolio_cap > 1e-9 ? FXAI_ManagedExposureLots("") / portfolio_cap : 0.0);
   double corr_ratio = (corr_cap > 1e-9 ? FXAI_ManagedCorrelatedExposureLots(symbol) / corr_cap : 0.0);
   double dir_ratio = (dir_cap > 1e-9 ? FXAI_ManagedDirectionalClusterLots(symbol, direction) / dir_cap : 0.0);
   double hierarchy_penalty = 1.0 - FXAI_Clamp(g_ai_last_hierarchy_score, 0.0, 1.0);
   double macro_penalty = (FXAI_MacroEventLeakageSafe() ? (1.0 - FXAI_Clamp(g_ai_last_macro_state_quality, 0.0, 1.0)) : 0.0);
   FXAIControlPlaneAggregate cp;
   FXAI_ReadControlPlaneAggregate(symbol, direction, cp);
   FXAIPortfolioSupervisorProfile supervisor;
   double supervisor_score = FXAI_PortfolioSupervisorScore(symbol, direction, cp, supervisor);
   FXAISupervisorServiceState service_state;
   FXAI_LoadSupervisorServiceState(symbol, service_state, false);
   FXAISupervisorCommandState command_state;
   FXAI_LoadSupervisorCommandState(symbol, command_state, false);
   double service_score = FXAI_SupervisorServiceScore(direction, service_state);
   double supervisor_blend = FXAI_Clamp(0.55 * FXAI_Clamp(supervisor.supervisor_weight, 0.0, 1.0) +
                                        0.45 * FXAI_Clamp(deploy_profile.supervisor_blend, 0.0, 1.0),
                                        0.0,
                                        1.0);
   g_control_plane_last_score = FXAI_Clamp((1.0 - supervisor_blend) * cp.score +
                                           0.55 * supervisor_blend * supervisor_score +
                                           0.45 * supervisor_blend * service_score,
                                           0.0,
                                           3.0);
   return FXAI_Clamp(0.30 * FXAI_Clamp(gross_ratio / MathMax(supervisor.gross_budget_bias, 0.40), 0.0, 2.0) +
                     0.28 * FXAI_Clamp(corr_ratio, 0.0, 2.0) +
                     0.22 * FXAI_Clamp(dir_ratio, 0.0, 2.0) +
                     0.12 * FXAI_Clamp(cp.score, 0.0, 1.5) +
                     0.08 * FXAI_Clamp(supervisor_score, 0.0, 1.5) +
                     0.08 * FXAI_Clamp(service_score, 0.0, 1.5) +
                     0.06 * FXAI_Clamp(cp.macro_overlap, 0.0, 1.0) +
                     0.06 * FXAI_Clamp(service_state.macro_pressure, 0.0, 1.0) +
                     0.10 * hierarchy_penalty +
                     0.06 * macro_penalty,
                     0.0,
                     1.5);
}

double FXAI_MoneyPerPointPerLot(const string symbol)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
   if(tick_value <= 0.0)
      tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   if(point <= 0.0 || tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;
   return tick_value * (point / tick_size);
}

double FXAI_EstimatedRiskPointsForDecision()
{
   double base_cost = MathMax(g_ai_last_min_move_points, 0.25);
   double expected_move = MathMax(g_ai_last_expected_move_points, base_cost);
   double configured_target = MathMax(RiskTargetMovePoints, 0.0);
   double risk_points = base_cost +
                        expected_move * (0.45 + 0.65 * FXAI_Clamp(g_ai_last_path_risk, 0.0, 1.0)) +
                        base_cost * (0.25 + 0.50 * FXAI_Clamp(g_ai_last_fill_risk, 0.0, 1.0));
   if(configured_target > 0.0 && configured_target > risk_points)
      risk_points = configured_target;
   if(!MathIsValidNumber(risk_points) || risk_points <= 0.0)
      risk_points = MathMax(configured_target, base_cost);
   return MathMax(risk_points, 0.25);
}

#endif // __FXAI_RUNTIME_TRADE_EXPOSURE_MQH__
