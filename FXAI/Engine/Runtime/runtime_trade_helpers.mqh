bool FXAI_IsTradeRetcodeSuccess(const uint retcode)
{
   return (retcode == TRADE_RETCODE_DONE ||
           retcode == TRADE_RETCODE_PLACED ||
           retcode == TRADE_RETCODE_DONE_PARTIAL);
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
   double supervisor_blend = FXAI_Clamp(0.55 * FXAI_Clamp(supervisor.supervisor_weight, 0.0, 1.0) +
                                        0.45 * FXAI_Clamp(deploy_profile.supervisor_blend, 0.0, 1.0),
                                        0.0,
                                        1.0);
   g_control_plane_last_score = FXAI_Clamp((1.0 - supervisor_blend) * cp.score +
                                           supervisor_blend * supervisor_score,
                                           0.0,
                                           2.0);
   return FXAI_Clamp(0.30 * FXAI_Clamp(gross_ratio / MathMax(supervisor.gross_budget_bias, 0.40), 0.0, 2.0) +
                     0.28 * FXAI_Clamp(corr_ratio, 0.0, 2.0) +
                     0.22 * FXAI_Clamp(dir_ratio, 0.0, 2.0) +
                     0.12 * FXAI_Clamp(cp.score, 0.0, 1.5) +
                     0.10 * FXAI_Clamp(supervisor_score, 0.0, 1.5) +
                     0.06 * FXAI_Clamp(cp.macro_overlap, 0.0, 1.0) +
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
   double control_plane_pressure = g_control_plane_last_score;
   if(direction == 1)
      control_plane_pressure = MathMax(control_plane_pressure, g_control_plane_last_buy_score);
   else if(direction == 0)
      control_plane_pressure = MathMax(control_plane_pressure, g_control_plane_last_sell_score);
   requested_lot *= FXAI_Clamp((1.08 - 0.60 * portfolio_pressure) *
                               (1.02 - 0.25 * FXAI_Clamp(control_plane_pressure, 0.0, 1.5)) *
                               FXAI_Clamp(1.04 - 0.22 * supervisor_score, 0.25, 1.10) *
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

//------------------------- CLOSE ALL --------------------------------
bool CloseAll()
{
   const int max_passes = 25;
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   int op_pause_ms = (in_tester ? 0 : 100);

   for(int pass=0; pass<max_passes; pass++)
   {
      int pos_before = FXAI_ManagedPositionsTotal(_Symbol);
      int ord_before = FXAI_ManagedOrdersTotal(_Symbol);

      for(int i=PositionsTotal() - 1; i>=0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if((ulong)PositionGetInteger(POSITION_MAGIC) != TradeMagic) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
         trade.PositionClose(ticket);
         if(op_pause_ms > 0) Sleep(op_pause_ms);
      }

      for(int i=OrdersTotal() - 1; i>=0; i--)
      {
         ulong ticket = OrderGetTicket(i);
         if(ticket == 0) continue;
         if(!OrderSelect(ticket)) continue;
         if((ulong)OrderGetInteger(ORDER_MAGIC) != TradeMagic) continue;
         if(OrderGetString(ORDER_SYMBOL) != _Symbol) continue;

         long orderType = OrderGetInteger(ORDER_TYPE);
         if(orderType == ORDER_TYPE_BUY_LIMIT      || orderType == ORDER_TYPE_SELL_LIMIT ||
            orderType == ORDER_TYPE_BUY_STOP       || orderType == ORDER_TYPE_SELL_STOP  ||
            orderType == ORDER_TYPE_BUY_STOP_LIMIT || orderType == ORDER_TYPE_SELL_STOP_LIMIT)
         {
            trade.OrderDelete(ticket);
            if(op_pause_ms > 0) Sleep(op_pause_ms);
         }
      }

      int pos_after = FXAI_ManagedPositionsTotal(_Symbol);
      int ord_after = FXAI_ManagedOrdersTotal(_Symbol);
      if(pos_after == 0 && ord_after == 0)
         return true;

      // no progress; avoid hard lock in OnTick if broker rejects close/delete
      if(pos_after >= pos_before && ord_after >= ord_before)
         break;
   }

   Print("FXAI warning: CloseAll incomplete. Remaining managed positions=",
         FXAI_ManagedPositionsTotal(_Symbol),
         ", managed orders=", FXAI_ManagedOrdersTotal(_Symbol));
   return (FXAI_ManagedPositionsTotal(_Symbol) == 0 &&
           FXAI_ManagedOrdersTotal(_Symbol) == 0);
}

//---------------------- TRADE POSSIBLE ------------------------------
int TradePossible(const string symbol, string &reason)
{
   bool in_tester = (MQLInfoInteger(MQL_TESTER) != 0);
   reason = "ok";

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
   if(!SymbolInfoTick(symbol, last_tick))
   {
      reason = "symbol_tick_failed";
      return 0;
   }

   datetime lastTickTime = last_tick.time;
   datetime currentTime  = TimeCurrent();
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
