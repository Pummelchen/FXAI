#ifndef __FXAI_RUNTIME_FACTOR_CONTEXT_MQH__
#define __FXAI_RUNTIME_FACTOR_CONTEXT_MQH__

#define FXAI_FACTOR_CONTEXT_MAX_CURRENCIES 16
#define FXAI_FACTOR_CONTEXT_MAX_PAIRS 32

struct FXAICurrencyFactorState
{
   string currency;
   bool   ready;
   double trend_score;
   double carry_score;
   double policy_score;
   double value_score;
   double commodity_score;
   double blended_score;
};

struct FXAIPairFactorContext
{
   bool     ready;
   bool     stale;
   string   symbol;
   double   trend_score;
   double   carry_score;
   double   policy_score;
   double   value_score;
   double   commodity_score;
   double   blended_score;
   int      bias_direction;
   double   alignment_score;
   datetime generated_at;
   string   rationale;
};

FXAIPairFactorContext g_factor_context_last_state;
bool                  g_factor_context_last_ready = false;
datetime              g_factor_context_last_generated_at = 0;
string                g_factor_context_last_symbol = "";
double                g_factor_context_last_blended_score = 0.0;
double                g_factor_context_last_alignment = 0.0;
int                   g_factor_context_last_bias_direction = -1;
string                g_factor_context_last_rationale = "";

void FXAI_ResetCurrencyFactorState(FXAICurrencyFactorState &out)
{
   out.currency = "";
   out.ready = false;
   out.trend_score = 0.0;
   out.carry_score = 0.0;
   out.policy_score = 0.0;
   out.value_score = 0.0;
   out.commodity_score = 0.0;
   out.blended_score = 0.0;
}

void FXAI_ResetPairFactorContext(FXAIPairFactorContext &out)
{
   out.ready = false;
   out.stale = true;
   out.symbol = "";
   out.trend_score = 0.0;
   out.carry_score = 0.0;
   out.policy_score = 0.0;
   out.value_score = 0.0;
   out.commodity_score = 0.0;
   out.blended_score = 0.0;
   out.bias_direction = -1;
   out.alignment_score = 0.0;
   out.generated_at = 0;
   out.rationale = "";
}

double FXAI_FactorD1Return(const string symbol,
                           const int shift_now,
                           const int shift_then)
{
   double now_close = iClose(symbol, PERIOD_D1, shift_now);
   double then_close = iClose(symbol, PERIOD_D1, shift_then);
   if(now_close <= 0.0 || then_close <= 0.0)
      return 0.0;
   return ((now_close / then_close) - 1.0);
}

double FXAI_FactorTrendScore(const string symbol)
{
   double r21 = FXAI_FactorD1Return(symbol, 1, 21);
   double r63 = FXAI_FactorD1Return(symbol, 1, 63);
   double r126 = FXAI_FactorD1Return(symbol, 1, 126);
   double blended = 0.20 * r21 + 0.35 * r63 + 0.45 * r126;
   return FXAI_Clamp(blended * 8.0, -1.0, 1.0);
}

double FXAI_FactorCarryDirectional(const string symbol,
                                   const int direction)
{
   double swap_long = SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG);
   double swap_short = SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT);
   double scale = MathMax(MathMax(MathAbs(swap_long), MathAbs(swap_short)), 0.50);
   if(direction == 1)
      return FXAI_Clamp(swap_long / scale, -1.0, 1.0);
   if(direction == 0)
      return FXAI_Clamp(swap_short / scale, -1.0, 1.0);
   return FXAI_Clamp((swap_long - swap_short) / scale, -1.0, 1.0);
}

double FXAI_FactorValueScore(const string symbol)
{
   double close_now = iClose(symbol, PERIOD_D1, 1);
   double close_63 = iClose(symbol, PERIOD_D1, 63);
   double close_252 = iClose(symbol, PERIOD_D1, 252);
   if(close_now <= 0.0 || close_63 <= 0.0 || close_252 <= 0.0)
      return 0.0;

   double medium_anchor = 0.5 * (close_63 + close_252);
   if(medium_anchor <= 0.0)
      return 0.0;
   double gap = (close_now - medium_anchor) / medium_anchor;
   return FXAI_Clamp(-gap * 6.0, -1.0, 1.0);
}

double FXAI_FactorPolicyPressureForCurrency(const string currency)
{
   FXAICalendarCachePairState calendar_state;
   FXAI_ResetCalendarCachePairState(calendar_state);
   string synthetic = currency + "USD";
   if(currency == "USD")
      synthetic = "EURUSD";
   if(!FXAI_ReadCalendarCachePairState(synthetic, calendar_state))
      return 0.0;
   if(!calendar_state.ready)
      return 0.0;

   double score = 0.0;
   bool has_central_bank = false;
   bool has_inflation = false;
   for(int i=0; i<calendar_state.reason_count; i++)
   {
      if(StringFind(calendar_state.reasons[i], "central_bank") >= 0)
         has_central_bank = true;
      if(StringFind(calendar_state.reasons[i], "inflation") >= 0)
         has_inflation = true;
   }
   if(StringFind(calendar_state.trade_gate, "BLOCK") >= 0)
      score += 0.20;
   if(has_central_bank)
      score += 0.35;
   if(has_inflation)
      score += 0.18;
   if(calendar_state.next_event_eta_min >= -120 && calendar_state.next_event_eta_min <= 240)
      score += 0.15;
   return FXAI_Clamp(score, -1.0, 1.0);
}

double FXAI_FactorCommodityScore(const string currency)
{
   string proxies[4];
   proxies[0] = "XAUUSD";
   proxies[1] = "XTIUSD";
   proxies[2] = "BRENT";
   proxies[3] = "XAGUSD";

   double commodity_move = 0.0;
   int used = 0;
   for(int i=0; i<4; i++)
   {
      if(!SymbolSelect(proxies[i], true))
         continue;
      double r20 = FXAI_FactorD1Return(proxies[i], 1, 20);
      double r60 = FXAI_FactorD1Return(proxies[i], 1, 60);
      commodity_move += 0.45 * r20 + 0.55 * r60;
      used++;
   }
   if(used <= 0)
      return 0.0;

   commodity_move /= (double)used;
   if(currency == "AUD" || currency == "NZD" || currency == "CAD" || currency == "NOK")
      return FXAI_Clamp(commodity_move * 6.0, -1.0, 1.0);
   if(currency == "CHF" || currency == "JPY")
      return FXAI_Clamp(-commodity_move * 4.0, -1.0, 1.0);
   return 0.0;
}

bool FXAI_BuildCurrencyFactorState(const string currency,
                                   FXAICurrencyFactorState &out)
{
   FXAI_ResetCurrencyFactorState(out);
   if(StringLen(currency) != 3)
      return false;

   out.currency = currency;

   string anchor_usd = currency + "USD";
   string anchor_rev = "USD" + currency;
   string pair_symbol = "";
   bool inverted = false;
   if(SymbolSelect(anchor_usd, true) && iClose(anchor_usd, PERIOD_D1, 1) > 0.0)
   {
      pair_symbol = anchor_usd;
      inverted = false;
   }
   else if(SymbolSelect(anchor_rev, true) && iClose(anchor_rev, PERIOD_D1, 1) > 0.0)
   {
      pair_symbol = anchor_rev;
      inverted = true;
   }
   else
   {
      return false;
   }

   out.trend_score = FXAI_FactorTrendScore(pair_symbol);
   out.value_score = FXAI_FactorValueScore(pair_symbol);
   out.policy_score = FXAI_FactorPolicyPressureForCurrency(currency);
   out.commodity_score = FXAI_FactorCommodityScore(currency);
   out.carry_score = FXAI_FactorCarryDirectional(pair_symbol, inverted ? 0 : 1);

   if(inverted)
   {
      out.trend_score *= -1.0;
      out.value_score *= -1.0;
      out.policy_score *= -1.0;
      out.commodity_score *= -1.0;
      out.carry_score *= -1.0;
   }

   out.blended_score = FXAI_Clamp(0.32 * out.trend_score +
                                  0.22 * out.carry_score +
                                  0.22 * out.policy_score +
                                  0.18 * out.value_score +
                                  0.06 * out.commodity_score,
                                  -1.0,
                                  1.0);
   out.ready = true;
   return true;
}

bool FXAI_BuildPairFactorContext(const string symbol,
                                 FXAIPairFactorContext &out)
{
   FXAI_ResetPairFactorContext(out);
   out.symbol = symbol;

   string base = "";
   string quote = "";
   FXAI_ParseSymbolLegs(symbol, base, quote);
   if(StringLen(base) != 3 || StringLen(quote) != 3)
      return false;

   FXAICurrencyFactorState base_state;
   FXAICurrencyFactorState quote_state;
   if(!FXAI_BuildCurrencyFactorState(base, base_state))
      return false;
   if(!FXAI_BuildCurrencyFactorState(quote, quote_state))
      return false;

   out.trend_score = FXAI_Clamp(base_state.trend_score - quote_state.trend_score, -1.0, 1.0);
   out.carry_score = FXAI_Clamp(base_state.carry_score - quote_state.carry_score, -1.0, 1.0);
   out.policy_score = FXAI_Clamp(base_state.policy_score - quote_state.policy_score, -1.0, 1.0);
   out.value_score = FXAI_Clamp(base_state.value_score - quote_state.value_score, -1.0, 1.0);
   out.commodity_score = FXAI_Clamp(base_state.commodity_score - quote_state.commodity_score, -1.0, 1.0);
   out.blended_score = FXAI_Clamp(base_state.blended_score - quote_state.blended_score, -1.0, 1.0);
   out.bias_direction = (out.blended_score > 0.08 ? 1 : (out.blended_score < -0.08 ? 0 : -1));
   out.alignment_score = FXAI_Clamp(0.50 + 0.50 * MathAbs(out.blended_score), 0.0, 1.0);
   out.generated_at = FXAI_ServerNow();
   out.stale = false;
   out.rationale = StringFormat("trend=%.2f carry=%.2f policy=%.2f value=%.2f commodity=%.2f",
                                out.trend_score,
                                out.carry_score,
                                out.policy_score,
                                out.value_score,
                                out.commodity_score);
   out.ready = true;
   return true;
}

bool FXAI_RefreshFactorContext(const string symbol,
                               FXAIPairFactorContext &out)
{
   if(!FXAI_BuildPairFactorContext(symbol, out))
      return false;

   g_factor_context_last_state = out;
   g_factor_context_last_ready = out.ready;
   g_factor_context_last_generated_at = out.generated_at;
   g_factor_context_last_symbol = out.symbol;
   g_factor_context_last_blended_score = out.blended_score;
   g_factor_context_last_alignment = out.alignment_score;
   g_factor_context_last_bias_direction = out.bias_direction;
   g_factor_context_last_rationale = out.rationale;
   return out.ready;
}

#endif // __FXAI_RUNTIME_FACTOR_CONTEXT_MQH__
