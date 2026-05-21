#ifndef __FXAI_RUNTIME_TRADE_EXECUTION_QUALITY_MQH__
#define __FXAI_RUNTIME_TRADE_EXECUTION_QUALITY_MQH__

#ifndef FXAI_EXEC_QUALITY_MAX_REASONS
#define FXAI_EXEC_QUALITY_MAX_REASONS 12
#endif

struct FXAIExecutionQualityPairState
{
   bool ready;
   bool available;
   bool stale;
   bool fallback_used;
   bool memory_stale;
   bool data_stale;
   bool support_usable;
   datetime generated_at;
   string method;
   string session_label;
   string regime_label;
   string selected_tier_kind;
   string selected_tier_key;
   int    selected_support;
   double selected_quality;
   double spread_now_points;
   double spread_expected_points;
   double spread_widening_risk;
   double expected_slippage_points;
   double slippage_risk;
   double fill_quality_score;
   double latency_sensitivity_score;
   double liquidity_fragility_score;
   double execution_quality_score;
   double allowed_deviation_points;
   double caution_lot_scale;
   double caution_enter_prob_buffer;
   string execution_state;
   int    reason_count;
   string reasons[FXAI_EXEC_QUALITY_MAX_REASONS];
};

void FXAI_ResetExecutionQualityPairState(FXAIExecutionQualityPairState &out)
{
   out.ready = false;
   out.available = false;
   out.stale = true;
   out.fallback_used = false;
   out.memory_stale = true;
   out.data_stale = true;
   out.support_usable = false;
   out.generated_at = 0;
   out.method = "SCORECARD_V1";
   out.session_label = "UNKNOWN";
   out.regime_label = "UNKNOWN";
   out.selected_tier_kind = "GLOBAL";
   out.selected_tier_key = "GLOBAL|*|*|*";
   out.selected_support = 0;
   out.selected_quality = 0.0;
   out.spread_now_points = 0.0;
   out.spread_expected_points = 0.0;
   out.spread_widening_risk = 0.0;
   out.expected_slippage_points = 0.0;
   out.slippage_risk = 0.0;
   out.fill_quality_score = 0.0;
   out.latency_sensitivity_score = 0.0;
   out.liquidity_fragility_score = 0.0;
   out.execution_quality_score = 0.0;
   out.allowed_deviation_points = 0.0;
   out.caution_lot_scale = 1.0;
   out.caution_enter_prob_buffer = 0.0;
   out.execution_state = "UNKNOWN";
   out.reason_count = 0;
   for(int i=0; i<FXAI_EXEC_QUALITY_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetExecutionQualityGlobals(void)
{
   g_execution_quality_last_ready = false;
   g_execution_quality_last_fallback_used = false;
   g_execution_quality_last_memory_stale = true;
   g_execution_quality_last_data_stale = true;
   g_execution_quality_last_support_usable = false;
   g_execution_quality_last_generated_at = 0;
   g_execution_quality_last_method = "SCORECARD_V1";
   g_execution_quality_last_tier_kind = "GLOBAL";
   g_execution_quality_last_tier_key = "GLOBAL|*|*|*";
   g_execution_quality_last_support = 0;
   g_execution_quality_last_quality = 0.0;
   g_execution_quality_last_spread_now = 0.0;
   g_execution_quality_last_spread_expected = 0.0;
   g_execution_quality_last_spread_widening_risk = 0.0;
   g_execution_quality_last_expected_slippage = 0.0;
   g_execution_quality_last_slippage_risk = 0.0;
   g_execution_quality_last_fill_quality = 0.0;
   g_execution_quality_last_latency_sensitivity = 0.0;
   g_execution_quality_last_liquidity_fragility = 0.0;
   g_execution_quality_last_quality_score = 0.0;
   g_execution_quality_last_allowed_deviation = 0.0;
   g_execution_quality_last_caution_lot_scale = 1.0;
   g_execution_quality_last_caution_enter_prob_buffer = 0.0;
   g_execution_quality_last_state = "UNKNOWN";
   g_execution_quality_last_reasons_csv = "";
}

void FXAI_ExecutionQualityAppendReason(FXAIExecutionQualityPairState &state,
                                       const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_EXEC_QUALITY_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_ExecutionQualityReasonsCSV(const FXAIExecutionQualityPairState &state)
{
   string joined = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reasons[i];
   }
   return joined;
}

bool FXAI_ReadExecutionQualityPairState(const string symbol,
                                        FXAIExecutionQualityPairState &out)
{
   FXAI_ResetExecutionQualityPairState(out);
   int handle = FileOpen("FXAI\\Runtime\\fxai_execution_quality_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv",
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      FXAI_ResetExecutionQualityGlobals();
      return false;
   }

   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 2)
         continue;
      string key = parts[0];
      string value = parts[1];
      out.available = true;
      out.ready = true;
      if(key == "generated_at") out.generated_at = (datetime)StringToInteger(value);
      else if(key == "method") out.method = value;
      else if(key == "session_label") out.session_label = value;
      else if(key == "regime_label") out.regime_label = value;
      else if(key == "selected_tier_kind") out.selected_tier_kind = value;
      else if(key == "selected_tier_key") out.selected_tier_key = value;
      else if(key == "selected_support") out.selected_support = (int)StringToInteger(value);
      else if(key == "selected_quality") out.selected_quality = StringToDouble(value);
      else if(key == "fallback_used") out.fallback_used = (StringToInteger(value) != 0);
      else if(key == "memory_stale") out.memory_stale = (StringToInteger(value) != 0);
      else if(key == "data_stale") out.data_stale = (StringToInteger(value) != 0);
      else if(key == "support_usable") out.support_usable = (StringToInteger(value) != 0);
      else if(key == "spread_now_points") out.spread_now_points = StringToDouble(value);
      else if(key == "spread_expected_points") out.spread_expected_points = StringToDouble(value);
      else if(key == "spread_widening_risk") out.spread_widening_risk = StringToDouble(value);
      else if(key == "expected_slippage_points") out.expected_slippage_points = StringToDouble(value);
      else if(key == "slippage_risk") out.slippage_risk = StringToDouble(value);
      else if(key == "fill_quality_score") out.fill_quality_score = StringToDouble(value);
      else if(key == "latency_sensitivity_score") out.latency_sensitivity_score = StringToDouble(value);
      else if(key == "liquidity_fragility_score") out.liquidity_fragility_score = StringToDouble(value);
      else if(key == "execution_quality_score") out.execution_quality_score = StringToDouble(value);
      else if(key == "allowed_deviation_points") out.allowed_deviation_points = StringToDouble(value);
      else if(key == "caution_lot_scale") out.caution_lot_scale = StringToDouble(value);
      else if(key == "caution_enter_prob_buffer") out.caution_enter_prob_buffer = StringToDouble(value);
      else if(key == "execution_state") out.execution_state = value;
      else if(key == "reasons_csv")
      {
         string reason_parts[];
         int rc = StringSplit(value, ';', reason_parts);
         for(int i=0; i<rc; i++)
         {
            string reason = reason_parts[i];
            StringTrimLeft(reason);
            StringTrimRight(reason);
            FXAI_ExecutionQualityAppendReason(out, reason);
         }
      }
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = TimeCurrent();
      if(now_time <= 0)
         now_time = TimeTradeServer();
      if(now_time > 0 && out.generated_at > 0)
         out.stale = ((now_time - out.generated_at) > MathMax(ExecutionQualityFreshnessMaxSec, 30));
      else
         out.stale = true;

      g_execution_quality_last_ready = out.ready;
      g_execution_quality_last_fallback_used = out.fallback_used;
      g_execution_quality_last_memory_stale = out.memory_stale;
      g_execution_quality_last_data_stale = out.data_stale || out.stale;
      g_execution_quality_last_support_usable = out.support_usable;
      g_execution_quality_last_generated_at = out.generated_at;
      g_execution_quality_last_method = out.method;
      g_execution_quality_last_tier_kind = out.selected_tier_kind;
      g_execution_quality_last_tier_key = out.selected_tier_key;
      g_execution_quality_last_support = out.selected_support;
      g_execution_quality_last_quality = out.selected_quality;
      g_execution_quality_last_spread_now = out.spread_now_points;
      g_execution_quality_last_spread_expected = out.spread_expected_points;
      g_execution_quality_last_spread_widening_risk = out.spread_widening_risk;
      g_execution_quality_last_expected_slippage = out.expected_slippage_points;
      g_execution_quality_last_slippage_risk = out.slippage_risk;
      g_execution_quality_last_fill_quality = out.fill_quality_score;
      g_execution_quality_last_latency_sensitivity = out.latency_sensitivity_score;
      g_execution_quality_last_liquidity_fragility = out.liquidity_fragility_score;
      g_execution_quality_last_quality_score = out.execution_quality_score;
      g_execution_quality_last_allowed_deviation = out.allowed_deviation_points;
      g_execution_quality_last_caution_lot_scale = out.caution_lot_scale;
      g_execution_quality_last_caution_enter_prob_buffer = out.caution_enter_prob_buffer;
      g_execution_quality_last_state = out.execution_state;
      g_execution_quality_last_reasons_csv = FXAI_ExecutionQualityReasonsCSV(out);
   }

   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_EXECUTION_QUALITY_MQH__
