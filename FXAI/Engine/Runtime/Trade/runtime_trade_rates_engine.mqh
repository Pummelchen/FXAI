#ifndef __FXAI_RUNTIME_TRADE_RATES_ENGINE_MQH__
#define __FXAI_RUNTIME_TRADE_RATES_ENGINE_MQH__

#define FXAI_RATES_ENGINE_FLAT_FILE "FXAI\\Runtime\\rates_snapshot_flat.tsv"
#define FXAI_RATES_ENGINE_SYMBOL_MAP_FILE "FXAI\\Runtime\\rates_symbol_map.tsv"
#define FXAI_RATES_ENGINE_MAX_REASONS 6

struct FXAIRatesEnginePairState
{
   bool ready;
   bool available;
   bool stale;
   datetime generated_at;
   double front_end_diff;
   double expected_path_diff;
   double curve_divergence_score;
   double policy_divergence_score;
   double rates_risk_score;
   double macro_to_rates_transmission_score;
   bool meeting_path_reprice_now;
   string rates_regime;
   string trade_gate;
   string policy_alignment;
   int reason_count;
   string reasons[FXAI_RATES_ENGINE_MAX_REASONS];
};

struct FXAIRatesEngineSymbolMapEntry
{
   string symbol;
   string pair_id;
};

double g_rates_engine_last_risk_score = 0.0;
double g_rates_engine_last_policy_divergence = 0.0;
bool g_rates_engine_last_stale = true;
bool g_rates_engine_last_meeting_reprice = false;
datetime g_rates_engine_last_generated_at = 0;
string g_rates_engine_last_trade_gate = "UNKNOWN";
string g_rates_engine_last_regime = "UNKNOWN";
string g_rates_engine_last_reasons_csv = "";
FXAIRatesEngineSymbolMapEntry g_rates_engine_symbol_map[];
datetime g_rates_engine_symbol_map_last_load = 0;

void FXAI_ResetRatesEnginePairState(FXAIRatesEnginePairState &out)
{
   out.ready = false;
   out.available = false;
   out.stale = true;
   out.generated_at = 0;
   out.front_end_diff = 0.0;
   out.expected_path_diff = 0.0;
   out.curve_divergence_score = 0.0;
   out.policy_divergence_score = 0.0;
   out.rates_risk_score = 0.0;
   out.macro_to_rates_transmission_score = 0.0;
   out.meeting_path_reprice_now = false;
   out.rates_regime = "UNKNOWN";
   out.trade_gate = "UNKNOWN";
   out.policy_alignment = "balanced";
   out.reason_count = 0;
   for(int i = 0; i < FXAI_RATES_ENGINE_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetRatesEngineGlobals(void)
{
   g_rates_engine_last_risk_score = 0.0;
   g_rates_engine_last_policy_divergence = 0.0;
   g_rates_engine_last_stale = true;
   g_rates_engine_last_meeting_reprice = false;
   g_rates_engine_last_generated_at = 0;
   g_rates_engine_last_trade_gate = "UNKNOWN";
   g_rates_engine_last_regime = "UNKNOWN";
   g_rates_engine_last_reasons_csv = "";
}

void FXAI_RatesEngineLoadSymbolMap(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 && g_rates_engine_symbol_map_last_load > 0 && (now_time - g_rates_engine_symbol_map_last_load) < 60)
      return;

   ArrayResize(g_rates_engine_symbol_map, 0);
   g_rates_engine_symbol_map_last_load = now_time;

   int handle = FileOpen(FXAI_RATES_ENGINE_SYMBOL_MAP_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 3)
         continue;
      if(parts[0] != "symbol")
         continue;
      string raw_symbol = parts[1];
      string pair_id = parts[2];
      StringToUpper(raw_symbol);
      StringToUpper(pair_id);
      if(StringLen(raw_symbol) <= 0 || StringLen(pair_id) != 6)
         continue;
      int idx = ArraySize(g_rates_engine_symbol_map);
      ArrayResize(g_rates_engine_symbol_map, idx + 1);
      g_rates_engine_symbol_map[idx].symbol = raw_symbol;
      g_rates_engine_symbol_map[idx].pair_id = pair_id;
   }
   FileClose(handle);
}

string FXAI_RatesEngineMappedPairId(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   for(int i = 0; i < ArraySize(g_rates_engine_symbol_map); i++)
   {
      if(g_rates_engine_symbol_map[i].symbol == symbol)
         return g_rates_engine_symbol_map[i].pair_id;
   }
   return "";
}

string FXAI_RatesEnginePairId(const string symbol)
{
   FXAI_RatesEngineLoadSymbolMap();
   string mapped = FXAI_RatesEngineMappedPairId(symbol);
   if(StringLen(mapped) == 6)
      return mapped;
   return FXAI_NewsPulsePairId(symbol);
}

void FXAI_RatesEngineAppendReason(FXAIRatesEnginePairState &state,
                                  const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i = 0; i < state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_RATES_ENGINE_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_RatesEngineReasonsCSV(const FXAIRatesEnginePairState &state)
{
   string joined = "";
   for(int i = 0; i < state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reasons[i];
   }
   return joined;
}

void FXAI_ApplyRatesEnginePairState(const FXAIRatesEnginePairState &state)
{
   g_rates_engine_last_risk_score = state.rates_risk_score;
   g_rates_engine_last_policy_divergence = state.policy_divergence_score;
   g_rates_engine_last_stale = state.stale;
   g_rates_engine_last_meeting_reprice = state.meeting_path_reprice_now;
   g_rates_engine_last_generated_at = state.generated_at;
   g_rates_engine_last_trade_gate = state.trade_gate;
   g_rates_engine_last_regime = state.rates_regime;
   g_rates_engine_last_reasons_csv = FXAI_RatesEngineReasonsCSV(state);
}

bool FXAI_ReadRatesEnginePairState(const string symbol,
                                   FXAIRatesEnginePairState &out)
{
   FXAI_ResetRatesEnginePairState(out);
   string pair_id = FXAI_RatesEnginePairId(symbol);
   if(StringLen(pair_id) != 6)
      return false;

   int handle = FileOpen(FXAI_RATES_ENGINE_FLAT_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      FXAI_ResetRatesEngineGlobals();
      return false;
   }

   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 4)
         continue;
      string kind = parts[0];
      string target = parts[1];
      string key = parts[2];
      string value = parts[3];

      if(kind == "meta" && target == "global")
      {
         if(key == "generated_at_unix")
            out.generated_at = (datetime)StringToInteger(value);
         continue;
      }

      if(kind == "pair" && target == pair_id)
      {
         out.available = true;
         out.ready = true;
         if(key == "front_end_diff")
            out.front_end_diff = StringToDouble(value);
         else if(key == "expected_path_diff")
            out.expected_path_diff = StringToDouble(value);
         else if(key == "curve_divergence_score")
            out.curve_divergence_score = StringToDouble(value);
         else if(key == "policy_divergence_score")
            out.policy_divergence_score = StringToDouble(value);
         else if(key == "rates_risk_score")
            out.rates_risk_score = StringToDouble(value);
         else if(key == "macro_to_rates_transmission_score")
            out.macro_to_rates_transmission_score = StringToDouble(value);
         else if(key == "meeting_path_reprice_now")
            out.meeting_path_reprice_now = (StringToInteger(value) != 0);
         else if(key == "stale")
            out.stale = (StringToInteger(value) != 0);
         else if(key == "rates_regime")
            out.rates_regime = value;
         else if(key == "trade_gate")
            out.trade_gate = value;
         else if(key == "policy_alignment")
            out.policy_alignment = value;
         continue;
      }

      if(kind == "pair_reason" && target == pair_id)
         FXAI_RatesEngineAppendReason(out, value);
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = TimeCurrent();
      if(now_time <= 0)
         now_time = TimeTradeServer();
      if(now_time > 0 && out.generated_at > 0)
      {
         if((now_time - out.generated_at) > MathMax(RatesEngineFreshnessMaxSec, 60))
            out.stale = true;
      }
      out.front_end_diff = FXAI_Clamp(out.front_end_diff, -10.0, 10.0);
      out.expected_path_diff = FXAI_Clamp(out.expected_path_diff, -10.0, 10.0);
      out.curve_divergence_score = FXAI_Clamp(out.curve_divergence_score, 0.0, 1.0);
      out.policy_divergence_score = FXAI_Clamp(out.policy_divergence_score, 0.0, 1.0);
      out.rates_risk_score = FXAI_Clamp(out.rates_risk_score, 0.0, 1.0);
      out.macro_to_rates_transmission_score = FXAI_Clamp(out.macro_to_rates_transmission_score, 0.0, 1.0);
      if(StringLen(out.rates_regime) <= 0)
         out.rates_regime = "UNKNOWN";
      if(StringLen(out.trade_gate) <= 0)
         out.trade_gate = "UNKNOWN";
      if(StringLen(out.policy_alignment) <= 0)
         out.policy_alignment = "balanced";
   }

   FXAI_ApplyRatesEnginePairState(out);
   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_RATES_ENGINE_MQH__
