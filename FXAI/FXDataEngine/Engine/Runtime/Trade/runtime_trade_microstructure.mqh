#ifndef __FXAI_RUNTIME_TRADE_MICROSTRUCTURE_MQH__
#define __FXAI_RUNTIME_TRADE_MICROSTRUCTURE_MQH__

#define FXAI_MICROSTRUCTURE_FLAT_FILE "FXAI\\Runtime\\microstructure_snapshot_flat.tsv"
#define FXAI_MICROSTRUCTURE_SYMBOL_MAP_FILE "FXAI\\Runtime\\microstructure_symbol_map.tsv"
#define FXAI_MICROSTRUCTURE_MAX_REASONS 6

struct FXAIMicrostructurePairState
{
   bool ready;
   bool available;
   bool stale;
   datetime generated_at;
   double tick_imbalance_30s;
   double directional_efficiency_60s;
   double spread_current;
   double spread_zscore_60s;
   double tick_rate_60s;
   double tick_rate_zscore_60s;
   double realized_vol_5m;
   double vol_burst_score_5m;
   double local_extrema_breach_score_60s;
   bool sweep_and_reject_flag_60s;
   double breakout_reversal_score_60s;
   double exhaustion_proxy_60s;
   double liquidity_stress_score;
   double hostile_execution_score;
   string microstructure_regime;
   string session_tag;
   bool handoff_flag;
   double session_open_burst_score;
   double session_spread_behavior_score;
   string trade_gate;
   double caution_lot_scale;
   double caution_enter_prob_buffer;
   int reason_count;
   string reasons[FXAI_MICROSTRUCTURE_MAX_REASONS];
};

struct FXAIMicrostructureSymbolMapEntry
{
   string symbol;
   string pair_id;
};

double g_microstructure_last_liquidity_stress = 0.0;
double g_microstructure_last_hostile_execution = 0.0;
double g_microstructure_last_tick_imbalance = 0.0;
double g_microstructure_last_spread_zscore = 0.0;
bool g_microstructure_last_stale = true;
datetime g_microstructure_last_generated_at = 0;
string g_microstructure_last_trade_gate = "UNKNOWN";
string g_microstructure_last_regime = "UNKNOWN";
string g_microstructure_last_reasons_csv = "";
FXAIMicrostructureSymbolMapEntry g_microstructure_symbol_map[];
datetime g_microstructure_symbol_map_last_load = 0;

void FXAI_ResetMicrostructurePairState(FXAIMicrostructurePairState &out)
{
   out.ready = false;
   out.available = false;
   out.stale = true;
   out.generated_at = 0;
   out.tick_imbalance_30s = 0.0;
   out.directional_efficiency_60s = 0.0;
   out.spread_current = 0.0;
   out.spread_zscore_60s = 0.0;
   out.tick_rate_60s = 0.0;
   out.tick_rate_zscore_60s = 0.0;
   out.realized_vol_5m = 0.0;
   out.vol_burst_score_5m = 0.0;
   out.local_extrema_breach_score_60s = 0.0;
   out.sweep_and_reject_flag_60s = false;
   out.breakout_reversal_score_60s = 0.0;
   out.exhaustion_proxy_60s = 0.0;
   out.liquidity_stress_score = 0.0;
   out.hostile_execution_score = 0.0;
   out.microstructure_regime = "UNKNOWN";
   out.session_tag = "UNKNOWN";
   out.handoff_flag = false;
   out.session_open_burst_score = 0.0;
   out.session_spread_behavior_score = 0.0;
   out.trade_gate = "UNKNOWN";
   out.caution_lot_scale = -1.0;
   out.caution_enter_prob_buffer = -1.0;
   out.reason_count = 0;
   for(int i = 0; i < FXAI_MICROSTRUCTURE_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetMicrostructureGlobals(void)
{
   g_microstructure_last_liquidity_stress = 0.0;
   g_microstructure_last_hostile_execution = 0.0;
   g_microstructure_last_tick_imbalance = 0.0;
   g_microstructure_last_spread_zscore = 0.0;
   g_microstructure_last_stale = true;
   g_microstructure_last_generated_at = 0;
   g_microstructure_last_trade_gate = "UNKNOWN";
   g_microstructure_last_regime = "UNKNOWN";
   g_microstructure_last_reasons_csv = "";
}

void FXAI_MicrostructureLoadSymbolMap(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 && g_microstructure_symbol_map_last_load > 0 && (now_time - g_microstructure_symbol_map_last_load) < 60)
      return;

   ArrayResize(g_microstructure_symbol_map, 0);
   g_microstructure_symbol_map_last_load = now_time;

   int handle = FileOpen(FXAI_MICROSTRUCTURE_SYMBOL_MAP_FILE,
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
      int idx = ArraySize(g_microstructure_symbol_map);
      ArrayResize(g_microstructure_symbol_map, idx + 1);
      g_microstructure_symbol_map[idx].symbol = raw_symbol;
      g_microstructure_symbol_map[idx].pair_id = pair_id;
   }
   FileClose(handle);
}

string FXAI_MicrostructureMappedPairId(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   for(int i = 0; i < ArraySize(g_microstructure_symbol_map); i++)
   {
      if(g_microstructure_symbol_map[i].symbol == symbol)
         return g_microstructure_symbol_map[i].pair_id;
   }
   return "";
}

string FXAI_MicrostructurePairId(const string symbol)
{
   FXAI_MicrostructureLoadSymbolMap();
   string mapped = FXAI_MicrostructureMappedPairId(symbol);
   if(StringLen(mapped) == 6)
      return mapped;
   return FXAI_NewsPulsePairId(symbol);
}

void FXAI_MicrostructureAppendReason(FXAIMicrostructurePairState &state,
                                     const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i = 0; i < state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_MICROSTRUCTURE_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_MicrostructureReasonsCSV(const FXAIMicrostructurePairState &state)
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

void FXAI_ApplyMicrostructurePairState(const FXAIMicrostructurePairState &state)
{
   g_microstructure_last_liquidity_stress = state.liquidity_stress_score;
   g_microstructure_last_hostile_execution = state.hostile_execution_score;
   g_microstructure_last_tick_imbalance = state.tick_imbalance_30s;
   g_microstructure_last_spread_zscore = state.spread_zscore_60s;
   g_microstructure_last_stale = state.stale;
   g_microstructure_last_generated_at = state.generated_at;
   g_microstructure_last_trade_gate = state.trade_gate;
   g_microstructure_last_regime = state.microstructure_regime;
   g_microstructure_last_reasons_csv = FXAI_MicrostructureReasonsCSV(state);
}

bool FXAI_ReadMicrostructurePairState(const string symbol,
                                      FXAIMicrostructurePairState &out)
{
   FXAI_ResetMicrostructurePairState(out);
   string pair_id = FXAI_MicrostructurePairId(symbol);
   if(StringLen(pair_id) != 6)
      return false;

   int handle = FileOpen(FXAI_MICROSTRUCTURE_FLAT_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      FXAI_ResetMicrostructureGlobals();
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
         if(key == "tick_imbalance_30s")
            out.tick_imbalance_30s = StringToDouble(value);
         else if(key == "directional_efficiency_60s")
            out.directional_efficiency_60s = StringToDouble(value);
         else if(key == "spread_current")
            out.spread_current = StringToDouble(value);
         else if(key == "spread_zscore_60s")
            out.spread_zscore_60s = StringToDouble(value);
         else if(key == "tick_rate_60s")
            out.tick_rate_60s = StringToDouble(value);
         else if(key == "tick_rate_zscore_60s")
            out.tick_rate_zscore_60s = StringToDouble(value);
         else if(key == "realized_vol_5m")
            out.realized_vol_5m = StringToDouble(value);
         else if(key == "vol_burst_score_5m")
            out.vol_burst_score_5m = StringToDouble(value);
         else if(key == "local_extrema_breach_score_60s")
            out.local_extrema_breach_score_60s = StringToDouble(value);
         else if(key == "sweep_and_reject_flag_60s")
            out.sweep_and_reject_flag_60s = (StringToInteger(value) != 0);
         else if(key == "breakout_reversal_score_60s")
            out.breakout_reversal_score_60s = StringToDouble(value);
         else if(key == "exhaustion_proxy_60s")
            out.exhaustion_proxy_60s = StringToDouble(value);
         else if(key == "liquidity_stress_score")
            out.liquidity_stress_score = StringToDouble(value);
         else if(key == "hostile_execution_score")
            out.hostile_execution_score = StringToDouble(value);
         else if(key == "microstructure_regime")
            out.microstructure_regime = value;
         else if(key == "session_tag")
            out.session_tag = value;
         else if(key == "handoff_flag")
            out.handoff_flag = (StringToInteger(value) != 0);
         else if(key == "session_open_burst_score")
            out.session_open_burst_score = StringToDouble(value);
         else if(key == "session_spread_behavior_score")
            out.session_spread_behavior_score = StringToDouble(value);
         else if(key == "trade_gate")
            out.trade_gate = value;
         else if(key == "stale")
            out.stale = (StringToInteger(value) != 0);
         else if(key == "caution_lot_scale")
            out.caution_lot_scale = StringToDouble(value);
         else if(key == "caution_enter_prob_buffer")
            out.caution_enter_prob_buffer = StringToDouble(value);
         continue;
      }

      if(kind == "pair_reason" && target == pair_id)
         FXAI_MicrostructureAppendReason(out, value);
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = TimeCurrent();
      if(now_time <= 0)
         now_time = TimeTradeServer();
      if(now_time > 0 && out.generated_at > 0)
      {
         if((now_time - out.generated_at) > MathMax(MicrostructureFreshnessMaxSec, 10))
            out.stale = true;
      }
      out.tick_imbalance_30s = FXAI_Clamp(out.tick_imbalance_30s, -1.0, 1.0);
      out.directional_efficiency_60s = FXAI_Clamp(out.directional_efficiency_60s, 0.0, 1.0);
      out.spread_current = MathMax(out.spread_current, 0.0);
      out.spread_zscore_60s = FXAI_Clamp(out.spread_zscore_60s, -8.0, 8.0);
      out.tick_rate_60s = MathMax(out.tick_rate_60s, 0.0);
      out.tick_rate_zscore_60s = FXAI_Clamp(out.tick_rate_zscore_60s, -8.0, 8.0);
      out.realized_vol_5m = MathMax(out.realized_vol_5m, 0.0);
      out.vol_burst_score_5m = FXAI_Clamp(out.vol_burst_score_5m, 0.0, 8.0);
      out.local_extrema_breach_score_60s = FXAI_Clamp(out.local_extrema_breach_score_60s, 0.0, 1.0);
      out.breakout_reversal_score_60s = FXAI_Clamp(out.breakout_reversal_score_60s, 0.0, 1.0);
      out.exhaustion_proxy_60s = FXAI_Clamp(out.exhaustion_proxy_60s, 0.0, 1.0);
      out.liquidity_stress_score = FXAI_Clamp(out.liquidity_stress_score, 0.0, 1.0);
      out.hostile_execution_score = FXAI_Clamp(out.hostile_execution_score, 0.0, 1.0);
      out.session_open_burst_score = FXAI_Clamp(out.session_open_burst_score, 0.0, 1.0);
      out.session_spread_behavior_score = FXAI_Clamp(out.session_spread_behavior_score, 0.0, 1.0);
      if(StringLen(out.microstructure_regime) <= 0)
         out.microstructure_regime = "UNKNOWN";
      if(StringLen(out.session_tag) <= 0)
         out.session_tag = "UNKNOWN";
      if(StringLen(out.trade_gate) <= 0)
         out.trade_gate = "UNKNOWN";
   }

   FXAI_ApplyMicrostructurePairState(out);
   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_MICROSTRUCTURE_MQH__
