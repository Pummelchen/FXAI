#ifndef __FXAI_RUNTIME_TRADE_CROSS_ASSET_STATE_MQH__
#define __FXAI_RUNTIME_TRADE_CROSS_ASSET_STATE_MQH__

#include "runtime_trade_newspulse.mqh"

#define FXAI_CROSS_ASSET_FLAT_FILE "FXAI\\Runtime\\cross_asset_snapshot_flat.tsv"
#define FXAI_CROSS_ASSET_SYMBOL_MAP_FILE "FXAI\\Runtime\\cross_asset_symbol_map.tsv"
#define FXAI_CROSS_ASSET_MAX_REASONS 8

struct FXAICrossAssetPairState
{
   bool ready;
   bool available;
   bool stale;
   datetime generated_at;
   double rates_repricing_score;
   double risk_off_score;
   double commodity_shock_score;
   double volatility_shock_score;
   double usd_liquidity_stress_score;
   double cross_asset_dislocation_score;
   double pair_cross_asset_risk_score;
   double pair_sensitivity;
   string macro_state;
   string risk_state;
   string liquidity_state;
   string trade_gate;
   int reason_count;
   string reasons[FXAI_CROSS_ASSET_MAX_REASONS];
};

struct FXAICrossAssetSymbolMapEntry
{
   string symbol;
   string pair_id;
};

double g_cross_asset_last_pair_risk = 0.0;
double g_cross_asset_last_rates_repricing = 0.0;
double g_cross_asset_last_risk_off = 0.0;
double g_cross_asset_last_liquidity_stress = 0.0;
bool g_cross_asset_last_stale = true;
datetime g_cross_asset_last_generated_at = 0;
string g_cross_asset_last_macro_state = "UNKNOWN";
string g_cross_asset_last_trade_gate = "UNKNOWN";
string g_cross_asset_last_reasons_csv = "";
FXAICrossAssetSymbolMapEntry g_cross_asset_symbol_map[];
datetime g_cross_asset_symbol_map_last_load = 0;

void FXAI_ResetCrossAssetPairState(FXAICrossAssetPairState &out)
{
   out.ready = false;
   out.available = false;
   out.stale = true;
   out.generated_at = 0;
   out.rates_repricing_score = 0.0;
   out.risk_off_score = 0.0;
   out.commodity_shock_score = 0.0;
   out.volatility_shock_score = 0.0;
   out.usd_liquidity_stress_score = 0.0;
   out.cross_asset_dislocation_score = 0.0;
   out.pair_cross_asset_risk_score = 0.0;
   out.pair_sensitivity = 0.0;
   out.macro_state = "UNKNOWN";
   out.risk_state = "UNKNOWN";
   out.liquidity_state = "UNKNOWN";
   out.trade_gate = "UNKNOWN";
   out.reason_count = 0;
   for(int i=0; i<FXAI_CROSS_ASSET_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetCrossAssetGlobals(void)
{
   g_cross_asset_last_pair_risk = 0.0;
   g_cross_asset_last_rates_repricing = 0.0;
   g_cross_asset_last_risk_off = 0.0;
   g_cross_asset_last_liquidity_stress = 0.0;
   g_cross_asset_last_stale = true;
   g_cross_asset_last_generated_at = 0;
   g_cross_asset_last_macro_state = "UNKNOWN";
   g_cross_asset_last_trade_gate = "UNKNOWN";
   g_cross_asset_last_reasons_csv = "";
}

void FXAI_CrossAssetLoadSymbolMap(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 && g_cross_asset_symbol_map_last_load > 0 && (now_time - g_cross_asset_symbol_map_last_load) < 60)
      return;

   ArrayResize(g_cross_asset_symbol_map, 0);
   g_cross_asset_symbol_map_last_load = now_time;

   int handle = FileOpen(FXAI_CROSS_ASSET_SYMBOL_MAP_FILE,
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
      int idx = ArraySize(g_cross_asset_symbol_map);
      ArrayResize(g_cross_asset_symbol_map, idx + 1);
      g_cross_asset_symbol_map[idx].symbol = raw_symbol;
      g_cross_asset_symbol_map[idx].pair_id = pair_id;
   }
   FileClose(handle);
}

string FXAI_CrossAssetMappedPairId(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   for(int i=0; i<ArraySize(g_cross_asset_symbol_map); i++)
   {
      if(g_cross_asset_symbol_map[i].symbol == symbol)
         return g_cross_asset_symbol_map[i].pair_id;
   }
   return "";
}

string FXAI_CrossAssetPairId(const string symbol)
{
   FXAI_CrossAssetLoadSymbolMap();
   string mapped = FXAI_CrossAssetMappedPairId(symbol);
   if(StringLen(mapped) == 6)
      return mapped;
   return FXAI_NewsPulsePairId(symbol);
}

void FXAI_CrossAssetAppendReason(FXAICrossAssetPairState &state,
                                 const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_CROSS_ASSET_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_CrossAssetReasonsCSV(const FXAICrossAssetPairState &state)
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

void FXAI_ApplyCrossAssetPairState(const FXAICrossAssetPairState &state)
{
   g_cross_asset_last_pair_risk = state.pair_cross_asset_risk_score;
   g_cross_asset_last_rates_repricing = state.rates_repricing_score;
   g_cross_asset_last_risk_off = state.risk_off_score;
   g_cross_asset_last_liquidity_stress = state.usd_liquidity_stress_score;
   g_cross_asset_last_stale = state.stale;
   g_cross_asset_last_generated_at = state.generated_at;
   g_cross_asset_last_macro_state = state.macro_state;
   g_cross_asset_last_trade_gate = state.trade_gate;
   g_cross_asset_last_reasons_csv = FXAI_CrossAssetReasonsCSV(state);
}

bool FXAI_ReadCrossAssetPairState(const string symbol,
                                  FXAICrossAssetPairState &out)
{
   FXAI_ResetCrossAssetPairState(out);
   string pair_id = FXAI_CrossAssetPairId(symbol);
   if(StringLen(pair_id) != 6)
      return false;

   int handle = FileOpen(FXAI_CROSS_ASSET_FLAT_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      FXAI_ResetCrossAssetGlobals();
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

      if(target == "global")
      {
         if(kind == "meta" && key == "generated_at_unix")
            out.generated_at = (datetime)StringToInteger(value);
         else if(kind == "score")
         {
            if(key == "rates_repricing_score")
               out.rates_repricing_score = StringToDouble(value);
            else if(key == "risk_off_score")
               out.risk_off_score = StringToDouble(value);
            else if(key == "commodity_shock_score")
               out.commodity_shock_score = StringToDouble(value);
            else if(key == "volatility_shock_score")
               out.volatility_shock_score = StringToDouble(value);
            else if(key == "usd_liquidity_stress_score")
               out.usd_liquidity_stress_score = StringToDouble(value);
            else if(key == "cross_asset_dislocation_score")
               out.cross_asset_dislocation_score = StringToDouble(value);
         }
         continue;
      }

      if(kind == "pair" && target == pair_id)
      {
         out.available = true;
         out.ready = true;
         if(key == "pair_cross_asset_risk_score")
            out.pair_cross_asset_risk_score = StringToDouble(value);
         else if(key == "pair_sensitivity")
            out.pair_sensitivity = StringToDouble(value);
         else if(key == "macro_state")
            out.macro_state = value;
         else if(key == "risk_state")
            out.risk_state = value;
         else if(key == "liquidity_state")
            out.liquidity_state = value;
         else if(key == "trade_gate")
            out.trade_gate = value;
         else if(key == "stale")
            out.stale = (StringToInteger(value) != 0);
         continue;
      }

      if(kind == "pair_reason" && target == pair_id)
         FXAI_CrossAssetAppendReason(out, value);
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = TimeCurrent();
      if(now_time <= 0)
         now_time = TimeTradeServer();
      if(now_time > 0 && out.generated_at > 0)
         out.stale = out.stale || ((now_time - out.generated_at) > MathMax(CrossAssetFreshnessMaxSec, 60));
      else
         out.stale = true;

      FXAI_ApplyCrossAssetPairState(out);
   }
   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_CROSS_ASSET_STATE_MQH__
