#ifndef __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__
#define __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__

#define FXAI_NEWSPULSE_FLAT_FILE "FXAI\\Runtime\\news_snapshot_flat.tsv"
#define FXAI_NEWSPULSE_SYMBOL_MAP_FILE "FXAI\\Runtime\\news_symbol_map.tsv"
#define FXAI_NEWSPULSE_MAX_REASONS 6

struct FXAINewsPulsePairState
{
   bool ready;
   bool available;
   bool stale;
   datetime generated_at;
   int event_eta_min;
   double news_risk_score;
   double news_pressure;
   string trade_gate;
   string session_profile;
   string calibration_profile;
   string watchlist_tags_csv;
   double caution_lot_scale;
   double caution_enter_prob_buffer;
   int reason_count;
   string reasons[FXAI_NEWSPULSE_MAX_REASONS];
};

struct FXAINewsPulseSymbolMapEntry
{
   string symbol;
   string pair_id;
};

double g_newspulse_last_risk_score = 0.0;
double g_newspulse_last_news_pressure = 0.0;
int g_newspulse_last_event_eta_min = -1;
bool g_newspulse_last_stale = true;
datetime g_newspulse_last_generated_at = 0;
string g_newspulse_last_trade_gate = "UNKNOWN";
string g_newspulse_last_reasons_csv = "";
FXAINewsPulseSymbolMapEntry g_newspulse_symbol_map[];
datetime g_newspulse_symbol_map_last_load = 0;

void FXAI_ResetNewsPulsePairState(FXAINewsPulsePairState &out)
{
   out.ready = false;
   out.available = false;
   out.stale = true;
   out.generated_at = 0;
   out.event_eta_min = -1;
   out.news_risk_score = 0.0;
   out.news_pressure = 0.0;
   out.trade_gate = "UNKNOWN";
   out.session_profile = "default";
   out.calibration_profile = "default";
   out.watchlist_tags_csv = "";
   out.caution_lot_scale = -1.0;
   out.caution_enter_prob_buffer = -1.0;
   out.reason_count = 0;
   for(int i = 0; i < FXAI_NEWSPULSE_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetNewsPulseGlobals(void)
{
   g_newspulse_last_risk_score = 0.0;
   g_newspulse_last_news_pressure = 0.0;
   g_newspulse_last_event_eta_min = -1;
   g_newspulse_last_stale = true;
   g_newspulse_last_generated_at = 0;
   g_newspulse_last_trade_gate = "UNKNOWN";
   g_newspulse_last_reasons_csv = "";
}

bool FXAI_NewsPulseSupportedCurrency(const string code)
{
   return (code == "USD" || code == "EUR" || code == "GBP" || code == "JPY" ||
           code == "CHF" || code == "CAD" || code == "AUD" || code == "NZD" ||
           code == "SEK" || code == "NOK");
}

string FXAI_NewsPulseAlphaOnly(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   string out = "";
   int len = StringLen(symbol);
   for(int i = 0; i < len; i++)
   {
      string token = StringSubstr(symbol, i, 1);
      if(token >= "A" && token <= "Z")
         out += token;
   }
   return out;
}

void FXAI_NewsPulseLoadSymbolMap(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 && g_newspulse_symbol_map_last_load > 0 && (now_time - g_newspulse_symbol_map_last_load) < 60)
      return;

   ArrayResize(g_newspulse_symbol_map, 0);
   g_newspulse_symbol_map_last_load = now_time;

   int handle = FileOpen(FXAI_NEWSPULSE_SYMBOL_MAP_FILE,
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
      int idx = ArraySize(g_newspulse_symbol_map);
      ArrayResize(g_newspulse_symbol_map, idx + 1);
      g_newspulse_symbol_map[idx].symbol = raw_symbol;
      g_newspulse_symbol_map[idx].pair_id = pair_id;
   }
   FileClose(handle);
}

string FXAI_NewsPulseMappedPairId(const string raw_symbol)
{
   string symbol = raw_symbol;
   StringToUpper(symbol);
   for(int i = 0; i < ArraySize(g_newspulse_symbol_map); i++)
   {
      if(g_newspulse_symbol_map[i].symbol == symbol)
         return g_newspulse_symbol_map[i].pair_id;
   }
   return "";
}

string FXAI_NewsPulseHeuristicPairId(const string raw_symbol)
{
   string alpha = FXAI_NewsPulseAlphaOnly(raw_symbol);
   int len = StringLen(alpha);
   if(len < 6)
      return "";
   for(int i = 0; i <= len - 6; i++)
   {
      string candidate = StringSubstr(alpha, i, 6);
      string base = StringSubstr(candidate, 0, 3);
      string quote = StringSubstr(candidate, 3, 3);
      if(base == quote)
         continue;
      if(FXAI_NewsPulseSupportedCurrency(base) && FXAI_NewsPulseSupportedCurrency(quote))
         return candidate;
   }
   return "";
}

string FXAI_NewsPulsePairId(const string symbol)
{
   FXAI_NewsPulseLoadSymbolMap();
   string mapped = FXAI_NewsPulseMappedPairId(symbol);
   if(StringLen(mapped) == 6)
      return mapped;
   return FXAI_NewsPulseHeuristicPairId(symbol);
}

void FXAI_NewsPulseAppendReason(FXAINewsPulsePairState &state,
                                const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i = 0; i < state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_NEWSPULSE_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_NewsPulseReasonsCSV(const FXAINewsPulsePairState &state)
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

void FXAI_ApplyNewsPulsePairState(const FXAINewsPulsePairState &state)
{
   g_newspulse_last_risk_score = state.news_risk_score;
   g_newspulse_last_news_pressure = state.news_pressure;
   g_newspulse_last_event_eta_min = state.event_eta_min;
   g_newspulse_last_stale = state.stale;
   g_newspulse_last_generated_at = state.generated_at;
   g_newspulse_last_trade_gate = state.trade_gate;
   g_newspulse_last_reasons_csv = FXAI_NewsPulseReasonsCSV(state);
}

bool FXAI_ApplyNewsPulseCalendarFallback(const string symbol,
                                         FXAINewsPulsePairState &out)
{
   FXAICalendarCachePairState calendar_state;
   if(!FXAI_ReadCalendarCachePairState(symbol, calendar_state))
      return false;

   out.available = calendar_state.ready;
   out.ready = calendar_state.ready;
   out.stale = calendar_state.stale;
   out.generated_at = calendar_state.generated_at;
   out.event_eta_min = calendar_state.next_event_eta_min;
   out.news_risk_score = FXAI_Clamp(calendar_state.event_risk_score, 0.0, 1.0);
   out.news_pressure = FXAI_Clamp(0.50 * calendar_state.event_risk_score, -1.0, 1.0);
   out.trade_gate = calendar_state.trade_gate;
   out.session_profile = "calendar_cache";
   out.calibration_profile = "calendar_cache";
   out.watchlist_tags_csv = "mt5_calendar_cache";
   out.caution_lot_scale = calendar_state.caution_lot_scale;
   out.caution_enter_prob_buffer = calendar_state.caution_enter_prob_buffer;
   for(int i=0; i<calendar_state.reason_count; i++)
      FXAI_NewsPulseAppendReason(out, calendar_state.reasons[i]);
   FXAI_NewsPulseAppendReason(out, "calendar_cache_fallback");
   return out.ready;
}

bool FXAI_ReadNewsPulsePairState(const string symbol,
                                 FXAINewsPulsePairState &out)
{
   FXAI_ResetNewsPulsePairState(out);
   string pair_id = FXAI_NewsPulsePairId(symbol);
   if(StringLen(pair_id) != 6)
      return false;

   int handle = FileOpen(FXAI_NEWSPULSE_FLAT_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      bool fallback_ok = FXAI_ApplyNewsPulseCalendarFallback(symbol, out);
      if(!fallback_ok)
      {
         FXAI_ResetNewsPulseGlobals();
         return false;
      }
      FXAI_ApplyNewsPulsePairState(out);
      return true;
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
         if(key == "event_eta_min")
            out.event_eta_min = (StringLen(value) > 0 ? (int)StringToInteger(value) : -1);
         else if(key == "news_risk_score")
            out.news_risk_score = StringToDouble(value);
         else if(key == "trade_gate")
            out.trade_gate = value;
         else if(key == "news_pressure")
            out.news_pressure = StringToDouble(value);
         else if(key == "stale")
            out.stale = (StringToInteger(value) != 0);
         else if(key == "session_profile")
            out.session_profile = value;
         else if(key == "calibration_profile")
            out.calibration_profile = value;
         else if(key == "watchlist_tags")
            out.watchlist_tags_csv = value;
         else if(key == "caution_lot_scale")
            out.caution_lot_scale = StringToDouble(value);
         else if(key == "caution_enter_prob_buffer")
            out.caution_enter_prob_buffer = StringToDouble(value);
         continue;
      }

      if(kind == "pair_reason" && target == pair_id)
         FXAI_NewsPulseAppendReason(out, value);
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = FXAI_ServerNow();
      if(now_time > 0 && out.generated_at > 0)
      {
         if((now_time - out.generated_at) > MathMax(NewsPulseFreshnessMaxSec, 60))
            out.stale = true;
      }
      if(StringLen(out.trade_gate) <= 0)
         out.trade_gate = "UNKNOWN";
      if(StringLen(out.session_profile) <= 0)
         out.session_profile = "default";
      if(StringLen(out.calibration_profile) <= 0)
         out.calibration_profile = out.session_profile;
      out.news_risk_score = FXAI_Clamp(out.news_risk_score, 0.0, 1.0);
      out.news_pressure = FXAI_Clamp(out.news_pressure, -1.0, 1.0);
      if(out.caution_lot_scale >= 0.0)
         out.caution_lot_scale = FXAI_Clamp(out.caution_lot_scale, 0.10, 1.0);
      if(out.caution_enter_prob_buffer >= 0.0)
         out.caution_enter_prob_buffer = FXAI_Clamp(out.caution_enter_prob_buffer, 0.0, 0.25);
   }

    if((!out.available || out.stale || StringLen(out.trade_gate) <= 0 || out.trade_gate == "UNKNOWN") &&
       FXAI_ApplyNewsPulseCalendarFallback(symbol, out))
    {
       out.available = true;
       out.ready = true;
    }

   FXAI_ApplyNewsPulsePairState(out);
   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__
