#ifndef __FXAI_RUNTIME_CALENDAR_CACHE_MQH__
#define __FXAI_RUNTIME_CALENDAR_CACHE_MQH__

#define FXAI_CALENDAR_CACHE_STATE_FILE "FXAI\\Runtime\\news_calendar_state.tsv"
#define FXAI_CALENDAR_CACHE_FEED_FILE  "FXAI\\Runtime\\news_calendar_feed.tsv"
#define FXAI_CALENDAR_CACHE_MAX_REASONS 6

struct FXAICalendarCacheState
{
   bool     ready;
   bool     ok;
   bool     stale;
   datetime last_update_trade_server;
   datetime collector_generated_at;
   int      trade_server_offset_sec;
   int      record_count;
   string   last_error;
};

struct FXAICalendarCachePairState
{
   bool     ready;
   bool     stale;
   datetime generated_at;
   int      trade_server_offset_sec;
   int      next_event_eta_min;
   string   trade_gate;
   double   event_risk_score;
   double   caution_lot_scale;
   double   caution_enter_prob_buffer;
   int      reason_count;
   string   reasons[FXAI_CALENDAR_CACHE_MAX_REASONS];
};

FXAICalendarCacheState g_calendar_cache_state;
bool                   g_calendar_cache_state_ready = false;
datetime               g_calendar_cache_last_load_time = 0;

void FXAI_ResetCalendarCacheState(FXAICalendarCacheState &out)
{
   out.ready = false;
   out.ok = false;
   out.stale = true;
   out.last_update_trade_server = 0;
   out.collector_generated_at = 0;
   out.trade_server_offset_sec = 0;
   out.record_count = 0;
   out.last_error = "";
}

void FXAI_ResetCalendarCachePairState(FXAICalendarCachePairState &out)
{
   out.ready = false;
   out.stale = true;
   out.generated_at = 0;
   out.trade_server_offset_sec = 0;
   out.next_event_eta_min = -1;
   out.trade_gate = "UNKNOWN";
   out.event_risk_score = 0.0;
   out.caution_lot_scale = 1.0;
   out.caution_enter_prob_buffer = 0.0;
   out.reason_count = 0;
   for(int i=0; i<FXAI_CALENDAR_CACHE_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_CalendarCacheAppendReason(FXAICalendarCachePairState &state,
                                    const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_CALENDAR_CACHE_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_CalendarCacheReasonsCSV(const FXAICalendarCachePairState &state)
{
   string csv = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(csv) > 0)
         csv += "; ";
      csv += state.reasons[i];
   }
   return csv;
}

int FXAI_CalendarEventClassFromTitle(const string raw_title)
{
   string title = raw_title;
   StringToLower(title);
   if(StringFind(title, "rate") >= 0 || StringFind(title, "fomc") >= 0 ||
      StringFind(title, "ecb") >= 0 || StringFind(title, "boe") >= 0 ||
      StringFind(title, "boj") >= 0 || StringFind(title, "rba") >= 0 ||
      StringFind(title, "rbnz") >= 0 || StringFind(title, "boc") >= 0 ||
      StringFind(title, "snb") >= 0)
      return 1;
   if(StringFind(title, "cpi") >= 0 || StringFind(title, "ppi") >= 0 ||
      StringFind(title, "pce") >= 0 || StringFind(title, "inflation") >= 0 ||
      StringFind(title, "price") >= 0)
      return 2;
   if(StringFind(title, "payroll") >= 0 || StringFind(title, "employment") >= 0 ||
      StringFind(title, "job") >= 0 || StringFind(title, "wage") >= 0 ||
      StringFind(title, "unemployment") >= 0)
      return 3;
   if(StringFind(title, "gdp") >= 0 || StringFind(title, "pmi") >= 0 ||
      StringFind(title, "retail") >= 0 || StringFind(title, "production") >= 0 ||
      StringFind(title, "confidence") >= 0 || StringFind(title, "sentiment") >= 0)
      return 4;
   if(StringFind(title, "speech") >= 0 || StringFind(title, "testimony") >= 0)
      return 5;
   return 0;
}

double FXAI_CalendarEventImportanceWeight(const int importance)
{
   if(importance >= 3)
      return 1.0;
   if(importance == 2)
      return 0.60;
   if(importance == 1)
      return 0.30;
   return 0.10;
}

datetime FXAI_CalendarCacheParseTime(const string raw_value)
{
   string value = raw_value;
   if(StringLen(value) <= 0)
      return 0;
   StringReplace(value, "T", " ");
   StringReplace(value, "Z", "");
   StringReplace(value, "-", ".");
   return StringToTime(value);
}

bool FXAI_LoadCalendarCacheState(FXAICalendarCacheState &out,
                                 const bool force_reload = false)
{
   datetime now_time = FXAI_ServerNow();
   if(!force_reload &&
      g_calendar_cache_state_ready &&
      g_calendar_cache_last_load_time > 0 &&
      now_time > 0 &&
      (now_time - g_calendar_cache_last_load_time) < 30)
   {
      out = g_calendar_cache_state;
      return out.ready;
   }

   FXAI_ResetCalendarCacheState(out);
   int handle = FileOpen(FXAI_CALENDAR_CACHE_STATE_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;

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
      if(key == "ok")
         out.ok = (StringToInteger(value) != 0);
      else if(key == "stale")
         out.stale = (StringToInteger(value) != 0);
      else if(key == "last_update_trade_server")
         out.last_update_trade_server = FXAI_CalendarCacheParseTime(value);
      else if(key == "collector_generated_at")
         out.collector_generated_at = FXAI_CalendarCacheParseTime(value);
      else if(key == "trade_server_offset_sec")
         out.trade_server_offset_sec = (int)StringToInteger(value);
      else if(key == "record_count")
         out.record_count = (int)StringToInteger(value);
      else if(key == "last_error")
         out.last_error = value;
   }
   FileClose(handle);

   out.ready = (out.ok || out.record_count > 0 || StringLen(out.last_error) > 0);
   g_calendar_cache_state = out;
   g_calendar_cache_state_ready = out.ready;
   g_calendar_cache_last_load_time = now_time;
   return out.ready;
}

bool FXAI_CalendarEventAffectsSymbol(const string symbol,
                                     const string currency)
{
   string base = "";
   string quote = "";
   FXAI_ParseSymbolLegs(symbol, base, quote);
   if(StringLen(base) != 3 || StringLen(quote) != 3 || StringLen(currency) != 3)
      return false;
   return (base == currency || quote == currency);
}

bool FXAI_ReadCalendarCachePairState(const string symbol,
                                     FXAICalendarCachePairState &out)
{
   FXAI_ResetCalendarCachePairState(out);

   FXAICalendarCacheState state;
   if(!FXAI_LoadCalendarCacheState(state))
      return false;

   out.ready = state.ready;
   out.stale = state.stale;
   out.generated_at = state.last_update_trade_server;
   out.trade_server_offset_sec = state.trade_server_offset_sec;

   int handle = FileOpen(FXAI_CALENDAR_CACHE_FEED_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return out.ready;

   datetime now_time = FXAI_ServerNow();
   if(now_time <= 0)
      now_time = state.last_update_trade_server;
   if(now_time <= 0)
      now_time = TimeCurrent();

   bool header_seen = false;
   int event_count = 0;
   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      if(!header_seen)
      {
         header_seen = true;
         continue;
      }

      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 8)
         continue;

      string currency = parts[5];
      StringToUpper(currency);
      if(!FXAI_CalendarEventAffectsSymbol(symbol, currency))
         continue;

      datetime event_time = (datetime)StringToInteger(parts[6]);
      int importance = (int)StringToInteger(parts[7]);
      string title = (n > 2 ? parts[2] : "");
      int event_class = FXAI_CalendarEventClassFromTitle(title);
      double importance_w = FXAI_CalendarEventImportanceWeight(importance);
      int eta_min = (int)((event_time - now_time) / 60);
      if(eta_min < out.next_event_eta_min || out.next_event_eta_min < 0)
         out.next_event_eta_min = eta_min;

      if(MathAbs(eta_min) > 360)
         continue;

      double event_weight = importance_w;
      if(event_class == 1)
         event_weight *= 1.20;
      else if(event_class == 2 || event_class == 3)
         event_weight *= 1.05;
      else if(event_class == 5)
         event_weight *= 0.65;

      if(eta_min >= -20 && eta_min <= 30)
      {
         out.trade_gate = "BLOCK";
         out.event_risk_score = MathMax(out.event_risk_score, FXAI_Clamp(event_weight, 0.0, 1.0));
         FXAI_CalendarCacheAppendReason(out, "calendar_blackout");
      }
      else if(eta_min >= -90 && eta_min <= 90)
      {
         if(out.trade_gate != "BLOCK")
            out.trade_gate = "CAUTION";
         out.event_risk_score = MathMax(out.event_risk_score, FXAI_Clamp(0.70 * event_weight, 0.0, 1.0));
         FXAI_CalendarCacheAppendReason(out, "calendar_caution");
      }

      if(event_class == 1)
         FXAI_CalendarCacheAppendReason(out, "calendar_central_bank");
      else if(event_class == 2)
         FXAI_CalendarCacheAppendReason(out, "calendar_inflation");
      else if(event_class == 3)
         FXAI_CalendarCacheAppendReason(out, "calendar_labor");

      event_count++;
   }
   FileClose(handle);

   if(out.next_event_eta_min < 0 && out.trade_gate == "UNKNOWN")
      out.trade_gate = "SAFE";
   if(StringLen(out.trade_gate) <= 0)
      out.trade_gate = "UNKNOWN";

   if(out.trade_gate == "BLOCK")
   {
      out.caution_lot_scale = 0.0;
      out.caution_enter_prob_buffer = 0.10;
   }
   else if(out.trade_gate == "CAUTION")
   {
      out.caution_lot_scale = 0.55;
      out.caution_enter_prob_buffer = 0.05;
   }
   else
   {
      out.caution_lot_scale = 1.0;
      out.caution_enter_prob_buffer = 0.0;
      out.event_risk_score = FXAI_Clamp(out.event_risk_score, 0.0, 0.35);
   }

   if(state.last_update_trade_server > 0 && now_time > state.last_update_trade_server &&
      (now_time - state.last_update_trade_server) > MathMax(NewsPulseFreshnessMaxSec, 600))
      out.stale = true;

   if(event_count <= 0 && !state.ok && StringLen(state.last_error) > 0)
      FXAI_CalendarCacheAppendReason(out, "calendar_state_error");

   return out.ready;
}

#endif // __FXAI_RUNTIME_CALENDAR_CACHE_MQH__
