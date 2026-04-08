//+------------------------------------------------------------------+
//|                                      FXAI_NewsPulseCalendar.mq5  |
//| Shared MT5 Economic Calendar collector for the FXAI NewsPulse    |
//| runtime. Installed into MQL5/Services via Offline Lab tooling.   |
//+------------------------------------------------------------------+
#property strict
#property service

#define NP_RUNTIME_DIR              "FXAI\\Runtime"
#define NP_CALENDAR_FEED_FILE       "FXAI\\Runtime\\news_calendar_feed.tsv"
#define NP_CALENDAR_STATE_FILE      "FXAI\\Runtime\\news_calendar_state.tsv"
#define NP_CALENDAR_HISTORY_FILE    "FXAI\\Runtime\\news_calendar_history.ndjson"
#define NP_SCHEMA_VERSION           1
#define NP_POLL_INTERVAL_MS         60000
#define NP_BOOTSTRAP_LOOKBACK_HOURS 36
#define NP_LOOKAHEAD_DAYS           7
#define NP_PRUNE_PAST_HOURS         48
#define NP_MAX_REASONS              6

struct FXAINewsPulseCalendarState
{
   bool ok;
   bool stale;
   ulong change_id;
   datetime last_update_at;
   datetime last_update_trade_server;
   datetime collector_generated_at;
   datetime last_bootstrap_at;
   datetime last_bootstrap_trade_server;
   int trade_server_offset_sec;
   int record_count;
   string last_error;
};

struct FXAINewsPulseCalendarRecord
{
   ulong event_id;
   string event_key;
   string title;
   string country_code;
   string country_name;
   string currency;
   datetime event_time;
   int importance;
   double actual;
   double forecast;
   double previous;
   double revised_previous;
   double surprise_proxy;
   datetime collector_seen_at;
   datetime collector_seen_trade_server;
   datetime event_time_utc;
   int trade_server_offset_sec;
   ulong change_id;
};

FXAINewsPulseCalendarState g_np_state;
FXAINewsPulseCalendarRecord g_np_records[];

string FXAI_NP_JsonEscape(const string value)
{
   string escaped = value;
   StringReplace(escaped, "\\", "\\\\");
   StringReplace(escaped, "\"", "\\\"");
   StringReplace(escaped, "\r", "\\r");
   StringReplace(escaped, "\n", "\\n");
   StringReplace(escaped, "\t", "\\t");
   return escaped;
}

string FXAI_NP_Iso8601Utc(const datetime value)
{
   if(value <= 0)
      return "";
   MqlDateTime parts;
   TimeToStruct(value, parts);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       parts.year,
                       parts.mon,
                       parts.day,
                       parts.hour,
                       parts.min,
                       parts.sec);
}

string FXAI_NP_Iso8601TradeServer(const datetime value)
{
   if(value <= 0)
      return "";
   MqlDateTime parts;
   TimeToStruct(value, parts);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02d",
                       parts.year,
                       parts.mon,
                       parts.day,
                       parts.hour,
                       parts.min,
                       parts.sec);
}

datetime FXAI_NP_ParseIso8601(const string value)
{
   string text = value;
   if(StringLen(text) <= 0)
      return 0;
   StringReplace(text, "T", " ");
   StringReplace(text, "Z", "");
   StringReplace(text, "-", ".");
   return StringToTime(text);
}

int FXAI_NP_TradeServerOffsetSec(const datetime server_now)
{
   datetime gmt_now = TimeGMT();
   if(gmt_now <= 0)
      gmt_now = TimeCurrent();
   return (int)(server_now - gmt_now);
}

datetime FXAI_NP_ToUtcApprox(const datetime server_time,
                             const int offset_sec)
{
   if(server_time <= 0)
      return 0;
   return (server_time - offset_sec);
}

string FXAI_NP_BoolText(const bool value)
{
   return (value ? "1" : "0");
}

double FXAI_NP_NaN(void)
{
   return MathArcsin(2.0);
}

bool FXAI_NP_IsNaN(const double value)
{
   return (value != value);
}

double FXAI_NP_CalendarFieldValue(const long raw_value)
{
   if(raw_value == LONG_MIN)
      return FXAI_NP_NaN();
   return ((double)raw_value / 1000000.0);
}

string FXAI_NP_FieldText(const double value)
{
   if(FXAI_NP_IsNaN(value))
      return "";
   return DoubleToString(value, 6);
}

void FXAI_NP_ResetState(FXAINewsPulseCalendarState &out)
{
   out.ok = false;
   out.stale = true;
   out.change_id = 0;
   out.last_update_at = 0;
   out.last_update_trade_server = 0;
   out.collector_generated_at = 0;
   out.last_bootstrap_at = 0;
   out.last_bootstrap_trade_server = 0;
   out.trade_server_offset_sec = 0;
   out.record_count = 0;
   out.last_error = "";
}

int FXAI_NP_RecordIndexByKey(const string event_key)
{
   for(int i = 0; i < ArraySize(g_np_records); i++)
   {
      if(g_np_records[i].event_key == event_key)
         return i;
   }
   return -1;
}

string FXAI_NP_RecordKey(const ulong event_id,
                         const datetime event_time,
                         const int revision)
{
   return StringFormat("%I64u|%d|%d", event_id, (int)event_time, revision);
}

void FXAI_NP_EnsureFolders(void)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(NP_RUNTIME_DIR, FILE_COMMON);
}

datetime FXAI_NP_Now(void)
{
   datetime now = TimeTradeServer();
   if(now <= 0)
      now = TimeCurrent();
   return now;
}

bool FXAI_NP_LoadState(FXAINewsPulseCalendarState &out)
{
   FXAI_NP_ResetState(out);
   int handle = FileOpen(NP_CALENDAR_STATE_FILE,
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
      int count = StringSplit(line, '\t', parts);
      if(count < 2)
         continue;
      string key = parts[0];
      string value = parts[1];
      if(key == "ok")
         out.ok = (StringToInteger(value) != 0);
      else if(key == "stale")
         out.stale = (StringToInteger(value) != 0);
      else if(key == "change_id")
         out.change_id = (ulong)StringToInteger(value);
      else if(key == "last_update_at")
         out.last_update_at = FXAI_NP_ParseIso8601(value);
      else if(key == "last_update_trade_server")
         out.last_update_trade_server = FXAI_NP_ParseIso8601(value);
      else if(key == "collector_generated_at")
         out.collector_generated_at = FXAI_NP_ParseIso8601(value);
      else if(key == "last_bootstrap_at")
         out.last_bootstrap_at = FXAI_NP_ParseIso8601(value);
      else if(key == "last_bootstrap_trade_server")
         out.last_bootstrap_trade_server = FXAI_NP_ParseIso8601(value);
      else if(key == "trade_server_offset_sec")
         out.trade_server_offset_sec = (int)StringToInteger(value);
      else if(key == "record_count")
         out.record_count = (int)StringToInteger(value);
      else if(key == "last_error")
         out.last_error = value;
   }
   FileClose(handle);
   return true;
}

void FXAI_NP_WriteState(const FXAINewsPulseCalendarState &state)
{
   FXAI_NP_EnsureFolders();
   int handle = FileOpen(NP_CALENDAR_STATE_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle, "schema_version\t" + IntegerToString(NP_SCHEMA_VERSION) + "\r\n");
   FileWriteString(handle, "ok\t" + FXAI_NP_BoolText(state.ok) + "\r\n");
   FileWriteString(handle, "stale\t" + FXAI_NP_BoolText(state.stale) + "\r\n");
    FileWriteString(handle, "time_basis\ttrade_server\r\n");
   FileWriteString(handle, "trade_server_offset_sec\t" + IntegerToString(state.trade_server_offset_sec) + "\r\n");
   FileWriteString(handle, "change_id\t" + StringFormat("%I64u", state.change_id) + "\r\n");
   FileWriteString(handle, "last_update_at\t" + FXAI_NP_Iso8601Utc(state.last_update_at) + "\r\n");
   FileWriteString(handle, "last_update_trade_server\t" + FXAI_NP_Iso8601TradeServer(state.last_update_trade_server) + "\r\n");
   FileWriteString(handle, "collector_generated_at\t" + FXAI_NP_Iso8601Utc(state.collector_generated_at) + "\r\n");
   FileWriteString(handle, "last_bootstrap_at\t" + FXAI_NP_Iso8601Utc(state.last_bootstrap_at) + "\r\n");
   FileWriteString(handle, "last_bootstrap_trade_server\t" + FXAI_NP_Iso8601TradeServer(state.last_bootstrap_trade_server) + "\r\n");
   FileWriteString(handle, "record_count\t" + IntegerToString(state.record_count) + "\r\n");
   FileWriteString(handle, "last_error\t" + state.last_error + "\r\n");
   FileClose(handle);
}

bool FXAI_NP_LoadFeed(void)
{
   ArrayResize(g_np_records, 0);
   int handle = FileOpen(NP_CALENDAR_FEED_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;

   bool header_seen = false;
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
      int count = StringSplit(line, '\t', parts);
      if(count < 15)
         continue;
      int idx = ArraySize(g_np_records);
      ArrayResize(g_np_records, idx + 1);
      g_np_records[idx].event_id = (ulong)StringToInteger(parts[0]);
      g_np_records[idx].event_key = parts[1];
      g_np_records[idx].title = parts[2];
      g_np_records[idx].country_code = parts[3];
      g_np_records[idx].country_name = parts[4];
      g_np_records[idx].currency = parts[5];
      g_np_records[idx].event_time = (datetime)StringToInteger(parts[6]);
      g_np_records[idx].importance = (int)StringToInteger(parts[7]);
      g_np_records[idx].actual = (StringLen(parts[8]) > 0 ? StringToDouble(parts[8]) : FXAI_NP_NaN());
      g_np_records[idx].forecast = (StringLen(parts[9]) > 0 ? StringToDouble(parts[9]) : FXAI_NP_NaN());
      g_np_records[idx].previous = (StringLen(parts[10]) > 0 ? StringToDouble(parts[10]) : FXAI_NP_NaN());
      g_np_records[idx].revised_previous = (StringLen(parts[11]) > 0 ? StringToDouble(parts[11]) : FXAI_NP_NaN());
      g_np_records[idx].surprise_proxy = (StringLen(parts[12]) > 0 ? StringToDouble(parts[12]) : FXAI_NP_NaN());
      g_np_records[idx].collector_seen_at = (datetime)StringToInteger(parts[13]);
      g_np_records[idx].change_id = (ulong)StringToInteger(parts[14]);
      g_np_records[idx].collector_seen_trade_server = g_np_records[idx].collector_seen_at;
      g_np_records[idx].event_time_utc = g_np_records[idx].event_time;
      g_np_records[idx].trade_server_offset_sec = 0;
      if(count >= 20)
      {
         g_np_records[idx].collector_seen_trade_server = FXAI_NP_ParseIso8601(parts[16]);
         g_np_records[idx].event_time_utc = (datetime)StringToInteger(parts[17]);
         g_np_records[idx].collector_seen_at = (datetime)StringToInteger(parts[18]);
         g_np_records[idx].trade_server_offset_sec = (int)StringToInteger(parts[19]);
      }
   }
   FileClose(handle);
   return (ArraySize(g_np_records) > 0);
}

void FXAI_NP_WriteFeed(void)
{
   FXAI_NP_EnsureFolders();
   int handle = FileOpen(NP_CALENDAR_FEED_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle,
                   "event_id\tevent_key\ttitle\tcountry_code\tcountry_name\tcurrency\t"
                   "event_time_unix\timportance\tactual\tforecast\tprevious\trevised_previous\t"
                   "surprise_proxy\tcollector_seen_unix\tchange_id\t"
                   "event_time_trade_server\tcollector_seen_trade_server\t"
                   "event_time_utc_unix\tcollector_seen_utc_unix\ttrade_server_offset_sec\r\n");
   for(int i = 0; i < ArraySize(g_np_records); i++)
   {
      FXAINewsPulseCalendarRecord record = g_np_records[i];
      string line =
         StringFormat("%I64u", record.event_id) + "\t" +
         record.event_key + "\t" +
         record.title + "\t" +
         record.country_code + "\t" +
         record.country_name + "\t" +
         record.currency + "\t" +
         IntegerToString((int)record.event_time) + "\t" +
         IntegerToString(record.importance) + "\t" +
         FXAI_NP_FieldText(record.actual) + "\t" +
         FXAI_NP_FieldText(record.forecast) + "\t" +
         FXAI_NP_FieldText(record.previous) + "\t" +
         FXAI_NP_FieldText(record.revised_previous) + "\t" +
         FXAI_NP_FieldText(record.surprise_proxy) + "\t" +
         IntegerToString((int)record.collector_seen_trade_server) + "\t" +
         StringFormat("%I64u", record.change_id) + "\t" +
         FXAI_NP_Iso8601TradeServer(record.event_time) + "\t" +
         FXAI_NP_Iso8601TradeServer(record.collector_seen_trade_server) + "\t" +
         IntegerToString((int)record.event_time_utc) + "\t" +
         IntegerToString((int)record.collector_seen_at) + "\t" +
         IntegerToString(record.trade_server_offset_sec) + "\r\n";
      FileWriteString(handle, line);
   }
   FileClose(handle);
}

void FXAI_NP_AppendHistory(const FXAINewsPulseCalendarRecord &record,
                           const datetime collector_generated_at_utc,
                           const datetime collector_generated_trade_server)
{
   FXAI_NP_EnsureFolders();
   int handle = FileOpen(NP_CALENDAR_HISTORY_FILE,
                         FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      handle = FileOpen(NP_CALENDAR_HISTORY_FILE,
                        FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                        FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;
   FileSeek(handle, 0, SEEK_END);

   string json =
      "{" +
      "\"record_type\":\"calendar_value\"," +
      "\"time_basis\":\"trade_server\"," +
      "\"collector_generated_at\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601Utc(collector_generated_at_utc)) + "\"," +
      "\"collector_generated_trade_server\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601TradeServer(collector_generated_trade_server)) + "\"," +
      "\"event_id\":\"" + FXAI_NP_JsonEscape(StringFormat("%I64u", record.event_id)) + "\"," +
      "\"event_key\":\"" + FXAI_NP_JsonEscape(record.event_key) + "\"," +
      "\"currency\":\"" + FXAI_NP_JsonEscape(record.currency) + "\"," +
      "\"country_code\":\"" + FXAI_NP_JsonEscape(record.country_code) + "\"," +
      "\"title\":\"" + FXAI_NP_JsonEscape(record.title) + "\"," +
      "\"event_time\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601Utc(record.event_time_utc)) + "\"," +
      "\"event_time_trade_server\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601TradeServer(record.event_time)) + "\"," +
      "\"importance\":" + IntegerToString(record.importance) + "," +
      "\"actual\":\"" + FXAI_NP_JsonEscape(FXAI_NP_FieldText(record.actual)) + "\"," +
      "\"forecast\":\"" + FXAI_NP_JsonEscape(FXAI_NP_FieldText(record.forecast)) + "\"," +
      "\"previous\":\"" + FXAI_NP_JsonEscape(FXAI_NP_FieldText(record.previous)) + "\"," +
      "\"revised_previous\":\"" + FXAI_NP_JsonEscape(FXAI_NP_FieldText(record.revised_previous)) + "\"," +
      "\"surprise_proxy\":\"" + FXAI_NP_JsonEscape(FXAI_NP_FieldText(record.surprise_proxy)) + "\"," +
      "\"collector_seen_at\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601Utc(record.collector_seen_at)) + "\"," +
      "\"collector_seen_trade_server\":\"" + FXAI_NP_JsonEscape(FXAI_NP_Iso8601TradeServer(record.collector_seen_trade_server)) + "\"," +
      "\"trade_server_offset_sec\":" + IntegerToString(record.trade_server_offset_sec) + "," +
      "\"change_id\":\"" + FXAI_NP_JsonEscape(StringFormat("%I64u", record.change_id)) + "\"" +
      "}";
   FileWriteString(handle, json + "\n");
   FileClose(handle);
}

double FXAI_NP_SurpriseProxy(const double actual,
                             const double forecast,
                             const double previous)
{
   if(FXAI_NP_IsNaN(actual))
      return FXAI_NP_NaN();

   double anchor = FXAI_NP_NaN();
   if(!FXAI_NP_IsNaN(forecast))
      anchor = forecast;
   else if(!FXAI_NP_IsNaN(previous))
      anchor = previous;

   if(FXAI_NP_IsNaN(anchor))
      return 0.0;

   double denom = MathMax(MathAbs(anchor), 1e-6);
   return ((actual - anchor) / denom);
}

void FXAI_NP_ApplyRecord(const FXAINewsPulseCalendarRecord &record)
{
   int idx = FXAI_NP_RecordIndexByKey(record.event_key);
   if(idx >= 0)
      g_np_records[idx] = record;
   else
   {
      int total = ArraySize(g_np_records);
      ArrayResize(g_np_records, total + 1);
      g_np_records[total] = record;
   }
}

void FXAI_NP_PruneRecords(const datetime now_time)
{
   datetime min_time = now_time - NP_PRUNE_PAST_HOURS * 3600;
   datetime max_time = now_time + NP_LOOKAHEAD_DAYS * 86400;
   FXAINewsPulseCalendarRecord filtered[];
   ArrayResize(filtered, 0);
   for(int i = 0; i < ArraySize(g_np_records); i++)
   {
      if(g_np_records[i].event_time < min_time || g_np_records[i].event_time > max_time)
         continue;
      int idx = ArraySize(filtered);
      ArrayResize(filtered, idx + 1);
      filtered[idx] = g_np_records[i];
   }
   ArrayResize(g_np_records, ArraySize(filtered));
   for(int i = 0; i < ArraySize(filtered); i++)
      g_np_records[i] = filtered[i];
}

bool FXAI_NP_RecordFromValue(const MqlCalendarValue &value,
                             const ulong cursor_change_id,
                             const datetime collector_seen_trade_server,
                             const int trade_server_offset_sec,
                             FXAINewsPulseCalendarRecord &out)
{
   MqlCalendarEvent event;
   if(!CalendarEventById(value.event_id, event))
      return false;

   MqlCalendarCountry country;
   if(!CalendarCountryById((long)event.country_id, country))
   {
      country.code = "";
      country.name = "";
      country.currency = "";
   }

   string currency = country.currency;
   StringToUpper(currency);
   if(StringLen(currency) != 3)
      return false;

   double actual = FXAI_NP_CalendarFieldValue(value.actual_value);
   double forecast = FXAI_NP_CalendarFieldValue(value.forecast_value);
   double previous = FXAI_NP_CalendarFieldValue(value.prev_value);
   double revised_previous = FXAI_NP_CalendarFieldValue(value.revised_prev_value);

   out.event_id = value.event_id;
   out.event_key = FXAI_NP_RecordKey(value.event_id, value.time, value.revision);
   out.title = event.name;
   out.country_code = country.code;
   out.country_name = country.name;
   out.currency = currency;
   out.event_time = value.time;
   out.importance = (int)event.importance;
   out.actual = actual;
   out.forecast = forecast;
   out.previous = previous;
   out.revised_previous = revised_previous;
   out.surprise_proxy = FXAI_NP_SurpriseProxy(actual, forecast, previous);
   out.collector_seen_trade_server = collector_seen_trade_server;
   out.collector_seen_at = FXAI_NP_ToUtcApprox(collector_seen_trade_server, trade_server_offset_sec);
   out.event_time_utc = FXAI_NP_ToUtcApprox(value.time, trade_server_offset_sec);
   out.trade_server_offset_sec = trade_server_offset_sec;
   out.change_id = cursor_change_id;
   return true;
}

void FXAI_NP_ProcessValues(MqlCalendarValue &values[],
                           const ulong cursor_change_id,
                           const datetime collector_generated_at_utc,
                           const datetime collector_generated_trade_server,
                           const int trade_server_offset_sec)
{
   for(int i = 0; i < ArraySize(values); i++)
   {
      FXAINewsPulseCalendarRecord record;
      if(!FXAI_NP_RecordFromValue(values[i],
                                  cursor_change_id,
                                  collector_generated_trade_server,
                                  trade_server_offset_sec,
                                  record))
         continue;
      FXAI_NP_ApplyRecord(record);
      FXAI_NP_AppendHistory(record, collector_generated_at_utc, collector_generated_trade_server);
   }
}

bool FXAI_NP_Bootstrap(FXAINewsPulseCalendarState &state,
                       const datetime now_time_trade_server,
                       const datetime now_time_utc,
                       const int trade_server_offset_sec)
{
   MqlCalendarValue values[];
   datetime from_time = now_time_trade_server - NP_BOOTSTRAP_LOOKBACK_HOURS * 3600;
   datetime to_time = now_time_trade_server + NP_LOOKAHEAD_DAYS * 86400;
   ResetLastError();
   int total = CalendarValueHistory(values, from_time, to_time);
   if(total < 0)
   {
      state.last_error = "calendar_history_error_" + IntegerToString(GetLastError());
      return false;
   }

   ArrayResize(g_np_records, 0);
   FXAI_NP_ProcessValues(values, 0, now_time_utc, now_time_trade_server, trade_server_offset_sec);
   state.last_bootstrap_at = now_time_utc;
   state.last_bootstrap_trade_server = now_time_trade_server;
   state.trade_server_offset_sec = trade_server_offset_sec;

   ulong bootstrap_cursor = 0;
   MqlCalendarValue cursor_values[];
   ResetLastError();
   int cursor_total = CalendarValueLast(bootstrap_cursor, cursor_values);
   if(cursor_total >= 0 && bootstrap_cursor > 0)
      state.change_id = bootstrap_cursor;
   else if(cursor_total < 0)
      state.last_error = "calendar_last_sync_error_" + IntegerToString(GetLastError());

   return true;
}

bool FXAI_NP_PollIncremental(FXAINewsPulseCalendarState &state,
                             const datetime now_time_trade_server,
                             const datetime now_time_utc,
                             const int trade_server_offset_sec)
{
   if(state.change_id == 0)
      return FXAI_NP_Bootstrap(state, now_time_trade_server, now_time_utc, trade_server_offset_sec);

   MqlCalendarValue values[];
   ulong cursor = state.change_id;
   ResetLastError();
   int total = CalendarValueLast(cursor, values);
   if(total < 0)
   {
      state.last_error = "calendar_last_error_" + IntegerToString(GetLastError());
      return false;
   }

   state.change_id = cursor;
   if(total > 0)
      FXAI_NP_ProcessValues(values, cursor, now_time_utc, now_time_trade_server, trade_server_offset_sec);
   state.trade_server_offset_sec = trade_server_offset_sec;
   return true;
}

void FXAI_NP_MarkFailure(FXAINewsPulseCalendarState &state,
                         const string error_text,
                         const datetime now_time_trade_server,
                         const datetime now_time_utc,
                         const int trade_server_offset_sec)
{
   state.ok = false;
   state.stale = true;
   state.last_error = error_text;
   state.collector_generated_at = now_time_utc;
   state.last_update_trade_server = now_time_trade_server;
   state.trade_server_offset_sec = trade_server_offset_sec;
   FXAI_NP_WriteState(state);
}

bool FXAI_NP_RunCollectionCycle(FXAINewsPulseCalendarState &state)
{
   datetime now_time_trade_server = FXAI_NP_Now();
   datetime now_time_utc = TimeGMT();
   if(now_time_utc <= 0)
      now_time_utc = TimeCurrent();
   int trade_server_offset_sec = FXAI_NP_TradeServerOffsetSec(now_time_trade_server);
   bool need_bootstrap = (state.change_id == 0 || ArraySize(g_np_records) <= 0);
   bool ok = (need_bootstrap ? FXAI_NP_Bootstrap(state, now_time_trade_server, now_time_utc, trade_server_offset_sec)
                             : FXAI_NP_PollIncremental(state, now_time_trade_server, now_time_utc, trade_server_offset_sec));
   if(!ok)
      return false;

   FXAI_NP_PruneRecords(now_time_trade_server);
   state.ok = true;
   state.stale = false;
   state.last_update_at = now_time_utc;
   state.last_update_trade_server = now_time_trade_server;
   state.collector_generated_at = now_time_utc;
   state.trade_server_offset_sec = trade_server_offset_sec;
   state.record_count = ArraySize(g_np_records);
   state.last_error = "";
   FXAI_NP_WriteFeed();
   FXAI_NP_WriteState(state);
   return true;
}

void OnStart(void)
{
   FXAI_NP_EnsureFolders();
   FXAI_NP_ResetState(g_np_state);
   FXAI_NP_LoadState(g_np_state);
   FXAI_NP_LoadFeed();

   while(!IsStopped())
   {
      datetime cycle_time_trade_server = FXAI_NP_Now();
      datetime cycle_time_utc = TimeGMT();
      if(cycle_time_utc <= 0)
         cycle_time_utc = TimeCurrent();
      int trade_server_offset_sec = FXAI_NP_TradeServerOffsetSec(cycle_time_trade_server);
      if(!FXAI_NP_RunCollectionCycle(g_np_state))
      {
         string error_text = g_np_state.last_error;
         if(StringLen(error_text) <= 0)
            error_text = "collection_failed";
         FXAI_NP_MarkFailure(g_np_state,
                             error_text,
                             cycle_time_trade_server,
                             cycle_time_utc,
                             trade_server_offset_sec);
         PrintFormat("FXAI NewsPulse calendar collector failed: %s", error_text);
      }
      Sleep(NP_POLL_INTERVAL_MS);
   }
}
