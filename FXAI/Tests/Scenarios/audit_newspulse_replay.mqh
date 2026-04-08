#ifndef __FXAI_AUDIT_NEWSPULSE_REPLAY_MQH__
#define __FXAI_AUDIT_NEWSPULSE_REPLAY_MQH__

#define FXAI_AUDIT_NEWSPULSE_REPLAY_FILE "FXAI\\Runtime\\news_replay_timeline.tsv"
#define FXAI_AUDIT_NEWSPULSE_SYMBOL_MAP_FILE "FXAI\\Runtime\\news_symbol_map.tsv"

struct FXAINewsPulseReplayRecord
{
   string pair_id;
   datetime observed_at;
   int event_eta_min;
   double news_risk_score;
   double news_pressure;
   bool stale;
   string trade_gate;
};

struct FXAINewsPulseReplaySymbolMapEntry
{
   string symbol;
   string pair_id;
};

FXAINewsPulseReplayRecord g_audit_newspulse_records[];
FXAINewsPulseReplaySymbolMapEntry g_audit_newspulse_symbol_map[];
string g_audit_newspulse_pair_id = "";
bool g_audit_newspulse_loaded = false;
datetime g_audit_newspulse_symbol_map_last_load = 0;

bool FXAI_AuditNewsPulseSupportedCurrency(const string code)
{
   return (code == "USD" || code == "EUR" || code == "GBP" || code == "JPY" ||
           code == "CHF" || code == "CAD" || code == "AUD" || code == "NZD" ||
           code == "SEK" || code == "NOK");
}

string FXAI_AuditNewsPulseAlphaOnly(const string raw_symbol)
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

void FXAI_AuditLoadNewsPulseSymbolMap(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 && g_audit_newspulse_symbol_map_last_load > 0 && (now_time - g_audit_newspulse_symbol_map_last_load) < 60)
      return;

   ArrayResize(g_audit_newspulse_symbol_map, 0);
   g_audit_newspulse_symbol_map_last_load = now_time;

   int handle = FileOpen(FXAI_AUDIT_NEWSPULSE_SYMBOL_MAP_FILE,
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
      int idx = ArraySize(g_audit_newspulse_symbol_map);
      ArrayResize(g_audit_newspulse_symbol_map, idx + 1);
      g_audit_newspulse_symbol_map[idx].symbol = raw_symbol;
      g_audit_newspulse_symbol_map[idx].pair_id = pair_id;
   }
   FileClose(handle);
}

string FXAI_AuditMappedNewsPulsePairId(const string raw_symbol)
{
   FXAI_AuditLoadNewsPulseSymbolMap();
   string symbol = raw_symbol;
   StringToUpper(symbol);
   for(int i = 0; i < ArraySize(g_audit_newspulse_symbol_map); i++)
   {
      if(g_audit_newspulse_symbol_map[i].symbol == symbol)
         return g_audit_newspulse_symbol_map[i].pair_id;
   }
   return "";
}

string FXAI_AuditNewsPulsePairId(const string raw_symbol)
{
   string mapped = FXAI_AuditMappedNewsPulsePairId(raw_symbol);
   if(StringLen(mapped) == 6)
      return mapped;
   string alpha = FXAI_AuditNewsPulseAlphaOnly(raw_symbol);
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
      if(FXAI_AuditNewsPulseSupportedCurrency(base) && FXAI_AuditNewsPulseSupportedCurrency(quote))
         return candidate;
   }
   return "";
}

void FXAI_AuditClearNewsPulseReplay(void)
{
   ArrayResize(g_audit_newspulse_records, 0);
   g_audit_newspulse_pair_id = "";
   g_audit_newspulse_loaded = false;
}

bool FXAI_AuditLoadNewsPulseReplay(const string symbol)
{
   string pair_id = FXAI_AuditNewsPulsePairId(symbol);
   if(StringLen(pair_id) != 6)
      return false;
   if(g_audit_newspulse_loaded && g_audit_newspulse_pair_id == pair_id)
      return (ArraySize(g_audit_newspulse_records) > 0);

   FXAI_AuditClearNewsPulseReplay();
   g_audit_newspulse_pair_id = pair_id;
   g_audit_newspulse_loaded = true;

   int handle = FileOpen(FXAI_AUDIT_NEWSPULSE_REPLAY_FILE,
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
      if(n < 8)
         continue;
      if(parts[0] == "pair_id")
         continue;
      if(parts[0] != pair_id)
         continue;

      int idx = ArraySize(g_audit_newspulse_records);
      ArrayResize(g_audit_newspulse_records, idx + 1);
      g_audit_newspulse_records[idx].pair_id = parts[0];
      g_audit_newspulse_records[idx].observed_at = (datetime)StringToInteger(parts[2]);
      g_audit_newspulse_records[idx].trade_gate = parts[3];
      g_audit_newspulse_records[idx].news_risk_score = StringToDouble(parts[4]);
      g_audit_newspulse_records[idx].news_pressure = StringToDouble(parts[5]);
      g_audit_newspulse_records[idx].stale = (StringToInteger(parts[6]) != 0);
      g_audit_newspulse_records[idx].event_eta_min = (StringLen(parts[7]) > 0 ? (int)StringToInteger(parts[7]) : -1);
   }
   FileClose(handle);
   return (ArraySize(g_audit_newspulse_records) > 0);
}

int FXAI_AuditNewsPulseReplayIndex(const datetime query_time)
{
   int best = -1;
   for(int i = 0; i < ArraySize(g_audit_newspulse_records); i++)
   {
      if(g_audit_newspulse_records[i].observed_at <= query_time)
         best = i;
      else
         break;
   }
   if(best < 0 && ArraySize(g_audit_newspulse_records) > 0)
      best = 0;
   return best;
}

bool FXAI_AuditNewsPulseStateAt(const string symbol,
                                const datetime query_time,
                                FXAINewsPulseReplayRecord &out)
{
   if(!FXAI_AuditLoadNewsPulseReplay(symbol))
      return false;
   int idx = FXAI_AuditNewsPulseReplayIndex(query_time);
   if(idx < 0 || idx >= ArraySize(g_audit_newspulse_records))
      return false;
   out = g_audit_newspulse_records[idx];
   return true;
}

double FXAI_NewsPulseReplayWindowScoreRates(const string symbol,
                                            const MqlRates &rates_m1[],
                                            const int start,
                                            const int bars)
{
   if(bars <= 0 || start < 0 || start + bars - 1 >= ArraySize(rates_m1))
      return 0.0;

   FXAINewsPulseReplayRecord replay_state;
   double best = 0.0;
   int checkpoints[4];
   checkpoints[0] = start;
   checkpoints[1] = start + bars / 3;
   checkpoints[2] = start + (2 * bars) / 3;
   checkpoints[3] = start + bars - 1;

   for(int i = 0; i < 4; i++)
   {
      int idx = checkpoints[i];
      if(idx < start)
         idx = start;
      if(idx > start + bars - 1)
         idx = start + bars - 1;
      if(!FXAI_AuditNewsPulseStateAt(symbol, rates_m1[idx].time, replay_state))
         continue;

      double score = FXAI_Clamp(replay_state.news_risk_score, 0.0, 1.0);
      if(replay_state.stale)
         score = MathMax(score, 0.90);
      if(replay_state.trade_gate == "BLOCK")
         score = MathMax(score, 0.98);
      else if(replay_state.trade_gate == "CAUTION")
         score = MathMax(score, 0.72);
      if(replay_state.event_eta_min >= 0 && replay_state.event_eta_min <= 15)
         score = MathMax(score, 0.84);
      best = MathMax(best, score);
   }
   return FXAI_Clamp(best, 0.0, 1.0);
}

#endif // __FXAI_AUDIT_NEWSPULSE_REPLAY_MQH__
