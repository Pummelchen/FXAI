#ifndef __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__
#define __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__

#define FXAI_NEWSPULSE_FLAT_FILE "FXAI\\Runtime\\news_snapshot_flat.tsv"
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
   int reason_count;
   string reasons[FXAI_NEWSPULSE_MAX_REASONS];
};

double g_newspulse_last_risk_score = 0.0;
double g_newspulse_last_news_pressure = 0.0;
int g_newspulse_last_event_eta_min = -1;
bool g_newspulse_last_stale = true;
datetime g_newspulse_last_generated_at = 0;
string g_newspulse_last_trade_gate = "UNKNOWN";
string g_newspulse_last_reasons_csv = "";

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

string FXAI_NewsPulsePairId(const string symbol)
{
   string base = FXAI_RuntimeBaseCurrency(symbol);
   string quote = FXAI_RuntimeQuoteCurrency(symbol);
   if(StringLen(base) != 3 || StringLen(quote) != 3)
      return "";
   return base + quote;
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
      FXAI_ResetNewsPulseGlobals();
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
         continue;
      }

      if(kind == "pair_reason" && target == pair_id)
         FXAI_NewsPulseAppendReason(out, value);
   }
   FileClose(handle);

   if(out.available)
   {
      datetime now_time = TimeCurrent();
      if(now_time <= 0)
         now_time = TimeTradeServer();
      if(now_time > 0 && out.generated_at > 0)
      {
         if((now_time - out.generated_at) > MathMax(NewsPulseFreshnessMaxSec, 60))
            out.stale = true;
      }
      if(StringLen(out.trade_gate) <= 0)
         out.trade_gate = "UNKNOWN";
      out.news_risk_score = FXAI_Clamp(out.news_risk_score, 0.0, 1.0);
      out.news_pressure = FXAI_Clamp(out.news_pressure, -1.0, 1.0);
   }

   FXAI_ApplyNewsPulsePairState(out);
   return out.available;
}

#endif // __FXAI_RUNTIME_TRADE_NEWSPULSE_MQH__
