#ifndef __FXAI_EVENT_MACRO_MQH__
#define __FXAI_EVENT_MACRO_MQH__

#define FXAI_MACRO_EVENT_FILE "FXAI\\Runtime\\macro_events.tsv"

struct FXAIMacroEventRecord
{
   string symbol;
   datetime event_time;
   int pre_window_min;
   int post_window_min;
   double importance;
   double surprise;
   double actual_delta;
   double forecast_delta;
   int event_class;
};

struct FXAIMacroEventDatasetStats
{
   int record_count;
   int parse_errors;
   int distinct_symbols;
   datetime first_event_time;
   datetime last_event_time;
   double avg_importance;
   double avg_pre_window_min;
   double avg_post_window_min;
   double checksum01;
   double leakage_guard_score;
};

FXAIMacroEventRecord g_macro_events[];
bool g_macro_events_loaded = false;
bool g_macro_events_available = false;
FXAIMacroEventDatasetStats g_macro_event_stats;

string FXAI_StripUtfBom(const string raw_value)
{
   string v = raw_value;
   if(StringLen(v) > 0 && StringGetCharacter(v, 0) == 65279)
      v = StringSubstr(v, 1);
   return v;
}

void FXAI_ClearMacroEventDatasetStats(FXAIMacroEventDatasetStats &stats)
{
   stats.record_count = 0;
   stats.parse_errors = 0;
   stats.distinct_symbols = 0;
   stats.first_event_time = 0;
   stats.last_event_time = 0;
   stats.avg_importance = 0.0;
   stats.avg_pre_window_min = 0.0;
   stats.avg_post_window_min = 0.0;
   stats.checksum01 = 0.0;
   stats.leakage_guard_score = 1.0;
}

double FXAI_MacroEventStringChecksum01(const string raw_value)
{
   string v = FXAI_StripUtfBom(raw_value);
   StringTrimLeft(v);
   StringTrimRight(v);
   StringToUpper(v);
   if(StringLen(v) <= 0)
      return 0.0;

   double acc = 0.0;
   double scale = 1.0;
   int n = StringLen(v);
   for(int i=0; i<n; i++)
   {
      int ch = StringGetCharacter(v, i);
      if(ch < 0)
         ch = 0;
      acc += scale * (double)((ch % 97) + 1);
      scale *= 1.131;
      if(scale > 17.0)
         scale = 1.0 + 0.17 * (double)(i + 1);
   }
   double frac = acc / 104729.0;
   return frac - MathFloor(frac);
}

string FXAI_NormalizeMacroTimeToken(const string raw_value)
{
   string v = FXAI_StripUtfBom(raw_value);
   StringTrimLeft(v);
   StringTrimRight(v);
   StringReplace(v, "T", " ");
   StringReplace(v, "-", ".");
   return v;
}

int FXAI_ParseMacroEventClass(const string raw_value)
{
   string v = FXAI_StripUtfBom(raw_value);
   StringTrimLeft(v);
   StringTrimRight(v);
   string lower = v;
   StringToLower(lower);
   int parsed = (int)StringToInteger(v);
   if(IntegerToString(parsed) == v)
      return parsed;
   if(lower == "rates" || lower == "central_bank" || lower == "cb")
      return 1;
   if(lower == "inflation" || lower == "cpi" || lower == "ppi")
      return 2;
   if(lower == "labor" || lower == "employment" || lower == "nfp")
      return 3;
   if(lower == "growth" || lower == "gdp" || lower == "pmi")
      return 4;
   if(lower == "trade" || lower == "balance")
      return 5;
   return 0;
}

double FXAI_MacroEventClassBias(const int event_class)
{
   switch(event_class)
   {
      case 1: return 0.50;
      case 2: return 0.30;
      case 3: return 0.40;
      case 4: return 0.20;
      case 5: return 0.10;
      default: return 0.0;
   }
}

bool FXAI_MacroEventAffectsSymbol(const string raw_event_symbol,
                                  const string raw_symbol)
{
   string event_symbol = FXAI_StripUtfBom(raw_event_symbol);
   StringTrimLeft(event_symbol);
   StringTrimRight(event_symbol);
   StringToUpper(event_symbol);

   string symbol = FXAI_StripUtfBom(raw_symbol);
   StringToUpper(symbol);

   if(StringLen(event_symbol) <= 0 || event_symbol == "ALL" || event_symbol == "*")
      return true;
   if(event_symbol == symbol)
      return true;
   if(StringLen(event_symbol) == 3 && StringFind(symbol, event_symbol) >= 0)
      return true;
   return false;
}

void FXAI_ResetMacroEventStore(void)
{
   ArrayResize(g_macro_events, 0);
   g_macro_events_loaded = false;
   g_macro_events_available = false;
   FXAI_ClearMacroEventDatasetStats(g_macro_event_stats);
}

bool FXAI_EnsureMacroEventStoreLoaded(void)
{
   if(g_macro_events_loaded)
      return g_macro_events_available;

   g_macro_events_loaded = true;
   g_macro_events_available = false;
   ArrayResize(g_macro_events, 0);
   FXAI_ClearMacroEventDatasetStats(g_macro_event_stats);

   int handle = FileOpen(FXAI_MACRO_EVENT_FILE,
                         FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI,
                         '\t');
   if(handle == INVALID_HANDLE)
      return false;

   while(!FileIsEnding(handle))
   {
      string symbol = FileReadString(handle);
      if(FileIsEnding(handle) && StringLen(symbol) <= 0)
         break;

      string event_time_s = FileReadString(handle);
      string pre_s = FileReadString(handle);
      string post_s = FileReadString(handle);
      string importance_s = FileReadString(handle);
      string surprise_s = FileReadString(handle);
      string actual_s = FileReadString(handle);
      string forecast_s = FileReadString(handle);
      string class_s = FileReadString(handle);

      string symbol_trim = FXAI_StripUtfBom(symbol);
      StringTrimLeft(symbol_trim);
      StringTrimRight(symbol_trim);
      string symbol_lower = symbol_trim;
      StringToLower(symbol_lower);
      if(StringLen(symbol_trim) <= 0 || StringGetCharacter(symbol_trim, 0) == '#')
         continue;
      if(symbol_lower == "symbol")
         continue;

      datetime event_time = StringToTime(FXAI_NormalizeMacroTimeToken(event_time_s));
      if(event_time <= 0)
      {
         g_macro_event_stats.parse_errors++;
         continue;
      }

      int pre_window = MathMax((int)StringToInteger(pre_s), 0);
      int post_window = MathMax((int)StringToInteger(post_s), 0);
      if(pre_window <= 0 && post_window <= 0)
      {
         g_macro_event_stats.parse_errors++;
         continue;
      }

      double importance = FXAI_Clamp(StringToDouble(importance_s), 0.0, 1.0);
      double surprise = FXAI_Clamp(StringToDouble(surprise_s), -6.0, 6.0);
      double actual_delta = FXAI_Clamp(StringToDouble(actual_s), -12.0, 12.0);
      double forecast_delta = FXAI_Clamp(StringToDouble(forecast_s), -12.0, 12.0);
      int event_class = FXAI_ParseMacroEventClass(class_s);

      int idx = ArraySize(g_macro_events);
      ArrayResize(g_macro_events, idx + 1);
      g_macro_events[idx].symbol = symbol_trim;
      g_macro_events[idx].event_time = event_time;
      g_macro_events[idx].pre_window_min = pre_window;
      g_macro_events[idx].post_window_min = post_window;
      g_macro_events[idx].importance = importance;
      g_macro_events[idx].surprise = surprise;
      g_macro_events[idx].actual_delta = actual_delta;
      g_macro_events[idx].forecast_delta = forecast_delta;
      g_macro_events[idx].event_class = event_class;

      g_macro_event_stats.record_count++;
      g_macro_event_stats.avg_importance += importance;
      g_macro_event_stats.avg_pre_window_min += (double)pre_window;
      g_macro_event_stats.avg_post_window_min += (double)post_window;
      g_macro_event_stats.checksum01 +=
         0.31 * FXAI_MacroEventStringChecksum01(symbol_trim) +
         0.11 * FXAI_MacroEventStringChecksum01(class_s) +
         0.07 * importance +
         0.03 * MathAbs(surprise);
      if(g_macro_event_stats.first_event_time <= 0 || event_time < g_macro_event_stats.first_event_time)
         g_macro_event_stats.first_event_time = event_time;
      if(g_macro_event_stats.last_event_time <= 0 || event_time > g_macro_event_stats.last_event_time)
         g_macro_event_stats.last_event_time = event_time;

      bool seen_symbol = false;
      for(int j=0; j<idx; j++)
      {
         if(g_macro_events[j].symbol == symbol_trim)
         {
            seen_symbol = true;
            break;
         }
      }
      if(!seen_symbol)
         g_macro_event_stats.distinct_symbols++;
   }

   FileClose(handle);
   g_macro_events_available = (ArraySize(g_macro_events) > 0);
   if(g_macro_event_stats.record_count > 0)
   {
      double denom = (double)g_macro_event_stats.record_count;
      g_macro_event_stats.avg_importance /= denom;
      g_macro_event_stats.avg_pre_window_min /= denom;
      g_macro_event_stats.avg_post_window_min /= denom;
      double frac = g_macro_event_stats.checksum01 / 8192.0;
      g_macro_event_stats.checksum01 = frac - MathFloor(frac);
   }
   else
   {
      FXAI_ClearMacroEventDatasetStats(g_macro_event_stats);
   }
   if(g_macro_event_stats.parse_errors > 0 && g_macro_event_stats.record_count > 0)
      g_macro_event_stats.leakage_guard_score = 0.85;
   return g_macro_events_available;
}

void FXAI_GetMacroEventDatasetStats(FXAIMacroEventDatasetStats &out)
{
   FXAI_ClearMacroEventDatasetStats(out);
   if(!FXAI_EnsureMacroEventStoreLoaded())
      return;
   out = g_macro_event_stats;
}

bool FXAI_HasMacroEventDataset(void)
{
   return FXAI_EnsureMacroEventStoreLoaded();
}

bool FXAI_MacroEventLeakageSafe(void)
{
   FXAIMacroEventDatasetStats stats;
   FXAI_GetMacroEventDatasetStats(stats);
   return (stats.record_count > 0 && stats.leakage_guard_score >= 0.999);
}

double FXAI_MacroEventWindowScoreRates(const string symbol,
                                       const MqlRates &rates_arr[],
                                       const int start_idx,
                                       const int bars)
{
   if(!FXAI_EnsureMacroEventStoreLoaded())
      return 0.0;
   int n = ArraySize(rates_arr);
   if(start_idx < 0 || bars <= 0 || start_idx + bars > n)
      return 0.0;

   int stride = MathMax(1, bars / 24);
   int samples = 0;
   double activity_sum = 0.0;
   double importance_sum = 0.0;
   double surprise_sum = 0.0;
   for(int idx=start_idx; idx<start_idx + bars; idx += stride)
   {
      double pre = 0.0;
      double post = 0.0;
      double importance = 0.0;
      double surprise = 0.0;
      double surprise_abs = 0.0;
      double class_bias = 0.0;
      FXAI_GetMacroEventFeatures(symbol,
                                 rates_arr[idx].time,
                                 pre,
                                 post,
                                 importance,
                                 surprise,
                                 surprise_abs,
                                 class_bias);
      double activity = MathMax(importance, MathMax(pre, post));
      activity_sum += activity;
      importance_sum += importance;
      surprise_sum += surprise_abs;
      samples++;
   }

   if(samples <= 0)
      return 0.0;
   double coverage = activity_sum / (double)samples;
   double importance_mean = importance_sum / (double)samples;
   double surprise_mean = surprise_sum / (double)samples;
   return FXAI_Clamp(0.55 * coverage +
                     0.25 * importance_mean +
                     0.20 * FXAI_Clamp(surprise_mean / 6.0, 0.0, 1.0),
                     0.0,
                     1.0);
}

void FXAI_GetMacroEventFeatures(const string symbol,
                                const datetime sample_time,
                                double &pre_embargo,
                                double &post_embargo,
                                double &event_importance,
                                double &surprise_signed,
                                double &surprise_abs,
                                double &event_class_bias)
{
   pre_embargo = 0.0;
   post_embargo = 0.0;
   event_importance = 0.0;
   surprise_signed = 0.0;
   surprise_abs = 0.0;
   event_class_bias = 0.0;

   if(sample_time <= 0 || !FXAI_EnsureMacroEventStoreLoaded())
      return;

   double signed_weight = 0.0;
   double class_weight = 0.0;
   for(int i=0; i<ArraySize(g_macro_events); i++)
   {
      FXAIMacroEventRecord ev = g_macro_events[i];
      if(!FXAI_MacroEventAffectsSymbol(ev.symbol, symbol))
         continue;

      double dt_minutes = (double)(sample_time - ev.event_time) / 60.0;
      bool in_pre = (dt_minutes < 0.0 && -dt_minutes <= (double)MathMax(ev.pre_window_min, 1));
      bool in_post = (dt_minutes >= 0.0 && dt_minutes <= (double)MathMax(ev.post_window_min, 1));
      if(!in_pre && !in_post)
         continue;

      double base_importance = FXAI_Clamp(ev.importance, 0.0, 1.0);
      double proximity = 0.0;
      if(in_pre)
         proximity = 1.0 - ((-dt_minutes) / (double)MathMax(ev.pre_window_min, 1));
      else
         proximity = 1.0 - (dt_minutes / (double)MathMax(ev.post_window_min, 1));
      proximity = FXAI_Clamp(proximity, 0.0, 1.0);

      double known_weight = FXAI_Clamp(base_importance * (0.35 + 0.65 * proximity), 0.0, 1.0);
      event_importance = MathMax(event_importance, known_weight);
      if(in_pre)
      {
         pre_embargo = MathMax(pre_embargo, known_weight);
         event_class_bias = FXAI_Clamp(event_class_bias + known_weight * FXAI_MacroEventClassBias(ev.event_class),
                                       -2.0,
                                       2.0);
         class_weight += known_weight;
         continue;
      }

      post_embargo = MathMax(post_embargo, known_weight);
      double realized_surprise = FXAI_Clamp(ev.surprise +
                                            0.20 * (ev.actual_delta - ev.forecast_delta),
                                            -6.0,
                                            6.0);
      surprise_signed += known_weight * realized_surprise;
      surprise_abs += known_weight * MathAbs(realized_surprise);
      signed_weight += known_weight;

      event_class_bias = FXAI_Clamp(event_class_bias + known_weight * FXAI_MacroEventClassBias(ev.event_class),
                                    -2.0,
                                    2.0);
      class_weight += known_weight;
   }

   if(signed_weight > 1e-6)
   {
      surprise_signed /= signed_weight;
      surprise_abs /= signed_weight;
   }
   else
   {
      surprise_signed = 0.0;
      surprise_abs = 0.0;
   }

   if(class_weight > 1e-6)
      event_class_bias /= class_weight;
   else
      event_class_bias = 0.0;

   pre_embargo = FXAI_Clamp(pre_embargo, 0.0, 1.0);
   post_embargo = FXAI_Clamp(post_embargo, 0.0, 1.0);
   event_importance = FXAI_Clamp(event_importance, 0.0, 1.0);
   surprise_signed = FXAI_Clamp(surprise_signed, -6.0, 6.0);
   surprise_abs = FXAI_Clamp(surprise_abs, 0.0, 6.0);
   event_class_bias = FXAI_Clamp(event_class_bias, -1.0, 1.0);
}

#endif // __FXAI_EVENT_MACRO_MQH__
