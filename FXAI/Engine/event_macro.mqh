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

FXAIMacroEventRecord g_macro_events[];
bool g_macro_events_loaded = false;
bool g_macro_events_available = false;

string FXAI_StripUtfBom(const string raw_value)
{
   string v = raw_value;
   if(StringLen(v) > 0 && StringGetCharacter(v, 0) == 65279)
      v = StringSubstr(v, 1);
   return v;
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
}

bool FXAI_EnsureMacroEventStoreLoaded(void)
{
   if(g_macro_events_loaded)
      return g_macro_events_available;

   g_macro_events_loaded = true;
   g_macro_events_available = false;
   ArrayResize(g_macro_events, 0);

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
         continue;

      int idx = ArraySize(g_macro_events);
      ArrayResize(g_macro_events, idx + 1);
      g_macro_events[idx].symbol = symbol_trim;
      g_macro_events[idx].event_time = event_time;
      g_macro_events[idx].pre_window_min = MathMax((int)StringToInteger(pre_s), 0);
      g_macro_events[idx].post_window_min = MathMax((int)StringToInteger(post_s), 0);
      g_macro_events[idx].importance = FXAI_Clamp(StringToDouble(importance_s), 0.0, 1.0);
      g_macro_events[idx].surprise = FXAI_Clamp(StringToDouble(surprise_s), -6.0, 6.0);
      g_macro_events[idx].actual_delta = FXAI_Clamp(StringToDouble(actual_s), -12.0, 12.0);
      g_macro_events[idx].forecast_delta = FXAI_Clamp(StringToDouble(forecast_s), -12.0, 12.0);
      g_macro_events[idx].event_class = FXAI_ParseMacroEventClass(class_s);
   }

   FileClose(handle);
   g_macro_events_available = (ArraySize(g_macro_events) > 0);
   return g_macro_events_available;
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
      const FXAIMacroEventRecord &ev = g_macro_events[i];
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
