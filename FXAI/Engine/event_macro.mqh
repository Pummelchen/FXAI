#ifndef __FXAI_EVENT_MACRO_MQH__
#define __FXAI_EVENT_MACRO_MQH__

#define FXAI_MACRO_EVENT_FILE "FXAI\\Runtime\\macro_events.tsv"

struct FXAIMacroEventRecord
{
   string symbol;
   string event_id;
   string country;
   string currency;
   string source;
   datetime event_time;
   int pre_window_min;
   int post_window_min;
   double importance;
   double surprise;
   double actual_delta;
   double forecast_delta;
   double prior_delta;
   double revision_delta;
   double surprise_z;
   double relevance_hint;
   double provenance_hash01;
   int event_class;
};

struct FXAIMacroEventDatasetStats
{
   int record_count;
   int parse_errors;
   int distinct_symbols;
   int distinct_sources;
   int distinct_event_ids;
   int family_rates_count;
   int family_inflation_count;
   int family_labor_count;
   int family_growth_count;
   int family_trade_count;
   datetime first_event_time;
   datetime last_event_time;
   double avg_importance;
   double avg_pre_window_min;
   double avg_post_window_min;
   double avg_surprise_z_abs;
   double avg_revision_abs;
   double checksum01;
   double provenance_hash01;
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
   stats.distinct_sources = 0;
   stats.distinct_event_ids = 0;
   stats.family_rates_count = 0;
   stats.family_inflation_count = 0;
   stats.family_labor_count = 0;
   stats.family_growth_count = 0;
   stats.family_trade_count = 0;
   stats.first_event_time = 0;
   stats.last_event_time = 0;
   stats.avg_importance = 0.0;
   stats.avg_pre_window_min = 0.0;
   stats.avg_post_window_min = 0.0;
   stats.avg_surprise_z_abs = 0.0;
   stats.avg_revision_abs = 0.0;
   stats.checksum01 = 0.0;
   stats.provenance_hash01 = 0.0;
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
   if(lower == "rates" || lower == "central_bank" || lower == "cb" ||
      StringFind(lower, "rate") >= 0 || StringFind(lower, "yield") >= 0 ||
      StringFind(lower, "fomc") >= 0 || StringFind(lower, "ecb") >= 0 ||
      StringFind(lower, "boe") >= 0 || StringFind(lower, "boj") >= 0 ||
      StringFind(lower, "rba") >= 0 || StringFind(lower, "rbnz") >= 0 ||
      StringFind(lower, "boc") >= 0 || StringFind(lower, "snb") >= 0)
      return 1;
   if(lower == "inflation" || lower == "cpi" || lower == "ppi" ||
      StringFind(lower, "pce") >= 0 || StringFind(lower, "hicp") >= 0 ||
      StringFind(lower, "price") >= 0 || StringFind(lower, "deflator") >= 0)
      return 2;
   if(lower == "labor" || lower == "employment" || lower == "nfp" ||
      StringFind(lower, "payroll") >= 0 || StringFind(lower, "unemployment") >= 0 ||
      StringFind(lower, "jobless") >= 0 || StringFind(lower, "wage") >= 0 ||
      StringFind(lower, "earnings") >= 0)
      return 3;
   if(lower == "growth" || lower == "gdp" || lower == "pmi" ||
      StringFind(lower, "manufacturing") >= 0 || StringFind(lower, "services") >= 0 ||
      StringFind(lower, "retail") >= 0 || StringFind(lower, "industrial") >= 0 ||
      StringFind(lower, "production") >= 0 || StringFind(lower, "consumer") >= 0 ||
      StringFind(lower, "confidence") >= 0 || StringFind(lower, "sentiment") >= 0 ||
      StringFind(lower, "housing") >= 0)
      return 4;
   if(lower == "trade" || lower == "balance" ||
      StringFind(lower, "current account") >= 0 ||
      StringFind(lower, "export") >= 0 ||
      StringFind(lower, "import") >= 0)
      return 5;
   return 0;
}

string FXAI_NormalizeMacroToken(const string raw_value)
{
   string v = FXAI_StripUtfBom(raw_value);
   StringTrimLeft(v);
   StringTrimRight(v);
   return v;
}

string FXAI_NormalizeMacroCurrencyToken(const string raw_value)
{
   string v = FXAI_NormalizeMacroToken(raw_value);
   StringToUpper(v);
   if(StringLen(v) > 3)
      v = StringSubstr(v, 0, 3);
   return v;
}

string FXAI_MacroCountryToCurrency(const string raw_country)
{
   string c = FXAI_NormalizeMacroToken(raw_country);
   StringToUpper(c);
   if(c == "US" || c == "USA" || c == "UNITED STATES") return "USD";
   if(c == "EU" || c == "EUR" || c == "EUROZONE") return "EUR";
   if(c == "GB" || c == "UK" || c == "UNITED KINGDOM") return "GBP";
   if(c == "JP" || c == "JAPAN") return "JPY";
   if(c == "AU" || c == "AUSTRALIA") return "AUD";
   if(c == "NZ" || c == "NEW ZEALAND") return "NZD";
   if(c == "CA" || c == "CANADA") return "CAD";
   if(c == "CH" || c == "SWITZERLAND") return "CHF";
   if(c == "CN" || c == "CHINA") return "CNY";
   if(c == "SE" || c == "SWEDEN") return "SEK";
   if(c == "NO" || c == "NORWAY") return "NOK";
   if(c == "DK" || c == "DENMARK") return "DKK";
   if(c == "SG" || c == "SINGAPORE") return "SGD";
   if(c == "HK" || c == "HONG KONG") return "HKD";
   if(c == "MX" || c == "MEXICO") return "MXN";
   if(c == "ZA" || c == "SOUTH AFRICA") return "ZAR";
   return "";
}

int FXAI_MacroCurrencyBlock(const string raw_currency)
{
   string c = FXAI_NormalizeMacroCurrencyToken(raw_currency);
   if(c == "EUR" || c == "CHF" || c == "GBP" || c == "SEK" || c == "NOK" || c == "DKK")
      return 1;
   if(c == "AUD" || c == "NZD" || c == "CAD")
      return 2;
   if(c == "JPY" || c == "CNY" || c == "HKD" || c == "SGD" || c == "KRW")
      return 3;
   if(c == "USD")
      return 4;
   return 0;
}

string FXAI_MacroSymbolBaseCurrency(const string raw_symbol)
{
   string symbol = FXAI_NormalizeMacroToken(raw_symbol);
   StringToUpper(symbol);
   if(StringLen(symbol) < 6)
      return "";
   return StringSubstr(symbol, 0, 3);
}

string FXAI_MacroSymbolQuoteCurrency(const string raw_symbol)
{
   string symbol = FXAI_NormalizeMacroToken(raw_symbol);
   StringToUpper(symbol);
   if(StringLen(symbol) < 6)
      return "";
   return StringSubstr(symbol, 3, 3);
}

double FXAI_MacroSourceTrust(const string raw_source)
{
   string src = FXAI_NormalizeMacroToken(raw_source);
   string lower = src;
   StringToLower(lower);
   if(StringLen(src) <= 0)
      return 0.45;
   if(StringFind(lower, "official") >= 0 ||
      StringFind(lower, "central bank") >= 0 ||
      StringFind(lower, "statistics") >= 0 ||
      StringFind(lower, "bureau") >= 0 ||
      StringFind(lower, "ministry") >= 0 ||
      StringFind(lower, "government") >= 0)
      return 1.0;
   if(StringFind(lower, "reuters") >= 0 ||
      StringFind(lower, "bloomberg") >= 0 ||
      StringFind(lower, "econoday") >= 0)
      return 0.92;
   if(StringFind(lower, "calendar") >= 0 ||
      StringFind(lower, "consensus") >= 0 ||
      StringFind(lower, "forexfactory") >= 0 ||
      StringFind(lower, "investing") >= 0)
      return 0.85;
   return 0.70;
}

double FXAI_MacroCurrencyRelevance(const string raw_currency,
                                   const string raw_symbol)
{
   string currency = FXAI_NormalizeMacroCurrencyToken(raw_currency);
   string base = FXAI_MacroSymbolBaseCurrency(raw_symbol);
   string quote = FXAI_MacroSymbolQuoteCurrency(raw_symbol);
   if(StringLen(currency) <= 0)
      return 0.0;
   if(currency == base || currency == quote)
      return 1.0;
   int ev_block = FXAI_MacroCurrencyBlock(currency);
   int base_block = FXAI_MacroCurrencyBlock(base);
   int quote_block = FXAI_MacroCurrencyBlock(quote);
   if(ev_block > 0 && (ev_block == base_block || ev_block == quote_block))
      return 0.55;
   if(currency == "USD" && (base == "CAD" || quote == "CAD" || base == "MXN" || quote == "MXN"))
      return 0.45;
   if(currency == "CNY" && (base == "AUD" || quote == "AUD" || base == "NZD" || quote == "NZD"))
      return 0.40;
   return 0.0;
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
                                  const string raw_currency,
                                  const string raw_country,
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
   string currency = FXAI_NormalizeMacroCurrencyToken(raw_currency);
   if(StringLen(currency) <= 0)
      currency = FXAI_MacroCountryToCurrency(raw_country);
   if(StringLen(currency) == 3 && FXAI_MacroCurrencyRelevance(currency, symbol) > 0.35)
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
      string event_id_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string country_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string currency_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string source_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string revision_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string prior_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");
      string surprise_z_s = (!FileIsLineEnding(handle) ? FileReadString(handle) : "");

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
      double revision_delta = FXAI_Clamp(StringToDouble(revision_s), -12.0, 12.0);
      double prior_delta = FXAI_Clamp(StringToDouble(prior_s), -12.0, 12.0);
      double surprise_z = FXAI_Clamp(StringToDouble(surprise_z_s), -8.0, 8.0);
      int event_class = FXAI_ParseMacroEventClass(class_s);
      string event_id = FXAI_NormalizeMacroToken(event_id_s);
      if(StringLen(event_id) <= 0)
         event_id = symbol_trim + "|" + IntegerToString((int)event_time) + "|" + IntegerToString(event_class);
      string country = FXAI_NormalizeMacroToken(country_s);
      string currency = FXAI_NormalizeMacroCurrencyToken(currency_s);
      if(StringLen(currency) <= 0)
      {
         if(StringLen(symbol_trim) == 3)
            currency = FXAI_NormalizeMacroCurrencyToken(symbol_trim);
         else
            currency = FXAI_MacroCountryToCurrency(country);
      }
      string source = FXAI_NormalizeMacroToken(source_s);
      if(MathAbs(surprise_z) <= 1e-9)
      {
         double realized_delta = surprise + 0.20 * (actual_delta - forecast_delta) + 0.15 * revision_delta;
         double scale = 0.50 + 0.40 * MathAbs(forecast_delta) + 0.25 * MathAbs(prior_delta);
         surprise_z = FXAI_Clamp(realized_delta / MathMax(scale, 0.25), -8.0, 8.0);
      }
      double provenance_hash = FXAI_MacroEventStringChecksum01(event_id + "|" + source + "|" + currency);
      double relevance_hint = FXAI_MacroCurrencyRelevance(currency, symbol_trim);

      int idx = ArraySize(g_macro_events);
      ArrayResize(g_macro_events, idx + 1);
      g_macro_events[idx].symbol = symbol_trim;
      g_macro_events[idx].event_id = event_id;
      g_macro_events[idx].country = country;
      g_macro_events[idx].currency = currency;
      g_macro_events[idx].source = source;
      g_macro_events[idx].event_time = event_time;
      g_macro_events[idx].pre_window_min = pre_window;
      g_macro_events[idx].post_window_min = post_window;
      g_macro_events[idx].importance = importance;
      g_macro_events[idx].surprise = surprise;
      g_macro_events[idx].actual_delta = actual_delta;
      g_macro_events[idx].forecast_delta = forecast_delta;
      g_macro_events[idx].prior_delta = prior_delta;
      g_macro_events[idx].revision_delta = revision_delta;
      g_macro_events[idx].surprise_z = surprise_z;
      g_macro_events[idx].relevance_hint = relevance_hint;
      g_macro_events[idx].provenance_hash01 = provenance_hash;
      g_macro_events[idx].event_class = event_class;

      g_macro_event_stats.record_count++;
      g_macro_event_stats.avg_importance += importance;
      g_macro_event_stats.avg_pre_window_min += (double)pre_window;
      g_macro_event_stats.avg_post_window_min += (double)post_window;
      g_macro_event_stats.avg_surprise_z_abs += MathAbs(surprise_z);
      g_macro_event_stats.avg_revision_abs += MathAbs(revision_delta);
      g_macro_event_stats.checksum01 +=
         0.31 * FXAI_MacroEventStringChecksum01(symbol_trim) +
         0.11 * FXAI_MacroEventStringChecksum01(class_s) +
         0.07 * importance +
         0.03 * MathAbs(surprise);
      g_macro_event_stats.provenance_hash01 += provenance_hash;
      if(g_macro_event_stats.first_event_time <= 0 || event_time < g_macro_event_stats.first_event_time)
         g_macro_event_stats.first_event_time = event_time;
      if(g_macro_event_stats.last_event_time <= 0 || event_time > g_macro_event_stats.last_event_time)
         g_macro_event_stats.last_event_time = event_time;
      switch(event_class)
      {
         case 1: g_macro_event_stats.family_rates_count++; break;
         case 2: g_macro_event_stats.family_inflation_count++; break;
         case 3: g_macro_event_stats.family_labor_count++; break;
         case 4: g_macro_event_stats.family_growth_count++; break;
         case 5: g_macro_event_stats.family_trade_count++; break;
      }

      bool seen_symbol = false;
      bool seen_source = false;
      bool seen_event_id = false;
      for(int j=0; j<idx; j++)
      {
         if(g_macro_events[j].symbol == symbol_trim)
            seen_symbol = true;
         if(g_macro_events[j].source == source)
            seen_source = true;
         if(g_macro_events[j].event_id == event_id)
            seen_event_id = true;
      }
      if(!seen_symbol)
         g_macro_event_stats.distinct_symbols++;
      if(!seen_source && StringLen(source) > 0)
         g_macro_event_stats.distinct_sources++;
      if(!seen_event_id)
         g_macro_event_stats.distinct_event_ids++;
   }

   FileClose(handle);
   g_macro_events_available = (ArraySize(g_macro_events) > 0);
   if(g_macro_event_stats.record_count > 0)
   {
      double denom = (double)g_macro_event_stats.record_count;
      g_macro_event_stats.avg_importance /= denom;
      g_macro_event_stats.avg_pre_window_min /= denom;
      g_macro_event_stats.avg_post_window_min /= denom;
      g_macro_event_stats.avg_surprise_z_abs /= denom;
      g_macro_event_stats.avg_revision_abs /= denom;
      double frac = g_macro_event_stats.checksum01 / 8192.0;
      g_macro_event_stats.checksum01 = frac - MathFloor(frac);
      double prov_frac = g_macro_event_stats.provenance_hash01 / denom;
      g_macro_event_stats.provenance_hash01 = prov_frac - MathFloor(prov_frac);
   }
   else
   {
      FXAI_ClearMacroEventDatasetStats(g_macro_event_stats);
   }
   if(g_macro_event_stats.parse_errors > 0 && g_macro_event_stats.record_count > 0)
      g_macro_event_stats.leakage_guard_score = 0.85;
   else if(g_macro_event_stats.record_count > 0)
      g_macro_event_stats.leakage_guard_score = 1.0;
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
      double surprise_z = 0.0;
      double revision_abs = 0.0;
      double currency_relevance = 0.0;
      double provenance_trust = 0.0;
      double rates_activity = 0.0;
      double inflation_activity = 0.0;
      double labor_activity = 0.0;
      double growth_activity = 0.0;
      FXAI_GetMacroEventFeatures(symbol,
                                 rates_arr[idx].time,
                                 pre,
                                 post,
                                 importance,
                                 surprise,
                                 surprise_abs,
                                 class_bias,
                                 surprise_z,
                                 revision_abs,
                                 currency_relevance,
                                 provenance_trust,
                                 rates_activity,
                                 inflation_activity,
                                 labor_activity,
                                 growth_activity);
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
                                double &event_class_bias,
                                double &surprise_zscore,
                                double &revision_abs,
                                double &currency_relevance,
                                double &provenance_trust,
                                double &rates_activity,
                                double &inflation_activity,
                                double &labor_activity,
                                double &growth_activity)
{
   pre_embargo = 0.0;
   post_embargo = 0.0;
   event_importance = 0.0;
   surprise_signed = 0.0;
   surprise_abs = 0.0;
   event_class_bias = 0.0;
   surprise_zscore = 0.0;
   revision_abs = 0.0;
   currency_relevance = 0.0;
   provenance_trust = 0.0;
   rates_activity = 0.0;
   inflation_activity = 0.0;
   labor_activity = 0.0;
   growth_activity = 0.0;

   if(sample_time <= 0 || !FXAI_EnsureMacroEventStoreLoaded())
      return;

   double signed_weight = 0.0;
   double z_weight = 0.0;
   double class_weight = 0.0;
   double revision_weight = 0.0;
   for(int i=0; i<ArraySize(g_macro_events); i++)
   {
      FXAIMacroEventRecord ev = g_macro_events[i];
      if(!FXAI_MacroEventAffectsSymbol(ev.symbol, ev.currency, ev.country, symbol))
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

      double src_trust = FXAI_MacroSourceTrust(ev.source);
      double relevance = MathMax(FXAI_MacroCurrencyRelevance(ev.currency, symbol), FXAI_Clamp(ev.relevance_hint, 0.0, 1.0));
      if(relevance <= 1e-6 && StringLen(ev.currency) <= 0 && StringLen(ev.symbol) <= 0)
         relevance = 1.0;
      double known_weight = FXAI_Clamp(base_importance * (0.35 + 0.65 * proximity) *
                                       MathMax(src_trust, 0.25) *
                                       MathMax(relevance, 0.35),
                                       0.0,
                                       1.0);
      event_importance = MathMax(event_importance, known_weight);
      currency_relevance = MathMax(currency_relevance, FXAI_Clamp(relevance, 0.0, 1.0));
      provenance_trust = MathMax(provenance_trust, FXAI_Clamp(src_trust, 0.0, 1.0));
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
                                            0.20 * (ev.actual_delta - ev.forecast_delta) +
                                            0.15 * ev.revision_delta,
                                            -6.0,
                                            6.0);
      surprise_signed += known_weight * realized_surprise;
      surprise_abs += known_weight * MathAbs(realized_surprise);
      signed_weight += known_weight;
      surprise_zscore += known_weight * ev.surprise_z;
      z_weight += known_weight;
      revision_abs += known_weight * MathAbs(ev.revision_delta);
      revision_weight += known_weight;

      event_class_bias = FXAI_Clamp(event_class_bias + known_weight * FXAI_MacroEventClassBias(ev.event_class),
                                    -2.0,
                                    2.0);
      class_weight += known_weight;
      if(ev.event_class == 1) rates_activity = MathMax(rates_activity, known_weight);
      else if(ev.event_class == 2) inflation_activity = MathMax(inflation_activity, known_weight);
      else if(ev.event_class == 3) labor_activity = MathMax(labor_activity, known_weight);
      else if(ev.event_class == 4) growth_activity = MathMax(growth_activity, known_weight);
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
   if(z_weight > 1e-6)
      surprise_zscore /= z_weight;
   else
      surprise_zscore = 0.0;
   if(revision_weight > 1e-6)
      revision_abs /= revision_weight;
   else
      revision_abs = 0.0;

   pre_embargo = FXAI_Clamp(pre_embargo, 0.0, 1.0);
   post_embargo = FXAI_Clamp(post_embargo, 0.0, 1.0);
   event_importance = FXAI_Clamp(event_importance, 0.0, 1.0);
   surprise_signed = FXAI_Clamp(surprise_signed, -6.0, 6.0);
   surprise_abs = FXAI_Clamp(surprise_abs, 0.0, 6.0);
   event_class_bias = FXAI_Clamp(event_class_bias, -1.0, 1.0);
   surprise_zscore = FXAI_Clamp(surprise_zscore, -8.0, 8.0);
   revision_abs = FXAI_Clamp(revision_abs, 0.0, 8.0);
   currency_relevance = FXAI_Clamp(currency_relevance, 0.0, 1.0);
   provenance_trust = FXAI_Clamp(provenance_trust, 0.0, 1.0);
   rates_activity = FXAI_Clamp(rates_activity, 0.0, 1.0);
   inflation_activity = FXAI_Clamp(inflation_activity, 0.0, 1.0);
   labor_activity = FXAI_Clamp(labor_activity, 0.0, 1.0);
   growth_activity = FXAI_Clamp(growth_activity, 0.0, 1.0);
}

#endif // __FXAI_EVENT_MACRO_MQH__
