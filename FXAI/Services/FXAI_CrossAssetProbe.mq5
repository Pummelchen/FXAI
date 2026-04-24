//+------------------------------------------------------------------+
//|                                         FXAI_CrossAssetProbe.mq5  |
//| Shared MT5 context-symbol probe for the FXAI cross-asset engine.  |
//| Exports broad macro/liquidity proxy state from broker symbols.     |
//+------------------------------------------------------------------+
#property strict
#property service

#include "..\Engine\market_data_gateway.mqh"

#define FXAI_CA_RUNTIME_DIR                 "FXAI\\Runtime"
#define FXAI_CA_CONFIG_FILE                 "FXAI\\Runtime\\cross_asset_probe_config.tsv"
#define FXAI_CA_SNAPSHOT_FILE               "FXAI\\Runtime\\cross_asset_probe_snapshot.json"
#define FXAI_CA_STATUS_FILE                 "FXAI\\Runtime\\cross_asset_probe_status.json"
#define FXAI_CA_HISTORY_FILE                "FXAI\\Runtime\\cross_asset_probe_history.ndjson"
#define FXAI_CA_SCHEMA_VERSION              1
#define FXAI_CA_MAX_SYMBOLS                 192

struct FXAICrossAssetConfig
{
   bool enabled;
   int poll_interval_ms;
   int snapshot_stale_after_sec;
   int symbol_count;
   string symbols[FXAI_CA_MAX_SYMBOLS];
};

struct FXAICrossAssetSymbolState
{
   string symbol;
   bool available;
   datetime updated_at;
   double last_price;
   double change_pct_1h;
   double change_pct_4h;
   double change_pct_1d;
   double change_pct_5d;
   double range_ratio_1d;
};

FXAICrossAssetConfig g_ca_cfg;
datetime g_ca_last_success_at = 0;
datetime g_ca_last_poll_at = 0;
string g_ca_last_error = "";

string FXAI_CA_JSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   StringReplace(out, "\t", " ");
   return out;
}

string FXAI_CA_ISO8601(const datetime value)
{
   if(value <= 0)
      return "";
   MqlDateTime dt;
   TimeToStruct(value, dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       dt.year,
                       dt.mon,
                       dt.day,
                       dt.hour,
                       dt.min,
                       dt.sec);
}

datetime FXAI_CA_Now(void)
{
   datetime now_time = TimeTradeServer();
   if(now_time <= 0)
      now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeGMT();
   return now_time;
}

void FXAI_CA_EnsureFolders(void)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_CA_RUNTIME_DIR, FILE_COMMON);
}

void FXAI_CA_ResetConfig(FXAICrossAssetConfig &cfg)
{
   cfg.enabled = true;
   cfg.poll_interval_ms = 15000;
   cfg.snapshot_stale_after_sec = 300;
   cfg.symbol_count = 0;
   for(int i=0; i<FXAI_CA_MAX_SYMBOLS; i++)
      cfg.symbols[i] = "";
}

bool FXAI_CA_LoadConfig(FXAICrossAssetConfig &cfg)
{
   FXAI_CA_ResetConfig(cfg);
   int handle = FileOpen(FXAI_CA_CONFIG_FILE,
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
      if(key == "enabled")
         cfg.enabled = (StringToInteger(value) != 0);
      else if(key == "poll_interval_ms")
         cfg.poll_interval_ms = (int)StringToInteger(value);
      else if(key == "snapshot_stale_after_sec")
         cfg.snapshot_stale_after_sec = (int)StringToInteger(value);
      else if(key == "symbol")
      {
         string symbol = value;
         StringTrimLeft(symbol);
         StringTrimRight(symbol);
         if(StringLen(symbol) > 0 && cfg.symbol_count < FXAI_CA_MAX_SYMBOLS)
         {
            cfg.symbols[cfg.symbol_count] = symbol;
            cfg.symbol_count++;
         }
      }
   }
   FileClose(handle);

   if(cfg.poll_interval_ms < 1000)
      cfg.poll_interval_ms = 1000;
   if(cfg.snapshot_stale_after_sec < 30)
      cfg.snapshot_stale_after_sec = 30;
   return true;
}

double FXAI_CA_LastPrice(const string symbol)
{
   MqlTick tick;
   if(!FXAI_MarketDataGetLatestTick(symbol, tick))
      return 0.0;
   if(tick.bid > 0.0 && tick.ask > 0.0)
      return 0.5 * (tick.bid + tick.ask);
   if(tick.last > 0.0)
      return tick.last;
   if(tick.bid > 0.0)
      return tick.bid;
   if(tick.ask > 0.0)
      return tick.ask;
   return 0.0;
}

double FXAI_CA_ChangePct(const string symbol,
                         const ENUM_TIMEFRAMES timeframe,
                         const int lookback_bars)
{
   if(lookback_bars < 1)
      return 0.0;
   double close_arr[];
   if(!FXAI_MarketDataCopyCloseByPos(symbol, timeframe, 0, lookback_bars + 2, close_arr) ||
      ArraySize(close_arr) <= lookback_bars)
      return 0.0;
   double current = close_arr[0];
   double past = close_arr[lookback_bars];
   if(current <= 0.0 || past <= 0.0)
      return 0.0;
   return 100.0 * ((current / past) - 1.0);
}

double FXAI_CA_RangeRatio1D(const string symbol)
{
   MqlRates rates[];
   if(!FXAI_MarketDataCopyRatesByPos(symbol, PERIOD_H1, 0, 24, rates) || ArraySize(rates) <= 0)
      return 0.0;
   int copied = ArraySize(rates);
   double hi = rates[0].high;
   double lo = rates[0].low;
   double close_sum = 0.0;
   for(int i=0; i<copied; i++)
   {
      if(rates[i].high > hi)
         hi = rates[i].high;
      if(rates[i].low < lo)
         lo = rates[i].low;
      close_sum += rates[i].close;
   }
   double avg_close = close_sum / MathMax(copied, 1);
   if(avg_close <= 0.0)
      return 0.0;
   return 100.0 * ((hi - lo) / avg_close);
}

bool FXAI_CA_Collect(const string symbol,
                     FXAICrossAssetSymbolState &out)
{
   out.symbol = symbol;
   out.available = false;
   out.updated_at = 0;
   out.last_price = 0.0;
   out.change_pct_1h = 0.0;
   out.change_pct_4h = 0.0;
   out.change_pct_1d = 0.0;
   out.change_pct_5d = 0.0;
   out.range_ratio_1d = 0.0;

   if(StringLen(symbol) <= 0)
      return false;
   SymbolSelect(symbol, true);
   double last_price = FXAI_CA_LastPrice(symbol);
   if(last_price <= 0.0)
      return false;

   out.available = true;
   out.updated_at = FXAI_CA_Now();
   out.last_price = last_price;
   out.change_pct_1h = FXAI_CA_ChangePct(symbol, PERIOD_H1, 1);
   out.change_pct_4h = FXAI_CA_ChangePct(symbol, PERIOD_H1, 4);
   out.change_pct_1d = FXAI_CA_ChangePct(symbol, PERIOD_D1, 1);
   out.change_pct_5d = FXAI_CA_ChangePct(symbol, PERIOD_D1, 5);
   out.range_ratio_1d = FXAI_CA_RangeRatio1D(symbol);
   return true;
}

bool FXAI_CA_WriteArtifacts(FXAICrossAssetSymbolState &states[],
                            const datetime generated_at)
{
   int available_count = 0;
   for(int i=0; i<ArraySize(states); i++)
   {
      if(states[i].available)
         available_count++;
   }

   int handle = FileOpen(FXAI_CA_SNAPSHOT_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;

   FileWriteString(handle, "{\n");
   FileWriteString(handle, "  \"schema_version\": " + IntegerToString(FXAI_CA_SCHEMA_VERSION) + ",\n");
   FileWriteString(handle, "  \"generated_at\": \"" + FXAI_CA_ISO8601(generated_at) + "\",\n");
   FileWriteString(handle, "  \"symbols\": {\n");
   bool first_symbol = true;
   for(int i=0; i<ArraySize(states); i++)
   {
      if(!first_symbol)
         FileWriteString(handle, ",\n");
      first_symbol = false;
      FileWriteString(handle, "    \"" + FXAI_CA_JSONEscape(states[i].symbol) + "\": {\n");
      FileWriteString(handle, "      \"available\": " + (states[i].available ? "true" : "false") + ",\n");
      FileWriteString(handle, "      \"updated_at\": \"" + FXAI_CA_ISO8601(states[i].updated_at) + "\",\n");
      FileWriteString(handle, "      \"last_price\": " + DoubleToString(states[i].last_price, 8) + ",\n");
      FileWriteString(handle, "      \"change_pct_1h\": " + DoubleToString(states[i].change_pct_1h, 6) + ",\n");
      FileWriteString(handle, "      \"change_pct_4h\": " + DoubleToString(states[i].change_pct_4h, 6) + ",\n");
      FileWriteString(handle, "      \"change_pct_1d\": " + DoubleToString(states[i].change_pct_1d, 6) + ",\n");
      FileWriteString(handle, "      \"change_pct_5d\": " + DoubleToString(states[i].change_pct_5d, 6) + ",\n");
      FileWriteString(handle, "      \"range_ratio_1d\": " + DoubleToString(states[i].range_ratio_1d, 6) + "\n");
      FileWriteString(handle, "    }");
   }
   FileWriteString(handle, "\n  }\n");
   FileWriteString(handle, "}\n");
   FileClose(handle);

   handle = FileOpen(FXAI_CA_STATUS_FILE,
                     FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
      return false;
   FileWriteString(handle, "{\n");
   FileWriteString(handle, "  \"generated_at\": \"" + FXAI_CA_ISO8601(generated_at) + "\",\n");
   FileWriteString(handle, "  \"service\": {\n");
   FileWriteString(handle, "    \"ok\": true,\n");
   FileWriteString(handle, "    \"stale\": false,\n");
   FileWriteString(handle, "    \"enabled\": " + (g_ca_cfg.enabled ? "true" : "false") + ",\n");
   FileWriteString(handle, "    \"poll_interval_ms\": " + IntegerToString(g_ca_cfg.poll_interval_ms) + ",\n");
   FileWriteString(handle, "    \"snapshot_stale_after_sec\": " + IntegerToString(g_ca_cfg.snapshot_stale_after_sec) + ",\n");
   FileWriteString(handle, "    \"last_poll_at\": \"" + FXAI_CA_ISO8601(g_ca_last_poll_at) + "\",\n");
   FileWriteString(handle, "    \"last_success_at\": \"" + FXAI_CA_ISO8601(g_ca_last_success_at) + "\",\n");
   FileWriteString(handle, "    \"configured_symbols\": " + IntegerToString(g_ca_cfg.symbol_count) + ",\n");
   FileWriteString(handle, "    \"available_symbols\": " + IntegerToString(available_count) + ",\n");
   FileWriteString(handle, "    \"last_error\": \"" + FXAI_CA_JSONEscape(g_ca_last_error) + "\"\n");
   FileWriteString(handle, "  }\n");
   FileWriteString(handle, "}\n");
   FileClose(handle);

   handle = FileOpen(FXAI_CA_HISTORY_FILE,
                     FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                     FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      handle = FileOpen(FXAI_CA_HISTORY_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      FileSeek(handle, 0, SEEK_END);
      FileWriteString(handle,
                      "{\"generated_at\":\"" + FXAI_CA_ISO8601(generated_at) +
                      "\",\"configured_symbols\":" + IntegerToString(g_ca_cfg.symbol_count) +
                      ",\"available_symbols\":" + IntegerToString(available_count) + "}\n");
      FileClose(handle);
   }
   return true;
}

void OnStart()
{
   FXAI_CA_EnsureFolders();
   while(!IsStopped())
   {
      g_ca_last_poll_at = FXAI_CA_Now();
      if(!FXAI_CA_LoadConfig(g_ca_cfg))
      {
         g_ca_last_error = "cross-asset probe config missing";
         Sleep(5000);
         continue;
      }
      if(!g_ca_cfg.enabled)
      {
         Sleep(MathMax(g_ca_cfg.poll_interval_ms, 1000));
         continue;
      }

      FXAICrossAssetSymbolState states[];
      ArrayResize(states, g_ca_cfg.symbol_count);
      int available_count = 0;
      for(int i=0; i<g_ca_cfg.symbol_count; i++)
      {
         FXAI_CA_Collect(g_ca_cfg.symbols[i], states[i]);
         if(states[i].available)
            available_count++;
      }

      datetime generated_at = FXAI_CA_Now();
      if(FXAI_CA_WriteArtifacts(states, generated_at))
      {
         g_ca_last_success_at = generated_at;
         g_ca_last_error = "";
      }
      else
      {
         g_ca_last_error = "cross-asset probe artifact write failed";
      }

      int sleep_ms = g_ca_cfg.poll_interval_ms;
      if(sleep_ms < 1000)
         sleep_ms = 1000;
      int slept = 0;
      while(!IsStopped() && slept < sleep_ms)
      {
         int chunk = MathMin(250, sleep_ms - slept);
         Sleep(chunk);
         slept += chunk;
      }
   }
}
