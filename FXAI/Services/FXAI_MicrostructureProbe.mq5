//+------------------------------------------------------------------+
//|                                       FXAI_MicrostructureProbe.mq5|
//| Shared MT5 microstructure and order-flow proxy collector for FXAI |
//| Computes broker-visible short-horizon liquidity/flow proxies.     |
//+------------------------------------------------------------------+
#property strict
#property service

#define FXAI_MS_RUNTIME_DIR                 "FXAI\\Runtime"
#define FXAI_MS_CONFIG_FILE                 "FXAI\\Runtime\\microstructure_service_config.tsv"
#define FXAI_MS_SNAPSHOT_FILE               "FXAI\\Runtime\\microstructure_snapshot.json"
#define FXAI_MS_STATUS_FILE                 "FXAI\\Runtime\\microstructure_status.json"
#define FXAI_MS_FLAT_FILE                   "FXAI\\Runtime\\microstructure_snapshot_flat.tsv"
#define FXAI_MS_HISTORY_FILE                "FXAI\\Runtime\\microstructure_history.ndjson"
#define FXAI_MS_SYMBOL_MAP_FILE             "FXAI\\Runtime\\microstructure_symbol_map.tsv"
#define FXAI_MS_SCHEMA_VERSION              1
#define FXAI_MS_MAX_WINDOWS                 5
#define FXAI_MS_MAX_PAIRS                   64
#define FXAI_MS_MAX_REASONS                 6
#define FXAI_MS_BASELINE_BUCKETS            4
#define FXAI_MS_HISTORY_HEARTBEAT_SEC       60

struct FXAIMicrostructureConfig
{
   bool enabled;
   int poll_interval_ms;
   int symbol_refresh_sec;
   int snapshot_stale_after_sec;
   int max_history_window_sec;
   int window_count;
   int windows_sec[FXAI_MS_MAX_WINDOWS];

   double wide_spread_zscore;
   double wide_spread_absolute_points_floor;
   double spread_instability_caution;
   double spread_instability_block;
   double tick_burst_ratio_caution;
   double tick_burst_ratio_block;
   double vol_burst_ratio_caution;
   double vol_burst_ratio_block;
   double shock_move_points_factor;
   double stop_run_reversal_fraction;
   double stop_run_rejection_score_flag;
   double liquidity_stress_caution;
   double liquidity_stress_block;
   double hostile_execution_caution;
   double hostile_execution_block;
   double clean_trend_efficiency_floor;
   double clean_trend_imbalance_floor;

   bool runtime_block_on_unknown;
   double caution_lot_scale;
   double caution_enter_prob_buffer;

   int handoff_minutes;
   int session_count;
   string session_labels[6];
   int session_start_hour[6];
   int session_end_hour[6];

   int pair_count;
   string canonical_pairs[FXAI_MS_MAX_PAIRS];
};

struct FXAIMicrostructureResolvedSymbol
{
   string canonical_pair;
   string broker_symbol;
   double point;
   int digits;
   bool active;
};

struct FXAIMicrostructureWindowStats
{
   int tick_count;
   int quote_change_count;
   int up_count;
   int down_count;
   int spread_widen_events;
   int shock_move_count;
   int directional_run_length_current;
   double signed_mid_change_sum_pts;
   double total_abs_change_sum_pts;
   double spread_current;
   double spread_mean;
   double spread_std;
   double spread_zscore;
   double wide_spread_fraction;
   double spread_instability;
   double tick_rate;
   double tick_rate_zscore;
   double intensity_burst_score;
   double quote_change_rate;
   double realized_vol;
   double realized_vol_zscore;
   double vol_burst_score;
   double range_expansion;
   double directional_efficiency;
   double local_extrema_breach_score;
   bool sweep_and_reject_flag;
   double breakout_reversal_score;
   double exhaustion_proxy;
};

struct FXAIMicrostructurePairState
{
   string canonical_pair;
   string broker_symbol;
   bool available;
   bool stale;
   datetime generated_at;
   double spread_current;
   double silent_gap_seconds_current;
   string session_tag;
   bool handoff_flag;
   int minutes_since_session_open;
   int minutes_to_session_close;
   double session_open_burst_score;
   double session_spread_behavior_score;
   double liquidity_stress_score;
   double hostile_execution_score;
   string microstructure_regime;
   string trade_gate;
   int reason_count;
   string reasons[FXAI_MS_MAX_REASONS];

   int tick_up_count[FXAI_MS_MAX_WINDOWS];
   int tick_down_count[FXAI_MS_MAX_WINDOWS];
   double tick_imbalance[FXAI_MS_MAX_WINDOWS];
   double signed_mid_change_sum_pts[FXAI_MS_MAX_WINDOWS];
   double directional_efficiency[FXAI_MS_MAX_WINDOWS];
   double spread_mean[FXAI_MS_MAX_WINDOWS];
   double spread_std[FXAI_MS_MAX_WINDOWS];
   double spread_zscore[FXAI_MS_MAX_WINDOWS];
   int spread_widen_events[FXAI_MS_MAX_WINDOWS];
   double spread_instability[FXAI_MS_MAX_WINDOWS];
   double wide_spread_fraction[FXAI_MS_MAX_WINDOWS];
   int tick_count[FXAI_MS_MAX_WINDOWS];
   double tick_rate[FXAI_MS_MAX_WINDOWS];
   double tick_rate_zscore[FXAI_MS_MAX_WINDOWS];
   double intensity_burst_score[FXAI_MS_MAX_WINDOWS];
   double quote_change_rate[FXAI_MS_MAX_WINDOWS];
   double realized_vol[FXAI_MS_MAX_WINDOWS];
   double realized_vol_zscore[FXAI_MS_MAX_WINDOWS];
   double vol_burst_score[FXAI_MS_MAX_WINDOWS];
   double range_expansion[FXAI_MS_MAX_WINDOWS];
   int shock_move_count[FXAI_MS_MAX_WINDOWS];
   double local_extrema_breach_score[FXAI_MS_MAX_WINDOWS];
   bool sweep_and_reject_flag[FXAI_MS_MAX_WINDOWS];
   double breakout_reversal_score[FXAI_MS_MAX_WINDOWS];
   double exhaustion_proxy[FXAI_MS_MAX_WINDOWS];
   int directional_run_length_current[FXAI_MS_MAX_WINDOWS];
};

FXAIMicrostructureConfig g_ms_cfg;
FXAIMicrostructureResolvedSymbol g_ms_symbols[];
datetime g_ms_last_symbol_refresh = 0;
datetime g_ms_last_success_at = 0;
datetime g_ms_last_poll_at = 0;
string g_ms_last_error = "";
datetime g_ms_history_last_at[];
string g_ms_history_last_regime[];
string g_ms_history_last_gate[];
double g_ms_history_last_hostile[];

double FXAI_MS_Clamp(const double value,
                     const double min_v,
                     const double max_v)
{
   if(value < min_v)
      return min_v;
   if(value > max_v)
      return max_v;
   return value;
}

string FXAI_MS_JSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   StringReplace(out, "\t", " ");
   return out;
}

string FXAI_MS_ISO8601(const datetime value)
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

datetime FXAI_MS_Now(void)
{
   datetime now = TimeTradeServer();
   if(now <= 0)
      now = TimeCurrent();
   if(now <= 0)
      now = TimeGMT();
   return now;
}

void FXAI_MS_EnsureFolders(void)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_MS_RUNTIME_DIR, FILE_COMMON);
}

void FXAI_MS_ResetConfig(FXAIMicrostructureConfig &cfg)
{
   cfg.enabled = true;
   cfg.poll_interval_ms = 5000;
   cfg.symbol_refresh_sec = 300;
   cfg.snapshot_stale_after_sec = 45;
   cfg.max_history_window_sec = 960;
   cfg.window_count = 5;
   cfg.windows_sec[0] = 10;
   cfg.windows_sec[1] = 30;
   cfg.windows_sec[2] = 60;
   cfg.windows_sec[3] = 300;
   cfg.windows_sec[4] = 900;
   cfg.wide_spread_zscore = 1.45;
   cfg.wide_spread_absolute_points_floor = 2.0;
   cfg.spread_instability_caution = 0.56;
   cfg.spread_instability_block = 0.82;
   cfg.tick_burst_ratio_caution = 1.35;
   cfg.tick_burst_ratio_block = 1.85;
   cfg.vol_burst_ratio_caution = 1.40;
   cfg.vol_burst_ratio_block = 1.90;
   cfg.shock_move_points_factor = 1.80;
   cfg.stop_run_reversal_fraction = 0.45;
   cfg.stop_run_rejection_score_flag = 0.58;
   cfg.liquidity_stress_caution = 0.58;
   cfg.liquidity_stress_block = 0.86;
   cfg.hostile_execution_caution = 0.56;
   cfg.hostile_execution_block = 0.84;
   cfg.clean_trend_efficiency_floor = 0.62;
   cfg.clean_trend_imbalance_floor = 0.22;
   cfg.runtime_block_on_unknown = false;
   cfg.caution_lot_scale = 0.72;
   cfg.caution_enter_prob_buffer = 0.04;
   cfg.handoff_minutes = 20;
   cfg.session_count = 5;
   cfg.session_labels[0] = "ASIA";
   cfg.session_start_hour[0] = 0;
   cfg.session_end_hour[0] = 7;
   cfg.session_labels[1] = "LONDON";
   cfg.session_start_hour[1] = 7;
   cfg.session_end_hour[1] = 12;
   cfg.session_labels[2] = "LONDON_NEWYORK_OVERLAP";
   cfg.session_start_hour[2] = 12;
   cfg.session_end_hour[2] = 16;
   cfg.session_labels[3] = "NEWYORK";
   cfg.session_start_hour[3] = 16;
   cfg.session_end_hour[3] = 21;
   cfg.session_labels[4] = "ROLLOVER";
   cfg.session_start_hour[4] = 21;
   cfg.session_end_hour[4] = 24;
   cfg.pair_count = 0;
}

void FXAI_MS_SetDefaultPairs(FXAIMicrostructureConfig &cfg)
{
   string defaults[] = {
      "AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDSGD","AUDUSD","CADCHF","CADJPY","CHFJPY","CHFSGD",
      "EURAUD","EURCAD","EURCHF","EURGBP","EURHKD","EURNOK","EURPLN","EURSEK","EURSGD","EURUSD",
      "EURZAR","GBPAUD","GBPCAD","GBPCHF","GBPDKK","GBPJPY","GBPNOK","GBPNZD","GBPSEK","GBPSGD",
      "GBPUSD","NOKJPY","NOKSEK","NZDCAD","NZDCHF","NZDJPY","NZDUSD","SEKJPY","SGDJPY","USDCAD",
      "USDCHF","USDCNH","USDCZK","USDDKK","USDHKD","USDHUF","USDJPY","USDMXN","USDNOK","USDPLN",
      "USDSEK","USDSGD","USDTHB","USDZAR"
   };
   cfg.pair_count = ArraySize(defaults);
   for(int i = 0; i < cfg.pair_count && i < FXAI_MS_MAX_PAIRS; i++)
      cfg.canonical_pairs[i] = defaults[i];
}

string FXAI_MS_Upper(const string raw)
{
   string out = raw;
   StringToUpper(out);
   return out;
}

string FXAI_MS_AlphaOnly(const string raw)
{
   string upper = FXAI_MS_Upper(raw);
   string out = "";
   int n = StringLen(upper);
   for(int i = 0; i < n; i++)
   {
      string token = StringSubstr(upper, i, 1);
      if(token >= "A" && token <= "Z")
         out += token;
   }
   return out;
}

string FXAI_MS_CanonicalPairFromRaw(const string raw)
{
   string alpha = FXAI_MS_AlphaOnly(raw);
   for(int i = 0; i < g_ms_cfg.pair_count; i++)
   {
      string pair = g_ms_cfg.canonical_pairs[i];
      if(StringLen(pair) == 6 && StringFind(alpha, pair) >= 0)
         return pair;
   }
   return "";
}

string FXAI_MS_WindowSuffix(const int seconds)
{
   if(seconds % 60 == 0 && seconds >= 60)
      return IntegerToString(seconds / 60) + "m";
   return IntegerToString(seconds) + "s";
}

void FXAI_MS_ResetWindowStats(FXAIMicrostructureWindowStats &stats)
{
   stats.tick_count = 0;
   stats.quote_change_count = 0;
   stats.up_count = 0;
   stats.down_count = 0;
   stats.spread_widen_events = 0;
   stats.shock_move_count = 0;
   stats.directional_run_length_current = 0;
   stats.signed_mid_change_sum_pts = 0.0;
   stats.total_abs_change_sum_pts = 0.0;
   stats.spread_current = 0.0;
   stats.spread_mean = 0.0;
   stats.spread_std = 0.0;
   stats.spread_zscore = 0.0;
   stats.wide_spread_fraction = 0.0;
   stats.spread_instability = 0.0;
   stats.tick_rate = 0.0;
   stats.tick_rate_zscore = 0.0;
   stats.intensity_burst_score = 1.0;
   stats.quote_change_rate = 0.0;
   stats.realized_vol = 0.0;
   stats.realized_vol_zscore = 0.0;
   stats.vol_burst_score = 1.0;
   stats.range_expansion = 0.0;
   stats.directional_efficiency = 0.0;
   stats.local_extrema_breach_score = 0.0;
   stats.sweep_and_reject_flag = false;
   stats.breakout_reversal_score = 0.0;
   stats.exhaustion_proxy = 0.0;
}

void FXAI_MS_ResetPairState(FXAIMicrostructurePairState &state)
{
   state.canonical_pair = "";
   state.broker_symbol = "";
   state.available = false;
   state.stale = true;
   state.generated_at = 0;
   state.spread_current = 0.0;
   state.silent_gap_seconds_current = 0.0;
   state.session_tag = "UNKNOWN";
   state.handoff_flag = false;
   state.minutes_since_session_open = -1;
   state.minutes_to_session_close = -1;
   state.session_open_burst_score = 0.0;
   state.session_spread_behavior_score = 0.0;
   state.liquidity_stress_score = 0.0;
   state.hostile_execution_score = 0.0;
   state.microstructure_regime = "UNKNOWN";
   state.trade_gate = "UNKNOWN";
   state.reason_count = 0;
   for(int i = 0; i < FXAI_MS_MAX_REASONS; i++)
      state.reasons[i] = "";
   for(int w = 0; w < FXAI_MS_MAX_WINDOWS; w++)
   {
      state.tick_up_count[w] = 0;
      state.tick_down_count[w] = 0;
      state.tick_imbalance[w] = 0.0;
      state.signed_mid_change_sum_pts[w] = 0.0;
      state.directional_efficiency[w] = 0.0;
      state.spread_mean[w] = 0.0;
      state.spread_std[w] = 0.0;
      state.spread_zscore[w] = 0.0;
      state.spread_widen_events[w] = 0;
      state.spread_instability[w] = 0.0;
      state.wide_spread_fraction[w] = 0.0;
      state.tick_count[w] = 0;
      state.tick_rate[w] = 0.0;
      state.tick_rate_zscore[w] = 0.0;
      state.intensity_burst_score[w] = 1.0;
      state.quote_change_rate[w] = 0.0;
      state.realized_vol[w] = 0.0;
      state.realized_vol_zscore[w] = 0.0;
      state.vol_burst_score[w] = 1.0;
      state.range_expansion[w] = 0.0;
      state.shock_move_count[w] = 0;
      state.local_extrema_breach_score[w] = 0.0;
      state.sweep_and_reject_flag[w] = false;
      state.breakout_reversal_score[w] = 0.0;
      state.exhaustion_proxy[w] = 0.0;
      state.directional_run_length_current[w] = 0;
   }
}

void FXAI_MS_AppendReason(FXAIMicrostructurePairState &state,
                          const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i = 0; i < state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_MS_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

bool FXAI_MS_LoadConfig(FXAIMicrostructureConfig &cfg)
{
   FXAI_MS_ResetConfig(cfg);
   FXAI_MS_SetDefaultPairs(cfg);

   int handle = FileOpen(FXAI_MS_CONFIG_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;

   cfg.pair_count = 0;
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
      if(key == "enabled")
         cfg.enabled = (StringToInteger(value) != 0);
      else if(key == "poll_interval_ms")
         cfg.poll_interval_ms = (int)StringToInteger(value);
      else if(key == "symbol_refresh_sec")
         cfg.symbol_refresh_sec = (int)StringToInteger(value);
      else if(key == "snapshot_stale_after_sec")
         cfg.snapshot_stale_after_sec = (int)StringToInteger(value);
      else if(key == "max_history_window_sec")
         cfg.max_history_window_sec = (int)StringToInteger(value);
      else if(key == "windows_csv")
      {
         string win_parts[];
         int wn = StringSplit(value, ',', win_parts);
         cfg.window_count = 0;
         for(int i = 0; i < wn && i < FXAI_MS_MAX_WINDOWS; i++)
         {
            int window_sec = (int)StringToInteger(win_parts[i]);
            if(window_sec > 0)
               cfg.windows_sec[cfg.window_count++] = window_sec;
         }
      }
      else if(key == "threshold_wide_spread_zscore")
         cfg.wide_spread_zscore = StringToDouble(value);
      else if(key == "threshold_wide_spread_absolute_points_floor")
         cfg.wide_spread_absolute_points_floor = StringToDouble(value);
      else if(key == "threshold_spread_instability_caution")
         cfg.spread_instability_caution = StringToDouble(value);
      else if(key == "threshold_spread_instability_block")
         cfg.spread_instability_block = StringToDouble(value);
      else if(key == "threshold_tick_burst_ratio_caution")
         cfg.tick_burst_ratio_caution = StringToDouble(value);
      else if(key == "threshold_tick_burst_ratio_block")
         cfg.tick_burst_ratio_block = StringToDouble(value);
      else if(key == "threshold_vol_burst_ratio_caution")
         cfg.vol_burst_ratio_caution = StringToDouble(value);
      else if(key == "threshold_vol_burst_ratio_block")
         cfg.vol_burst_ratio_block = StringToDouble(value);
      else if(key == "threshold_shock_move_points_factor")
         cfg.shock_move_points_factor = StringToDouble(value);
      else if(key == "threshold_stop_run_reversal_fraction")
         cfg.stop_run_reversal_fraction = StringToDouble(value);
      else if(key == "threshold_stop_run_rejection_score_flag")
         cfg.stop_run_rejection_score_flag = StringToDouble(value);
      else if(key == "threshold_liquidity_stress_caution")
         cfg.liquidity_stress_caution = StringToDouble(value);
      else if(key == "threshold_liquidity_stress_block")
         cfg.liquidity_stress_block = StringToDouble(value);
      else if(key == "threshold_hostile_execution_caution")
         cfg.hostile_execution_caution = StringToDouble(value);
      else if(key == "threshold_hostile_execution_block")
         cfg.hostile_execution_block = StringToDouble(value);
      else if(key == "threshold_clean_trend_efficiency_floor")
         cfg.clean_trend_efficiency_floor = StringToDouble(value);
      else if(key == "threshold_clean_trend_imbalance_floor")
         cfg.clean_trend_imbalance_floor = StringToDouble(value);
      else if(key == "runtime_block_on_unknown")
         cfg.runtime_block_on_unknown = (StringToInteger(value) != 0);
      else if(key == "runtime_caution_lot_scale")
         cfg.caution_lot_scale = StringToDouble(value);
      else if(key == "runtime_caution_enter_prob_buffer")
         cfg.caution_enter_prob_buffer = StringToDouble(value);
      else if(key == "session_handoff_minutes")
         cfg.handoff_minutes = (int)StringToInteger(value);
      else if(key == "session_window")
      {
         string session_parts[];
         int sn = StringSplit(value, ',', session_parts);
         if(sn >= 3 && cfg.session_count < 6)
         {
            int idx = cfg.session_count++;
            cfg.session_labels[idx] = session_parts[0];
            cfg.session_start_hour[idx] = (int)StringToInteger(session_parts[1]);
            cfg.session_end_hour[idx] = (int)StringToInteger(session_parts[2]);
         }
      }
      else if(key == "pair")
      {
         string pair = FXAI_MS_Upper(value);
         if(StringLen(pair) == 6 && cfg.pair_count < FXAI_MS_MAX_PAIRS)
            cfg.canonical_pairs[cfg.pair_count++] = pair;
      }
   }
   FileClose(handle);

   if(cfg.window_count <= 0)
   {
      cfg.window_count = 5;
      cfg.windows_sec[0] = 10;
      cfg.windows_sec[1] = 30;
      cfg.windows_sec[2] = 60;
      cfg.windows_sec[3] = 300;
      cfg.windows_sec[4] = 900;
   }
   if(cfg.pair_count <= 0)
      FXAI_MS_SetDefaultPairs(cfg);
   if(cfg.poll_interval_ms < 1000)
      cfg.poll_interval_ms = 1000;
   if(cfg.max_history_window_sec < cfg.windows_sec[cfg.window_count - 1])
      cfg.max_history_window_sec = cfg.windows_sec[cfg.window_count - 1];
   return true;
}

string FXAI_MS_ResolveBrokerSymbol(const string canonical_pair)
{
   string best = "";
   int best_score = 1000000;
   string upper_pair = FXAI_MS_Upper(canonical_pair);

   if(SymbolSelect(upper_pair, true))
      return upper_pair;

   for(int selected = 1; selected >= 0; selected--)
   {
      int total = SymbolsTotal((bool)(selected != 0));
      for(int i = 0; i < total; i++)
      {
         string candidate = SymbolName(i, (bool)(selected != 0));
         if(StringLen(candidate) <= 0)
            continue;
         if(FXAI_MS_CanonicalPairFromRaw(candidate) != upper_pair)
            continue;
         int score = StringLen(candidate) - 6;
         if(selected != 0)
            score -= 10;
         string upper_cand = FXAI_MS_Upper(candidate);
         if(StringFind(upper_cand, upper_pair) == 0)
            score -= 2;
         if(best_score > score)
         {
            best = candidate;
            best_score = score;
         }
      }
   }

   if(StringLen(best) > 0)
      SymbolSelect(best, true);
   return best;
}

void FXAI_MS_RefreshResolvedSymbols(const bool force_refresh = false)
{
   datetime now_time = FXAI_MS_Now();
   if(!force_refresh && g_ms_last_symbol_refresh > 0 &&
      (now_time - g_ms_last_symbol_refresh) < MathMax(g_ms_cfg.symbol_refresh_sec, 30))
      return;

   ArrayResize(g_ms_symbols, g_ms_cfg.pair_count);
   ArrayResize(g_ms_history_last_at, g_ms_cfg.pair_count);
   ArrayResize(g_ms_history_last_regime, g_ms_cfg.pair_count);
   ArrayResize(g_ms_history_last_gate, g_ms_cfg.pair_count);
   ArrayResize(g_ms_history_last_hostile, g_ms_cfg.pair_count);

   for(int i = 0; i < g_ms_cfg.pair_count; i++)
   {
      string canonical_pair = g_ms_cfg.canonical_pairs[i];
      string broker_symbol = FXAI_MS_ResolveBrokerSymbol(canonical_pair);
      g_ms_symbols[i].canonical_pair = canonical_pair;
      g_ms_symbols[i].broker_symbol = broker_symbol;
      g_ms_symbols[i].active = (StringLen(broker_symbol) > 0);
      g_ms_symbols[i].point = (g_ms_symbols[i].active ? SymbolInfoDouble(broker_symbol, SYMBOL_POINT) : 0.0);
      g_ms_symbols[i].digits = (g_ms_symbols[i].active ? (int)SymbolInfoInteger(broker_symbol, SYMBOL_DIGITS) : 0);
      if(ArraySize(g_ms_history_last_regime) > i && StringLen(g_ms_history_last_regime[i]) <= 0)
      {
         g_ms_history_last_regime[i] = "";
         g_ms_history_last_gate[i] = "";
         g_ms_history_last_hostile[i] = -1.0;
         g_ms_history_last_at[i] = 0;
      }
   }
   g_ms_last_symbol_refresh = now_time;
}

void FXAI_MS_WriteSymbolMap(void)
{
   FXAI_MS_EnsureFolders();
   int handle = FileOpen(FXAI_MS_SYMBOL_MAP_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   for(int i = 0; i < ArraySize(g_ms_symbols); i++)
   {
      if(!g_ms_symbols[i].active)
         continue;
      FileWriteString(handle,
                      "symbol\t" + FXAI_MS_Upper(g_ms_symbols[i].broker_symbol) +
                      "\t" + g_ms_symbols[i].canonical_pair + "\r\n");
   }
   FileClose(handle);
}

bool FXAI_MS_LoadTicks(const string symbol,
                       const int lookback_sec,
                       MqlTick &ticks[])
{
   ArrayResize(ticks, 0);
   if(StringLen(symbol) <= 0 || lookback_sec <= 0)
      return false;

   datetime now_time = FXAI_MS_Now();
   long to_msc = (long)now_time * 1000;
   long from_msc = to_msc - (long)lookback_sec * 1000;
   int copied = CopyTicksRange(symbol, ticks, COPY_TICKS_ALL, from_msc, to_msc);
   if(copied <= 0)
      copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 8192);
   if(copied <= 0)
   {
      ArrayResize(ticks, 0);
      return false;
   }
   if(ArraySize(ticks) >= 2 && ticks[0].time_msc > ticks[ArraySize(ticks) - 1].time_msc)
   {
      int left = 0;
      int right = ArraySize(ticks) - 1;
      while(left < right)
      {
         MqlTick temp = ticks[left];
         ticks[left] = ticks[right];
         ticks[right] = temp;
         left++;
         right--;
      }
   }
   return (ArraySize(ticks) > 0);
}

bool FXAI_MS_BuildWindowStats(const MqlTick &ticks[],
                              const long from_msc,
                              const long to_msc,
                              const double point_value,
                              FXAIMicrostructureWindowStats &stats)
{
   FXAI_MS_ResetWindowStats(stats);
   if(point_value <= 0.0 || ArraySize(ticks) <= 0 || to_msc <= from_msc)
      return false;

   int first = -1;
   int last = -1;
   int n = ArraySize(ticks);
   for(int i = 0; i < n; i++)
   {
      if(ticks[i].time_msc < from_msc)
         continue;
      if(ticks[i].time_msc > to_msc)
         break;
      if(first < 0)
         first = i;
      last = i;
   }
   if(first < 0 || last < first)
      return false;

   stats.tick_count = last - first + 1;
   stats.spread_current = MathMax((ticks[last].ask - ticks[last].bid) / point_value, 0.0);

   double spread_sum = 0.0;
   double spread_sum_sq = 0.0;
   double total_abs_points = 0.0;
   double net_move_points = 0.0;
   double rv_sum_sq = 0.0;
   double max_mid = 0.0;
   double min_mid = 0.0;
   double prev_spread = 0.0;
   bool have_prev_spread = false;

   double first_mid = 0.0;
   double last_mid = 0.0;
   bool have_prev_mid = false;
   double prev_mid = 0.0;
   int prev_sign = 0;
   int current_run = 0;

   for(int i = first; i <= last; i++)
   {
      double bid = ticks[i].bid;
      double ask = ticks[i].ask;
      double mid = 0.5 * (bid + ask);
      double spread_points = MathMax((ask - bid) / point_value, 0.0);

      if(i == first)
      {
         first_mid = mid;
         max_mid = mid;
         min_mid = mid;
      }
      if(mid > max_mid) max_mid = mid;
      if(mid < min_mid) min_mid = mid;
      last_mid = mid;

      spread_sum += spread_points;
      spread_sum_sq += spread_points * spread_points;
      if(have_prev_spread)
      {
         if((spread_points - prev_spread) > MathMax(0.35 * MathMax(prev_spread, 1.0), 0.5))
            stats.spread_widen_events++;
      }
      prev_spread = spread_points;
      have_prev_spread = true;

      if(have_prev_mid)
      {
         double delta_points = (mid - prev_mid) / point_value;
         stats.signed_mid_change_sum_pts += delta_points;
         double abs_points = MathAbs(delta_points);
         total_abs_points += abs_points;
         rv_sum_sq += delta_points * delta_points;

         int sign = (delta_points > 1e-9 ? 1 : (delta_points < -1e-9 ? -1 : 0));
         if(sign > 0)
            stats.up_count++;
         else if(sign < 0)
            stats.down_count++;

         if(sign != 0)
         {
            if(sign == prev_sign)
               current_run++;
            else
               current_run = 1;
            prev_sign = sign;
         }
         if(abs_points >= MathMax(g_ms_cfg.shock_move_points_factor * MathMax(spread_points, 1.0), 2.0))
            stats.shock_move_count++;

         if(ticks[i].bid != ticks[i - 1].bid || ticks[i].ask != ticks[i - 1].ask)
            stats.quote_change_count++;
      }
      else
      {
         current_run = 0;
      }

      prev_mid = mid;
      have_prev_mid = true;
   }

   double duration_sec = MathMax((double)(to_msc - from_msc) / 1000.0, 1.0);
   double mean_spread = spread_sum / (double)MathMax(stats.tick_count, 1);
   double var_spread = (spread_sum_sq / (double)MathMax(stats.tick_count, 1)) - mean_spread * mean_spread;
   if(var_spread < 0.0)
      var_spread = 0.0;
   double spread_std = MathSqrt(var_spread);
   double spread_zscore = 0.0;
   if(spread_std > 1e-6)
      spread_zscore = (stats.spread_current - mean_spread) / spread_std;

   int wide_count = 0;
   for(int i = first; i <= last; i++)
   {
      double spread_points = MathMax((ticks[i].ask - ticks[i].bid) / point_value, 0.0);
      double wide_threshold = MathMax(mean_spread + g_ms_cfg.wide_spread_zscore * spread_std,
                                      g_ms_cfg.wide_spread_absolute_points_floor);
      if(spread_points >= wide_threshold)
         wide_count++;
   }

   stats.spread_mean = mean_spread;
   stats.spread_std = spread_std;
   stats.spread_zscore = spread_zscore;
   stats.wide_spread_fraction = FXAI_MS_Clamp((double)wide_count / (double)MathMax(stats.tick_count, 1), 0.0, 1.0);
   stats.tick_rate = (double)stats.tick_count / duration_sec;
   stats.quote_change_rate = (double)MathMax(stats.quote_change_count, 0) / duration_sec;
   stats.realized_vol = MathSqrt(rv_sum_sq / (double)MathMax(stats.tick_count - 1, 1));
   stats.range_expansion = MathMax(0.0, (max_mid - min_mid) / point_value);
   stats.directional_efficiency = FXAI_MS_Clamp(MathAbs((last_mid - first_mid) / point_value) / MathMax(total_abs_points, 1e-6), 0.0, 1.0);
   stats.directional_run_length_current = current_run;
   stats.total_abs_change_sum_pts = total_abs_points;
   net_move_points = (last_mid - first_mid) / point_value;
   stats.signed_mid_change_sum_pts = net_move_points;
   stats.spread_instability = FXAI_MS_Clamp(0.38 * FXAI_MS_Clamp(spread_std / MathMax(mean_spread, 0.25), 0.0, 4.0) / 2.0 +
                                            0.34 * stats.wide_spread_fraction +
                                            0.28 * FXAI_MS_Clamp((double)stats.spread_widen_events / (double)MathMax(stats.tick_count - 1, 1), 0.0, 1.0),
                                            0.0,
                                            1.0);

   int max_index = first;
   int min_index = first;
   for(int i = first + 1; i <= last; i++)
   {
      double mid = 0.5 * (ticks[i].bid + ticks[i].ask);
      if(mid > max_mid)
      {
         max_mid = mid;
         max_index = i;
      }
      if(mid < min_mid)
      {
         min_mid = mid;
         min_index = i;
      }
   }

   double prev_high_before_max = max_mid;
   if(max_index > first)
   {
      prev_high_before_max = 0.5 * (ticks[first].bid + ticks[first].ask);
      for(int i = first; i < max_index; i++)
      {
         double mid = 0.5 * (ticks[i].bid + ticks[i].ask);
         if(mid > prev_high_before_max)
            prev_high_before_max = mid;
      }
   }

   double prev_low_before_min = min_mid;
   if(min_index > first)
   {
      prev_low_before_min = 0.5 * (ticks[first].bid + ticks[first].ask);
      for(int i = first; i < min_index; i++)
      {
         double mid = 0.5 * (ticks[i].bid + ticks[i].ask);
         if(mid < prev_low_before_min)
            prev_low_before_min = mid;
      }
   }

   double breach_up_pts = MathMax(0.0, (max_mid - prev_high_before_max) / point_value);
   double breach_down_pts = MathMax(0.0, (prev_low_before_min - min_mid) / point_value);
   double breach_norm = FXAI_MS_Clamp(MathMax(breach_up_pts, breach_down_pts) / MathMax(stats.range_expansion, 1.0), 0.0, 1.0);
   double rejection_score = 0.0;
   bool rejection_flag = false;
   double snapback_frac = FXAI_MS_Clamp(g_ms_cfg.stop_run_reversal_fraction, 0.10, 0.90);
   if(breach_up_pts > MathMax(g_ms_cfg.shock_move_points_factor * MathMax(stats.spread_current, 1.0), 2.0))
   {
      double rejection_pts = MathMax(0.0, (max_mid - last_mid) / point_value);
      if(rejection_pts >= breach_up_pts * snapback_frac)
      {
         rejection_flag = true;
         rejection_score = MathMax(rejection_score, FXAI_MS_Clamp(rejection_pts / MathMax(breach_up_pts, 1.0), 0.0, 1.0));
      }
   }
   if(breach_down_pts > MathMax(g_ms_cfg.shock_move_points_factor * MathMax(stats.spread_current, 1.0), 2.0))
   {
      double rejection_pts = MathMax(0.0, (last_mid - min_mid) / point_value);
      if(rejection_pts >= breach_down_pts * snapback_frac)
      {
         rejection_flag = true;
         rejection_score = MathMax(rejection_score, FXAI_MS_Clamp(rejection_pts / MathMax(breach_down_pts, 1.0), 0.0, 1.0));
      }
   }
   stats.local_extrema_breach_score = breach_norm;
   stats.sweep_and_reject_flag = rejection_flag;
   stats.breakout_reversal_score = FXAI_MS_Clamp(0.55 * breach_norm + 0.45 * rejection_score, 0.0, 1.0);
   stats.exhaustion_proxy = FXAI_MS_Clamp(0.46 * stats.breakout_reversal_score +
                                          0.30 * (1.0 - stats.directional_efficiency) +
                                          0.24 * FXAI_MS_Clamp((double)stats.shock_move_count / MathMax((double)stats.tick_count, 1.0), 0.0, 1.0),
                                          0.0,
                                          1.0);
   return true;
}

void FXAI_MS_BaselineStats(const MqlTick &ticks[],
                           const long now_msc,
                           const int window_sec,
                           const double point_value,
                           double &tick_rate_mean,
                           double &tick_rate_std,
                           double &realized_vol_mean,
                           double &realized_vol_std)
{
   tick_rate_mean = 0.0;
   tick_rate_std = 0.0;
   realized_vol_mean = 0.0;
   realized_vol_std = 0.0;

   double tick_values[FXAI_MS_BASELINE_BUCKETS];
   double vol_values[FXAI_MS_BASELINE_BUCKETS];
   int used = 0;
   long span_msc = (long)window_sec * 1000;
   for(int bucket = 1; bucket <= FXAI_MS_BASELINE_BUCKETS; bucket++)
   {
      long bucket_to = now_msc - (long)bucket * span_msc;
      long bucket_from = bucket_to - span_msc;
      FXAIMicrostructureWindowStats stats;
      if(!FXAI_MS_BuildWindowStats(ticks, bucket_from, bucket_to, point_value, stats))
         continue;
      tick_values[used] = stats.tick_rate;
      vol_values[used] = stats.realized_vol;
      used++;
   }
   if(used <= 0)
      return;

   double tick_sum = 0.0, tick_sq = 0.0;
   double vol_sum = 0.0, vol_sq = 0.0;
   for(int i = 0; i < used; i++)
   {
      tick_sum += tick_values[i];
      tick_sq += tick_values[i] * tick_values[i];
      vol_sum += vol_values[i];
      vol_sq += vol_values[i] * vol_values[i];
   }
   tick_rate_mean = tick_sum / (double)used;
   realized_vol_mean = vol_sum / (double)used;
   double tick_var = tick_sq / (double)used - tick_rate_mean * tick_rate_mean;
   double vol_var = vol_sq / (double)used - realized_vol_mean * realized_vol_mean;
   if(tick_var < 0.0) tick_var = 0.0;
   if(vol_var < 0.0) vol_var = 0.0;
   tick_rate_std = MathSqrt(tick_var);
   realized_vol_std = MathSqrt(vol_var);
}

void FXAI_MS_ApplyBurstScores(FXAIMicrostructureWindowStats &stats,
                              const double baseline_tick_mean,
                              const double baseline_tick_std,
                              const double baseline_vol_mean,
                              const double baseline_vol_std)
{
   double tick_den = MathMax(MathMax(baseline_tick_std, baseline_tick_mean * 0.25), 1e-6);
   double vol_den = MathMax(MathMax(baseline_vol_std, baseline_vol_mean * 0.25), 1e-6);
   stats.tick_rate_zscore = FXAI_MS_Clamp((stats.tick_rate - baseline_tick_mean) / tick_den, -8.0, 8.0);
   stats.realized_vol_zscore = FXAI_MS_Clamp((stats.realized_vol - baseline_vol_mean) / vol_den, -8.0, 8.0);
   stats.intensity_burst_score = FXAI_MS_Clamp(stats.tick_rate / MathMax(baseline_tick_mean, 1e-6), 0.0, 8.0);
   stats.vol_burst_score = FXAI_MS_Clamp(stats.realized_vol / MathMax(baseline_vol_mean, 1e-6), 0.0, 8.0);
   stats.exhaustion_proxy = FXAI_MS_Clamp(stats.exhaustion_proxy +
                                          0.18 * FXAI_MS_Clamp(stats.vol_burst_score - 1.0, 0.0, 3.0) / 3.0 +
                                          0.12 * FXAI_MS_Clamp(stats.intensity_burst_score - 1.0, 0.0, 3.0) / 3.0,
                                          0.0,
                                          1.0);
}

void FXAI_MS_ResolveSession(const datetime now_utc,
                            string &session_tag,
                            bool &handoff_flag,
                            int &minutes_since_open,
                            int &minutes_to_close)
{
   MqlDateTime dt;
   TimeToStruct(now_utc, dt);
   int minute_of_day = dt.hour * 60 + dt.min;
   session_tag = "UNKNOWN";
   handoff_flag = false;
   minutes_since_open = -1;
   minutes_to_close = -1;

   for(int i = 0; i < g_ms_cfg.session_count; i++)
   {
      int start_min = g_ms_cfg.session_start_hour[i] * 60;
      int end_min = g_ms_cfg.session_end_hour[i] * 60;
      bool in_session = false;
      if(end_min > start_min)
         in_session = (minute_of_day >= start_min && minute_of_day < end_min);
      else
         in_session = (minute_of_day >= start_min || minute_of_day < end_min);
      if(!in_session)
         continue;
      session_tag = g_ms_cfg.session_labels[i];
      minutes_since_open = minute_of_day - start_min;
      if(minutes_since_open < 0)
         minutes_since_open += 1440;
      minutes_to_close = end_min - minute_of_day;
      if(minutes_to_close < 0)
         minutes_to_close += 1440;
      int handoff = MathMax(g_ms_cfg.handoff_minutes, 5);
      handoff_flag = (minutes_since_open <= handoff || minutes_to_close <= handoff);
      return;
   }
}

void FXAI_MS_ClassifyState(FXAIMicrostructurePairState &state,
                           const int idx_10s,
                           const int idx_30s,
                           const int idx_60s,
                           const int idx_5m)
{
   double trend_eff = (idx_60s >= 0 ? state.directional_efficiency[idx_60s] : 0.0);
   double trend_imbalance = (idx_30s >= 0 ? MathAbs(state.tick_imbalance[idx_30s]) : 0.0);
   double burst_30s = (idx_30s >= 0 ? state.intensity_burst_score[idx_30s] : 1.0);
   double vol_5m = (idx_5m >= 0 ? state.vol_burst_score[idx_5m] : 1.0);
   double sweep_risk = MathMax((idx_60s >= 0 ? state.breakout_reversal_score[idx_60s] : 0.0),
                               (idx_30s >= 0 ? state.exhaustion_proxy[idx_30s] : 0.0));
   double spread_instability = (idx_60s >= 0 ? state.spread_instability[idx_60s] : 0.0);
   double spread_level_stress = (idx_60s >= 0
                                 ? FXAI_MS_Clamp(MathMax(state.spread_zscore[idx_60s], 0.0) / 3.0, 0.0, 1.0)
                                 : 0.0);
   double wide_fraction = (idx_60s >= 0 ? state.wide_spread_fraction[idx_60s] : 0.0);
   bool thin_and_wide_flag = (spread_instability >= g_ms_cfg.spread_instability_block ||
                              (spread_level_stress >= 0.45 && wide_fraction >= 0.10) ||
                              (state.silent_gap_seconds_current >= 8.0 && spread_level_stress >= 0.35));

   state.liquidity_stress_score = FXAI_MS_Clamp(0.24 * spread_instability +
                                                0.18 * spread_level_stress +
                                                0.20 * state.session_spread_behavior_score +
                                                0.15 * FXAI_MS_Clamp(vol_5m - 1.0, 0.0, 3.0) / 3.0 +
                                                0.13 * FXAI_MS_Clamp(burst_30s - 1.0, 0.0, 3.0) / 3.0 +
                                                0.10 * FXAI_MS_Clamp(state.silent_gap_seconds_current / 10.0, 0.0, 1.0),
                                                0.0,
                                                1.0);
   state.hostile_execution_score = FXAI_MS_Clamp(0.34 * state.liquidity_stress_score +
                                                 0.18 * spread_instability +
                                                 0.12 * spread_level_stress +
                                                 0.18 * sweep_risk +
                                                 0.10 * (state.handoff_flag ? 1.0 : 0.0) +
                                                 0.08 * FXAI_MS_Clamp(vol_5m - 1.0, 0.0, 3.0) / 3.0,
                                                 0.0,
                                                 1.0);

   if(state.hostile_execution_score >= g_ms_cfg.hostile_execution_block ||
      state.liquidity_stress_score >= g_ms_cfg.liquidity_stress_block)
   {
      state.trade_gate = "BLOCK";
   }
   else if(state.hostile_execution_score >= g_ms_cfg.hostile_execution_caution ||
           state.liquidity_stress_score >= g_ms_cfg.liquidity_stress_caution)
   {
      state.trade_gate = "CAUTION";
   }
   else
   {
      state.trade_gate = "ALLOW";
   }
   if(thin_and_wide_flag && state.trade_gate == "ALLOW")
      state.trade_gate = "CAUTION";

   if(thin_and_wide_flag)
      state.microstructure_regime = "THIN_AND_WIDE";
   else if(sweep_risk >= g_ms_cfg.stop_run_rejection_score_flag)
      state.microstructure_regime = "STOP_RUN_RISK";
   else if(trend_eff >= g_ms_cfg.clean_trend_efficiency_floor &&
           trend_imbalance >= g_ms_cfg.clean_trend_imbalance_floor &&
           state.hostile_execution_score < 0.40)
      state.microstructure_regime = "TRENDING_CLEAN";
   else if(trend_eff >= g_ms_cfg.clean_trend_efficiency_floor &&
           state.hostile_execution_score >= 0.40)
      state.microstructure_regime = "TRENDING_FRAGILE";
   else if((burst_30s >= g_ms_cfg.tick_burst_ratio_caution && trend_eff < 0.45) ||
           (vol_5m >= g_ms_cfg.vol_burst_ratio_caution && trend_eff < 0.45))
      state.microstructure_regime = "CHOPPY_HIGH_ACTIVITY";
   else if((burst_30s >= g_ms_cfg.tick_burst_ratio_block && vol_5m >= g_ms_cfg.vol_burst_ratio_caution) ||
           state.session_open_burst_score >= 0.65)
      state.microstructure_regime = "VOLATILE_NEWSLIKE";
   else
      state.microstructure_regime = "NORMAL";

   if(state.trade_gate == "BLOCK")
      FXAI_MS_AppendReason(state, "hostile execution block threshold exceeded");
   else if(state.trade_gate == "CAUTION")
      FXAI_MS_AppendReason(state, "hostile execution caution threshold exceeded");
   if(spread_instability >= g_ms_cfg.spread_instability_caution)
      FXAI_MS_AppendReason(state, "spread instability elevated");
   if(thin_and_wide_flag)
      FXAI_MS_AppendReason(state, "current spread regime is abnormally wide");
   if(idx_30s >= 0 && state.intensity_burst_score[idx_30s] >= g_ms_cfg.tick_burst_ratio_caution)
      FXAI_MS_AppendReason(state, "tick intensity burst above baseline");
   if(idx_5m >= 0 && state.vol_burst_score[idx_5m] >= g_ms_cfg.vol_burst_ratio_caution)
      FXAI_MS_AppendReason(state, "realized volatility burst above baseline");
   if(idx_60s >= 0 && state.sweep_and_reject_flag[idx_60s])
      FXAI_MS_AppendReason(state, "recent breakout rejection detected");
   if(state.handoff_flag && state.session_open_burst_score >= 0.45)
      FXAI_MS_AppendReason(state, "session handoff burst active");
}

bool FXAI_MS_ComputePairState(const FXAIMicrostructureResolvedSymbol &entry,
                              const datetime now_server,
                              FXAIMicrostructurePairState &state)
{
   FXAI_MS_ResetPairState(state);
   state.canonical_pair = entry.canonical_pair;
   state.broker_symbol = entry.broker_symbol;
   state.generated_at = now_server;
   state.stale = true;

   if(!entry.active || StringLen(entry.broker_symbol) <= 0 || entry.point <= 0.0)
      return false;

   MqlTick ticks[];
   if(!FXAI_MS_LoadTicks(entry.broker_symbol, g_ms_cfg.max_history_window_sec, ticks))
      return false;

   if(ArraySize(ticks) < 2)
      return false;

   datetime now_utc = TimeGMT();
   if(now_utc <= 0)
      now_utc = now_server;
   FXAI_MS_ResolveSession(now_utc,
                          state.session_tag,
                          state.handoff_flag,
                          state.minutes_since_session_open,
                          state.minutes_to_session_close);

   long now_msc = (long)now_server * 1000;
   state.silent_gap_seconds_current = FXAI_MS_Clamp((double)(now_msc - ticks[ArraySize(ticks) - 1].time_msc) / 1000.0, 0.0, 3600.0);
   state.available = true;
   state.stale = false;

   int idx_10s = -1;
   int idx_30s = -1;
   int idx_60s = -1;
   int idx_5m = -1;
   for(int w = 0; w < g_ms_cfg.window_count; w++)
   {
      int window_sec = g_ms_cfg.windows_sec[w];
      long from_msc = now_msc - (long)window_sec * 1000;
      FXAIMicrostructureWindowStats stats;
      if(!FXAI_MS_BuildWindowStats(ticks, from_msc, now_msc, entry.point, stats))
         continue;

      double baseline_tick_mean = 0.0, baseline_tick_std = 0.0;
      double baseline_vol_mean = 0.0, baseline_vol_std = 0.0;
      FXAI_MS_BaselineStats(ticks,
                            now_msc,
                            window_sec,
                            entry.point,
                            baseline_tick_mean,
                            baseline_tick_std,
                            baseline_vol_mean,
                            baseline_vol_std);
      FXAI_MS_ApplyBurstScores(stats,
                               baseline_tick_mean,
                               baseline_tick_std,
                               baseline_vol_mean,
                               baseline_vol_std);

      state.tick_up_count[w] = stats.up_count;
      state.tick_down_count[w] = stats.down_count;
      state.tick_count[w] = stats.tick_count;
      state.tick_imbalance[w] = FXAI_MS_Clamp((double)(stats.up_count - stats.down_count) / (double)MathMax(stats.up_count + stats.down_count, 1), -1.0, 1.0);
      state.signed_mid_change_sum_pts[w] = stats.signed_mid_change_sum_pts;
      state.directional_efficiency[w] = stats.directional_efficiency;
      state.spread_mean[w] = stats.spread_mean;
      state.spread_std[w] = stats.spread_std;
      state.spread_zscore[w] = stats.spread_zscore;
      state.spread_widen_events[w] = stats.spread_widen_events;
      state.spread_instability[w] = stats.spread_instability;
      state.wide_spread_fraction[w] = stats.wide_spread_fraction;
      state.tick_rate[w] = stats.tick_rate;
      state.tick_rate_zscore[w] = stats.tick_rate_zscore;
      state.intensity_burst_score[w] = stats.intensity_burst_score;
      state.quote_change_rate[w] = stats.quote_change_rate;
      state.realized_vol[w] = stats.realized_vol;
      state.realized_vol_zscore[w] = stats.realized_vol_zscore;
      state.vol_burst_score[w] = stats.vol_burst_score;
      state.range_expansion[w] = stats.range_expansion;
      state.shock_move_count[w] = stats.shock_move_count;
      state.local_extrema_breach_score[w] = stats.local_extrema_breach_score;
      state.sweep_and_reject_flag[w] = stats.sweep_and_reject_flag;
      state.breakout_reversal_score[w] = stats.breakout_reversal_score;
      state.exhaustion_proxy[w] = stats.exhaustion_proxy;
      state.directional_run_length_current[w] = stats.directional_run_length_current;
      if(window_sec == 10) idx_10s = w;
      if(window_sec == 30) idx_30s = w;
      if(window_sec == 60) idx_60s = w;
      if(window_sec == 300) idx_5m = w;
   }

   if(idx_60s >= 0)
      state.spread_current = state.spread_mean[idx_60s] + state.spread_zscore[idx_60s] * state.spread_std[idx_60s];
   else
      state.spread_current = MathMax((ticks[ArraySize(ticks) - 1].ask - ticks[ArraySize(ticks) - 1].bid) / entry.point, 0.0);

   state.session_open_burst_score = 0.0;
   state.session_spread_behavior_score = 0.0;
   if(state.handoff_flag)
   {
      double burst_factor = (idx_30s >= 0 ? FXAI_MS_Clamp(state.intensity_burst_score[idx_30s] - 1.0, 0.0, 3.0) / 3.0 : 0.0);
      double spread_factor = (idx_60s >= 0 ? state.spread_instability[idx_60s] : 0.0);
      state.session_open_burst_score = FXAI_MS_Clamp(0.62 * burst_factor + 0.38 * (idx_5m >= 0 ? FXAI_MS_Clamp(state.vol_burst_score[idx_5m] - 1.0, 0.0, 3.0) / 3.0 : 0.0), 0.0, 1.0);
      state.session_spread_behavior_score = FXAI_MS_Clamp(0.58 * spread_factor + 0.42 * burst_factor, 0.0, 1.0);
   }

   FXAI_MS_ClassifyState(state, idx_10s, idx_30s, idx_60s, idx_5m);
   return true;
}

string FXAI_MS_StateToJSON(const FXAIMicrostructurePairState &state)
{
   string json = "{";
   json += "\"symbol\":\"" + FXAI_MS_JSONEscape(state.canonical_pair) + "\"";
   json += ",\"broker_symbol\":\"" + FXAI_MS_JSONEscape(state.broker_symbol) + "\"";
   json += ",\"available\":" + IntegerToString(state.available ? 1 : 0);
   json += ",\"stale\":" + IntegerToString(state.stale ? 1 : 0);
   json += ",\"generated_at\":\"" + FXAI_MS_ISO8601(state.generated_at) + "\"";
   json += ",\"spread_current\":" + DoubleToString(state.spread_current, 6);
   json += ",\"silent_gap_seconds_current\":" + DoubleToString(state.silent_gap_seconds_current, 3);
   json += ",\"session_tag\":\"" + FXAI_MS_JSONEscape(state.session_tag) + "\"";
   json += ",\"handoff_flag\":" + IntegerToString(state.handoff_flag ? 1 : 0);
   json += ",\"minutes_since_session_open\":" + IntegerToString(state.minutes_since_session_open);
   json += ",\"minutes_to_session_close\":" + IntegerToString(state.minutes_to_session_close);
   json += ",\"session_open_burst_score\":" + DoubleToString(state.session_open_burst_score, 6);
   json += ",\"session_spread_behavior_score\":" + DoubleToString(state.session_spread_behavior_score, 6);
   json += ",\"liquidity_stress_score\":" + DoubleToString(state.liquidity_stress_score, 6);
   json += ",\"hostile_execution_score\":" + DoubleToString(state.hostile_execution_score, 6);
   json += ",\"microstructure_regime\":\"" + FXAI_MS_JSONEscape(state.microstructure_regime) + "\"";
   json += ",\"trade_gate\":\"" + FXAI_MS_JSONEscape(state.trade_gate) + "\"";
   for(int w = 0; w < g_ms_cfg.window_count; w++)
   {
      string suffix = FXAI_MS_WindowSuffix(g_ms_cfg.windows_sec[w]);
      json += ",\"tick_up_count_" + suffix + "\":" + IntegerToString(state.tick_up_count[w]);
      json += ",\"tick_down_count_" + suffix + "\":" + IntegerToString(state.tick_down_count[w]);
      json += ",\"tick_count_" + suffix + "\":" + IntegerToString(state.tick_count[w]);
      json += ",\"tick_imbalance_" + suffix + "\":" + DoubleToString(state.tick_imbalance[w], 6);
      json += ",\"signed_mid_change_sum_" + suffix + "\":" + DoubleToString(state.signed_mid_change_sum_pts[w], 6);
      json += ",\"directional_efficiency_" + suffix + "\":" + DoubleToString(state.directional_efficiency[w], 6);
      json += ",\"directional_run_length_current_" + suffix + "\":" + IntegerToString(state.directional_run_length_current[w]);
      json += ",\"spread_mean_" + suffix + "\":" + DoubleToString(state.spread_mean[w], 6);
      json += ",\"spread_std_" + suffix + "\":" + DoubleToString(state.spread_std[w], 6);
      json += ",\"spread_zscore_" + suffix + "\":" + DoubleToString(state.spread_zscore[w], 6);
      json += ",\"spread_widen_events_" + suffix + "\":" + IntegerToString(state.spread_widen_events[w]);
      json += ",\"spread_instability_" + suffix + "\":" + DoubleToString(state.spread_instability[w], 6);
      json += ",\"wide_spread_fraction_" + suffix + "\":" + DoubleToString(state.wide_spread_fraction[w], 6);
      json += ",\"tick_rate_" + suffix + "\":" + DoubleToString(state.tick_rate[w], 6);
      json += ",\"tick_rate_zscore_" + suffix + "\":" + DoubleToString(state.tick_rate_zscore[w], 6);
      json += ",\"intensity_burst_score_" + suffix + "\":" + DoubleToString(state.intensity_burst_score[w], 6);
      json += ",\"quote_change_rate_" + suffix + "\":" + DoubleToString(state.quote_change_rate[w], 6);
      json += ",\"realized_vol_" + suffix + "\":" + DoubleToString(state.realized_vol[w], 6);
      json += ",\"realized_vol_zscore_" + suffix + "\":" + DoubleToString(state.realized_vol_zscore[w], 6);
      json += ",\"vol_burst_score_" + suffix + "\":" + DoubleToString(state.vol_burst_score[w], 6);
      json += ",\"range_expansion_" + suffix + "\":" + DoubleToString(state.range_expansion[w], 6);
      json += ",\"shock_move_count_" + suffix + "\":" + IntegerToString(state.shock_move_count[w]);
      json += ",\"local_extrema_breach_score_" + suffix + "\":" + DoubleToString(state.local_extrema_breach_score[w], 6);
      json += ",\"sweep_and_reject_flag_" + suffix + "\":" + IntegerToString(state.sweep_and_reject_flag[w] ? 1 : 0);
      json += ",\"breakout_reversal_score_" + suffix + "\":" + DoubleToString(state.breakout_reversal_score[w], 6);
      json += ",\"exhaustion_proxy_" + suffix + "\":" + DoubleToString(state.exhaustion_proxy[w], 6);
   }
   json += ",\"reasons\":[";
   for(int i = 0; i < state.reason_count; i++)
   {
      if(i > 0)
         json += ",";
      json += "\"" + FXAI_MS_JSONEscape(state.reasons[i]) + "\"";
   }
   json += "]}";
   return json;
}

void FXAI_MS_WriteFlat(const FXAIMicrostructurePairState &states[],
                       const datetime generated_at)
{
   FXAI_MS_EnsureFolders();
   int handle = FileOpen(FXAI_MS_FLAT_FILE,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   FileWriteString(handle, "meta\tglobal\tgenerated_at_unix\t" + IntegerToString((int)generated_at) + "\r\n");
   FileWriteString(handle, "meta\tglobal\tschema_version\t" + IntegerToString(FXAI_MS_SCHEMA_VERSION) + "\r\n");
   for(int i = 0; i < ArraySize(states); i++)
   {
      FXAIMicrostructurePairState state = states[i];
      if(!state.available)
         continue;
      string pair = state.canonical_pair;
      FileWriteString(handle, "pair\t" + pair + "\tstale\t" + IntegerToString(state.stale ? 1 : 0) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tspread_current\t" + DoubleToString(state.spread_current, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tsilent_gap_seconds_current\t" + DoubleToString(state.silent_gap_seconds_current, 3) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tsession_tag\t" + state.session_tag + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\thandoff_flag\t" + IntegerToString(state.handoff_flag ? 1 : 0) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tsession_open_burst_score\t" + DoubleToString(state.session_open_burst_score, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tsession_spread_behavior_score\t" + DoubleToString(state.session_spread_behavior_score, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tliquidity_stress_score\t" + DoubleToString(state.liquidity_stress_score, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\thostile_execution_score\t" + DoubleToString(state.hostile_execution_score, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tmicrostructure_regime\t" + state.microstructure_regime + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\ttrade_gate\t" + state.trade_gate + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tcaution_lot_scale\t" + DoubleToString(g_ms_cfg.caution_lot_scale, 6) + "\r\n");
      FileWriteString(handle, "pair\t" + pair + "\tcaution_enter_prob_buffer\t" + DoubleToString(g_ms_cfg.caution_enter_prob_buffer, 6) + "\r\n");
      for(int w = 0; w < g_ms_cfg.window_count; w++)
      {
         string suffix = FXAI_MS_WindowSuffix(g_ms_cfg.windows_sec[w]);
         FileWriteString(handle, "pair\t" + pair + "\ttick_imbalance_" + suffix + "\t" + DoubleToString(state.tick_imbalance[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tdirectional_efficiency_" + suffix + "\t" + DoubleToString(state.directional_efficiency[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tspread_zscore_" + suffix + "\t" + DoubleToString(state.spread_zscore[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\ttick_rate_" + suffix + "\t" + DoubleToString(state.tick_rate[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\ttick_rate_zscore_" + suffix + "\t" + DoubleToString(state.tick_rate_zscore[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\trealized_vol_" + suffix + "\t" + DoubleToString(state.realized_vol[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tvol_burst_score_" + suffix + "\t" + DoubleToString(state.vol_burst_score[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tlocal_extrema_breach_score_" + suffix + "\t" + DoubleToString(state.local_extrema_breach_score[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tsweep_and_reject_flag_" + suffix + "\t" + IntegerToString(state.sweep_and_reject_flag[w] ? 1 : 0) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\tbreakout_reversal_score_" + suffix + "\t" + DoubleToString(state.breakout_reversal_score[w], 6) + "\r\n");
         FileWriteString(handle, "pair\t" + pair + "\texhaustion_proxy_" + suffix + "\t" + DoubleToString(state.exhaustion_proxy[w], 6) + "\r\n");
      }
      for(int r = 0; r < state.reason_count; r++)
         FileWriteString(handle, "pair_reason\t" + pair + "\treason\t" + state.reasons[r] + "\r\n");
   }
   FileClose(handle);
}

void FXAI_MS_WriteHistoryIfNeeded(const int index,
                                  const FXAIMicrostructurePairState &state)
{
   if(index < 0 || index >= ArraySize(g_ms_history_last_at) || !state.available)
      return;

   bool should_write = false;
   if(g_ms_history_last_at[index] <= 0)
      should_write = true;
   else if(state.trade_gate != g_ms_history_last_gate[index])
      should_write = true;
   else if(state.microstructure_regime != g_ms_history_last_regime[index])
      should_write = true;
   else if(MathAbs(state.hostile_execution_score - g_ms_history_last_hostile[index]) >= 0.08)
      should_write = true;
   else if((state.generated_at - g_ms_history_last_at[index]) >= FXAI_MS_HISTORY_HEARTBEAT_SEC)
      should_write = true;

   if(!should_write)
      return;

   int handle = FileOpen(FXAI_MS_HISTORY_FILE,
                         FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      handle = FileOpen(FXAI_MS_HISTORY_FILE,
                        FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                        FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;
   FileSeek(handle, 0, SEEK_END);
   string line = "{\"schema_version\":" + IntegerToString(FXAI_MS_SCHEMA_VERSION) +
                 ",\"generated_at\":\"" + FXAI_MS_ISO8601(state.generated_at) + "\"" +
                 ",\"symbol\":\"" + FXAI_MS_JSONEscape(state.canonical_pair) + "\"" +
                 ",\"state\":" + FXAI_MS_StateToJSON(state) + "}\r\n";
   FileWriteString(handle, line);
   FileClose(handle);

   g_ms_history_last_at[index] = state.generated_at;
   g_ms_history_last_gate[index] = state.trade_gate;
   g_ms_history_last_regime[index] = state.microstructure_regime;
   g_ms_history_last_hostile[index] = state.hostile_execution_score;
}

void FXAI_MS_WriteJSONArtifacts(const FXAIMicrostructurePairState &states[],
                                const datetime generated_at,
                                const bool ok,
                                const string last_error)
{
   FXAI_MS_EnsureFolders();
   string json = "{";
   json += "\"schema_version\":" + IntegerToString(FXAI_MS_SCHEMA_VERSION);
   json += ",\"generated_at\":\"" + FXAI_MS_ISO8601(generated_at) + "\"";
   json += ",\"service\":{";
   json += "\"ok\":" + IntegerToString(ok ? 1 : 0);
   json += ",\"stale\":0";
   json += ",\"enabled\":" + IntegerToString(g_ms_cfg.enabled ? 1 : 0);
   json += ",\"poll_interval_ms\":" + IntegerToString(g_ms_cfg.poll_interval_ms);
   json += ",\"symbol_refresh_sec\":" + IntegerToString(g_ms_cfg.symbol_refresh_sec);
   json += ",\"snapshot_stale_after_sec\":" + IntegerToString(g_ms_cfg.snapshot_stale_after_sec);
   json += ",\"last_poll_at\":\"" + FXAI_MS_ISO8601(g_ms_last_poll_at) + "\"";
   json += ",\"last_success_at\":\"" + FXAI_MS_ISO8601(g_ms_last_success_at) + "\"";
   json += ",\"last_symbol_refresh_at\":\"" + FXAI_MS_ISO8601(g_ms_last_symbol_refresh) + "\"";
   json += ",\"last_error\":\"" + FXAI_MS_JSONEscape(last_error) + "\"";
   json += "}";
   json += ",\"health\":{";
   json += "\"resolved_symbol_count\":" + IntegerToString(ArraySize(g_ms_symbols));
   json += ",\"active_symbol_count\":";
   int active_count = 0;
   for(int i = 0; i < ArraySize(g_ms_symbols); i++)
      if(g_ms_symbols[i].active) active_count++;
   json += IntegerToString(active_count);
   json += ",\"snapshot_stale_after_sec\":" + IntegerToString(g_ms_cfg.snapshot_stale_after_sec);
   json += "}";
   json += ",\"symbols\":{";
   bool first_symbol = true;
   for(int i = 0; i < ArraySize(states); i++)
   {
      if(!states[i].available)
         continue;
      if(!first_symbol)
         json += ",";
      json += "\"" + FXAI_MS_JSONEscape(states[i].canonical_pair) + "\":" + FXAI_MS_StateToJSON(states[i]);
      first_symbol = false;
   }
   json += "}";
   json += "}";

   int handle_snapshot = FileOpen(FXAI_MS_SNAPSHOT_FILE,
                                  FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                                  FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle_snapshot != INVALID_HANDLE)
   {
      FileWriteString(handle_snapshot, json);
      FileClose(handle_snapshot);
   }
   int handle_status = FileOpen(FXAI_MS_STATUS_FILE,
                                FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                                FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle_status != INVALID_HANDLE)
   {
      FileWriteString(handle_status, json);
      FileClose(handle_status);
   }
}

bool FXAI_MS_RunCycle(void)
{
   g_ms_last_poll_at = FXAI_MS_Now();
   FXAI_MS_LoadConfig(g_ms_cfg);
   FXAI_MS_RefreshResolvedSymbols(false);
   FXAI_MS_WriteSymbolMap();

   FXAIMicrostructurePairState states[];
   ArrayResize(states, ArraySize(g_ms_symbols));
   bool any_ok = false;
   string last_error = "";
   datetime now_time = FXAI_MS_Now();

   for(int i = 0; i < ArraySize(g_ms_symbols); i++)
   {
      FXAI_MS_ResetPairState(states[i]);
      if(!FXAI_MS_ComputePairState(g_ms_symbols[i], now_time, states[i]))
      {
         states[i].canonical_pair = g_ms_symbols[i].canonical_pair;
         states[i].broker_symbol = g_ms_symbols[i].broker_symbol;
         states[i].generated_at = now_time;
         states[i].stale = true;
         states[i].trade_gate = "BLOCK";
         states[i].microstructure_regime = "UNKNOWN";
         FXAI_MS_AppendReason(states[i], "microstructure snapshot unavailable");
         last_error = "one_or_more_symbols_unavailable";
         continue;
      }
      any_ok = true;
      FXAI_MS_WriteHistoryIfNeeded(i, states[i]);
   }

   if(any_ok)
      g_ms_last_success_at = now_time;
   g_ms_last_error = last_error;
   FXAI_MS_WriteFlat(states, now_time);
   FXAI_MS_WriteJSONArtifacts(states, now_time, any_ok, last_error);
   return any_ok;
}

int OnStart()
{
   FXAI_MS_EnsureFolders();
   FXAI_MS_LoadConfig(g_ms_cfg);
   FXAI_MS_RefreshResolvedSymbols(true);
   FXAI_MS_WriteSymbolMap();

   while(!IsStopped())
   {
      FXAI_MS_RunCycle();
      int sleep_ms = (g_ms_cfg.poll_interval_ms > 0 ? g_ms_cfg.poll_interval_ms : 5000);
      int slices = MathMax(sleep_ms / 250, 1);
      for(int i = 0; i < slices && !IsStopped(); i++)
         Sleep(250);
   }
   return 0;
}
