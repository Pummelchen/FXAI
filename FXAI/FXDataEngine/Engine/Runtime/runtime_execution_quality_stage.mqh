#ifndef __FXAI_RUNTIME_EXECUTION_QUALITY_STAGE_MQH__
#define __FXAI_RUNTIME_EXECUTION_QUALITY_STAGE_MQH__

#include "Trade\\runtime_trade_cross_asset_state.mqh"

#ifndef FXAI_EXEC_QUALITY_MAX_REASONS
#define FXAI_EXEC_QUALITY_MAX_REASONS 12
#endif
#define FXAI_EXEC_QUALITY_MAX_BUCKETS 5
#define FXAI_EXEC_QUALITY_MAX_TIERS 128

struct FXAIExecutionQualityConfig
{
   bool   ready;
   bool   enabled;
   bool   block_on_unknown;
   bool   allow_block_state;
   int    support_soft_floor;
   int    support_hard_floor;
   int    memory_stale_after_hours;
   double threshold_normal_min;
   double threshold_caution_min;
   double threshold_stressed_min;
   double lot_scale_normal;
   double lot_scale_caution;
   double lot_scale_stressed;
   double lot_scale_blocked;
   double enter_prob_buffer_normal;
   double enter_prob_buffer_caution;
   double enter_prob_buffer_stressed;
   double enter_prob_buffer_blocked;
   double cap_spread_expected_mult;
   double cap_expected_slippage_points;
   double cap_allowed_deviation_points_min;
   double cap_allowed_deviation_points_max;
   double weight_spread_zscore;
   double weight_news_risk;
   double weight_rates_risk;
   double weight_micro_liquidity;
   double weight_micro_hostile;
   double weight_volatility_burst;
   double weight_tick_rate_burst;
   double weight_session_thinness;
   double weight_broker_reject;
   double weight_broker_partial;
   double weight_broker_latency;
   double weight_broker_event_burst;
   double weight_stale_context;
   double weight_support_shortfall;
   int    bucket_count;
   string bucket_hierarchy[FXAI_EXEC_QUALITY_MAX_BUCKETS];
};

struct FXAIExecutionQualityTier
{
   bool   ready;
   string kind;
   string symbol;
   string session;
   string regime;
   int    support;
   double quality;
   double spread_mult;
   double slippage_mult;
   double fill_quality_bias;
   double latency_mult;
   double fragility_mult;
   double deviation_mult;
};

struct FXAIExecutionQualityRuntimeState
{
   bool     ready;
   bool     fallback_used;
   bool     memory_stale;
   bool     data_stale;
   bool     support_usable;
   bool     news_window_active;
   bool     rates_repricing_active;
   datetime generated_at;
   string   symbol;
   string   method;
   string   session_label;
   string   regime_label;
   string   selected_tier_kind;
   string   selected_tier_key;
   int      selected_support;
   double   selected_quality;
   double   broker_coverage;
   double   broker_reject_prob;
   double   broker_partial_fill_prob;
   double   spread_now_points;
   double   spread_expected_points;
   double   spread_widening_risk;
   double   expected_slippage_points;
   double   slippage_risk;
   double   fill_quality_score;
   double   latency_sensitivity_score;
   double   liquidity_fragility_score;
   double   execution_quality_score;
   double   allowed_deviation_points;
   double   caution_lot_scale;
   double   caution_enter_prob_buffer;
   string   execution_state;
   int      reason_count;
   string   reason_codes[FXAI_EXEC_QUALITY_MAX_REASONS];
};

FXAIExecutionQualityConfig g_exec_quality_cfg_cache;
datetime g_exec_quality_cfg_cache_loaded_at = 0;
FXAIExecutionQualityTier g_exec_quality_tiers[];
int      g_exec_quality_tier_count = 0;
datetime g_exec_quality_memory_loaded_at = 0;
datetime g_exec_quality_memory_generated_at = 0;
string   g_exec_quality_memory_method = "SCORECARD_V1";

string FXAI_ExecutionQualityConfigFile(void)
{
   return "FXAI\\Runtime\\execution_quality_config.tsv";
}

string FXAI_ExecutionQualityMemoryFile(void)
{
   return "FXAI\\Runtime\\execution_quality_memory.tsv";
}

string FXAI_ExecutionQualityRuntimeStateFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_execution_quality_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_ExecutionQualityRuntimeHistoryFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_execution_quality_history_" + FXAI_ControlPlaneSafeToken(symbol) + ".ndjson";
}

string FXAI_ExecutionQualityISO8601(const datetime value)
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

datetime FXAI_ExecutionQualityParseISO8601(const string raw)
{
   string text = raw;
   if(StringLen(text) < 19)
      return 0;
   MqlDateTime dt;
   ZeroMemory(dt);
   dt.year = (int)StringToInteger(StringSubstr(text, 0, 4));
   dt.mon = (int)StringToInteger(StringSubstr(text, 5, 2));
   dt.day = (int)StringToInteger(StringSubstr(text, 8, 2));
   dt.hour = (int)StringToInteger(StringSubstr(text, 11, 2));
   dt.min = (int)StringToInteger(StringSubstr(text, 14, 2));
   dt.sec = (int)StringToInteger(StringSubstr(text, 17, 2));
   if(dt.year < 2000 || dt.mon < 1 || dt.mon > 12 || dt.day < 1 || dt.day > 31)
      return 0;
   return StructToTime(dt);
}

string FXAI_ExecutionQualityJSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   return out;
}

void FXAI_ResetExecutionQualityConfig(FXAIExecutionQualityConfig &out)
{
   out.ready = true;
   out.enabled = true;
   out.block_on_unknown = true;
   out.allow_block_state = true;
   out.support_soft_floor = 64;
   out.support_hard_floor = 16;
   out.memory_stale_after_hours = 168;
   out.threshold_normal_min = 0.72;
   out.threshold_caution_min = 0.54;
   out.threshold_stressed_min = 0.36;
   out.lot_scale_normal = 1.00;
   out.lot_scale_caution = 0.82;
   out.lot_scale_stressed = 0.58;
   out.lot_scale_blocked = 0.00;
   out.enter_prob_buffer_normal = 0.00;
   out.enter_prob_buffer_caution = 0.04;
   out.enter_prob_buffer_stressed = 0.08;
   out.enter_prob_buffer_blocked = 1.00;
   out.cap_spread_expected_mult = 4.50;
   out.cap_expected_slippage_points = 18.0;
   out.cap_allowed_deviation_points_min = 2.0;
   out.cap_allowed_deviation_points_max = 25.0;
   out.weight_spread_zscore = 0.22;
   out.weight_news_risk = 0.18;
   out.weight_rates_risk = 0.10;
   out.weight_micro_liquidity = 0.18;
   out.weight_micro_hostile = 0.18;
   out.weight_volatility_burst = 0.14;
   out.weight_tick_rate_burst = 0.12;
   out.weight_session_thinness = 0.10;
   out.weight_broker_reject = 0.16;
   out.weight_broker_partial = 0.14;
   out.weight_broker_latency = 0.14;
   out.weight_broker_event_burst = 0.12;
   out.weight_stale_context = 0.10;
   out.weight_support_shortfall = 0.08;
   out.bucket_count = 5;
   out.bucket_hierarchy[0] = "PAIR_SESSION_REGIME";
   out.bucket_hierarchy[1] = "PAIR_REGIME";
   out.bucket_hierarchy[2] = "SESSION_REGIME";
   out.bucket_hierarchy[3] = "REGIME";
   out.bucket_hierarchy[4] = "GLOBAL";
}

void FXAI_ResetExecutionQualityTier(FXAIExecutionQualityTier &out)
{
   out.ready = false;
   out.kind = "GLOBAL";
   out.symbol = "*";
   out.session = "*";
   out.regime = "*";
   out.support = 0;
   out.quality = 0.34;
   out.spread_mult = 1.08;
   out.slippage_mult = 1.12;
   out.fill_quality_bias = -0.06;
   out.latency_mult = 1.08;
   out.fragility_mult = 1.10;
   out.deviation_mult = 1.06;
}

void FXAI_ResetExecutionQualityRuntimeState(FXAIExecutionQualityRuntimeState &out)
{
   out.ready = false;
   out.fallback_used = false;
   out.memory_stale = true;
   out.data_stale = true;
   out.support_usable = false;
   out.news_window_active = false;
   out.rates_repricing_active = false;
   out.generated_at = 0;
   out.symbol = "";
   out.method = "SCORECARD_V1";
   out.session_label = "UNKNOWN";
   out.regime_label = "UNKNOWN";
   out.selected_tier_kind = "GLOBAL";
   out.selected_tier_key = "GLOBAL|*|*|*";
   out.selected_support = 0;
   out.selected_quality = 0.0;
   out.broker_coverage = 0.0;
   out.broker_reject_prob = 0.0;
   out.broker_partial_fill_prob = 0.0;
   out.spread_now_points = 0.0;
   out.spread_expected_points = 0.0;
   out.spread_widening_risk = 0.0;
   out.expected_slippage_points = 0.0;
   out.slippage_risk = 0.0;
   out.fill_quality_score = 0.0;
   out.latency_sensitivity_score = 0.0;
   out.liquidity_fragility_score = 0.0;
   out.execution_quality_score = 0.0;
   out.allowed_deviation_points = 0.0;
   out.caution_lot_scale = 1.0;
   out.caution_enter_prob_buffer = 0.0;
   out.execution_state = "UNKNOWN";
   out.reason_count = 0;
   for(int i=0; i<FXAI_EXEC_QUALITY_MAX_REASONS; i++)
      out.reason_codes[i] = "";
}

void FXAI_ExecutionQualityAppendReason(FXAIExecutionQualityRuntimeState &state,
                                       const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reason_codes[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_EXEC_QUALITY_MAX_REASONS)
      return;
   state.reason_codes[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_ExecutionQualityReasonsCSV(const FXAIExecutionQualityRuntimeState &state)
{
   string joined = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reason_codes[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reason_codes[i];
   }
   return joined;
}

string FXAI_ExecutionQualityTierKey(const FXAIExecutionQualityTier &tier)
{
   return tier.kind + "|" + tier.symbol + "|" + tier.session + "|" + tier.regime;
}

string FXAI_ExecutionQualitySessionLabel(const FXAINewsPulsePairState &news_state,
                                         const FXAIMicrostructurePairState &micro_state)
{
   string label = "";
   if(micro_state.ready && StringLen(micro_state.session_tag) > 0)
      label = micro_state.session_tag;
   else if(news_state.ready && StringLen(news_state.session_profile) > 0)
      label = news_state.session_profile;
   if(StringLen(label) <= 0)
      label = FXAI_AdaptiveRouterSessionLabel(TimeCurrent());
   StringToUpper(label);
   return label;
}

string FXAI_ExecutionQualityRegimeLabel(const FXAIDynamicEnsembleRuntimeState &dynamic_state,
                                        const FXAIAdaptiveRegimeState &adaptive_state)
{
   string label = "";
   if(dynamic_state.ready && StringLen(dynamic_state.top_regime) > 0)
      label = dynamic_state.top_regime;
   else if(adaptive_state.valid && StringLen(adaptive_state.top_label) > 0)
      label = adaptive_state.top_label;
   else
      label = FXAI_AdaptiveRouterRegimeLabel(g_ai_last_regime_id);
   StringToUpper(label);
   return label;
}

void FXAI_ExecutionQualityLoadConfig(FXAIExecutionQualityConfig &out)
{
   FXAI_ResetExecutionQualityConfig(out);
   int handle = FileOpen(FXAI_ExecutionQualityConfigFile(),
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
      if(n < 2)
         continue;
      string key = parts[0];
      string value = parts[1];
      if(key == "enabled") out.enabled = (StringToInteger(value) != 0);
      else if(key == "block_on_unknown") out.block_on_unknown = (StringToInteger(value) != 0);
      else if(key == "allow_block_state") out.allow_block_state = (StringToInteger(value) != 0);
      else if(key == "support_soft_floor") out.support_soft_floor = (int)StringToInteger(value);
      else if(key == "support_hard_floor") out.support_hard_floor = (int)StringToInteger(value);
      else if(key == "memory_stale_after_hours") out.memory_stale_after_hours = (int)StringToInteger(value);
      else if(key == "threshold_normal_min") out.threshold_normal_min = StringToDouble(value);
      else if(key == "threshold_caution_min") out.threshold_caution_min = StringToDouble(value);
      else if(key == "threshold_stressed_min") out.threshold_stressed_min = StringToDouble(value);
      else if(key == "lot_scale_normal") out.lot_scale_normal = StringToDouble(value);
      else if(key == "lot_scale_caution") out.lot_scale_caution = StringToDouble(value);
      else if(key == "lot_scale_stressed") out.lot_scale_stressed = StringToDouble(value);
      else if(key == "lot_scale_blocked") out.lot_scale_blocked = StringToDouble(value);
      else if(key == "enter_prob_buffer_normal") out.enter_prob_buffer_normal = StringToDouble(value);
      else if(key == "enter_prob_buffer_caution") out.enter_prob_buffer_caution = StringToDouble(value);
      else if(key == "enter_prob_buffer_stressed") out.enter_prob_buffer_stressed = StringToDouble(value);
      else if(key == "enter_prob_buffer_blocked") out.enter_prob_buffer_blocked = StringToDouble(value);
      else if(key == "cap_spread_expected_mult") out.cap_spread_expected_mult = StringToDouble(value);
      else if(key == "cap_expected_slippage_points") out.cap_expected_slippage_points = StringToDouble(value);
      else if(key == "cap_allowed_deviation_points_min") out.cap_allowed_deviation_points_min = StringToDouble(value);
      else if(key == "cap_allowed_deviation_points_max") out.cap_allowed_deviation_points_max = StringToDouble(value);
      else if(key == "weight_spread_zscore") out.weight_spread_zscore = StringToDouble(value);
      else if(key == "weight_news_risk") out.weight_news_risk = StringToDouble(value);
      else if(key == "weight_rates_risk") out.weight_rates_risk = StringToDouble(value);
      else if(key == "weight_micro_liquidity") out.weight_micro_liquidity = StringToDouble(value);
      else if(key == "weight_micro_hostile") out.weight_micro_hostile = StringToDouble(value);
      else if(key == "weight_volatility_burst") out.weight_volatility_burst = StringToDouble(value);
      else if(key == "weight_tick_rate_burst") out.weight_tick_rate_burst = StringToDouble(value);
      else if(key == "weight_session_thinness") out.weight_session_thinness = StringToDouble(value);
      else if(key == "weight_broker_reject") out.weight_broker_reject = StringToDouble(value);
      else if(key == "weight_broker_partial") out.weight_broker_partial = StringToDouble(value);
      else if(key == "weight_broker_latency") out.weight_broker_latency = StringToDouble(value);
      else if(key == "weight_broker_event_burst") out.weight_broker_event_burst = StringToDouble(value);
      else if(key == "weight_stale_context") out.weight_stale_context = StringToDouble(value);
      else if(key == "weight_support_shortfall") out.weight_support_shortfall = StringToDouble(value);
      else if(key == "bucket_count") out.bucket_count = (int)StringToInteger(value);
      else if(StringFind(key, "bucket_", 0) == 0)
      {
         int idx = (int)StringToInteger(StringSubstr(key, 7));
         if(idx >= 0 && idx < FXAI_EXEC_QUALITY_MAX_BUCKETS)
            out.bucket_hierarchy[idx] = value;
      }
   }
   FileClose(handle);
}

void FXAI_ExecutionQualityEnsureConfigLoaded(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(g_exec_quality_cfg_cache_loaded_at > 0 && now_time > 0 && (now_time - g_exec_quality_cfg_cache_loaded_at) < 60)
      return;
   FXAI_ExecutionQualityLoadConfig(g_exec_quality_cfg_cache);
   g_exec_quality_cfg_cache_loaded_at = now_time;
}

void FXAI_ExecutionQualityEnsureMemoryLoaded(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(g_exec_quality_memory_loaded_at > 0 && now_time > 0 && (now_time - g_exec_quality_memory_loaded_at) < 60)
      return;

   ArrayResize(g_exec_quality_tiers, 0);
   g_exec_quality_tier_count = 0;
   g_exec_quality_memory_generated_at = 0;
   g_exec_quality_memory_method = "SCORECARD_V1";

   int handle = FileOpen(FXAI_ExecutionQualityMemoryFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      g_exec_quality_memory_loaded_at = now_time;
      return;
   }

   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 3)
         continue;
      string kind = parts[0];
      if(kind == "meta")
      {
         if(parts[1] == "generated_at")
            g_exec_quality_memory_generated_at = FXAI_ExecutionQualityParseISO8601(parts[2]);
         else if(parts[1] == "default_method")
            g_exec_quality_memory_method = parts[2];
         continue;
      }
      if(kind != "tier" || n < 13 || g_exec_quality_tier_count >= FXAI_EXEC_QUALITY_MAX_TIERS)
         continue;

      int idx = g_exec_quality_tier_count;
      ArrayResize(g_exec_quality_tiers, idx + 1);
      FXAI_ResetExecutionQualityTier(g_exec_quality_tiers[idx]);
      g_exec_quality_tiers[idx].ready = true;
      g_exec_quality_tiers[idx].kind = parts[1];
      g_exec_quality_tiers[idx].symbol = parts[2];
      g_exec_quality_tiers[idx].session = parts[3];
      g_exec_quality_tiers[idx].regime = parts[4];
      g_exec_quality_tiers[idx].support = (int)StringToInteger(parts[5]);
      g_exec_quality_tiers[idx].quality = StringToDouble(parts[6]);
      g_exec_quality_tiers[idx].spread_mult = StringToDouble(parts[7]);
      g_exec_quality_tiers[idx].slippage_mult = StringToDouble(parts[8]);
      g_exec_quality_tiers[idx].fill_quality_bias = StringToDouble(parts[9]);
      g_exec_quality_tiers[idx].latency_mult = StringToDouble(parts[10]);
      g_exec_quality_tiers[idx].fragility_mult = StringToDouble(parts[11]);
      g_exec_quality_tiers[idx].deviation_mult = StringToDouble(parts[12]);
      StringToUpper(g_exec_quality_tiers[idx].kind);
      StringToUpper(g_exec_quality_tiers[idx].symbol);
      StringToUpper(g_exec_quality_tiers[idx].session);
      StringToUpper(g_exec_quality_tiers[idx].regime);
      g_exec_quality_tier_count++;
   }
   FileClose(handle);
   g_exec_quality_memory_loaded_at = now_time;
}

int FXAI_ExecutionQualityTierHierarchyIndex(const string kind)
{
   if(kind == "PAIR_SESSION_REGIME") return 0;
   if(kind == "PAIR_REGIME") return 1;
   if(kind == "SESSION_REGIME") return 2;
   if(kind == "REGIME") return 3;
   if(kind == "GLOBAL") return 4;
   return 99;
}

bool FXAI_ExecutionQualityTierMatches(const FXAIExecutionQualityTier &tier,
                                      const string kind,
                                      const string symbol,
                                      const string session,
                                      const string regime)
{
   if(tier.kind != kind)
      return false;
   if(kind == "PAIR_SESSION_REGIME")
      return tier.symbol == symbol && tier.session == session && tier.regime == regime;
   if(kind == "PAIR_REGIME")
      return tier.symbol == symbol && tier.regime == regime;
   if(kind == "SESSION_REGIME")
      return tier.session == session && tier.regime == regime;
   if(kind == "REGIME")
      return tier.regime == regime;
   if(kind == "GLOBAL")
      return true;
   return false;
}

bool FXAI_SelectExecutionQualityTier(const string symbol,
                                     const string session,
                                     const string regime,
                                     FXAIExecutionQualityTier &selected,
                                     bool &fallback_used,
                                     bool &support_usable)
{
   FXAI_ResetExecutionQualityTier(selected);
   fallback_used = false;
   support_usable = false;
   if(g_exec_quality_tier_count <= 0)
      return false;

   for(int order=0; order<g_exec_quality_cfg_cache.bucket_count; order++)
   {
      string kind = g_exec_quality_cfg_cache.bucket_hierarchy[order];
      if(StringLen(kind) <= 0)
         continue;
      FXAIExecutionQualityTier best_pref;
      FXAIExecutionQualityTier best_fallback;
      bool pref_found = false;
      bool fallback_found = false;
      int best_pref_support = -1;
      int best_fallback_support = -1;
      double best_pref_quality = -1.0;
      double best_fallback_quality = -1.0;

      for(int i=0; i<g_exec_quality_tier_count; i++)
      {
         if(!FXAI_ExecutionQualityTierMatches(g_exec_quality_tiers[i], kind, symbol, session, regime))
            continue;
         int support = g_exec_quality_tiers[i].support;
         double quality = g_exec_quality_tiers[i].quality;
         if(support >= g_exec_quality_cfg_cache.support_soft_floor)
         {
            if(!pref_found || support > best_pref_support || (support == best_pref_support && quality > best_pref_quality))
            {
               best_pref = g_exec_quality_tiers[i];
               best_pref_support = support;
               best_pref_quality = quality;
               pref_found = true;
            }
         }
         else if(support >= g_exec_quality_cfg_cache.support_hard_floor)
         {
            if(!fallback_found || support > best_fallback_support || (support == best_fallback_support && quality > best_fallback_quality))
            {
               best_fallback = g_exec_quality_tiers[i];
               best_fallback_support = support;
               best_fallback_quality = quality;
               fallback_found = true;
            }
         }
      }

      if(pref_found)
      {
         selected = best_pref;
         fallback_used = false;
         support_usable = true;
         return true;
      }
      if(fallback_found)
      {
         selected = best_fallback;
         fallback_used = true;
         support_usable = true;
         return true;
      }
   }
   return false;
}

void FXAI_ExecutionQualityBuildFallbackTier(FXAIExecutionQualityTier &tier)
{
   FXAI_ResetExecutionQualityTier(tier);
   tier.ready = true;
}

double FXAI_ExecutionQualityCurrentSpreadPoints(const string symbol,
                                                const double fallback_spread_points)
{
   MqlTick tick;
   if(FXAI_MarketDataGetLatestTick(symbol, tick) && tick.ask > 0.0 && tick.bid > 0.0)
   {
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(point <= 0.0)
         point = (_Point > 0.0 ? _Point : 1.0);
      if(point > 0.0)
         return MathMax((tick.ask - tick.bid) / point, 0.0);
   }
   return MathMax(fallback_spread_points, 0.0);
}

double FXAI_ExecutionQualitySessionThinness(const string session_label,
                                            const bool handoff_flag)
{
   string session = session_label;
   StringToUpper(session);
   double thinness = 0.18;
   if(StringFind(session, "ASIA", 0) >= 0)
      thinness = 0.42;
   if(StringFind(session, "OVERLAP", 0) >= 0)
      thinness = 0.22;
   if(StringFind(session, "ROLLOVER", 0) >= 0 || StringFind(session, "OFF", 0) >= 0)
      thinness = 0.60;
   if(handoff_flag)
      thinness = MathMax(thinness, 0.55);
   return FXAI_Clamp(thinness, 0.0, 1.0);
}

void FXAI_ExecutionQualityApply(const string symbol,
                                const FXAIExecutionProfile &exec_profile,
                                const FXAINewsPulsePairState &news_state,
                                const FXAIRatesEnginePairState &rates_state,
                                const FXAICrossAssetPairState &cross_state,
                                const FXAIMicrostructurePairState &micro_state,
                                const FXAIAdaptiveRegimeState &adaptive_state,
                                const FXAIDynamicEnsembleRuntimeState &dynamic_state,
                                const double spread_pred_points,
                                const int horizon_minutes,
                                const int upstream_decision,
                                FXAIExecutionQualityRuntimeState &state)
{
   FXAI_ResetExecutionQualityRuntimeState(state);
   FXAI_ExecutionQualityEnsureConfigLoaded();
   FXAI_ExecutionQualityEnsureMemoryLoaded();
   state.method = g_exec_quality_memory_method;
   state.symbol = symbol;
   state.generated_at = TimeCurrent();
   if(state.generated_at <= 0)
      state.generated_at = TimeTradeServer();
   if(state.generated_at <= 0)
      state.generated_at = TimeLocal();
   state.session_label = FXAI_ExecutionQualitySessionLabel(news_state, micro_state);
   state.regime_label = FXAI_ExecutionQualityRegimeLabel(dynamic_state, adaptive_state);
   state.news_window_active = (news_state.ready && news_state.available &&
                               !news_state.stale &&
                               (news_state.trade_gate == "BLOCK" || news_state.trade_gate == "CAUTION" ||
                                (news_state.event_eta_min >= 0 && news_state.event_eta_min <= 30) ||
                                news_state.news_risk_score >= 0.64));
   state.rates_repricing_active = (rates_state.ready && rates_state.available &&
                                   !rates_state.stale &&
                                   (rates_state.meeting_path_reprice_now ||
                                    rates_state.trade_gate == "BLOCK" ||
                                    rates_state.rates_risk_score >= 0.64));

   if(!g_exec_quality_cfg_cache.enabled)
      return;

   string tier_symbol = symbol;
   StringToUpper(tier_symbol);
   string tier_session = state.session_label;
   string tier_regime = state.regime_label;
   FXAIExecutionQualityTier tier;
   bool tier_fallback = false;
   bool tier_support = false;
   bool tier_found = FXAI_SelectExecutionQualityTier(tier_symbol,
                                                     tier_session,
                                                     tier_regime,
                                                     tier,
                                                     tier_fallback,
                                                     tier_support);
   if(!tier_found)
   {
      FXAI_ExecutionQualityBuildFallbackTier(tier);
      tier_fallback = true;
      tier_support = false;
   }
   state.selected_tier_kind = tier.kind;
   state.selected_tier_key = FXAI_ExecutionQualityTierKey(tier);
   state.selected_support = tier.support;
   state.selected_quality = FXAI_Clamp(tier.quality, 0.0, 1.0);
   state.fallback_used = tier_fallback;

   int stale_context_count = 0;
   bool news_stale = (news_state.ready && news_state.available && news_state.stale);
   bool rates_stale = (rates_state.ready && rates_state.available && rates_state.stale);
   bool cross_stale = (cross_state.ready && cross_state.available && cross_state.stale);
   bool micro_stale = (micro_state.ready && micro_state.available && micro_state.stale);
   if(news_stale) stale_context_count++;
   if(rates_stale) stale_context_count++;
   if(cross_stale) stale_context_count++;
   if(micro_stale) stale_context_count++;

   state.memory_stale = (g_exec_quality_memory_generated_at <= 0 ||
                         (g_exec_quality_cfg_cache.memory_stale_after_hours > 0 &&
                          (state.generated_at - g_exec_quality_memory_generated_at) >
                          g_exec_quality_cfg_cache.memory_stale_after_hours * 3600));

   state.spread_now_points = FXAI_ExecutionQualityCurrentSpreadPoints(symbol, spread_pred_points);

   int order_side = 0;
   if(upstream_decision == 1) order_side = 1;
   else if(upstream_decision == 0) order_side = -1;
   FXAIBrokerExecutionStats broker_stats;
   FXAI_GetBrokerExecutionStressEx(state.generated_at,
                                   symbol,
                                   horizon_minutes,
                                   order_side,
                                   1,
                                   broker_stats);
   state.broker_coverage = FXAI_Clamp(broker_stats.coverage, 0.0, 1.0);
   state.broker_reject_prob = FXAI_Clamp(broker_stats.reject_prob, 0.0, 1.0);
   state.broker_partial_fill_prob = FXAI_Clamp(MathMax(broker_stats.partial_fill_prob,
                                                       1.0 - FXAI_Clamp(broker_stats.fill_ratio_mean, 0.0, 1.0)),
                                               0.0,
                                               1.0);
   state.support_usable = (tier_support && state.broker_coverage >= 0.05);
   state.data_stale = (state.memory_stale ||
                       state.spread_now_points <= 0.0 ||
                       (g_exec_quality_cfg_cache.block_on_unknown && stale_context_count > 0));

   double news_risk = (news_state.ready && news_state.available
                       ? FXAI_Clamp(news_state.news_risk_score, 0.0, 1.0)
                       : (news_stale ? 0.45 : 0.12));
   double rates_risk = (rates_state.ready && rates_state.available
                        ? FXAI_Clamp(rates_state.rates_risk_score, 0.0, 1.0)
                        : (rates_stale ? 0.32 : 0.10));
   double cross_risk = (cross_state.ready && cross_state.available
                        ? FXAI_Clamp(MathMax(cross_state.pair_cross_asset_risk_score,
                                             MathMax(cross_state.usd_liquidity_stress_score,
                                                     cross_state.cross_asset_dislocation_score)),
                                     0.0,
                                     1.0)
                        : (cross_stale ? 0.34 : 0.12));
   double micro_hostile = (micro_state.ready && micro_state.available
                           ? FXAI_Clamp(micro_state.hostile_execution_score, 0.0, 1.0)
                           : (micro_stale ? 0.42 : 0.10));
   double micro_liquidity = (micro_state.ready && micro_state.available
                             ? FXAI_Clamp(micro_state.liquidity_stress_score, 0.0, 1.0)
                             : (micro_stale ? 0.44 : 0.12));
   double spread_z_norm = (micro_state.ready && micro_state.available
                           ? FXAI_Clamp(micro_state.spread_zscore_60s / 4.0, 0.0, 1.0)
                           : 0.0);
   double vol_burst_norm = (micro_state.ready && micro_state.available
                            ? FXAI_Clamp(micro_state.vol_burst_score_5m / 3.0, 0.0, 1.0)
                            : 0.0);
   double tick_rate_norm = (micro_state.ready && micro_state.available
                            ? FXAI_Clamp(micro_state.tick_rate_zscore_60s / 3.0, 0.0, 1.0)
                            : 0.0);
   double tick_imbalance_norm = (micro_state.ready && micro_state.available
                                 ? FXAI_Clamp(MathAbs(micro_state.tick_imbalance_30s), 0.0, 1.0)
                                 : 0.0);
   double session_thinness = FXAI_ExecutionQualitySessionThinness(state.session_label,
                                                                  (micro_state.ready && micro_state.available && micro_state.handoff_flag));
   double stale_norm = FXAI_Clamp((double)stale_context_count / 3.0, 0.0, 1.0);
   double support_shortfall = FXAI_Clamp((double)(g_exec_quality_cfg_cache.support_soft_floor - tier.support) /
                                         (double)MathMax(g_exec_quality_cfg_cache.support_soft_floor, 1),
                                         0.0,
                                         1.0);
   state.spread_widening_risk = FXAI_Clamp(0.10 +
                                           g_exec_quality_cfg_cache.weight_spread_zscore * spread_z_norm +
                                           g_exec_quality_cfg_cache.weight_news_risk * news_risk +
                                           g_exec_quality_cfg_cache.weight_rates_risk * rates_risk +
                                           0.12 * cross_risk +
                                           g_exec_quality_cfg_cache.weight_micro_liquidity * micro_liquidity +
                                           g_exec_quality_cfg_cache.weight_volatility_burst * vol_burst_norm +
                                           g_exec_quality_cfg_cache.weight_session_thinness * session_thinness +
                                           g_exec_quality_cfg_cache.weight_broker_reject * state.broker_reject_prob * 0.45 +
                                           g_exec_quality_cfg_cache.weight_broker_partial * state.broker_partial_fill_prob * 0.40 +
                                           g_exec_quality_cfg_cache.weight_broker_event_burst * FXAI_Clamp(broker_stats.event_burst_penalty, 0.0, 1.0) +
                                           g_exec_quality_cfg_cache.weight_stale_context * stale_norm +
                                           (state.news_window_active ? 0.08 : 0.0) +
                                           (state.rates_repricing_active ? 0.05 : 0.0) -
                                           0.10 * state.selected_quality,
                                           0.0,
                                           1.0);

   double spread_expected_mult = FXAI_Clamp(0.96 +
                                            0.38 * tier.spread_mult +
                                            0.64 * state.spread_widening_risk +
                                            0.14 * spread_z_norm +
                                            0.06 * session_thinness,
                                            1.0,
                                            g_exec_quality_cfg_cache.cap_spread_expected_mult);
   state.spread_expected_points = MathMax(state.spread_now_points,
                                          state.spread_now_points * spread_expected_mult +
                                          0.12 * MathMax(broker_stats.slippage_points, 0.0));

   state.expected_slippage_points = FXAI_Clamp(MathMax(broker_stats.slippage_points, 0.0) * tier.slippage_mult +
                                               0.16 * state.spread_expected_points +
                                               0.55 * micro_hostile +
                                               0.38 * vol_burst_norm +
                                               0.26 * session_thinness +
                                               0.24 * news_risk +
                                               0.18 * rates_risk +
                                               0.22 * cross_risk +
                                               0.28 * FXAI_Clamp(broker_stats.event_burst_penalty, 0.0, 1.0) +
                                               0.32 * state.broker_reject_prob +
                                               0.30 * state.broker_partial_fill_prob +
                                               0.18 * FXAI_Clamp(broker_stats.latency_points / 5.0, 0.0, 1.0) * tier.latency_mult,
                                               0.0,
                                               g_exec_quality_cfg_cache.cap_expected_slippage_points);

   state.slippage_risk = FXAI_Clamp(0.12 +
                                    0.24 * FXAI_Clamp(state.expected_slippage_points / MathMax(state.spread_expected_points + 0.5, 1.0), 0.0, 3.0) / 3.0 +
                                    0.18 * micro_hostile +
                                    0.12 * vol_burst_norm +
                                    0.12 * news_risk +
                                    0.08 * rates_risk +
                                    0.10 * FXAI_Clamp(broker_stats.event_burst_penalty, 0.0, 1.0) +
                                    0.12 * state.broker_reject_prob +
                                    0.10 * state.broker_partial_fill_prob,
                                    0.0,
                                    1.0);

   state.latency_sensitivity_score = FXAI_Clamp(0.14 +
                                                0.22 * tick_rate_norm +
                                                0.18 * vol_burst_norm +
                                                0.16 * news_risk +
                                                0.10 * rates_risk +
                                                0.08 * cross_risk +
                                                0.12 * FXAI_Clamp(broker_stats.latency_points / 5.0, 0.0, 1.0) * tier.latency_mult +
                                                0.08 * micro_hostile +
                                                0.08 * session_thinness +
                                                0.06 * tick_imbalance_norm,
                                                0.0,
                                                1.0);

   state.liquidity_fragility_score = FXAI_Clamp(0.10 +
                                                0.26 * micro_liquidity * tier.fragility_mult +
                                                0.16 * micro_hostile +
                                                0.12 * spread_z_norm +
                                                0.08 * news_risk +
                                                0.08 * rates_risk +
                                                0.12 * cross_risk +
                                                0.12 * state.broker_partial_fill_prob +
                                                0.10 * state.broker_reject_prob +
                                                0.10 * session_thinness +
                                                0.06 * FXAI_Clamp(broker_stats.event_burst_penalty, 0.0, 1.0) -
                                                0.08 * state.selected_quality,
                                                0.0,
                                                1.0);

   state.fill_quality_score = FXAI_Clamp(0.86 +
                                         tier.fill_quality_bias -
                                         0.28 * state.slippage_risk -
                                         0.24 * state.latency_sensitivity_score -
                                         0.22 * state.liquidity_fragility_score -
                                         0.14 * state.broker_reject_prob -
                                         0.12 * state.broker_partial_fill_prob -
                                         0.08 * session_thinness,
                                         0.0,
                                         1.0);

   state.execution_quality_score = FXAI_Clamp(0.40 * state.fill_quality_score +
                                              0.18 * (1.0 - state.spread_widening_risk) +
                                              0.18 * (1.0 - state.slippage_risk) +
                                              0.12 * (1.0 - state.latency_sensitivity_score) +
                                              0.12 * (1.0 - state.liquidity_fragility_score) -
                                              g_exec_quality_cfg_cache.weight_stale_context * 0.80 * stale_norm -
                                              g_exec_quality_cfg_cache.weight_support_shortfall * support_shortfall,
                                              0.0,
                                              1.0);

   double block_threshold = g_exec_quality_cfg_cache.threshold_stressed_min * 0.72;
   if(state.data_stale && g_exec_quality_cfg_cache.block_on_unknown)
      state.execution_state = "BLOCKED";
   else if(g_exec_quality_cfg_cache.allow_block_state &&
           (state.execution_quality_score < block_threshold ||
            state.spread_widening_risk >= 0.90 ||
            state.slippage_risk >= 0.90 ||
            state.fill_quality_score <= 0.20))
      state.execution_state = "BLOCKED";
   else if(state.execution_quality_score < g_exec_quality_cfg_cache.threshold_stressed_min)
      state.execution_state = "STRESSED";
   else if(state.execution_quality_score < g_exec_quality_cfg_cache.threshold_caution_min)
      state.execution_state = "CAUTION";
   else
      state.execution_state = "NORMAL";

   double base_deviation = FXAI_ExecutionAllowedDeviationPoints(exec_profile,
                                                                g_ai_last_path_risk,
                                                                g_ai_last_fill_risk);
   state.allowed_deviation_points = FXAI_Clamp(base_deviation *
                                               tier.deviation_mult *
                                               (1.0 + 0.14 * state.spread_widening_risk +
                                                0.18 * state.slippage_risk +
                                                0.10 * state.latency_sensitivity_score),
                                               g_exec_quality_cfg_cache.cap_allowed_deviation_points_min,
                                               g_exec_quality_cfg_cache.cap_allowed_deviation_points_max);

   if(state.execution_state == "BLOCKED")
   {
      state.caution_lot_scale = g_exec_quality_cfg_cache.lot_scale_blocked;
      state.caution_enter_prob_buffer = g_exec_quality_cfg_cache.enter_prob_buffer_blocked;
   }
   else if(state.execution_state == "STRESSED")
   {
      state.caution_lot_scale = g_exec_quality_cfg_cache.lot_scale_stressed;
      state.caution_enter_prob_buffer = g_exec_quality_cfg_cache.enter_prob_buffer_stressed;
   }
   else if(state.execution_state == "CAUTION")
   {
      state.caution_lot_scale = g_exec_quality_cfg_cache.lot_scale_caution;
      state.caution_enter_prob_buffer = g_exec_quality_cfg_cache.enter_prob_buffer_caution;
   }
   else
   {
      state.caution_lot_scale = g_exec_quality_cfg_cache.lot_scale_normal;
      state.caution_enter_prob_buffer = g_exec_quality_cfg_cache.enter_prob_buffer_normal;
   }

   if(state.data_stale)
      FXAI_ExecutionQualityAppendReason(state, "DATA_STALE");
   if(state.memory_stale)
      FXAI_ExecutionQualityAppendReason(state, "MEMORY_STALE");
   if(!state.support_usable)
      FXAI_ExecutionQualityAppendReason(state, "SUPPORT_TOO_LOW");
   if(state.news_window_active || news_risk >= 0.68)
      FXAI_ExecutionQualityAppendReason(state, "NEWS_WINDOW_ACTIVE");
   if(state.rates_repricing_active || rates_risk >= 0.68)
      FXAI_ExecutionQualityAppendReason(state, "RATES_REPRICING_ACTIVE");
   if(cross_risk >= 0.58)
      FXAI_ExecutionQualityAppendReason(state, "CROSS_ASSET_STRESS_ELEVATED");
   if(spread_z_norm >= 0.55)
      FXAI_ExecutionQualityAppendReason(state, "SPREAD_ALREADY_ELEVATED");
   if(micro_hostile >= 0.62)
      FXAI_ExecutionQualityAppendReason(state, "MICROSTRUCTURE_HOSTILE");
   if(micro_liquidity >= 0.62)
      FXAI_ExecutionQualityAppendReason(state, "LIQUIDITY_STRESS_ELEVATED");
   if(vol_burst_norm >= 0.58)
      FXAI_ExecutionQualityAppendReason(state, "VOLATILITY_BURST");
   if(session_thinness >= 0.52)
      FXAI_ExecutionQualityAppendReason(state, "LOW_LIQUIDITY_SESSION");
   if(state.slippage_risk >= 0.66)
      FXAI_ExecutionQualityAppendReason(state, "SLIPPAGE_RISK_ELEVATED");
   if(state.latency_sensitivity_score >= 0.66)
      FXAI_ExecutionQualityAppendReason(state, "LATENCY_SENSITIVITY_HIGH");
   if(state.broker_reject_prob >= 0.40)
      FXAI_ExecutionQualityAppendReason(state, "BROKER_REJECT_RISK_ELEVATED");
   if(state.broker_partial_fill_prob >= 0.42)
      FXAI_ExecutionQualityAppendReason(state, "BROKER_PARTIAL_FILL_RISK_ELEVATED");
   if(state.execution_state == "BLOCKED")
      FXAI_ExecutionQualityAppendReason(state, "EXECUTION_STATE_BLOCKED");
   else if(state.execution_state == "STRESSED")
      FXAI_ExecutionQualityAppendReason(state, "EXECUTION_STATE_STRESSED");
   else if(state.execution_state == "CAUTION")
      FXAI_ExecutionQualityAppendReason(state, "EXECUTION_STATE_CAUTION");

   state.ready = true;
   g_execution_quality_last_ready = state.ready;
   g_execution_quality_last_fallback_used = state.fallback_used;
   g_execution_quality_last_memory_stale = state.memory_stale;
   g_execution_quality_last_data_stale = state.data_stale;
   g_execution_quality_last_support_usable = state.support_usable;
   g_execution_quality_last_generated_at = state.generated_at;
   g_execution_quality_last_method = state.method;
   g_execution_quality_last_tier_kind = state.selected_tier_kind;
   g_execution_quality_last_tier_key = state.selected_tier_key;
   g_execution_quality_last_support = state.selected_support;
   g_execution_quality_last_quality = state.selected_quality;
   g_execution_quality_last_spread_now = state.spread_now_points;
   g_execution_quality_last_spread_expected = state.spread_expected_points;
   g_execution_quality_last_spread_widening_risk = state.spread_widening_risk;
   g_execution_quality_last_expected_slippage = state.expected_slippage_points;
   g_execution_quality_last_slippage_risk = state.slippage_risk;
   g_execution_quality_last_fill_quality = state.fill_quality_score;
   g_execution_quality_last_latency_sensitivity = state.latency_sensitivity_score;
   g_execution_quality_last_liquidity_fragility = state.liquidity_fragility_score;
   g_execution_quality_last_quality_score = state.execution_quality_score;
   g_execution_quality_last_allowed_deviation = state.allowed_deviation_points;
   g_execution_quality_last_caution_lot_scale = state.caution_lot_scale;
   g_execution_quality_last_caution_enter_prob_buffer = state.caution_enter_prob_buffer;
   g_execution_quality_last_state = state.execution_state;
   g_execution_quality_last_reasons_csv = FXAI_ExecutionQualityReasonsCSV(state);
}

void FXAI_ExecutionQualityWriteRuntimeArtifacts(const string symbol,
                                                const FXAIExecutionQualityRuntimeState &state)
{
   if(StringLen(symbol) <= 0)
      return;
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate("FXAI\\Runtime", FILE_COMMON);

   if(state.ready)
   {
      int handle = FileOpen(FXAI_ExecutionQualityRuntimeStateFile(symbol),
                            FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(handle != INVALID_HANDLE)
      {
         FileWriteString(handle, "symbol\t" + symbol + "\r\n");
         FileWriteString(handle, "generated_at\t" + IntegerToString((int)state.generated_at) + "\r\n");
         FileWriteString(handle, "method\t" + state.method + "\r\n");
         FileWriteString(handle, "session_label\t" + state.session_label + "\r\n");
         FileWriteString(handle, "regime_label\t" + state.regime_label + "\r\n");
         FileWriteString(handle, "selected_tier_kind\t" + state.selected_tier_kind + "\r\n");
         FileWriteString(handle, "selected_tier_key\t" + state.selected_tier_key + "\r\n");
         FileWriteString(handle, "selected_support\t" + IntegerToString(state.selected_support) + "\r\n");
         FileWriteString(handle, "selected_quality\t" + DoubleToString(state.selected_quality, 6) + "\r\n");
         FileWriteString(handle, "fallback_used\t" + IntegerToString(state.fallback_used ? 1 : 0) + "\r\n");
         FileWriteString(handle, "memory_stale\t" + IntegerToString(state.memory_stale ? 1 : 0) + "\r\n");
         FileWriteString(handle, "data_stale\t" + IntegerToString(state.data_stale ? 1 : 0) + "\r\n");
         FileWriteString(handle, "support_usable\t" + IntegerToString(state.support_usable ? 1 : 0) + "\r\n");
         FileWriteString(handle, "news_window_active\t" + IntegerToString(state.news_window_active ? 1 : 0) + "\r\n");
         FileWriteString(handle, "rates_repricing_active\t" + IntegerToString(state.rates_repricing_active ? 1 : 0) + "\r\n");
         FileWriteString(handle, "broker_coverage\t" + DoubleToString(state.broker_coverage, 6) + "\r\n");
         FileWriteString(handle, "broker_reject_prob\t" + DoubleToString(state.broker_reject_prob, 6) + "\r\n");
         FileWriteString(handle, "broker_partial_fill_prob\t" + DoubleToString(state.broker_partial_fill_prob, 6) + "\r\n");
         FileWriteString(handle, "spread_now_points\t" + DoubleToString(state.spread_now_points, 6) + "\r\n");
         FileWriteString(handle, "spread_expected_points\t" + DoubleToString(state.spread_expected_points, 6) + "\r\n");
         FileWriteString(handle, "spread_widening_risk\t" + DoubleToString(state.spread_widening_risk, 6) + "\r\n");
         FileWriteString(handle, "expected_slippage_points\t" + DoubleToString(state.expected_slippage_points, 6) + "\r\n");
         FileWriteString(handle, "slippage_risk\t" + DoubleToString(state.slippage_risk, 6) + "\r\n");
         FileWriteString(handle, "fill_quality_score\t" + DoubleToString(state.fill_quality_score, 6) + "\r\n");
         FileWriteString(handle, "latency_sensitivity_score\t" + DoubleToString(state.latency_sensitivity_score, 6) + "\r\n");
         FileWriteString(handle, "liquidity_fragility_score\t" + DoubleToString(state.liquidity_fragility_score, 6) + "\r\n");
         FileWriteString(handle, "execution_quality_score\t" + DoubleToString(state.execution_quality_score, 6) + "\r\n");
         FileWriteString(handle, "allowed_deviation_points\t" + DoubleToString(state.allowed_deviation_points, 6) + "\r\n");
         FileWriteString(handle, "caution_lot_scale\t" + DoubleToString(state.caution_lot_scale, 6) + "\r\n");
         FileWriteString(handle, "caution_enter_prob_buffer\t" + DoubleToString(state.caution_enter_prob_buffer, 6) + "\r\n");
         FileWriteString(handle, "execution_state\t" + state.execution_state + "\r\n");
         FileWriteString(handle, "reasons_csv\t" + FXAI_ExecutionQualityReasonsCSV(state) + "\r\n");
         FileClose(handle);
      }
   }

   if(state.ready)
   {
      int hist = FileOpen(FXAI_ExecutionQualityRuntimeHistoryFile(symbol),
                          FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                          FILE_READ | FILE_SHARE_READ | FILE_SHARE_WRITE);
      if(hist == INVALID_HANDLE)
         hist = FileOpen(FXAI_ExecutionQualityRuntimeHistoryFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
      if(hist != INVALID_HANDLE)
      {
         FileSeek(hist, 0, SEEK_END);
         string json = "{";
         json += "\"generated_at\":\"" + FXAI_ExecutionQualityISO8601(state.generated_at) + "\",";
         json += "\"symbol\":\"" + FXAI_ExecutionQualityJSONEscape(symbol) + "\",";
         json += "\"state\":{";
         json += "\"symbol\":\"" + FXAI_ExecutionQualityJSONEscape(symbol) + "\",";
         json += "\"method\":\"" + FXAI_ExecutionQualityJSONEscape(state.method) + "\",";
         json += "\"session_label\":\"" + FXAI_ExecutionQualityJSONEscape(state.session_label) + "\",";
         json += "\"regime_label\":\"" + FXAI_ExecutionQualityJSONEscape(state.regime_label) + "\",";
         json += "\"selected_tier_kind\":\"" + FXAI_ExecutionQualityJSONEscape(state.selected_tier_kind) + "\",";
         json += "\"selected_tier_key\":\"" + FXAI_ExecutionQualityJSONEscape(state.selected_tier_key) + "\",";
         json += "\"selected_support\":" + IntegerToString(state.selected_support) + ",";
         json += "\"selected_quality\":" + DoubleToString(state.selected_quality, 6) + ",";
         json += "\"fallback_used\":" + IntegerToString(state.fallback_used ? 1 : 0) + ",";
         json += "\"memory_stale\":" + IntegerToString(state.memory_stale ? 1 : 0) + ",";
         json += "\"data_stale\":" + IntegerToString(state.data_stale ? 1 : 0) + ",";
         json += "\"support_usable\":" + IntegerToString(state.support_usable ? 1 : 0) + ",";
         json += "\"news_window_active\":" + IntegerToString(state.news_window_active ? 1 : 0) + ",";
         json += "\"rates_repricing_active\":" + IntegerToString(state.rates_repricing_active ? 1 : 0) + ",";
         json += "\"broker_coverage\":" + DoubleToString(state.broker_coverage, 6) + ",";
         json += "\"broker_reject_prob\":" + DoubleToString(state.broker_reject_prob, 6) + ",";
         json += "\"broker_partial_fill_prob\":" + DoubleToString(state.broker_partial_fill_prob, 6) + ",";
         json += "\"spread_now_points\":" + DoubleToString(state.spread_now_points, 6) + ",";
         json += "\"spread_expected_points\":" + DoubleToString(state.spread_expected_points, 6) + ",";
         json += "\"spread_widening_risk\":" + DoubleToString(state.spread_widening_risk, 6) + ",";
         json += "\"expected_slippage_points\":" + DoubleToString(state.expected_slippage_points, 6) + ",";
         json += "\"slippage_risk\":" + DoubleToString(state.slippage_risk, 6) + ",";
         json += "\"fill_quality_score\":" + DoubleToString(state.fill_quality_score, 6) + ",";
         json += "\"latency_sensitivity_score\":" + DoubleToString(state.latency_sensitivity_score, 6) + ",";
         json += "\"liquidity_fragility_score\":" + DoubleToString(state.liquidity_fragility_score, 6) + ",";
         json += "\"execution_quality_score\":" + DoubleToString(state.execution_quality_score, 6) + ",";
         json += "\"allowed_deviation_points\":" + DoubleToString(state.allowed_deviation_points, 6) + ",";
         json += "\"caution_lot_scale\":" + DoubleToString(state.caution_lot_scale, 6) + ",";
         json += "\"caution_enter_prob_buffer\":" + DoubleToString(state.caution_enter_prob_buffer, 6) + ",";
         json += "\"execution_state\":\"" + FXAI_ExecutionQualityJSONEscape(state.execution_state) + "\",";
         json += "\"reason_codes\":[";
         for(int i=0; i<state.reason_count; i++)
         {
            if(i > 0) json += ",";
            json += "\"" + FXAI_ExecutionQualityJSONEscape(state.reason_codes[i]) + "\"";
         }
         json += "]";
         json += "}}";
         FileWriteString(hist, json + "\r\n");
         FileClose(hist);
      }
   }
}

#endif // __FXAI_RUNTIME_EXECUTION_QUALITY_STAGE_MQH__
