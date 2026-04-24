#ifndef __FXAI_RUNTIME_CONTROL_PLANE_PROFILES_MQH__
#define __FXAI_RUNTIME_CONTROL_PLANE_PROFILES_MQH__
string FXAI_ControlPlaneSafeToken(const string raw)
{
   string clean = raw;
   if(StringLen(clean) <= 0)
      clean = "default";
   StringReplace(clean, "\\", "_");
   StringReplace(clean, "/", "_");
   StringReplace(clean, ":", "_");
   StringReplace(clean, "*", "_");
   StringReplace(clean, "?", "_");
   StringReplace(clean, "\"", "_");
   StringReplace(clean, "<", "_");
   StringReplace(clean, ">", "_");
   StringReplace(clean, "|", "_");
   StringReplace(clean, " ", "_");
   return clean;
}

string FXAI_FamilyName(const int family_id)
{
   switch(family_id)
   {
      case FXAI_FAMILY_LINEAR: return "linear";
      case FXAI_FAMILY_TREE: return "tree";
      case FXAI_FAMILY_RECURRENT: return "recurrent";
      case FXAI_FAMILY_CONVOLUTIONAL: return "convolutional";
      case FXAI_FAMILY_TRANSFORMER: return "transformer";
      case FXAI_FAMILY_STATE_SPACE: return "state_space";
      case FXAI_FAMILY_DISTRIBUTIONAL: return "distribution";
      case FXAI_FAMILY_MIXTURE: return "mixture";
      case FXAI_FAMILY_RETRIEVAL: return "memory";
      case FXAI_FAMILY_WORLD_MODEL: return "world";
      case FXAI_FAMILY_RULE_BASED: return "rule";
      default: return "other";
   }
}

string FXAI_SupervisorServiceSymbolFile(const string symbol)
{
   return "FXAI\\Offline\\Promotions\\fxai_supervisor_service_" +
          FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_SupervisorCommandSymbolFile(const string symbol)
{
   return "FXAI\\Offline\\Promotions\\fxai_supervisor_command_" +
          FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_StudentRouterProfileFile(const string symbol)
{
   return "FXAI\\Offline\\Promotions\\fxai_student_router_" +
          FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_AdaptiveRouterProfileFile(const string symbol)
{
   return "FXAI\\Offline\\Promotions\\fxai_adaptive_router_" +
          FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_ControlPlaneSnapshotFile(const string symbol)
{
   long login = (long)AccountInfoInteger(ACCOUNT_LOGIN);
   ulong magic = TradeMagic;
   long chart_id = (long)ChartID();
   return FXAI_CONTROL_PLANE_DIR + "\\cp_" +
          IntegerToString((int)login) + "_" +
          IntegerToString((int)magic) + "_" +
          FXAI_ControlPlaneSafeToken(symbol) + "_" +
          IntegerToString((int)(chart_id % 2147483647)) + ".tsv";
}

string FXAI_LiveDeploymentProfileFile(const string symbol)
{
   return "FXAI\\Offline\\Promotions\\fxai_live_deploy_" +
          FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

double FXAI_ControlPlaneCapitalRiskPct(const string symbol,
                                       const int direction,
                                       const double additional_lot = 0.0)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity <= 0.0)
      return 0.0;
   double money_per_point = FXAI_MoneyPerPointPerLot(symbol);
   if(money_per_point <= 0.0)
      return 0.0;
   double risk_points = FXAI_EstimatedRiskPointsForDecision();
   double exposure_lots = FXAI_ManagedDirectionalClusterLots(symbol, direction);
   if(additional_lot > 0.0)
      exposure_lots += additional_lot;
   if(exposure_lots <= 0.0)
      return 0.0;
   double risk_money = exposure_lots * risk_points * money_per_point;
   return 100.0 * risk_money / equity;
}

bool FXAI_ControlPlaneReadKV(const string file_name,
                             const string key,
                             string &value_out)
{
   int handle = FileOpen(file_name,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;

   bool found = false;
   while(!FileIsEnding(handle))
   {
      string line = FileReadString(handle);
      if(StringLen(line) <= 0)
         continue;
      string parts[];
      int n = StringSplit(line, '\t', parts);
      if(n < 2)
         continue;
      if(parts[0] == key)
      {
         value_out = parts[1];
         found = true;
         break;
      }
   }
   FileClose(handle);
   return found;
}

bool FXAI_CSVContainsToken(const string csv,
                           const string token)
{
   string clean_token = token;
   StringTrimLeft(clean_token);
   StringTrimRight(clean_token);
   if(StringLen(clean_token) <= 0)
      return false;
   string work = csv;
   string parts[];
   int n = StringSplit(work, ',', parts);
   for(int i=0; i<n; i++)
   {
      string item = parts[i];
      StringTrimLeft(item);
      StringTrimRight(item);
      if(item == clean_token)
         return true;
   }
   return false;
}

double FXAI_CSVMapWeight(const string csv,
                         const string key,
                         const double default_value = 1.0)
{
   string clean_key = key;
   StringTrimLeft(clean_key);
   StringTrimRight(clean_key);
   if(StringLen(clean_key) <= 0)
      return default_value;
   string parts[];
   int n = StringSplit(csv, ',', parts);
   for(int i=0; i<n; i++)
   {
      string item = parts[i];
      StringTrimLeft(item);
      StringTrimRight(item);
      if(StringLen(item) <= 0)
         continue;
      string kv[];
      int kv_n = StringSplit(item, '=', kv);
      if(kv_n < 2)
         continue;
      string item_key = kv[0];
      StringTrimLeft(item_key);
      StringTrimRight(item_key);
      if(item_key != clean_key)
         continue;
      return StringToDouble(kv[1]);
   }
   return default_value;
}

bool FXAI_ControlPlaneArtifactFresh(const datetime generated_at,
                                    const datetime expires_at,
                                    const int fallback_ttl_sec)
{
   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;
   if(now <= 0)
      return true;
   if(expires_at > 0 && now > expires_at)
      return false;
   if(generated_at > 0 && fallback_ttl_sec > 0 && now > generated_at &&
      (now - generated_at) > fallback_ttl_sec)
      return false;
   return true;
}

bool FXAI_LoadLiveDeploymentProfile(const string symbol,
                                    FXAILiveDeploymentProfile &out,
                                    const bool force_reload = false)
{
   static FXAILiveDeploymentProfile cache;
   static string cache_symbol = "";
   static datetime cache_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache.ready &&
      cache_symbol == symbol &&
      cache_time > 0 &&
      now > 0 &&
      (now - cache_time) < 300)
   {
      out = cache;
      return true;
   }

   FXAI_ResetLiveDeploymentProfile(out);
   out.symbol = symbol;
   string file_name = FXAI_LiveDeploymentProfileFile(symbol);
   int handle = FileOpen(file_name,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      cache = out;
      cache_symbol = symbol;
      cache_time = now;
      return false;
   }

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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "teacher_weight")
         out.teacher_weight = StringToDouble(value);
      else if(key == "student_weight")
         out.student_weight = StringToDouble(value);
      else if(key == "analog_weight")
         out.analog_weight = StringToDouble(value);
      else if(key == "foundation_weight")
         out.foundation_weight = StringToDouble(value);
      else if(key == "policy_trade_floor")
         out.policy_trade_floor = StringToDouble(value);
      else if(key == "policy_size_bias")
         out.policy_size_bias = StringToDouble(value);
      else if(key == "portfolio_budget_bias")
         out.portfolio_budget_bias = StringToDouble(value);
      else if(key == "challenger_promote_margin")
         out.challenger_promote_margin = StringToDouble(value);
      else if(key == "regime_transition_weight")
         out.regime_transition_weight = StringToDouble(value);
      else if(key == "macro_quality_floor")
         out.macro_quality_floor = StringToDouble(value);
      else if(key == "policy_no_trade_cap")
         out.policy_no_trade_cap = StringToDouble(value);
      else if(key == "capital_efficiency_bias")
         out.capital_efficiency_bias = StringToDouble(value);
      else if(key == "supervisor_blend")
         out.supervisor_blend = StringToDouble(value);
      else if(key == "teacher_signal_gain")
         out.teacher_signal_gain = StringToDouble(value);
      else if(key == "student_signal_gain")
         out.student_signal_gain = StringToDouble(value);
      else if(key == "foundation_quality_gain")
         out.foundation_quality_gain = StringToDouble(value);
      else if(key == "macro_state_gain")
         out.macro_state_gain = StringToDouble(value);
      else if(key == "policy_lifecycle_gain")
         out.policy_lifecycle_gain = StringToDouble(value);
      else if(key == "policy_hold_floor")
         out.policy_hold_floor = StringToDouble(value);
      else if(key == "policy_exit_floor")
         out.policy_exit_floor = StringToDouble(value);
      else if(key == "policy_add_floor")
         out.policy_add_floor = StringToDouble(value);
      else if(key == "policy_reduce_floor")
         out.policy_reduce_floor = StringToDouble(value);
      else if(key == "policy_timeout_floor")
         out.policy_timeout_floor = StringToDouble(value);
      else if(key == "max_add_fraction")
         out.max_add_fraction = StringToDouble(value);
      else if(key == "reduce_fraction")
         out.reduce_fraction = StringToDouble(value);
      else if(key == "soft_timeout_bars")
         out.soft_timeout_bars = (int)StringToInteger(value);
      else if(key == "hard_timeout_bars")
         out.hard_timeout_bars = (int)StringToInteger(value);
      else if(key == "runtime_mode")
         out.runtime_mode = value;
      else if(key == "telemetry_level")
         out.telemetry_level = value;
      else if(key == "performance_budget_ms")
         out.performance_budget_ms = StringToDouble(value);
      else if(key == "shadow_enabled")
         out.shadow_enabled = (StringToInteger(value) != 0);
      else if(key == "snapshot_detail")
         out.snapshot_detail = value;
      else if(key == "max_runtime_models")
         out.max_runtime_models = (int)StringToInteger(value);
      else if(key == "promotion_tier")
         out.promotion_tier = value;
   }
   FileClose(handle);

   out.teacher_weight = FXAI_Clamp(out.teacher_weight, 0.05, 0.95);
   out.student_weight = FXAI_Clamp(out.student_weight, 0.05, 0.95);
   out.analog_weight = FXAI_Clamp(out.analog_weight, 0.0, 0.80);
   out.foundation_weight = FXAI_Clamp(out.foundation_weight, 0.0, 0.90);
   out.policy_trade_floor = FXAI_Clamp(out.policy_trade_floor, 0.20, 0.90);
   out.policy_size_bias = FXAI_Clamp(out.policy_size_bias, 0.40, 1.60);
   out.portfolio_budget_bias = FXAI_Clamp(out.portfolio_budget_bias, 0.40, 1.60);
   out.challenger_promote_margin = FXAI_Clamp(out.challenger_promote_margin, 0.50, 3.00);
   out.regime_transition_weight = FXAI_Clamp(out.regime_transition_weight, 0.0, 1.0);
   out.macro_quality_floor = FXAI_Clamp(out.macro_quality_floor, 0.0, 1.0);
   out.policy_no_trade_cap = FXAI_Clamp(out.policy_no_trade_cap, 0.25, 0.95);
   out.capital_efficiency_bias = FXAI_Clamp(out.capital_efficiency_bias, 0.40, 1.80);
   out.supervisor_blend = FXAI_Clamp(out.supervisor_blend, 0.0, 1.0);
   out.teacher_signal_gain = FXAI_Clamp(out.teacher_signal_gain, 0.40, 1.80);
   out.student_signal_gain = FXAI_Clamp(out.student_signal_gain, 0.40, 1.80);
   out.foundation_quality_gain = FXAI_Clamp(out.foundation_quality_gain, 0.40, 1.80);
   out.macro_state_gain = FXAI_Clamp(out.macro_state_gain, 0.40, 1.80);
   out.policy_lifecycle_gain = FXAI_Clamp(out.policy_lifecycle_gain, 0.40, 1.80);
   out.policy_hold_floor = FXAI_Clamp(out.policy_hold_floor, 0.20, 0.95);
   out.policy_exit_floor = FXAI_Clamp(out.policy_exit_floor, 0.20, 0.99);
   out.policy_add_floor = FXAI_Clamp(out.policy_add_floor, 0.20, 0.99);
   out.policy_reduce_floor = FXAI_Clamp(out.policy_reduce_floor, 0.20, 0.99);
   out.policy_timeout_floor = FXAI_Clamp(out.policy_timeout_floor, 0.20, 0.99);
   out.max_add_fraction = FXAI_Clamp(out.max_add_fraction, 0.05, 1.00);
   out.reduce_fraction = FXAI_Clamp(out.reduce_fraction, 0.05, 0.95);
   out.soft_timeout_bars = (int)FXAI_Clamp((double)out.soft_timeout_bars, 1.0, 10000.0);
   out.hard_timeout_bars = (int)FXAI_Clamp((double)out.hard_timeout_bars, (double)(out.soft_timeout_bars + 1), 20000.0);
   if(out.runtime_mode != "production")
      out.runtime_mode = "research";
   if(out.telemetry_level != "lean")
      out.telemetry_level = "full";
   if(out.snapshot_detail != "lean")
      out.snapshot_detail = "full";
   out.performance_budget_ms = FXAI_Clamp(out.performance_budget_ms, 2.0, 100.0);
   out.max_runtime_models = (int)FXAI_Clamp((double)out.max_runtime_models, 1.0, (double)FXAI_AI_COUNT);
   if(out.promotion_tier != "production-approved" &&
      out.promotion_tier != "audit-approved" &&
      out.promotion_tier != "research-approved")
      out.promotion_tier = "experimental";
   out.ready = true;
   out.loaded_at = now;

   cache = out;
   cache_symbol = symbol;
   cache_time = now;
   return true;
}

bool FXAI_LoadPortfolioSupervisorProfile(FXAIPortfolioSupervisorProfile &out,
                                         const bool force_reload = false)
{
   static FXAIPortfolioSupervisorProfile cache;
   static datetime cache_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache.ready &&
      cache_time > 0 &&
      now > 0 &&
      (now - cache_time) < 300)
   {
      out = cache;
      return true;
   }

   FXAI_ResetPortfolioSupervisorProfile(out);
   int handle = FileOpen(FXAI_PORTFOLIO_SUPERVISOR_FILE,
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      cache = out;
      cache_time = now;
      return false;
   }

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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "gross_budget_bias")
         out.gross_budget_bias = StringToDouble(value);
      else if(key == "correlated_budget_bias")
         out.correlated_budget_bias = StringToDouble(value);
      else if(key == "directional_budget_bias")
         out.directional_budget_bias = StringToDouble(value);
      else if(key == "capital_risk_cap_pct")
         out.capital_risk_cap_pct = StringToDouble(value);
      else if(key == "macro_overlap_cap")
         out.macro_overlap_cap = StringToDouble(value);
      else if(key == "concentration_cap")
         out.concentration_cap = StringToDouble(value);
      else if(key == "supervisor_weight")
         out.supervisor_weight = StringToDouble(value);
      else if(key == "hard_block_score")
         out.hard_block_score = StringToDouble(value);
      else if(key == "policy_enter_floor")
         out.policy_enter_floor = StringToDouble(value);
      else if(key == "policy_no_trade_ceiling")
         out.policy_no_trade_ceiling = StringToDouble(value);
   }
   FileClose(handle);

   out.gross_budget_bias = FXAI_Clamp(out.gross_budget_bias, 0.40, 1.60);
   out.correlated_budget_bias = FXAI_Clamp(out.correlated_budget_bias, 0.40, 1.60);
   out.directional_budget_bias = FXAI_Clamp(out.directional_budget_bias, 0.40, 1.60);
   out.capital_risk_cap_pct = FXAI_Clamp(out.capital_risk_cap_pct, 0.10, 10.0);
   out.macro_overlap_cap = FXAI_Clamp(out.macro_overlap_cap, 0.10, 2.0);
   out.concentration_cap = FXAI_Clamp(out.concentration_cap, 0.10, 2.0);
   out.supervisor_weight = FXAI_Clamp(out.supervisor_weight, 0.0, 1.0);
   out.hard_block_score = FXAI_Clamp(out.hard_block_score, 0.20, 3.0);
   out.policy_enter_floor = FXAI_Clamp(out.policy_enter_floor, 0.10, 0.95);
   out.policy_no_trade_ceiling = FXAI_Clamp(out.policy_no_trade_ceiling, 0.10, 0.99);
   out.ready = true;
   out.loaded_at = now;

   cache = out;
   cache_time = now;
   return true;
}

bool FXAI_LoadSupervisorServiceStateFromFile(const string file_name,
                                             FXAISupervisorServiceState &out)
{
   FXAI_ResetSupervisorServiceState(out);
   int handle = FileOpen(file_name,
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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "generated_at")
         out.generated_at = (datetime)StringToInteger(value);
      else if(key == "expires_at")
         out.expires_at = (datetime)StringToInteger(value);
      else if(key == "snapshot_count")
         out.snapshot_count = (int)StringToInteger(value);
      else if(key == "gross_pressure")
         out.gross_pressure = StringToDouble(value);
      else if(key == "directional_long_pressure")
         out.directional_long_pressure = StringToDouble(value);
      else if(key == "directional_short_pressure")
         out.directional_short_pressure = StringToDouble(value);
      else if(key == "macro_pressure")
         out.macro_pressure = StringToDouble(value);
      else if(key == "concentration_pressure")
         out.concentration_pressure = StringToDouble(value);
      else if(key == "freshness_penalty")
         out.freshness_penalty = StringToDouble(value);
      else if(key == "pressure_velocity")
         out.pressure_velocity = StringToDouble(value);
      else if(key == "gross_velocity")
         out.gross_velocity = StringToDouble(value);
      else if(key == "long_entry_budget_mult")
         out.long_entry_budget_mult = StringToDouble(value);
      else if(key == "short_entry_budget_mult")
         out.short_entry_budget_mult = StringToDouble(value);
      else if(key == "budget_multiplier")
         out.budget_multiplier = StringToDouble(value);
      else if(key == "add_multiplier")
         out.add_multiplier = StringToDouble(value);
      else if(key == "reduce_bias")
         out.reduce_bias = StringToDouble(value);
      else if(key == "exit_bias")
         out.exit_bias = StringToDouble(value);
      else if(key == "entry_floor")
         out.entry_floor = StringToDouble(value);
      else if(key == "block_score")
         out.block_score = StringToDouble(value);
      else if(key == "supervisor_score")
         out.supervisor_score = StringToDouble(value);
   }
   FileClose(handle);

   out.snapshot_count = (int)FXAI_Clamp((double)out.snapshot_count, 0.0, 10000.0);
   out.gross_pressure = FXAI_Clamp(out.gross_pressure, 0.0, 2.0);
   out.directional_long_pressure = FXAI_Clamp(out.directional_long_pressure, 0.0, 2.0);
   out.directional_short_pressure = FXAI_Clamp(out.directional_short_pressure, 0.0, 2.0);
   out.macro_pressure = FXAI_Clamp(out.macro_pressure, 0.0, 1.5);
   out.concentration_pressure = FXAI_Clamp(out.concentration_pressure, 0.0, 1.0);
   out.freshness_penalty = FXAI_Clamp(out.freshness_penalty, 0.0, 1.0);
   out.pressure_velocity = FXAI_Clamp(out.pressure_velocity, -1.0, 1.0);
   out.gross_velocity = FXAI_Clamp(out.gross_velocity, -1.0, 1.0);
   out.long_entry_budget_mult = FXAI_Clamp(out.long_entry_budget_mult, 0.10, 1.20);
   out.short_entry_budget_mult = FXAI_Clamp(out.short_entry_budget_mult, 0.10, 1.20);
   out.budget_multiplier = FXAI_Clamp(out.budget_multiplier, 0.10, 1.20);
   if(MathAbs(out.long_entry_budget_mult - 1.0) < 1e-6 &&
      MathAbs(out.short_entry_budget_mult - 1.0) < 1e-6 &&
      MathAbs(out.budget_multiplier - 1.0) > 1e-6)
   {
      out.long_entry_budget_mult = out.budget_multiplier;
      out.short_entry_budget_mult = out.budget_multiplier;
   }
   out.add_multiplier = FXAI_Clamp(out.add_multiplier, 0.10, 1.40);
   out.reduce_bias = FXAI_Clamp(out.reduce_bias, 0.0, 1.0);
   out.exit_bias = FXAI_Clamp(out.exit_bias, 0.0, 1.0);
   out.entry_floor = FXAI_Clamp(out.entry_floor, 0.10, 0.95);
   out.block_score = FXAI_Clamp(out.block_score, 0.20, 3.0);
   out.supervisor_score = FXAI_Clamp(out.supervisor_score, 0.0, 3.0);
   out.ready = (StringLen(out.symbol) > 0 &&
                FXAI_ControlPlaneArtifactFresh(out.generated_at, out.expires_at, 240));
   out.loaded_at = TimeCurrent();
   if(out.loaded_at <= 0)
      out.loaded_at = TimeTradeServer();
   return out.ready;
}

bool FXAI_LoadSupervisorServiceState(const string symbol,
                                     FXAISupervisorServiceState &out,
                                     const bool force_reload = false)
{
   static FXAISupervisorServiceState cache_symbol;
   static FXAISupervisorServiceState cache_global;
   static string cache_symbol_name = "";
   static datetime cache_symbol_time = 0;
   static datetime cache_global_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache_symbol.ready &&
      cache_symbol_name == symbol &&
      cache_symbol_time > 0 &&
      now > 0 &&
      (now - cache_symbol_time) < 90)
   {
      out = cache_symbol;
      return true;
   }

   if(FXAI_LoadSupervisorServiceStateFromFile(FXAI_SupervisorServiceSymbolFile(symbol), out))
   {
      cache_symbol = out;
      cache_symbol_name = symbol;
      cache_symbol_time = now;
      return true;
   }

   if(!force_reload &&
      cache_global.ready &&
      cache_global_time > 0 &&
      now > 0 &&
      (now - cache_global_time) < 90)
   {
      out = cache_global;
      if(StringLen(out.symbol) <= 0)
         out.symbol = symbol;
      return true;
   }

   if(FXAI_LoadSupervisorServiceStateFromFile(FXAI_SUPERVISOR_SERVICE_GLOBAL_FILE, out))
   {
      cache_global = out;
      cache_global_time = now;
      if(StringLen(out.symbol) <= 0 || out.symbol == "__GLOBAL__")
         out.symbol = symbol;
      return true;
   }

   FXAI_ResetSupervisorServiceState(out);
   out.symbol = symbol;
   return false;
}

bool FXAI_LoadStudentRouterProfile(const string symbol,
                                   FXAIStudentRouterProfile &out,
                                   const bool force_reload = false)
{
   static FXAIStudentRouterProfile cache;
   static string cache_symbol = "";
   static datetime cache_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache.ready &&
      cache_symbol == symbol &&
      cache_time > 0 &&
      now > 0 &&
      (now - cache_time) < 300)
   {
      out = cache;
      return true;
   }

   FXAI_ResetStudentRouterProfile(out);
   out.symbol = symbol;
   int handle = FileOpen(FXAI_StudentRouterProfileFile(symbol),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      cache = out;
      cache_symbol = symbol;
      cache_time = now;
      return false;
   }

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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "champion_only")
         out.champion_only = (StringToInteger(value) != 0);
      else if(key == "max_active_models")
         out.max_active_models = (int)StringToInteger(value);
      else if(key == "min_meta_weight")
         out.min_meta_weight = StringToDouble(value);
      else if(key == "allow_plugins_csv")
         out.allow_plugins_csv = value;
      else if(key == "plugin_weights_csv")
         out.plugin_weights_csv = value;
      else if(StringFind(key, "family_weight_") == 0)
      {
         string family_name = StringSubstr(key, 14);
         for(int fam=0; fam<=FXAI_FAMILY_OTHER; fam++)
         {
            if(FXAI_FamilyName(fam) == family_name)
            {
               out.family_weight[fam] = StringToDouble(value);
               break;
            }
         }
      }
   }
   FileClose(handle);

   out.max_active_models = (int)FXAI_Clamp((double)out.max_active_models, 1.0, (double)FXAI_AI_COUNT);
   out.min_meta_weight = FXAI_Clamp(out.min_meta_weight, 0.0, 0.25);
   for(int fam=0; fam<=FXAI_FAMILY_OTHER; fam++)
      out.family_weight[fam] = FXAI_Clamp(out.family_weight[fam], 0.05, 1.50);
   out.ready = true;
   out.loaded_at = now;
   cache = out;
   cache_symbol = symbol;
   cache_time = now;
   return true;
}

int FXAI_AdaptiveRouterRegimeIndexByLabel(const string label)
{
   if(label == "TREND_PERSISTENT") return 0;
   if(label == "RANGE_MEAN_REVERTING") return 1;
   if(label == "BREAKOUT_TRANSITION") return 2;
   if(label == "HIGH_VOL_EVENT") return 3;
   if(label == "RISK_ON_OFF_MACRO") return 4;
   if(label == "LIQUIDITY_STRESS") return 5;
   if(label == "SESSION_FLOW") return 6;
   return -1;
}

int FXAI_AdaptiveRouterSessionIndexByLabel(const string label)
{
   if(label == "ASIA") return 0;
   if(label == "LONDON") return 1;
   if(label == "NEWYORK") return 2;
   if(label == "LONDON_NY_OVERLAP") return 3;
   if(label == "ROLLOVER") return 4;
   return -1;
}

bool FXAI_LoadAdaptiveRouterProfile(const string symbol,
                                    FXAIAdaptiveRouterProfile &out,
                                    const bool force_reload = false)
{
   static FXAIAdaptiveRouterProfile cache;
   static string cache_symbol = "";
   static datetime cache_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache.ready &&
      cache_symbol == symbol &&
      cache_time > 0 &&
      now > 0 &&
      (now - cache_time) < 300)
   {
      out = cache;
      return true;
   }

   FXAI_ResetAdaptiveRouterProfile(out);
   out.symbol = symbol;
   int handle = FileOpen(FXAI_AdaptiveRouterProfileFile(symbol),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      cache = out;
      cache_symbol = symbol;
      cache_time = now;
      return false;
   }

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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "enabled")
         out.enabled = (StringToInteger(value) != 0);
      else if(key == "router_mode")
         out.router_mode = value;
      else if(key == "fallback_to_student_router_only")
         out.fallback_to_student_router_only = (StringToInteger(value) != 0);
      else if(key == "pair_tags_csv")
         out.pair_tags_csv = value;
      else if(key == "caution_threshold")
         out.caution_threshold = StringToDouble(value);
      else if(key == "abstain_threshold")
         out.abstain_threshold = StringToDouble(value);
      else if(key == "block_threshold")
         out.block_threshold = StringToDouble(value);
      else if(key == "confidence_floor")
         out.confidence_floor = StringToDouble(value);
      else if(key == "suppression_threshold")
         out.suppression_threshold = StringToDouble(value);
      else if(key == "downweight_threshold")
         out.downweight_threshold = StringToDouble(value);
      else if(key == "stale_news_abstain_bias")
         out.stale_news_abstain_bias = StringToDouble(value);
      else if(key == "stale_news_force_caution")
         out.stale_news_force_caution = (StringToInteger(value) != 0);
      else if(key == "min_plugin_weight")
         out.min_plugin_weight = StringToDouble(value);
      else if(key == "max_plugin_weight")
         out.max_plugin_weight = StringToDouble(value);
      else if(key == "max_active_weight_share")
         out.max_active_weight_share = StringToDouble(value);
      else if(key == "plugin_global_weights_csv")
         out.plugin_global_weights_csv = value;
      else if(key == "plugin_news_compatibility_csv")
         out.plugin_news_compatibility_csv = value;
      else if(key == "plugin_liquidity_robustness_csv")
         out.plugin_liquidity_robustness_csv = value;
      else if(StringFind(key, "plugin_regime_") == 0 && StringFind(key, "_csv") > 0)
      {
         string regime_label = StringSubstr(key, 14, StringLen(key) - 18);
         int regime_index = FXAI_AdaptiveRouterRegimeIndexByLabel(regime_label);
         if(regime_index >= 0 && regime_index < FXAI_ADAPTIVE_ROUTER_REGIME_COUNT)
            out.plugin_regime_weights_csv[regime_index] = value;
      }
      else if(StringFind(key, "plugin_session_") == 0 && StringFind(key, "_csv") > 0)
      {
         string session_label = StringSubstr(key, 15, StringLen(key) - 19);
         int session_index = FXAI_AdaptiveRouterSessionIndexByLabel(session_label);
         if(session_index >= 0 && session_index < FXAI_ADAPTIVE_ROUTER_SESSION_COUNT)
            out.plugin_session_weights_csv[session_index] = value;
      }
   }
   FileClose(handle);

   out.caution_threshold = FXAI_Clamp(out.caution_threshold, 0.10, 1.50);
   out.abstain_threshold = FXAI_Clamp(out.abstain_threshold, 0.05, out.caution_threshold);
   out.block_threshold = FXAI_Clamp(out.block_threshold, 0.01, out.abstain_threshold);
   out.confidence_floor = FXAI_Clamp(out.confidence_floor, 0.0, 1.0);
   out.suppression_threshold = FXAI_Clamp(out.suppression_threshold, 0.05, 2.50);
   out.downweight_threshold = FXAI_Clamp(out.downweight_threshold, 0.05, 2.50);
   out.stale_news_abstain_bias = FXAI_Clamp(out.stale_news_abstain_bias, 0.0, 1.0);
   out.min_plugin_weight = FXAI_Clamp(out.min_plugin_weight, 0.01, 1.0);
   out.max_plugin_weight = FXAI_Clamp(out.max_plugin_weight, out.min_plugin_weight, 3.0);
   out.max_active_weight_share = FXAI_Clamp(out.max_active_weight_share, 0.10, 0.99);
   if(out.router_mode != "WEIGHTED_ENSEMBLE")
      out.router_mode = "WEIGHTED_ENSEMBLE";
   out.ready = true;
   out.loaded_at = now;

   cache = out;
   cache_symbol = symbol;
   cache_time = now;
   return true;
}

double FXAI_AdaptiveRouterPluginGlobalWeight(const FXAIAdaptiveRouterProfile &profile,
                                             const string plugin_name)
{
   if(!profile.ready || !profile.enabled)
      return 1.0;
   double value = FXAI_CSVMapWeight(profile.plugin_global_weights_csv, plugin_name, 1.0);
   return FXAI_Clamp(value, profile.min_plugin_weight, profile.max_plugin_weight);
}

double FXAI_AdaptiveRouterPluginNewsCompatibility(const FXAIAdaptiveRouterProfile &profile,
                                                  const string plugin_name)
{
   if(!profile.ready || !profile.enabled)
      return 1.0;
   double value = FXAI_CSVMapWeight(profile.plugin_news_compatibility_csv, plugin_name, 1.0);
   return FXAI_Clamp(value, 0.05, 2.50);
}

double FXAI_AdaptiveRouterPluginLiquidityRobustness(const FXAIAdaptiveRouterProfile &profile,
                                                    const string plugin_name)
{
   if(!profile.ready || !profile.enabled)
      return 1.0;
   double value = FXAI_CSVMapWeight(profile.plugin_liquidity_robustness_csv, plugin_name, 1.0);
   return FXAI_Clamp(value, 0.05, 2.50);
}

double FXAI_AdaptiveRouterPluginRegimeWeight(const FXAIAdaptiveRouterProfile &profile,
                                             const string plugin_name,
                                             const string regime_label)
{
   if(!profile.ready || !profile.enabled)
      return 1.0;
   int regime_index = FXAI_AdaptiveRouterRegimeIndexByLabel(regime_label);
   if(regime_index < 0 || regime_index >= FXAI_ADAPTIVE_ROUTER_REGIME_COUNT)
      return 1.0;
   double value = FXAI_CSVMapWeight(profile.plugin_regime_weights_csv[regime_index], plugin_name, 1.0);
   return FXAI_Clamp(value, 0.05, 2.50);
}

double FXAI_AdaptiveRouterPluginSessionWeight(const FXAIAdaptiveRouterProfile &profile,
                                              const string plugin_name,
                                              const string session_label)
{
   if(!profile.ready || !profile.enabled)
      return 1.0;
   int session_index = FXAI_AdaptiveRouterSessionIndexByLabel(session_label);
   if(session_index < 0 || session_index >= FXAI_ADAPTIVE_ROUTER_SESSION_COUNT)
      return 1.0;
   double value = FXAI_CSVMapWeight(profile.plugin_session_weights_csv[session_index], plugin_name, 1.0);
   return FXAI_Clamp(value, 0.05, 2.50);
}

double FXAI_StudentRouterFamilyWeight(const FXAIStudentRouterProfile &profile,
                                      const int family_id)
{
   if(!profile.ready || family_id < 0 || family_id > FXAI_FAMILY_OTHER)
      return 1.0;
   return FXAI_Clamp(profile.family_weight[family_id], 0.05, 1.50);
}

double FXAI_StudentRouterPluginWeight(const FXAIStudentRouterProfile &profile,
                                      const string plugin_name,
                                      const int family_id)
{
   double family_weight = FXAI_StudentRouterFamilyWeight(profile, family_id);
   if(!profile.ready)
      return family_weight;
   double plugin_weight = FXAI_CSVMapWeight(profile.plugin_weights_csv, plugin_name, family_weight);
   return FXAI_Clamp(plugin_weight, 0.01, 1.60);
}

bool FXAI_StudentRouterAllowsPlugin(const FXAIStudentRouterProfile &profile,
                                    const string plugin_name,
                                    const int family_id)
{
   if(!profile.ready)
      return true;
   if(family_id >= 0 && family_id <= FXAI_FAMILY_OTHER &&
      profile.family_weight[family_id] <= 0.051)
      return false;
   if(FXAI_StudentRouterPluginWeight(profile, plugin_name, family_id) <= 0.021)
      return false;
   if(!profile.champion_only)
      return true;
   if(StringLen(profile.allow_plugins_csv) <= 0)
      return true;
   return FXAI_CSVContainsToken(profile.allow_plugins_csv, plugin_name);
}

bool FXAI_LoadSupervisorCommandState(const string symbol,
                                     FXAISupervisorCommandState &out,
                                     const bool force_reload = false)
{
   static FXAISupervisorCommandState cache_symbol;
   static FXAISupervisorCommandState cache_global;
   static string cache_symbol_name = "";
   static datetime cache_symbol_time = 0;
   static datetime cache_global_time = 0;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0 && !FXAI_MarketDataBarTime(_Symbol, PERIOD_M1, 0, now))
      now = 0;

   if(!force_reload &&
      cache_symbol.ready &&
      cache_symbol_name == symbol &&
      cache_symbol_time > 0 &&
      now > 0 &&
      (now - cache_symbol_time) < 90)
   {
      out = cache_symbol;
      return true;
   }

   FXAI_ResetSupervisorCommandState(out);
   out.symbol = symbol;
   int handle = FileOpen(FXAI_SupervisorCommandSymbolFile(symbol),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      if(!force_reload &&
         cache_global.ready &&
         cache_global_time > 0 &&
         now > 0 &&
         (now - cache_global_time) < 90)
      {
         out = cache_global;
         if(StringLen(out.symbol) <= 0 || out.symbol == "__GLOBAL__")
            out.symbol = symbol;
         return true;
      }
      int global_handle = FileOpen(FXAI_SUPERVISOR_COMMAND_GLOBAL_FILE,
                                   FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                                   FILE_SHARE_READ | FILE_SHARE_WRITE);
      if(global_handle == INVALID_HANDLE)
         return false;
      handle = global_handle;
   }

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
      if(key == "profile_name")
         out.profile_name = value;
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "generated_at")
         out.generated_at = (datetime)StringToInteger(value);
      else if(key == "expires_at")
         out.expires_at = (datetime)StringToInteger(value);
      else if(key == "entry_budget_mult")
         out.entry_budget_mult = StringToDouble(value);
      else if(key == "long_entry_budget_mult")
         out.long_entry_budget_mult = StringToDouble(value);
      else if(key == "short_entry_budget_mult")
         out.short_entry_budget_mult = StringToDouble(value);
      else if(key == "hold_budget_mult")
         out.hold_budget_mult = StringToDouble(value);
      else if(key == "add_cap_mult")
         out.add_cap_mult = StringToDouble(value);
      else if(key == "reduce_bias")
         out.reduce_bias = StringToDouble(value);
      else if(key == "exit_bias")
         out.exit_bias = StringToDouble(value);
      else if(key == "tighten_bias")
         out.tighten_bias = StringToDouble(value);
      else if(key == "timeout_bias")
         out.timeout_bias = StringToDouble(value);
      else if(key == "long_block")
         out.long_block = (StringToInteger(value) != 0);
      else if(key == "short_block")
         out.short_block = (StringToInteger(value) != 0);
      else if(key == "block_score")
         out.block_score = StringToDouble(value);
      else if(key == "max_active_models")
         out.max_active_models = (int)StringToInteger(value);
      else if(key == "champion_only")
         out.champion_only = (StringToInteger(value) != 0);
   }
   FileClose(handle);

   out.entry_budget_mult = FXAI_Clamp(out.entry_budget_mult, 0.10, 1.20);
   out.long_entry_budget_mult = FXAI_Clamp(out.long_entry_budget_mult, 0.10, 1.20);
   out.short_entry_budget_mult = FXAI_Clamp(out.short_entry_budget_mult, 0.10, 1.20);
   if(MathAbs(out.long_entry_budget_mult - 1.0) < 1e-6 &&
      MathAbs(out.short_entry_budget_mult - 1.0) < 1e-6 &&
      MathAbs(out.entry_budget_mult - 1.0) > 1e-6)
   {
      out.long_entry_budget_mult = out.entry_budget_mult;
      out.short_entry_budget_mult = out.entry_budget_mult;
   }
   out.hold_budget_mult = FXAI_Clamp(out.hold_budget_mult, 0.10, 1.20);
   out.add_cap_mult = FXAI_Clamp(out.add_cap_mult, 0.05, 1.20);
   out.reduce_bias = FXAI_Clamp(out.reduce_bias, 0.0, 1.0);
   out.exit_bias = FXAI_Clamp(out.exit_bias, 0.0, 1.0);
   out.tighten_bias = FXAI_Clamp(out.tighten_bias, 0.0, 1.0);
   out.timeout_bias = FXAI_Clamp(out.timeout_bias, 0.0, 1.0);
   out.block_score = FXAI_Clamp(out.block_score, 0.20, 3.0);
   out.max_active_models = (int)FXAI_Clamp((double)out.max_active_models, 1.0, (double)FXAI_AI_COUNT);
   if(!FXAI_ControlPlaneArtifactFresh(out.generated_at, out.expires_at, 240))
      return false;
   out.ready = true;
   out.loaded_at = now;
   if(out.symbol == "__GLOBAL__")
   {
      cache_global = out;
      cache_global_time = now;
      out.symbol = symbol;
   }
   else
   {
      cache_symbol = out;
      cache_symbol_name = symbol;
      cache_symbol_time = now;
   }
   return true;
}

bool FXAI_SupervisorCommandBlocksDirection(const FXAISupervisorCommandState &state,
                                           const int direction)
{
   if(!state.ready)
      return false;
   if(direction == 1)
      return state.long_block;
   if(direction == 0)
      return state.short_block;
   return (state.long_block && state.short_block);
}

double FXAI_SupervisorCommandEntryBudgetMult(const FXAISupervisorCommandState &state,
                                             const int direction)
{
   if(!state.ready)
      return 1.0;
   if(direction == 1)
      return FXAI_Clamp(state.long_entry_budget_mult, 0.10, 1.20);
   if(direction == 0)
      return FXAI_Clamp(state.short_entry_budget_mult, 0.10, 1.20);
   return FXAI_Clamp(state.entry_budget_mult, 0.10, 1.20);
}
#endif // __FXAI_RUNTIME_CONTROL_PLANE_PROFILES_MQH__
