#ifndef __FXAI_RUNTIME_CONTROL_PLANE_MQH__
#define __FXAI_RUNTIME_CONTROL_PLANE_MQH__

#define FXAI_CONTROL_PLANE_DIR "FXAI\\ControlPlane"
#define FXAI_CONTROL_PLANE_TTL_SEC 7200
#define FXAI_PORTFOLIO_SUPERVISOR_FILE "FXAI\\Offline\\Promotions\\fxai_portfolio_supervisor.tsv"

struct FXAILiveDeploymentProfile
{
   bool   ready;
   string profile_name;
   string symbol;
   double teacher_weight;
   double student_weight;
   double analog_weight;
   double foundation_weight;
   double policy_trade_floor;
   double policy_size_bias;
   double portfolio_budget_bias;
   double challenger_promote_margin;
   double regime_transition_weight;
   double macro_quality_floor;
   double policy_no_trade_cap;
   double capital_efficiency_bias;
   double supervisor_blend;
   datetime loaded_at;
};

struct FXAIPortfolioSupervisorProfile
{
   bool   ready;
   string profile_name;
   double gross_budget_bias;
   double correlated_budget_bias;
   double directional_budget_bias;
   double capital_risk_cap_pct;
   double macro_overlap_cap;
   double concentration_cap;
   double supervisor_weight;
   double hard_block_score;
   double policy_enter_floor;
   double policy_no_trade_ceiling;
   datetime loaded_at;
};

struct FXAIControlPlaneSnapshot
{
   bool   valid;
   long   login;
   ulong  magic;
   long   chart_id;
   string symbol;
   datetime bar_time;
   int    direction;
   double signal_intensity;
   double confidence;
   double reliability;
   double trade_gate;
   double hierarchy_score;
   double macro_quality;
   double trade_edge_norm;
   double expected_move_norm;
   double policy_trade_prob;
   double policy_no_trade_prob;
   double policy_enter_prob;
   double policy_exit_prob;
   double policy_size_mult;
   double policy_portfolio_fit;
   double policy_capital_efficiency;
   double gross_exposure_lots;
   double correlated_exposure_lots;
   double directional_cluster_lots;
   double capital_risk_pct;
   double portfolio_pressure;
};

struct FXAIControlPlaneAggregate
{
   int    peer_count;
   double gross_intensity;
   double correlated_intensity;
   double directional_intensity;
   double macro_overlap;
   double quality_overlap;
   double diversity_bonus;
   double concentration_penalty;
   double max_capital_risk_pct;
   double mean_trade_prob;
   double mean_no_trade_prob;
   double mean_capital_efficiency;
   double mean_portfolio_fit;
   double supervisor_score;
   double score;
};

double g_control_plane_last_score = 0.0;
double g_control_plane_last_buy_score = 0.0;
double g_control_plane_last_sell_score = 0.0;
string g_control_plane_last_symbol = "";
datetime g_control_plane_last_bar_time = 0;
double g_portfolio_supervisor_last_score = 0.0;
double g_portfolio_supervisor_last_capital_risk_pct = 0.0;

void FXAI_ParseSymbolLegs(const string symbol,
                         string &base_out,
                         string &quote_out);
double FXAI_CorrelationExposureWeight(const string anchor_symbol,
                                      const string other_symbol);
double FXAI_DirectionalClusterAlignment(const string anchor_symbol,
                                        const int anchor_direction,
                                        const string other_symbol,
                                        const int other_direction);
double FXAI_ManagedExposureLots(const string symbol);
double FXAI_ManagedCorrelatedExposureLots(const string symbol);
double FXAI_ManagedDirectionalClusterLots(const string symbol,
                                          const int direction);
double FXAI_EstimatedRiskPointsForDecision();
double FXAI_MoneyPerPointPerLot(const string symbol);

void FXAI_ResetLiveDeploymentProfile(FXAILiveDeploymentProfile &out)
{
   out.ready = false;
   out.profile_name = "";
   out.symbol = "";
   out.teacher_weight = 0.58;
   out.student_weight = 0.42;
   out.analog_weight = 0.18;
   out.foundation_weight = 0.24;
   out.policy_trade_floor = 0.52;
   out.policy_size_bias = 1.0;
   out.portfolio_budget_bias = 1.0;
   out.challenger_promote_margin = 1.0;
   out.regime_transition_weight = 0.35;
   out.macro_quality_floor = 0.24;
   out.policy_no_trade_cap = 0.62;
   out.capital_efficiency_bias = 1.0;
   out.supervisor_blend = 0.45;
   out.loaded_at = 0;
}

void FXAI_ResetPortfolioSupervisorProfile(FXAIPortfolioSupervisorProfile &out)
{
   out.ready = false;
   out.profile_name = "";
   out.gross_budget_bias = 1.0;
   out.correlated_budget_bias = 1.0;
   out.directional_budget_bias = 1.0;
   out.capital_risk_cap_pct = 1.20;
   out.macro_overlap_cap = 0.92;
   out.concentration_cap = 0.82;
   out.supervisor_weight = 0.45;
   out.hard_block_score = 1.08;
   out.policy_enter_floor = 0.42;
   out.policy_no_trade_ceiling = 0.74;
   out.loaded_at = 0;
}

void FXAI_ResetControlPlaneSnapshot(FXAIControlPlaneSnapshot &out)
{
   out.valid = false;
   out.login = 0;
   out.magic = 0;
   out.chart_id = 0;
   out.symbol = "";
   out.bar_time = 0;
   out.direction = -1;
   out.signal_intensity = 0.0;
   out.confidence = 0.0;
   out.reliability = 0.0;
   out.trade_gate = 0.0;
   out.hierarchy_score = 0.0;
   out.macro_quality = 0.0;
   out.trade_edge_norm = 0.0;
   out.expected_move_norm = 0.0;
   out.policy_trade_prob = 0.0;
   out.policy_no_trade_prob = 1.0;
   out.policy_enter_prob = 0.0;
   out.policy_exit_prob = 0.0;
   out.policy_size_mult = 0.0;
   out.policy_portfolio_fit = 0.0;
   out.policy_capital_efficiency = 0.0;
   out.gross_exposure_lots = 0.0;
   out.correlated_exposure_lots = 0.0;
   out.directional_cluster_lots = 0.0;
   out.capital_risk_pct = 0.0;
   out.portfolio_pressure = 0.0;
}

void FXAI_ResetControlPlaneAggregate(FXAIControlPlaneAggregate &out)
{
   out.peer_count = 0;
   out.gross_intensity = 0.0;
   out.correlated_intensity = 0.0;
   out.directional_intensity = 0.0;
   out.macro_overlap = 0.0;
   out.quality_overlap = 0.0;
   out.diversity_bonus = 0.0;
   out.concentration_penalty = 0.0;
   out.max_capital_risk_pct = 0.0;
   out.mean_trade_prob = 0.0;
   out.mean_no_trade_prob = 0.0;
   out.mean_capital_efficiency = 0.0;
   out.mean_portfolio_fit = 0.0;
   out.supervisor_score = 0.0;
   out.score = 0.0;
}

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
   if(now <= 0)
      now = iTime(_Symbol, PERIOD_M1, 0);

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
   if(now <= 0)
      now = iTime(_Symbol, PERIOD_M1, 0);

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

bool FXAI_ReadControlPlaneSnapshotFile(const string file_name,
                                       FXAIControlPlaneSnapshot &out)
{
   FXAI_ResetControlPlaneSnapshot(out);
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
      if(key == "login")
         out.login = (long)StringToInteger(value);
      else if(key == "magic")
         out.magic = (ulong)StringToInteger(value);
      else if(key == "chart_id")
         out.chart_id = (long)StringToInteger(value);
      else if(key == "symbol")
         out.symbol = value;
      else if(key == "bar_time")
         out.bar_time = (datetime)StringToInteger(value);
      else if(key == "direction")
         out.direction = (int)StringToInteger(value);
      else if(key == "signal_intensity")
         out.signal_intensity = StringToDouble(value);
      else if(key == "confidence")
         out.confidence = StringToDouble(value);
      else if(key == "reliability")
         out.reliability = StringToDouble(value);
      else if(key == "trade_gate")
         out.trade_gate = StringToDouble(value);
      else if(key == "hierarchy_score")
         out.hierarchy_score = StringToDouble(value);
      else if(key == "macro_quality")
         out.macro_quality = StringToDouble(value);
      else if(key == "trade_edge_norm")
         out.trade_edge_norm = StringToDouble(value);
      else if(key == "expected_move_norm")
         out.expected_move_norm = StringToDouble(value);
      else if(key == "policy_trade_prob")
         out.policy_trade_prob = StringToDouble(value);
      else if(key == "policy_no_trade_prob")
         out.policy_no_trade_prob = StringToDouble(value);
      else if(key == "policy_enter_prob")
         out.policy_enter_prob = StringToDouble(value);
      else if(key == "policy_exit_prob")
         out.policy_exit_prob = StringToDouble(value);
      else if(key == "policy_size_mult")
         out.policy_size_mult = StringToDouble(value);
      else if(key == "policy_portfolio_fit")
         out.policy_portfolio_fit = StringToDouble(value);
      else if(key == "policy_capital_efficiency")
         out.policy_capital_efficiency = StringToDouble(value);
      else if(key == "gross_exposure_lots")
         out.gross_exposure_lots = StringToDouble(value);
      else if(key == "correlated_exposure_lots")
         out.correlated_exposure_lots = StringToDouble(value);
      else if(key == "directional_cluster_lots")
         out.directional_cluster_lots = StringToDouble(value);
      else if(key == "capital_risk_pct")
         out.capital_risk_pct = StringToDouble(value);
      else if(key == "portfolio_pressure")
         out.portfolio_pressure = StringToDouble(value);
   }
   FileClose(handle);

   out.signal_intensity = FXAI_Clamp(out.signal_intensity, 0.0, 4.0);
   out.confidence = FXAI_Clamp(out.confidence, 0.0, 1.0);
   out.reliability = FXAI_Clamp(out.reliability, 0.0, 1.0);
   out.trade_gate = FXAI_Clamp(out.trade_gate, 0.0, 1.0);
   out.hierarchy_score = FXAI_Clamp(out.hierarchy_score, 0.0, 1.0);
   out.macro_quality = FXAI_Clamp(out.macro_quality, 0.0, 1.0);
   out.trade_edge_norm = FXAI_Clamp(out.trade_edge_norm, -1.0, 1.0);
   out.expected_move_norm = FXAI_Clamp(out.expected_move_norm, 0.0, 4.0);
   out.policy_trade_prob = FXAI_Clamp(out.policy_trade_prob, 0.0, 1.0);
   out.policy_no_trade_prob = FXAI_Clamp(out.policy_no_trade_prob, 0.0, 1.0);
   out.policy_enter_prob = FXAI_Clamp(out.policy_enter_prob, 0.0, 1.0);
   out.policy_exit_prob = FXAI_Clamp(out.policy_exit_prob, 0.0, 1.0);
   out.policy_size_mult = FXAI_Clamp(out.policy_size_mult, 0.0, 2.0);
   out.policy_portfolio_fit = FXAI_Clamp(out.policy_portfolio_fit, 0.0, 1.0);
   out.policy_capital_efficiency = FXAI_Clamp(out.policy_capital_efficiency, 0.0, 1.0);
   out.gross_exposure_lots = FXAI_Clamp(out.gross_exposure_lots, 0.0, 1000.0);
   out.correlated_exposure_lots = FXAI_Clamp(out.correlated_exposure_lots, 0.0, 1000.0);
   out.directional_cluster_lots = FXAI_Clamp(out.directional_cluster_lots, 0.0, 1000.0);
   out.capital_risk_pct = FXAI_Clamp(out.capital_risk_pct, 0.0, 100.0);
   out.portfolio_pressure = FXAI_Clamp(out.portfolio_pressure, 0.0, 2.0);
   out.valid = (out.login > 0 && out.magic > 0 && StringLen(out.symbol) > 0 && out.bar_time > 0);
   return out.valid;
}

bool FXAI_WriteControlPlaneLocalSnapshot(const string symbol,
                                         const int direction,
                                         const double signal_intensity)
{
   FolderCreate("FXAI", FILE_COMMON);
   FolderCreate(FXAI_CONTROL_PLANE_DIR, FILE_COMMON);

   string file_name = FXAI_ControlPlaneSnapshotFile(symbol);
   int handle = FileOpen(file_name,
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return false;

   datetime bar_time = iTime(symbol, PERIOD_M1, 1);
   if(bar_time <= 0)
      bar_time = TimeCurrent();
   if(bar_time <= 0)
      bar_time = TimeTradeServer();

   double mm = MathMax(g_ai_last_min_move_points, 0.10);
   double edge_norm = FXAI_Clamp(g_ai_last_trade_edge_points / mm, -4.0, 4.0) / 4.0;
   double move_norm = FXAI_Clamp(g_ai_last_expected_move_points / mm, 0.0, 8.0) / 2.0;
   double gross_exposure = FXAI_ManagedExposureLots("");
   double corr_exposure = FXAI_ManagedCorrelatedExposureLots(symbol);
   double dir_exposure = FXAI_ManagedDirectionalClusterLots(symbol, direction);
   double capital_risk_pct = FXAI_ControlPlaneCapitalRiskPct(symbol, direction, 0.0);
   double intensity = FXAI_Clamp(signal_intensity, 0.0, 4.0);
   if(direction < 0)
      intensity *= 0.35;

   FileWriteString(handle, "login\t" + IntegerToString((int)AccountInfoInteger(ACCOUNT_LOGIN)) + "\r\n");
   FileWriteString(handle, "magic\t" + IntegerToString((int)TradeMagic) + "\r\n");
   FileWriteString(handle, "chart_id\t" + IntegerToString((int)((long)ChartID() % 2147483647)) + "\r\n");
   FileWriteString(handle, "symbol\t" + symbol + "\r\n");
   FileWriteString(handle, "bar_time\t" + IntegerToString((int)bar_time) + "\r\n");
   FileWriteString(handle, "direction\t" + IntegerToString(direction) + "\r\n");
   FileWriteString(handle, "signal_intensity\t" + DoubleToString(intensity, 6) + "\r\n");
   FileWriteString(handle, "confidence\t" + DoubleToString(FXAI_Clamp(g_ai_last_confidence, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "reliability\t" + DoubleToString(FXAI_Clamp(g_ai_last_reliability, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "trade_gate\t" + DoubleToString(FXAI_Clamp(g_ai_last_trade_gate, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "hierarchy_score\t" + DoubleToString(FXAI_Clamp(g_ai_last_hierarchy_score, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "macro_quality\t" + DoubleToString(FXAI_Clamp(g_ai_last_macro_state_quality, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "trade_edge_norm\t" + DoubleToString(edge_norm, 6) + "\r\n");
   FileWriteString(handle, "expected_move_norm\t" + DoubleToString(move_norm, 6) + "\r\n");
   FileWriteString(handle, "policy_trade_prob\t" + DoubleToString(FXAI_Clamp(g_policy_last_trade_prob, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "policy_no_trade_prob\t" + DoubleToString(FXAI_Clamp(g_policy_last_no_trade_prob, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "policy_enter_prob\t" + DoubleToString(FXAI_Clamp(g_policy_last_enter_prob, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "policy_exit_prob\t" + DoubleToString(FXAI_Clamp(g_policy_last_exit_prob, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "policy_size_mult\t" + DoubleToString(FXAI_Clamp(g_policy_last_size_mult, 0.0, 2.0), 6) + "\r\n");
   FileWriteString(handle, "policy_portfolio_fit\t" + DoubleToString(FXAI_Clamp(g_policy_last_portfolio_fit, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "policy_capital_efficiency\t" + DoubleToString(FXAI_Clamp(g_policy_last_capital_efficiency, 0.0, 1.0), 6) + "\r\n");
   FileWriteString(handle, "gross_exposure_lots\t" + DoubleToString(gross_exposure, 6) + "\r\n");
   FileWriteString(handle, "correlated_exposure_lots\t" + DoubleToString(corr_exposure, 6) + "\r\n");
   FileWriteString(handle, "directional_cluster_lots\t" + DoubleToString(dir_exposure, 6) + "\r\n");
   FileWriteString(handle, "capital_risk_pct\t" + DoubleToString(capital_risk_pct, 6) + "\r\n");
   FileWriteString(handle, "portfolio_pressure\t" + DoubleToString(FXAI_Clamp(g_ai_last_portfolio_pressure, 0.0, 2.0), 6) + "\r\n");
   FileClose(handle);

   g_control_plane_last_symbol = symbol;
   g_control_plane_last_bar_time = bar_time;
   return true;
}

void FXAI_RemoveControlPlaneLocalSnapshot(const string symbol)
{
   string file_name = FXAI_ControlPlaneSnapshotFile(symbol);
   FileDelete(file_name, FILE_COMMON);
}

void FXAI_ReadControlPlaneAggregate(const string symbol,
                                    const int direction,
                                    FXAIControlPlaneAggregate &out)
{
   FXAI_ResetControlPlaneAggregate(out);

   string filter = FXAI_CONTROL_PLANE_DIR + "\\cp_*.tsv";
   string found = "";
   long search = FileFindFirst(filter, found, FILE_COMMON);
   if(search == INVALID_HANDLE)
      return;

   datetime now = TimeCurrent();
   if(now <= 0)
      now = TimeTradeServer();
   if(now <= 0)
      now = iTime(symbol, PERIOD_M1, 1);

   long login = (long)AccountInfoInteger(ACCOUNT_LOGIN);
   ulong magic = TradeMagic;
   long self_chart = (long)ChartID();
   int unique_symbols = 0;
   string seen_symbols[64];
   double mean_trade_prob_sum = 0.0;
   double mean_no_trade_prob_sum = 0.0;
   double mean_capital_efficiency_sum = 0.0;
   double mean_portfolio_fit_sum = 0.0;
   double max_symbol_intensity = 0.0;

   bool cont = true;
   while(cont)
   {
      string file_name = FXAI_CONTROL_PLANE_DIR + "\\" + found;
      FXAIControlPlaneSnapshot snap;
      if(FXAI_ReadControlPlaneSnapshotFile(file_name, snap))
      {
         bool stale = (now > 0 && snap.bar_time > 0 && (now - snap.bar_time) > FXAI_CONTROL_PLANE_TTL_SEC);
         if(stale)
         {
            FileDelete(file_name, FILE_COMMON);
         }
         else if(snap.login == login &&
                 snap.magic == magic &&
                 snap.chart_id != self_chart)
         {
            double corr = FXAI_CorrelationExposureWeight(symbol, snap.symbol);
            if(corr > 0.0)
            {
               out.peer_count++;
               out.gross_intensity += snap.signal_intensity;
               out.correlated_intensity += corr * snap.signal_intensity;
               mean_trade_prob_sum += snap.policy_trade_prob;
               mean_no_trade_prob_sum += snap.policy_no_trade_prob;
               mean_capital_efficiency_sum += snap.policy_capital_efficiency;
               mean_portfolio_fit_sum += snap.policy_portfolio_fit;
               out.max_capital_risk_pct = MathMax(out.max_capital_risk_pct, snap.capital_risk_pct);
               if(snap.signal_intensity > max_symbol_intensity)
                  max_symbol_intensity = snap.signal_intensity;
               if(direction == 0 || direction == 1)
               {
                  double align = FXAI_DirectionalClusterAlignment(symbol, direction, snap.symbol, snap.direction);
                  if(align > 0.0)
                  {
                     out.directional_intensity += align * snap.signal_intensity;
                     out.macro_overlap += align * snap.signal_intensity * snap.macro_quality;
                     out.quality_overlap += align * snap.signal_intensity *
                                            (0.55 * snap.confidence + 0.45 * snap.reliability);
                  }
               }

               bool known_symbol = false;
               for(int s=0; s<unique_symbols; s++)
               {
                  if(seen_symbols[s] == snap.symbol)
                  {
                     known_symbol = true;
                     break;
                  }
               }
               if(!known_symbol && unique_symbols < 64)
               {
                  seen_symbols[unique_symbols] = snap.symbol;
                  unique_symbols++;
               }
            }
         }
      }

      cont = FileFindNext(search, found);
   }
   FileFindClose(search);

   out.diversity_bonus = FXAI_Clamp((double)unique_symbols / 6.0, 0.0, 1.0);
   if(out.peer_count > 0)
   {
      out.mean_trade_prob = FXAI_Clamp(mean_trade_prob_sum / (double)out.peer_count, 0.0, 1.0);
      out.mean_no_trade_prob = FXAI_Clamp(mean_no_trade_prob_sum / (double)out.peer_count, 0.0, 1.0);
      out.mean_capital_efficiency = FXAI_Clamp(mean_capital_efficiency_sum / (double)out.peer_count, 0.0, 1.0);
      out.mean_portfolio_fit = FXAI_Clamp(mean_portfolio_fit_sum / (double)out.peer_count, 0.0, 1.0);
   }
   out.concentration_penalty = FXAI_Clamp(max_symbol_intensity / MathMax(out.gross_intensity, 1e-6), 0.0, 1.0);
   out.score = FXAI_Clamp(0.34 * FXAI_Clamp(out.gross_intensity / 2.0, 0.0, 1.5) +
                          0.30 * FXAI_Clamp(out.correlated_intensity / 1.6, 0.0, 1.5) +
                          0.22 * FXAI_Clamp(out.directional_intensity / 1.4, 0.0, 1.5) +
                          0.08 * FXAI_Clamp(out.macro_overlap / 1.2, 0.0, 1.0) +
                          0.08 * FXAI_Clamp(out.quality_overlap / 1.2, 0.0, 1.0) -
                          0.10 * out.diversity_bonus +
                          0.10 * out.concentration_penalty +
                          0.08 * FXAI_Clamp(out.max_capital_risk_pct / 2.0, 0.0, 1.0),
                          0.0,
                          2.0);
}

double FXAI_PortfolioSupervisorScore(const string symbol,
                                     const int direction,
                                     const FXAIControlPlaneAggregate &cp,
                                     FXAIPortfolioSupervisorProfile &profile_out)
{
   FXAI_LoadPortfolioSupervisorProfile(profile_out, false);
   double score = 0.0;
   score += 0.30 * FXAI_Clamp(cp.score, 0.0, 2.0);
   score += 0.14 * FXAI_Clamp(cp.max_capital_risk_pct / MathMax(profile_out.capital_risk_cap_pct, 0.10), 0.0, 2.0);
   score += 0.12 * FXAI_Clamp(cp.macro_overlap / MathMax(profile_out.macro_overlap_cap, 0.10), 0.0, 2.0);
   score += 0.10 * FXAI_Clamp(cp.concentration_penalty / MathMax(profile_out.concentration_cap, 0.10), 0.0, 2.0);
   score += 0.12 * FXAI_Clamp(cp.mean_no_trade_prob / MathMax(profile_out.policy_no_trade_ceiling, 0.10), 0.0, 2.0);
   score -= 0.10 * FXAI_Clamp(cp.mean_capital_efficiency, 0.0, 1.0);
   score -= 0.08 * FXAI_Clamp(cp.mean_portfolio_fit, 0.0, 1.0);
   score -= 0.06 * FXAI_Clamp(cp.diversity_bonus, 0.0, 1.0);
   if(direction < 0)
      score *= 0.45;
   score = FXAI_Clamp(score, 0.0, 3.0);
   g_portfolio_supervisor_last_score = score;
   g_portfolio_supervisor_last_capital_risk_pct = cp.max_capital_risk_pct;
   return score;
}

#endif // __FXAI_RUNTIME_CONTROL_PLANE_MQH__
