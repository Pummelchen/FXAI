#ifndef __FXAI_RUNTIME_PROB_CALIBRATION_STAGE_MQH__
#define __FXAI_RUNTIME_PROB_CALIBRATION_STAGE_MQH__

#define FXAI_PROB_CAL_MAX_REASONS 10
#define FXAI_PROB_CAL_MAX_BUCKETS 4
#define FXAI_PROB_CAL_MAX_TIERS 96

struct FXAIProbCalibrationConfig
{
   bool   ready;
   bool   enabled;
   bool   allow_abstain_flag;
   double neutral_blend_gain;
   double skip_uncertainty_gain;
   double skip_calibration_credit;
   double skip_floor;
   double skip_cap;
   double base_uncertainty_score;
   int    support_soft_floor;
   int    support_hard_floor;
   int    memory_stale_after_hours;
   double min_calibration_quality;
   double max_uncertainty_score;
   double signal_zero_band;
   double edge_floor_mult;
   double trade_edge_floor_points;
   double soft_prob_scale;
   double soft_skip_bias;
   double soft_move_mean_scale;
   double soft_move_q25_scale;
   double soft_move_q50_scale;
   double soft_move_q75_scale;
   double soft_confidence_cap;
   double uncertainty_support_penalty;
   double uncertainty_quality_penalty;
   double uncertainty_disagreement_penalty;
   double uncertainty_distribution_width_penalty;
   double uncertainty_news_penalty;
   double uncertainty_rates_penalty;
   double uncertainty_micro_penalty;
   double uncertainty_dynamic_abstain_penalty;
   double uncertainty_adaptive_abstain_penalty;
   double uncertainty_stale_context_penalty;
   double risk_news_block_mult;
   double risk_rates_block_mult;
   double risk_micro_block_mult;
   double risk_caution_posture_mult;
   double risk_abstain_posture_mult;
   double risk_block_posture_mult;
   double risk_fill_mult;
   double risk_path_mult;
   int    bucket_count;
   string bucket_hierarchy[FXAI_PROB_CAL_MAX_BUCKETS];
};

struct FXAIProbCalibrationTier
{
   bool   ready;
   string kind;
   string symbol;
   string session;
   string regime;
   int    support;
   double prob_scale;
   double prob_bias;
   double skip_bias;
   double move_mean_scale;
   double move_q25_scale;
   double move_q50_scale;
   double move_q75_scale;
   double calibration_quality;
   double uncertainty_mult;
   double confidence_cap;
};

struct FXAIProbCalibrationRuntimeState
{
   bool     ready;
   bool     fallback_used;
   bool     calibration_stale;
   bool     input_stale;
   bool     news_risk_block;
   bool     rates_risk_block;
   bool     microstructure_stress;
   bool     support_usable;
   datetime generated_at;
   string   symbol;
   string   method;
   string   session_label;
   string   regime_label;
   string   selected_tier_kind;
   string   selected_tier_key;
   int      selected_support;
   double   selected_quality;
   double   raw_buy_prob;
   double   raw_sell_prob;
   double   raw_skip_prob;
   double   raw_score;
   string   raw_action;
   double   calibrated_buy_prob;
   double   calibrated_sell_prob;
   double   calibrated_skip_prob;
   double   calibrated_confidence;
   double   expected_move_mean_points;
   double   expected_move_q25_points;
   double   expected_move_q50_points;
   double   expected_move_q75_points;
   double   spread_cost_points;
   double   slippage_cost_points;
   double   uncertainty_score;
   double   uncertainty_penalty_points;
   double   risk_penalty_points;
   double   expected_gross_edge_points;
   double   edge_after_costs_points;
   string   final_action;
   bool     abstain;
   int      reason_count;
   string   reason_codes[FXAI_PROB_CAL_MAX_REASONS];
};

FXAIProbCalibrationConfig g_prob_cal_cfg_cache;
datetime g_prob_cal_cfg_cache_loaded_at = 0;
FXAIProbCalibrationTier g_prob_cal_tiers[];
int      g_prob_cal_tier_count = 0;
datetime g_prob_cal_memory_loaded_at = 0;
datetime g_prob_cal_memory_generated_at = 0;
string   g_prob_cal_memory_method = "LOGISTIC_AFFINE";

string FXAI_ProbCalibrationConfigFile(void)
{
   return "FXAI\\Runtime\\prob_calibration_config.tsv";
}

string FXAI_ProbCalibrationMemoryFile(void)
{
   return "FXAI\\Runtime\\prob_calibration_memory.tsv";
}

string FXAI_ProbCalibrationRuntimeStateFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_prob_calibration_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_ProbCalibrationRuntimeHistoryFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_prob_calibration_history_" + FXAI_ControlPlaneSafeToken(symbol) + ".ndjson";
}

string FXAI_ProbCalibrationISO8601(const datetime value)
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

string FXAI_ProbCalibrationJSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   return out;
}

void FXAI_ResetProbCalibrationConfig(FXAIProbCalibrationConfig &out)
{
   out.ready = true;
   out.enabled = true;
   out.allow_abstain_flag = true;
   out.neutral_blend_gain = 0.65;
   out.skip_uncertainty_gain = 0.12;
   out.skip_calibration_credit = 0.05;
   out.skip_floor = 0.02;
   out.skip_cap = 0.96;
   out.base_uncertainty_score = 0.18;
   out.support_soft_floor = 64;
   out.support_hard_floor = 16;
   out.memory_stale_after_hours = 96;
   out.min_calibration_quality = 0.44;
   out.max_uncertainty_score = 0.92;
   out.signal_zero_band = 0.035;
   out.edge_floor_mult = 0.08;
   out.trade_edge_floor_points = 0.05;
   out.soft_prob_scale = 1.60;
   out.soft_skip_bias = 0.08;
   out.soft_move_mean_scale = 0.78;
   out.soft_move_q25_scale = 0.60;
   out.soft_move_q50_scale = 0.72;
   out.soft_move_q75_scale = 0.88;
   out.soft_confidence_cap = 0.58;
   out.uncertainty_support_penalty = 0.34;
   out.uncertainty_quality_penalty = 0.28;
   out.uncertainty_disagreement_penalty = 0.26;
   out.uncertainty_distribution_width_penalty = 0.22;
   out.uncertainty_news_penalty = 0.18;
   out.uncertainty_rates_penalty = 0.14;
   out.uncertainty_micro_penalty = 0.24;
   out.uncertainty_dynamic_abstain_penalty = 0.20;
   out.uncertainty_adaptive_abstain_penalty = 0.22;
   out.uncertainty_stale_context_penalty = 0.16;
   out.risk_news_block_mult = 0.32;
   out.risk_rates_block_mult = 0.24;
   out.risk_micro_block_mult = 0.36;
   out.risk_caution_posture_mult = 0.14;
   out.risk_abstain_posture_mult = 0.24;
   out.risk_block_posture_mult = 0.42;
   out.risk_fill_mult = 0.20;
   out.risk_path_mult = 0.16;
   out.bucket_count = 4;
   out.bucket_hierarchy[0] = "PAIR_SESSION_REGIME";
   out.bucket_hierarchy[1] = "PAIR_REGIME";
   out.bucket_hierarchy[2] = "REGIME";
   out.bucket_hierarchy[3] = "GLOBAL";
}

void FXAI_ResetProbCalibrationTier(FXAIProbCalibrationTier &out)
{
   out.ready = false;
   out.kind = "GLOBAL";
   out.symbol = "*";
   out.session = "*";
   out.regime = "*";
   out.support = 0;
   out.prob_scale = 1.60;
   out.prob_bias = 0.0;
   out.skip_bias = 0.08;
   out.move_mean_scale = 0.78;
   out.move_q25_scale = 0.60;
   out.move_q50_scale = 0.72;
   out.move_q75_scale = 0.88;
   out.calibration_quality = 0.34;
   out.uncertainty_mult = 1.30;
   out.confidence_cap = 0.58;
}

void FXAI_ResetProbCalibrationRuntimeState(FXAIProbCalibrationRuntimeState &out)
{
   out.ready = false;
   out.fallback_used = false;
   out.calibration_stale = true;
   out.input_stale = true;
   out.news_risk_block = false;
   out.rates_risk_block = false;
   out.microstructure_stress = false;
   out.support_usable = false;
   out.generated_at = 0;
   out.symbol = "";
   out.method = "LOGISTIC_AFFINE";
   out.session_label = "UNKNOWN";
   out.regime_label = "UNKNOWN";
   out.selected_tier_kind = "GLOBAL";
   out.selected_tier_key = "GLOBAL|*|*|*";
   out.selected_support = 0;
   out.selected_quality = 0.0;
   out.raw_buy_prob = 0.0;
   out.raw_sell_prob = 0.0;
   out.raw_skip_prob = 1.0;
   out.raw_score = 0.0;
   out.raw_action = "SKIP";
   out.calibrated_buy_prob = 0.0;
   out.calibrated_sell_prob = 0.0;
   out.calibrated_skip_prob = 1.0;
   out.calibrated_confidence = 0.0;
   out.expected_move_mean_points = 0.0;
   out.expected_move_q25_points = 0.0;
   out.expected_move_q50_points = 0.0;
   out.expected_move_q75_points = 0.0;
   out.spread_cost_points = 0.0;
   out.slippage_cost_points = 0.0;
   out.uncertainty_score = 0.0;
   out.uncertainty_penalty_points = 0.0;
   out.risk_penalty_points = 0.0;
   out.expected_gross_edge_points = 0.0;
   out.edge_after_costs_points = 0.0;
   out.final_action = "SKIP";
   out.abstain = false;
   out.reason_count = 0;
   for(int i=0; i<FXAI_PROB_CAL_MAX_REASONS; i++)
      out.reason_codes[i] = "";
}

double FXAI_ProbCalibrationSigmoid(const double value)
{
   if(value >= 0.0)
   {
      double exp_neg = MathExp(-value);
      return 1.0 / (1.0 + exp_neg);
   }
   double exp_pos = MathExp(value);
   return exp_pos / (1.0 + exp_pos);
}

double FXAI_ProbCalibrationLogit(const double probability)
{
   double p = FXAI_Clamp(probability, 1e-6, 1.0 - 1e-6);
   return MathLog(p / (1.0 - p));
}

void FXAI_ProbCalibrationAppendReason(FXAIProbCalibrationRuntimeState &state,
                                      const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reason_codes[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_PROB_CAL_MAX_REASONS)
      return;
   state.reason_codes[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_ProbCalibrationReasonsCSV(const FXAIProbCalibrationRuntimeState &state)
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

string FXAI_ProbCalibrationTierKey(const FXAIProbCalibrationTier &tier)
{
   return tier.kind + "|" + tier.symbol + "|" + tier.session + "|" + tier.regime;
}

string FXAI_ProbCalibrationDecisionLabel(const int decision)
{
   if(decision == 1)
      return "BUY";
   if(decision == 0)
      return "SELL";
   return "SKIP";
}

string FXAI_ProbCalibrationSessionLabel(const FXAINewsPulsePairState &news_state,
                                        const FXAIMicrostructurePairState &micro_state,
                                        const FXAIAdaptiveRegimeState &adaptive_state)
{
   if(adaptive_state.valid && StringLen(adaptive_state.session_label) > 0)
      return adaptive_state.session_label;
   if(micro_state.ready && micro_state.available && StringLen(micro_state.session_tag) > 0)
      return micro_state.session_tag;
   if(news_state.ready && news_state.available && StringLen(news_state.session_profile) > 0)
      return news_state.session_profile;
   return "UNKNOWN";
}

string FXAI_ProbCalibrationRegimeLabel(const FXAIDynamicEnsembleRuntimeState &dynamic_state,
                                       const FXAIAdaptiveRegimeState &adaptive_state,
                                       const int regime_id)
{
   int fallback_regime_id = regime_id;
   if(fallback_regime_id < 0)
      fallback_regime_id = 0;
   if(fallback_regime_id >= FXAI_ADAPTIVE_ROUTER_REGIME_COUNT)
      fallback_regime_id = FXAI_ADAPTIVE_ROUTER_REGIME_COUNT - 1;
   if(adaptive_state.valid && StringLen(adaptive_state.top_label) > 0)
      return adaptive_state.top_label;
   if(dynamic_state.ready && StringLen(dynamic_state.top_regime) > 0)
      return dynamic_state.top_regime;
   return FXAI_AdaptiveRouterRegimeLabel(fallback_regime_id);
}

void FXAI_ProbCalibrationLoadConfig(FXAIProbCalibrationConfig &out)
{
   FXAI_ResetProbCalibrationConfig(out);
   int handle = FileOpen(FXAI_ProbCalibrationConfigFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
      return;

   out.bucket_count = 0;
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
      double dv = StringToDouble(value);
      if(key == "enabled")
         out.enabled = (StringToInteger(value) != 0);
      else if(key == "allow_abstain_flag")
         out.allow_abstain_flag = (StringToInteger(value) != 0);
      else if(key == "neutral_blend_gain")
         out.neutral_blend_gain = dv;
      else if(key == "skip_uncertainty_gain")
         out.skip_uncertainty_gain = dv;
      else if(key == "skip_calibration_credit")
         out.skip_calibration_credit = dv;
      else if(key == "skip_floor")
         out.skip_floor = dv;
      else if(key == "skip_cap")
         out.skip_cap = dv;
      else if(key == "base_uncertainty_score")
         out.base_uncertainty_score = dv;
      else if(key == "support_soft_floor")
         out.support_soft_floor = (int)MathRound(dv);
      else if(key == "support_hard_floor")
         out.support_hard_floor = (int)MathRound(dv);
      else if(key == "memory_stale_after_hours")
         out.memory_stale_after_hours = (int)MathRound(dv);
      else if(key == "min_calibration_quality")
         out.min_calibration_quality = dv;
      else if(key == "max_uncertainty_score")
         out.max_uncertainty_score = dv;
      else if(key == "signal_zero_band")
         out.signal_zero_band = dv;
      else if(key == "edge_floor_mult")
         out.edge_floor_mult = dv;
      else if(key == "trade_edge_floor_points")
         out.trade_edge_floor_points = dv;
      else if(key == "soft_prob_scale")
         out.soft_prob_scale = dv;
      else if(key == "soft_skip_bias")
         out.soft_skip_bias = dv;
      else if(key == "soft_move_mean_scale")
         out.soft_move_mean_scale = dv;
      else if(key == "soft_move_q25_scale")
         out.soft_move_q25_scale = dv;
      else if(key == "soft_move_q50_scale")
         out.soft_move_q50_scale = dv;
      else if(key == "soft_move_q75_scale")
         out.soft_move_q75_scale = dv;
      else if(key == "soft_confidence_cap")
         out.soft_confidence_cap = dv;
      else if(key == "uncertainty_support")
         out.uncertainty_support_penalty = dv;
      else if(key == "uncertainty_quality")
         out.uncertainty_quality_penalty = dv;
      else if(key == "uncertainty_disagreement")
         out.uncertainty_disagreement_penalty = dv;
      else if(key == "uncertainty_distribution_width")
         out.uncertainty_distribution_width_penalty = dv;
      else if(key == "uncertainty_news")
         out.uncertainty_news_penalty = dv;
      else if(key == "uncertainty_rates")
         out.uncertainty_rates_penalty = dv;
      else if(key == "uncertainty_micro")
         out.uncertainty_micro_penalty = dv;
      else if(key == "uncertainty_dynamic_abstain")
         out.uncertainty_dynamic_abstain_penalty = dv;
      else if(key == "uncertainty_adaptive_abstain")
         out.uncertainty_adaptive_abstain_penalty = dv;
      else if(key == "uncertainty_stale_context")
         out.uncertainty_stale_context_penalty = dv;
      else if(key == "risk_news_block_mult")
         out.risk_news_block_mult = dv;
      else if(key == "risk_rates_block_mult")
         out.risk_rates_block_mult = dv;
      else if(key == "risk_micro_block_mult")
         out.risk_micro_block_mult = dv;
      else if(key == "risk_caution_posture_mult")
         out.risk_caution_posture_mult = dv;
      else if(key == "risk_abstain_posture_mult")
         out.risk_abstain_posture_mult = dv;
      else if(key == "risk_block_posture_mult")
         out.risk_block_posture_mult = dv;
      else if(key == "risk_fill_mult")
         out.risk_fill_mult = dv;
      else if(key == "risk_path_mult")
         out.risk_path_mult = dv;
      else if(key == "bucket_hierarchy")
      {
         if(out.bucket_count < FXAI_PROB_CAL_MAX_BUCKETS)
         {
            out.bucket_hierarchy[out.bucket_count] = value;
            out.bucket_count++;
         }
      }
   }
   FileClose(handle);
   if(out.bucket_count <= 0)
   {
      out.bucket_count = 4;
      out.bucket_hierarchy[0] = "PAIR_SESSION_REGIME";
      out.bucket_hierarchy[1] = "PAIR_REGIME";
      out.bucket_hierarchy[2] = "REGIME";
      out.bucket_hierarchy[3] = "GLOBAL";
   }
}

void FXAI_ProbCalibrationEnsureConfigLoaded(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(g_prob_cal_cfg_cache_loaded_at > 0 && now_time > 0 && (now_time - g_prob_cal_cfg_cache_loaded_at) < 60)
      return;
   FXAI_ProbCalibrationLoadConfig(g_prob_cal_cfg_cache);
   g_prob_cal_cfg_cache_loaded_at = now_time;
}

void FXAI_ProbCalibrationEnsureMemoryLoaded(void)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(g_prob_cal_memory_loaded_at > 0 && now_time > 0 && (now_time - g_prob_cal_memory_loaded_at) < 60)
      return;

   ArrayResize(g_prob_cal_tiers, 0);
   g_prob_cal_tier_count = 0;
   g_prob_cal_memory_generated_at = 0;
   g_prob_cal_memory_method = "LOGISTIC_AFFINE";

   int handle = FileOpen(FXAI_ProbCalibrationMemoryFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      g_prob_cal_memory_loaded_at = now_time;
      return;
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
      if(parts[0] == "generated_at_unix")
      {
         g_prob_cal_memory_generated_at = (datetime)StringToInteger(parts[1]);
      }
      else if(parts[0] == "default_method")
      {
         g_prob_cal_memory_method = parts[1];
      }
      else if(parts[0] == "tier" && n >= 16)
      {
         int idx = ArraySize(g_prob_cal_tiers);
         if(idx >= FXAI_PROB_CAL_MAX_TIERS)
            continue;
         ArrayResize(g_prob_cal_tiers, idx + 1);
         FXAI_ResetProbCalibrationTier(g_prob_cal_tiers[idx]);
         g_prob_cal_tiers[idx].ready = true;
         g_prob_cal_tiers[idx].kind = parts[1];
         g_prob_cal_tiers[idx].symbol = parts[2];
         g_prob_cal_tiers[idx].session = parts[3];
         g_prob_cal_tiers[idx].regime = parts[4];
         g_prob_cal_tiers[idx].support = (int)StringToInteger(parts[5]);
         g_prob_cal_tiers[idx].prob_scale = StringToDouble(parts[6]);
         g_prob_cal_tiers[idx].prob_bias = StringToDouble(parts[7]);
         g_prob_cal_tiers[idx].skip_bias = StringToDouble(parts[8]);
         g_prob_cal_tiers[idx].move_mean_scale = StringToDouble(parts[9]);
         g_prob_cal_tiers[idx].move_q25_scale = StringToDouble(parts[10]);
         g_prob_cal_tiers[idx].move_q50_scale = StringToDouble(parts[11]);
         g_prob_cal_tiers[idx].move_q75_scale = StringToDouble(parts[12]);
         g_prob_cal_tiers[idx].calibration_quality = StringToDouble(parts[13]);
         g_prob_cal_tiers[idx].uncertainty_mult = StringToDouble(parts[14]);
         g_prob_cal_tiers[idx].confidence_cap = StringToDouble(parts[15]);
      }
   }
   FileClose(handle);
   g_prob_cal_tier_count = ArraySize(g_prob_cal_tiers);
   g_prob_cal_memory_loaded_at = now_time;
}

int FXAI_ProbCalibrationTierHierarchyIndex(const string kind)
{
   for(int i=0; i<g_prob_cal_cfg_cache.bucket_count; i++)
   {
      if(g_prob_cal_cfg_cache.bucket_hierarchy[i] == kind)
         return i;
   }
   return FXAI_PROB_CAL_MAX_BUCKETS;
}

bool FXAI_ProbCalibrationTierMatches(const FXAIProbCalibrationTier &tier,
                                     const string kind,
                                     const string symbol,
                                     const string session,
                                     const string regime)
{
   if(!tier.ready || tier.kind != kind)
      return false;
   if(kind == "PAIR_SESSION_REGIME")
      return tier.symbol == symbol && tier.session == session && tier.regime == regime;
   if(kind == "PAIR_REGIME")
      return tier.symbol == symbol && tier.regime == regime;
   if(kind == "REGIME")
      return tier.regime == regime;
   if(kind == "GLOBAL")
      return true;
   return false;
}

bool FXAI_SelectProbCalibrationTier(const string symbol,
                                    const string session,
                                    const string regime,
                                    FXAIProbCalibrationTier &selected,
                                    bool &fallback_used,
                                    bool &support_usable)
{
   FXAI_ResetProbCalibrationTier(selected);
   fallback_used = false;
   support_usable = false;
   if(g_prob_cal_tier_count <= 0)
      return false;

   for(int h=0; h<g_prob_cal_cfg_cache.bucket_count; h++)
   {
      string kind = g_prob_cal_cfg_cache.bucket_hierarchy[h];
      int preferred_idx = -1;
      int preferred_support = -1;
      double preferred_quality = -1.0;
      int fallback_idx = -1;
      int fallback_support = -1;
      double fallback_quality = -1.0;

      for(int i=0; i<g_prob_cal_tier_count; i++)
      {
         if(!FXAI_ProbCalibrationTierMatches(g_prob_cal_tiers[i], kind, symbol, session, regime))
            continue;

         int support = g_prob_cal_tiers[i].support;
         double quality = g_prob_cal_tiers[i].calibration_quality;
         if(support >= g_prob_cal_cfg_cache.support_soft_floor)
         {
            if(support > preferred_support || (support == preferred_support && quality > preferred_quality))
            {
               preferred_idx = i;
               preferred_support = support;
               preferred_quality = quality;
            }
         }
         if(support >= g_prob_cal_cfg_cache.support_hard_floor)
         {
            if(support > fallback_support || (support == fallback_support && quality > fallback_quality))
            {
               fallback_idx = i;
               fallback_support = support;
               fallback_quality = quality;
            }
         }
      }

      if(preferred_idx >= 0)
      {
         selected = g_prob_cal_tiers[preferred_idx];
         fallback_used = false;
         support_usable = true;
         return true;
      }
      if(fallback_idx >= 0)
      {
         selected = g_prob_cal_tiers[fallback_idx];
         fallback_used = true;
         support_usable = true;
         return true;
      }
   }
   return false;
}

void FXAI_ProbCalibrationBuildFallbackTier(FXAIProbCalibrationTier &tier)
{
   FXAI_ResetProbCalibrationTier(tier);
   tier.ready = true;
   tier.kind = "GLOBAL";
   tier.symbol = "*";
   tier.session = "*";
   tier.regime = "*";
   tier.prob_scale = g_prob_cal_cfg_cache.soft_prob_scale;
   tier.skip_bias = g_prob_cal_cfg_cache.soft_skip_bias;
   tier.move_mean_scale = g_prob_cal_cfg_cache.soft_move_mean_scale;
   tier.move_q25_scale = g_prob_cal_cfg_cache.soft_move_q25_scale;
   tier.move_q50_scale = g_prob_cal_cfg_cache.soft_move_q50_scale;
   tier.move_q75_scale = g_prob_cal_cfg_cache.soft_move_q75_scale;
   tier.confidence_cap = g_prob_cal_cfg_cache.soft_confidence_cap;
}

void FXAI_ProbCalibrationApply(const string symbol,
                               const FXAIExecutionProfile &exec_profile,
                               const FXAINewsPulsePairState &news_state,
                               const FXAIRatesEnginePairState &rates_state,
                               const FXAIMicrostructurePairState &micro_state,
                               const FXAIAdaptiveRegimeState &adaptive_state,
                               const string adaptive_router_posture,
                               const double adaptive_router_abstain_bias,
                               const FXAIDynamicEnsembleRuntimeState &dynamic_state,
                               const FXAIExecutionQualityRuntimeState &execution_quality_state,
                               const double raw_buy_prob,
                               const double raw_sell_prob,
                               const double raw_skip_prob,
                               const double move_mean_points,
                               const double move_q25_points,
                               const double move_q50_points,
                               const double move_q75_points,
                               const double agreement_score,
                               const double min_move_points,
                               const double spread_points,
                               const double commission_points,
                               const double cost_buffer_points,
                               const int horizon_minutes,
                               int &decision,
                               FXAIProbCalibrationRuntimeState &state)
{
   FXAI_ResetProbCalibrationRuntimeState(state);
   FXAI_ProbCalibrationEnsureConfigLoaded();
   FXAI_ProbCalibrationEnsureMemoryLoaded();
   state.ready = true;
   state.generated_at = TimeCurrent();
   if(state.generated_at <= 0)
      state.generated_at = TimeTradeServer();
   state.symbol = symbol;
   state.method = g_prob_cal_memory_method;

   if(!g_prob_cal_cfg_cache.enabled)
      return;

   double rb = FXAI_Clamp(raw_buy_prob, 0.0, 1.0);
   double rs = FXAI_Clamp(raw_sell_prob, 0.0, 1.0);
   double rk = FXAI_Clamp(raw_skip_prob, 0.0, 1.0);
   double total = rb + rs + rk;
   if(total <= 0.0)
   {
      rb = 0.0;
      rs = 0.0;
      rk = 1.0;
      total = 1.0;
   }
   rb /= total;
   rs /= total;
   rk /= total;
   state.raw_buy_prob = rb;
   state.raw_sell_prob = rs;
   state.raw_skip_prob = rk;
   state.raw_score = rb - rs;
   state.raw_action = (rk >= rb && rk >= rs ? "SKIP" : (rb >= rs ? "BUY" : "SELL"));

   state.session_label = FXAI_ProbCalibrationSessionLabel(news_state, micro_state, adaptive_state);
   state.regime_label = FXAI_ProbCalibrationRegimeLabel(dynamic_state, adaptive_state, g_ai_last_regime_id);

   FXAIProbCalibrationTier tier;
   bool tier_found = false;
   tier_found = FXAI_SelectProbCalibrationTier(symbol,
                                               state.session_label,
                                               state.regime_label,
                                               tier,
                                               state.fallback_used,
                                               state.support_usable);
   if(!tier_found)
   {
      FXAI_ProbCalibrationBuildFallbackTier(tier);
      state.fallback_used = true;
      state.support_usable = false;
   }
   state.selected_tier_kind = tier.kind;
   state.selected_tier_key = FXAI_ProbCalibrationTierKey(tier);
   state.selected_support = tier.support;
   state.selected_quality = FXAI_Clamp(tier.calibration_quality, 0.0, 1.0);

   datetime now_time = state.generated_at;
   state.calibration_stale = (g_prob_cal_memory_generated_at <= 0 ||
                              (g_prob_cal_cfg_cache.memory_stale_after_hours > 0 &&
                               (now_time - g_prob_cal_memory_generated_at) > g_prob_cal_cfg_cache.memory_stale_after_hours * 3600));
   bool news_stale = (news_state.ready && news_state.available && news_state.stale);
   bool rates_stale = (rates_state.ready && rates_state.available && rates_state.stale);
   bool micro_stale = (micro_state.ready && micro_state.available && micro_state.stale);
   bool execution_quality_unknown = (ExecutionQualityEnabled && !execution_quality_state.ready);
   bool execution_quality_stale = (execution_quality_state.ready && execution_quality_state.data_stale);
   int stale_context_count = 0;
   if(news_stale) stale_context_count++;
   if(rates_stale) stale_context_count++;
   if(micro_stale) stale_context_count++;
   if(execution_quality_unknown || execution_quality_stale) stale_context_count++;
   state.input_stale = (stale_context_count > 0);
   state.news_risk_block = (news_state.ready && news_state.available &&
                            (news_state.trade_gate == "BLOCK" || news_state.news_risk_score >= 0.84));
   state.rates_risk_block = (rates_state.ready && rates_state.available &&
                             (rates_state.trade_gate == "BLOCK" || rates_state.rates_risk_score >= 0.82 ||
                              rates_state.meeting_path_reprice_now));
   state.microstructure_stress = (micro_state.ready && micro_state.available &&
                                  (micro_state.trade_gate == "BLOCK" ||
                                   micro_state.hostile_execution_score >= 0.82 ||
                                   micro_state.liquidity_stress_score >= 0.84));

   double news_risk = (news_state.ready && news_state.available ? FXAI_Clamp(news_state.news_risk_score, 0.0, 1.0) :
                       (news_stale ? 0.45 : 0.15));
   double rates_risk = (rates_state.ready && rates_state.available ? FXAI_Clamp(rates_state.rates_risk_score, 0.0, 1.0) :
                        (rates_stale ? 0.35 : 0.12));
   double micro_risk = (micro_state.ready && micro_state.available ? FXAI_Clamp(MathMax(micro_state.hostile_execution_score,
                                                                                         micro_state.liquidity_stress_score),
                                                                                 0.0,
                                                                                 1.0) :
                        (micro_stale ? 0.42 : 0.14));
   double dynamic_abstain = (dynamic_state.ready ? FXAI_Clamp(dynamic_state.abstain_bias, 0.0, 1.0) : 0.0);
   double adaptive_abstain = FXAI_Clamp(adaptive_router_abstain_bias, 0.0, 1.0);
   double agree = FXAI_Clamp(agreement_score, 0.0, 1.0);

   double distribution_width = MathMax(move_q75_points - move_q25_points, 0.0);
   double distribution_ratio = FXAI_Clamp(distribution_width / MathMax(MathMax(move_mean_points, min_move_points), 0.25), 0.0, 3.0) / 3.0;
   double support_shortfall = FXAI_Clamp((double)(g_prob_cal_cfg_cache.support_soft_floor - tier.support) /
                                         (double)MathMax(g_prob_cal_cfg_cache.support_soft_floor, 1),
                                         0.0,
                                         1.0);
   double quality_shortfall = FXAI_Clamp(g_prob_cal_cfg_cache.min_calibration_quality - tier.calibration_quality, 0.0, 1.0);
   double uncertainty_score = g_prob_cal_cfg_cache.base_uncertainty_score +
                              g_prob_cal_cfg_cache.uncertainty_support_penalty * support_shortfall +
                              g_prob_cal_cfg_cache.uncertainty_quality_penalty * quality_shortfall +
                              g_prob_cal_cfg_cache.uncertainty_disagreement_penalty * (1.0 - agree) +
                              g_prob_cal_cfg_cache.uncertainty_distribution_width_penalty * distribution_ratio +
                              g_prob_cal_cfg_cache.uncertainty_news_penalty * news_risk +
                              g_prob_cal_cfg_cache.uncertainty_rates_penalty * rates_risk +
                              g_prob_cal_cfg_cache.uncertainty_micro_penalty * micro_risk +
                              g_prob_cal_cfg_cache.uncertainty_dynamic_abstain_penalty * dynamic_abstain +
                              g_prob_cal_cfg_cache.uncertainty_adaptive_abstain_penalty * adaptive_abstain +
                              g_prob_cal_cfg_cache.uncertainty_stale_context_penalty * FXAI_Clamp((double)stale_context_count / 3.0, 0.0, 1.0);
   uncertainty_score *= FXAI_Clamp(tier.uncertainty_mult, 0.40, 2.50);
   state.uncertainty_score = uncertainty_score;

   double directional_mass = MathMax(rb + rs, 1e-6);
   double directional_share = rb / directional_mass;
   double dir_buy = FXAI_ProbCalibrationSigmoid(tier.prob_bias + tier.prob_scale * FXAI_ProbCalibrationLogit(directional_share));
   double neutral_blend = FXAI_Clamp(g_prob_cal_cfg_cache.neutral_blend_gain * (1.0 - FXAI_Clamp(tier.calibration_quality, 0.0, 1.0)),
                                     0.0,
                                     0.85);
   dir_buy = neutral_blend * 0.5 + (1.0 - neutral_blend) * dir_buy;
   double dir_distance = dir_buy - 0.5;
   double max_distance = MathMax(FXAI_Clamp(tier.confidence_cap, 0.50, 0.95) - 0.5, 0.0);
   dir_buy = 0.5 + FXAI_Clamp(dir_distance, -max_distance, max_distance);

   double cal_skip = FXAI_Clamp(rk +
                                tier.skip_bias +
                                g_prob_cal_cfg_cache.skip_uncertainty_gain * uncertainty_score -
                                g_prob_cal_cfg_cache.skip_calibration_credit * FXAI_Clamp(tier.calibration_quality, 0.0, 1.0),
                                g_prob_cal_cfg_cache.skip_floor,
                                g_prob_cal_cfg_cache.skip_cap);
   double cal_dir_mass = MathMax(1.0 - cal_skip, 1e-6);
   state.calibrated_buy_prob = FXAI_Clamp(cal_dir_mass * dir_buy, 0.0, 1.0);
   state.calibrated_sell_prob = FXAI_Clamp(cal_dir_mass * (1.0 - dir_buy), 0.0, 1.0);
   state.calibrated_skip_prob = FXAI_Clamp(1.0 - state.calibrated_buy_prob - state.calibrated_sell_prob, 0.0, 1.0);
   state.calibrated_confidence = MathMax(state.calibrated_buy_prob, state.calibrated_sell_prob);

   double uncertainty_mean_mult = FXAI_Clamp(1.0 - 0.18 * uncertainty_score, 0.35, 1.0);
   double uncertainty_q25_mult = FXAI_Clamp(1.0 - 0.24 * uncertainty_score, 0.20, 1.0);
   double uncertainty_q50_mult = FXAI_Clamp(1.0 - 0.16 * uncertainty_score, 0.25, 1.0);
   double uncertainty_q75_mult = FXAI_Clamp(1.0 - 0.10 * uncertainty_score, 0.35, 1.0);
   state.expected_move_q25_points = MathMax(move_q25_points * tier.move_q25_scale * uncertainty_q25_mult, 0.0);
   state.expected_move_q50_points = MathMax(move_q50_points * tier.move_q50_scale * uncertainty_q50_mult,
                                            state.expected_move_q25_points);
   state.expected_move_q75_points = MathMax(move_q75_points * tier.move_q75_scale * uncertainty_q75_mult,
                                            state.expected_move_q50_points);
   state.expected_move_mean_points = MathMax(move_mean_points * tier.move_mean_scale * uncertainty_mean_mult,
                                             state.expected_move_q50_points);

   double base_spread_cost = MathMax(spread_points, 0.0) +
                             MathMax(commission_points, 0.0) +
                             MathMax(cost_buffer_points, 0.0) +
                             MathMax(exec_profile.cost_buffer_points, 0.0);
   double spread_stress = (micro_state.ready && micro_state.available ? FXAI_Clamp(MathMax(micro_state.spread_zscore_60s, 0.0), 0.0, 4.0) : 0.0);
   int path_flags = 0;
   if(state.news_risk_block || state.rates_risk_block || state.microstructure_stress)
      path_flags |= 4;
   if(g_ai_last_path_risk >= 0.72 || (micro_state.ready && micro_state.available && micro_state.sweep_and_reject_flag_60s))
      path_flags |= 1;
   if(g_ai_last_fill_risk >= 0.72 || (micro_state.ready && micro_state.available && micro_state.handoff_flag))
      path_flags |= 8;

   double spread_cost = base_spread_cost;
   double slippage_cost = FXAI_ExecutionSlippagePoints(exec_profile,
                                                       base_spread_cost,
                                                       horizon_minutes,
                                                       spread_stress,
                                                       path_flags);
   bool execution_quality_usable = (execution_quality_state.ready &&
                                    !execution_quality_state.data_stale &&
                                    execution_quality_state.spread_expected_points >= 0.0);
   if(execution_quality_usable)
   {
      spread_cost = MathMax(MathMax(execution_quality_state.spread_expected_points, MathMax(spread_points, 0.0)), 0.0) +
                    MathMax(commission_points, 0.0) +
                    MathMax(cost_buffer_points, 0.0) +
                    MathMax(exec_profile.cost_buffer_points, 0.0);
      slippage_cost = MathMax(slippage_cost,
                              MathMax(execution_quality_state.expected_slippage_points, 0.0));
   }
   state.spread_cost_points = spread_cost;
   state.slippage_cost_points = slippage_cost;
   state.uncertainty_penalty_points = MathMax(min_move_points, 0.25) * uncertainty_score;
   double risk_penalty = FXAI_ExecutionFillPenaltyPoints(exec_profile,
                                                         spread_cost,
                                                         spread_stress,
                                                         path_flags) +
                         MathMax(min_move_points, 0.25) *
                         (g_prob_cal_cfg_cache.risk_fill_mult * FXAI_Clamp(g_ai_last_fill_risk, 0.0, 1.0) +
                          g_prob_cal_cfg_cache.risk_path_mult * FXAI_Clamp(g_ai_last_path_risk, 0.0, 1.0) +
                          (state.news_risk_block ? g_prob_cal_cfg_cache.risk_news_block_mult : 0.0) +
                          (state.rates_risk_block ? g_prob_cal_cfg_cache.risk_rates_block_mult : 0.0) +
                          (state.microstructure_stress ? g_prob_cal_cfg_cache.risk_micro_block_mult : 0.0) +
                          ((adaptive_router_posture == "CAUTION" || dynamic_state.trade_posture == "CAUTION") ? g_prob_cal_cfg_cache.risk_caution_posture_mult : 0.0) +
                          ((adaptive_router_posture == "ABSTAIN_BIAS" || dynamic_state.trade_posture == "ABSTAIN_BIAS") ? g_prob_cal_cfg_cache.risk_abstain_posture_mult : 0.0) +
                          ((adaptive_router_posture == "BLOCK" || dynamic_state.trade_posture == "BLOCK") ? g_prob_cal_cfg_cache.risk_block_posture_mult : 0.0));
   if(execution_quality_usable)
   {
      double execution_penalty_mult = MathMax(min_move_points, 0.25);
      risk_penalty += execution_penalty_mult *
                      (0.42 * FXAI_Clamp(1.0 - execution_quality_state.fill_quality_score, 0.0, 1.0) +
                       0.32 * FXAI_Clamp(execution_quality_state.latency_sensitivity_score, 0.0, 1.0) +
                       0.26 * FXAI_Clamp(execution_quality_state.liquidity_fragility_score, 0.0, 1.0));
      if(execution_quality_state.execution_state == "BLOCKED")
         risk_penalty += execution_penalty_mult * 0.60;
      else if(execution_quality_state.execution_state == "STRESSED")
         risk_penalty += execution_penalty_mult * 0.35;
      else if(execution_quality_state.execution_state == "CAUTION")
         risk_penalty += execution_penalty_mult * 0.18;
   }
   state.risk_penalty_points = risk_penalty;

   state.expected_gross_edge_points = MathAbs(state.calibrated_buy_prob - state.calibrated_sell_prob) *
                                      state.expected_move_mean_points;
   state.edge_after_costs_points = state.expected_gross_edge_points -
                                   state.spread_cost_points -
                                   state.slippage_cost_points -
                                   state.uncertainty_penalty_points -
                                   state.risk_penalty_points;

   double edge_floor_points = MathMax(g_prob_cal_cfg_cache.trade_edge_floor_points,
                                      g_prob_cal_cfg_cache.edge_floor_mult * MathMax(min_move_points, 0.25));
   double cost_floor_points = state.spread_cost_points + state.slippage_cost_points + state.risk_penalty_points;
   string calibrated_direction = (state.calibrated_buy_prob >= state.calibrated_sell_prob ? "BUY" : "SELL");
   state.final_action = FXAI_ProbCalibrationDecisionLabel(decision);

   if(state.calibration_stale)
      FXAI_ProbCalibrationAppendReason(state, "CALIBRATION_STALE");
   if(state.input_stale)
      FXAI_ProbCalibrationAppendReason(state, "INPUT_STALE");
   if(execution_quality_unknown)
      FXAI_ProbCalibrationAppendReason(state, "EXECUTION_QUALITY_UNKNOWN");
   if(execution_quality_stale)
      FXAI_ProbCalibrationAppendReason(state, "EXECUTION_QUALITY_STALE");
   if(!state.support_usable)
      FXAI_ProbCalibrationAppendReason(state, "SUPPORT_TOO_LOW");
   if(tier.calibration_quality < g_prob_cal_cfg_cache.min_calibration_quality)
      FXAI_ProbCalibrationAppendReason(state, "CALIBRATION_WEAK");
   if(MathAbs(state.raw_score) < g_prob_cal_cfg_cache.signal_zero_band)
      FXAI_ProbCalibrationAppendReason(state, "SIGNAL_TOO_CLOSE_TO_ZERO");
   if(state.expected_move_q25_points <= cost_floor_points)
      FXAI_ProbCalibrationAppendReason(state, "MOVE_DISTRIBUTION_TOO_WEAK");
   if(state.expected_gross_edge_points <= cost_floor_points)
      FXAI_ProbCalibrationAppendReason(state, "COST_TOO_HIGH");
   if(state.uncertainty_score >= g_prob_cal_cfg_cache.max_uncertainty_score)
      FXAI_ProbCalibrationAppendReason(state, "UNCERTAINTY_TOO_HIGH");
   if(state.edge_after_costs_points <= edge_floor_points)
      FXAI_ProbCalibrationAppendReason(state, "EDGE_TOO_SMALL");
   if(state.news_risk_block)
      FXAI_ProbCalibrationAppendReason(state, "NEWS_RISK_BLOCK");
   if(state.rates_risk_block)
      FXAI_ProbCalibrationAppendReason(state, "RATES_RISK_BLOCK");
   if(state.microstructure_stress)
      FXAI_ProbCalibrationAppendReason(state, "MICROSTRUCTURE_STRESS");
   if(execution_quality_usable)
   {
      if(execution_quality_state.execution_state == "BLOCKED")
         FXAI_ProbCalibrationAppendReason(state, "EXECUTION_QUALITY_BLOCK");
      else if(execution_quality_state.execution_state == "STRESSED")
         FXAI_ProbCalibrationAppendReason(state, "EXECUTION_QUALITY_STRESSED");
      else if(execution_quality_state.execution_state == "CAUTION")
         FXAI_ProbCalibrationAppendReason(state, "EXECUTION_QUALITY_CAUTION");
   }

   state.abstain = false;
   if(decision != -1)
   {
      string upstream_action = FXAI_ProbCalibrationDecisionLabel(decision);
      if(upstream_action != calibrated_direction && MathAbs(state.calibrated_buy_prob - state.calibrated_sell_prob) >= 0.08)
      {
         FXAI_ProbCalibrationAppendReason(state, "CALIBRATED_DIRECTION_CONFLICT");
         decision = -1;
      }
      else if(state.reason_count > 0)
      {
         decision = -1;
      }
   }
   else
   {
      state.abstain = true;
   }

   if(decision == -1)
   {
      state.final_action = "SKIP";
      state.abstain = true;
   }
   else
   {
      state.final_action = FXAI_ProbCalibrationDecisionLabel(decision);
   }

   g_ai_last_confidence = FXAI_Clamp(MathMin(g_ai_last_confidence, state.calibrated_confidence), 0.0, 1.0);
   g_ai_last_expected_move_points = MathMax(state.expected_move_mean_points, 0.0);
   g_ai_last_trade_edge_points = state.edge_after_costs_points;
   g_policy_last_confidence = FXAI_Clamp(MathMin(g_policy_last_confidence, state.calibrated_confidence), 0.0, 1.0);
   g_policy_last_enter_prob = FXAI_Clamp(MathMin(g_policy_last_enter_prob, state.calibrated_confidence), 0.0, 1.0);
   g_policy_last_no_trade_prob = FXAI_Clamp(MathMax(g_policy_last_no_trade_prob,
                                                    1.0 - FXAI_Clamp(state.calibrated_confidence + 0.10, 0.0, 1.0)),
                                            0.0,
                                            1.0);
   g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * FXAI_Clamp(1.0 - 0.20 * uncertainty_score, 0.20, 1.0), 0.0, 1.0);
   if(state.abstain)
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.35, 0.05, 1.60);
      g_policy_last_enter_prob = FXAI_Clamp(g_policy_last_enter_prob * 0.55, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(MathMax(g_policy_last_no_trade_prob, 0.84), 0.0, 1.0);
   }

   g_prob_calibration_last_ready = state.ready;
   g_prob_calibration_last_fallback_used = state.fallback_used;
   g_prob_calibration_last_abstain = state.abstain;
   g_prob_calibration_last_calibration_stale = state.calibration_stale;
   g_prob_calibration_last_input_stale = state.input_stale;
   g_prob_calibration_last_generated_at = state.generated_at;
   g_prob_calibration_last_method = state.method;
   g_prob_calibration_last_tier_kind = state.selected_tier_kind;
   g_prob_calibration_last_tier_key = state.selected_tier_key;
   g_prob_calibration_last_support = state.selected_support;
   g_prob_calibration_last_quality = state.selected_quality;
   g_prob_calibration_last_raw_score = state.raw_score;
   g_prob_calibration_last_raw_action = state.raw_action;
   g_prob_calibration_last_raw_buy_prob = state.raw_buy_prob;
   g_prob_calibration_last_raw_sell_prob = state.raw_sell_prob;
   g_prob_calibration_last_raw_skip_prob = state.raw_skip_prob;
   g_prob_calibration_last_buy_prob = state.calibrated_buy_prob;
   g_prob_calibration_last_sell_prob = state.calibrated_sell_prob;
   g_prob_calibration_last_skip_prob = state.calibrated_skip_prob;
   g_prob_calibration_last_confidence = state.calibrated_confidence;
   g_prob_calibration_last_move_mean = state.expected_move_mean_points;
   g_prob_calibration_last_move_q25 = state.expected_move_q25_points;
   g_prob_calibration_last_move_q50 = state.expected_move_q50_points;
   g_prob_calibration_last_move_q75 = state.expected_move_q75_points;
   g_prob_calibration_last_spread_cost = state.spread_cost_points;
   g_prob_calibration_last_slippage_cost = state.slippage_cost_points;
   g_prob_calibration_last_uncertainty_score = state.uncertainty_score;
   g_prob_calibration_last_uncertainty_penalty = state.uncertainty_penalty_points;
   g_prob_calibration_last_risk_penalty = state.risk_penalty_points;
   g_prob_calibration_last_gross_edge = state.expected_gross_edge_points;
   g_prob_calibration_last_edge_after_costs = state.edge_after_costs_points;
   g_prob_calibration_last_final_action = state.final_action;
   g_prob_calibration_last_session = state.session_label;
   g_prob_calibration_last_regime = state.regime_label;
   g_prob_calibration_last_reasons_csv = FXAI_ProbCalibrationReasonsCSV(state);
   g_prob_calibration_last_primary_reason = (state.reason_count > 0 ? state.reason_codes[0] : "");
}

void FXAI_ProbCalibrationWriteRuntimeArtifacts(const string symbol,
                                               const FXAIProbCalibrationRuntimeState &state)
{
   if(!state.ready || StringLen(symbol) <= 0)
      return;

   int handle = FileOpen(FXAI_ProbCalibrationRuntimeStateFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, "schema_version\t1\r\n");
      FileWriteString(handle, "symbol\t" + symbol + "\r\n");
      FileWriteString(handle, "generated_at\t" + IntegerToString((int)state.generated_at) + "\r\n");
      FileWriteString(handle, "method\t" + state.method + "\r\n");
      FileWriteString(handle, "session_label\t" + state.session_label + "\r\n");
      FileWriteString(handle, "regime_label\t" + state.regime_label + "\r\n");
      FileWriteString(handle, "selected_tier_kind\t" + state.selected_tier_kind + "\r\n");
      FileWriteString(handle, "selected_tier_key\t" + state.selected_tier_key + "\r\n");
      FileWriteString(handle, "selected_support\t" + IntegerToString(state.selected_support) + "\r\n");
      FileWriteString(handle, "selected_quality\t" + DoubleToString(state.selected_quality, 6) + "\r\n");
      FileWriteString(handle, "raw_action\t" + state.raw_action + "\r\n");
      FileWriteString(handle, "raw_score\t" + DoubleToString(state.raw_score, 6) + "\r\n");
      FileWriteString(handle, "raw_buy_prob\t" + DoubleToString(state.raw_buy_prob, 6) + "\r\n");
      FileWriteString(handle, "raw_sell_prob\t" + DoubleToString(state.raw_sell_prob, 6) + "\r\n");
      FileWriteString(handle, "raw_skip_prob\t" + DoubleToString(state.raw_skip_prob, 6) + "\r\n");
      FileWriteString(handle, "calibrated_buy_prob\t" + DoubleToString(state.calibrated_buy_prob, 6) + "\r\n");
      FileWriteString(handle, "calibrated_sell_prob\t" + DoubleToString(state.calibrated_sell_prob, 6) + "\r\n");
      FileWriteString(handle, "calibrated_skip_prob\t" + DoubleToString(state.calibrated_skip_prob, 6) + "\r\n");
      FileWriteString(handle, "calibrated_confidence\t" + DoubleToString(state.calibrated_confidence, 6) + "\r\n");
      FileWriteString(handle, "expected_move_mean_points\t" + DoubleToString(state.expected_move_mean_points, 6) + "\r\n");
      FileWriteString(handle, "expected_move_q25_points\t" + DoubleToString(state.expected_move_q25_points, 6) + "\r\n");
      FileWriteString(handle, "expected_move_q50_points\t" + DoubleToString(state.expected_move_q50_points, 6) + "\r\n");
      FileWriteString(handle, "expected_move_q75_points\t" + DoubleToString(state.expected_move_q75_points, 6) + "\r\n");
      FileWriteString(handle, "spread_cost_points\t" + DoubleToString(state.spread_cost_points, 6) + "\r\n");
      FileWriteString(handle, "slippage_cost_points\t" + DoubleToString(state.slippage_cost_points, 6) + "\r\n");
      FileWriteString(handle, "uncertainty_score\t" + DoubleToString(state.uncertainty_score, 6) + "\r\n");
      FileWriteString(handle, "uncertainty_penalty_points\t" + DoubleToString(state.uncertainty_penalty_points, 6) + "\r\n");
      FileWriteString(handle, "risk_penalty_points\t" + DoubleToString(state.risk_penalty_points, 6) + "\r\n");
      FileWriteString(handle, "expected_gross_edge_points\t" + DoubleToString(state.expected_gross_edge_points, 6) + "\r\n");
      FileWriteString(handle, "edge_after_costs_points\t" + DoubleToString(state.edge_after_costs_points, 6) + "\r\n");
      FileWriteString(handle, "final_action\t" + state.final_action + "\r\n");
      FileWriteString(handle, "abstain\t" + IntegerToString(state.abstain ? 1 : 0) + "\r\n");
      FileWriteString(handle, "fallback_used\t" + IntegerToString(state.fallback_used ? 1 : 0) + "\r\n");
      FileWriteString(handle, "calibration_stale\t" + IntegerToString(state.calibration_stale ? 1 : 0) + "\r\n");
      FileWriteString(handle, "input_stale\t" + IntegerToString(state.input_stale ? 1 : 0) + "\r\n");
      FileWriteString(handle, "support_usable\t" + IntegerToString(state.support_usable ? 1 : 0) + "\r\n");
      FileWriteString(handle, "reasons_csv\t" + FXAI_ProbCalibrationReasonsCSV(state) + "\r\n");
      FileClose(handle);
   }

   int hist = FileOpen(FXAI_ProbCalibrationRuntimeHistoryFile(symbol),
                       FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                       FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist == INVALID_HANDLE)
      return;
   FileSeek(hist, 0, SEEK_END);
   string json = "{";
   json += "\"schema_version\":1,";
   json += "\"generated_at\":\"" + FXAI_ProbCalibrationISO8601(state.generated_at) + "\",";
   json += "\"symbol\":\"" + FXAI_ProbCalibrationJSONEscape(symbol) + "\",";
   json += "\"state\":{";
   json += "\"method\":\"" + FXAI_ProbCalibrationJSONEscape(state.method) + "\",";
   json += "\"session_label\":\"" + FXAI_ProbCalibrationJSONEscape(state.session_label) + "\",";
   json += "\"regime_label\":\"" + FXAI_ProbCalibrationJSONEscape(state.regime_label) + "\",";
   json += "\"selected_tier_kind\":\"" + FXAI_ProbCalibrationJSONEscape(state.selected_tier_kind) + "\",";
   json += "\"selected_tier_key\":\"" + FXAI_ProbCalibrationJSONEscape(state.selected_tier_key) + "\",";
   json += "\"selected_support\":" + IntegerToString(state.selected_support) + ",";
   json += "\"selected_quality\":" + DoubleToString(state.selected_quality, 6) + ",";
   json += "\"raw_action\":\"" + state.raw_action + "\",";
   json += "\"raw_score\":" + DoubleToString(state.raw_score, 6) + ",";
   json += "\"raw_buy_prob\":" + DoubleToString(state.raw_buy_prob, 6) + ",";
   json += "\"raw_sell_prob\":" + DoubleToString(state.raw_sell_prob, 6) + ",";
   json += "\"raw_skip_prob\":" + DoubleToString(state.raw_skip_prob, 6) + ",";
   json += "\"calibrated_buy_prob\":" + DoubleToString(state.calibrated_buy_prob, 6) + ",";
   json += "\"calibrated_sell_prob\":" + DoubleToString(state.calibrated_sell_prob, 6) + ",";
   json += "\"calibrated_skip_prob\":" + DoubleToString(state.calibrated_skip_prob, 6) + ",";
   json += "\"calibrated_confidence\":" + DoubleToString(state.calibrated_confidence, 6) + ",";
   json += "\"expected_move_mean_points\":" + DoubleToString(state.expected_move_mean_points, 6) + ",";
   json += "\"expected_move_q25_points\":" + DoubleToString(state.expected_move_q25_points, 6) + ",";
   json += "\"expected_move_q50_points\":" + DoubleToString(state.expected_move_q50_points, 6) + ",";
   json += "\"expected_move_q75_points\":" + DoubleToString(state.expected_move_q75_points, 6) + ",";
   json += "\"spread_cost_points\":" + DoubleToString(state.spread_cost_points, 6) + ",";
   json += "\"slippage_cost_points\":" + DoubleToString(state.slippage_cost_points, 6) + ",";
   json += "\"uncertainty_score\":" + DoubleToString(state.uncertainty_score, 6) + ",";
   json += "\"uncertainty_penalty_points\":" + DoubleToString(state.uncertainty_penalty_points, 6) + ",";
   json += "\"risk_penalty_points\":" + DoubleToString(state.risk_penalty_points, 6) + ",";
   json += "\"expected_gross_edge_points\":" + DoubleToString(state.expected_gross_edge_points, 6) + ",";
   json += "\"edge_after_costs_points\":" + DoubleToString(state.edge_after_costs_points, 6) + ",";
   json += "\"final_action\":\"" + state.final_action + "\",";
   json += "\"abstain\":" + (state.abstain ? "true" : "false") + ",";
   json += "\"fallback_used\":" + (state.fallback_used ? "true" : "false") + ",";
   json += "\"calibration_stale\":" + (state.calibration_stale ? "true" : "false") + ",";
   json += "\"input_stale\":" + (state.input_stale ? "true" : "false") + ",";
   json += "\"support_usable\":" + (state.support_usable ? "true" : "false") + ",";
   json += "\"reason_codes\":[";
   bool first_reason = true;
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reason_codes[i]) <= 0)
         continue;
      if(!first_reason)
         json += ",";
      json += "\"" + FXAI_ProbCalibrationJSONEscape(state.reason_codes[i]) + "\"";
      first_reason = false;
   }
   json += "]";
   json += "}}";
   FileWriteString(hist, json + "\r\n");
   FileClose(hist);
}

#endif // __FXAI_RUNTIME_PROB_CALIBRATION_STAGE_MQH__
