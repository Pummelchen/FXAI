#ifndef __FXAI_RUNTIME_DYNAMIC_ENSEMBLE_STAGE_MQH__
#define __FXAI_RUNTIME_DYNAMIC_ENSEMBLE_STAGE_MQH__

#include "Trade\\runtime_trade_rates_engine.mqh"

#define FXAI_DYNAMIC_ENSEMBLE_MAX_PLUGIN_REASONS 4
#define FXAI_DYNAMIC_ENSEMBLE_MAX_REASONS 8

#define FXAI_DYNAMIC_ENSEMBLE_STATUS_EXCLUDED 0
#define FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED 1
#define FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED 2
#define FXAI_DYNAMIC_ENSEMBLE_STATUS_ACTIVE 3

struct FXAIDynamicEnsembleConfig
{
   bool ready;
   bool enabled;
   bool fallback_to_routed_ensemble;
   double suppress_trust_threshold;
   double downweight_trust_threshold;
   double caution_quality_threshold;
   double abstain_quality_threshold;
   double block_quality_threshold;
   double min_effective_weight;
   double max_weight_share;
   int    min_active_plugins;
   double penalty_confidence_gap;
   double penalty_context_regret;
   double penalty_disagreement;
   double penalty_drift;
   double penalty_spread_cost;
   double penalty_news;
   double penalty_rates;
   double penalty_micro;
   double penalty_stale_context;
   double penalty_single_plugin_quality;
   double penalty_concentration_quality;
   double weight_reliability_gain;
   double weight_context_edge_gain;
   double weight_global_edge_gain;
   double weight_portfolio_gain;
   double weight_context_trust_gain;
   double weight_adaptive_upweight_gain;
   double weight_adaptive_downweight_penalty;
   double family_news_compat[FXAI_FAMILY_OTHER + 1];
   double family_rates_compat[FXAI_FAMILY_OTHER + 1];
   double family_micro_compat[FXAI_FAMILY_OTHER + 1];
   double family_cost_robustness[FXAI_FAMILY_OTHER + 1];
   double family_confidence_cap[FXAI_FAMILY_OTHER + 1];
   double family_disagreement_tolerance[FXAI_FAMILY_OTHER + 1];
};

struct FXAIDynamicEnsemblePluginRecord
{
   bool   ready;
   int    ai_idx;
   string ai_name;
   int    family_id;
   int    signal;
   double buy_prob;
   double sell_prob;
   double skip_prob;
   double expected_move;
   double move_q25;
   double move_q50;
   double move_q75;
   double confidence;
   double reliability;
   double margin;
   double hit_time_frac;
   double path_risk;
   double fill_risk;
   double mfe_ratio;
   double mae_ratio;
   double buy_ev;
   double sell_ev;
   double base_meta_weight;
   double adaptive_suitability;
   int    adaptive_status;
   double ctx_edge_norm;
   double ctx_regret;
   double global_edge_norm;
   double port_edge_norm;
   double port_stability;
   double port_corr;
   double port_div;
   double ctx_trust;
   double calibration_shrink;
   double trust_score;
   double normalized_weight;
   int    status;
   int    reason_count;
   string reasons[FXAI_DYNAMIC_ENSEMBLE_MAX_PLUGIN_REASONS];
};

struct FXAIDynamicEnsembleRuntimeState
{
   bool     ready;
   bool     fallback_used;
   datetime generated_at;
   string   symbol;
   string   top_regime;
   string   session_label;
   string   trade_posture;
   double   ensemble_quality;
   double   abstain_bias;
   double   agreement_score;
   double   context_fit_score;
   double   dominant_plugin_share;
   int      participating_count;
   int      downweighted_count;
   int      suppressed_count;
   double   buy_support;
   double   sell_support;
   double   skip_support;
   double   buy_prob;
   double   sell_prob;
   double   skip_prob;
   double   final_score;
   int      final_action;
   int      reason_count;
   string   reasons[FXAI_DYNAMIC_ENSEMBLE_MAX_REASONS];
};

FXAIDynamicEnsembleConfig g_dynamic_ensemble_cfg_cache;
datetime g_dynamic_ensemble_cfg_cache_loaded_at = 0;

string FXAI_DynamicEnsembleConfigFile(void)
{
   return "FXAI\\Runtime\\dynamic_ensemble_config.tsv";
}

string FXAI_DynamicEnsembleRuntimeStateFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_dynamic_ensemble_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_DynamicEnsembleRuntimeHistoryFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_dynamic_ensemble_history_" + FXAI_ControlPlaneSafeToken(symbol) + ".ndjson";
}

string FXAI_DynamicEnsembleJSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   return out;
}

string FXAI_DynamicEnsembleISO8601(const datetime value)
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

string FXAI_DynamicEnsembleStatusLabel(const int status)
{
   switch(status)
   {
      case FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED: return "SUPPRESSED";
      case FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED: return "DOWNWEIGHTED";
      case FXAI_DYNAMIC_ENSEMBLE_STATUS_ACTIVE: return "ACTIVE";
      default: return "EXCLUDED";
   }
}

string FXAI_DynamicEnsembleActionLabel(const int action)
{
   if(action == 1)
      return "BUY";
   if(action == 0)
      return "SELL";
   return "SKIP";
}

void FXAI_ResetDynamicEnsembleConfig(FXAIDynamicEnsembleConfig &out)
{
   out.ready = true;
   out.enabled = true;
   out.fallback_to_routed_ensemble = true;
   out.suppress_trust_threshold = 0.30;
   out.downweight_trust_threshold = 0.72;
   out.caution_quality_threshold = 0.56;
   out.abstain_quality_threshold = 0.36;
   out.block_quality_threshold = 0.18;
   out.min_effective_weight = 0.04;
   out.max_weight_share = 0.66;
   out.min_active_plugins = 1;
   out.penalty_confidence_gap = 0.52;
   out.penalty_context_regret = 0.38;
   out.penalty_disagreement = 0.28;
   out.penalty_drift = 0.18;
   out.penalty_spread_cost = 0.28;
   out.penalty_news = 0.24;
   out.penalty_rates = 0.20;
   out.penalty_micro = 0.30;
   out.penalty_stale_context = 0.24;
   out.penalty_single_plugin_quality = 0.16;
   out.penalty_concentration_quality = 0.22;
   out.weight_reliability_gain = 0.34;
   out.weight_context_edge_gain = 0.18;
   out.weight_global_edge_gain = 0.10;
   out.weight_portfolio_gain = 0.16;
   out.weight_context_trust_gain = 0.18;
   out.weight_adaptive_upweight_gain = 0.05;
   out.weight_adaptive_downweight_penalty = 0.14;
   for(int fam=0; fam<=FXAI_FAMILY_OTHER; fam++)
   {
      out.family_news_compat[fam] = 1.0;
      out.family_rates_compat[fam] = 1.0;
      out.family_micro_compat[fam] = 1.0;
      out.family_cost_robustness[fam] = 1.0;
      out.family_confidence_cap[fam] = 0.96;
      out.family_disagreement_tolerance[fam] = 1.0;
   }
   out.family_news_compat[FXAI_FAMILY_LINEAR] = 0.82;
   out.family_rates_compat[FXAI_FAMILY_LINEAR] = 0.92;
   out.family_micro_compat[FXAI_FAMILY_LINEAR] = 0.96;
   out.family_cost_robustness[FXAI_FAMILY_LINEAR] = 0.98;
   out.family_confidence_cap[FXAI_FAMILY_LINEAR] = 0.84;
   out.family_disagreement_tolerance[FXAI_FAMILY_LINEAR] = 0.92;

   out.family_news_compat[FXAI_FAMILY_TRANSFORMER] = 1.08;
   out.family_rates_compat[FXAI_FAMILY_TRANSFORMER] = 1.00;
   out.family_micro_compat[FXAI_FAMILY_TRANSFORMER] = 0.84;
   out.family_cost_robustness[FXAI_FAMILY_TRANSFORMER] = 0.82;
   out.family_confidence_cap[FXAI_FAMILY_TRANSFORMER] = 0.86;
   out.family_disagreement_tolerance[FXAI_FAMILY_TRANSFORMER] = 0.92;

   out.family_news_compat[FXAI_FAMILY_RULE_BASED] = 0.76;
   out.family_rates_compat[FXAI_FAMILY_RULE_BASED] = 0.88;
   out.family_micro_compat[FXAI_FAMILY_RULE_BASED] = 1.02;
   out.family_cost_robustness[FXAI_FAMILY_RULE_BASED] = 1.02;
   out.family_confidence_cap[FXAI_FAMILY_RULE_BASED] = 0.78;
   out.family_disagreement_tolerance[FXAI_FAMILY_RULE_BASED] = 0.90;
}

void FXAI_ResetDynamicEnsemblePluginRecord(FXAIDynamicEnsemblePluginRecord &out)
{
   out.ready = false;
   out.ai_idx = -1;
   out.ai_name = "";
   out.family_id = FXAI_FAMILY_OTHER;
   out.signal = -1;
   out.buy_prob = 0.0;
   out.sell_prob = 0.0;
   out.skip_prob = 1.0;
   out.expected_move = 0.0;
   out.move_q25 = 0.0;
   out.move_q50 = 0.0;
   out.move_q75 = 0.0;
   out.confidence = 0.0;
   out.reliability = 0.0;
   out.margin = 0.0;
   out.hit_time_frac = 0.0;
   out.path_risk = 1.0;
   out.fill_risk = 1.0;
   out.mfe_ratio = 0.0;
   out.mae_ratio = 0.0;
   out.buy_ev = 0.0;
   out.sell_ev = 0.0;
   out.base_meta_weight = 0.0;
   out.adaptive_suitability = 1.0;
   out.adaptive_status = FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE;
   out.ctx_edge_norm = 0.0;
   out.ctx_regret = 0.0;
   out.global_edge_norm = 0.0;
   out.port_edge_norm = 0.0;
   out.port_stability = 0.0;
   out.port_corr = 0.0;
   out.port_div = 0.0;
   out.ctx_trust = 0.0;
   out.calibration_shrink = 1.0;
   out.trust_score = 0.0;
   out.normalized_weight = 0.0;
   out.status = FXAI_DYNAMIC_ENSEMBLE_STATUS_EXCLUDED;
   out.reason_count = 0;
   for(int i=0; i<FXAI_DYNAMIC_ENSEMBLE_MAX_PLUGIN_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_ResetDynamicEnsembleRuntimeState(FXAIDynamicEnsembleRuntimeState &out)
{
   out.ready = false;
   out.fallback_used = false;
   out.generated_at = 0;
   out.symbol = "";
   out.top_regime = "UNKNOWN";
   out.session_label = "UNKNOWN";
   out.trade_posture = "NORMAL";
   out.ensemble_quality = 0.0;
   out.abstain_bias = 0.0;
   out.agreement_score = 0.0;
   out.context_fit_score = 0.0;
   out.dominant_plugin_share = 0.0;
   out.participating_count = 0;
   out.downweighted_count = 0;
   out.suppressed_count = 0;
   out.buy_support = 0.0;
   out.sell_support = 0.0;
   out.skip_support = 1.0;
   out.buy_prob = 0.0;
   out.sell_prob = 0.0;
   out.skip_prob = 1.0;
   out.final_score = 0.0;
   out.final_action = -1;
   out.reason_count = 0;
   for(int i=0; i<FXAI_DYNAMIC_ENSEMBLE_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_DynamicEnsembleAppendReason(FXAIDynamicEnsembleRuntimeState &state,
                                      const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_DYNAMIC_ENSEMBLE_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

void FXAI_DynamicEnsembleAppendPluginReason(FXAIDynamicEnsemblePluginRecord &state,
                                            const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_DYNAMIC_ENSEMBLE_MAX_PLUGIN_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_DynamicEnsembleReasonsCSV(const FXAIDynamicEnsembleRuntimeState &state)
{
   string joined = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reasons[i];
   }
   return joined;
}

string FXAI_DynamicEnsemblePluginReasonsCSV(const FXAIDynamicEnsemblePluginRecord &state)
{
   string joined = "";
   for(int i=0; i<state.reason_count; i++)
   {
      if(StringLen(state.reasons[i]) <= 0)
         continue;
      if(StringLen(joined) > 0)
         joined += "; ";
      joined += state.reasons[i];
   }
   return joined;
}

int FXAI_DynamicEnsembleFamilySlot(const string family_name)
{
   for(int fam=0; fam<=FXAI_FAMILY_OTHER; fam++)
   {
      if(FXAI_FamilyName(fam) == family_name)
         return fam;
   }
   return FXAI_FAMILY_OTHER;
}

bool FXAI_LoadDynamicEnsembleConfig(FXAIDynamicEnsembleConfig &out)
{
   datetime now_time = TimeCurrent();
   if(now_time <= 0)
      now_time = TimeTradeServer();
   if(now_time > 0 &&
      g_dynamic_ensemble_cfg_cache_loaded_at > 0 &&
      (now_time - g_dynamic_ensemble_cfg_cache_loaded_at) < 60)
   {
      out = g_dynamic_ensemble_cfg_cache;
      return out.ready;
   }

   FXAI_ResetDynamicEnsembleConfig(out);

   int handle = FileOpen(FXAI_DynamicEnsembleConfigFile(),
                         FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
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
         int iv = (int)StringToInteger(value);
         if(key == "enabled")
            out.enabled = (iv != 0);
         else if(key == "fallback_to_routed_ensemble")
            out.fallback_to_routed_ensemble = (iv != 0);
         else if(key == "threshold_suppress_trust_threshold")
            out.suppress_trust_threshold = dv;
         else if(key == "threshold_downweight_trust_threshold")
            out.downweight_trust_threshold = dv;
         else if(key == "threshold_caution_quality_threshold")
            out.caution_quality_threshold = dv;
         else if(key == "threshold_abstain_quality_threshold")
            out.abstain_quality_threshold = dv;
         else if(key == "threshold_block_quality_threshold")
            out.block_quality_threshold = dv;
         else if(key == "threshold_min_effective_weight")
            out.min_effective_weight = dv;
         else if(key == "threshold_max_weight_share")
            out.max_weight_share = dv;
         else if(key == "threshold_min_active_plugins")
            out.min_active_plugins = iv;
         else if(key == "penalty_confidence_gap_penalty")
            out.penalty_confidence_gap = dv;
         else if(key == "penalty_context_regret_penalty")
            out.penalty_context_regret = dv;
         else if(key == "penalty_disagreement_penalty")
            out.penalty_disagreement = dv;
         else if(key == "penalty_drift_penalty")
            out.penalty_drift = dv;
         else if(key == "penalty_spread_cost_penalty")
            out.penalty_spread_cost = dv;
         else if(key == "penalty_news_penalty")
            out.penalty_news = dv;
         else if(key == "penalty_rates_penalty")
            out.penalty_rates = dv;
         else if(key == "penalty_micro_penalty")
            out.penalty_micro = dv;
         else if(key == "penalty_stale_context_penalty")
            out.penalty_stale_context = dv;
         else if(key == "penalty_single_plugin_quality_penalty")
            out.penalty_single_plugin_quality = dv;
         else if(key == "penalty_concentration_quality_penalty")
            out.penalty_concentration_quality = dv;
         else if(key == "weight_reliability_gain")
            out.weight_reliability_gain = dv;
         else if(key == "weight_context_edge_gain")
            out.weight_context_edge_gain = dv;
         else if(key == "weight_global_edge_gain")
            out.weight_global_edge_gain = dv;
         else if(key == "weight_portfolio_gain")
            out.weight_portfolio_gain = dv;
         else if(key == "weight_context_trust_gain")
            out.weight_context_trust_gain = dv;
         else if(key == "weight_adaptive_upweight_gain")
            out.weight_adaptive_upweight_gain = dv;
         else if(key == "weight_adaptive_downweight_penalty")
            out.weight_adaptive_downweight_penalty = dv;
         else if(StringFind(key, "family_news_compat_") == 0)
         {
            int fam0 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 19));
            out.family_news_compat[fam0] = dv;
         }
         else if(StringFind(key, "family_rates_compat_") == 0)
         {
            int fam1 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 20));
            out.family_rates_compat[fam1] = dv;
         }
         else if(StringFind(key, "family_micro_compat_") == 0)
         {
            int fam2 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 20));
            out.family_micro_compat[fam2] = dv;
         }
         else if(StringFind(key, "family_cost_robustness_") == 0)
         {
            int fam3 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 23));
            out.family_cost_robustness[fam3] = dv;
         }
         else if(StringFind(key, "family_confidence_cap_") == 0)
         {
            int fam4 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 22));
            out.family_confidence_cap[fam4] = dv;
         }
         else if(StringFind(key, "family_disagreement_tolerance_") == 0)
         {
            int fam5 = FXAI_DynamicEnsembleFamilySlot(StringSubstr(key, 30));
            out.family_disagreement_tolerance[fam5] = dv;
         }
      }
      FileClose(handle);
   }

   g_dynamic_ensemble_cfg_cache = out;
   g_dynamic_ensemble_cfg_cache_loaded_at = now_time;
   return out.ready;
}

double FXAI_DynamicEnsembleCenterDirection(const FXAIDynamicEnsemblePluginRecord &records[])
{
   double numer = 0.0;
   double denom = 0.0;
   for(int i=0; i<ArraySize(records); i++)
   {
      if(!records[i].ready || records[i].base_meta_weight <= 0.0)
         continue;
      double direction = FXAI_Clamp(records[i].buy_prob - records[i].sell_prob, -1.0, 1.0);
      numer += records[i].base_meta_weight * direction;
      denom += records[i].base_meta_weight;
   }
   if(denom <= 0.0)
      return 0.0;
   return FXAI_Clamp(numer / denom, -1.0, 1.0);
}

double FXAI_DynamicEnsembleRiskStress(const FXAINewsPulsePairState &news_state,
                                      const FXAIRatesEnginePairState &rates_state,
                                      const FXAIMicrostructurePairState &micro_state,
                                      const double drift_norm)
{
   double news_stress = 0.0;
   if(!news_state.ready || news_state.stale)
      news_stress = 0.50;
   else
      news_stress = FXAI_Clamp(MathMax(news_state.news_risk_score,
                                       news_state.trade_gate == "BLOCK" ? 0.95 :
                                       (news_state.trade_gate == "CAUTION" ? 0.64 : 0.0)),
                               0.0,
                               1.0);

   double rates_stress = 0.0;
   if(!rates_state.ready || rates_state.stale)
      rates_stress = 0.40;
   else
      rates_stress = FXAI_Clamp(MathMax(rates_state.rates_risk_score,
                                        rates_state.trade_gate == "BLOCK" ? 0.90 :
                                        (rates_state.trade_gate == "CAUTION" ? 0.60 : 0.0)),
                                0.0,
                                1.0);

   double micro_stress = 0.0;
   if(!micro_state.ready || micro_state.stale)
      micro_stress = 0.50;
   else
      micro_stress = FXAI_Clamp(MathMax(micro_state.hostile_execution_score,
                                        MathMax(micro_state.liquidity_stress_score,
                                                micro_state.trade_gate == "BLOCK" ? 0.96 :
                                                (micro_state.trade_gate == "CAUTION" ? 0.64 : 0.0))),
                                0.0,
                                1.0);

   return FXAI_Clamp(0.34 * news_stress + 0.22 * rates_stress + 0.30 * micro_stress + 0.14 * drift_norm,
                     0.0,
                     1.0);
}

void FXAI_DynamicEnsembleNormalizeWeights(FXAIDynamicEnsemblePluginRecord &records[],
                                          const double max_share)
{
   double total = 0.0;
   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].status >= FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED &&
         records[i].trust_score > 0.0)
         total += records[i].trust_score;
   }
   if(total <= 0.0)
      return;

   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].status >= FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED &&
         records[i].trust_score > 0.0)
         records[i].normalized_weight = records[i].trust_score / total;
      else
         records[i].normalized_weight = 0.0;
   }

   double capped_sum = 0.0;
   double uncapped_total = 0.0;
   bool needs_redistribution = false;
   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].normalized_weight <= 0.0)
         continue;
      if(records[i].normalized_weight > max_share)
      {
         records[i].normalized_weight = max_share;
         needs_redistribution = true;
      }
      else
      {
         uncapped_total += records[i].trust_score;
      }
      capped_sum += records[i].normalized_weight;
   }
   if(!needs_redistribution)
      return;

   double residual = 1.0 - capped_sum;
   if(residual <= 0.0 || uncapped_total <= 0.0)
      return;
   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].normalized_weight <= 0.0 || records[i].normalized_weight >= max_share - 1e-9)
         continue;
      records[i].normalized_weight = residual * records[i].trust_score / uncapped_total;
   }
}

bool FXAI_DynamicEnsembleEvaluate(const string symbol,
                                  const datetime sample_time,
                                  const double spread_points,
                                  const double min_move_points,
                                  const double drift_norm,
                                  const FXAIAdaptiveRegimeState &regime_state,
                                  const FXAINewsPulsePairState &news_state,
                                  const FXAIRatesEnginePairState &rates_state,
                                  const FXAIMicrostructurePairState &micro_state,
                                  FXAIDynamicEnsemblePluginRecord &records[],
                                  FXAIDynamicEnsembleRuntimeState &out)
{
   FXAI_ResetDynamicEnsembleRuntimeState(out);
   FXAIDynamicEnsembleConfig cfg;
   FXAI_LoadDynamicEnsembleConfig(cfg);
   if(!DynamicEnsembleEnabled || !cfg.ready || !cfg.enabled || ArraySize(records) <= 0)
   {
      out.fallback_used = true;
      return false;
   }

   out.ready = true;
   out.generated_at = sample_time;
   out.symbol = symbol;
   out.top_regime = regime_state.top_label;
   out.session_label = regime_state.session_label;

   double center_direction = FXAI_DynamicEnsembleCenterDirection(records);
   double direction_numer = 0.0;
   double direction_abs = 0.0;
   double quality_ctx_sum = 0.0;
   double quality_trust_sum = 0.0;
   double quality_ctx_denom = 0.0;
   double dominant_share = 0.0;

   for(int i=0; i<ArraySize(records); i++)
   {
      FXAIDynamicEnsemblePluginRecord rec = records[i];
      if(!rec.ready || rec.base_meta_weight <= 0.0)
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_EXCLUDED;
         records[i].normalized_weight = 0.0;
         continue;
      }

      if(rec.adaptive_status == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED;
         records[i].trust_score = 0.0;
         records[i].normalized_weight = 0.0;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "suppressed_by_adaptive_router");
         continue;
      }

      int family = rec.family_id;
      if(family < 0 || family > FXAI_FAMILY_OTHER)
         family = FXAI_FAMILY_OTHER;

      double prior_mult = 1.0;
      if(rec.adaptive_status == FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED)
      {
         prior_mult += cfg.weight_adaptive_upweight_gain;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "adaptive_router_upweighted");
      }
      else if(rec.adaptive_status == FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED)
      {
         prior_mult *= MathMax(0.35, 1.0 - cfg.weight_adaptive_downweight_penalty);
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "adaptive_router_downweighted");
      }

      double empirical_mult = FXAI_Clamp(0.48 +
                                         cfg.weight_reliability_gain * FXAI_Clamp(rec.reliability, 0.0, 1.0) +
                                         cfg.weight_context_edge_gain * MathMax(rec.ctx_edge_norm, 0.0) +
                                         cfg.weight_global_edge_gain * MathMax(rec.global_edge_norm, 0.0) +
                                         cfg.weight_portfolio_gain * FXAI_Clamp(0.45 * rec.port_stability +
                                                                                0.35 * rec.port_div +
                                                                                0.20 * MathMax(rec.port_edge_norm, 0.0),
                                                                                0.0,
                                                                                1.0) +
                                         cfg.weight_context_trust_gain * FXAI_Clamp(rec.ctx_trust, 0.0, 1.0) -
                                         0.24 * FXAI_Clamp(rec.ctx_regret, 0.0, 1.0) -
                                         0.12 * FXAI_Clamp(rec.port_corr, 0.0, 1.0),
                                         0.20,
                                         1.80);

      double confidence_gap = MathMax(rec.confidence - rec.reliability, 0.0);
      double confidence_cap = FXAI_Clamp(cfg.family_confidence_cap[family], 0.55, 1.00);
      double calibration_mult = FXAI_Clamp(confidence_cap -
                                           cfg.penalty_confidence_gap * confidence_gap -
                                           cfg.penalty_context_regret * FXAI_Clamp(rec.ctx_regret, 0.0, 1.0),
                                           0.30,
                                           1.00);
      records[i].calibration_shrink = calibration_mult;
      if(calibration_mult < 0.70)
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "confidence_shrunk");

      double direction_score = FXAI_Clamp(rec.buy_prob - rec.sell_prob, -1.0, 1.0);
      double disagreement = MathAbs(direction_score - center_direction);
      double stability_mult = FXAI_Clamp(1.0 -
                                         cfg.penalty_disagreement * disagreement /
                                         MathMax(cfg.family_disagreement_tolerance[family], 0.20) -
                                         cfg.penalty_drift * drift_norm,
                                         0.35,
                                         1.10);
      if(disagreement >= 0.45)
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "directional_disagreement");

      double cost_ratio = FXAI_Clamp(spread_points / MathMax(min_move_points, 0.50), 0.0, 2.5) / 2.5;
      double risk_mult = 1.0;
      double family_news = FXAI_Clamp(cfg.family_news_compat[family], 0.40, 1.30);
      double family_rates = FXAI_Clamp(cfg.family_rates_compat[family], 0.40, 1.30);
      double family_micro = FXAI_Clamp(cfg.family_micro_compat[family], 0.40, 1.30);
      double family_cost = FXAI_Clamp(cfg.family_cost_robustness[family], 0.40, 1.30);

      if(!news_state.ready || news_state.stale)
      {
         risk_mult -= cfg.penalty_stale_context * 0.45;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "newspulse_stale");
      }
      else if(news_state.trade_gate == "BLOCK" || news_state.trade_gate == "CAUTION")
      {
         risk_mult -= cfg.penalty_news *
                      FXAI_Clamp(MathMax(news_state.news_risk_score, 0.40), 0.0, 1.0) *
                      FXAI_Clamp(1.20 - family_news, 0.25, 1.20);
         if(news_state.trade_gate == "BLOCK")
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "newspulse_block_context");
         else
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "newspulse_caution_context");
      }

      if(!rates_state.ready || rates_state.stale)
      {
         risk_mult -= cfg.penalty_stale_context * 0.30;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "rates_state_stale");
      }
      else if(rates_state.trade_gate == "BLOCK" || rates_state.trade_gate == "CAUTION")
      {
         risk_mult -= cfg.penalty_rates *
                      FXAI_Clamp(MathMax(rates_state.rates_risk_score, 0.35), 0.0, 1.0) *
                      FXAI_Clamp(1.20 - family_rates, 0.25, 1.20);
         if(rates_state.trade_gate == "BLOCK")
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "rates_block_context");
         else
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "rates_caution_context");
      }

      if(!micro_state.ready || micro_state.stale)
      {
         risk_mult -= cfg.penalty_stale_context * 0.42;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "microstructure_stale");
      }
      else if(micro_state.trade_gate == "BLOCK" || micro_state.trade_gate == "CAUTION")
      {
         double micro_stress = MathMax(micro_state.hostile_execution_score, micro_state.liquidity_stress_score);
         risk_mult -= cfg.penalty_micro *
                      FXAI_Clamp(MathMax(micro_stress, 0.35), 0.0, 1.0) *
                      FXAI_Clamp(1.20 - family_micro, 0.25, 1.20);
         if(micro_state.trade_gate == "BLOCK")
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "microstructure_block_context");
         else
            FXAI_DynamicEnsembleAppendPluginReason(records[i], "microstructure_caution_context");
      }

      risk_mult -= cfg.penalty_spread_cost * cost_ratio * FXAI_Clamp(1.15 - family_cost, 0.25, 1.20);
      risk_mult = FXAI_Clamp(risk_mult, 0.10, 1.20);

      double trust = rec.base_meta_weight * prior_mult * empirical_mult * calibration_mult * stability_mult * risk_mult;
      trust = FXAI_Clamp(trust, 0.0, 4.0);
      records[i].trust_score = trust;

      if(trust < cfg.suppress_trust_threshold)
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED;
         records[i].normalized_weight = 0.0;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "trust_below_suppress_threshold");
      }
      else if(trust < cfg.downweight_trust_threshold)
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "trust_below_active_threshold");
      }
      else
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_ACTIVE;
      }
   }

   FXAI_DynamicEnsembleNormalizeWeights(records, cfg.max_weight_share);

   int weighted_candidates = 0;
   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].status >= FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED &&
         records[i].normalized_weight > 0.0)
         weighted_candidates++;
   }

   double active_sum = 0.0;
   double buy_prob = 0.0;
   double sell_prob = 0.0;
   double skip_prob = 0.0;
   double agreement_numer = 0.0;
   double agreement_denom = 0.0;

   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].normalized_weight <= 0.0)
         continue;
      if(records[i].normalized_weight < cfg.min_effective_weight &&
         weighted_candidates > cfg.min_active_plugins)
      {
         records[i].status = FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED;
         records[i].normalized_weight = 0.0;
         weighted_candidates--;
         FXAI_DynamicEnsembleAppendPluginReason(records[i], "weight_below_min_effective_weight");
         continue;
      }
      active_sum += records[i].normalized_weight;
   }

   if(active_sum <= 0.0)
   {
      out.ready = false;
      out.fallback_used = true;
      return false;
   }

   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].normalized_weight <= 0.0)
         continue;
      records[i].normalized_weight /= active_sum;
      if(records[i].normalized_weight > dominant_share)
         dominant_share = records[i].normalized_weight;
      if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_ACTIVE)
         out.participating_count++;
      else if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED)
         out.downweighted_count++;
      double direction = FXAI_Clamp(records[i].buy_prob - records[i].sell_prob, -1.0, 1.0);
      agreement_numer += records[i].normalized_weight * direction;
      agreement_denom += records[i].normalized_weight * MathAbs(direction);
      buy_prob += records[i].normalized_weight * FXAI_Clamp(records[i].buy_prob * records[i].calibration_shrink, 0.0, 1.0);
      sell_prob += records[i].normalized_weight * FXAI_Clamp(records[i].sell_prob * records[i].calibration_shrink, 0.0, 1.0);
      skip_prob += records[i].normalized_weight * FXAI_Clamp(records[i].skip_prob + (1.0 - records[i].calibration_shrink) * 0.35, 0.0, 1.0);
      quality_ctx_sum += records[i].normalized_weight *
                         FXAI_Clamp(0.40 * MathMax(records[i].ctx_edge_norm, 0.0) +
                                    0.18 * MathMax(records[i].global_edge_norm, 0.0) +
                                    0.22 * FXAI_Clamp(records[i].ctx_trust, 0.0, 1.0) +
                                    0.10 * FXAI_Clamp(records[i].port_stability, 0.0, 1.0) +
                                    0.10 * FXAI_Clamp(records[i].port_div, 0.0, 1.0),
                                    0.0,
                                    1.0);
      quality_trust_sum += records[i].normalized_weight * FXAI_Clamp(records[i].trust_score, 0.0, 1.20);
      quality_ctx_denom += records[i].normalized_weight;
      if(records[i].signal == 1)
         out.buy_support += records[i].normalized_weight;
      else if(records[i].signal == 0)
         out.sell_support += records[i].normalized_weight;
      else
         out.skip_support += records[i].normalized_weight;
   }

   for(int i=0; i<ArraySize(records); i++)
   {
      if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED)
         out.suppressed_count++;
   }

   double prob_sum = buy_prob + sell_prob + skip_prob;
   if(prob_sum <= 0.0)
      prob_sum = 1.0;
   out.buy_prob = FXAI_Clamp(buy_prob / prob_sum, 0.0, 1.0);
   out.sell_prob = FXAI_Clamp(sell_prob / prob_sum, 0.0, 1.0);
   out.skip_prob = FXAI_Clamp(skip_prob / prob_sum, 0.0, 1.0);
   out.final_score = FXAI_Clamp(out.buy_prob - out.sell_prob, -1.0, 1.0);
   out.agreement_score = FXAI_Clamp(MathAbs(agreement_numer) / MathMax(agreement_denom, 1e-6), 0.0, 1.0);
   out.context_fit_score = FXAI_Clamp(quality_ctx_sum / MathMax(quality_ctx_denom, 1e-6), 0.0, 1.0);
   out.dominant_plugin_share = dominant_share;

   int effective_participants = out.participating_count + out.downweighted_count;
   double trust_strength = FXAI_Clamp(quality_trust_sum / MathMax(quality_ctx_denom, 1e-6), 0.0, 1.20) / 1.20;
   double risk_stress = FXAI_DynamicEnsembleRiskStress(news_state, rates_state, micro_state, drift_norm);
   double execution_safety = FXAI_Clamp(1.0 - risk_stress, 0.0, 1.0);
   double concentration_penalty = cfg.penalty_concentration_quality *
                                  FXAI_Clamp((dominant_share - cfg.max_weight_share) /
                                             MathMax(1.0 - cfg.max_weight_share, 0.10),
                                             0.0,
                                             1.0);
   double single_penalty = (effective_participants <= 1 ? cfg.penalty_single_plugin_quality : 0.0);
   out.ensemble_quality = FXAI_Clamp(0.34 * out.agreement_score +
                                     0.26 * trust_strength +
                                     0.18 * out.context_fit_score +
                                     0.22 * execution_safety -
                                     concentration_penalty -
                                     single_penalty,
                                     0.0,
                                     1.0);

   if(effective_participants <= 0 || out.ensemble_quality <= cfg.block_quality_threshold)
      out.trade_posture = "BLOCK";
   else if(out.ensemble_quality <= cfg.abstain_quality_threshold)
      out.trade_posture = "ABSTAIN_BIAS";
   else if(out.ensemble_quality <= cfg.caution_quality_threshold ||
           risk_stress >= 0.56 ||
           out.agreement_score <= 0.20)
      out.trade_posture = "CAUTION";
   else
      out.trade_posture = "NORMAL";

   if(out.trade_posture == "BLOCK")
      out.abstain_bias = 0.92;
   else if(out.trade_posture == "ABSTAIN_BIAS")
      out.abstain_bias = 0.34;
   else if(out.trade_posture == "CAUTION")
      out.abstain_bias = 0.14;
   else
      out.abstain_bias = 0.04;

   if(!news_state.ready || news_state.stale || !rates_state.ready || rates_state.stale || !micro_state.ready || micro_state.stale)
      out.abstain_bias = FXAI_Clamp(out.abstain_bias + 0.10, 0.0, 0.98);
   if(out.agreement_score <= 0.20)
      out.abstain_bias = FXAI_Clamp(out.abstain_bias + 0.08, 0.0, 0.98);

   if(out.trade_posture == "BLOCK")
      FXAI_DynamicEnsembleAppendReason(out, "ensemble_quality_below_block_floor");
   else if(out.trade_posture == "ABSTAIN_BIAS")
      FXAI_DynamicEnsembleAppendReason(out, "ensemble_quality_below_abstain_floor");
   else if(out.trade_posture == "CAUTION")
      FXAI_DynamicEnsembleAppendReason(out, "ensemble_quality_caution");
   if(out.agreement_score >= 0.66)
      FXAI_DynamicEnsembleAppendReason(out, "strong_plugin_agreement");
   else if(out.agreement_score <= 0.24)
      FXAI_DynamicEnsembleAppendReason(out, "plugin_disagreement_elevated");
   if(dominant_share >= 0.58)
      FXAI_DynamicEnsembleAppendReason(out, "plugin_concentration_elevated");
   if(news_state.ready && !news_state.stale && news_state.trade_gate == "CAUTION")
      FXAI_DynamicEnsembleAppendReason(out, "newspulse_caution_active");
   if(rates_state.ready && !rates_state.stale && rates_state.trade_gate == "CAUTION")
      FXAI_DynamicEnsembleAppendReason(out, "rates_caution_active");
   if(micro_state.ready && !micro_state.stale && micro_state.trade_gate == "CAUTION")
      FXAI_DynamicEnsembleAppendReason(out, "microstructure_caution_active");
   if(!news_state.ready || news_state.stale || !rates_state.ready || rates_state.stale || !micro_state.ready || micro_state.stale)
      FXAI_DynamicEnsembleAppendReason(out, "context_state_stale");

   if(out.buy_prob >= out.sell_prob && out.buy_prob > out.skip_prob)
      out.final_action = 1;
   else if(out.sell_prob > out.buy_prob && out.sell_prob > out.skip_prob)
      out.final_action = 0;
   else
      out.final_action = -1;

   return true;
}

void FXAI_DynamicEnsembleApplyPosture(const FXAIDynamicEnsembleRuntimeState &state,
                                      int &decision)
{
   if(!DynamicEnsembleEnabled || !state.ready)
      return;

   if(state.trade_posture == "CAUTION")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.86, 0.10, 1.60);
      g_policy_last_enter_prob = FXAI_Clamp(g_policy_last_enter_prob - FXAI_Clamp(DynamicEnsembleCautionEnterProbBuffer, 0.0, 0.25),
                                            0.0,
                                            1.0);
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.93, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(g_policy_last_no_trade_prob + state.abstain_bias, 0.0, 1.0);
   }
   else if(state.trade_posture == "ABSTAIN_BIAS")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.72, 0.05, 1.60);
      g_policy_last_enter_prob = FXAI_Clamp(g_policy_last_enter_prob - MathMax(0.08, FXAI_Clamp(DynamicEnsembleCautionEnterProbBuffer, 0.0, 0.25)),
                                            0.0,
                                            1.0);
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.84, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(g_policy_last_no_trade_prob + MathMax(state.abstain_bias, 0.18), 0.0, 1.0);
      if(g_policy_last_enter_prob < FXAI_Clamp(DynamicEnsembleAbstainEnterProbFloor, 0.05, 0.95))
         decision = -1;
   }
   else if(state.trade_posture == "BLOCK")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.25, 0.01, 1.60);
      g_policy_last_enter_prob = 0.0;
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.42, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(MathMax(g_policy_last_no_trade_prob, 0.96), 0.0, 1.0);
      decision = -1;
   }
}

void FXAI_DynamicEnsemblePublishGlobals(const FXAIDynamicEnsembleRuntimeState &state,
                                        const string active_csv,
                                        const string downweighted_csv,
                                        const string suppressed_csv)
{
   g_dynamic_ensemble_last_ready = state.ready;
   g_dynamic_ensemble_last_quality = state.ensemble_quality;
   g_dynamic_ensemble_last_abstain_bias = state.abstain_bias;
   g_dynamic_ensemble_last_trade_posture = state.trade_posture;
   g_dynamic_ensemble_last_top_regime = state.top_regime;
   g_dynamic_ensemble_last_session = state.session_label;
   g_dynamic_ensemble_last_generated_at = state.generated_at;
   g_dynamic_ensemble_last_buy_prob = state.buy_prob;
   g_dynamic_ensemble_last_sell_prob = state.sell_prob;
   g_dynamic_ensemble_last_skip_prob = state.skip_prob;
   g_dynamic_ensemble_last_reasons_csv = FXAI_DynamicEnsembleReasonsCSV(state);
   g_dynamic_ensemble_last_active_plugins_csv = active_csv;
   g_dynamic_ensemble_last_downweighted_plugins_csv = downweighted_csv;
   g_dynamic_ensemble_last_suppressed_plugins_csv = suppressed_csv;
}

void FXAI_DynamicEnsembleWriteRuntimeArtifacts(const string symbol,
                                               FXAIDynamicEnsembleRuntimeState &state,
                                               const FXAIDynamicEnsemblePluginRecord &records[],
                                               const int final_decision)
{
   if(!DynamicEnsembleEnabled || StringLen(symbol) <= 0 || !state.ready)
      return;

   state.final_action = final_decision;
   string active_csv = "";
   string downweighted_csv = "";
   string suppressed_csv = "";
   string plugins_json = "";
   for(int i=0; i<ArraySize(records); i++)
   {
      if(!records[i].ready)
         continue;
      string token = records[i].ai_name + ":" +
                     DoubleToString(records[i].normalized_weight, 4) + ":" +
                     DoubleToString(records[i].trust_score, 4);
      if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_SUPPRESSED)
      {
         if(StringLen(suppressed_csv) > 0)
            suppressed_csv += "|";
         suppressed_csv += token;
      }
      else if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_DOWNWEIGHTED)
      {
         if(StringLen(downweighted_csv) > 0)
            downweighted_csv += "|";
         downweighted_csv += token;
      }
      else if(records[i].status == FXAI_DYNAMIC_ENSEMBLE_STATUS_ACTIVE)
      {
         if(StringLen(active_csv) > 0)
            active_csv += "|";
         active_csv += token;
      }

      if(StringLen(plugins_json) > 0)
         plugins_json += ",";
      plugins_json += "{\"name\":\"" + FXAI_DynamicEnsembleJSONEscape(records[i].ai_name) +
                      "\",\"family\":\"" + FXAI_DynamicEnsembleJSONEscape(FXAI_FamilyName(records[i].family_id)) +
                      "\",\"status\":\"" + FXAI_DynamicEnsembleStatusLabel(records[i].status) +
                      "\",\"signal\":\"" + FXAI_DynamicEnsembleActionLabel(records[i].signal) +
                      "\",\"weight\":" + DoubleToString(records[i].normalized_weight, 6) +
                      ",\"trust\":" + DoubleToString(records[i].trust_score, 6) +
                      ",\"calibration_shrink\":" + DoubleToString(records[i].calibration_shrink, 6) +
                      ",\"reasons\":[";
      for(int r=0; r<records[i].reason_count; r++)
      {
         if(r > 0)
            plugins_json += ",";
         plugins_json += "\"" + FXAI_DynamicEnsembleJSONEscape(records[i].reasons[r]) + "\"";
      }
      plugins_json += "]}";
   }

   FXAI_DynamicEnsemblePublishGlobals(state, active_csv, downweighted_csv, suppressed_csv);

   int handle = FileOpen(FXAI_DynamicEnsembleRuntimeStateFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, "schema_version\t1\r\n");
      FileWriteString(handle, "symbol\t" + symbol + "\r\n");
      FileWriteString(handle, "generated_at\t" + IntegerToString((int)state.generated_at) + "\r\n");
      FileWriteString(handle, "top_regime\t" + state.top_regime + "\r\n");
      FileWriteString(handle, "session_label\t" + state.session_label + "\r\n");
      FileWriteString(handle, "trade_posture\t" + state.trade_posture + "\r\n");
      FileWriteString(handle, "ensemble_quality\t" + DoubleToString(state.ensemble_quality, 6) + "\r\n");
      FileWriteString(handle, "abstain_bias\t" + DoubleToString(state.abstain_bias, 6) + "\r\n");
      FileWriteString(handle, "agreement_score\t" + DoubleToString(state.agreement_score, 6) + "\r\n");
      FileWriteString(handle, "context_fit_score\t" + DoubleToString(state.context_fit_score, 6) + "\r\n");
      FileWriteString(handle, "dominant_plugin_share\t" + DoubleToString(state.dominant_plugin_share, 6) + "\r\n");
      FileWriteString(handle, "participating_count\t" + IntegerToString(state.participating_count) + "\r\n");
      FileWriteString(handle, "downweighted_count\t" + IntegerToString(state.downweighted_count) + "\r\n");
      FileWriteString(handle, "suppressed_count\t" + IntegerToString(state.suppressed_count) + "\r\n");
      FileWriteString(handle, "buy_support\t" + DoubleToString(state.buy_support, 6) + "\r\n");
      FileWriteString(handle, "sell_support\t" + DoubleToString(state.sell_support, 6) + "\r\n");
      FileWriteString(handle, "skip_support\t" + DoubleToString(state.skip_support, 6) + "\r\n");
      FileWriteString(handle, "buy_prob\t" + DoubleToString(state.buy_prob, 6) + "\r\n");
      FileWriteString(handle, "sell_prob\t" + DoubleToString(state.sell_prob, 6) + "\r\n");
      FileWriteString(handle, "skip_prob\t" + DoubleToString(state.skip_prob, 6) + "\r\n");
      FileWriteString(handle, "final_score\t" + DoubleToString(state.final_score, 6) + "\r\n");
      FileWriteString(handle, "final_action\t" + FXAI_DynamicEnsembleActionLabel(state.final_action) + "\r\n");
      FileWriteString(handle, "fallback_used\t" + (state.fallback_used ? "1" : "0") + "\r\n");
      FileWriteString(handle, "reasons_csv\t" + FXAI_DynamicEnsembleReasonsCSV(state) + "\r\n");
      FileWriteString(handle, "active_plugins_csv\t" + active_csv + "\r\n");
      FileWriteString(handle, "downweighted_plugins_csv\t" + downweighted_csv + "\r\n");
      FileWriteString(handle, "suppressed_plugins_csv\t" + suppressed_csv + "\r\n");
      FileClose(handle);
   }

   int hist = FileOpen(FXAI_DynamicEnsembleRuntimeHistoryFile(symbol),
                       FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                       FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist != INVALID_HANDLE)
   {
      FileSeek(hist, 0, SEEK_END);
      string json = "{\"schema_version\":1," +
                    "\"generated_at\":\"" + FXAI_DynamicEnsembleISO8601(state.generated_at) + "\"," +
                    "\"symbol\":\"" + FXAI_DynamicEnsembleJSONEscape(symbol) + "\"," +
                    "\"ensemble\":{\"top_regime\":\"" + FXAI_DynamicEnsembleJSONEscape(state.top_regime) +
                    "\",\"session_label\":\"" + FXAI_DynamicEnsembleJSONEscape(state.session_label) +
                    "\",\"trade_posture\":\"" + FXAI_DynamicEnsembleJSONEscape(state.trade_posture) +
                    "\",\"ensemble_quality\":" + DoubleToString(state.ensemble_quality, 6) +
                    ",\"abstain_bias\":" + DoubleToString(state.abstain_bias, 6) +
                    ",\"agreement_score\":" + DoubleToString(state.agreement_score, 6) +
                    ",\"context_fit_score\":" + DoubleToString(state.context_fit_score, 6) +
                    ",\"dominant_plugin_share\":" + DoubleToString(state.dominant_plugin_share, 6) +
                    ",\"buy_prob\":" + DoubleToString(state.buy_prob, 6) +
                    ",\"sell_prob\":" + DoubleToString(state.sell_prob, 6) +
                    ",\"skip_prob\":" + DoubleToString(state.skip_prob, 6) +
                    ",\"buy_support\":" + DoubleToString(state.buy_support, 6) +
                    ",\"sell_support\":" + DoubleToString(state.sell_support, 6) +
                    ",\"skip_support\":" + DoubleToString(state.skip_support, 6) +
                    ",\"final_score\":" + DoubleToString(state.final_score, 6) +
                    ",\"final_action\":\"" + FXAI_DynamicEnsembleActionLabel(state.final_action) +
                    "\",\"reasons\":[";
      for(int i=0; i<state.reason_count; i++)
      {
         if(i > 0)
            json += ",";
         json += "\"" + FXAI_DynamicEnsembleJSONEscape(state.reasons[i]) + "\"";
      }
      json += "]},\"plugins\":[" + plugins_json + "]}\r\n";
      FileWriteString(hist, json);
      FileClose(hist);
   }
}

#endif // __FXAI_RUNTIME_DYNAMIC_ENSEMBLE_STAGE_MQH__
