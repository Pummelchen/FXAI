#ifndef __FXAI_FEATURE_REGISTRY_MQH__
#define __FXAI_FEATURE_REGISTRY_MQH__

#define FXAI_FEATURE_GROUP_COUNT ((int)FXAI_FEAT_GROUP_FILTERS + 1)

enum ENUM_FXAI_FEATURE_PROVENANCE
{
   FXAI_PROV_PRICE_BAR = 0,
   FXAI_PROV_MULTI_TIMEFRAME,
   FXAI_PROV_CONTEXT_SYMBOL,
   FXAI_PROV_TIME_CALENDAR,
   FXAI_PROV_SYMBOL_CONTRACT,
   FXAI_PROV_EVENT_MACRO,
   FXAI_PROV_DERIVED_FILTER
};

bool g_feature_drift_ready = false;
datetime g_feature_drift_last_time = 0;
int g_feature_drift_baseline_obs[FXAI_FEATURE_GROUP_COUNT];
int g_feature_drift_live_obs[FXAI_FEATURE_GROUP_COUNT];
double g_feature_drift_baseline_mean[FXAI_FEATURE_GROUP_COUNT];
double g_feature_drift_baseline_abs[FXAI_FEATURE_GROUP_COUNT];
double g_feature_drift_live_mean[FXAI_FEATURE_GROUP_COUNT];
   double g_feature_drift_live_abs[FXAI_FEATURE_GROUP_COUNT];
double g_feature_drift_ema[FXAI_FEATURE_GROUP_COUNT];

string FXAI_FeatureGroupName(const int group_id)
{
   switch(group_id)
   {
      case FXAI_FEAT_GROUP_PRICE: return "price";
      case FXAI_FEAT_GROUP_MULTI_TIMEFRAME: return "multi_timeframe";
      case FXAI_FEAT_GROUP_VOLATILITY: return "volatility";
      case FXAI_FEAT_GROUP_TIME: return "time_calendar";
      case FXAI_FEAT_GROUP_CONTEXT: return "context";
      case FXAI_FEAT_GROUP_COST: return "cost";
      case FXAI_FEAT_GROUP_MICROSTRUCTURE: return "microstructure";
      case FXAI_FEAT_GROUP_FILTERS: return "filters";
      default: return "unknown";
   }
}

string FXAI_FeatureProvenanceName(const int provenance_id)
{
   switch(provenance_id)
   {
      case FXAI_PROV_PRICE_BAR: return "price_bar";
      case FXAI_PROV_MULTI_TIMEFRAME: return "multi_timeframe";
      case FXAI_PROV_CONTEXT_SYMBOL: return "context_symbol";
      case FXAI_PROV_TIME_CALENDAR: return "time_calendar";
      case FXAI_PROV_SYMBOL_CONTRACT: return "symbol_contract";
      case FXAI_PROV_EVENT_MACRO: return "event_macro";
      case FXAI_PROV_DERIVED_FILTER: return "derived_filter";
      default: return "unknown";
   }
}

string FXAI_MTFMetricName(const int metric_id)
{
   switch(metric_id)
   {
      case FXAI_MTF_BODY_BIAS: return "body_bias";
      case FXAI_MTF_CLOSE_LOCATION: return "close_location";
      case FXAI_MTF_RANGE_PRESSURE: return "range_pressure";
      case FXAI_MTF_SPREAD_PRESSURE: return "spread_pressure";
      default: return "unknown";
   }
}

string FXAI_MainMTFSlotName(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return "m5";
      case 1: return "m15";
      case 2: return "m30";
      case 3: return "h1";
      default: return "m1";
   }
}

string FXAI_ContextMTFSlotName(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return "m1";
      case 1: return "m5";
      case 2: return "m15";
      case 3: return "m30";
      case 4: return "h1";
      default: return "m1";
   }
}

string FXAI_FeatureName(const int feature_idx)
{
   switch(feature_idx)
   {
      case 0: return "m1_ret_1";
      case 1: return "m1_ret_3";
      case 2: return "m1_ret_5";
      case 3: return "m1_slope_10";
      case 4: return "m1_zscore_10";
      case 5: return "m1_return_vol_10";
      case 6: return "spread_norm";
      case 7: return "m5_ret";
      case 8: return "m15_ret";
      case 9: return "h1_ret";
      case 10: return "ctx_ret_mean";
      case 11: return "ctx_ret_std";
      case 12: return "ctx_up_ratio";
      case 13: return "m5_slope";
      case 14: return "h1_slope";
      case 15: return "weekday_norm";
      case 16: return "hour_norm";
      case 17: return "minute_norm";
      case 18: return "body_edge";
      case 19: return "upper_wick_edge";
      case 20: return "lower_wick_edge";
      case 21: return "bar_range_norm";
      case 22: return "m5_sma100_edge";
      case 23: return "m5_sma200_edge";
      case 24: return "m15_sma100_edge";
      case 25: return "m15_sma200_edge";
      case 26: return "m30_sma100_edge";
      case 27: return "m30_sma200_edge";
      case 28: return "h1_sma100_edge";
      case 29: return "h1_sma200_edge";
      case 30: return "m5_ema100_edge";
      case 31: return "m5_ema200_edge";
      case 32: return "m15_ema100_edge";
      case 33: return "m15_ema200_edge";
      case 34: return "m30_ema100_edge";
      case 35: return "m30_ema200_edge";
      case 36: return "h1_ema100_edge";
      case 37: return "h1_ema200_edge";
      case 38: return "qsdema100_edge";
      case 39: return "qsdema200_edge";
      case 40: return "rsi14";
      case 41: return "atr14_unit";
      case 42: return "natr14";
      case 43: return "parkinson20";
      case 44: return "rogers_satchell20";
      case 45: return "garman_klass20";
      case 46: return "median21_edge";
      case 47: return "hampel21";
      case 48: return "kalman34_edge";
      case 49: return "supersmoother20_edge";
      case 50: return "ctx_top1_ret";
      case 51: return "ctx_top1_lag";
      case 52: return "ctx_top1_rel";
      case 53: return "ctx_top1_corr";
      case 54: return "ctx_top2_ret";
      case 55: return "ctx_top2_lag";
      case 56: return "ctx_top2_rel";
      case 57: return "ctx_top2_corr";
      case 58: return "ctx_top3_ret";
      case 59: return "ctx_top3_lag";
      case 60: return "ctx_top3_rel";
      case 61: return "ctx_top3_corr";
      case 62: return "shared_util";
      case 63: return "shared_stability";
      case 64: return "shared_lead";
      case 65: return "shared_coverage";
      case 66: return "close_location";
      case 67: return "wick_imbalance";
      case 68: return "spread_shock";
      case 69: return "spread_accel";
      case 70: return "spread_to_range";
      case 71: return "micro_trend";
      case 72: return "session_transition";
      case 73: return "session_overlap";
      case 74: return "rollover_proximity";
      case 75: return "triple_swap_bias";
      case 76: return "swap_long_pressure";
      case 77: return "swap_short_pressure";
      case 78: return "carry_trend_alignment";
      case 79: return "feature_family_drift";
      case 80: return "spread_points_log";
      case 81: return "spread_zscore_20";
      case 82: return "spread_vol_ratio_20";
      case 83: return "spread_rank_20";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 0: return "macro_pre_event_embargo";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 1: return "macro_post_event_embargo";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 2: return "macro_event_importance";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 3: return "macro_surprise_signed";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 4: return "macro_surprise_abs";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 5: return "macro_event_class_bias";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 6: return "macro_surprise_zscore";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 7: return "macro_revision_abs";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 8: return "macro_currency_relevance";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 9: return "macro_provenance_trust";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 10: return "macro_rates_activity";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 11: return "macro_inflation_activity";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 12: return "macro_labor_activity";
      case FXAI_MACRO_EVENT_FEATURE_OFFSET + 13: return "macro_growth_activity";
      default:
      {
         if(feature_idx >= FXAI_MAIN_MTF_FEATURE_OFFSET && feature_idx < FXAI_CONTEXT_MTF_FEATURE_OFFSET)
         {
            int rel = feature_idx - FXAI_MAIN_MTF_FEATURE_OFFSET;
            int tf_slot = rel / FXAI_MTF_STATE_FEATURES_PER_TF;
            int metric = rel % FXAI_MTF_STATE_FEATURES_PER_TF;
            return FXAI_MainMTFSlotName(tf_slot) + "_" + FXAI_MTFMetricName(metric);
         }
         if(feature_idx >= FXAI_CONTEXT_MTF_FEATURE_OFFSET && feature_idx < FXAI_MACRO_EVENT_FEATURE_OFFSET)
         {
            int rel = feature_idx - FXAI_CONTEXT_MTF_FEATURE_OFFSET;
            int slot = rel / FXAI_CONTEXT_SLOT_MTF_FEATS;
            int slot_rel = rel % FXAI_CONTEXT_SLOT_MTF_FEATS;
            int tf_slot = slot_rel / FXAI_MTF_STATE_FEATURES_PER_TF;
            int metric = slot_rel % FXAI_MTF_STATE_FEATURES_PER_TF;
            return "ctx_top" + IntegerToString(slot + 1) + "_" +
                   FXAI_ContextMTFSlotName(tf_slot) + "_" +
                   FXAI_MTFMetricName(metric);
         }
         return "";
      }
   }
}

int FXAI_FeatureProvenance(const int feature_idx)
{
   if(feature_idx < 0 || feature_idx >= FXAI_AI_FEATURES)
      return (int)FXAI_PROV_DERIVED_FILTER;
   if(feature_idx >= FXAI_MAIN_MTF_FEATURE_OFFSET && feature_idx < FXAI_CONTEXT_MTF_FEATURE_OFFSET)
      return (int)FXAI_PROV_MULTI_TIMEFRAME;
    if(feature_idx >= FXAI_CONTEXT_MTF_FEATURE_OFFSET && feature_idx < FXAI_MACRO_EVENT_FEATURE_OFFSET)
      return (int)FXAI_PROV_CONTEXT_SYMBOL;
   if(feature_idx >= FXAI_MACRO_EVENT_FEATURE_OFFSET)
      return (int)FXAI_PROV_EVENT_MACRO;
   if(feature_idx <= 6 || (feature_idx >= 18 && feature_idx <= 21) || (feature_idx >= 66 && feature_idx <= 71) || (feature_idx >= 80 && feature_idx <= 83))
      return (int)FXAI_PROV_PRICE_BAR;
   if((feature_idx >= 7 && feature_idx <= 9) || (feature_idx >= 13 && feature_idx <= 37))
      return (int)FXAI_PROV_MULTI_TIMEFRAME;
   if((feature_idx >= 10 && feature_idx <= 12) || (feature_idx >= 50 && feature_idx <= 65))
      return (int)FXAI_PROV_CONTEXT_SYMBOL;
   if((feature_idx >= 15 && feature_idx <= 17) || feature_idx == 72 || feature_idx == 73 || feature_idx == 74 || feature_idx == 75)
      return (int)FXAI_PROV_TIME_CALENDAR;
   if(feature_idx >= 76 && feature_idx <= 78)
      return (int)FXAI_PROV_SYMBOL_CONTRACT;
   return (int)FXAI_PROV_DERIVED_FILTER;
}

bool FXAI_FeatureLeakageGuarded(const int feature_idx)
{
   int prov = FXAI_FeatureProvenance(feature_idx);
   return (prov >= (int)FXAI_PROV_PRICE_BAR && prov <= (int)FXAI_PROV_DERIVED_FILTER);
}

bool FXAI_FeatureRegistrySelfTest(void)
{
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      if(StringLen(FXAI_FeatureName(f)) <= 0)
         return false;
      int prov = FXAI_FeatureProvenance(f);
      if(prov < (int)FXAI_PROV_PRICE_BAR || prov > (int)FXAI_PROV_DERIVED_FILTER)
         return false;
      if(!FXAI_FeatureLeakageGuarded(f))
         return false;
   }
   return true;
}

void FXAI_ResetFeatureDriftDiagnostics(void)
{
   g_feature_drift_ready = true;
   g_feature_drift_last_time = 0;
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      g_feature_drift_baseline_obs[g] = 0;
      g_feature_drift_live_obs[g] = 0;
      g_feature_drift_baseline_mean[g] = 0.0;
      g_feature_drift_baseline_abs[g] = 0.0;
      g_feature_drift_live_mean[g] = 0.0;
      g_feature_drift_live_abs[g] = 0.0;
      g_feature_drift_ema[g] = 0.0;
   }
}

void FXAI_BuildFeatureFamilySnapshot(const double &x[],
                                     double &family_mean[],
                                     double &family_abs[])
{
   ArrayResize(family_mean, FXAI_FEATURE_GROUP_COUNT);
   ArrayResize(family_abs, FXAI_FEATURE_GROUP_COUNT);
   int family_count[FXAI_FEATURE_GROUP_COUNT];
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      family_mean[g] = 0.0;
      family_abs[g] = 0.0;
      family_count[g] = 0;
   }

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int g = FXAI_GetFeatureGroupForIndex(f);
      if(g < 0 || g >= FXAI_FEATURE_GROUP_COUNT)
         continue;
      double v = FXAI_GetInputFeature(x, f);
      family_mean[g] += v;
      family_abs[g] += MathAbs(v);
      family_count[g]++;
   }

   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      if(family_count[g] <= 0)
         continue;
      family_mean[g] /= (double)family_count[g];
      family_abs[g] /= (double)family_count[g];
   }
}

void FXAI_UpdateFeatureDriftBaselineFromInput(const double &x[])
{
   if(!g_feature_drift_ready)
      FXAI_ResetFeatureDriftDiagnostics();

   double mean[];
   double absv[];
   FXAI_BuildFeatureFamilySnapshot(x, mean, absv);
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      int obs = g_feature_drift_baseline_obs[g];
      double alpha = (obs <= 0 ? 1.0 : FXAI_Clamp(1.0 / (double)MathMin(obs + 1, 128), 0.01, 1.0));
      g_feature_drift_baseline_mean[g] = (obs <= 0 ? mean[g] : (1.0 - alpha) * g_feature_drift_baseline_mean[g] + alpha * mean[g]);
      g_feature_drift_baseline_abs[g] = (obs <= 0 ? absv[g] : (1.0 - alpha) * g_feature_drift_baseline_abs[g] + alpha * absv[g]);
      g_feature_drift_baseline_obs[g] = obs + 1;
   }
}

void FXAI_UpdateFeatureDriftLiveFromInput(const double &x[],
                                          const datetime sample_time)
{
   if(!g_feature_drift_ready)
      FXAI_ResetFeatureDriftDiagnostics();

   double mean[];
   double absv[];
   FXAI_BuildFeatureFamilySnapshot(x, mean, absv);
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      int obs = g_feature_drift_live_obs[g];
      double alpha = (obs <= 0 ? 1.0 : 0.08);
      g_feature_drift_live_mean[g] = (obs <= 0 ? mean[g] : (1.0 - alpha) * g_feature_drift_live_mean[g] + alpha * mean[g]);
      g_feature_drift_live_abs[g] = (obs <= 0 ? absv[g] : (1.0 - alpha) * g_feature_drift_live_abs[g] + alpha * absv[g]);
      double denom = MathMax(g_feature_drift_baseline_abs[g], 0.10);
      double mean_drift = MathAbs(g_feature_drift_live_mean[g] - g_feature_drift_baseline_mean[g]) / denom;
      double abs_drift = MathAbs(g_feature_drift_live_abs[g] - g_feature_drift_baseline_abs[g]) / denom;
      double drift = FXAI_Clamp(0.55 * mean_drift + 0.45 * abs_drift, 0.0, 4.0);
      g_feature_drift_ema[g] = (obs <= 0 ? drift : 0.85 * g_feature_drift_ema[g] + 0.15 * drift);
      g_feature_drift_live_obs[g] = obs + 1;
   }
   g_feature_drift_last_time = sample_time;
}

double FXAI_GetFeatureFamilyDrift(const int group_id)
{
   if(group_id < 0 || group_id >= FXAI_FEATURE_GROUP_COUNT)
      return 0.0;
   return FXAI_Clamp(g_feature_drift_ema[group_id], 0.0, 4.0);
}

double FXAI_GetFeatureDriftPenalty(void)
{
   double sum = 0.0;
   int used = 0;
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      if(g_feature_drift_live_obs[g] <= 0 || g_feature_drift_baseline_obs[g] <= 0)
         continue;
      sum += FXAI_GetFeatureFamilyDrift(g);
      used++;
   }
   if(used <= 0)
      return 0.0;
   return FXAI_Clamp(sum / (double)used, 0.0, 4.0);
}

#endif // __FXAI_FEATURE_REGISTRY_MQH__
