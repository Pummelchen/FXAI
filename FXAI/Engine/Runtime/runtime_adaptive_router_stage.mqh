#ifndef __FXAI_RUNTIME_ADAPTIVE_ROUTER_STAGE_MQH__
#define __FXAI_RUNTIME_ADAPTIVE_ROUTER_STAGE_MQH__

#include "Trade\\runtime_trade_newspulse.mqh"
#include "Trade\\runtime_trade_cross_asset_state.mqh"
#include "Trade\\runtime_trade_microstructure.mqh"

#define FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED 0
#define FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED 1
#define FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE 2
#define FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED 3

struct FXAIAdaptiveRegimeState
{
   bool   valid;
   string symbol;
   datetime generated_at;
   string top_label;
   double confidence;
   double probabilities[FXAI_ADAPTIVE_ROUTER_REGIME_COUNT];
   string session_label;
   string spread_regime;
   string volatility_regime;
   double news_risk_score;
   double news_pressure;
   int    event_eta_min;
   bool   stale_news;
   double liquidity_stress;
   double breakout_pressure;
   double trend_strength;
   double range_pressure;
   double macro_pressure;
   int    reason_count;
   string reasons[FXAI_ADAPTIVE_ROUTER_MAX_REASONS];
};

string FXAI_AdaptiveRouterRuntimeStateFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_regime_router_" + FXAI_ControlPlaneSafeToken(symbol) + ".tsv";
}

string FXAI_AdaptiveRouterRuntimeHistoryFile(const string symbol)
{
   return "FXAI\\Runtime\\fxai_regime_router_history_" + FXAI_ControlPlaneSafeToken(symbol) + ".ndjson";
}

string FXAI_AdaptiveRouterRegimeLabel(const int regime_index)
{
   switch(regime_index)
   {
      case 0: return "TREND_PERSISTENT";
      case 1: return "RANGE_MEAN_REVERTING";
      case 2: return "BREAKOUT_TRANSITION";
      case 3: return "HIGH_VOL_EVENT";
      case 4: return "RISK_ON_OFF_MACRO";
      case 5: return "LIQUIDITY_STRESS";
      case 6: return "SESSION_FLOW";
      default: return "TREND_PERSISTENT";
   }
}

string FXAI_AdaptiveRouterSessionLabel(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0)
      hour = 0;
   if(hour > 23)
      hour = 23;
   if(hour >= 21 || hour < 1)
      return "ROLLOVER";
   if(hour < 7)
      return "ASIA";
   if(hour < 12)
      return "LONDON";
   if(hour < 16)
      return "LONDON_NY_OVERLAP";
   return "NEWYORK";
}

string FXAI_AdaptiveRouterStatusLabel(const int status)
{
   switch(status)
   {
      case FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED: return "SUPPRESSED";
      case FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED: return "DOWNWEIGHTED";
      case FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED: return "UPWEIGHTED";
      default: return "ACTIVE";
   }
}

void FXAI_ResetAdaptiveRegimeState(FXAIAdaptiveRegimeState &out)
{
   out.valid = false;
   out.symbol = "";
   out.generated_at = 0;
   out.top_label = "TREND_PERSISTENT";
   out.confidence = 0.0;
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
      out.probabilities[i] = 0.0;
   out.session_label = "ASIA";
   out.spread_regime = "NORMAL";
   out.volatility_regime = "NORMAL";
   out.news_risk_score = 0.0;
   out.news_pressure = 0.0;
   out.event_eta_min = -1;
   out.stale_news = true;
   out.liquidity_stress = 0.0;
   out.breakout_pressure = 0.0;
   out.trend_strength = 0.0;
   out.range_pressure = 0.0;
   out.macro_pressure = 0.0;
   out.reason_count = 0;
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_MAX_REASONS; i++)
      out.reasons[i] = "";
}

void FXAI_AdaptiveRouterAppendReason(FXAIAdaptiveRegimeState &state,
                                     const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   for(int i=0; i<state.reason_count; i++)
   {
      if(state.reasons[i] == reason)
         return;
   }
   if(state.reason_count >= FXAI_ADAPTIVE_ROUTER_MAX_REASONS)
      return;
   state.reasons[state.reason_count] = reason;
   state.reason_count++;
}

string FXAI_AdaptiveRouterReasonsCSV(const FXAIAdaptiveRegimeState &state)
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

double FXAI_AdaptiveRouterPriceDeltaPoints(const double &close_arr[],
                                           const int lookback,
                                           const double point_value)
{
   if(ArraySize(close_arr) <= lookback || point_value <= 0.0)
      return 0.0;
   return (close_arr[0] - close_arr[lookback]) / point_value;
}

double FXAI_AdaptiveRouterRecentFlipRatio(const double &close_arr[],
                                          const int count)
{
   if(ArraySize(close_arr) <= count + 1 || count <= 1)
      return 0.0;
   int flips = 0;
   int prev_sign = 0;
   for(int i=0; i<count; i++)
   {
      double delta = close_arr[i] - close_arr[i + 1];
      int sign = (delta > 0.0 ? 1 : (delta < 0.0 ? -1 : 0));
      if(sign == 0)
         continue;
      if(prev_sign != 0 && sign != prev_sign)
         flips++;
      prev_sign = sign;
   }
   return FXAI_Clamp((double)flips / (double)MathMax(count - 1, 1), 0.0, 1.0);
}

double FXAI_AdaptiveRouterRangePoints(const double &high_arr[],
                                      const double &low_arr[],
                                      const int count,
                                      const double point_value)
{
   if(ArraySize(high_arr) <= 0 || ArraySize(low_arr) <= 0 || count <= 0 || point_value <= 0.0)
      return 0.0;
   int limit = MathMin(count, MathMin(ArraySize(high_arr), ArraySize(low_arr)));
   if(limit <= 0)
      return 0.0;
   double hi = high_arr[0];
   double lo = low_arr[0];
   for(int i=1; i<limit; i++)
   {
      if(high_arr[i] > hi)
         hi = high_arr[i];
      if(low_arr[i] < lo)
         lo = low_arr[i];
   }
   return MathMax(0.0, (hi - lo) / point_value);
}

double FXAI_AdaptiveRouterMacroPairSensitivity(const string symbol)
{
   string upper = symbol;
   StringToUpper(upper);
   double score = 0.10;
   if(StringFind(upper, "JPY") >= 0 || StringFind(upper, "CHF") >= 0)
      score += 0.32;
   if(StringFind(upper, "USD") >= 0)
      score += 0.22;
   if(StringFind(upper, "AUD") >= 0 || StringFind(upper, "NZD") >= 0 || StringFind(upper, "CAD") >= 0)
      score += 0.18;
   if(StringFind(upper, "EUR") >= 0 || StringFind(upper, "GBP") >= 0)
      score += 0.16;
   return FXAI_Clamp(score, 0.10, 1.0);
}

string FXAI_AdaptiveRouterJSONEscape(const string raw)
{
   string out = raw;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   StringReplace(out, "\r", " ");
   StringReplace(out, "\n", " ");
   return out;
}

string FXAI_AdaptiveRouterProbabilitiesCSV(const FXAIAdaptiveRegimeState &state)
{
   string csv = "";
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
   {
      if(i > 0)
         csv += ",";
      csv += FXAI_AdaptiveRouterRegimeLabel(i) + "=" + DoubleToString(state.probabilities[i], 6);
   }
   return csv;
}

string FXAI_AdaptiveRouterISO8601(const datetime value)
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

void FXAI_BuildAdaptiveRegimeState(const string symbol,
                                   const FXAIDataSnapshot &snapshot,
                                   const double spread_points,
                                   const double vol_proxy_abs,
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   const double min_move_points,
                                   const double context_strength,
                                   const double context_quality,
                                   const FXAIRegimeGraphQuery &regime_graph,
                                   const FXAINewsPulsePairState &news_state,
                                   const FXAICrossAssetPairState &cross_asset_state,
                                   const FXAIMicrostructurePairState &micro_state,
                                   FXAIAdaptiveRegimeState &out)
{
   FXAI_ResetAdaptiveRegimeState(out);
   out.valid = true;
   out.symbol = symbol;
   out.generated_at = snapshot.bar_time;
   out.session_label = FXAI_AdaptiveRouterSessionLabel(snapshot.bar_time);
   out.news_risk_score = FXAI_Clamp((news_state.ready ? news_state.news_risk_score : 0.0), 0.0, 1.0);
   out.news_pressure = FXAI_Clamp((news_state.ready ? news_state.news_pressure : 0.0), -1.0, 1.0);
   out.event_eta_min = (news_state.ready ? news_state.event_eta_min : -1);
   out.stale_news = (!news_state.ready || news_state.stale);
   bool cross_ready = (cross_asset_state.ready && !cross_asset_state.stale);
   double cross_pair_risk = (cross_ready ? FXAI_Clamp(cross_asset_state.pair_cross_asset_risk_score, 0.0, 1.0) : 0.0);
   double cross_risk_off = (cross_ready ? FXAI_Clamp(cross_asset_state.risk_off_score, 0.0, 1.0) : 0.0);
   double cross_liquidity = (cross_ready ? FXAI_Clamp(MathMax(cross_asset_state.usd_liquidity_stress_score,
                                                              cross_asset_state.cross_asset_dislocation_score),
                                                      0.0,
                                                      1.0) : 0.0);
   bool micro_ready = (micro_state.ready && !micro_state.stale);
   double micro_liquidity = (micro_ready ? FXAI_Clamp(micro_state.liquidity_stress_score, 0.0, 1.0) : 0.0);
   double micro_hostile = (micro_ready ? FXAI_Clamp(micro_state.hostile_execution_score, 0.0, 1.0) : 0.0);
   double micro_breakout = (micro_ready ? FXAI_Clamp(micro_state.local_extrema_breach_score_60s, 0.0, 1.0) : 0.0);
   double micro_tick_pressure = (micro_ready ? FXAI_Clamp(MathAbs(micro_state.tick_imbalance_30s), 0.0, 1.0) : 0.0);
   double micro_directional_eff = (micro_ready ? FXAI_Clamp(micro_state.directional_efficiency_60s, 0.0, 1.0) : 0.0);
   double micro_sweep_risk = (micro_ready ? FXAI_Clamp(MathMax(micro_state.breakout_reversal_score_60s,
                                                               micro_state.exhaustion_proxy_60s),
                                                       0.0,
                                                       1.0) : 0.0);
   bool micro_handoff = (micro_ready && micro_state.handoff_flag);
   double micro_session_burst = (micro_ready ? FXAI_Clamp(MathMax(micro_state.session_open_burst_score,
                                                                  micro_state.session_spread_behavior_score),
                                                          0.0,
                                                          1.0) : 0.0);

   double point_value = (snapshot.point > 0.0 ? snapshot.point : (_Point > 0.0 ? _Point : 0.0001));
   double spread_ref = (g_regime_ema_ready ? MathMax(g_regime_spread_ema, 0.10) : MathMax(spread_points, 0.10));
   double vol_ref = (g_regime_ema_ready ? MathMax(g_regime_vol_ema, 1e-6) : MathMax(MathAbs(vol_proxy_abs), 1e-6));
   double spread_ratio = FXAI_Clamp(spread_points / spread_ref, 0.25, 4.0);
   double vol_ratio = FXAI_Clamp(MathAbs(vol_proxy_abs) / vol_ref, 0.25, 4.0);
   if(micro_ready)
   {
      double micro_spread_ratio = FXAI_Clamp(1.0 + MathMax(micro_state.spread_zscore_60s, 0.0) / 2.0, 0.60, 4.0);
      double micro_vol_ratio = FXAI_Clamp(1.0 + MathMax(micro_state.vol_burst_score_5m - 1.0, 0.0), 0.60, 4.0);
      spread_ratio = FXAI_Clamp(0.72 * spread_ratio + 0.28 * micro_spread_ratio, 0.25, 4.0);
      vol_ratio = FXAI_Clamp(0.70 * vol_ratio + 0.30 * micro_vol_ratio, 0.25, 4.0);
   }
   out.spread_regime = (spread_ratio >= 1.45 ? "ELEVATED" : (spread_ratio <= 0.85 ? "CALM" : "NORMAL"));
   out.volatility_regime = (vol_ratio >= 1.35 ? "HIGH" : (vol_ratio <= 0.82 ? "LOW" : "NORMAL"));

   double slope_points = MathAbs(FXAI_AdaptiveRouterPriceDeltaPoints(close_arr, 6, point_value));
   double long_slope_points = MathAbs(FXAI_AdaptiveRouterPriceDeltaPoints(close_arr, 12, point_value));
   double move_floor = MathMax(min_move_points, 0.50);
   double slope_norm = FXAI_Clamp((0.55 * slope_points + 0.45 * long_slope_points) / MathMax(move_floor, 0.50), 0.0, 3.0) / 3.0;
   double flip_ratio = FXAI_AdaptiveRouterRecentFlipRatio(close_arr, 8);
   double range_points = FXAI_AdaptiveRouterRangePoints(high_arr, low_arr, 12, point_value);
   double range_tightness = 1.0 - FXAI_Clamp(range_points / MathMax(4.0 * move_floor, 1.0), 0.0, 1.0);
   double reversal_pressure = FXAI_Clamp(0.55 * flip_ratio + 0.45 * range_tightness, 0.0, 1.0);
   double breakout_pressure = FXAI_Clamp(0.34 * FXAI_Clamp(vol_ratio - 1.0, 0.0, 1.5) +
                                         0.20 * FXAI_Clamp(MathAbs(regime_graph.edge_bias), 0.0, 1.0) +
                                         0.18 * FXAI_Clamp(regime_graph.transition_confidence, 0.0, 1.0) +
                                         0.14 * (1.0 - range_tightness) +
                                         0.14 * micro_breakout,
                                         0.0,
                                         1.0);
   double trend_strength = FXAI_Clamp(0.30 * FXAI_Clamp(regime_graph.persistence, 0.0, 1.0) +
                                      0.26 * slope_norm +
                                      0.14 * (1.0 - reversal_pressure) +
                                      0.10 * FXAI_Clamp(context_strength / 2.0, 0.0, 1.0) +
                                      0.10 * micro_tick_pressure +
                                      0.10 * micro_directional_eff,
                                      0.0,
                                      1.0);
   double range_pressure = FXAI_Clamp(0.31 * reversal_pressure +
                                      0.24 * range_tightness +
                                      0.18 * (1.0 - slope_norm) +
                                      0.13 * (1.0 - breakout_pressure) +
                                      0.08 * (micro_ready && micro_state.microstructure_regime == "CHOPPY_HIGH_ACTIVITY" ? 1.0 : 0.0) +
                                      0.06 * (1.0 - micro_directional_eff),
                                      0.0,
                                      1.0);
   double pair_macro_sensitivity = FXAI_AdaptiveRouterMacroPairSensitivity(symbol);
   double macro_pressure = FXAI_Clamp(0.36 * FXAI_Clamp(regime_graph.macro_alignment, 0.0, 1.0) +
                                      0.24 * MathAbs(out.news_pressure) +
                                      0.20 * pair_macro_sensitivity +
                                      0.12 * FXAI_Clamp(context_strength / 2.0, 0.0, 1.0) +
                                      0.08 * cross_pair_risk,
                                      0.0,
                                      1.0);
   double liquidity_stress = FXAI_Clamp(0.28 * FXAI_Clamp(spread_ratio - 1.0, 0.0, 2.0) +
                                        0.14 * FXAI_Clamp(regime_graph.instability, 0.0, 1.0) +
                                        0.12 * (out.session_label == "ROLLOVER" ? 1.0 : 0.0) +
                                        0.10 * FXAI_Clamp(vol_ratio - 1.0, 0.0, 2.0) +
                                        0.10 * (out.stale_news ? 1.0 : 0.0) +
                                        0.16 * micro_liquidity +
                                        0.06 * micro_hostile +
                                        0.04 * cross_liquidity,
                                        0.0,
                                        1.0);
   out.breakout_pressure = breakout_pressure;
   out.trend_strength = trend_strength;
   out.range_pressure = range_pressure;
   out.macro_pressure = macro_pressure;
   out.liquidity_stress = liquidity_stress;

   double session_flow_score = 0.12;
   if(out.session_label == "ASIA")
      session_flow_score += 0.20 * range_pressure + 0.12 * (1.0 - out.news_risk_score) + 0.08 * micro_session_burst;
   else if(out.session_label == "LONDON")
      session_flow_score += 0.22 * breakout_pressure + 0.10 * trend_strength + 0.08 * micro_session_burst;
   else if(out.session_label == "LONDON_NY_OVERLAP")
      session_flow_score += 0.18 * macro_pressure + 0.16 * breakout_pressure + 0.08 * micro_session_burst;
   else if(out.session_label == "NEWYORK")
      session_flow_score += 0.18 * trend_strength + 0.12 * macro_pressure + 0.08 * micro_session_burst;
   else
      session_flow_score += 0.20 * liquidity_stress + 0.08 * micro_session_burst;
   session_flow_score = FXAI_Clamp(session_flow_score, 0.0, 1.0);

   bool event_window = (news_state.ready && !news_state.stale &&
                        ((news_state.event_eta_min >= 0 && news_state.event_eta_min <= 45) ||
                         news_state.trade_gate == "BLOCK" ||
                         news_state.trade_gate == "CAUTION"));

   double raw[FXAI_ADAPTIVE_ROUTER_REGIME_COUNT];
   raw[0] = 0.14 + 0.46 * trend_strength + 0.16 * FXAI_Clamp(regime_graph.persistence, 0.0, 1.0) + 0.10 * (1.0 - liquidity_stress) + 0.14 * (1.0 - out.news_risk_score);
   raw[1] = 0.12 + 0.44 * range_pressure + 0.12 * (1.0 - breakout_pressure) + 0.16 * (1.0 - out.news_risk_score) + 0.16 * (1.0 - liquidity_stress);
   raw[2] = 0.12 + 0.42 * breakout_pressure + 0.14 * trend_strength + 0.10 * FXAI_Clamp(regime_graph.transition_confidence, 0.0, 1.0) + 0.10 * FXAI_Clamp(vol_ratio - 1.0, 0.0, 2.0) + 0.12 * micro_sweep_risk;
   raw[3] = 0.08 + 0.52 * out.news_risk_score + 0.18 * (event_window ? 1.0 : 0.0) + 0.10 * FXAI_Clamp(vol_ratio - 1.0, 0.0, 2.0) + 0.12 * MathAbs(out.news_pressure);
   raw[4] = 0.08 + 0.38 * macro_pressure + 0.16 * MathAbs(out.news_pressure) + 0.18 * pair_macro_sensitivity + 0.10 * FXAI_Clamp(context_quality, 0.0, 1.0) + 0.10 * cross_risk_off;
   raw[5] = 0.08 + 0.42 * liquidity_stress + 0.10 * FXAI_Clamp(regime_graph.instability, 0.0, 1.0) + 0.08 * (out.session_label == "ROLLOVER" ? 1.0 : 0.0) + 0.10 * (out.stale_news ? 1.0 : 0.0) + 0.12 * micro_hostile + 0.10 * cross_liquidity;
   raw[6] = 0.12 + 0.48 * session_flow_score + 0.14 * (1.0 - out.news_risk_score) + 0.08 * FXAI_Clamp(context_strength / 2.0, 0.0, 1.0) + 0.08 * (1.0 - liquidity_stress) + 0.10 * (micro_handoff ? 1.0 : 0.0);

   double total = 0.0;
   int top_index = 0;
   int second_index = 1;
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
   {
      raw[i] = MathMax(raw[i], 0.0001);
      total += raw[i];
      if(raw[i] > raw[top_index])
      {
         second_index = top_index;
         top_index = i;
      }
      else if(i != top_index && raw[i] > raw[second_index])
      {
         second_index = i;
      }
   }
   if(total <= 0.0)
      total = 1.0;
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
      out.probabilities[i] = raw[i] / total;
   out.top_label = FXAI_AdaptiveRouterRegimeLabel(top_index);
   out.confidence = FXAI_Clamp(0.60 * out.probabilities[top_index] +
                               0.40 * (out.probabilities[top_index] - out.probabilities[second_index]),
                               0.0,
                               1.0);

   if(event_window)
      FXAI_AdaptiveRouterAppendReason(out, "NewsPulse event window active");
   if(out.stale_news)
      FXAI_AdaptiveRouterAppendReason(out, "NewsPulse stale or unavailable");
   if(spread_ratio >= 1.35)
      FXAI_AdaptiveRouterAppendReason(out, "Spread regime elevated");
   if(vol_ratio >= 1.25)
      FXAI_AdaptiveRouterAppendReason(out, "Volatility expansion detected");
   if(trend_strength >= 0.62)
      FXAI_AdaptiveRouterAppendReason(out, "Directional persistence elevated");
   if(range_pressure >= 0.62)
      FXAI_AdaptiveRouterAppendReason(out, "Range reversion pressure elevated");
   if(macro_pressure >= 0.58)
      FXAI_AdaptiveRouterAppendReason(out, "Macro repricing pressure elevated");
   if(liquidity_stress >= 0.58)
      FXAI_AdaptiveRouterAppendReason(out, "Liquidity stress elevated");
   if(cross_ready && cross_asset_state.macro_state != "NORMAL")
      FXAI_AdaptiveRouterAppendReason(out, "Cross-asset macro regime active");
   if(cross_ready && cross_asset_state.trade_gate == "BLOCK")
      FXAI_AdaptiveRouterAppendReason(out, "Cross-asset stress blocking");
   if(micro_ready && micro_hostile >= 0.58)
      FXAI_AdaptiveRouterAppendReason(out, "Microstructure hostile execution elevated");
   if(micro_ready && micro_sweep_risk >= 0.58)
      FXAI_AdaptiveRouterAppendReason(out, "Microstructure sweep rejection risk elevated");
   if(micro_handoff && micro_session_burst >= 0.50)
      FXAI_AdaptiveRouterAppendReason(out, "Session handoff burst active");
   if(session_flow_score >= 0.58)
      FXAI_AdaptiveRouterAppendReason(out, out.session_label + " session flow dominant");
}

double FXAI_AdaptiveRouterComputeSuitability(const FXAIAdaptiveRouterProfile &profile,
                                             const FXAIAdaptiveRegimeState &state,
                                             const string plugin_name)
{
   if(!AdaptiveRouterEnabled || !profile.ready || !profile.enabled)
      return 1.0;

   double regime_blend = 0.0;
   for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
   {
      string label = FXAI_AdaptiveRouterRegimeLabel(i);
      regime_blend += state.probabilities[i] *
                      FXAI_AdaptiveRouterPluginRegimeWeight(profile, plugin_name, label);
   }
   if(regime_blend <= 0.0)
      regime_blend = 1.0;

   double session_weight = FXAI_AdaptiveRouterPluginSessionWeight(profile,
                                                                  plugin_name,
                                                                  state.session_label);
   double global_weight = FXAI_AdaptiveRouterPluginGlobalWeight(profile, plugin_name);
   double news_compatibility = FXAI_AdaptiveRouterPluginNewsCompatibility(profile, plugin_name);
   double liquidity_robustness = FXAI_AdaptiveRouterPluginLiquidityRobustness(profile, plugin_name);

   double news_factor = 1.0;
   if(state.stale_news)
      news_factor = (profile.stale_news_force_caution ? 0.86 : 0.95);
   else if(state.news_risk_score > 0.45 || state.top_label == "HIGH_VOL_EVENT")
      news_factor = FXAI_Clamp(0.70 + 0.30 * news_compatibility, 0.20, 1.60);

   double liquidity_factor = 1.0;
   if(state.liquidity_stress > 0.40)
      liquidity_factor = FXAI_Clamp(0.68 + 0.32 * liquidity_robustness, 0.20, 1.60);

   double suitability = global_weight * regime_blend * session_weight * news_factor * liquidity_factor;
   return FXAI_Clamp(suitability, profile.min_plugin_weight, profile.max_plugin_weight);
}

int FXAI_AdaptiveRouterSuitabilityStatus(const FXAIAdaptiveRouterProfile &profile,
                                         const double suitability)
{
   if(!AdaptiveRouterEnabled || !profile.ready || !profile.enabled)
      return FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE;
   if(suitability < profile.suppression_threshold)
      return FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED;
   if(suitability < profile.downweight_threshold)
      return FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED;
   if(suitability > 1.05)
      return FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED;
   return FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE;
}

string FXAI_AdaptiveRouterComputePosture(const FXAIAdaptiveRouterProfile &profile,
                                         const FXAIAdaptiveRegimeState &state,
                                         const double best_suitability,
                                         const int eligible_count)
{
   if(!AdaptiveRouterEnabled || !profile.ready || !profile.enabled)
      return "NORMAL";
   if(eligible_count <= 0 || best_suitability <= profile.block_threshold)
      return "BLOCK";
   if(state.stale_news && profile.stale_news_force_caution)
   {
      if(best_suitability <= profile.abstain_threshold)
         return "ABSTAIN_BIAS";
      return "CAUTION";
   }
   if(state.top_label == "LIQUIDITY_STRESS" && (state.liquidity_stress >= 0.74 || best_suitability <= profile.abstain_threshold))
      return "BLOCK";
   if(state.top_label == "HIGH_VOL_EVENT")
   {
      if(best_suitability <= profile.abstain_threshold || state.news_risk_score >= 0.82)
         return "ABSTAIN_BIAS";
      return "CAUTION";
   }
   if(best_suitability <= profile.abstain_threshold || state.confidence < profile.confidence_floor)
      return "ABSTAIN_BIAS";
   if(best_suitability <= profile.caution_threshold ||
      state.liquidity_stress >= 0.56 ||
      state.breakout_pressure >= 0.72)
      return "CAUTION";
   return "NORMAL";
}

double FXAI_AdaptiveRouterPostureAbstainBias(const FXAIAdaptiveRouterProfile &profile,
                                             const FXAIAdaptiveRegimeState &state,
                                             const string posture)
{
   double bias = 0.0;
   if(posture == "CAUTION")
      bias = 0.12;
   else if(posture == "ABSTAIN_BIAS")
      bias = 0.30;
   else if(posture == "BLOCK")
      bias = 0.92;
   if(state.stale_news)
      bias += profile.stale_news_abstain_bias;
   return FXAI_Clamp(bias, 0.0, 0.98);
}

void FXAI_AdaptiveRouterApplyPosture(const string posture,
                                     const double abstain_bias,
                                     int &decision)
{
   if(posture == "CAUTION")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.84, 0.10, 1.60);
      g_policy_last_enter_prob = FXAI_Clamp(g_policy_last_enter_prob - 0.05, 0.0, 1.0);
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.92, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(g_policy_last_no_trade_prob + abstain_bias, 0.0, 1.0);
   }
   else if(posture == "ABSTAIN_BIAS")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.68, 0.05, 1.60);
      g_policy_last_enter_prob = FXAI_Clamp(g_policy_last_enter_prob - 0.14, 0.0, 1.0);
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.82, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(g_policy_last_no_trade_prob + MathMax(abstain_bias, 0.18), 0.0, 1.0);
      if(g_policy_last_enter_prob < 0.32)
         decision = -1;
   }
   else if(posture == "BLOCK")
   {
      g_policy_last_size_mult = FXAI_Clamp(g_policy_last_size_mult * 0.25, 0.01, 1.60);
      g_policy_last_enter_prob = 0.0;
      g_ai_last_trade_gate = FXAI_Clamp(g_ai_last_trade_gate * 0.40, 0.0, 1.0);
      g_policy_last_no_trade_prob = FXAI_Clamp(MathMax(g_policy_last_no_trade_prob, 0.96), 0.0, 1.0);
      decision = -1;
   }
}

void FXAI_AdaptiveRouterPublishGlobals(const FXAIAdaptiveRegimeState &state,
                                       const string posture,
                                       const double abstain_bias,
                                       const string active_plugins_csv,
                                       const string downweighted_plugins_csv,
                                       const string suppressed_plugins_csv)
{
   g_adaptive_router_last_ready = state.valid;
   g_adaptive_router_last_top_label = state.top_label;
   g_adaptive_router_last_confidence = state.confidence;
   g_adaptive_router_last_posture = posture;
   g_adaptive_router_last_abstain_bias = abstain_bias;
   g_adaptive_router_last_session = state.session_label;
   g_adaptive_router_last_spread_regime = state.spread_regime;
   g_adaptive_router_last_volatility_regime = state.volatility_regime;
   g_adaptive_router_last_news_risk = state.news_risk_score;
   g_adaptive_router_last_liquidity_stress = state.liquidity_stress;
   g_adaptive_router_last_generated_at = state.generated_at;
   g_adaptive_router_last_reasons_csv = FXAI_AdaptiveRouterReasonsCSV(state);
   g_adaptive_router_last_active_plugins_csv = active_plugins_csv;
   g_adaptive_router_last_downweighted_plugins_csv = downweighted_plugins_csv;
   g_adaptive_router_last_suppressed_plugins_csv = suppressed_plugins_csv;
}

void FXAI_AdaptiveRouterWriteRuntimeArtifacts(const string symbol,
                                             const FXAIAdaptiveRouterProfile &profile,
                                             const FXAIAdaptiveRegimeState &state,
                                             const string posture,
                                             const double abstain_bias,
                                             const double &suitability[],
                                             const double &routed_weight[],
                                             const bool &selected[],
                                             const int &status[])
{
   if(!AdaptiveRouterEnabled || StringLen(symbol) <= 0 || !state.valid)
      return;

   string active_csv = "";
   string downweighted_csv = "";
   string suppressed_csv = "";
   string plugins_json = "";
   double selected_total = 0.0;
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      if(selected[ai] && routed_weight[ai] > 0.0)
         selected_total += routed_weight[ai];
   }

   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      if(suitability[ai] <= 0.0 && routed_weight[ai] <= 0.0 && !selected[ai])
         continue;
      CFXAIAIPlugin *plugin = g_plugins.Get(ai);
      if(plugin == NULL)
         continue;
      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      string token = manifest.ai_name + ":" +
                     DoubleToString((selected_total > 0.0 && selected[ai] ? routed_weight[ai] / selected_total : 0.0), 4) + ":" +
                     DoubleToString(suitability[ai], 4);
      if(status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
      {
         if(StringLen(suppressed_csv) > 0)
            suppressed_csv += "|";
         suppressed_csv += token;
      }
      else if(status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED)
      {
         if(StringLen(downweighted_csv) > 0)
            downweighted_csv += "|";
         downweighted_csv += token;
      }
      else if(selected[ai] || status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED || status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_ACTIVE)
      {
         if(StringLen(active_csv) > 0)
            active_csv += "|";
         active_csv += token;
      }

      string plugin_reason = "Balanced regime fit";
      if(status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_SUPPRESSED)
      {
         if(state.top_label == "HIGH_VOL_EVENT")
            plugin_reason = "Suppressed in event regime";
         else if(state.top_label == "LIQUIDITY_STRESS")
            plugin_reason = "Suppressed in liquidity stress";
         else
            plugin_reason = "Suppressed by low regime fit";
      }
      else if(status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_DOWNWEIGHTED)
         plugin_reason = "Downweighted by moderate regime fit";
      else if(status[ai] == FXAI_ADAPTIVE_ROUTER_STATUS_UPWEIGHTED)
         plugin_reason = "Upweighted by strong regime fit";

      if(StringLen(plugins_json) > 0)
         plugins_json += ",";
      plugins_json += "{\"name\":\"" + FXAI_AdaptiveRouterJSONEscape(manifest.ai_name) +
                      "\",\"eligible\":" + (selected[ai] ? "true" : "false") +
                      ",\"weight\":" + DoubleToString((selected_total > 0.0 && selected[ai] ? routed_weight[ai] / selected_total : 0.0), 6) +
                      ",\"suitability\":" + DoubleToString(suitability[ai], 6) +
                      ",\"status\":\"" + FXAI_AdaptiveRouterStatusLabel(status[ai]) +
                      "\",\"reasons\":[\"" + FXAI_AdaptiveRouterJSONEscape(plugin_reason) + "\"]}";
   }

   FXAI_AdaptiveRouterPublishGlobals(state,
                                     posture,
                                     abstain_bias,
                                     active_csv,
                                     downweighted_csv,
                                     suppressed_csv);

   int handle = FileOpen(FXAI_AdaptiveRouterRuntimeStateFile(symbol),
                         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                         FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(handle != INVALID_HANDLE)
   {
      FileWriteString(handle, "schema_version\t1\r\n");
      FileWriteString(handle, "symbol\t" + symbol + "\r\n");
      FileWriteString(handle, "generated_at\t" + IntegerToString((int)state.generated_at) + "\r\n");
      FileWriteString(handle, "top_regime_label\t" + state.top_label + "\r\n");
      FileWriteString(handle, "regime_confidence\t" + DoubleToString(state.confidence, 6) + "\r\n");
      FileWriteString(handle, "trade_posture\t" + posture + "\r\n");
      FileWriteString(handle, "abstain_bias\t" + DoubleToString(abstain_bias, 6) + "\r\n");
      FileWriteString(handle, "session_label\t" + state.session_label + "\r\n");
      FileWriteString(handle, "spread_regime\t" + state.spread_regime + "\r\n");
      FileWriteString(handle, "volatility_regime\t" + state.volatility_regime + "\r\n");
      FileWriteString(handle, "news_risk_score\t" + DoubleToString(state.news_risk_score, 6) + "\r\n");
      FileWriteString(handle, "news_pressure\t" + DoubleToString(state.news_pressure, 6) + "\r\n");
      FileWriteString(handle, "event_eta_min\t" + IntegerToString(state.event_eta_min) + "\r\n");
      FileWriteString(handle, "stale_news\t" + (state.stale_news ? "1" : "0") + "\r\n");
      FileWriteString(handle, "liquidity_stress\t" + DoubleToString(state.liquidity_stress, 6) + "\r\n");
      FileWriteString(handle, "breakout_pressure\t" + DoubleToString(state.breakout_pressure, 6) + "\r\n");
      FileWriteString(handle, "trend_strength\t" + DoubleToString(state.trend_strength, 6) + "\r\n");
      FileWriteString(handle, "range_pressure\t" + DoubleToString(state.range_pressure, 6) + "\r\n");
      FileWriteString(handle, "macro_pressure\t" + DoubleToString(state.macro_pressure, 6) + "\r\n");
      FileWriteString(handle, "reasons_csv\t" + FXAI_AdaptiveRouterReasonsCSV(state) + "\r\n");
      FileWriteString(handle, "probabilities_csv\t" + FXAI_AdaptiveRouterProbabilitiesCSV(state) + "\r\n");
      FileWriteString(handle, "active_plugins_csv\t" + active_csv + "\r\n");
      FileWriteString(handle, "downweighted_plugins_csv\t" + downweighted_csv + "\r\n");
      FileWriteString(handle, "suppressed_plugins_csv\t" + suppressed_csv + "\r\n");
      FileClose(handle);
   }

   int hist = FileOpen(FXAI_AdaptiveRouterRuntimeHistoryFile(symbol),
                       FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                       FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist == INVALID_HANDLE)
      hist = FileOpen(FXAI_AdaptiveRouterRuntimeHistoryFile(symbol),
                      FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON |
                      FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(hist != INVALID_HANDLE)
   {
      FileSeek(hist, 0, SEEK_END);
      string json = "{\"schema_version\":1," +
                    "\"generated_at\":\"" + FXAI_AdaptiveRouterISO8601(state.generated_at) + "\"," +
                    "\"symbol\":\"" + FXAI_AdaptiveRouterJSONEscape(symbol) + "\"," +
                    "\"regime\":{\"top_label\":\"" + state.top_label +
                    "\",\"confidence\":" + DoubleToString(state.confidence, 6) +
                    ",\"probabilities\":{";
      for(int i=0; i<FXAI_ADAPTIVE_ROUTER_REGIME_COUNT; i++)
      {
         if(i > 0)
            json += ",";
         json += "\"" + FXAI_AdaptiveRouterRegimeLabel(i) + "\":" + DoubleToString(state.probabilities[i], 6);
      }
      json += "},\"reasons\":[";
      for(int i=0; i<state.reason_count; i++)
      {
         if(i > 0)
            json += ",";
         json += "\"" + FXAI_AdaptiveRouterJSONEscape(state.reasons[i]) + "\"";
      }
      json += "],\"session\":\"" + state.session_label +
              "\",\"spread_regime\":\"" + state.spread_regime +
              "\",\"volatility_regime\":\"" + state.volatility_regime +
              "\",\"news_risk_score\":" + DoubleToString(state.news_risk_score, 6) +
              ",\"news_pressure\":" + DoubleToString(state.news_pressure, 6) +
              ",\"event_eta_min\":" + IntegerToString(state.event_eta_min) +
              ",\"stale_news\":" + (state.stale_news ? "true" : "false") +
              "},\"router\":{\"mode\":\"" + profile.router_mode +
              "\",\"top_regime\":\"" + state.top_label +
              "\",\"trade_posture\":\"" + posture +
              "\",\"abstain_bias\":" + DoubleToString(abstain_bias, 6) +
              ",\"reasons\":[";
      for(int i=0; i<state.reason_count; i++)
      {
         if(i > 0)
            json += ",";
         json += "\"" + FXAI_AdaptiveRouterJSONEscape(state.reasons[i]) + "\"";
      }
      json += "]},\"plugins\":[" + plugins_json + "]}\r\n";
      FileWriteString(hist, json);
      FileClose(hist);
   }
}

#endif // __FXAI_RUNTIME_ADAPTIVE_ROUTER_STAGE_MQH__
