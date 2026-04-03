#ifndef __FXAI_AUDIT_WORLD_PLAN_MQH__
#define __FXAI_AUDIT_WORLD_PLAN_MQH__
string FXAI_AuditWorldPlanFile(const string symbol)
{
   string clean = symbol;
   if(StringLen(clean) <= 0)
      clean = _Symbol;
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
   return "FXAI\\Offline\\Promotions\\fxai_world_plan_" + clean + ".tsv";
}

double FXAI_AuditWorldHashUnit(const datetime t,
                               const int salt)
{
   long x = (long)t + (long)(salt + 1) * 1315423911;
   x ^= (x << 13);
   x ^= (x >> 17);
   x ^= (x << 5);
   if(x < 0)
      x = -x;
   return (double)(x % 1000000) / 1000000.0;
}

double FXAI_AuditWorldSign(const datetime t,
                           const int salt)
{
   return (FXAI_AuditWorldHashUnit(t, salt) < 0.5 ? -1.0 : 1.0);
}

double FXAI_AuditSessionEdgeStrength(const datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   double edge = 0.0;
   double dist_lon = MathAbs((double)dt.hour - 8.0);
   double dist_ny = MathAbs((double)dt.hour - 16.0);
   double dist_roll = MathMin(MathAbs((double)dt.hour - 23.0), MathAbs((double)dt.hour - 0.0));
   edge = MathMax(edge, 1.0 - dist_lon / 4.0);
   edge = MathMax(edge, 1.0 - dist_ny / 4.0);
   edge = MathMax(edge, 1.0 - dist_roll / 3.0);
   return FXAI_Clamp(edge, 0.0, 1.0);
}

int FXAI_AuditHourOf(const datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return dt.hour;
}

double FXAI_AuditSessionSigmaScale(const FXAIAuditScenarioSpec &spec,
                                   const int hour)
{
   if(hour >= 7 && hour <= 12)
      return FXAI_Clamp(spec.world_london_sigma_scale, 0.50, 3.00);
   if(hour >= 13 && hour <= 20)
      return FXAI_Clamp(spec.world_newyork_sigma_scale, 0.50, 3.00);
   return FXAI_Clamp(spec.world_asia_sigma_scale, 0.50, 3.00);
}

double FXAI_AuditSessionSpreadScale(const FXAIAuditScenarioSpec &spec,
                                    const int hour)
{
   if(hour >= 7 && hour <= 12)
      return FXAI_Clamp(spec.world_london_spread_scale, 0.50, 4.00);
   if(hour >= 13 && hour <= 20)
      return FXAI_Clamp(spec.world_newyork_spread_scale, 0.50, 4.00);
   return FXAI_Clamp(spec.world_asia_spread_scale, 0.50, 4.00);
}

void FXAI_AuditNormalizeRateBar(MqlRates &bar,
                                const double point)
{
   double pt = (point > 0.0 ? point : 1e-5);
   if(bar.open <= pt)
      bar.open = MathMax(bar.close, 10.0 * pt);
   if(bar.close <= pt)
      bar.close = MathMax(bar.open, 10.0 * pt);
   double body_hi = MathMax(bar.open, bar.close);
   double body_lo = MathMin(bar.open, bar.close);
   if(bar.high < body_hi)
      bar.high = body_hi;
   if(bar.low <= 0.0 || bar.low > body_lo)
      bar.low = MathMax(pt, body_lo - pt);
   if(bar.high - bar.low < 2.0 * pt)
   {
      bar.high = body_hi + pt;
      bar.low = MathMax(pt, body_lo - pt);
   }
   if(bar.spread < 1)
      bar.spread = 1;
}

void FXAI_AuditApplyWorldPlanToMarketRates(MqlRates &rates_m1[],
                                           const FXAIAuditScenarioSpec &spec,
                                           const double point)
{
   int bars = ArraySize(rates_m1);
   if(bars <= 0 || point <= 0.0)
      return;

   double prev_close_t = 0.0;
   double prev_ret_t = 0.0;
   double prev_shock_strength = 0.0;
   double pt = MathMax(point, 1e-5);

   for(int i=bars - 1; i>=0; i--)
   {
      MqlRates bar = rates_m1[i];
      double orig_open = MathMax(bar.open, pt);
      double orig_close = MathMax(bar.close, pt);
      double orig_high = MathMax(bar.high, MathMax(orig_open, orig_close));
      double orig_low = MathMax(pt, MathMin(bar.low, MathMin(orig_open, orig_close)));
      double orig_prev_close = orig_open;
      if(i < bars - 1)
         orig_prev_close = MathMax(rates_m1[i + 1].close, pt);
      if(prev_close_t <= pt)
         prev_close_t = orig_prev_close;

      double base_ret = (orig_prev_close > pt ? (orig_close - orig_prev_close) / orig_prev_close : 0.0);
      double base_gap = (orig_prev_close > pt ? (orig_open - orig_prev_close) / orig_prev_close : 0.0);
      double base_range = MathMax(orig_high - orig_low, 2.0 * pt);
      double body_hi = MathMax(orig_open, orig_close);
      double body_lo = MathMin(orig_open, orig_close);
      double upper_ratio = MathMax(orig_high - body_hi, 0.0) / base_range;
      double lower_ratio = MathMax(body_lo - orig_low, 0.0) / base_range;
      double session_edge = FXAI_AuditSessionEdgeStrength(bar.time);
      double sigma_scale = FXAI_Clamp(spec.world_sigma_scale, 0.50, 3.00);
      double edge_focus = FXAI_Clamp(spec.world_session_edge_focus, 0.0, 1.5);
      double persistence = FXAI_Clamp(spec.world_trend_persistence, 0.0, 1.0);
      double shock_memory = FXAI_Clamp(spec.world_shock_memory, 0.0, 1.0);
      double shock_decay = FXAI_Clamp(spec.world_shock_decay, 0.0, 1.5);
      double recovery_bias = FXAI_Clamp(spec.world_recovery_bias, -1.0, 1.0);
      double transition_burst = FXAI_Clamp(spec.world_regime_transition_burst, 0.0, 1.0);
      double transition_entropy = FXAI_Clamp(spec.world_transition_entropy, 0.0, 1.0);
      double mean_revert_bias = FXAI_Clamp(spec.world_mean_revert_bias, 0.0, 1.0);
      double vol_cluster_bias = FXAI_Clamp(spec.world_vol_cluster_bias, 0.0, 1.0);
      double gap_prob = FXAI_Clamp(spec.world_gap_prob, 0.0, 0.30);
      double gap_scale = FXAI_Clamp(spec.world_gap_scale, 0.0, 8.0);
      double spread_scale = FXAI_Clamp(spec.world_spread_scale, 0.50, 4.00);
      int session_hour = FXAI_AuditHourOf(bar.time);
      sigma_scale *= FXAI_AuditSessionSigmaScale(spec, session_hour);
      spread_scale *= FXAI_AuditSessionSpreadScale(spec, session_hour);
      double liquidity = FXAI_Clamp(spec.world_liquidity_stress, 0.0, 3.0);
      double spread_shock_prob = FXAI_Clamp(spec.world_spread_shock_prob, 0.0, 0.50);
      double spread_shock_scale = FXAI_Clamp(spec.world_spread_shock_scale, 1.0, 8.0);
      double flip_prob = FXAI_Clamp(spec.world_flip_prob, 0.0, 0.50);

      double edge_vol_mult = 1.0 + 0.32 * edge_focus * session_edge +
                             0.18 * vol_cluster_bias * prev_shock_strength;
      double trend_bias = (persistence - 0.50) * (0.30 - 0.18 * mean_revert_bias) * MathAbs(prev_ret_t);
      if(prev_ret_t >= 0.0)
         base_ret += trend_bias;
      else
         base_ret -= trend_bias;
      if(prev_shock_strength > 0.0)
      {
         double shock_term = prev_shock_strength * (0.18 * shock_memory - 0.14 * recovery_bias);
         if(prev_ret_t >= 0.0)
            base_ret += shock_term;
         else
            base_ret -= shock_term;
      }

      double ret = base_ret * sigma_scale * edge_vol_mult + spec.world_drift_bias * (0.70 + 0.30 * session_edge);
      double live_flip_prob = FXAI_Clamp(flip_prob + 0.14 * mean_revert_bias +
                                         0.10 * transition_burst * (1.0 - session_edge) +
                                         0.08 * transition_entropy,
                                         0.0,
                                         0.65);
      if(FXAI_AuditWorldHashUnit(bar.time, 5) < live_flip_prob)
         ret *= -1.0;

      double micro_gap = 0.40 * pt / MathMax(prev_close_t, pt);
      double gap_term = base_gap * (0.65 + 0.35 * sigma_scale) +
                        0.06 * session_edge * edge_focus * FXAI_AuditWorldSign(bar.time, 7) * micro_gap;
      if(FXAI_AuditWorldHashUnit(bar.time, 11) <
         FXAI_Clamp(gap_prob + 0.10 * transition_burst, 0.0, 0.40))
         gap_term += gap_scale * MathMax(MathAbs(base_ret), 0.20 * pt / MathMax(prev_close_t, pt)) * FXAI_AuditWorldSign(bar.time, 13);

      double new_open = prev_close_t * (1.0 + gap_term);
      if(new_open <= pt)
         new_open = prev_close_t;
      if(new_open <= pt)
         new_open = orig_open;
      double new_close = prev_close_t * (1.0 + ret);
      if(new_close <= pt)
         new_close = MathMax(new_open, orig_close);

      double range_scale = (0.78 + 0.22 * sigma_scale) *
                           (1.0 + 0.22 * edge_focus * session_edge + 0.10 * liquidity + 0.12 * transition_burst);
      if(prev_shock_strength > 0.0)
         range_scale *= (1.0 + 0.20 * prev_shock_strength * (0.60 + shock_memory + 0.35 * vol_cluster_bias) *
                         (1.0 + 0.20 * transition_entropy));
      double new_range = MathMax(2.0 * pt, base_range * range_scale);
      double wick_up = MathMax(0.5 * pt, new_range * MathMax(upper_ratio, 0.15));
      double wick_dn = MathMax(0.5 * pt, new_range * MathMax(lower_ratio, 0.15));
      if(recovery_bias > 0.0 && prev_shock_strength > 0.0)
      {
         if(ret >= 0.0)
            wick_dn *= (1.0 + 0.20 * recovery_bias * prev_shock_strength);
         else
            wick_up *= (1.0 + 0.20 * recovery_bias * prev_shock_strength);
      }

      double new_body_hi = MathMax(new_open, new_close);
      double new_body_lo = MathMin(new_open, new_close);
      bar.open = new_open;
      bar.close = new_close;
      bar.high = new_body_hi + wick_up;
      bar.low = MathMax(pt, new_body_lo - wick_dn);
      double live_spread_scale = spread_scale *
                                 (1.0 + 0.20 * edge_focus * session_edge + 0.16 * liquidity);
      if(FXAI_AuditWorldHashUnit(bar.time, 17) < spread_shock_prob)
         live_spread_scale *= spread_shock_scale;
      bar.spread = (int)MathMax(1.0, MathRound(MathMax((double)bar.spread, 1.0) * live_spread_scale));
      FXAI_AuditNormalizeRateBar(bar, pt);
      rates_m1[i] = bar;

      prev_ret_t = (prev_close_t > pt ? (bar.close - prev_close_t) / prev_close_t : 0.0);
      prev_shock_strength = FXAI_Clamp((MathAbs(prev_ret_t) /
                                       MathMax(MathAbs(base_ret) + 1e-6, 1e-6)) *
                                       (1.0 - 0.25 * shock_decay) +
                                       0.10 * transition_entropy,
                                       0.0,
                                       3.0);
      prev_close_t = bar.close;
   }
}

void FXAI_AuditApplyWorldPlan(FXAIAuditScenarioSpec &spec,
                              const string symbol)
{
   string file_name = FXAI_AuditWorldPlanFile(symbol);
   int handle = FileOpen(file_name,
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
      if(key == "sigma_scale")
         spec.world_sigma_scale = StringToDouble(value);
      else if(key == "drift_bias")
         spec.world_drift_bias = StringToDouble(value);
      else if(key == "spread_scale")
         spec.world_spread_scale = StringToDouble(value);
      else if(key == "gap_prob")
         spec.world_gap_prob = StringToDouble(value);
      else if(key == "gap_scale")
         spec.world_gap_scale = StringToDouble(value);
      else if(key == "flip_prob")
         spec.world_flip_prob = StringToDouble(value);
      else if(key == "context_corr_bias")
         spec.world_context_corr_bias = StringToDouble(value);
      else if(key == "liquidity_stress")
         spec.world_liquidity_stress = StringToDouble(value);
      else if(key == "session_edge_focus")
         spec.world_session_edge_focus = StringToDouble(value);
      else if(key == "trend_persistence")
         spec.world_trend_persistence = StringToDouble(value);
      else if(key == "shock_memory")
         spec.world_shock_memory = StringToDouble(value);
      else if(key == "recovery_bias")
         spec.world_recovery_bias = StringToDouble(value);
      else if(key == "spread_shock_prob")
         spec.world_spread_shock_prob = StringToDouble(value);
      else if(key == "spread_shock_scale")
         spec.world_spread_shock_scale = StringToDouble(value);
      else if(key == "regime_transition_burst")
         spec.world_regime_transition_burst = StringToDouble(value);
      else if(key == "transition_entropy")
         spec.world_transition_entropy = StringToDouble(value);
      else if(key == "mean_revert_bias")
         spec.world_mean_revert_bias = StringToDouble(value);
      else if(key == "vol_cluster_bias")
         spec.world_vol_cluster_bias = StringToDouble(value);
      else if(key == "shock_decay")
         spec.world_shock_decay = StringToDouble(value);
      else if(key == "asia_sigma_scale")
         spec.world_asia_sigma_scale = StringToDouble(value);
      else if(key == "london_sigma_scale")
         spec.world_london_sigma_scale = StringToDouble(value);
      else if(key == "newyork_sigma_scale")
         spec.world_newyork_sigma_scale = StringToDouble(value);
      else if(key == "asia_spread_scale")
         spec.world_asia_spread_scale = StringToDouble(value);
      else if(key == "london_spread_scale")
         spec.world_london_spread_scale = StringToDouble(value);
      else if(key == "newyork_spread_scale")
         spec.world_newyork_spread_scale = StringToDouble(value);
      else if(key == "macro_focus")
         spec.macro_focus = StringToDouble(value);
   }
   FileClose(handle);

   spec.world_sigma_scale = FXAI_Clamp(spec.world_sigma_scale, 0.50, 3.00);
   spec.world_drift_bias = FXAI_Clamp(spec.world_drift_bias, -3.0 * spec.sigma_per_bar, 3.0 * spec.sigma_per_bar);
   spec.world_spread_scale = FXAI_Clamp(spec.world_spread_scale, 0.50, 4.00);
   spec.world_gap_prob = FXAI_Clamp(spec.world_gap_prob, 0.0, 0.30);
   spec.world_gap_scale = FXAI_Clamp(spec.world_gap_scale, 0.0, 8.0);
   spec.world_flip_prob = FXAI_Clamp(spec.world_flip_prob, 0.0, 0.50);
   spec.world_context_corr_bias = FXAI_Clamp(spec.world_context_corr_bias, -1.0, 1.0);
   spec.world_liquidity_stress = FXAI_Clamp(spec.world_liquidity_stress, 0.0, 3.0);
   spec.world_session_edge_focus = FXAI_Clamp(spec.world_session_edge_focus, 0.0, 1.5);
   spec.world_trend_persistence = FXAI_Clamp(spec.world_trend_persistence, 0.0, 1.0);
   spec.world_shock_memory = FXAI_Clamp(spec.world_shock_memory, 0.0, 1.0);
   spec.world_recovery_bias = FXAI_Clamp(spec.world_recovery_bias, -1.0, 1.0);
   spec.world_spread_shock_prob = FXAI_Clamp(spec.world_spread_shock_prob, 0.0, 0.50);
   spec.world_spread_shock_scale = FXAI_Clamp(spec.world_spread_shock_scale, 1.0, 8.0);
   spec.world_regime_transition_burst = FXAI_Clamp(spec.world_regime_transition_burst, 0.0, 1.0);
   spec.world_transition_entropy = FXAI_Clamp(spec.world_transition_entropy, 0.0, 1.0);
   spec.world_mean_revert_bias = FXAI_Clamp(spec.world_mean_revert_bias, 0.0, 1.0);
   spec.world_vol_cluster_bias = FXAI_Clamp(spec.world_vol_cluster_bias, 0.0, 1.0);
   spec.world_shock_decay = FXAI_Clamp(spec.world_shock_decay, 0.0, 1.5);
   spec.world_asia_sigma_scale = FXAI_Clamp(spec.world_asia_sigma_scale, 0.50, 3.00);
   spec.world_london_sigma_scale = FXAI_Clamp(spec.world_london_sigma_scale, 0.50, 3.00);
   spec.world_newyork_sigma_scale = FXAI_Clamp(spec.world_newyork_sigma_scale, 0.50, 3.00);
   spec.world_asia_spread_scale = FXAI_Clamp(spec.world_asia_spread_scale, 0.50, 4.00);
   spec.world_london_spread_scale = FXAI_Clamp(spec.world_london_spread_scale, 0.50, 4.00);
   spec.world_newyork_spread_scale = FXAI_Clamp(spec.world_newyork_spread_scale, 0.50, 4.00);
   spec.macro_focus = FXAI_Clamp(spec.macro_focus, 0.0, 1.5);
}
#endif // __FXAI_AUDIT_WORLD_PLAN_MQH__
