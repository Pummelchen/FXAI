#ifndef __FXAI_AUDIT_SCENARIOS_MQH__
#define __FXAI_AUDIT_SCENARIOS_MQH__

void FXAI_AuditFillScenarioSpec(const int scenario_id,
                                FXAIAuditScenarioSpec &spec)
{
   spec.id = scenario_id;
   spec.name = "random_walk";
   spec.drift_per_bar = 0.0;
   spec.sigma_per_bar = 0.00018;
   spec.mean_revert_strength = 0.0;
   spec.vol_cluster = 0.0;
   spec.spike_prob = 0.0;
   spec.spike_scale = 0.0;
   spec.spread_points = 1.2;
   spec.macro_focus = 0.0;

   switch(scenario_id)
   {
      case 1:
         spec.name = "drift_up";
         spec.drift_per_bar = 0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 2:
         spec.name = "drift_down";
         spec.drift_per_bar = -0.00010;
         spec.sigma_per_bar = 0.00015;
         spec.spread_points = 1.0;
         break;
      case 3:
         spec.name = "mean_revert";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.mean_revert_strength = 0.22;
         spec.spread_points = 1.3;
         break;
      case 4:
         spec.name = "vol_cluster";
         spec.drift_per_bar = 0.0;
         spec.sigma_per_bar = 0.00018;
         spec.vol_cluster = 0.85;
         spec.spike_prob = 0.01;
         spec.spike_scale = 4.0;
         spec.spread_points = 1.8;
         break;
      case 5:
         spec.name = "monotonic_up";
         spec.drift_per_bar = 0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 6:
         spec.name = "monotonic_down";
         spec.drift_per_bar = -0.00022;
         spec.sigma_per_bar = 0.00003;
         spec.spread_points = 0.8;
         break;
      case 7:
         spec.name = "regime_shift";
         spec.drift_per_bar = 0.00008;
         spec.sigma_per_bar = 0.00015;
         spec.vol_cluster = 0.55;
         spec.spike_prob = 0.005;
         spec.spike_scale = 3.0;
         spec.spread_points = 1.5;
         break;
      case 8:
         spec.name = "market_recent";
         spec.spread_points = 1.2;
         break;
      case 9:
         spec.name = "market_trend";
         spec.spread_points = 1.2;
         break;
      case 10:
         spec.name = "market_chop";
         spec.spread_points = 1.4;
         break;
      case 11:
         spec.name = "market_session_edges";
         spec.spread_points = 1.6;
         break;
      case 12:
         spec.name = "market_spread_shock";
         spec.spread_points = 2.2;
         break;
      case 13:
         spec.name = "market_walkforward";
         spec.spread_points = 1.5;
         break;
      case 14:
         spec.name = "market_macro_event";
         spec.spread_points = 1.7;
         spec.macro_focus = 1.0;
         break;
      default:
         break;
   }
}

void FXAI_AuditResetMetrics(FXAIAuditScenarioMetrics &m,
                            const int ai_id,
                            const string ai_name,
                            const int family,
                            const string scenario,
                            const int bars_total)
{
   m.ai_id = ai_id;
   m.ai_name = ai_name;
   m.family = family;
   m.scenario = scenario;
   m.bars_total = bars_total;
   m.samples_total = 0;
   m.valid_preds = 0;
   m.invalid_preds = 0;
   m.buy_count = 0;
   m.sell_count = 0;
   m.skip_count = 0;
   m.true_buy_count = 0;
   m.true_sell_count = 0;
   m.true_skip_count = 0;
   m.exact_match_count = 0;
   m.directional_eval_count = 0;
   m.directional_correct_count = 0;
   m.trend_alignment_sum = 0.0;
   m.trend_alignment_count = 0;
   m.conf_sum = 0.0;
   m.rel_sum = 0.0;
   m.move_sum = 0.0;
   m.dir_conf_sum = 0.0;
   m.dir_hit_sum = 0.0;
   m.brier_sum = 0.0;
   m.calibration_abs_sum = 0.0;
   m.path_quality_abs_sum = 0.0;
   m.path_quality_count = 0;
   m.net_sum = 0.0;
   m.skip_ratio = 0.0;
   m.active_ratio = 0.0;
   m.bias_abs = 0.0;
   m.conf_drift = 0.0;
   m.brier_score = 0.0;
   m.calibration_error = 0.0;
   m.path_quality_error = 0.0;
   m.macro_event_rate = 0.0;
   m.macro_pre_rate = 0.0;
   m.macro_post_rate = 0.0;
   m.macro_importance_mean = 0.0;
   m.macro_surprise_abs_mean = 0.0;
   m.macro_data_coverage = 0.0;
   m.reset_delta = 0.0;
   m.sequence_delta = 0.0;
   m.wf_folds = 0;
   m.wf_train_samples = 0;
   m.wf_test_samples = 0;
   m.wf_train_score = 0.0;
   m.wf_test_score = 0.0;
   m.wf_test_score_std = 0.0;
   m.wf_gap = 0.0;
   m.wf_pbo = 0.0;
   m.wf_dsr = 0.0;
   m.wf_pass_rate = 0.0;
   m.score = 0.0;
   m.issue_flags = 0;
}

int FXAI_AuditDecisionFromPred(const FXAIAIPredictionV4 &pred)
{
   int best = (int)FXAI_LABEL_SKIP;
   double best_p = pred.class_probs[(int)FXAI_LABEL_SKIP];
   if(pred.class_probs[(int)FXAI_LABEL_BUY] > best_p)
   {
      best = (int)FXAI_LABEL_BUY;
      best_p = pred.class_probs[(int)FXAI_LABEL_BUY];
   }
   if(pred.class_probs[(int)FXAI_LABEL_SELL] > best_p)
      best = (int)FXAI_LABEL_SELL;
   return best;
}

double FXAI_AuditRollingCorr(const double &a[],
                             const double &b[],
                             const int start_idx,
                             const int width)
{
   int n = ArraySize(a);
   if(n != ArraySize(b) || width < 4 || start_idx < 0 || start_idx + width >= n)
      return 0.0;

   double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
   int used = 0;
   for(int k=0; k<width; k++)
   {
      double ra = FXAI_SafeReturn(a, start_idx + k, start_idx + k + 1);
      double rb = FXAI_SafeReturn(b, start_idx + k, start_idx + k + 1);
      sa += ra;
      sb += rb;
      saa += ra * ra;
      sbb += rb * rb;
      sab += ra * rb;
      used++;
   }
   if(used < 4) return 0.0;
   double ma = sa / (double)used;
   double mb = sb / (double)used;
   double va = saa / (double)used - ma * ma;
   double vb = sbb / (double)used - mb * mb;
   double cov = sab / (double)used - ma * mb;
   if(va <= 1e-12 || vb <= 1e-12) return 0.0;
   return FXAI_Clamp(cov / MathSqrt(va * vb), -1.0, 1.0);
}

void FXAI_AuditReverseChronoToSeries(const datetime &src_time[],
                                     const double &src_open[],
                                     const double &src_high[],
                                     const double &src_low[],
                                     const double &src_close[],
                                     const int &src_spread[],
                                     datetime &dst_time[],
                                     double &dst_open[],
                                     double &dst_high[],
                                     double &dst_low[],
                                     double &dst_close[],
                                     int &dst_spread[])
{
   int n = ArraySize(src_close);
   ArrayResize(dst_time, n);
   ArrayResize(dst_open, n);
   ArrayResize(dst_high, n);
   ArrayResize(dst_low, n);
   ArrayResize(dst_close, n);
   ArrayResize(dst_spread, n);
   ArraySetAsSeries(dst_time, true);
   ArraySetAsSeries(dst_open, true);
   ArraySetAsSeries(dst_high, true);
   ArraySetAsSeries(dst_low, true);
   ArraySetAsSeries(dst_close, true);
   ArraySetAsSeries(dst_spread, true);

   for(int i=0; i<n; i++)
   {
      int src = n - 1 - i;
      dst_time[i] = src_time[src];
      dst_open[i] = src_open[src];
      dst_high[i] = src_high[src];
      dst_low[i] = src_low[src];
      dst_close[i] = src_close[src];
      dst_spread[i] = src_spread[src];
   }
}

void FXAI_AuditReverseCloseSeries(const datetime &src_time[],
                                  const double &src_close[],
                                  datetime &dst_time[],
                                  double &dst_close[])
{
   int n = ArraySize(src_close);
   ArrayResize(dst_time, n);
   ArrayResize(dst_close, n);
   ArraySetAsSeries(dst_time, true);
   ArraySetAsSeries(dst_close, true);
   for(int i=0; i<n; i++)
   {
      int src = n - 1 - i;
      dst_time[i] = src_time[src];
      dst_close[i] = src_close[src];
   }
}

void FXAI_AuditDeriveContextSeriesFromBase(const double point,
                                           const double &base_open[],
                                           const double &base_high[],
                                           const double &base_low[],
                                           const double &base_close[],
                                           const int &base_spread[],
                                           const int transform_id,
                                           double &ctx_open[],
                                           double &ctx_high[],
                                           double &ctx_low[],
                                           double &ctx_close[],
                                           int &ctx_spread[])
{
   int n = ArraySize(base_close);
   ArrayResize(ctx_open, n);
   ArrayResize(ctx_high, n);
   ArrayResize(ctx_low, n);
   ArrayResize(ctx_close, n);
   ArrayResize(ctx_spread, n);
   ArraySetAsSeries(ctx_open, true);
   ArraySetAsSeries(ctx_high, true);
   ArraySetAsSeries(ctx_low, true);
   ArraySetAsSeries(ctx_close, true);
   ArraySetAsSeries(ctx_spread, true);

   double pt = (point > 0.0 ? point : 1e-5);
   for(int i=n - 1; i>=0; i--)
   {
      double c_base = FXAI_AuditGetArrayValue(base_close, i, 0.0);
      double prev_base = FXAI_AuditGetArrayValue(base_close, i + 1, c_base);
      double ret = (prev_base > 0.0 ? (c_base - prev_base) / prev_base : 0.0);
      double c = c_base;
      double gap_scale = 0.60;
      double range_scale = 1.0;
      double spread_scale = 1.0;

      if(transform_id == 0)
      {
         c = c_base * (1.0 + 0.60 * ret);
         gap_scale = 0.70;
         range_scale = 1.05;
         spread_scale = 1.05 + 0.20 * FXAI_Clamp(MathAbs(ret) * 4000.0, 0.0, 1.0);
      }
      else if(transform_id == 1)
      {
         c = 0.65 * c_base + 0.35 * prev_base;
         gap_scale = 0.45;
         range_scale = 0.90;
         spread_scale = 0.95 + 0.10 * FXAI_Clamp(MathAbs(ret) * 3000.0, 0.0, 1.0);
      }
      else
      {
         c = c_base * (1.0 - 0.35 * ret);
         gap_scale = -0.25;
         range_scale = 1.08;
         spread_scale = 1.08 + 0.15 * FXAI_Clamp(MathAbs(ret) * 3500.0, 0.0, 1.0);
      }

      if(c <= pt)
         c = MathMax(c_base, pt);

      double prev_ctx = FXAI_AuditGetArrayValue(ctx_close, i + 1, c);
      double open_gap = (prev_base > 0.0 ? (FXAI_AuditGetArrayValue(base_open, i, c_base) - prev_base) / prev_base : 0.0);
      double o = prev_ctx * (1.0 + gap_scale * open_gap);
      if(o <= pt)
         o = prev_ctx;
      if(o <= pt)
         o = c;

      double base_open_i = FXAI_AuditGetArrayValue(base_open, i, c_base);
      double base_high_i = FXAI_AuditGetArrayValue(base_high, i, MathMax(base_open_i, c_base));
      double base_low_i = FXAI_AuditGetArrayValue(base_low, i, MathMin(base_open_i, c_base));
      double base_range = MathMax(base_high_i - base_low_i, pt);
      double base_body_hi = MathMax(base_open_i, c_base);
      double base_body_lo = MathMin(base_open_i, c_base);
      double upper_ratio = MathMax(base_high_i - base_body_hi, 0.0) / base_range;
      double lower_ratio = MathMax(base_body_lo - base_low_i, 0.0) / base_range;
      double scaled_range = range_scale * base_range * MathMax(c / MathMax(c_base, pt), 0.25);
      double upper_wick = MathMax(0.25 * pt, scaled_range * upper_ratio);
      double lower_wick = MathMax(0.25 * pt, scaled_range * lower_ratio);
      double body_hi = MathMax(o, c);
      double body_lo = MathMin(o, c);

      ctx_open[i] = o;
      ctx_high[i] = body_hi + upper_wick;
      ctx_low[i] = MathMax(pt, body_lo - lower_wick);
      ctx_close[i] = c;

      int ref_spread = (i < ArraySize(base_spread) ? base_spread[i] : 1);
      if(ref_spread < 1)
         ref_spread = 1;
      ctx_spread[i] = (int)MathMax(1.0, MathRound((double)ref_spread * spread_scale));
   }
}

void FXAI_AuditAggregateCloseTF(const datetime &src_time_chrono[],
                                const double &src_open_chrono[],
                                const double &src_high_chrono[],
                                const double &src_low_chrono[],
                                const double &src_close_chrono[],
                                const int step,
                                datetime &out_time_series[],
                                double &out_close_series[])
{
   int n = ArraySize(src_close_chrono);
   int bars = (step > 0 ? n / step : 0);
   if(bars <= 0)
   {
      ArrayResize(out_time_series, 0);
      ArrayResize(out_close_series, 0);
      return;
   }

   datetime tmp_time[];
   double tmp_close[];
   ArrayResize(tmp_time, bars);
   ArrayResize(tmp_close, bars);

   for(int b=0; b<bars; b++)
   {
      int start = b * step;
      int end = start + step - 1;
      if(end >= n) end = n - 1;
      tmp_time[b] = src_time_chrono[end];
      tmp_close[b] = src_close_chrono[end];
   }

   FXAI_AuditReverseCloseSeries(tmp_time, tmp_close, out_time_series, out_close_series);
}

void FXAI_AuditSetContextSlotMTFExtras(double &ctx_extra_arr[],
                                       const int sample_idx,
                                       const int top_slot,
                                       const double point,
                                       const double &open_arr[],
                                       const double &high_arr[],
                                       const double &low_arr[],
                                       const double &close_arr[],
                                       const int &spread_arr[])
{
   for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
   {
      double body_bias = 0.0;
      double close_loc = 0.0;
      double range_pressure = 0.0;
      double spread_pressure = 0.0;
      int bars = FXAI_ContextMTFBarsForSlot(tf_slot);
      if(!FXAI_ComputeAggregatedCandleSpreadState(sample_idx,
                                                  bars,
                                                  open_arr,
                                                  high_arr,
                                                  low_arr,
                                                  close_arr,
                                                  spread_arr,
                                                  point,
                                                  body_bias,
                                                  close_loc,
                                                  range_pressure,
                                                  spread_pressure))
         continue;

      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_BODY_BIAS), body_bias);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_CLOSE_LOCATION), close_loc);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_RANGE_PRESSURE), range_pressure);
      FXAI_SetContextExtraValue(ctx_extra_arr, sample_idx, FXAI_ContextSlotMTFExtraIndex(top_slot, tf_slot, (int)FXAI_MTF_SPREAD_PRESSURE), spread_pressure);
   }
}

void FXAI_AuditBuildContextFeatures(const double &main_close[],
                                    const double point,
                                    const double &ctx1_open[],
                                    const double &ctx1_high[],
                                    const double &ctx1_low[],
                                    const double &ctx1_close[],
                                    const int &ctx1_spread[],
                                    const double &ctx2_open[],
                                    const double &ctx2_high[],
                                    const double &ctx2_low[],
                                    const double &ctx2_close[],
                                    const int &ctx2_spread[],
                                    const double &ctx3_open[],
                                    const double &ctx3_high[],
                                    const double &ctx3_low[],
                                    const double &ctx3_close[],
                                    const int &ctx3_spread[],
                                    double &ctx_mean_arr[],
                                    double &ctx_std_arr[],
                                    double &ctx_up_arr[],
                                    double &ctx_extra_arr[])
{
   int n = ArraySize(main_close);
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   ArraySetAsSeries(ctx_mean_arr, true);
   ArraySetAsSeries(ctx_std_arr, true);
   ArraySetAsSeries(ctx_up_arr, true);
   ArrayInitialize(ctx_extra_arr, 0.0);

   for(int i=0; i<n; i++)
   {
      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double ret[3];
      double lag[3];
      ret[0] = FXAI_SafeReturn(ctx1_close, i, i + 1);
      ret[1] = FXAI_SafeReturn(ctx2_close, i, i + 1);
      ret[2] = FXAI_SafeReturn(ctx3_close, i, i + 1);
      lag[0] = FXAI_SafeReturn(ctx1_close, i + 1, i + 2);
      lag[1] = FXAI_SafeReturn(ctx2_close, i + 1, i + 2);
      lag[2] = FXAI_SafeReturn(ctx3_close, i + 1, i + 2);

      double sum = 0.0;
      double sum2 = 0.0;
      int up = 0;
      for(int s=0; s<3; s++)
      {
         sum += ret[s];
         sum2 += ret[s] * ret[s];
         if(ret[s] > 0.0) up++;
      }
      double mean = sum / 3.0;
      double var = sum2 / 3.0 - mean * mean;
      if(var < 0.0) var = 0.0;
      ctx_mean_arr[i] = mean;
      ctx_std_arr[i] = MathSqrt(var);
      ctx_up_arr[i] = (double)up / 3.0;

      double corr1 = FXAI_AuditRollingCorr(main_close, ctx1_close, i, 16);
      double corr2 = FXAI_AuditRollingCorr(main_close, ctx2_close, i, 16);
      double corr3 = FXAI_AuditRollingCorr(main_close, ctx3_close, i, 16);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 0, ret[0]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 1, lag[0]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 2, ret[0] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 3, corr1);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 4, ret[1]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 5, lag[1]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 6, ret[1] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 7, corr2);

      FXAI_SetContextExtraValue(ctx_extra_arr, i, 8, ret[2]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 9, lag[2]);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 10, ret[2] - main_ret);
      FXAI_SetContextExtraValue(ctx_extra_arr, i, 11, corr3);

      double main_vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(main_vol < 1e-6) main_vol = MathAbs(main_ret);
      if(main_vol < 1e-6) main_vol = 1e-4;
      double stab = 1.0;
      for(int s=0; s<3; s++)
         stab -= 0.20 * FXAI_Clamp(MathAbs(ret[s] - lag[s]) / MathMax(main_vol, 1e-4), 0.0, 1.0);
      double lead = 0.0;
      for(int s=0; s<3; s++)
         lead += FXAI_Clamp(MathAbs(lag[s]) / MathMax(main_vol, 1e-4), 0.0, 4.0) / 4.0;
      lead /= 3.0;

      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 0, FXAI_Clamp(mean / MathMax(main_vol, 1e-4), -1.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 1, FXAI_Clamp(stab, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 2, FXAI_Clamp(lead, 0.0, 1.0));
      FXAI_SetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 3, 1.0);

      FXAI_AuditSetContextSlotMTFExtras(ctx_extra_arr, i, 0, point, ctx1_open, ctx1_high, ctx1_low, ctx1_close, ctx1_spread);
      FXAI_AuditSetContextSlotMTFExtras(ctx_extra_arr, i, 1, point, ctx2_open, ctx2_high, ctx2_low, ctx2_close, ctx2_spread);
      FXAI_AuditSetContextSlotMTFExtras(ctx_extra_arr, i, 2, point, ctx3_open, ctx3_high, ctx3_low, ctx3_close, ctx3_spread);
   }
}

bool FXAI_AuditGenerateScenarioSeries(const FXAIAuditScenarioSpec &spec,
                                      const int bars,
                                      const ulong seed,
                                      const double point,
                                      datetime &time_series[],
                                      double &open_series[],
                                      double &high_series[],
                                      double &low_series[],
                                      double &close_series[],
                                      int &spread_series[],
                                      datetime &time_m5[],
                                      double &close_m5[],
                                      int &map_m5[],
                                      datetime &time_m15[],
                                      double &close_m15[],
                                      int &map_m15[],
                                      datetime &time_m30[],
                                      double &close_m30[],
                                      int &map_m30[],
                                      datetime &time_h1[],
                                      double &close_h1[],
                                      int &map_h1[],
                                      double &ctx_mean_arr[],
                                      double &ctx_std_arr[],
                                      double &ctx_up_arr[],
                                      double &ctx_extra_arr[])
{
   if(bars < 512 || point <= 0.0) return false;

   if(spec.id >= 8)
   {
      MqlRates rates_m1[];
      ArraySetAsSeries(rates_m1, true);
      int search_bars = bars * 4;
      if(search_bars < bars + 512) search_bars = bars + 512;
      int got = CopyRates(_Symbol, PERIOD_M1, 1, search_bars, rates_m1);
      if(got < bars + 64) return false;

      int best_start = 0;
      double best_score = -1e18;
      int max_start = got - bars;
      if(max_start < 0) max_start = 0;
      for(int start=0; start<=max_start; start++)
      {
         double net = MathAbs(rates_m1[start].close - rates_m1[start + bars - 1].close);
         double abs_sum = 0.0;
         double vol_sum = 0.0;
         for(int j=start; j<start + bars - 1; j++)
         {
            double step = MathAbs(rates_m1[j].close - rates_m1[j + 1].close);
            abs_sum += step;
            vol_sum += MathAbs(rates_m1[j].high - rates_m1[j].low);
         }
         if(abs_sum < point) abs_sum = point;
         double trendiness = net / abs_sum;
         double avg_range = vol_sum / MathMax((double)(bars - 1), 1.0);
         double score = 0.0;
         if(spec.id == 8)
            score = -0.001 * (double)start;
         else if(spec.id == 9)
            score = trendiness + 0.05 * (avg_range / MathMax(point, 1e-6));
         else if(spec.id == 10)
            score = (1.0 - trendiness) + 0.03 * (avg_range / MathMax(point, 1e-6));
         else if(spec.id == 11)
         {
            MqlDateTime dt0;
            TimeToStruct(rates_m1[start + bars / 2].time, dt0);
            double session_edge = 1.0 - MathMin(MathAbs((double)dt0.hour - 8.0), MathAbs((double)dt0.hour - 16.0)) / 8.0;
            score = 0.60 * session_edge + 0.25 * (avg_range / MathMax(point, 1e-6)) + 0.15 * (1.0 - trendiness);
         }
         else if(spec.id == 12)
         {
            double avg_spread = 0.0;
            double max_spread = 0.0;
            for(int j=start; j<start + bars; j++)
            {
               double sp = (double)rates_m1[j].spread;
               avg_spread += sp;
               if(sp > max_spread) max_spread = sp;
            }
            avg_spread /= MathMax((double)bars, 1.0);
            score = (max_spread - avg_spread) + 0.04 * (avg_range / MathMax(point, 1e-6));
         }
         else if(spec.id == 14)
         {
            double macro_score = FXAI_MacroEventWindowScoreRates(_Symbol, rates_m1, start, bars);
            score = 0.80 * macro_score +
                    0.12 * trendiness +
                    0.08 * FXAI_Clamp(avg_range / MathMax(point, 1e-6), 0.0, 8.0);
         }
         else
         {
            double first_a = rates_m1[start].close;
            double first_b = rates_m1[start + bars / 3].close;
            double last_a = rates_m1[start + (2 * bars) / 3].close;
            double last_b = rates_m1[start + bars - 1].close;
            double first_ret = (first_b > 0.0 ? (first_a - first_b) / first_b : 0.0);
            double last_ret = (last_b > 0.0 ? (last_a - last_b) / last_b : 0.0);
            score = (1.0 - MathAbs(first_ret - last_ret)) + 0.20 * (1.0 - trendiness) + 0.03 * (avg_range / MathMax(point, 1e-6));
         }

         if(score > best_score)
         {
            best_score = score;
            best_start = start;
         }
      }

      MqlRates sel_m1[];
      ArrayResize(sel_m1, bars);
      ArraySetAsSeries(sel_m1, true);
      for(int i=0; i<bars; i++)
         sel_m1[i] = rates_m1[best_start + i];

      FXAI_ExtractRatesCloseTimeSpread(sel_m1, close_series, time_series, spread_series);
      FXAI_ExtractRatesOHLC(sel_m1, open_series, high_series, low_series, close_series);
      if(!FXAI_ValidateM1SeriesBundle(time_series, open_series, high_series, low_series, close_series, spread_series, bars))
         return false;

      int need_m5 = (search_bars / 5) + 220;
      int need_m15 = (search_bars / 15) + 220;
      int need_m30 = (search_bars / 30) + 220;
      int need_h1 = (search_bars / 60) + 220;
      MqlRates rates_tf[];
      ArraySetAsSeries(rates_tf, true);

      int got5 = CopyRates(_Symbol, PERIOD_M5, 1, need_m5, rates_tf);
      if(got5 > 0) FXAI_ExtractRatesCloseTime(rates_tf, close_m5, time_m5);
      else { ArrayResize(close_m5, 0); ArrayResize(time_m5, 0); }
      int got15 = CopyRates(_Symbol, PERIOD_M15, 1, need_m15, rates_tf);
      if(got15 > 0) FXAI_ExtractRatesCloseTime(rates_tf, close_m15, time_m15);
      else { ArrayResize(close_m15, 0); ArrayResize(time_m15, 0); }
      int got30 = CopyRates(_Symbol, PERIOD_M30, 1, need_m30, rates_tf);
      if(got30 > 0) FXAI_ExtractRatesCloseTime(rates_tf, close_m30, time_m30);
      else { ArrayResize(close_m30, 0); ArrayResize(time_m30, 0); }
      int got60 = CopyRates(_Symbol, PERIOD_H1, 1, need_h1, rates_tf);
      if(got60 > 0) FXAI_ExtractRatesCloseTime(rates_tf, close_h1, time_h1);
      else { ArrayResize(close_h1, 0); ArrayResize(time_h1, 0); }

      FXAI_BuildAlignedIndexMap(time_series, time_m5, 2 * PeriodSeconds(PERIOD_M5), map_m5);
      FXAI_BuildAlignedIndexMap(time_series, time_m15, 2 * PeriodSeconds(PERIOD_M15), map_m15);
      FXAI_BuildAlignedIndexMap(time_series, time_m30, 2 * PeriodSeconds(PERIOD_M30), map_m30);
      FXAI_BuildAlignedIndexMap(time_series, time_h1, 2 * PeriodSeconds(PERIOD_H1), map_h1);

      double ctx1_open[];
      double ctx1_high[];
      double ctx1_low[];
      double ctx1_close[];
      int ctx1_spread[];
      double ctx2_open[];
      double ctx2_high[];
      double ctx2_low[];
      double ctx2_close[];
      int ctx2_spread[];
      double ctx3_open[];
      double ctx3_high[];
      double ctx3_low[];
      double ctx3_close[];
      int ctx3_spread[];
      FXAI_AuditDeriveContextSeriesFromBase(point,
                                            open_series,
                                            high_series,
                                            low_series,
                                            close_series,
                                            spread_series,
                                            0,
                                            ctx1_open,
                                            ctx1_high,
                                            ctx1_low,
                                            ctx1_close,
                                            ctx1_spread);
      FXAI_AuditDeriveContextSeriesFromBase(point,
                                            open_series,
                                            high_series,
                                            low_series,
                                            close_series,
                                            spread_series,
                                            1,
                                            ctx2_open,
                                            ctx2_high,
                                            ctx2_low,
                                            ctx2_close,
                                            ctx2_spread);
      FXAI_AuditDeriveContextSeriesFromBase(point,
                                            open_series,
                                            high_series,
                                            low_series,
                                            close_series,
                                            spread_series,
                                            2,
                                            ctx3_open,
                                            ctx3_high,
                                            ctx3_low,
                                            ctx3_close,
                                            ctx3_spread);
      FXAI_AuditBuildContextFeatures(close_series,
                                     point,
                                     ctx1_open,
                                     ctx1_high,
                                     ctx1_low,
                                     ctx1_close,
                                     ctx1_spread,
                                     ctx2_open,
                                     ctx2_high,
                                     ctx2_low,
                                     ctx2_close,
                                     ctx2_spread,
                                     ctx3_open,
                                     ctx3_high,
                                     ctx3_low,
                                     ctx3_close,
                                     ctx3_spread,
                                     ctx_mean_arr,
                                     ctx_std_arr,
                                     ctx_up_arr,
                                     ctx_extra_arr);
      return true;
   }

   CFXAIAuditRng rng;
   rng.Seed(seed ^ ((ulong)(spec.id + 1) * (ulong)2654435761));

   datetime chrono_time[];
   double chrono_open[];
   double chrono_high[];
   double chrono_low[];
   double chrono_close[];
   int chrono_spread[];
   double chrono_ctx1_open[];
   double chrono_ctx1_high[];
   double chrono_ctx1_low[];
   double chrono_ctx1_close[];
   int chrono_ctx1_spread[];
   double chrono_ctx2_open[];
   double chrono_ctx2_high[];
   double chrono_ctx2_low[];
   double chrono_ctx2_close[];
   int chrono_ctx2_spread[];
   double chrono_ctx3_open[];
   double chrono_ctx3_high[];
   double chrono_ctx3_low[];
   double chrono_ctx3_close[];
   int chrono_ctx3_spread[];
   ArrayResize(chrono_time, bars);
   ArrayResize(chrono_open, bars);
   ArrayResize(chrono_high, bars);
   ArrayResize(chrono_low, bars);
   ArrayResize(chrono_close, bars);
   ArrayResize(chrono_spread, bars);
   ArrayResize(chrono_ctx1_open, bars);
   ArrayResize(chrono_ctx1_high, bars);
   ArrayResize(chrono_ctx1_low, bars);
   ArrayResize(chrono_ctx1_close, bars);
   ArrayResize(chrono_ctx1_spread, bars);
   ArrayResize(chrono_ctx2_open, bars);
   ArrayResize(chrono_ctx2_high, bars);
   ArrayResize(chrono_ctx2_low, bars);
   ArrayResize(chrono_ctx2_close, bars);
   ArrayResize(chrono_ctx2_spread, bars);
   ArrayResize(chrono_ctx3_open, bars);
   ArrayResize(chrono_ctx3_high, bars);
   ArrayResize(chrono_ctx3_low, bars);
   ArrayResize(chrono_ctx3_close, bars);
   ArrayResize(chrono_ctx3_spread, bars);

   datetime t0 = D'2024.01.01 00:00';
   double anchor = 1.10000;
   double prev = anchor;
   double prev_sigma = spec.sigma_per_bar;
   double ctx_prev1 = anchor * 0.97;
   double ctx_prev2 = anchor * 1.03;
   double ctx_prev3 = anchor * 0.91;

   for(int k=0; k<bars; k++)
   {
      double sigma = spec.sigma_per_bar;
      if(spec.vol_cluster > 0.0)
      {
         sigma = MathMax(1e-6, spec.vol_cluster * prev_sigma + (1.0 - spec.vol_cluster) * spec.sigma_per_bar * (0.5 + 1.5 * rng.NextUnit()));
         prev_sigma = sigma;
      }

      double drift = spec.drift_per_bar;
      if(spec.id == 7 && k > bars / 2)
      {
         drift = -spec.drift_per_bar * 1.25;
         sigma *= 1.8;
      }

      double ret = drift + sigma * rng.NextNormal();
      if(spec.mean_revert_strength > 0.0)
         ret += spec.mean_revert_strength * ((anchor - prev) / MathMax(prev, point));
      if(spec.spike_prob > 0.0 && rng.NextUnit() < spec.spike_prob)
         ret += spec.spike_scale * sigma * rng.NextNormal();

      if(spec.id == 5)
         ret = MathAbs(drift) + 0.10 * sigma * MathAbs(rng.NextNormal());
      else if(spec.id == 6)
         ret = -(MathAbs(drift) + 0.10 * sigma * MathAbs(rng.NextNormal()));

      double op = prev;
      double cl = prev * (1.0 + ret);
      if(cl <= point) cl = prev + 10.0 * point;
      double wick = MathMax(MathAbs(cl - op) * (0.30 + 0.40 * rng.NextUnit()), point * (2.0 + 8.0 * rng.NextUnit()));
      double hi = MathMax(op, cl) + wick;
      double lo = MathMin(op, cl) - wick;
      if(lo <= point) lo = point;

      chrono_time[k] = t0 + (datetime)(60 * k);
      chrono_open[k] = op;
      chrono_high[k] = hi;
      chrono_low[k] = lo;
      chrono_close[k] = cl;
      chrono_spread[k] = (int)MathMax(1.0, MathRound(spec.spread_points * (0.85 + 0.30 * rng.NextUnit())));

      double ctx_noise1 = 0.35 * sigma * rng.NextNormal();
      double ctx_noise2 = 0.45 * sigma * rng.NextNormal();
      double ctx_noise3 = 0.55 * sigma * rng.NextNormal();
      double ctx_ret1 = 0.80 * ret + ctx_noise1;
      double ctx_ret2 = -0.45 * ret + ctx_noise2;
      double ctx_ret3 = 0.35 * ret + 0.50 * ctx_noise3;

      double ctx_op1 = ctx_prev1;
      double ctx_op2 = ctx_prev2;
      double ctx_op3 = ctx_prev3;
      double ctx_cl1 = ctx_prev1 * (1.0 + ctx_ret1);
      double ctx_cl2 = ctx_prev2 * (1.0 + ctx_ret2);
      double ctx_cl3 = ctx_prev3 * (1.0 + ctx_ret3);
      if(ctx_cl1 <= point) ctx_cl1 = MathMax(ctx_op1, op);
      if(ctx_cl2 <= point) ctx_cl2 = MathMax(ctx_op2, op);
      if(ctx_cl3 <= point) ctx_cl3 = MathMax(ctx_op3, op);

      double ctx_wick1 = MathMax(MathAbs(ctx_cl1 - ctx_op1) * (0.25 + 0.35 * rng.NextUnit()), point * (1.5 + 5.0 * rng.NextUnit()));
      double ctx_wick2 = MathMax(MathAbs(ctx_cl2 - ctx_op2) * (0.25 + 0.35 * rng.NextUnit()), point * (1.5 + 5.5 * rng.NextUnit()));
      double ctx_wick3 = MathMax(MathAbs(ctx_cl3 - ctx_op3) * (0.25 + 0.35 * rng.NextUnit()), point * (1.5 + 6.0 * rng.NextUnit()));

      chrono_ctx1_open[k] = ctx_op1;
      chrono_ctx1_high[k] = MathMax(ctx_op1, ctx_cl1) + ctx_wick1;
      chrono_ctx1_low[k] = MathMax(point, MathMin(ctx_op1, ctx_cl1) - ctx_wick1);
      chrono_ctx1_close[k] = ctx_cl1;
      chrono_ctx1_spread[k] = (int)MathMax(1.0, MathRound(spec.spread_points *
                                                           (0.85 + 0.25 * rng.NextUnit() +
                                                            0.18 * FXAI_Clamp(MathAbs(ctx_ret1) / MathMax(sigma, 1e-6), 0.0, 2.0))));

      chrono_ctx2_open[k] = ctx_op2;
      chrono_ctx2_high[k] = MathMax(ctx_op2, ctx_cl2) + ctx_wick2;
      chrono_ctx2_low[k] = MathMax(point, MathMin(ctx_op2, ctx_cl2) - ctx_wick2);
      chrono_ctx2_close[k] = ctx_cl2;
      chrono_ctx2_spread[k] = (int)MathMax(1.0, MathRound(spec.spread_points *
                                                           (0.80 + 0.20 * rng.NextUnit() +
                                                            0.12 * FXAI_Clamp(MathAbs(ctx_ret2) / MathMax(sigma, 1e-6), 0.0, 2.0))));

      chrono_ctx3_open[k] = ctx_op3;
      chrono_ctx3_high[k] = MathMax(ctx_op3, ctx_cl3) + ctx_wick3;
      chrono_ctx3_low[k] = MathMax(point, MathMin(ctx_op3, ctx_cl3) - ctx_wick3);
      chrono_ctx3_close[k] = ctx_cl3;
      chrono_ctx3_spread[k] = (int)MathMax(1.0, MathRound(spec.spread_points *
                                                           (0.90 + 0.25 * rng.NextUnit() +
                                                            0.16 * FXAI_Clamp(MathAbs(ctx_ret3) / MathMax(sigma, 1e-6), 0.0, 2.0))));

      ctx_prev1 = ctx_cl1;
      ctx_prev2 = ctx_cl2;
      ctx_prev3 = ctx_cl3;

      prev = cl;
   }

   FXAI_AuditReverseChronoToSeries(chrono_time,
                                   chrono_open,
                                   chrono_high,
                                   chrono_low,
                                   chrono_close,
                                   chrono_spread,
                                   time_series,
                                   open_series,
                                   high_series,
                                   low_series,
                                   close_series,
                                   spread_series);

   datetime ctx1_time[];
   datetime ctx2_time[];
   datetime ctx3_time[];
   double ctx1_series[];
   double ctx2_series[];
   double ctx3_series[];
   double ctx1_open[];
   double ctx1_high[];
   double ctx1_low[];
   int ctx1_spread[];
   double ctx2_open[];
   double ctx2_high[];
   double ctx2_low[];
   int ctx2_spread[];
   double ctx3_open[];
   double ctx3_high[];
   double ctx3_low[];
   int ctx3_spread[];
   FXAI_AuditReverseChronoToSeries(chrono_time,
                                   chrono_ctx1_open,
                                   chrono_ctx1_high,
                                   chrono_ctx1_low,
                                   chrono_ctx1_close,
                                   chrono_ctx1_spread,
                                   ctx1_time,
                                   ctx1_open,
                                   ctx1_high,
                                   ctx1_low,
                                   ctx1_series,
                                   ctx1_spread);
   FXAI_AuditReverseChronoToSeries(chrono_time,
                                   chrono_ctx2_open,
                                   chrono_ctx2_high,
                                   chrono_ctx2_low,
                                   chrono_ctx2_close,
                                   chrono_ctx2_spread,
                                   ctx2_time,
                                   ctx2_open,
                                   ctx2_high,
                                   ctx2_low,
                                   ctx2_series,
                                   ctx2_spread);
   FXAI_AuditReverseChronoToSeries(chrono_time,
                                   chrono_ctx3_open,
                                   chrono_ctx3_high,
                                   chrono_ctx3_low,
                                   chrono_ctx3_close,
                                   chrono_ctx3_spread,
                                   ctx3_time,
                                   ctx3_open,
                                   ctx3_high,
                                   ctx3_low,
                                   ctx3_series,
                                   ctx3_spread);

   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 5, time_m5, close_m5);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 15, time_m15, close_m15);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 30, time_m30, close_m30);
   FXAI_AuditAggregateCloseTF(chrono_time, chrono_open, chrono_high, chrono_low, chrono_close, 60, time_h1, close_h1);

   FXAI_BuildAlignedIndexMap(time_series, time_m5, 2 * PeriodSeconds(PERIOD_M5), map_m5);
   FXAI_BuildAlignedIndexMap(time_series, time_m15, 2 * PeriodSeconds(PERIOD_M15), map_m15);
   FXAI_BuildAlignedIndexMap(time_series, time_m30, 2 * PeriodSeconds(PERIOD_M30), map_m30);
   FXAI_BuildAlignedIndexMap(time_series, time_h1, 2 * PeriodSeconds(PERIOD_H1), map_h1);

   FXAI_AuditBuildContextFeatures(close_series,
                                  point,
                                  ctx1_open,
                                  ctx1_high,
                                  ctx1_low,
                                  ctx1_series,
                                  ctx1_spread,
                                  ctx2_open,
                                  ctx2_high,
                                  ctx2_low,
                                  ctx2_series,
                                  ctx2_spread,
                                  ctx3_open,
                                  ctx3_high,
                                  ctx3_low,
                                  ctx3_series,
                                  ctx3_spread,
                                  ctx_mean_arr,
                                  ctx_std_arr,
                                  ctx_up_arr,
                                  ctx_extra_arr);
   return true;
}


#endif // __FXAI_AUDIT_SCENARIOS_MQH__
