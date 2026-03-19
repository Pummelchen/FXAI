#ifndef __FXAI_FEATURE_BUILD_MQH__
#define __FXAI_FEATURE_BUILD_MQH__

double FXAI_CyclicHourPulse(const double hour_value,
                            const double center_hour,
                            const double radius_hours)
{
   double radius = (radius_hours > 1e-6 ? radius_hours : 1.0);
   double d = MathAbs(hour_value - center_hour);
   if(d > 12.0)
      d = 24.0 - d;
   return FXAI_Clamp(1.0 - d / radius, 0.0, 1.0);
}

double FXAI_RolloverProximityFeature(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   double hour_value = (double)dt.hour + (double)dt.min / 60.0;
   return FXAI_CyclicHourPulse(hour_value, 23.0, 3.0);
}

double FXAI_SessionTransitionFeature(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   double hour_value = (double)dt.hour + (double)dt.min / 60.0;
   double asia_to_eu = FXAI_CyclicHourPulse(hour_value, 7.0, 2.5);
   double eu_to_us = FXAI_CyclicHourPulse(hour_value, 13.0, 2.5);
   double us_to_roll = FXAI_CyclicHourPulse(hour_value, 21.0, 2.5);
   return FXAI_Clamp(0.60 * asia_to_eu + 0.80 * eu_to_us - 0.70 * us_to_roll, -1.0, 1.0);
}

double FXAI_SessionOverlapFeature(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   double hour_value = (double)dt.hour + (double)dt.min / 60.0;
   double london_open = FXAI_CyclicHourPulse(hour_value, 8.0, 3.0);
   double ny_open = FXAI_CyclicHourPulse(hour_value, 13.5, 3.0);
   double overlap = MathMin(london_open, ny_open);
   return FXAI_Clamp(0.70 * overlap + 0.30 * FXAI_CyclicHourPulse(hour_value, 15.0, 2.0), 0.0, 1.0);
}

void FXAI_GetSwapCarryFeatures(const string symbol,
                               const datetime sample_time,
                               const double momentum_signal,
                               const double ctx_signal,
                               double &triple_swap_bias,
                               double &swap_long_pressure,
                               double &swap_short_pressure,
                               double &carry_alignment)
{
   long triple_roll = 3;
   if(!SymbolInfoInteger(symbol, SYMBOL_SWAP_ROLLOVER3DAYS, triple_roll))
      triple_roll = 3;

   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   int triple_dow = (int)triple_roll;
   if(triple_dow < 0 || triple_dow > 6)
      triple_dow = 3;
   double day_bias = (dt.day_of_week == triple_dow ? 1.0 : -0.25);
   double roll_bias = FXAI_RolloverProximityFeature(sample_time);
   triple_swap_bias = FXAI_Clamp(day_bias * (0.35 + 0.65 * roll_bias), -1.0, 1.0);

   double swap_long = SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG);
   double swap_short = SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT);
   double scale = MathMax(MathMax(MathAbs(swap_long), MathAbs(swap_short)), 0.25);
   swap_long_pressure = FXAI_Clamp(swap_long / scale, -4.0, 4.0);
   swap_short_pressure = FXAI_Clamp(swap_short / scale, -4.0, 4.0);

   double carry_skew = (swap_long - swap_short) / scale;
   double directional = FXAI_Clamp(0.70 * momentum_signal + 0.30 * ctx_signal, -4.0, 4.0);
   carry_alignment = FXAI_Clamp(carry_skew * directional, -6.0, 6.0);
}

double FXAI_LocalFeatureFamilyDrift(const double &features[])
{
   double family_sum[FXAI_FEATURE_GROUP_COUNT];
   int family_cnt[FXAI_FEATURE_GROUP_COUNT];
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
   {
      family_sum[g] = 0.0;
      family_cnt[g] = 0;
   }

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      if(f == 79)
         continue;
      int g = FXAI_GetFeatureGroupForIndex(f);
      if(g < 0 || g >= FXAI_FEATURE_GROUP_COUNT)
         continue;
      family_sum[g] += features[f];
      family_cnt[g]++;
   }

   double family_mean[FXAI_FEATURE_GROUP_COUNT];
   for(int g=0; g<FXAI_FEATURE_GROUP_COUNT; g++)
      family_mean[g] = (family_cnt[g] > 0 ? family_sum[g] / (double)family_cnt[g] : 0.0);

   double drift = 0.0;
   drift += MathAbs(family_mean[(int)FXAI_FEAT_GROUP_PRICE] - family_mean[(int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME]);
   drift += MathAbs(family_mean[(int)FXAI_FEAT_GROUP_CONTEXT] - family_mean[(int)FXAI_FEAT_GROUP_COST]);
   drift += 0.50 * MathAbs(family_mean[(int)FXAI_FEAT_GROUP_MICROSTRUCTURE] - family_mean[(int)FXAI_FEAT_GROUP_FILTERS]);
   return FXAI_Clamp(drift / 3.0, 0.0, 6.0);
}

void FXAI_GetSpreadStateFeatures(const int i,
                                 const double spread_points,
                                 const int &main_spread_arr[],
                                 double &avg_spread20,
                                 double &spread_prev,
                                 double &spread_z20,
                                 double &spread_vol_ratio20,
                                 double &spread_rank20)
{
   double cur_spread = MathMax(spread_points, 0.0);
   avg_spread20 = MathMax(cur_spread, 0.25);
   spread_prev = cur_spread;
   double sum = cur_spread;
   double sum2 = cur_spread * cur_spread;
   int used = 1;
   int rank_le = 1;

   for(int k=1; k<20; k++)
   {
      int ik = i + k;
      if(ik < 0 || ik >= ArraySize(main_spread_arr))
         break;

      double v = MathMax((double)main_spread_arr[ik], 0.0);
      if(k == 1)
         spread_prev = v;
      sum += v;
      sum2 += v * v;
      used++;
      if(v <= cur_spread)
         rank_le++;
   }

   avg_spread20 = MathMax(sum / (double)MathMax(used, 1), 0.25);
   double var = (sum2 / (double)MathMax(used, 1)) - avg_spread20 * avg_spread20;
   if(var < 0.0)
      var = 0.0;
   double spread_std20 = MathSqrt(var);
   spread_z20 = (cur_spread - avg_spread20) / MathMax(spread_std20, 0.25);
   spread_vol_ratio20 = spread_std20 / MathMax(avg_spread20, 0.25);
   spread_rank20 = FXAI_Clamp(2.0 * ((double)rank_le / (double)MathMax(used, 1)) - 1.0, -1.0, 1.0);
}

bool FXAI_ComputeFeatureVector(const int i,
                              const string symbol,
                              const double spread_points,
                              const datetime &main_t1[],
                              const double &main_o1[],
                              const double &main_h1_ohlc[],
                              const double &main_l1[],
                              const double &main_m1[],
                              const int &main_spread_arr[],
                              const datetime &main_t5[],
                              const double &main_m5[],
                              const int &map_m5[],
                              const datetime &main_t15[],
                              const double &main_m15[],
                              const int &map_m15[],
                              const datetime &main_t30[],
                              const double &main_m30[],
                              const int &map_m30[],
                              const datetime &main_h1_t[],
                              const double &main_h1[],
                              const int &map_h1[],
                              const double ctx_ret_mean,
                              const double ctx_ret_std,
                              const double ctx_up_ratio,
                              const double &ctx_extra_arr[],
                              const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                              double &features[])
{
   int n = ArraySize(main_m1);
   if(n < 40) return false;
   if(i < 0) return false;
   if(i + 10 >= n) return false;
   if(ArraySize(main_t1) != n) return false;
   if(ArraySize(main_o1) != n) return false;
   if(ArraySize(main_h1_ohlc) != n) return false;
   if(ArraySize(main_l1) != n) return false;
   if(ArraySize(main_spread_arr) != n) return false;
   if(i >= ArraySize(main_t1)) return false;

   datetime t_ref = main_t1[i];
   if(t_ref <= 0) return false;

   for(int f=0; f<FXAI_AI_FEATURES; f++) features[f] = 0.0;

   double c = main_m1[i];
   double c1 = main_m1[i + 1];
   double c3 = main_m1[i + 3];
   double c5 = main_m1[i + 5];
   if(c1 <= 0.0 || c3 <= 0.0 || c5 <= 0.0) return false;

   // Default volatility unit: rolling absolute return (past-only for as-series arrays).
   double vol_unit = FXAI_RollingAbsReturn(main_m1, i, 20);
   if(norm_method == FXAI_NORM_VOL_STD_RETURNS)
   {
      double ru = FXAI_RollingReturnStd(main_m1, i, 20);
      if(ru > 1e-8) vol_unit = ru;
   }

   // ATR/NATR-based volatility-unit normalization option.
   double atr14 = FXAI_ATRAt(main_h1_ohlc, main_l1, main_m1, i, 14);
   if(norm_method == FXAI_NORM_ATR_NATR_UNIT && c > 0.0)
   {
      double atr_unit = atr14 / c;
      if(atr_unit > 1e-8) vol_unit = atr_unit;
   }
   if(vol_unit < 1e-6) vol_unit = 1e-6;
   double spread_norm = 1.0 + (10000.0 * vol_unit);
   if(spread_norm < 1.0) spread_norm = 1.0;

   // M1 core features
   features[0] = ((c - c1) / c1) / vol_unit;
   features[1] = ((c - c3) / c3) / vol_unit;
   features[2] = ((c - c5) / c5) / vol_unit;
   features[3] = FXAI_NormalizedSlope(main_m1, i, 10) / vol_unit;

   double sumy = 0.0;
   for(int k=0; k<10; k++) sumy += main_m1[i + k];
   double mean = sumy / 10.0;

   double var = 0.0;
   for(int k=0; k<10; k++)
   {
      double d = main_m1[i + k] - mean;
      var += d * d;
   }
   double std = MathSqrt(var / 10.0);
   features[4] = (std > 0.0 ? ((c - mean) / std) : 0.0);

   double rsum = 0.0;
   double rsum2 = 0.0;
   for(int k=0; k<10; k++)
   {
      double r = FXAI_SafeReturn(main_m1, i + k, i + k + 1);
      rsum += r;
      rsum2 += r * r;
   }
   double rmean = rsum / 10.0;
   double rvar = (rsum2 / 10.0) - (rmean * rmean);
   features[5] = (rvar > 0.0 ? MathSqrt(rvar) / vol_unit : 0.0);

   features[6] = spread_points / spread_norm;

   int lag_m5 = 2 * PeriodSeconds(PERIOD_M5);
   int lag_m15 = 2 * PeriodSeconds(PERIOD_M15);
   int lag_m30 = 2 * PeriodSeconds(PERIOD_M30);
   int lag_h1 = 2 * PeriodSeconds(PERIOD_H1);
   if(lag_m5 <= 0) lag_m5 = 600;
   if(lag_m15 <= 0) lag_m15 = 1800;
   if(lag_m30 <= 0) lag_m30 = 3600;
   if(lag_h1 <= 0) lag_h1 = 7200;

   // Multi-timeframe trend/return context aligned by timestamp
   int i5 = -1;
   int i15 = -1;
   int i30 = -1;
   int i60 = -1;
   if(i >= 0 && i < ArraySize(map_m5)) i5 = map_m5[i];
   if(i >= 0 && i < ArraySize(map_m15)) i15 = map_m15[i];
   if(i >= 0 && i < ArraySize(map_m30)) i30 = map_m30[i];
   if(i >= 0 && i < ArraySize(map_h1)) i60 = map_h1[i];

   if(i5 < 0) i5 = FXAI_FindAlignedIndex(main_t5, t_ref, lag_m5);
   if(i15 < 0) i15 = FXAI_FindAlignedIndex(main_t15, t_ref, lag_m15);
   if(i30 < 0) i30 = FXAI_FindAlignedIndex(main_t30, t_ref, lag_m30);
   if(i60 < 0) i60 = FXAI_FindAlignedIndex(main_h1_t, t_ref, lag_h1);

   double w5 = FXAI_AlignedFreshnessWeight(main_t5, i5, t_ref, lag_m5);
   double w15 = FXAI_AlignedFreshnessWeight(main_t15, i15, t_ref, lag_m15);
   double w60 = FXAI_AlignedFreshnessWeight(main_h1_t, i60, t_ref, lag_h1);

   double miss_penalty = -0.25;
   double ret5 = FXAI_SafeReturn(main_m5, i5, i5 + 1) / vol_unit;
   double ret15 = FXAI_SafeReturn(main_m15, i15, i15 + 1) / vol_unit;
   double ret60 = FXAI_SafeReturn(main_h1, i60, i60 + 1) / vol_unit;

   features[7] = (w5 * ret5) + ((1.0 - w5) * miss_penalty);
   features[8] = (w15 * ret15) + ((1.0 - w15) * miss_penalty);
   features[9] = (w60 * ret60) + ((1.0 - w60) * miss_penalty);

   // Cross-symbol context (dynamic list, pre-aggregated in caller)
   // [10] mean return, [11] return dispersion, [12] up-breadth in [-1, +1]
   features[10] = ctx_ret_mean / vol_unit;
   features[11] = ctx_ret_std / vol_unit;
   features[12] = FXAI_Clamp((ctx_up_ratio - 0.5) * 2.0, -1.0, 1.0);

   // MTF slopes on aligned anchor bars
   double sl5 = FXAI_NormalizedSlope(main_m5, i5, 6) / vol_unit;
   double sl60 = FXAI_NormalizedSlope(main_h1, i60, 6) / vol_unit;
   features[13] = (w5 * sl5) + ((1.0 - w5) * miss_penalty);
   features[14] = (w60 * sl60) + ((1.0 - w60) * miss_penalty);

   MqlDateTime dt;
   TimeToStruct(t_ref, dt);
   int weekday = dt.day_of_week;
   if(weekday < 1) weekday = 1;
   if(weekday > 5) weekday = 5;
   int hh = dt.hour;
   int mm = dt.min;
   if(hh < 0) hh = 0;
   if(hh > 23) hh = 23;
   if(mm < 0) mm = 0;
   if(mm > 59) mm = 59;

   // Time features (normalized from requested MT5 ranges)
   features[15] = ((double)weekday - 3.0) / 2.0;
   features[16] = ((double)hh - 11.5) / 11.5;
   features[17] = ((double)mm - 29.5) / 29.5;

   // OHLC features from current M1 bar
   double o = main_o1[i];
   double h = main_h1_ohlc[i];
   double l = main_l1[i];
   if(norm_method == FXAI_NORM_CANDLE_GEOMETRY)
   {
      FXAI_CandleGeometryNormalize(o, h, l, c, c1, 1e-8,
                                   features[18], features[19], features[20], features[21]);
   }
   else
   {
      features[18] = FXAI_MAEdgeFeature(c, o, vol_unit);
      features[19] = FXAI_MAEdgeFeature(h, c, vol_unit);
      features[20] = FXAI_MAEdgeFeature(c, l, vol_unit);
      features[21] = (c > 0.0 ? ((h - l) / c) / vol_unit : 0.0);
   }

   // Multi-timeframe SMA distance features (100/200)
   double ma_m5_100 = FXAI_SMAAt(main_m5, i5, 100);
   double ma_m5_200 = FXAI_SMAAt(main_m5, i5, 200);
   double ma_m15_100 = FXAI_SMAAt(main_m15, i15, 100);
   double ma_m15_200 = FXAI_SMAAt(main_m15, i15, 200);
   double ma_m30_100 = FXAI_SMAAt(main_m30, i30, 100);
   double ma_m30_200 = FXAI_SMAAt(main_m30, i30, 200);
   double ma_h1_100 = FXAI_SMAAt(main_h1, i60, 100);
   double ma_h1_200 = FXAI_SMAAt(main_h1, i60, 200);

   features[22] = FXAI_MAEdgeFeature(c, ma_m5_100, vol_unit);
   features[23] = FXAI_MAEdgeFeature(c, ma_m5_200, vol_unit);
   features[24] = FXAI_MAEdgeFeature(c, ma_m15_100, vol_unit);
   features[25] = FXAI_MAEdgeFeature(c, ma_m15_200, vol_unit);
   features[26] = FXAI_MAEdgeFeature(c, ma_m30_100, vol_unit);
   features[27] = FXAI_MAEdgeFeature(c, ma_m30_200, vol_unit);
   features[28] = FXAI_MAEdgeFeature(c, ma_h1_100, vol_unit);
   features[29] = FXAI_MAEdgeFeature(c, ma_h1_200, vol_unit);

   // Multi-timeframe EMA distance features (100/200)
   double ema_m5_100 = FXAI_EMAAt(main_m5, i5, 100);
   double ema_m5_200 = FXAI_EMAAt(main_m5, i5, 200);
   double ema_m15_100 = FXAI_EMAAt(main_m15, i15, 100);
   double ema_m15_200 = FXAI_EMAAt(main_m15, i15, 200);
   double ema_m30_100 = FXAI_EMAAt(main_m30, i30, 100);
   double ema_m30_200 = FXAI_EMAAt(main_m30, i30, 200);
   double ema_h1_100 = FXAI_EMAAt(main_h1, i60, 100);
   double ema_h1_200 = FXAI_EMAAt(main_h1, i60, 200);

   features[30] = FXAI_MAEdgeFeature(c, ema_m5_100, vol_unit);
   features[31] = FXAI_MAEdgeFeature(c, ema_m5_200, vol_unit);
   features[32] = FXAI_MAEdgeFeature(c, ema_m15_100, vol_unit);
   features[33] = FXAI_MAEdgeFeature(c, ema_m15_200, vol_unit);
   features[34] = FXAI_MAEdgeFeature(c, ema_m30_100, vol_unit);
   features[35] = FXAI_MAEdgeFeature(c, ema_m30_200, vol_unit);
   features[36] = FXAI_MAEdgeFeature(c, ema_h1_100, vol_unit);
   features[37] = FXAI_MAEdgeFeature(c, ema_h1_200, vol_unit);

   // Additional volatility/momentum features
   double qsdema_100 = FXAI_QSDEMAAt(main_m1, i, 100);
   double qsdema_200 = FXAI_QSDEMAAt(main_m1, i, 200);
   features[38] = FXAI_MAEdgeFeature(c, qsdema_100, vol_unit);
   features[39] = FXAI_MAEdgeFeature(c, qsdema_200, vol_unit);

   double rsi14 = FXAI_RSIAt(main_m1, i, 14);
   features[40] = (rsi14 - 50.0) / 50.0;

   features[41] = (c > 0.0 ? ((atr14 / c) / vol_unit) : 0.0);

   double natr14 = (c > 0.0 ? (100.0 * atr14 / c) : 0.0);
   features[42] = natr14;

   double parkinson20 = FXAI_ParkinsonVolAt(main_h1_ohlc, main_l1, i, 20);
   features[43] = (vol_unit > 0.0 ? (parkinson20 / vol_unit) : 0.0);

   double rs20 = FXAI_RogersSatchellVolAt(main_o1, main_h1_ohlc, main_l1, main_m1, i, 20);
   features[44] = (vol_unit > 0.0 ? (rs20 / vol_unit) : 0.0);

   double gk20 = FXAI_GarmanKlassVolAt(main_o1, main_h1_ohlc, main_l1, main_m1, i, 20);
   features[45] = (vol_unit > 0.0 ? (gk20 / vol_unit) : 0.0);

   double med21 = FXAI_RollingMedianAt(main_m1, i, 21);
   features[46] = FXAI_MAEdgeFeature(c, med21, vol_unit);
   double mad21 = FXAI_RollingMADAt(main_m1, i, 21, med21);
   double hampel_denom = 1.4826 * mad21;
   if(hampel_denom < 1e-8) hampel_denom = 1e-8;
   features[47] = (c - med21) / hampel_denom;

   double kalman34 = FXAI_KalmanEstimateAt(main_m1, i, 34);
   features[48] = FXAI_MAEdgeFeature(c, kalman34, vol_unit);

   double ss20 = FXAI_EhlersSuperSmootherAt(main_m1, i, 20);
   features[49] = FXAI_MAEdgeFeature(c, ss20, vol_unit);

   // Detailed cross-symbol context: per-symbol aligned returns, lagged returns,
   // relative-strength residuals, and rolling correlation to the main symbol.
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      int base_f = 50 + slot * 4;
      double ctx_ret = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 0, 0.0);
      double ctx_lag = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 1, 0.0);
      double ctx_rel = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 2, 0.0);
      double ctx_corr = FXAI_GetContextExtraValue(ctx_extra_arr, i, slot * 4 + 3, 0.0);

      features[base_f + 0] = ctx_ret / vol_unit;
      features[base_f + 1] = ctx_lag / vol_unit;
      features[base_f + 2] = ctx_rel / vol_unit;
      features[base_f + 3] = ctx_corr;
   }

   double shared_util = FXAI_GetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 0, 0.0);
   double shared_stability = FXAI_GetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 1, 0.5);
   double shared_lead = FXAI_GetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 2, 0.5);
   double shared_coverage = FXAI_GetContextExtraValue(ctx_extra_arr, i, FXAI_CONTEXT_SHARED_OFFSET + 3, 0.0);
   features[62] = FXAI_Clamp(shared_util, -1.0, 1.0);
   features[63] = FXAI_Clamp(2.0 * shared_stability - 1.0, -1.0, 1.0);
   features[64] = FXAI_Clamp(2.0 * shared_lead - 1.0, -1.0, 1.0);
   features[65] = FXAI_Clamp(2.0 * shared_coverage - 1.0, -1.0, 1.0);

   double point_value = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point_value <= 0.0)
      point_value = (_Point > 0.0 ? _Point : 1.0);
   double bar_range = MathMax(h - l, point_value);
   double close_loc = ((c - l) - (h - c)) / bar_range;
   double wick_up = MathMax(0.0, h - MathMax(c, o));
   double wick_dn = MathMax(0.0, MathMin(c, o) - l);
   double wick_imbalance = (wick_up - wick_dn) / bar_range;
   double body_eff = MathAbs(c - o) / bar_range;

   double avg_spread20 = MathMax(spread_points, 0.25);
   double avg_range_points20 = MathMax(bar_range / point_value, 0.25);
   double spread_prev = spread_points;
   double spread_z20 = 0.0;
   double spread_vol_ratio20 = 0.0;
   double spread_rank20 = 0.0;
   FXAI_GetSpreadStateFeatures(i,
                               spread_points,
                               main_spread_arr,
                               avg_spread20,
                               spread_prev,
                               spread_z20,
                               spread_vol_ratio20,
                               spread_rank20);
   int range_used = 0;
   for(int k=0; k<20; k++)
   {
      int ik = i + k;
      if(ik >= 0 && ik < ArraySize(main_h1_ohlc) && ik < ArraySize(main_l1))
      {
         avg_range_points20 += MathMax(0.0, (main_h1_ohlc[ik] - main_l1[ik]) / point_value);
         range_used++;
      }
   }
   avg_range_points20 /= (double)MathMax(range_used + 1, 1);

   double bar_range_points = MathMax(0.0, (h - l) / point_value);
   double spread_shock = (spread_points / MathMax(avg_spread20, 0.25)) - 1.0;
   double spread_accel = (spread_points - spread_prev) / MathMax(avg_spread20, 0.25);
   double spread_to_range = spread_points / MathMax(bar_range_points + 0.25, 0.25);
   double range_pressure = (bar_range_points / MathMax(avg_range_points20, 0.25)) - 1.0;
   double micro_trend = close_loc * (0.60 + 0.40 * body_eff);

   features[66] = FXAI_Clamp(close_loc, -1.2, 1.2);
   features[67] = FXAI_Clamp(wick_imbalance, -1.2, 1.2);
   features[68] = FXAI_Clamp(spread_shock, -4.0, 8.0);
   features[69] = FXAI_Clamp(spread_accel, -4.0, 8.0);
   features[70] = FXAI_Clamp(spread_to_range, 0.0, 8.0);
   features[71] = FXAI_Clamp(micro_trend * (1.0 + 0.35 * range_pressure), -6.0, 6.0);

   features[72] = FXAI_SessionTransitionFeature(t_ref);
   features[73] = FXAI_SessionOverlapFeature(t_ref);
   features[74] = FXAI_RolloverProximityFeature(t_ref);
   FXAI_GetSwapCarryFeatures(symbol,
                             t_ref,
                             features[0] + 0.50 * features[3],
                             features[10] + 0.50 * features[12],
                             features[75],
                             features[76],
                             features[77],
                             features[78]);
   features[80] = FXAI_Clamp(MathLog(1.0 + MathMax(spread_points, 0.0)), 0.0, 6.0);
   features[81] = FXAI_Clamp(spread_z20, -8.0, 8.0);
   features[82] = FXAI_Clamp(spread_vol_ratio20, 0.0, 4.5);
   features[83] = FXAI_Clamp(spread_rank20, -1.0, 1.0);
   features[79] = FXAI_LocalFeatureFamilyDrift(features);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      double lo = -8.0;
      double hi = 8.0;
      FXAI_GetFeatureClipBounds(f, lo, hi);
      features[f] = FXAI_Clamp(features[f], lo, hi);
   }

   return true;
}

#endif // __FXAI_FEATURE_BUILD_MQH__
