double FXAI_SharedTransferWindowFeatureMean(const double &x_window[][FXAI_AI_WEIGHTS],
                                            const int window_size,
                                            const int feature_idx)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double sum = 0.0;
   int used = 0;
   for(int b=0; b<window_size; b++)
   {
      sum += x_window[b][input_idx];
      used++;
   }
   if(used <= 0)
      return 0.0;
   return sum / (double)used;
}

double FXAI_SharedTransferWindowFeatureEMAMean(const double &x_window[][FXAI_AI_WEIGHTS],
                                               const int window_size,
                                               const int feature_idx,
                                               const double decay = 0.72)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double a = FXAI_Clamp(decay, 0.05, 0.98);
   double w = 1.0;
   double sw = 0.0;
   double sum = 0.0;
   for(int b=0; b<window_size; b++)
   {
      sum += w * x_window[b][input_idx];
      sw += w;
      w *= a;
   }
   if(sw <= 0.0)
      return 0.0;
   return sum / sw;
}

double FXAI_SharedTransferWindowFeatureStd(const double &x_window[][FXAI_AI_WEIGHTS],
                                           const int window_size,
                                           const int feature_idx)
{
   if(window_size <= 1)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   double mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, feature_idx);
   double acc = 0.0;
   for(int b=0; b<window_size; b++)
   {
      double d = x_window[b][input_idx] - mean;
      acc += d * d;
   }
   return MathSqrt(acc / (double)MathMax(window_size, 1));
}

double FXAI_SharedTransferWindowFeatureRange(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const int feature_idx,
                                             const int recent_bars = 0)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   int n = recent_bars;
   if(n <= 0 || n > window_size)
      n = window_size;
   double lo = x_window[0][input_idx];
   double hi = lo;
   for(int b=0; b<n; b++)
   {
      double v = x_window[b][input_idx];
      if(v < lo)
         lo = v;
      if(v > hi)
         hi = v;
   }
   return hi - lo;
}

double FXAI_SharedTransferWindowFeatureSlope(const double &x_window[][FXAI_AI_WEIGHTS],
                                             const int window_size,
                                             const int feature_idx)
{
   if(window_size <= 1)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;
   double first = x_window[0][input_idx];
   double last = x_window[window_size - 1][input_idx];
   return (first - last) / (double)MathMax(window_size - 1, 1);
}

double FXAI_SharedTransferWindowFeatureRecentDelta(const double &x_window[][FXAI_AI_WEIGHTS],
                                                   const int window_size,
                                                   const int feature_idx,
                                                   const int recent_bars)
{
   if(window_size <= 0)
      return 0.0;
   int input_idx = feature_idx + 1;
   if(input_idx < 1 || input_idx >= FXAI_AI_WEIGHTS)
      return 0.0;

   int n = recent_bars;
   if(n <= 1)
      n = MathMax(window_size / 4, 2);
   if(n > window_size)
      n = window_size;
   int last_idx = n - 1;
   if(last_idx < 0)
      last_idx = 0;
   return x_window[0][input_idx] - x_window[last_idx][input_idx];
}

void FXAI_BuildSharedTransferInputGlobalBase(const double &x[],
                                             const double domain_hash,
                                             const int horizon_minutes,
                                             double &out[])
{
   ArrayResize(out, FXAI_SHARED_TRANSFER_FEATURES);
   for(int i=0; i<FXAI_SHARED_TRANSFER_FEATURES; i++)
      out[i] = 0.0;
   out[0] = 1.0;
   out[1] = FXAI_GetInputFeature(x, 62);
   out[2] = FXAI_GetInputFeature(x, 63);
   out[3] = FXAI_GetInputFeature(x, 64);
   out[4] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetInputFeature(x, 65), 0.0, 1.0);

   double ret_mix = 0.0;
   double lag_mix = 0.0;
   double rel_mix = 0.0;
   double corr_mix = 0.0;
   double weight_total = 0.0;
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      int base = 50 + slot * 4;
      double ctx_ret = FXAI_GetInputFeature(x, base + 0);
      double ctx_lag = FXAI_GetInputFeature(x, base + 1);
      double ctx_rel = FXAI_GetInputFeature(x, base + 2);
      double ctx_corr = FXAI_GetInputFeature(x, base + 3);
      double w = FXAI_Clamp((0.30 + 0.70 * MathAbs(ctx_corr)) *
                            (0.35 + 0.65 * out[4]) *
                            (0.35 + 0.25 * MathAbs(ctx_ret) + 0.25 * MathAbs(ctx_lag) + 0.15 * MathAbs(ctx_rel)),
                            0.0,
                            3.0);
      if(w <= 1e-6)
         continue;
      ret_mix += w * ctx_ret;
      lag_mix += w * ctx_lag;
      rel_mix += w * ctx_rel;
      corr_mix += w * ctx_corr;
      weight_total += w;
   }
   if(weight_total > 1e-6)
   {
      out[5] = ret_mix / weight_total;
      out[6] = lag_mix / weight_total;
      out[7] = rel_mix / weight_total;
      out[8] = corr_mix / weight_total;
   }
   else
   {
      out[5] = 0.0;
      out[6] = 0.0;
      out[7] = 0.0;
      out[8] = 0.0;
   }

   double domain = FXAI_Clamp(domain_hash, 0.0, 1.0);
   double horizon_scale = FXAI_Clamp(MathLog(1.0 + (double)MathMax(horizon_minutes, 1)) / MathLog(1.0 + 1440.0), 0.0, 1.0);
   double main_mtf_body = 0.0;
   double main_mtf_loc = 0.0;
   double main_mtf_range = 0.0;
   double main_mtf_spread = 0.0;
   for(int tf_slot=0; tf_slot<FXAI_MAIN_MTF_TF_COUNT; tf_slot++)
   {
      int base = FXAI_MainMTFFeatureIndex(tf_slot, 0);
      if(base < 0)
         continue;
      main_mtf_body += FXAI_GetInputFeature(x, base + 0);
      main_mtf_loc += FXAI_GetInputFeature(x, base + 1);
      main_mtf_range += FXAI_GetInputFeature(x, base + 2);
      main_mtf_spread += FXAI_GetInputFeature(x, base + 3);
   }
   main_mtf_body /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_loc /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_range /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);
   main_mtf_spread /= (double)MathMax(FXAI_MAIN_MTF_TF_COUNT, 1);

   double ctx_mtf_body = 0.0;
   double ctx_mtf_loc = 0.0;
   double ctx_mtf_range = 0.0;
   double ctx_mtf_spread = 0.0;
   double ctx_mtf_weight = 0.0;
   for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
   {
      double slot_corr = MathAbs(FXAI_GetInputFeature(x, 50 + slot * 4 + 3));
      double slot_weight = 0.35 + 0.65 * slot_corr;
      double slot_body = 0.0;
      double slot_loc = 0.0;
      double slot_range = 0.0;
      double slot_spread = 0.0;
      int slot_used = 0;
      for(int tf_slot=0; tf_slot<FXAI_CONTEXT_MTF_TF_COUNT; tf_slot++)
      {
         int base = FXAI_ContextMTFFeatureIndex(slot, tf_slot, 0);
         if(base < 0)
            continue;
         slot_body += FXAI_GetInputFeature(x, base + 0);
         slot_loc += FXAI_GetInputFeature(x, base + 1);
         slot_range += FXAI_GetInputFeature(x, base + 2);
         slot_spread += FXAI_GetInputFeature(x, base + 3);
         slot_used++;
      }
      if(slot_used <= 0)
         continue;
      slot_body /= (double)slot_used;
      slot_loc /= (double)slot_used;
      slot_range /= (double)slot_used;
      slot_spread /= (double)slot_used;
      ctx_mtf_body += slot_weight * slot_body;
      ctx_mtf_loc += slot_weight * slot_loc;
      ctx_mtf_range += slot_weight * slot_range;
      ctx_mtf_spread += slot_weight * slot_spread;
      ctx_mtf_weight += slot_weight;
   }
   if(ctx_mtf_weight > 1e-6)
   {
      ctx_mtf_body /= ctx_mtf_weight;
      ctx_mtf_loc /= ctx_mtf_weight;
      ctx_mtf_range /= ctx_mtf_weight;
      ctx_mtf_spread /= ctx_mtf_weight;
   }

   double macro_pre = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
   double macro_post = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
   double macro_imp = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
   double macro_surprise = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3);
   double macro_surprise_abs = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4);
   double macro_class = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
   double macro_surprise_z = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 6);
   double macro_revision_abs = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 7);
   double macro_currency_rel = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 8);
   double macro_provenance = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 9);
   double macro_policy_div = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 14);
   double macro_policy_pressure = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 15);
   double macro_inflation_pressure = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 16);
   double macro_growth_pressure = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 18);
   double macro_state_quality = FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19);

   out[9] = 2.0 * domain - 1.0;
   out[10] = 2.0 * horizon_scale - 1.0;
   out[11] = FXAI_Clamp(0.60 * FXAI_GetInputFeature(x, 72) +
                        0.15 * FXAI_GetInputFeature(x, 73) +
                        0.10 * macro_pre -
                        0.08 * macro_post +
                        0.07 * main_mtf_loc +
                        0.06 * ctx_mtf_loc,
                        -1.0,
                        1.0);
   out[12] = FXAI_Clamp(0.28 * FXAI_GetInputFeature(x, 74) +
                        0.16 * FXAI_GetInputFeature(x, 75) +
                        0.14 * FXAI_GetInputFeature(x, 78) +
                        0.10 * macro_imp +
                        0.08 * macro_class +
                        0.08 * main_mtf_body +
                        0.06 * ctx_mtf_body,
                        -1.0,
                        1.0);
   out[13] = FXAI_Clamp(0.16 * FXAI_GetInputFeature(x, 76) -
                        0.16 * FXAI_GetInputFeature(x, 77) -
                        0.10 * FXAI_GetInputFeature(x, 79) +
                        0.08 * FXAI_GetInputFeature(x, 6) +
                        0.08 * FXAI_GetInputFeature(x, 81) +
                        0.06 * macro_surprise_abs -
                        0.05 * main_mtf_spread -
                        0.05 * FXAI_GetInputFeature(x, 82) +
                        0.04 * ctx_mtf_spread,
                        -4.0,
                        4.0);
   out[14] = FXAI_Clamp(0.42 * FXAI_GetInputFeature(x, 18) +
                        0.18 * FXAI_GetInputFeature(x, 19) -
                        0.18 * FXAI_GetInputFeature(x, 20) +
                        0.12 * FXAI_GetInputFeature(x, 21) +
                        0.06 * macro_imp +
                        0.08 * main_mtf_body +
                        0.08 * main_mtf_loc,
                        -4.0,
                        4.0);
   out[15] = FXAI_Clamp(0.16 * FXAI_GetInputFeature(x, 66) +
                        0.12 * FXAI_GetInputFeature(x, 67) +
                        0.12 * FXAI_GetInputFeature(x, 68) +
                        0.10 * FXAI_GetInputFeature(x, 69) +
                        0.14 * FXAI_GetInputFeature(x, 71) +
                        0.10 * macro_surprise +
                        0.08 * macro_surprise_abs +
                        0.06 * FXAI_GetInputFeature(x, 81) +
                        0.04 * FXAI_GetInputFeature(x, 83) +
                        0.05 * main_mtf_range +
                        0.05 * ctx_mtf_range,
                        -6.0,
                        6.0);
   out[16] = FXAI_Clamp(0.48 * FXAI_GetInputFeature(x, 68) +
                        0.18 * FXAI_GetInputFeature(x, 81) +
                        0.12 * FXAI_GetInputFeature(x, 80) +
                        0.10 * macro_imp +
                        0.08 * macro_surprise_abs +
                        0.08 * main_mtf_spread,
                        -4.0,
                        8.0);
   out[17] = FXAI_Clamp(0.55 * FXAI_GetInputFeature(x, 70) +
                        0.18 * FXAI_GetInputFeature(x, 82) +
                        0.10 * macro_post +
                        0.08 * macro_surprise_abs +
                        0.05 * main_mtf_range +
                        0.04 * MathAbs(FXAI_GetInputFeature(x, 83)),
                        0.0,
                        8.0);
   out[18] = FXAI_Clamp(0.45 * macro_surprise +
                        0.20 * macro_class +
                        0.15 * FXAI_GetInputFeature(x, 78) +
                        0.10 * FXAI_GetInputFeature(x, 72) +
                        0.10 * FXAI_GetInputFeature(x, 73) +
                        0.12 * macro_surprise_z +
                        0.10 * macro_policy_div +
                        0.08 * macro_currency_rel,
                        -6.0,
                        6.0);
   out[19] = FXAI_Clamp(0.40 * macro_surprise_abs +
                        0.22 * macro_imp +
                        0.18 * macro_pre +
                        0.10 * macro_post +
                        0.10 * FXAI_GetInputFeature(x, 79) +
                        0.10 * macro_revision_abs +
                        0.08 * macro_provenance +
                        0.08 * macro_policy_pressure +
                        0.06 * macro_inflation_pressure +
                        0.06 * macro_growth_pressure +
                        0.08 * macro_state_quality,
                        0.0,
                        6.0);
}

void FXAI_BuildSharedTransferInputGlobal(const double &x[],
                                         const double domain_hash,
                                         const int horizon_minutes,
                                         double &out[])
{
   FXAI_BuildSharedTransferInputGlobalBase(x, domain_hash, horizon_minutes, out);
}

void FXAI_BuildSharedTransferInputGlobal(const double &x[],
                                         const double &x_window[][FXAI_AI_WEIGHTS],
                                         const int window_size,
                                         const double domain_hash,
                                         const int horizon_minutes,
                                         double &out[])
{
   FXAI_BuildSharedTransferInputGlobalBase(x, domain_hash, horizon_minutes, out);
   if(ArraySize(out) < FXAI_SHARED_TRANSFER_FEATURES)
      ArrayResize(out, FXAI_SHARED_TRANSFER_FEATURES);

   if(window_size <= 0)
   {
      out[20] = FXAI_GetInputFeature(x, 0);
      out[21] = FXAI_GetInputFeature(x, 3);
      out[22] = FXAI_GetInputFeature(x, 41);
      out[23] = 0.50 * FXAI_GetInputFeature(x, 68) + 0.25 * FXAI_GetInputFeature(x, 69) + 0.25 * FXAI_GetInputFeature(x, 80);
      out[24] = 0.35 * FXAI_GetInputFeature(x, 10) + 0.25 * FXAI_GetInputFeature(x, 62) + 0.20 * FXAI_GetInputFeature(x, 63) + 0.20 * FXAI_GetInputFeature(x, 64);
      out[25] = 0.25 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0) +
                0.15 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4) +
                0.20 * FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
      out[26] = 0.20 * FXAI_GetInputFeature(x, 72) +
                0.15 * FXAI_GetInputFeature(x, 73) +
                0.20 * FXAI_GetInputFeature(x, 74) +
                0.10 * FXAI_GetInputFeature(x, 75) +
                0.15 * FXAI_GetInputFeature(x, 78) +
                0.10 * (FXAI_GetInputFeature(x, 76) - FXAI_GetInputFeature(x, 77)) +
                0.10 * FXAI_GetInputFeature(x, 79);
      out[27] = 0.30 * FXAI_GetInputFeature(x, 79) +
                0.25 * MathAbs(FXAI_GetInputFeature(x, 63)) +
                0.20 * MathAbs(FXAI_GetInputFeature(x, 68)) +
                0.15 * FXAI_GetInputFeature(x, 82) +
                0.10 * MathAbs(FXAI_GetInputFeature(x, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4));
      return;
   }

   double ret_fast = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 0);
   double ret_mid = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 1);
   double ret_long = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 2);
   double slope_m1 = FXAI_SharedTransferWindowFeatureSlope(x_window, window_size, 3);
   double vol_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 5);
   double vol_std = FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 5);
   double atr_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 41);
   double ctx_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 10);
   double ctx_std = FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 10);
   double shared_util = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 62);
   double shared_stability = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 63);
   double shared_lead = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 64);
   double shared_coverage = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 65);
   double spread_shock = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 68);
   double spread_accel = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 69);
   double spread_to_range = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 70);
   double micro_trend = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, 71);
   double session_transition = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 72);
   double session_overlap = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 73);
   double rollover_prox = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 74);
   double triple_swap = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 75);
   double swap_bias = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 76) -
                      FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 77);
   double carry_align = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 78);
   double drift_mean = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 79);
   double spread_log = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 80);
   double spread_z = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 81);
   double spread_vol = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 82);
   double spread_rank = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 83);
   double macro_pre = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 0);
   double macro_post = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 1);
   double macro_imp = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 2);
   double macro_surprise = FXAI_SharedTransferWindowFeatureEMAMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 3);
   double macro_surprise_abs = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 4);
   double macro_class = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 5);
   double macro_policy_div = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 14);
   double macro_policy_pressure = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 15);
   double macro_inflation_pressure = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 16);
   double macro_growth_pressure = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 18);
   double macro_state_quality = FXAI_SharedTransferWindowFeatureMean(x_window, window_size, FXAI_MACRO_EVENT_FEATURE_OFFSET + 19);
   double ret_delta = FXAI_SharedTransferWindowFeatureRecentDelta(x_window, window_size, 0, MathMax(window_size / 4, 3));
   double spread_delta = FXAI_SharedTransferWindowFeatureRecentDelta(x_window, window_size, 80, MathMax(window_size / 4, 3));

   out[20] = FXAI_Clamp(0.42 * ret_fast + 0.33 * ret_mid + 0.15 * ret_long + 0.10 * ret_delta, -4.0, 4.0);
   out[21] = FXAI_Clamp(0.38 * slope_m1 + 0.22 * (ret_fast - ret_mid) + 0.20 * micro_trend + 0.20 * FXAI_SharedTransferWindowFeatureSlope(x_window, window_size, 71), -4.0, 4.0);
   out[22] = FXAI_Clamp(0.34 * vol_mean + 0.22 * vol_std + 0.22 * atr_mean + 0.22 * FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 43), 0.0, 6.0);
   out[23] = FXAI_Clamp(0.26 * FXAI_SharedTransferWindowFeatureMean(x_window, window_size, 6) +
                        0.24 * spread_shock +
                        0.18 * spread_accel +
                        0.18 * spread_log +
                        0.14 * spread_z,
                        -6.0,
                        8.0);
   out[24] = FXAI_Clamp(0.28 * ctx_mean +
                        0.14 * ctx_std +
                        0.22 * shared_util +
                        0.14 * shared_stability +
                        0.12 * shared_lead +
                        0.10 * shared_coverage,
                        -4.0,
                        4.0);
   out[25] = FXAI_Clamp(0.18 * macro_pre +
                        0.12 * macro_post +
                        0.22 * macro_imp +
                        0.18 * macro_surprise +
                        0.16 * macro_surprise_abs +
                        0.10 * macro_class +
                        0.08 * macro_policy_pressure +
                        0.06 * macro_state_quality,
                        -6.0,
                        6.0);
   out[26] = FXAI_Clamp(0.18 * session_transition +
                        0.14 * session_overlap +
                        0.20 * rollover_prox +
                        0.12 * triple_swap +
                        0.14 * carry_align +
                        0.12 * swap_bias +
                        0.08 * drift_mean +
                        0.10 * macro_policy_div +
                        0.08 * macro_growth_pressure,
                        -4.0,
                        4.0);
   out[27] = FXAI_Clamp(0.24 * drift_mean +
                        0.18 * FXAI_SharedTransferWindowFeatureStd(x_window, window_size, 63) +
                        0.18 * FXAI_SharedTransferWindowFeatureRange(x_window, window_size, 10) +
                        0.14 * MathAbs(spread_delta) +
                        0.14 * spread_vol +
                        0.10 * MathAbs(spread_rank) +
                        0.12 * macro_inflation_pressure +
                        0.10 * macro_state_quality,
                        0.0,
                        6.0);
}
