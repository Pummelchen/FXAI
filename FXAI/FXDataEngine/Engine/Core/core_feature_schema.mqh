void FXAI_BuildInputVector(const double &features[], double &x[])
{
   x[0] = 1.0;
   for(int i=0; i<FXAI_AI_FEATURES; i++)
      x[i + 1] = features[i];
}

void FXAI_ClearInputWindow(double &x_window[][FXAI_AI_WEIGHTS], int &window_size)
{
   window_size = 0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = 0.0;
}

void FXAI_CopyInputVector(const double &src[], double &dst[])
{
   int n = MathMin(ArraySize(src), ArraySize(dst));
   for(int i=0; i<n; i++)
      dst[i] = src[i];
}

double FXAI_GetInputFeature(const double &x[], const int feature_idx)
{
   int idx = feature_idx + 1;
   if(idx >= 0 && idx < ArraySize(x)) return x[idx];
   return 0.0;
}

void FXAI_SetInputFeature(double &x[], const int feature_idx, const double value)
{
   int idx = feature_idx + 1;
   if(idx >= 0 && idx < ArraySize(x))
      x[idx] = value;
}

double FXAI_MeanInputFeatureRange(const double &x[],
                                  const int first_feature_idx,
                                  const int last_feature_idx)
{
   if(last_feature_idx < first_feature_idx) return 0.0;
   double sum = 0.0;
   int used = 0;
   for(int f=first_feature_idx; f<=last_feature_idx; f++)
   {
      sum += FXAI_GetInputFeature(x, f);
      used++;
   }
   if(used <= 0) return 0.0;
   return sum / (double)used;
}

double FXAI_QuantizeSignedFeature(const double value, const double step)
{
   double s = (step > 1e-9 ? step : 0.25);
   return MathRound(value / s) * s;
}

ulong FXAI_FeatureGroupBit(const int group_id)
{
   if(group_id < 0 || group_id > (int)FXAI_FEAT_GROUP_FILTERS)
      return 0;
   return ((ulong)1 << (ulong)group_id);
}

int FXAI_GetFeatureGroupForIndex(const int feature_idx)
{
   if(feature_idx < 0 || feature_idx >= FXAI_AI_FEATURES)
      return (int)FXAI_FEAT_GROUP_PRICE;

   if(feature_idx <= 5) return (int)FXAI_FEAT_GROUP_PRICE;
   if(feature_idx == 6) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx <= 9) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 12) return (int)FXAI_FEAT_GROUP_CONTEXT;
   if(feature_idx <= 14) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 17) return (int)FXAI_FEAT_GROUP_TIME;
   if(feature_idx <= 21) return (int)FXAI_FEAT_GROUP_PRICE;
   if(feature_idx <= 37) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx <= 45) return (int)FXAI_FEAT_GROUP_VOLATILITY;
   if(feature_idx <= 49) return (int)FXAI_FEAT_GROUP_FILTERS;
   if(feature_idx <= 65) return (int)FXAI_FEAT_GROUP_CONTEXT;
   if(feature_idx <= 71) return (int)FXAI_FEAT_GROUP_MICROSTRUCTURE;
   if(feature_idx <= 73) return (int)FXAI_FEAT_GROUP_TIME;
   if(feature_idx <= 78) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx == 79) return (int)FXAI_FEAT_GROUP_FILTERS;
   if(feature_idx <= 83) return (int)FXAI_FEAT_GROUP_COST;
   if(feature_idx < FXAI_CONTEXT_MTF_FEATURE_OFFSET) return (int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME;
   if(feature_idx < FXAI_MACRO_EVENT_FEATURE_OFFSET) return (int)FXAI_FEAT_GROUP_CONTEXT;

   int macro_rel = feature_idx - FXAI_MACRO_EVENT_FEATURE_OFFSET;
   if(macro_rel <= 2) return (int)FXAI_FEAT_GROUP_TIME;
   if(macro_rel == 8) return (int)FXAI_FEAT_GROUP_CONTEXT;
   return (int)FXAI_FEAT_GROUP_FILTERS;
}

ulong FXAI_DefaultFeatureGroupsForFamily(const int family)
{
   ulong mask = 0;
   switch(family)
   {
      case FXAI_FAMILY_LINEAR:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         break;
      case FXAI_FAMILY_TREE:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_FILTERS);
         break;
      case FXAI_FAMILY_RULE_BASED:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         break;
      case FXAI_FAMILY_DISTRIBUTIONAL:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         break;
      default:
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_PRICE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MULTI_TIMEFRAME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_VOLATILITY);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_TIME);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_COST);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_MICROSTRUCTURE);
         mask |= FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_FILTERS);
         break;
   }
   return mask;
}

int FXAI_DefaultFeatureSchemaForFamily(const int family)
{
   switch(family)
   {
      case FXAI_FAMILY_LINEAR:
      case FXAI_FAMILY_DISTRIBUTIONAL:
         return (int)FXAI_SCHEMA_SPARSE_STAT;
      case FXAI_FAMILY_RULE_BASED:
         return (int)FXAI_SCHEMA_RULE;
      case FXAI_FAMILY_TREE:
         return (int)FXAI_SCHEMA_TREE;
      case FXAI_FAMILY_RECURRENT:
      case FXAI_FAMILY_CONVOLUTIONAL:
      case FXAI_FAMILY_TRANSFORMER:
      case FXAI_FAMILY_STATE_SPACE:
         return (int)FXAI_SCHEMA_SEQUENCE;
      case FXAI_FAMILY_RETRIEVAL:
      case FXAI_FAMILY_MIXTURE:
      case FXAI_FAMILY_WORLD_MODEL:
         return (int)FXAI_SCHEMA_CONTEXTUAL;
      default:
         return (int)FXAI_SCHEMA_FULL;
   }
}

bool FXAI_IsFeatureEnabledForSchema(const int feature_idx,
                                    const int schema_id,
                                    const ulong groups_mask)
{
   if(feature_idx >= FXAI_MAIN_MTF_FEATURE_OFFSET && feature_idx < FXAI_MACRO_EVENT_FEATURE_OFFSET)
      return true;

   int group_id = FXAI_GetFeatureGroupForIndex(feature_idx);
   ulong bit = FXAI_FeatureGroupBit(group_id);
   if(bit == 0 || (groups_mask & bit) == 0)
      return false;

   switch(schema_id)
   {
      case FXAI_SCHEMA_SPARSE_STAT:
         if(feature_idx >= 46 && feature_idx <= 49) return false;
         if(feature_idx >= 50 && feature_idx <= 71) return false;
         return true;
      case FXAI_SCHEMA_RULE:
         return (group_id == (int)FXAI_FEAT_GROUP_PRICE ||
                 group_id == (int)FXAI_FEAT_GROUP_TIME ||
                 group_id == (int)FXAI_FEAT_GROUP_COST);
      case FXAI_SCHEMA_CONTEXTUAL:
         if(group_id == (int)FXAI_FEAT_GROUP_TIME && feature_idx >= 15 && feature_idx <= 17)
            return false;
         return true;
      case FXAI_SCHEMA_TREE:
      case FXAI_SCHEMA_SEQUENCE:
      case FXAI_SCHEMA_FULL:
      default:
         return true;
   }
}

void FXAI_ApplyFeatureSchemaToInputEx(const int schema_id,
                                      const ulong groups_mask,
                                      const int sequence_bars,
                                      const double &x_window[][FXAI_AI_WEIGHTS],
                                      const int window_size,
                                      double &x[])
{
   if(ArraySize(x) < FXAI_AI_WEIGHTS)
      return;

   bool enabled_input[FXAI_AI_WEIGHTS];
   double masked_x[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      enabled_input[k] = true;
      masked_x[k] = 0.0;
   }

   enabled_input[0] = true;
   masked_x[0] = 1.0;
   x[0] = 1.0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int input_idx = f + 1;
      bool enabled = FXAI_IsFeatureEnabledForSchema(f, schema_id, groups_mask);
      enabled_input[input_idx] = enabled;
      masked_x[input_idx] = (enabled ? x[input_idx] : 0.0);
      x[input_idx] = masked_x[input_idx];
   }

   int seq_n = sequence_bars;
   if(seq_n < 1) seq_n = 1;
   if(seq_n > FXAI_MAX_SEQUENCE_BARS) seq_n = FXAI_MAX_SEQUENCE_BARS;
   if(window_size > 0 && window_size < seq_n) seq_n = window_size;

   double seq_mean[FXAI_AI_WEIGHTS];
   double seq_delta[FXAI_AI_WEIGHTS];
   double seq_std[FXAI_AI_WEIGHTS];
   double seq_short_mean[FXAI_AI_WEIGHTS];
   double seq_mid_mean[FXAI_AI_WEIGHTS];
   double seq_long_mean[FXAI_AI_WEIGHTS];
   double seq_short_delta[FXAI_AI_WEIGHTS];
   double seq_mid_delta[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
   {
      seq_mean[k] = masked_x[k];
      seq_delta[k] = 0.0;
      seq_std[k] = 0.0;
      seq_short_mean[k] = masked_x[k];
      seq_mid_mean[k] = masked_x[k];
      seq_long_mean[k] = masked_x[k];
      seq_short_delta[k] = 0.0;
      seq_mid_delta[k] = 0.0;
   }

   if(seq_n > 1)
   {
      int used = 0;
      for(int b=0; b<seq_n; b++)
      {
         if(b >= window_size) break;
         used++;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double wv = (enabled_input[k] ? x_window[b][k] : 0.0);
            seq_mean[k] += wv;
         }
      }
      if(used > 0)
      {
         double denom = (double)(used + 1);
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            seq_mean[k] /= denom;
      }

      for(int b=0; b<seq_n && b<window_size; b++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double wv = (enabled_input[k] ? x_window[b][k] : 0.0);
            double d = wv - seq_mean[k];
            seq_std[k] += d * d;
         }
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         seq_std[k] = MathSqrt(seq_std[k] / (double)MathMax(seq_n - 1, 1));

      int short_n = MathMax(seq_n / 4, 1);
      int mid_n = MathMax(seq_n / 2, 1);
      int long_n = seq_n;
      int short_used = 0;
      int mid_used = 0;
      int long_used = 0;
      for(int b=0; b<seq_n && b<window_size; b++)
      {
         bool use_short = (b < short_n);
         bool use_mid = (b < mid_n);
         bool use_long = (b < long_n);
         if(use_short) short_used++;
         if(use_mid) mid_used++;
         if(use_long) long_used++;
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double wv = (enabled_input[k] ? x_window[b][k] : 0.0);
            if(use_short) seq_short_mean[k] += wv;
            if(use_mid) seq_mid_mean[k] += wv;
            if(use_long) seq_long_mean[k] += wv;
         }
      }
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         seq_short_mean[k] /= (double)MathMax(short_used + 1, 1);
         seq_mid_mean[k] /= (double)MathMax(mid_used + 1, 1);
         seq_long_mean[k] /= (double)MathMax(long_used + 1, 1);
      }

      int last_idx = seq_n - 2;
      if(last_idx >= 0 && last_idx < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double prev_v = (enabled_input[k] ? x_window[last_idx][k] : 0.0);
            seq_delta[k] = masked_x[k] - prev_v;
         }
      }

      int short_last = short_n - 1;
      if(short_last >= 0 && short_last < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double prev_v = (enabled_input[k] ? x_window[short_last][k] : 0.0);
            seq_short_delta[k] = masked_x[k] - prev_v;
         }
      }
      int mid_last = mid_n - 1;
      if(mid_last >= 0 && mid_last < window_size)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         {
            double prev_v = (enabled_input[k] ? x_window[mid_last][k] : 0.0);
            seq_mid_delta[k] = masked_x[k] - prev_v;
         }
      }
   }

   // Schema-specific projection stage. This is intentionally stronger than
   // simple masking so each plugin family sees a representation aligned to its
   // inductive bias while staying on the shared feature contract.
   switch(schema_id)
   {
      case FXAI_SCHEMA_SPARSE_STAT:
      {
         double mtf_ret = FXAI_MeanInputFeatureRange(x, 7, 9);
         double mtf_slope = FXAI_MeanInputFeatureRange(x, 13, 14);
         double sma_fast = 0.25 * (FXAI_GetInputFeature(x, 22) + FXAI_GetInputFeature(x, 24) +
                                   FXAI_GetInputFeature(x, 26) + FXAI_GetInputFeature(x, 28));
         double sma_slow = 0.25 * (FXAI_GetInputFeature(x, 23) + FXAI_GetInputFeature(x, 25) +
                                   FXAI_GetInputFeature(x, 27) + FXAI_GetInputFeature(x, 29));
         double ema_fast = 0.25 * (FXAI_GetInputFeature(x, 30) + FXAI_GetInputFeature(x, 32) +
                                   FXAI_GetInputFeature(x, 34) + FXAI_GetInputFeature(x, 36));
         double ema_slow = 0.25 * (FXAI_GetInputFeature(x, 31) + FXAI_GetInputFeature(x, 33) +
                                   FXAI_GetInputFeature(x, 35) + FXAI_GetInputFeature(x, 37));
         double vol_pack = FXAI_MeanInputFeatureRange(x, 41, 45);
         double filt_pack = FXAI_MeanInputFeatureRange(x, 46, 49);
         double ctx_ret = FXAI_GetInputFeature(x, 10);
         double ctx_rel = FXAI_GetInputFeature(x, 12);
         double ctx_corr = 0.0;
         bool ctx_enabled = ((groups_mask & FXAI_FeatureGroupBit((int)FXAI_FEAT_GROUP_CONTEXT)) != 0);
         if(ctx_enabled)
         {
            ctx_ret = (FXAI_GetInputFeature(x, 10) + FXAI_GetInputFeature(masked_x, 50) +
                       FXAI_GetInputFeature(masked_x, 54) + FXAI_GetInputFeature(masked_x, 58)) / 4.0;
            ctx_rel = (FXAI_GetInputFeature(x, 12) + FXAI_GetInputFeature(masked_x, 52) +
                       FXAI_GetInputFeature(masked_x, 56) + FXAI_GetInputFeature(masked_x, 60)) / 4.0;
            ctx_corr = (FXAI_GetInputFeature(masked_x, 53) + FXAI_GetInputFeature(masked_x, 57) +
                        FXAI_GetInputFeature(masked_x, 61)) / 3.0;
         }

         FXAI_SetInputFeature(x, 7, mtf_ret);
         FXAI_SetInputFeature(x, 8, mtf_slope);
         FXAI_SetInputFeature(x, 9, 0.5 * (mtf_ret + mtf_slope));
         FXAI_SetInputFeature(x, 22, sma_fast);
         FXAI_SetInputFeature(x, 23, sma_slow);
         FXAI_SetInputFeature(x, 24, ema_fast);
         FXAI_SetInputFeature(x, 25, ema_slow);
         FXAI_SetInputFeature(x, 26, sma_fast - sma_slow);
         FXAI_SetInputFeature(x, 27, ema_fast - ema_slow);
         FXAI_SetInputFeature(x, 28, vol_pack);
         FXAI_SetInputFeature(x, 29, filt_pack);
         FXAI_SetInputFeature(x, 30, ctx_ret);
         FXAI_SetInputFeature(x, 31, ctx_rel);
         FXAI_SetInputFeature(x, 32, ctx_corr);
         for(int f=33; f<=39; f++)
            FXAI_SetInputFeature(x, f, 0.0);
         break;
      }

      case FXAI_SCHEMA_SEQUENCE:
      {
         double ret_short = FXAI_GetInputFeature(x, 0);
         double ret_mid = FXAI_GetInputFeature(x, 1);
         double ret_long = FXAI_GetInputFeature(x, 2);
         double mtf_fast = 0.5 * (FXAI_GetInputFeature(x, 7) + FXAI_GetInputFeature(x, 13));
         double mtf_slow = 0.5 * (FXAI_GetInputFeature(x, 9) + FXAI_GetInputFeature(x, 14));
         double seq_ret_short = FXAI_GetInputFeature(seq_short_mean, 0);
         double seq_ret_mid = FXAI_GetInputFeature(seq_mid_mean, 1);
         double seq_ret_long = FXAI_GetInputFeature(seq_long_mean, 2);
         double seq_ctx = 0.50 * FXAI_GetInputFeature(seq_short_mean, 10) +
                          0.30 * FXAI_GetInputFeature(seq_mid_mean, 10) +
                          0.20 * FXAI_GetInputFeature(seq_long_mean, 10);
         double seq_vol = 0.50 * FXAI_GetInputFeature(seq_short_mean, 41) +
                          0.30 * FXAI_GetInputFeature(seq_mid_mean, 41) +
                          0.20 * FXAI_GetInputFeature(seq_long_mean, 41);
         double seq_ret_accel = FXAI_GetInputFeature(seq_short_mean, 0) - FXAI_GetInputFeature(seq_mid_mean, 0);
         double seq_ctx_delta = FXAI_GetInputFeature(seq_short_mean, 10) - FXAI_GetInputFeature(seq_long_mean, 10);

         FXAI_SetInputFeature(x, 13, ret_short - ret_mid);
         FXAI_SetInputFeature(x, 14, ret_mid - ret_long);
         FXAI_SetInputFeature(x, 22, FXAI_GetInputFeature(x, 22) - FXAI_GetInputFeature(x, 23));
         FXAI_SetInputFeature(x, 23, FXAI_GetInputFeature(x, 24) - FXAI_GetInputFeature(x, 25));
         FXAI_SetInputFeature(x, 24, FXAI_GetInputFeature(x, 26) - FXAI_GetInputFeature(x, 27));
         FXAI_SetInputFeature(x, 25, FXAI_GetInputFeature(x, 28) - FXAI_GetInputFeature(x, 29));
         FXAI_SetInputFeature(x, 26, FXAI_GetInputFeature(x, 30) - FXAI_GetInputFeature(x, 31));
         FXAI_SetInputFeature(x, 27, FXAI_GetInputFeature(x, 32) - FXAI_GetInputFeature(x, 33));
         FXAI_SetInputFeature(x, 28, FXAI_GetInputFeature(x, 34) - FXAI_GetInputFeature(x, 35));
         FXAI_SetInputFeature(x, 29, FXAI_GetInputFeature(x, 36) - FXAI_GetInputFeature(x, 37));
         FXAI_SetInputFeature(x, 30, mtf_fast);
         FXAI_SetInputFeature(x, 31, mtf_slow);
         FXAI_SetInputFeature(x, 32, mtf_fast - mtf_slow);
         FXAI_SetInputFeature(x, 33, seq_ret_short - seq_ret_mid);
         FXAI_SetInputFeature(x, 34, seq_ret_mid - seq_ret_long);
         FXAI_SetInputFeature(x, 35, FXAI_GetInputFeature(seq_short_delta, 0));
         FXAI_SetInputFeature(x, 36, FXAI_GetInputFeature(seq_mid_delta, 1));
         FXAI_SetInputFeature(x, 37, seq_ret_accel);
         FXAI_SetInputFeature(x, 38, seq_ctx);
         FXAI_SetInputFeature(x, 39, seq_vol);
         // Preserve explicit lag/context blocks for sequence-capable plugins.
         FXAI_SetInputFeature(x, 50, FXAI_GetInputFeature(seq_short_mean, 50));
         FXAI_SetInputFeature(x, 51, FXAI_GetInputFeature(seq_short_mean, 51));
         FXAI_SetInputFeature(x, 52, FXAI_GetInputFeature(seq_short_mean, 52));
         FXAI_SetInputFeature(x, 53, FXAI_GetInputFeature(seq_short_mean, 53));
         FXAI_SetInputFeature(x, 54, FXAI_GetInputFeature(seq_mid_mean, 50));
         FXAI_SetInputFeature(x, 55, FXAI_GetInputFeature(seq_mid_mean, 51));
         FXAI_SetInputFeature(x, 56, FXAI_GetInputFeature(seq_mid_mean, 52));
         FXAI_SetInputFeature(x, 57, FXAI_GetInputFeature(seq_mid_mean, 53));
         FXAI_SetInputFeature(x, 58, FXAI_GetInputFeature(seq_short_delta, 50));
         FXAI_SetInputFeature(x, 59, seq_ctx_delta);
         FXAI_SetInputFeature(x, 60, FXAI_GetInputFeature(seq_std, 50));
         FXAI_SetInputFeature(x, 61, FXAI_GetInputFeature(seq_std, 51));
         break;
      }

      case FXAI_SCHEMA_RULE:
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            int group_id = FXAI_GetFeatureGroupForIndex(f);
            if(group_id == (int)FXAI_FEAT_GROUP_TIME)
               continue;

            double v = FXAI_GetInputFeature(x, f);
            double out_v = 0.0;
            if(v > 0.15) out_v = 1.0;
            else if(v < -0.15) out_v = -1.0;
            FXAI_SetInputFeature(x, f, out_v);
         }
         break;
      }

      case FXAI_SCHEMA_TREE:
      {
         for(int f=0; f<FXAI_AI_FEATURES; f++)
         {
            double v = FXAI_GetInputFeature(x, f);
            FXAI_SetInputFeature(x, f, FXAI_QuantizeSignedFeature(v, 0.25));
         }
         break;
      }

      case FXAI_SCHEMA_CONTEXTUAL:
      {
         double ctx_ret = (FXAI_GetInputFeature(x, 50) + FXAI_GetInputFeature(x, 54) +
                           FXAI_GetInputFeature(x, 58)) / 3.0;
         double ctx_lag = (FXAI_GetInputFeature(x, 51) + FXAI_GetInputFeature(x, 55) +
                           FXAI_GetInputFeature(x, 59)) / 3.0;
         double ctx_rel = (FXAI_GetInputFeature(x, 52) + FXAI_GetInputFeature(x, 56) +
                           FXAI_GetInputFeature(x, 60)) / 3.0;
         double ctx_corr = (FXAI_GetInputFeature(x, 53) + FXAI_GetInputFeature(x, 57) +
                            FXAI_GetInputFeature(x, 61)) / 3.0;
         double ctx_ret_fast = (FXAI_GetInputFeature(seq_short_mean, 50) +
                                FXAI_GetInputFeature(seq_short_mean, 54) +
                                FXAI_GetInputFeature(seq_short_mean, 58)) / 3.0;
         double ctx_ret_slow = (FXAI_GetInputFeature(seq_long_mean, 50) +
                                FXAI_GetInputFeature(seq_long_mean, 54) +
                                FXAI_GetInputFeature(seq_long_mean, 58)) / 3.0;
         double ctx_corr_fast = (FXAI_GetInputFeature(seq_short_mean, 53) +
                                 FXAI_GetInputFeature(seq_short_mean, 57) +
                                 FXAI_GetInputFeature(seq_short_mean, 61)) / 3.0;
         double ctx_strength = 0.30 * MathAbs(ctx_ret) +
                               0.30 * MathAbs(ctx_rel) +
                               0.25 * MathAbs(ctx_corr) +
                               0.15 * MathAbs(ctx_lag);

         FXAI_SetInputFeature(x, 10, 0.35 * FXAI_GetInputFeature(x, 10) + 0.35 * ctx_ret + 0.30 * ctx_ret_fast);
         FXAI_SetInputFeature(x, 11, 0.40 * FXAI_GetInputFeature(x, 11) + 0.35 * MathAbs(ctx_rel) + 0.25 * MathAbs(ctx_ret_fast - ctx_ret_slow));
         FXAI_SetInputFeature(x, 12, 0.35 * FXAI_GetInputFeature(x, 12) + 0.35 * ctx_corr + 0.30 * ctx_corr_fast);
         FXAI_SetInputFeature(x, 13, 0.50 * ctx_lag + 0.50 * (ctx_ret_fast - ctx_ret_slow));
         FXAI_SetInputFeature(x, 14, ctx_strength + 0.20 * MathAbs(ctx_ret_fast - ctx_ret_slow));
         // Keep explicit per-symbol context blocks instead of collapsing them all.
         FXAI_SetInputFeature(x, 15, FXAI_GetInputFeature(x, 50));
         FXAI_SetInputFeature(x, 16, FXAI_GetInputFeature(x, 51));
         FXAI_SetInputFeature(x, 17, FXAI_GetInputFeature(x, 52));
         FXAI_SetInputFeature(x, 18, FXAI_GetInputFeature(x, 53));
         FXAI_SetInputFeature(x, 19, FXAI_GetInputFeature(x, 54));
         FXAI_SetInputFeature(x, 20, FXAI_GetInputFeature(x, 55));
         FXAI_SetInputFeature(x, 21, FXAI_GetInputFeature(x, 56));
         FXAI_SetInputFeature(x, 22, FXAI_GetInputFeature(x, 57));
         FXAI_SetInputFeature(x, 50, FXAI_GetInputFeature(x, 58));
         FXAI_SetInputFeature(x, 51, FXAI_GetInputFeature(x, 59));
         FXAI_SetInputFeature(x, 52, FXAI_GetInputFeature(x, 60));
         FXAI_SetInputFeature(x, 53, FXAI_GetInputFeature(x, 61));
         FXAI_SetInputFeature(x, 54, FXAI_GetInputFeature(seq_mean, 50));
         FXAI_SetInputFeature(x, 55, FXAI_GetInputFeature(seq_mean, 51));
         FXAI_SetInputFeature(x, 56, FXAI_GetInputFeature(seq_mean, 52));
         FXAI_SetInputFeature(x, 57, FXAI_GetInputFeature(seq_mean, 53));
         FXAI_SetInputFeature(x, 58, FXAI_GetInputFeature(seq_delta, 50));
         FXAI_SetInputFeature(x, 59, FXAI_GetInputFeature(seq_short_delta, 51));
         FXAI_SetInputFeature(x, 60, FXAI_GetInputFeature(seq_std, 50));
         FXAI_SetInputFeature(x, 61, FXAI_GetInputFeature(seq_std, 51));
         break;
      }

      case FXAI_SCHEMA_FULL:
      default:
         break;
   }
}

void FXAI_ApplyFeatureSchemaToPayloadEx(const int schema_id,
                                        const ulong groups_mask,
                                        const int sequence_bars,
                                        double &x_window[][FXAI_AI_WEIGHTS],
                                        const int window_size,
                                        double &x[])
{
   int ws = window_size;
   if(ws < 0) ws = 0;
   if(ws > FXAI_MAX_SEQUENCE_BARS) ws = FXAI_MAX_SEQUENCE_BARS;

   double raw_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
   {
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      {
         raw_window[b][k] = 0.0;
         if(b < ws)
            raw_window[b][k] = x_window[b][k];
      }
   }

   // Project each historical row using only older rows. This gives plugins a
   // schema-native rolling payload instead of a raw normalized shared vector.
   for(int b=0; b<ws; b++)
   {
      double row[FXAI_AI_WEIGHTS];
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         row[k] = raw_window[b][k];

      double tail_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
      int tail_size = 0;
      for(int tb=b + 1; tb<ws && tail_size<FXAI_MAX_SEQUENCE_BARS; tb++, tail_size++)
      {
         for(int k=0; k<FXAI_AI_WEIGHTS; k++)
            tail_window[tail_size][k] = raw_window[tb][k];
      }

      FXAI_ApplyFeatureSchemaToInputEx(schema_id,
                                       groups_mask,
                                       sequence_bars,
                                       tail_window,
                                       tail_size,
                                       row);
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         x_window[b][k] = row[k];
   }

   FXAI_ApplyFeatureSchemaToInputEx(schema_id,
                                    groups_mask,
                                    sequence_bars,
                                    x_window,
                                    ws,
                                    x);
}

void FXAI_ApplyPayloadTransformPipelineEx(const int schema_id,
                                          const ulong groups_mask,
                                          const int normalization_method_id,
                                          const int horizon_minutes,
                                          const int sequence_bars,
                                          double &x_window[][FXAI_AI_WEIGHTS],
                                          const int window_size,
                                          double &x[])
{
   FXAI_ApplyPayloadAdaptiveNormalization(normalization_method_id,
                                          horizon_minutes,
                                          x_window,
                                          window_size,
                                          x);
   FXAI_ApplyFeatureSchemaToPayloadEx(schema_id,
                                      groups_mask,
                                      sequence_bars,
                                      x_window,
                                      window_size,
                                      x);
}

int FXAI_ContextExtraIndex(const int sample_idx, const int feat_idx)
{
   if(sample_idx < 0) return -1;
   if(feat_idx < 0 || feat_idx >= FXAI_CONTEXT_EXTRA_FEATS) return -1;
   return sample_idx * FXAI_CONTEXT_EXTRA_FEATS + feat_idx;
}

int FXAI_MainMTFBarsForSlot(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return 5;
      case 1: return 15;
      case 2: return 30;
      case 3: return 60;
      default: return 1;
   }
}

int FXAI_ContextMTFBarsForSlot(const int tf_slot)
{
   switch(tf_slot)
   {
      case 0: return 1;
      case 1: return 5;
      case 2: return 15;
      case 3: return 30;
      case 4: return 60;
      default: return 1;
   }
}

int FXAI_MainMTFFeatureIndex(const int tf_slot, const int metric)
{
   if(tf_slot < 0 || tf_slot >= FXAI_MAIN_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_MAIN_MTF_FEATURE_OFFSET + tf_slot * FXAI_MTF_STATE_FEATURES_PER_TF + metric;
}

int FXAI_ContextMTFFeatureIndex(const int slot,
                                const int tf_slot,
                                const int metric)
{
   if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
      return -1;
   if(tf_slot < 0 || tf_slot >= FXAI_CONTEXT_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_CONTEXT_MTF_FEATURE_OFFSET +
          (slot * FXAI_CONTEXT_MTF_TF_COUNT + tf_slot) * FXAI_MTF_STATE_FEATURES_PER_TF +
          metric;
}

int FXAI_ContextSlotMTFExtraIndex(const int slot,
                                  const int tf_slot,
                                  const int metric)
{
   if(slot < 0 || slot >= FXAI_CONTEXT_TOP_SYMBOLS)
      return -1;
   if(tf_slot < 0 || tf_slot >= FXAI_CONTEXT_MTF_TF_COUNT)
      return -1;
   if(metric < 0 || metric >= FXAI_MTF_STATE_FEATURES_PER_TF)
      return -1;
   return FXAI_CONTEXT_MTF_OFFSET +
          slot * FXAI_CONTEXT_SLOT_MTF_FEATS +
          tf_slot * FXAI_MTF_STATE_FEATURES_PER_TF +
          metric;
}

bool FXAI_ComputeAggregatedCandleSpreadState(const int idx,
                                             const int window_bars,
                                             const double &open_arr[],
                                             const double &high_arr[],
                                             const double &low_arr[],
                                             const double &close_arr[],
                                             const int &spread_arr[],
                                             const double point_value,
                                             double &body_bias,
                                             double &close_loc,
                                             double &range_pressure,
                                             double &spread_pressure)
{
   body_bias = 0.0;
   close_loc = 0.0;
   range_pressure = 0.0;
   spread_pressure = 0.0;

   int n = ArraySize(close_arr);
   if(idx < 0 || window_bars < 1 || n <= 0)
      return false;
   if(ArraySize(open_arr) != n || ArraySize(high_arr) != n || ArraySize(low_arr) != n || ArraySize(spread_arr) != n)
      return false;

   int last = idx + window_bars - 1;
   if(last >= n)
      return false;

   double point = (point_value > 0.0 ? point_value : 1.0);
   double agg_open = open_arr[last];
   double agg_close = close_arr[idx];
   double agg_high = high_arr[idx];
   double agg_low = low_arr[idx];
   double spread_sum_cur = 0.0;
   int spread_n_cur = 0;

   for(int k=0; k<window_bars; k++)
   {
      int ik = idx + k;
      if(ik < 0 || ik >= n)
         break;
      if(high_arr[ik] > agg_high) agg_high = high_arr[ik];
      if(low_arr[ik] < agg_low) agg_low = low_arr[ik];
      spread_sum_cur += MathMax((double)spread_arr[ik], 0.0);
      spread_n_cur++;
   }

   double bar_range = MathMax(agg_high - agg_low, point);
   double bar_range_points = MathMax(0.0, (agg_high - agg_low) / point);
   double spread_cur = spread_sum_cur / (double)MathMax(spread_n_cur, 1);

   double avg_range_points = 0.0;
   double avg_spread = 0.0;
   int windows_used = 0;
   for(int w=0; w<20; w++)
   {
      int base = idx + w * window_bars;
      int base_last = base + window_bars - 1;
      if(base < 0 || base_last >= n)
         break;

      double win_high = high_arr[base];
      double win_low = low_arr[base];
      double win_spread_sum = 0.0;
      int win_spread_n = 0;
      for(int k=0; k<window_bars; k++)
      {
         int ik = base + k;
         if(ik < 0 || ik >= n)
            break;
         if(high_arr[ik] > win_high) win_high = high_arr[ik];
         if(low_arr[ik] < win_low) win_low = low_arr[ik];
         win_spread_sum += MathMax((double)spread_arr[ik], 0.0);
         win_spread_n++;
      }

      avg_range_points += MathMax(0.0, (win_high - win_low) / point);
      avg_spread += win_spread_sum / (double)MathMax(win_spread_n, 1);
      windows_used++;
   }

   if(windows_used <= 0)
   {
      avg_range_points = MathMax(bar_range_points, 0.25);
      avg_spread = MathMax(spread_cur, 0.25);
      windows_used = 1;
   }
   else
   {
      avg_range_points /= (double)windows_used;
      avg_spread /= (double)windows_used;
   }

   body_bias = FXAI_Clamp((agg_close - agg_open) / bar_range, -1.2, 1.2);
   close_loc = FXAI_Clamp(((agg_close - agg_low) - (agg_high - agg_close)) / bar_range, -1.2, 1.2);
   range_pressure = FXAI_ClipSym((bar_range_points / MathMax(avg_range_points, 0.25)) - 1.0, 6.0);
   spread_pressure = FXAI_ClipSym((spread_cur / MathMax(avg_spread, 0.25)) - 1.0, 8.0);
   return true;
}

double FXAI_GetContextExtraValue(const double &arr[],
                                 const int sample_idx,
                                 const int feat_idx,
                                 const double def_value)
{
   int idx = FXAI_ContextExtraIndex(sample_idx, feat_idx);
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

void FXAI_SetContextExtraValue(double &arr[],
                               const int sample_idx,
                               const int feat_idx,
                               const double value)
{
   int idx = FXAI_ContextExtraIndex(sample_idx, feat_idx);
   if(idx >= 0 && idx < ArraySize(arr))
      arr[idx] = value;
}


void FXAI_ApplyFeatureSchemaToInput(const int schema_id,
                                    const ulong groups_mask,
                                    double &x[])
{
   double dummy_window[FXAI_MAX_SEQUENCE_BARS][FXAI_AI_WEIGHTS];
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         dummy_window[b][k] = 0.0;
   FXAI_ApplyFeatureSchemaToPayloadEx(schema_id, groups_mask, 1, dummy_window, 0, x);
}
