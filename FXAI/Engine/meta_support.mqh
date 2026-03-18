#ifndef __FXAI_META_SUPPORT_MQH__
#define __FXAI_META_SUPPORT_MQH__

double FXAI_BarRandom01(const datetime bar_time, const int salt)
{
   uint x = (uint)(bar_time & 0x7FFFFFFF);
   uint s = (uint)(salt + 1);
   x ^= (s * 1103515245U + 12345U);
   x ^= (x << 13);
   x ^= (x >> 17);
   x ^= (x << 5);
   return (double)(x % 100000U) / 100000.0;
}

bool FXAI_ShouldSampleByPct(const datetime bar_time, const int salt, const double pct)
{
   double p = FXAI_Clamp(pct, 0.0, 100.0);
   if(p <= 0.0) return false;
   if(p >= 100.0) return true;
   return (FXAI_BarRandom01(bar_time, salt) < (p / 100.0));
}

bool FXAI_IsShadowBar(const int cadence_bars, const int bar_seq)
{
   int c = cadence_bars;
   if(c <= 0) return false;
   if(c == 1) return true;
   if(bar_seq < 0) return false;
   return ((bar_seq % c) == 0);
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_GetFeatureNormalizationMethod()
{
   int v = (int)AI_FeatureNormalization;
   if(v < (int)FXAI_NORM_EXISTING || v > (int)FXAI_NORM_DAIN)
      return FXAI_NORM_EXISTING;
   return (ENUM_FXAI_FEATURE_NORMALIZATION)v;
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_SanitizeNormMethod(const int method_id)
{
   int v = method_id;
   if(v < (int)FXAI_NORM_EXISTING || v > (int)FXAI_NORM_DAIN)
      v = (int)FXAI_NORM_EXISTING;
   return (ENUM_FXAI_FEATURE_NORMALIZATION)v;
}

ENUM_FXAI_FEATURE_NORMALIZATION FXAI_GetModelNormMethodRouted(const int ai_idx,
                                                             const int regime_id,
                                                             const int horizon_minutes)
{
   ENUM_FXAI_FEATURE_NORMALIZATION method = FXAI_GetFeatureNormalizationMethod();
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT)
      return method;

   if(g_model_norm_ready[ai_idx])
      method = FXAI_SanitizeNormMethod(g_model_norm_method[ai_idx]);

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_norm_horizon_ready[ai_idx][hslot])
      method = FXAI_SanitizeNormMethod(g_model_norm_method_horizon[ai_idx][hslot]);

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_norm_bank_ready[ai_idx][regime_id][hslot])
   {
      method = FXAI_SanitizeNormMethod(g_model_norm_method_bank[ai_idx][regime_id][hslot]);
   }
   return method;
}

void FXAI_BuildNormMethodCandidateList(const int ai_idx, int &methods[])
{
   ArrayResize(methods, 0);

   int seed_methods[FXAI_NORM_CAND_MAX];
   int n_seed = 0;
   seed_methods[n_seed++] = (int)FXAI_GetFeatureNormalizationMethod();

   bool deep_model = (ai_idx == (int)AI_LSTM || ai_idx == (int)AI_LSTMG ||
                      ai_idx == (int)AI_TCN || ai_idx == (int)AI_TFT ||
                      ai_idx == (int)AI_TST || ai_idx == (int)AI_AUTOFORMER ||
                      ai_idx == (int)AI_PATCHTST || ai_idx == (int)AI_STMN ||
                      ai_idx == (int)AI_S4 || ai_idx == (int)AI_CHRONOS ||
                      ai_idx == (int)AI_TIMESFM || ai_idx == (int)AI_GEODESICATTENTION);

   if(deep_model)
   {
      seed_methods[n_seed++] = (int)FXAI_NORM_EXISTING;
      seed_methods[n_seed++] = (int)FXAI_NORM_VOL_STD_RETURNS;
      seed_methods[n_seed++] = (int)FXAI_NORM_ATR_NATR_UNIT;
      seed_methods[n_seed++] = (int)FXAI_NORM_ZSCORE;
      seed_methods[n_seed++] = (int)FXAI_NORM_REVIN;
      seed_methods[n_seed++] = (int)FXAI_NORM_DAIN;
      seed_methods[n_seed++] = (int)FXAI_NORM_ROBUST_MEDIAN_IQR;
   }
   else
   {
      seed_methods[n_seed++] = (int)FXAI_NORM_EXISTING;
      seed_methods[n_seed++] = (int)FXAI_NORM_ZSCORE;
      seed_methods[n_seed++] = (int)FXAI_NORM_ROBUST_MEDIAN_IQR;
      seed_methods[n_seed++] = (int)FXAI_NORM_QUANTILE_TO_NORMAL;
      seed_methods[n_seed++] = (int)FXAI_NORM_CHANGE_PERCENT;
      seed_methods[n_seed++] = (int)FXAI_NORM_VOL_STD_RETURNS;
      seed_methods[n_seed++] = (int)FXAI_NORM_ATR_NATR_UNIT;
   }

   for(int i=0; i<n_seed; i++)
   {
      int m = seed_methods[i];
      if(m < (int)FXAI_NORM_EXISTING || m > (int)FXAI_NORM_DAIN) continue;
      bool exists = false;
      for(int j=0; j<ArraySize(methods); j++)
      {
         if(methods[j] == m)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;
      int sz = ArraySize(methods);
      ArrayResize(methods, sz + 1);
      methods[sz] = m;
      if(ArraySize(methods) >= FXAI_NORM_CAND_MAX)
         break;
   }
}

int FXAI_GetNormDefaultWindow()
{
   int w = FXAI_NORM_ROLL_WINDOW_DEFAULT;
   if(PredictionTargetMinutes <= 2) w = 128;
   else if(PredictionTargetMinutes >= 30) w = 256;
   if(w < 32) w = 32;
   if(w > FXAI_NORM_ROLL_WINDOW_MAX) w = FXAI_NORM_ROLL_WINDOW_MAX;
   return w;
}

void FXAI_BuildNormWindowsFromGroups(const int w_fast,
                                     const int w_mid,
                                     const int w_slow,
                                     const int w_regime,
                                     int &windows_out[])
{
   if(ArraySize(windows_out) != FXAI_AI_FEATURES)
      ArrayResize(windows_out, FXAI_AI_FEATURES);

   int wf = FXAI_NormalizationWindowClamp(w_fast);
   int wm = FXAI_NormalizationWindowClamp(w_mid);
   int ws = FXAI_NormalizationWindowClamp(w_slow);
   int wr = FXAI_NormalizationWindowClamp(w_regime);

   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = wm;
      if(f <= 6) w = wf;            // ultra-short momentum/cost features
      else if(f <= 14) w = wm;      // MTF trend/returns
      else if(f <= 21) w = wr;      // time/candle geometry
      else if(f <= 33) w = ws;      // MA/EMA trend structure
      else if(f <= 49) w = wm;      // volatility/statistical filters
      else if(f <= 65) w = wm;      // detailed cross-symbol context
      else if(f <= 71) w = wf;      // microstructure and spread shocks
      else if(f <= 75) w = wr;      // session and rollover state
      else if(f <= 78) w = ws;      // carry and swap pressures
      else w = wm;                  // feature-family drift composite
      windows_out[f] = w;
   }
}

void FXAI_ApplyNormWindows(const int &windows[], const int default_window)
{
   FXAI_SetNormalizationWindows(windows, default_window);
   int n = ArraySize(windows);
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      int w = default_window;
      if(f < n) w = windows[f];
      g_norm_feature_windows[f] = FXAI_NormalizationWindowClamp(w);
   }
   g_norm_default_window = FXAI_NormalizationWindowClamp(default_window);
   g_norm_windows_ready = true;
}


#endif // __FXAI_META_SUPPORT_MQH__
