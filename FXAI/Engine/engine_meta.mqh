#ifndef __FXAI_ENGINE_META_MQH__
#define __FXAI_ENGINE_META_MQH__

double FXAI_GetArrayValue(const double &arr[], const int idx, const double def_value)
{
   if(idx >= 0 && idx < ArraySize(arr)) return arr[idx];
   return def_value;
}

double FXAI_GetIntArrayMean(const int &arr[],
                            const int start_idx,
                            const int width,
                            const double fallback)
{
   int n = ArraySize(arr);
   if(n <= 0 || start_idx < 0 || start_idx >= n || width <= 0)
      return MathMax(fallback, 0.10);

   int end = start_idx + width;
   if(end > n) end = n;
   double sum = 0.0;
   int used = 0;
   for(int i=start_idx; i<end; i++)
   {
      double v = (double)arr[i];
      if(v <= 0.0) continue;
      sum += v;
      used++;
   }
   if(used <= 0) return MathMax(fallback, 0.10);
   return sum / (double)used;
}

int FXAI_GetStaticRegimeId(const datetime sample_time,
                           const double spread_points,
                           const double spread_ref,
                           const double vol_proxy_abs,
                           const double vol_ref)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = MathMax(spread_ref, 0.10);
   double vp_ref = MathMax(MathAbs(vol_ref), 1e-6);
   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_REGIME_COUNT) regime = FXAI_REGIME_COUNT - 1;
   return regime;
}

int FXAI_ClampHorizon(const int h_in)
{
   int h = h_in;
   if(h < 1) h = 1;
   if(h > 720) h = 720;
   return h;
}

void FXAI_SortIntAsc(int &arr[])
{
   int n = ArraySize(arr);
   for(int i=1; i<n; i++)
   {
      int key = arr[i];
      int j = i - 1;
      while(j >= 0 && arr[j] > key)
      {
         arr[j + 1] = arr[j];
         j--;
      }
      arr[j + 1] = key;
   }
}

void FXAI_ParseHorizonList(const string raw, const int base_h, int &out_horizons[])
{
   ArrayResize(out_horizons, 0);

   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   for(int i=0; i<n; i++)
   {
      string tok = parts[i];
      StringTrimLeft(tok);
      StringTrimRight(tok);
      if(StringLen(tok) <= 0) continue;

      int hv = (int)StringToInteger(tok);
      hv = FXAI_ClampHorizon(hv);

      bool exists = false;
      for(int j=0; j<ArraySize(out_horizons); j++)
      {
         if(out_horizons[j] == hv)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      int sz = ArraySize(out_horizons);
      ArrayResize(out_horizons, sz + 1);
      out_horizons[sz] = hv;
      if(ArraySize(out_horizons) >= FXAI_MAX_HORIZONS)
         break;
   }

   int b = FXAI_ClampHorizon(base_h);
   bool has_base = false;
   for(int j=0; j<ArraySize(out_horizons); j++)
   {
      if(out_horizons[j] == b)
      {
         has_base = true;
         break;
      }
   }
   if(!has_base)
   {
      int sz2 = ArraySize(out_horizons);
      ArrayResize(out_horizons, sz2 + 1);
      out_horizons[sz2] = b;
   }

   if(ArraySize(out_horizons) <= 0)
   {
      ArrayResize(out_horizons, 1);
      out_horizons[0] = b;
   }

   FXAI_SortIntAsc(out_horizons);
}

int FXAI_GetMaxConfiguredHorizon(const int fallback_h)
{
   int hmax = FXAI_ClampHorizon(fallback_h);
   for(int i=0; i<ArraySize(g_horizon_minutes); i++)
   {
      int h = FXAI_ClampHorizon(g_horizon_minutes[i]);
      if(h > hmax) hmax = h;
   }
   return hmax;
}

int FXAI_GetHorizonSlot(const int horizon_minutes)
{
   int n = ArraySize(g_horizon_minutes);
   if(n <= 0) return 0;
   if(n > FXAI_MAX_HORIZONS) n = FXAI_MAX_HORIZONS;

   int h = FXAI_ClampHorizon(horizon_minutes);
   int best = 0;
   int best_diff = MathAbs(FXAI_ClampHorizon(g_horizon_minutes[0]) - h);
   for(int i=1; i<n; i++)
   {
      int hv = FXAI_ClampHorizon(g_horizon_minutes[i]);
      int d = MathAbs(hv - h);
      if(d < best_diff)
      {
         best_diff = d;
         best = i;
      }
   }
   if(best < 0) best = 0;
   if(best >= FXAI_MAX_HORIZONS) best = FXAI_MAX_HORIZONS - 1;
   return best;
}

void FXAI_BuildHorizonPolicyFeatures(const int horizon_minutes,
                                     const int base_h,
                                     const double expected_abs_points,
                                     const double min_move_points,
                                     const FXAIDataSnapshot &snapshot,
                                     const double current_vol,
                                     const int regime_id,
                                     const int ai_hint,
                                     double &feat[])
{
   MqlDateTime dt;
   TimeToStruct(snapshot.bar_time, dt);
   double hold_penalty = FXAI_Clamp(AI_HorizonPenaltyPerMinute, 0.0, 0.02);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((expected_abs_points - min_move_points) / MathMax(min_move_points, 0.50), -4.0, 6.0) / 4.0;
   feat[2] = FXAI_Clamp(expected_abs_points / MathMax(min_move_points, 0.50), 0.0, 8.0) / 4.0;
   feat[3] = 1.0 / MathSqrt((double)MathMax(horizon_minutes, 1));
   feat[4] = -hold_penalty * (double)horizon_minutes;
   feat[5] = FXAI_Clamp(FXAI_GetHorizonRegimeEdge(regime_id, horizon_minutes) / MathMax(min_move_points, 0.50), -3.0, 3.0) / 3.0;
   feat[6] = (ai_hint >= 0 ? FXAI_Clamp(FXAI_GetModelRegimeEdge(ai_hint, regime_id) / MathMax(min_move_points, 0.50), -3.0, 3.0) / 3.0 : 0.0);
   feat[7] = FXAI_Clamp(current_vol / MathMax(snapshot.point, 1e-6), 0.0, 50.0) / 25.0;
   feat[8] = FXAI_Clamp(snapshot.spread_points / MathMax(min_move_points, 0.50), 0.0, 2.0) - 0.5;
   feat[9] = ((double)dt.hour - 11.5) / 11.5;
   feat[10] = ((double)dt.min - 29.5) / 29.5;
   feat[11] = FXAI_Clamp(((double)horizon_minutes - (double)base_h) / (double)MathMax(base_h, 1), -2.0, 2.0) / 2.0;
}

int FXAI_SelectRoutedHorizon(const double &close_arr[],
                             const FXAIDataSnapshot &snapshot,
                             const double min_move_points,
                             const int ev_lookback,
                             const int fallback_h,
                             const int regime_id,
                             const int ai_hint)
{
   int base_h = FXAI_ClampHorizon(fallback_h);
   if(!AI_MultiHorizon) return base_h;
   if(ArraySize(g_horizon_minutes) <= 0) return base_h;

   double best_score = -1e18;
   int best_h = base_h;
   double hold_penalty = FXAI_Clamp(AI_HorizonPenaltyPerMinute, 0.0, 0.02);
   double current_vol = MathAbs(FXAI_SafeReturn(close_arr, 0, 1));

   for(int i=0; i<ArraySize(g_horizon_minutes); i++)
   {
      int h = FXAI_ClampHorizon(g_horizon_minutes[i]);
      double exp_abs = FXAI_EstimateExpectedAbsMovePoints(close_arr,
                                                          h,
                                                          ev_lookback,
                                                          snapshot.point);
      if(exp_abs <= 0.0) continue;

      double net = exp_abs - min_move_points;
      double score = (net / MathSqrt((double)h)) - (hold_penalty * (double)h);
      int slot = FXAI_GetHorizonSlot(h);

      // Learned global regime-aware horizon utility with UCB exploration.
      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
         slot >= 0 && slot < FXAI_MAX_HORIZONS &&
         g_horizon_regime_edge_ready[regime_id][slot])
      {
         double edge = g_horizon_regime_edge_ema[regime_id][slot];
         int obs = g_horizon_regime_obs[regime_id][slot];
         double total_obs = g_horizon_regime_total_obs[regime_id];
         if(total_obs < 1.0) total_obs = 1.0;
         double ucb = edge + (0.35 * MathSqrt(MathLog(1.0 + total_obs) / (1.0 + (double)obs)));
         score += 0.25 * (ucb / MathMax(min_move_points, 0.50));
      }

      // Optional model-specific horizon utility when single-model mode.
      if(ai_hint >= 0 && ai_hint < FXAI_AI_COUNT &&
         slot >= 0 && slot < FXAI_MAX_HORIZONS &&
         g_model_horizon_edge_ready[ai_hint][slot])
      {
         double medge = g_model_horizon_edge_ema[ai_hint][slot];
         int mobs = g_model_horizon_obs[ai_hint][slot];
         double mu = medge + (0.20 / MathSqrt(1.0 + (double)mobs));
         score += 0.15 * (mu / MathMax(min_move_points, 0.50));
      }

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_hpolicy_ready[regime_id])
      {
         double feat[FXAI_HPOL_FEATS];
         FXAI_BuildHorizonPolicyFeatures(h,
                                         base_h,
                                         exp_abs,
                                         min_move_points,
                                         snapshot,
                                         current_vol,
                                         regime_id,
                                         ai_hint,
                                         feat);

         double learned = 0.0;
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            learned += g_hpolicy_w[regime_id][k] * feat[k];
         score += 0.35 * learned;
      }

      if(score > best_score)
      {
         best_score = score;
         best_h = h;
      }
   }

   return FXAI_ClampHorizon(best_h);
}

bool FXAI_IsModelInList(const int ai_idx, const int &ai_list[])
{
   for(int i=0; i<ArraySize(ai_list); i++)
      if(ai_list[i] == ai_idx) return true;
   return false;
}

void FXAI_StackBuildFeatures(const double buy_pct,
                             const double sell_pct,
                             const double skip_pct,
                             const double avg_buy_ev,
                             const double avg_sell_ev,
                             const double min_move_points,
                             const double expected_move_points,
                             const double vol_proxy,
                             const int horizon_minutes,
                             double &feat[])
{
   double mm = MathMax(min_move_points, 0.10);
   double em = MathMax(expected_move_points, mm);
   double pb = FXAI_Clamp(buy_pct / 100.0, 0.0, 1.0);
   double ps = FXAI_Clamp(sell_pct / 100.0, 0.0, 1.0);
   double pk = FXAI_Clamp(skip_pct / 100.0, 0.0, 1.0);
   double sump = pb + ps + pk;
   if(sump <= 0.0) sump = 1.0;
   pb /= sump; ps /= sump; pk /= sump;

   double entropy = 0.0;
   if(pb > 1e-9) entropy -= pb * MathLog(pb);
   if(ps > 1e-9) entropy -= ps * MathLog(ps);
   if(pk > 1e-9) entropy -= pk * MathLog(pk);
   double entropy_norm = 1.0 - FXAI_Clamp(entropy / MathLog(3.0), 0.0, 1.0);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((pb - 0.5) * 2.0, -1.0, 1.0);
   feat[2] = FXAI_Clamp((ps - 0.5) * 2.0, -1.0, 1.0);
   feat[3] = FXAI_Clamp((pk - 0.5) * 2.0, -1.0, 1.0);
   feat[4] = FXAI_Clamp(avg_buy_ev / mm, -3.0, 3.0) / 3.0;
   feat[5] = FXAI_Clamp(avg_sell_ev / mm, -3.0, 3.0) / 3.0;
   feat[6] = FXAI_Clamp(pb - ps, -1.0, 1.0);
   feat[7] = FXAI_Clamp(MathMax(pb, ps), 0.0, 1.0);
   feat[8] = FXAI_Clamp(entropy_norm, 0.0, 1.0);
   feat[9] = FXAI_Clamp((avg_buy_ev - avg_sell_ev) / mm, -4.0, 4.0) / 4.0;
   feat[10] = FXAI_Clamp(min_move_points / em, 0.0, 2.0) - 0.5;
   feat[11] = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);
   feat[12] = FXAI_Clamp((double)horizon_minutes / (double)MathMax(FXAI_GetMaxConfiguredHorizon(horizon_minutes), 1), 0.0, 1.0);
   feat[13] = FXAI_Clamp(pk - MathMax(pb, ps), -1.0, 1.0);
}

void FXAI_StackPredict(const int regime_id, const double &feat[], double &probs[])
{
   if(ArraySize(probs) != 3) ArrayResize(probs, 3);
   probs[0] = 0.3333;
   probs[1] = 0.3333;
   probs[2] = 0.3334;

   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   if(!g_stack_ready[r])
   {
      double p_sell = FXAI_Clamp(0.32 + (0.35 * feat[2]) - (0.12 * feat[3]) + (0.18 * feat[5]) - (0.08 * feat[10]), 0.01, 0.98);
      double p_buy  = FXAI_Clamp(0.32 + (0.35 * feat[1]) - (0.12 * feat[3]) + (0.18 * feat[4]) - (0.08 * feat[10]), 0.01, 0.98);
      double p_skip = FXAI_Clamp(0.26 + (0.40 * feat[3]) + (0.18 * feat[10]) - (0.10 * feat[8]), 0.01, 0.98);
      double s0 = p_sell + p_buy + p_skip;
      if(s0 <= 0.0) s0 = 1.0;
      probs[0] = p_sell / s0;
      probs[1] = p_buy / s0;
      probs[2] = p_skip / s0;
      return;
   }

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z[3];
   for(int c=0; c<3; c++)
   {
      z[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z[0];
   if(z[1] > zmax) zmax = z[1];
   if(z[2] > zmax) zmax = z[2];
   double e0 = MathExp(z[0] - zmax);
   double e1 = MathExp(z[1] - zmax);
   double e2 = MathExp(z[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;
}

void FXAI_StackUpdate(const int regime_id,
                      const int label_class,
                      const double &feat[],
                      const double sample_weight)
{
   if(label_class < 0 || label_class > 2) return;
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z_out[3];
   double probs[3];
   for(int c=0; c<3; c++)
   {
      z_out[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z_out[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z_out[0];
   if(z_out[1] > zmax) zmax = z_out[1];
   if(z_out[2] > zmax) zmax = z_out[2];
   double e0 = MathExp(z_out[0] - zmax);
   double e1 = MathExp(z_out[1] - zmax);
   double e2 = MathExp(z_out[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;

   double lr = 0.025 / MathSqrt(1.0 + 0.02 * (double)g_stack_obs[r]);
   lr = FXAI_Clamp(lr, 0.002, 0.025);
   double sw = FXAI_Clamp(sample_weight, 0.20, 7.50);

   double delta_out[3];
   for(int c=0; c<3; c++)
   {
      double target = (c == label_class ? 1.0 : 0.0);
      delta_out[c] = FXAI_Clamp((target - probs[c]) * sw, -3.0, 3.0);
   }

   double delta_hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double back = 0.0;
      for(int c=0; c<3; c++)
         back += delta_out[c] * g_stack_w2[r][c][h];
      delta_hidden[h] = back * (1.0 - hidden[h] * hidden[h]);
      delta_hidden[h] = FXAI_Clamp(delta_hidden[h], -3.0, 3.0);
   }

   for(int c=0; c<3; c++)
   {
      g_stack_b2[r][c] += lr * delta_out[c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         double reg = 0.0005 * g_stack_w2[r][c][h];
         g_stack_w2[r][c][h] += lr * (delta_out[c] * hidden[h] - reg);
      }
   }

   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      g_stack_b1[r][h] += lr * delta_hidden[h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
      {
         double reg = (k == 0 ? 0.0 : 0.0004 * g_stack_w1[r][h][k]);
         g_stack_w1[r][h][k] += lr * (delta_hidden[h] * feat[k] - reg);
      }
   }

   g_stack_obs[r]++;
   if(g_stack_obs[r] > 200000) g_stack_obs[r] = 200000;
   g_stack_ready[r] = true;
}

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
      else w = wm;                  // detailed cross-symbol context
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

void FXAI_ResetRegimeCalibration()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SELL] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_BUY] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SKIP] = 1.2;
         g_regime_total[ai][r] = 3.2;
      }
   }
   g_regime_spread_ema = 0.0;
   g_regime_vol_ema = 0.0;
   g_regime_ema_ready = false;
}

void FXAI_UpdateRegimeEMAs(const double spread_points, const double vol_proxy_abs)
{
   double sp = MathMax(0.0, spread_points);
   double vp = MathMax(0.0, vol_proxy_abs);
   if(!g_regime_ema_ready)
   {
      g_regime_spread_ema = sp;
      g_regime_vol_ema = vp;
      g_regime_ema_ready = true;
      return;
   }

   g_regime_spread_ema = (0.98 * g_regime_spread_ema) + (0.02 * sp);
   g_regime_vol_ema = (0.98 * g_regime_vol_ema) + (0.02 * vp);
}

int FXAI_GetRegimeId(const datetime sample_time,
                     const double spread_points,
                     const double vol_proxy_abs)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = (g_regime_ema_ready ? MathMax(g_regime_spread_ema, 0.10) : MathMax(spread_points, 0.10));
   double vp_ref = (g_regime_ema_ready ? MathMax(g_regime_vol_ema, 1e-6) : MathMax(MathAbs(vol_proxy_abs), 1e-6));

   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_REGIME_COUNT) regime = FXAI_REGIME_COUNT - 1;
   return regime;
}

void FXAI_ApplyRegimeCalibration(const int ai_idx, const int regime_id, double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;

   double total = g_regime_total[ai_idx][regime_id];
   if(total < 8.0) return;

   double prior[3];
   double prior_total = 0.0;
   for(int c=0; c<3; c++)
      prior_total += g_regime_class_mass[ai_idx][regime_id][c];
   if(prior_total <= 1e-9) return;
   for(int c=0; c<3; c++)
      prior[c] = g_regime_class_mass[ai_idx][regime_id][c] / prior_total;

   double strength = FXAI_Clamp((total - 8.0) / 220.0, 0.0, 0.45);
   double s = 0.0;
   for(int c=0; c<3; c++)
   {
      probs[c] = FXAI_Clamp(((1.0 - strength) * probs[c]) + (strength * prior[c]), 0.0005, 0.9990);
      s += probs[c];
   }
   if(s <= 0.0) s = 1.0;
   for(int c=0; c<3; c++) probs[c] /= s;
}

void FXAI_UpdateRegimeCalibration(const int ai_idx,
                                  const int regime_id,
                                  const int label_class,
                                  const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double w = 1.0 + (0.5 * p_true);
   g_regime_class_mass[ai_idx][regime_id][label_class] += w;
   g_regime_total[ai_idx][regime_id] += w;

   if(g_regime_total[ai_idx][regime_id] > 20000.0)
   {
      for(int c=0; c<3; c++)
         g_regime_class_mass[ai_idx][regime_id][c] *= 0.5;
      g_regime_total[ai_idx][regime_id] *= 0.5;
   }
}

void FXAI_ResetModelPerformanceState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   g_model_meta_weight[ai_idx] = 1.0;
   g_model_global_edge_ema[ai_idx] = 0.0;
   g_model_global_edge_ready[ai_idx] = false;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_model_regime_edge_ema[ai_idx][r] = 0.0;
      g_model_regime_edge_ready[ai_idx][r] = false;
      g_model_regime_obs[ai_idx][r] = 0;
   }
}

double FXAI_GetModelRegimeEdge(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_model_regime_edge_ready[ai_idx][regime_id])
      return g_model_regime_edge_ema[ai_idx][regime_id];
   if(g_model_global_edge_ready[ai_idx]) return g_model_global_edge_ema[ai_idx];
   return 0.0;
}

double FXAI_GetHorizonRegimeEdge(const int regime_id, const int horizon_minutes)
{
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_horizon_regime_edge_ready[regime_id][hslot])
   {
      return g_horizon_regime_edge_ema[regime_id][hslot];
   }
   return 0.0;
}

bool FXAI_IsModelPruned(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return true;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.0, 3.0);
   if(rel < 0.30) return true;

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      int obs = g_model_regime_obs[ai_idx][regime_id];
      if(obs >= 24)
      {
         double edge = g_model_regime_edge_ema[ai_idx][regime_id];
         if(edge < -0.35) return true;
      }
   }

   if(g_model_global_edge_ready[ai_idx] && g_model_global_edge_ema[ai_idx] < -0.45)
      return true;
   return false;
}

double FXAI_GetModelMetaScore(const int ai_idx,
                              const int regime_id,
                              const double min_move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
   double meta = FXAI_Clamp(g_model_meta_weight[ai_idx], 0.20, 3.00);
   double edge = FXAI_GetModelRegimeEdge(ai_idx, regime_id);
   double edge_scale = 1.0 + FXAI_Clamp(edge / MathMax(min_move_points, 0.50), -0.70, 1.20);
   if(edge_scale < 0.15) edge_scale = 0.15;
   return rel * meta * edge_scale;
}

void FXAI_UpdateModelPerformance(const int ai_idx,
                                 const int regime_id,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const int horizon_minutes,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double min_mv = MathMax(min_move_points, 0.10);
   double realized_net = 0.0;
   double opportunity_penalty = 0.0;
   if(signal == 0 || signal == 1)
   {
      realized_net = FXAI_RealizedNetPointsForSignal(signal,
                                                     realized_move_points,
                                                     min_mv,
                                                     horizon_minutes);
   }
   else
   {
      if(label_class == (int)FXAI_LABEL_SKIP) realized_net = 0.05 * min_mv;
      else
      {
         opportunity_penalty = FXAI_Clamp((MathAbs(realized_move_points) - min_mv), 0.0, 8.0 * min_mv);
         realized_net = -0.25 * opportunity_penalty;
      }
   }

   double pred_edge = 0.0;
   if(signal == 1)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move_points - min_mv;
   else if(signal == 0)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move_points - min_mv;

   double err = FXAI_Clamp(realized_net - pred_edge, -8.0 * min_mv, 8.0 * min_mv);
   double alpha = 0.04;

   if(!g_model_global_edge_ready[ai_idx])
   {
      g_model_global_edge_ema[ai_idx] = realized_net;
      g_model_global_edge_ready[ai_idx] = true;
   }
   else
   {
      g_model_global_edge_ema[ai_idx] = (1.0 - alpha) * g_model_global_edge_ema[ai_idx] + alpha * realized_net;
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      if(!g_model_regime_edge_ready[ai_idx][regime_id])
      {
         g_model_regime_edge_ema[ai_idx][regime_id] = realized_net;
         g_model_regime_edge_ready[ai_idx][regime_id] = true;
      }
      else
      {
         g_model_regime_edge_ema[ai_idx][regime_id] =
            (1.0 - alpha) * g_model_regime_edge_ema[ai_idx][regime_id] + alpha * realized_net;
      }
      g_model_regime_obs[ai_idx][regime_id]++;
      if(g_model_regime_obs[ai_idx][regime_id] > 200000)
         g_model_regime_obs[ai_idx][regime_id] = 200000;
   }

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS)
   {
      if(!g_model_horizon_edge_ready[ai_idx][hslot])
      {
         g_model_horizon_edge_ema[ai_idx][hslot] = realized_net;
         g_model_horizon_edge_ready[ai_idx][hslot] = true;
      }
      else
      {
         g_model_horizon_edge_ema[ai_idx][hslot] =
            (1.0 - alpha) * g_model_horizon_edge_ema[ai_idx][hslot] + alpha * realized_net;
      }
      g_model_horizon_obs[ai_idx][hslot]++;
      if(g_model_horizon_obs[ai_idx][hslot] > 200000)
         g_model_horizon_obs[ai_idx][hslot] = 200000;

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         if(!g_horizon_regime_edge_ready[regime_id][hslot])
         {
            g_horizon_regime_edge_ema[regime_id][hslot] = realized_net;
            g_horizon_regime_edge_ready[regime_id][hslot] = true;
         }
         else
         {
            g_horizon_regime_edge_ema[regime_id][hslot] =
               (1.0 - alpha) * g_horizon_regime_edge_ema[regime_id][hslot] + alpha * realized_net;
         }
         g_horizon_regime_obs[regime_id][hslot]++;
         if(g_horizon_regime_obs[regime_id][hslot] > 200000)
            g_horizon_regime_obs[regime_id][hslot] = 200000;
         g_horizon_regime_total_obs[regime_id] += 1.0;
         if(g_horizon_regime_total_obs[regime_id] > 1e9)
            g_horizon_regime_total_obs[regime_id] = 1e9;
      }
   }

   // Online utility-driven threshold adaptation (model + regime + horizon aware).
   if(signal == 1 || signal == 0)
   {
      double utility = FXAI_Clamp(realized_net / min_mv, -2.5, 2.5);
      double mag = MathAbs(utility);
      double step = 0.004 + (0.004 * mag);
      step = FXAI_Clamp(step, 0.002, 0.02);

      double buy_u = g_model_buy_thr[ai_idx];
      double sell_u = g_model_sell_thr[ai_idx];
      if(signal == 1)
      {
         if(utility >= 0.0) buy_u -= step * MathMin(utility, 1.5) * 0.8;
         else               buy_u += step * mag;
      }
      else
      {
         // sell_thr is inverse-coded: higher sell_thr means looser sell gate.
         if(utility >= 0.0) sell_u += step * MathMin(utility, 1.5) * 0.8;
         else               sell_u -= step * mag;
      }
      g_model_buy_thr[ai_idx] = FXAI_Clamp(buy_u, 0.50, 0.95);
      g_model_sell_thr[ai_idx] = FXAI_Clamp(sell_u, 0.05, 0.50);
      FXAI_SanitizeThresholdPair(g_model_buy_thr[ai_idx], g_model_sell_thr[ai_idx]);
      g_model_thr_ready[ai_idx] = true;

      int hslot_thr = FXAI_GetHorizonSlot(horizon_minutes);
      if(hslot_thr >= 0 && hslot_thr < FXAI_MAX_HORIZONS)
      {
         double bh = g_model_buy_thr_horizon[ai_idx][hslot_thr];
         double sh = g_model_sell_thr_horizon[ai_idx][hslot_thr];
         if(!g_model_thr_horizon_ready[ai_idx][hslot_thr])
         {
            bh = g_model_buy_thr[ai_idx];
            sh = g_model_sell_thr[ai_idx];
            g_model_thr_horizon_ready[ai_idx][hslot_thr] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) bh -= step * MathMin(utility, 1.5);
            else               bh += step * mag;
         }
         else
         {
            if(utility >= 0.0) sh += step * MathMin(utility, 1.5);
            else               sh -= step * mag;
         }
         g_model_buy_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(bh, 0.50, 0.95);
         g_model_sell_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(sh, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_horizon[ai_idx][hslot_thr],
                                    g_model_sell_thr_horizon[ai_idx][hslot_thr]);
      }

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         double br = g_model_buy_thr_regime[ai_idx][regime_id];
         double sr = g_model_sell_thr_regime[ai_idx][regime_id];
         if(!g_model_thr_regime_ready[ai_idx][regime_id])
         {
            br = g_model_buy_thr[ai_idx];
            sr = g_model_sell_thr[ai_idx];
            g_model_thr_regime_ready[ai_idx][regime_id] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) br -= step * MathMin(utility, 1.5);
            else               br += step * mag;
         }
         else
         {
            if(utility >= 0.0) sr += step * MathMin(utility, 1.5);
            else               sr -= step * mag;
         }
         g_model_buy_thr_regime[ai_idx][regime_id] = FXAI_Clamp(br, 0.50, 0.95);
         g_model_sell_thr_regime[ai_idx][regime_id] = FXAI_Clamp(sr, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_regime[ai_idx][regime_id],
                                    g_model_sell_thr_regime[ai_idx][regime_id]);

         int hslot_bank = FXAI_GetHorizonSlot(horizon_minutes);
         if(hslot_bank >= 0 && hslot_bank < FXAI_MAX_HORIZONS)
         {
            double bb = g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank];
            double sb = g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank];
            if(!g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank])
            {
               bb = g_model_buy_thr_regime[ai_idx][regime_id];
               sb = g_model_sell_thr_regime[ai_idx][regime_id];
               g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank] = true;
            }
            if(signal == 1)
            {
               if(utility >= 0.0) bb -= step * MathMin(utility, 1.5);
               else               bb += step * mag;
            }
            else
            {
               if(utility >= 0.0) sb += step * MathMin(utility, 1.5);
               else               sb -= step * mag;
            }
            g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(bb, 0.50, 0.95);
            g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(sb, 0.05, 0.50);
            FXAI_SanitizeThresholdPair(g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank],
                                       g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank]);
         }
      }
   }

   double grad = FXAI_Clamp(err / min_mv, -2.0, 2.0);
   double lr = 0.015;
   double meta_new = g_model_meta_weight[ai_idx] + lr * grad;
   g_model_meta_weight[ai_idx] = FXAI_Clamp(meta_new, 0.20, 3.00);
}

void FXAI_ResetModelAuxState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   g_model_reliability[ai_idx] = 1.0;
   g_model_abs_move_ema[ai_idx] = 0.0;
   g_model_abs_move_ready[ai_idx] = false;
   FXAI_ResetModelPerformanceState(ai_idx);
}

void FXAI_UpdateModelMoveStats(const int ai_idx, const double move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   double abs_move = MathAbs(move_points);
   if(!MathIsValidNumber(abs_move) || abs_move <= 0.0) return;

   if(!g_model_abs_move_ready[ai_idx])
   {
      g_model_abs_move_ema[ai_idx] = abs_move;
      g_model_abs_move_ready[ai_idx] = true;
      return;
   }

   g_model_abs_move_ema[ai_idx] = (0.95 * g_model_abs_move_ema[ai_idx]) + (0.05 * abs_move);
}

double FXAI_GetModelExpectedMove(const int ai_idx, const double fallback_move)
{
   if(ai_idx >= 0 && ai_idx < FXAI_AI_COUNT && g_model_abs_move_ready[ai_idx] && g_model_abs_move_ema[ai_idx] > 0.0)
      return g_model_abs_move_ema[ai_idx];
   return fallback_move;
}

void FXAI_UpdateModelReliability(const int ai_idx,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   int best = 0;
   for(int c=1; c<3; c++)
      if(probs[c] > probs[best]) best = c;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double min_mv = MathMax(min_move_points, 0.10);
   double target = 1.0;

   if(signal == 1 || signal == 0)
   {
      double net_points = (signal == 1 ? realized_move_points : -realized_move_points) - min_mv;
      double edge_norm = FXAI_Clamp(net_points / min_mv, -2.5, 2.5);
      int pred_class = (signal == 1 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double cls_bonus = (pred_class == label_class ? 0.20 : -0.20);
      if(label_class == (int)FXAI_LABEL_SKIP) cls_bonus -= 0.15;
      double exp_mv = MathMax(expected_move_points, min_mv);
      double exp_fit = 1.0 - FXAI_Clamp(MathAbs(MathAbs(realized_move_points) - exp_mv) / MathMax(exp_mv, 0.10), 0.0, 1.5);
      target = 1.0 + (0.35 * edge_norm) + cls_bonus + (0.10 * (p_true - 0.5) * 2.0) + (0.08 * exp_fit);
   }
   else
   {
      // Abstention-aware: reward correct skips, penalize missed opportunities.
      if(label_class == (int)FXAI_LABEL_SKIP)
      {
         target = 1.10 + (0.10 * p_true);
      }
      else
      {
         double opportunity = FXAI_Clamp((MathAbs(realized_move_points) - min_mv) / min_mv, 0.0, 3.0);
         target = 0.95 - (0.20 * opportunity);
      }
   }

   if(best == label_class) target += 0.05;
   target = FXAI_Clamp(target, 0.20, 2.80);
   g_model_reliability[ai_idx] = FXAI_Clamp((0.97 * g_model_reliability[ai_idx]) + (0.03 * target), 0.20, 3.00);
}

void FXAI_ResetReliabilityPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_rel_pending_head[ai] = 0;
      g_rel_pending_tail[ai] = 0;
      for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
      {
         g_rel_pending_seq[ai][k] = -1;
         g_rel_pending_signal[ai][k] = -1;
         g_rel_pending_regime[ai][k] = -1;
         g_rel_pending_expected_move[ai][k] = 0.0;
         g_rel_pending_horizon[ai][k] = default_h;
      }
   }
   g_rel_clock_bar_time = 0;
   g_rel_clock_seq = 0;
}

void FXAI_ResetHorizonPolicyPending()
{
   g_hpolicy_pending_head = 0;
   g_hpolicy_pending_tail = 0;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_hpolicy_pending_seq[k] = -1;
      g_hpolicy_pending_regime[k] = 0;
      g_hpolicy_pending_horizon[k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_hpolicy_pending_min_move[k] = 0.0;
      for(int j=0; j<FXAI_HPOL_FEATS; j++)
         g_hpolicy_pending_feat[k][j] = 0.0;
   }
}

void FXAI_ResetStackPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   g_stack_pending_head = 0;
   g_stack_pending_tail = 0;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_stack_pending_seq[k] = -1;
      g_stack_pending_signal[k] = -1;
      g_stack_pending_regime[k] = -1;
      g_stack_pending_horizon[k] = default_h;
      g_stack_pending_prob[k][0] = 0.0;
      g_stack_pending_prob[k][1] = 0.0;
      g_stack_pending_prob[k][2] = 0.0;
      g_stack_pending_expected_move[k] = 0.0;
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = 0.0;
   }
}

void FXAI_ResetAdaptiveRoutingState()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_model_thr_regime_ready[ai][r] = false;
         g_model_buy_thr_regime[ai][r] = g_model_buy_thr[ai];
         g_model_sell_thr_regime[ai][r] = g_model_sell_thr[ai];
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_model_horizon_edge_ema[ai][h] = 0.0;
         g_model_horizon_edge_ready[ai][h] = false;
         g_model_horizon_obs[ai][h] = 0;
      }
   }

   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_horizon_regime_total_obs[r] = 0.0;
      g_stack_ready[r] = false;
      g_stack_obs[r] = 0;
      g_hpolicy_ready[r] = false;
      g_hpolicy_obs[r] = 0;
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            g_stack_w1[r][h][k] = 0.0;
         g_stack_b1[r][h] = 0.0;
      }
      for(int c=0; c<3; c++)
      {
         g_stack_b2[r][c] = 0.0;
         for(int h=0; h<FXAI_STACK_HIDDEN; h++)
            g_stack_w2[r][c][h] = 0.0;
      }
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         g_hpolicy_w[r][k] = 0.0;
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_horizon_regime_edge_ema[r][h] = 0.0;
         g_horizon_regime_edge_ready[r][h] = false;
         g_horizon_regime_obs[r][h] = 0;
      }
   }
}

void FXAI_EnqueueStackPending(const int signal_seq,
                              const int signal,
                              const int regime_id,
                              const int horizon_minutes,
                              const double expected_move_points,
                              const double &probs[],
                              const double &feat[])
{
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;

   if(head != tail && g_stack_pending_seq[prev] == signal_seq)
   {
      g_stack_pending_signal[prev] = signal;
      g_stack_pending_regime[prev] = regime_id;
      g_stack_pending_horizon[prev] = h;
      g_stack_pending_expected_move[prev] = expected_move_points;
      g_stack_pending_prob[prev][0] = probs[0];
      g_stack_pending_prob[prev][1] = probs[1];
      g_stack_pending_prob[prev][2] = probs[2];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[prev][j] = feat[j];
      return;
   }

   g_stack_pending_seq[tail] = signal_seq;
   g_stack_pending_signal[tail] = signal;
   g_stack_pending_regime[tail] = regime_id;
   g_stack_pending_horizon[tail] = h;
   g_stack_pending_expected_move[tail] = expected_move_points;
   g_stack_pending_prob[tail][0] = probs[0];
   g_stack_pending_prob[tail][1] = probs[1];
   g_stack_pending_prob[tail][2] = probs[2];
   for(int j=0; j<FXAI_STACK_FEATS; j++)
      g_stack_pending_feat[tail][j] = feat[j];

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_stack_pending_head = head;
   }
   g_stack_pending_tail = next_tail;
}

void FXAI_UpdateStackFromPending(const int current_signal_seq,
                                 const FXAIDataSnapshot &snapshot,
                                 const int &spread_m1[],
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 const double commission_points,
                                 const double cost_buffer_points,
                                 const double ev_threshold_points)
{
   if(current_signal_seq < 0) return;
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   double keep_feat[][FXAI_STACK_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);
   ArrayResize(keep_feat, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      int pending_signal = g_stack_pending_signal[idx];
      int pending_regime = g_stack_pending_regime[idx];
      int pending_h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
      bool consumed = false;

      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               int label_class = FXAI_BuildTripleBarrierLabel(idx_pred,
                                                              pending_h,
                                                              min_move_i,
                                                              ev_threshold_points,
                                                              snapshot,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              move_points);
               double feat[FXAI_STACK_FEATS];
               for(int j=0; j<FXAI_STACK_FEATS; j++)
                  feat[j] = g_stack_pending_feat[idx][j];
               double sw = FXAI_MoveEdgeWeight(move_points, min_move_i);
               if(pending_signal == -1 && label_class != (int)FXAI_LABEL_SKIP)
                  sw *= 0.80;
               FXAI_StackUpdate(pending_regime, label_class, feat, sw);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_signal, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_expected, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);
            ArrayResize(keep_feat, ks + 1);

            keep_seq[ks] = g_stack_pending_seq[idx];
            keep_signal[ks] = g_stack_pending_signal[idx];
            keep_regime[ks] = g_stack_pending_regime[idx];
            keep_horizon[ks] = g_stack_pending_horizon[idx];
            keep_expected[ks] = g_stack_pending_expected_move[idx];
            keep_prob0[ks] = g_stack_pending_prob[idx][0];
            keep_prob1[ks] = g_stack_pending_prob[idx][1];
            keep_prob2[ks] = g_stack_pending_prob[idx][2];
            for(int j=0; j<FXAI_STACK_FEATS; j++)
               keep_feat[ks][j] = g_stack_pending_feat[idx][j];
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   FXAI_ResetStackPending();
   int keep_n = ArraySize(keep_seq);
   int queue_cap = FXAI_REL_MAX_PENDING - 1;
   if(queue_cap < 0) queue_cap = 0;
   if(keep_n > queue_cap) keep_n = queue_cap;
   for(int k=0; k<keep_n; k++)
   {
      g_stack_pending_seq[k] = keep_seq[k];
      g_stack_pending_signal[k] = keep_signal[k];
      g_stack_pending_regime[k] = keep_regime[k];
      g_stack_pending_horizon[k] = keep_horizon[k];
      g_stack_pending_expected_move[k] = keep_expected[k];
      g_stack_pending_prob[k][0] = keep_prob0[k];
      g_stack_pending_prob[k][1] = keep_prob1[k];
      g_stack_pending_prob[k][2] = keep_prob2[k];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = keep_feat[k][j];
   }
   g_stack_pending_head = 0;
   g_stack_pending_tail = keep_n;
}

void FXAI_EnqueueHorizonPolicyPending(const int signal_seq,
                                      const int regime_id,
                                      const int horizon_minutes,
                                      const double min_move_points,
                                      const double &feat[])
{
   if(signal_seq < 0) return;
   int head = g_hpolicy_pending_head;
   int tail = g_hpolicy_pending_tail;
   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;

   if(head != tail && g_hpolicy_pending_seq[prev] == signal_seq)
   {
      g_hpolicy_pending_regime[prev] = regime_id;
      g_hpolicy_pending_horizon[prev] = FXAI_ClampHorizon(horizon_minutes);
      g_hpolicy_pending_min_move[prev] = min_move_points;
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         g_hpolicy_pending_feat[prev][k] = feat[k];
      return;
   }

   g_hpolicy_pending_seq[tail] = signal_seq;
   g_hpolicy_pending_regime[tail] = regime_id;
   g_hpolicy_pending_horizon[tail] = FXAI_ClampHorizon(horizon_minutes);
   g_hpolicy_pending_min_move[tail] = min_move_points;
   for(int k=0; k<FXAI_HPOL_FEATS; k++)
      g_hpolicy_pending_feat[tail][k] = feat[k];

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_hpolicy_pending_head = head;
   }
   g_hpolicy_pending_tail = next_tail;
}

void FXAI_UpdateHorizonPolicy(const int regime_id,
                              const double &feat[],
                              const double reward_scaled)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double pred = 0.0;
   for(int k=0; k<FXAI_HPOL_FEATS; k++)
      pred += g_hpolicy_w[r][k] * feat[k];

   double err = FXAI_Clamp(reward_scaled - pred, -4.0, 4.0);
   double lr = 0.020 / MathSqrt(1.0 + 0.02 * (double)g_hpolicy_obs[r]);
   lr = FXAI_Clamp(lr, 0.0015, 0.020);

   for(int k=0; k<FXAI_HPOL_FEATS; k++)
   {
      double reg = (k == 0 ? 0.0 : 0.0008 * g_hpolicy_w[r][k]);
      g_hpolicy_w[r][k] += lr * (err * feat[k] - reg);
   }

   g_hpolicy_obs[r]++;
   if(g_hpolicy_obs[r] > 200000) g_hpolicy_obs[r] = 200000;
   g_hpolicy_ready[r] = true;
}

void FXAI_UpdateHorizonPolicyFromPending(const int current_signal_seq,
                                         const FXAIDataSnapshot &snapshot,
                                         const int &spread_m1[],
                                         const double &high_arr[],
                                         const double &low_arr[],
                                         const double &close_arr[],
                                         const double commission_points,
                                         const double cost_buffer_points,
                                         const double ev_threshold_points)
{
   int head = g_hpolicy_pending_head;
   int tail = g_hpolicy_pending_tail;
   if(head == tail) return;

   int idx = head;
   int keep_seq[];
   int keep_regime[];
   int keep_horizon[];
   double keep_min_move[];
   double keep_feat[][FXAI_HPOL_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_min_move, 0);
   ArrayResize(keep_feat, 0);

   while(idx != tail)
   {
      bool consumed = false;
      int seq_pred = g_hpolicy_pending_seq[idx];
      int pending_h = FXAI_ClampHorizon(g_hpolicy_pending_horizon[idx]);
      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;
               double move_points = 0.0;
               double mfe_points = 0.0;
               double mae_points = 0.0;
               double time_to_hit_frac = 1.0;
               int path_flags = 0;
               int label_class = FXAI_BuildTripleBarrierLabelEx(idx_pred,
                                                                pending_h,
                                                                min_move_i,
                                                                ev_threshold_points,
                                                                snapshot,
                                                                high_arr,
                                                                low_arr,
                                                                close_arr,
                                                                move_points,
                                                                mfe_points,
                                                                mae_points,
                                                                time_to_hit_frac,
                                                                path_flags);
               double edge = MathMax(MathAbs(move_points) - min_move_i, 0.0);
               double reward = -0.25;
               if(label_class != (int)FXAI_LABEL_SKIP)
               {
                  double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
                  double quality = 1.0 + 0.20 * speed_bonus - 0.12 * FXAI_Clamp(mae_points / MathMax(mfe_points, min_move_i), 0.0, 3.0);
                  if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) quality -= 0.10;
                  reward = quality * edge / MathMax(min_move_i, 0.50);
               }
               reward = FXAI_Clamp(reward, -2.0, 6.0);
               double feat_local[FXAI_HPOL_FEATS];
               for(int k=0; k<FXAI_HPOL_FEATS; k++)
                  feat_local[k] = g_hpolicy_pending_feat[idx][k];
               FXAI_UpdateHorizonPolicy(g_hpolicy_pending_regime[idx], feat_local, reward);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         ArrayResize(keep_seq, ks + 1);
         ArrayResize(keep_regime, ks + 1);
         ArrayResize(keep_horizon, ks + 1);
         ArrayResize(keep_min_move, ks + 1);
         ArrayResize(keep_feat, ks + 1);
         keep_seq[ks] = g_hpolicy_pending_seq[idx];
         keep_regime[ks] = g_hpolicy_pending_regime[idx];
         keep_horizon[ks] = g_hpolicy_pending_horizon[idx];
         keep_min_move[ks] = g_hpolicy_pending_min_move[idx];
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            keep_feat[ks][k] = g_hpolicy_pending_feat[idx][k];
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   FXAI_ResetHorizonPolicyPending();
   int keep_n = ArraySize(keep_seq);
   int queue_cap = FXAI_REL_MAX_PENDING - 1;
   if(queue_cap < 0) queue_cap = 0;
   if(keep_n > queue_cap) keep_n = queue_cap;
   for(int k=0; k<keep_n; k++)
   {
      g_hpolicy_pending_seq[k] = keep_seq[k];
      g_hpolicy_pending_regime[k] = keep_regime[k];
      g_hpolicy_pending_horizon[k] = keep_horizon[k];
      g_hpolicy_pending_min_move[k] = keep_min_move[k];
      for(int j=0; j<FXAI_HPOL_FEATS; j++)
         g_hpolicy_pending_feat[k][j] = keep_feat[k][j];
   }
   g_hpolicy_pending_head = 0;
   g_hpolicy_pending_tail = keep_n;
}

void FXAI_AdvanceReliabilityClock(const datetime signal_bar)
{
   if(signal_bar <= 0) return;
   if(g_rel_clock_bar_time == 0)
   {
      g_rel_clock_bar_time = signal_bar;
      g_rel_clock_seq = 0;
      return;
   }

   if(signal_bar != g_rel_clock_bar_time)
   {
      g_rel_clock_seq++;
      g_rel_clock_bar_time = signal_bar;
   }
}

void FXAI_EnqueueReliabilityPending(const int ai_idx,
                                    const int signal_seq,
                                    const int signal,
                                    const int regime_id,
                                    const double expected_move_points,
                                    const int horizon_minutes,
                                    const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;
   if(head != tail && g_rel_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_rel_pending_prob[ai_idx][prev][0] = probs[0];
      g_rel_pending_prob[ai_idx][prev][1] = probs[1];
      g_rel_pending_prob[ai_idx][prev][2] = probs[2];
      g_rel_pending_signal[ai_idx][prev] = signal;
      g_rel_pending_regime[ai_idx][prev] = regime_id;
      g_rel_pending_expected_move[ai_idx][prev] = expected_move_points;
      g_rel_pending_horizon[ai_idx][prev] = h;
      return;
   }

   g_rel_pending_seq[ai_idx][tail] = signal_seq;
   g_rel_pending_prob[ai_idx][tail][0] = probs[0];
   g_rel_pending_prob[ai_idx][tail][1] = probs[1];
   g_rel_pending_prob[ai_idx][tail][2] = probs[2];
   g_rel_pending_signal[ai_idx][tail] = signal;
   g_rel_pending_regime[ai_idx][tail] = regime_id;
   g_rel_pending_expected_move[ai_idx][tail] = expected_move_points;
   g_rel_pending_horizon[ai_idx][tail] = h;

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_rel_pending_head[ai_idx] = head;
   }
   g_rel_pending_tail[ai_idx] = next_tail;
}

void FXAI_UpdateReliabilityFromPending(const int ai_idx,
                                      const int current_signal_seq,
                                      const FXAIDataSnapshot &snapshot,
                                      const int &spread_m1[],
                                      const datetime &time_arr[],
                                      const double &high_arr[],
                                      const double &low_arr[],
                                      const double &close_arr[],
                                      const double commission_points,
                                      const double cost_buffer_points,
                                      const double ev_threshold_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(current_signal_seq < 0) return;

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_rel_pending_seq[ai_idx][idx];
      int pending_signal = g_rel_pending_signal[ai_idx][idx];
      int pending_regime = g_rel_pending_regime[ai_idx][idx];
      double pending_expected_move = g_rel_pending_expected_move[ai_idx][idx];
      int pending_h = FXAI_ClampHorizon(g_rel_pending_horizon[ai_idx][idx]);

      double p0 = g_rel_pending_prob[ai_idx][idx][0];
      double p1 = g_rel_pending_prob[ai_idx][idx][1];
      double p2 = g_rel_pending_prob[ai_idx][idx][2];

      bool consumed = false;
      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            int idx_future = age - pending_h;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(time_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr) &&
               idx_future >= 0 && idx_future < ArraySize(close_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               int label_class = FXAI_BuildTripleBarrierLabel(idx_pred,
                                                              pending_h,
                                                              min_move_i,
                                                              ev_threshold_points,
                                                              snapshot,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              move_points);

               double probs_eval[3];
               probs_eval[0] = p0;
               probs_eval[1] = p1;
               probs_eval[2] = p2;

               FXAI_UpdateModelReliability(ai_idx,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_UpdateRegimeCalibration(ai_idx, pending_regime, label_class, probs_eval);
               FXAI_UpdateModelPerformance(ai_idx,
                                           pending_regime,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_h,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_BoostReplayPriorityByOutcome(time_arr[idx_pred],
                                                pending_h,
                                                pending_regime,
                                                label_class,
                                                pending_signal,
                                                move_points,
                                                min_move_i);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_signal, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_expected, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);

            keep_seq[ks] = seq_pred;
            keep_signal[ks] = pending_signal;
            keep_regime[ks] = pending_regime;
            keep_horizon[ks] = pending_h;
            keep_expected[ks] = pending_expected_move;
            keep_prob0[ks] = p0;
            keep_prob1[ks] = p1;
            keep_prob2[ks] = p2;
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   int keep_n = ArraySize(keep_seq);
   int queue_cap = FXAI_REL_MAX_PENDING - 1;
   if(queue_cap < 0) queue_cap = 0;
   if(keep_n > queue_cap) keep_n = queue_cap;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_rel_pending_seq[ai_idx][k] = -1;
      g_rel_pending_signal[ai_idx][k] = -1;
      g_rel_pending_regime[ai_idx][k] = -1;
      g_rel_pending_expected_move[ai_idx][k] = 0.0;
      g_rel_pending_horizon[ai_idx][k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_rel_pending_prob[ai_idx][k][0] = 0.0;
      g_rel_pending_prob[ai_idx][k][1] = 0.0;
      g_rel_pending_prob[ai_idx][k][2] = 0.0;
   }

   for(int k=0; k<keep_n; k++)
   {
      g_rel_pending_seq[ai_idx][k] = keep_seq[k];
      g_rel_pending_signal[ai_idx][k] = keep_signal[k];
      g_rel_pending_regime[ai_idx][k] = keep_regime[k];
      g_rel_pending_expected_move[ai_idx][k] = keep_expected[k];
      g_rel_pending_horizon[ai_idx][k] = keep_horizon[k];
      g_rel_pending_prob[ai_idx][k][0] = keep_prob0[k];
      g_rel_pending_prob[ai_idx][k][1] = keep_prob1[k];
      g_rel_pending_prob[ai_idx][k][2] = keep_prob2[k];
   }

   g_rel_pending_head[ai_idx] = 0;
   g_rel_pending_tail[ai_idx] = keep_n;
}

int FXAI_GetMaxPendingHorizon(const int fallback_h)
{
   int hmax = FXAI_ClampHorizon(fallback_h);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      int head = g_rel_pending_head[ai];
      int tail = g_rel_pending_tail[ai];
      int idx = head;
      while(idx != tail)
      {
         int seq_pred = g_rel_pending_seq[ai][idx];
         if(seq_pred >= 0)
         {
            int h = FXAI_ClampHorizon(g_rel_pending_horizon[ai][idx]);
            if(h > hmax) hmax = h;
         }
         idx++;
         if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
      }
   }

   int idx = g_stack_pending_head;
   while(idx != g_stack_pending_tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   idx = g_hpolicy_pending_head;
   while(idx != g_hpolicy_pending_tail)
   {
      int seq_pred = g_hpolicy_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_hpolicy_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }
   return hmax;
}

void FXAI_ProcessReliabilityBar(const string symbol)
{
   if(StringLen(symbol) <= 0) return;

   int H = FXAI_ClampHorizon(PredictionTargetMinutes);
   H = FXAI_GetMaxConfiguredHorizon(H);
   H = FXAI_GetMaxPendingHorizon(H);

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar <= 0) return;

   static string rel_symbol = "";
   static datetime rel_last_processed_bar = 0;
   static datetime rel_last_rates_bar = 0;
   static MqlRates rel_rates_m1[];
   static double rel_open_arr[];
   static double rel_high_arr[];
   static double rel_low_arr[];
   static double rel_close_arr[];
   static datetime rel_time_arr[];
   static int rel_spread_arr[];

   if(rel_symbol != symbol)
   {
      rel_symbol = symbol;
      rel_last_processed_bar = 0;
      rel_last_rates_bar = 0;
      ArrayResize(rel_rates_m1, 0);
      ArrayResize(rel_open_arr, 0);
      ArrayResize(rel_high_arr, 0);
      ArrayResize(rel_low_arr, 0);
      ArrayResize(rel_close_arr, 0);
      ArrayResize(rel_time_arr, 0);
      ArrayResize(rel_spread_arr, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   if(signal_bar == rel_last_processed_bar) return;
   rel_last_processed_bar = signal_bar;

   int needed = H + 64;
   if(needed < 128) needed = 128;
   if(needed > 1500) needed = 1500;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, rel_last_rates_bar, rel_rates_m1))
      return;

   FXAI_ExtractRatesCloseTimeSpread(rel_rates_m1, rel_close_arr, rel_time_arr, rel_spread_arr);
   FXAI_ExtractRatesOHLC(rel_rates_m1, rel_open_arr, rel_high_arr, rel_low_arr, rel_close_arr);
   if(ArraySize(rel_close_arr) <= H || ArraySize(rel_spread_arr) <= H)
      return;

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return;
   snapshot.bar_time = signal_bar;

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int signal_seq = g_rel_clock_seq;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      FXAI_UpdateReliabilityFromPending(ai_idx,
                                       signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_time_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
   }

   FXAI_UpdateStackFromPending(signal_seq,
                               snapshot,
                               rel_spread_arr,
                               rel_high_arr,
                               rel_low_arr,
                               rel_close_arr,
                               commission_points,
                               cost_buffer_points,
                               evThresholdPoints);
   FXAI_UpdateHorizonPolicyFromPending(signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
}

double FXAI_GetModelVoteWeight(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 1.0;
   return FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
}

void FXAI_DeriveAdaptiveThresholds(const double base_buy_threshold,
                                  const double base_sell_threshold,
                                  const double min_move_points,
                                  const double expected_move_points,
                                  const double vol_proxy,
                                  double &buy_min_prob,
                                  double &sell_min_prob,
                                  double &skip_min_prob)
{
   double buy_base = FXAI_Clamp(base_buy_threshold, 0.50, 0.95);
   double sell_base = FXAI_Clamp(1.0 - base_sell_threshold, 0.50, 0.95);

   double em = MathMax(expected_move_points, min_move_points + 0.10);
   double cost_ratio = FXAI_Clamp(min_move_points / em, 0.0, 2.0);
   double vol_ratio = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);

   double tighten = FXAI_Clamp(((cost_ratio - 0.35) * 0.35) + (0.10 * vol_ratio), 0.0, 0.25);

   buy_min_prob = FXAI_Clamp(buy_base + tighten, 0.50, 0.96);
   sell_min_prob = FXAI_Clamp(sell_base + tighten, 0.50, 0.96);
   skip_min_prob = FXAI_Clamp(0.45 + (0.20 * cost_ratio) + (0.10 * vol_ratio), 0.35, 0.85);
}

int FXAI_ClassSignalFromEV(const double &probs[],
                          const double buy_min_prob,
                          const double sell_min_prob,
                          const double skip_min_prob,
                          const double expected_move_points,
                          const double min_move_points,
                          const double ev_threshold_points)
{
   if(expected_move_points <= 0.0) return -1;

   double p_sell = probs[(int)FXAI_LABEL_SELL];
   double p_buy = probs[(int)FXAI_LABEL_BUY];
   double p_skip = probs[(int)FXAI_LABEL_SKIP];

   if(p_skip >= skip_min_prob) return -1;

   double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move_points - min_move_points;
   double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move_points - min_move_points;

   if(p_buy >= buy_min_prob && buy_ev >= ev_threshold_points && buy_ev > sell_ev)
      return 1;
   if(p_sell >= sell_min_prob && sell_ev >= ev_threshold_points && sell_ev > buy_ev)
      return 0;

   return -1;
}


#endif // __FXAI_ENGINE_META_MQH__
