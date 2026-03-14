#ifndef __FXAI_META_HORIZON_MQH__
#define __FXAI_META_HORIZON_MQH__

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
                                     const double context_strength,
                                     const double context_quality,
                                     const double model_reliability_hint,
                                     double &feat[])
{
   for(int k=0; k<FXAI_HPOL_FEATS; k++)
      feat[k] = 0.0;

   MqlDateTime dt;
   TimeToStruct(snapshot.bar_time, dt);
   double hold_penalty = FXAI_Clamp(AI_HorizonPenaltyPerMinute, 0.0, 0.02);
   double mm = MathMax(min_move_points, 0.50);
   double ctx_strength = FXAI_Clamp(context_strength, 0.0, 4.0);
   double ctx_quality = FXAI_Clamp(context_quality, -1.0, 2.0);
   double rel_hint = FXAI_Clamp(model_reliability_hint, 0.0, 1.0);
   int session_bucket = FXAI_DeriveSessionBucket(snapshot.bar_time);
   double vol_points = current_vol / MathMax(snapshot.point, 1e-6);
   double net_edge = expected_abs_points - min_move_points;
   double regime_edge = FXAI_GetHorizonRegimeEdge(regime_id, horizon_minutes) / mm;
   double model_edge = (ai_hint >= 0 ? FXAI_GetModelRegimeEdge(ai_hint, regime_id) / mm : 0.0);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((expected_abs_points - min_move_points) / mm, -4.0, 6.0) / 4.0;
   feat[2] = FXAI_Clamp(expected_abs_points / mm, 0.0, 8.0) / 4.0;
   feat[3] = 1.0 / MathSqrt((double)MathMax(horizon_minutes, 1));
   feat[4] = -hold_penalty * (double)horizon_minutes;
   feat[5] = FXAI_Clamp(regime_edge, -3.0, 3.0) / 3.0;
   feat[6] = FXAI_Clamp(model_edge, -3.0, 3.0) / 3.0;
   feat[7] = FXAI_Clamp(vol_points, 0.0, 50.0) / 25.0;
   feat[8] = FXAI_Clamp(snapshot.spread_points / mm, 0.0, 2.0) - 0.5;
   feat[9] = ((double)dt.hour - 11.5) / 11.5;
   feat[10] = ((double)dt.min - 29.5) / 29.5;
   feat[11] = FXAI_Clamp(((double)horizon_minutes - (double)base_h) / (double)MathMax(base_h, 1), -2.0, 2.0) / 2.0;
   feat[12] = FXAI_Clamp((expected_abs_points - min_move_points) / MathMax((double)horizon_minutes, 1.0), -2.0, 6.0) / 4.0;
   feat[13] = FXAI_Clamp(expected_abs_points / MathSqrt((double)MathMax(horizon_minutes, 1)), 0.0, 20.0) / 10.0;
   feat[14] = FXAI_Clamp(((double)horizon_minutes / (double)MathMax(base_h, 1)) - 1.0, -2.0, 4.0) / 2.0;
   feat[15] = FXAI_Clamp((double)regime_id / (double)MathMax(FXAI_REGIME_COUNT - 1, 1), 0.0, 1.0) - 0.5;
   feat[16] = FXAI_Clamp(ctx_strength / 2.0, 0.0, 1.5) - 0.25;
   feat[17] = FXAI_Clamp(ctx_quality, -1.0, 2.0) / 2.0;
   feat[18] = rel_hint - 0.5;
   feat[19] = FXAI_Clamp(((expected_abs_points - min_move_points) / mm) * (0.5 + ctx_quality), -6.0, 6.0) / 6.0;
   feat[20] = FXAI_Clamp(vol_points * (0.25 + ctx_strength), 0.0, 80.0) / 40.0;
   feat[21] = FXAI_Clamp(snapshot.spread_points / MathMax(vol_points, 1.0), 0.0, 4.0) / 2.0 - 0.5;
   feat[22] = ((double)session_bucket / (double)MathMax(FXAI_PLUGIN_SESSION_BUCKETS - 1, 1)) - 0.5;
   feat[23] = ((double)FXAI_GetHorizonSlot(horizon_minutes) / (double)MathMax(FXAI_MAX_HORIZONS - 1, 1)) - 0.5;
   feat[24] = FXAI_Clamp((net_edge / mm) * rel_hint, -6.0, 6.0) / 6.0;
   feat[25] = FXAI_Clamp(expected_abs_points / MathMax(vol_points * MathSqrt((double)MathMax(horizon_minutes, 1)), 1.0), 0.0, 6.0) / 3.0 - 0.5;
   feat[26] = FXAI_Clamp(min_move_points / MathMax(snapshot.spread_points, 0.10), 0.0, 6.0) / 3.0 - 0.5;
   feat[27] = FXAI_Clamp(snapshot.spread_points / MathMax(expected_abs_points, mm), 0.0, 2.0) - 0.5;
   feat[28] = FXAI_Clamp((ctx_strength * (ctx_quality + 1.0)) / 4.0, 0.0, 2.0) - 0.5;
   feat[29] = FXAI_Clamp(rel_hint * (1.0 + FXAI_Clamp(regime_edge + model_edge, -2.0, 2.0)), 0.0, 2.0) - 0.5;
   feat[30] = ((double)dt.day_of_week - 2.5) / 2.5;
   feat[31] = FXAI_Clamp((hold_penalty * (double)horizon_minutes) / MathMax(MathAbs(net_edge / mm), 0.25), 0.0, 2.0) - 0.5;
   feat[32] = FXAI_Clamp(net_edge / MathMax(expected_abs_points, mm), -1.0, 1.0);
   feat[33] = FXAI_Clamp(ctx_quality * rel_hint, -1.0, 2.0) / 2.0;
   feat[34] = FXAI_Clamp(ctx_strength * rel_hint, 0.0, 4.0) / 2.0 - 0.5;
   feat[35] = FXAI_Clamp(regime_edge * MathMax(ctx_quality, 0.0), -4.0, 4.0) / 4.0;
   feat[36] = FXAI_Clamp(model_edge * MathMax(ctx_quality, 0.0), -4.0, 4.0) / 4.0;
   feat[37] = FXAI_Clamp((expected_abs_points / MathMax((double)horizon_minutes, 1.0)) / MathMax(vol_points, 0.50), 0.0, 6.0) / 3.0 - 0.5;
   feat[38] = FXAI_Clamp((double)session_bucket / (double)MathMax(FXAI_PLUGIN_SESSION_BUCKETS - 1, 1), 0.0, 1.0) * rel_hint - 0.5;
   feat[39] = FXAI_Clamp((MathAbs(net_edge) / mm) * (0.50 + MathAbs(ctx_quality)), 0.0, 8.0) / 4.0 - 0.5;
   feat[40] = FXAI_Clamp(MathAbs(regime_edge), 0.0, 4.0) / 2.0 - 0.5;
   feat[41] = FXAI_Clamp(MathAbs(model_edge), 0.0, 4.0) / 2.0 - 0.5;
   feat[42] = FXAI_Clamp(ctx_strength * MathMax(ctx_quality + 1.0, 0.0), 0.0, 8.0) / 4.0 - 0.5;
   feat[43] = FXAI_Clamp((net_edge / mm) * MathMax(rel_hint, 0.05) / MathSqrt((double)MathMax(horizon_minutes, 1)), -4.0, 4.0) / 4.0;
   feat[44] = FXAI_Clamp((hold_penalty * (double)horizon_minutes) * MathMax(vol_points, 0.25), 0.0, 6.0) / 3.0 - 0.5;
   feat[45] = FXAI_Clamp((snapshot.spread_points / MathMax(expected_abs_points + mm, mm)) * (0.50 + rel_hint), 0.0, 2.0) - 0.5;
   feat[46] = FXAI_Clamp((double)(session_bucket + 1) / (double)MathMax(FXAI_PLUGIN_SESSION_BUCKETS, 1) * (ctx_strength + 0.50), 0.0, 3.0) / 1.5 - 0.5;
   feat[47] = FXAI_Clamp(((double)base_h / (double)MathMax(horizon_minutes, 1) - 1.0) * (0.50 + MathMax(ctx_quality, 0.0)), -3.0, 3.0) / 3.0;
}

double FXAI_HorizonPolicyPredictValue(const int regime_id,
                                      const double &feat[],
                                      double &hidden[])
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
   {
      double z = g_hpolicy_b1[r][h];
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
         z += g_hpolicy_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double pred = g_hpolicy_b2[r];
   for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
      pred += g_hpolicy_w2[r][h] * hidden[h];
   return pred;
}

double FXAI_HorizonPolicyPredictValue(const int regime_id,
                                      const double &feat[])
{
   double hidden[FXAI_HPOL_HIDDEN];
   return FXAI_HorizonPolicyPredictValue(regime_id, feat, hidden);
}

int FXAI_SelectRoutedHorizon(const double &close_arr[],
                             const FXAIDataSnapshot &snapshot,
                             const double min_move_points,
                             const int ev_lookback,
                             const int fallback_h,
                             const int regime_id,
                             const int ai_hint,
                             const double context_strength,
                             const double context_quality,
                             const double model_reliability_hint)
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
                                         context_strength,
                                         context_quality,
                                         model_reliability_hint,
                                         feat);

         double learned = FXAI_HorizonPolicyPredictValue(regime_id, feat);
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

   double hidden[FXAI_HPOL_HIDDEN];
   double w2_old[FXAI_HPOL_HIDDEN];
   for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
      w2_old[h] = g_hpolicy_w2[r][h];
   double pred = FXAI_HorizonPolicyPredictValue(r, feat, hidden);

   double err = FXAI_Clamp(reward_scaled - pred, -4.0, 4.0);
   double lr = 0.020 / MathSqrt(1.0 + 0.02 * (double)g_hpolicy_obs[r]);
   lr = FXAI_Clamp(lr, 0.0015, 0.020);

   g_hpolicy_b2[r] += lr * err;
   for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
   {
      double reg2 = 0.0008 * g_hpolicy_w2[r][h];
      g_hpolicy_w2[r][h] += lr * (err * hidden[h] - reg2);
   }

   for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
   {
      double dh = (1.0 - hidden[h] * hidden[h]) * w2_old[h] * err;
      g_hpolicy_b1[r][h] += lr * dh;
      for(int k=0; k<FXAI_HPOL_FEATS; k++)
      {
         double reg1 = 0.0006 * g_hpolicy_w1[r][h][k];
         g_hpolicy_w1[r][h][k] += lr * (dh * feat[k] - reg1);
      }
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


#endif // __FXAI_META_HORIZON_MQH__
