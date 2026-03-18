#ifndef __FXAI_ENGINE_SAMPLES_MQH__
#define __FXAI_ENGINE_SAMPLES_MQH__

void FXAI_SanitizeThresholdPair(double &buy_threshold, double &sell_threshold)
{
   buy_threshold = FXAI_Clamp(buy_threshold, 0.50, 0.95);
   sell_threshold = FXAI_Clamp(sell_threshold, 0.05, 0.50);

   if(sell_threshold >= buy_threshold)
   {
      sell_threshold = FXAI_Clamp(sell_threshold, 0.05, 0.49);
      buy_threshold = FXAI_Clamp(MathMax(buy_threshold, sell_threshold + 0.01), 0.50, 0.95);
      if(sell_threshold >= buy_threshold)
      {
         sell_threshold = 0.49;
         buy_threshold = 0.50;
      }
   }
}

void FXAI_AssignExcursionsForRealizedMove(const double realized_move_points,
                                          const double best_up_points,
                                          const double best_dn_points,
                                          double &mfe_points,
                                          double &mae_points)
{
   if(realized_move_points > 0.0)
   {
      mfe_points = best_up_points;
      mae_points = best_dn_points;
      return;
   }
   if(realized_move_points < 0.0)
   {
      mfe_points = best_dn_points;
      mae_points = best_up_points;
      return;
   }

   mfe_points = MathMax(best_up_points, best_dn_points);
   mae_points = MathMax(best_up_points, best_dn_points);
}

int FXAI_BuildTripleBarrierLabelEx(const int i,
                                   const int H,
                                   const double roundtrip_cost_points,
                                   const double ev_threshold_points,
                                   const FXAIDataSnapshot &snapshot,
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   double &realized_move_points,
                                   double &mfe_points,
                                   double &mae_points,
                                   double &time_to_hit_frac,
                                   int &path_flags)
{
   realized_move_points = 0.0;
   mfe_points = 0.0;
   mae_points = 0.0;
   time_to_hit_frac = 1.0;
   path_flags = 0;
   if(i < 0 || H < 1) return (int)FXAI_LABEL_SKIP;
   if(i >= ArraySize(close_arr) || i >= ArraySize(high_arr) || i >= ArraySize(low_arr))
      return (int)FXAI_LABEL_SKIP;

   double entry = close_arr[i];
   if(entry <= 0.0 || snapshot.point <= 0.0)
      return (int)FXAI_LABEL_SKIP;

   int max_step = H;
   if(i - max_step < 0) max_step = i;
   if(max_step < 1) return (int)FXAI_LABEL_SKIP;

   double ev_min = (ev_threshold_points > 0.0 ? ev_threshold_points : 0.0);
   double barrier = roundtrip_cost_points + ev_min;
   if(barrier < 0.10) barrier = 0.10;

   // Asymmetric barrier shaping: blend short momentum and local range regime.
   // This reduces excessive SKIP labels while staying cost-aware.
   double mom = 0.0;
   if(i + 5 < ArraySize(close_arr) && snapshot.point > 0.0)
      mom = FXAI_MovePoints(close_arr[i + 5], close_arr[i], snapshot.point);
   double drift = FXAI_Clamp(mom / MathMax(barrier, 0.10), -1.0, 1.0);

   double range_sum = 0.0;
   int range_n = 0;
   for(int k=0; k<10; k++)
   {
      int ik = i + k;
      if(ik < 0 || ik >= ArraySize(high_arr) || ik >= ArraySize(low_arr)) break;
      range_sum += MathMax(0.0, (high_arr[ik] - low_arr[ik]) / snapshot.point);
      range_n++;
   }
   double range_avg = (range_n > 0 ? (range_sum / (double)range_n) : barrier);
   double vol_scale = FXAI_Clamp(range_avg / MathMax(barrier, 0.10), 0.7, 1.8);

   double buy_barrier = barrier * vol_scale * (1.0 - 0.10 * drift);
   double sell_barrier = barrier * vol_scale * (1.0 + 0.10 * drift);
   if(buy_barrier < 0.10) buy_barrier = 0.10;
   if(sell_barrier < 0.10) sell_barrier = 0.10;

   double best_up = 0.0;
   double best_dn = 0.0;

   for(int step=1; step<=max_step; step++)
   {
      int idx = i - step;
      if(idx < 0) break;
      if(idx >= ArraySize(high_arr) || idx >= ArraySize(low_arr)) break;

      double up_mv = FXAI_MovePoints(entry, high_arr[idx], snapshot.point);
      double dn_mv = FXAI_MovePoints(entry, low_arr[idx], snapshot.point);
      if(up_mv > best_up) best_up = up_mv;
      double dn_abs = MathAbs(dn_mv);
      if(dn_abs > best_dn) best_dn = dn_abs;
      bool hit_up = (up_mv >= buy_barrier);
      bool hit_dn = (dn_mv <= -sell_barrier);

      if(hit_up && !hit_dn)
      {
         realized_move_points = MathMax(up_mv, buy_barrier);
         mfe_points = best_up;
         mae_points = best_dn;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_BUY;
      }
      if(hit_dn && !hit_up)
      {
         realized_move_points = MathMin(dn_mv, -sell_barrier);
         mfe_points = best_dn;
         mae_points = best_up;
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         if(time_to_hit_frac > 0.75) path_flags |= FXAI_PATHFLAG_SLOW_HIT;
         return (int)FXAI_LABEL_SELL;
      }
      if(hit_up && hit_dn)
      {
         path_flags |= FXAI_PATHFLAG_DUAL_HIT;
         // Lower-timeframe disambiguation proxy: use close direction and
         // distance-to-barrier to reduce skip inflation on dual-hit bars.
         double close_mv = FXAI_MovePoints(entry, close_arr[idx], snapshot.point);
         double up_excess = up_mv - buy_barrier;
         double dn_excess = -dn_mv - sell_barrier;
         if(close_mv > 0.0 && up_excess >= dn_excess)
         {
            realized_move_points = MathMax(close_mv, buy_barrier);
            mfe_points = best_up;
            mae_points = best_dn;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_BUY;
         }
         if(close_mv < 0.0 && dn_excess >= up_excess)
         {
            realized_move_points = MathMin(close_mv, -sell_barrier);
            mfe_points = best_dn;
            mae_points = best_up;
            time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
            return (int)FXAI_LABEL_SELL;
         }
         realized_move_points = close_mv;
         FXAI_AssignExcursionsForRealizedMove(realized_move_points, best_up, best_dn, mfe_points, mae_points);
         time_to_hit_frac = FXAI_Clamp((double)step / (double)MathMax(max_step, 1), 0.0, 1.0);
         return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
      }
   }

   int idx_term = i - max_step;
   if(idx_term < 0) idx_term = 0;
   realized_move_points = FXAI_MovePoints(entry, close_arr[idx_term], snapshot.point);
   FXAI_AssignExcursionsForRealizedMove(realized_move_points, best_up, best_dn, mfe_points, mae_points);
   if(TradeKiller > 0 && H > TradeKiller)
      path_flags |= FXAI_PATHFLAG_KILLED_EARLY;
   return FXAI_BuildEVClassLabel(realized_move_points, roundtrip_cost_points, ev_threshold_points);
}

int FXAI_BuildTripleBarrierLabel(const int i,
                                 const int H,
                                 const double roundtrip_cost_points,
                                 const double ev_threshold_points,
                                 const FXAIDataSnapshot &snapshot,
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 double &realized_move_points)
{
   double mfe_points = 0.0;
   double mae_points = 0.0;
   double time_to_hit_frac = 1.0;
   int path_flags = 0;
   return FXAI_BuildTripleBarrierLabelEx(i,
                                         H,
                                         roundtrip_cost_points,
                                         ev_threshold_points,
                                         snapshot,
                                         high_arr,
                                         low_arr,
                                         close_arr,
                                         realized_move_points,
                                         mfe_points,
                                         mae_points,
                                         time_to_hit_frac,
                                         path_flags);
}

double FXAI_RealizedNetPointsForSignal(const int signal,
                                       const double realized_move_points,
                                       const double roundtrip_cost_points,
                                       const int horizon_minutes,
                                       const double spread_stress,
                                       const int path_flags)
{
   if(signal != 0 && signal != 1) return 0.0;

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double slippage_points = FXAI_ExecutionSlippagePoints(exec_profile,
                                                         roundtrip_cost_points,
                                                         horizon_minutes,
                                                         spread_stress,
                                                         path_flags);
   double fill_penalty_points = FXAI_ExecutionFillPenaltyPoints(exec_profile,
                                                                roundtrip_cost_points,
                                                                spread_stress,
                                                                path_flags);

   double kill_penalty = 0.0;
   if(TradeKiller > 0 && horizon_minutes > TradeKiller)
   {
      double frac_cut = 1.0 - ((double)TradeKiller / (double)horizon_minutes);
      kill_penalty = FXAI_Clamp(frac_cut * 0.10 * MathAbs(realized_move_points), 0.0, 10.0);
   }

   double gross = (signal == 1 ? realized_move_points : -realized_move_points);
   return gross - MathMax(roundtrip_cost_points, 0.0) - slippage_points - fill_penalty_points - kill_penalty;
}

void FXAI_ResetPreparedSample(FXAIPreparedSample &sample)
{
   sample.valid = false;
   sample.label_class = (int)FXAI_LABEL_SKIP;
   sample.regime_id = 0;
   sample.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   sample.horizon_slot = 0;
   sample.move_points = 0.0;
   sample.min_move_points = 0.0;
   sample.cost_points = 0.0;
   sample.sample_weight = 1.0;
   sample.quality_score = 1.0;
   sample.mfe_points = 0.0;
   sample.mae_points = 0.0;
   sample.spread_stress = 0.0;
   sample.time_to_hit_frac = 1.0;
   sample.path_flags = 0;
   sample.masked_step_target = 0.0;
   sample.next_vol_target = 0.0;
   sample.regime_shift_target = 0.0;
   sample.context_lead_target = 0.5;
   sample.sample_time = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      sample.x[k] = 0.0;
}

void FXAI_CopyPreparedSamples(const FXAIPreparedSample &src[], FXAIPreparedSample &dst[])
{
   int n = ArraySize(src);
   ArrayResize(dst, n);
   for(int i=0; i<n; i++)
      dst[i] = src[i];
}

bool FXAI_PrepareTrainingSample(const int i,
                               const int H,
                               const double commission_points,
                               const double cost_buffer_points,
                               const double ev_threshold_points,
                               const FXAIDataSnapshot &snapshot,
                               const int &spread_m1[],
                               const datetime &time_arr[],
                               const double &open_arr[],
                               const double &high_arr[],
                               const double &low_arr[],
                               const double &close_arr[],
                               const datetime &time_m5[],
                               const double &close_m5[],
                               const int &map_m5[],
                               const datetime &time_m15[],
                               const double &close_m15[],
                               const int &map_m15[],
                               const datetime &time_m30[],
                               const double &close_m30[],
                               const int &map_m30[],
                               const datetime &time_h1[],
                               const double &close_h1[],
                               const int &map_h1[],
                               const double &ctx_mean_arr[],
                               const double &ctx_std_arr[],
                               const double &ctx_up_arr[],
                               const double &ctx_extra_arr[],
                               const int norm_method_override,
                               FXAIPreparedSample &sample)
{
   FXAI_ResetPreparedSample(sample);

   if(i < 0 || i >= ArraySize(close_arr)) return false;
   bool has_label = (i - H >= 0 && i - H < ArraySize(close_arr));
   double move_points = 0.0;
   double mfe_points = 0.0;
   double mae_points = 0.0;
   double time_to_hit_frac = 1.0;
   int path_flags = 0;
   double spread_i = FXAI_GetSpreadAtIndex(i, spread_m1, snapshot.spread_points);
   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   double min_move_i = FXAI_ExecutionEntryCostPoints(spread_i,
                                                     commission_points,
                                                     cost_buffer_points,
                                                     exec_profile);
   if(min_move_i < 0.0) min_move_i = 0.0;

   int label_class = (int)FXAI_LABEL_SKIP;
   if(has_label)
      label_class = FXAI_BuildTripleBarrierLabelEx(i,
                                                   H,
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

   double ctx_mean_i = FXAI_GetArrayValue(ctx_mean_arr, i, 0.0);
   double ctx_std_i = FXAI_GetArrayValue(ctx_std_arr, i, 0.0);
   double ctx_up_i = FXAI_GetArrayValue(ctx_up_arr, i, 0.5);

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method =
      (norm_method_override >= 0 ? FXAI_SanitizeNormMethod(norm_method_override)
                                 : FXAI_GetFeatureNormalizationMethod());
   double feat[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(i,
                                spread_i,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
                                spread_m1,
                                time_m5,
                                close_m5,
                                map_m5,
                                time_m15,
                                close_m15,
                                map_m15,
                                time_m30,
                                close_m30,
                                map_m30,
                                time_h1,
                                close_h1,
                                map_h1,
                                ctx_mean_i,
                                ctx_std_i,
                                ctx_up_i,
                                ctx_extra_arr,
                                norm_method,
                                feat))
      return false;

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev_feat = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_prev[f] = 0.0;

   if(need_prev && (i + 1) < ArraySize(close_arr))
   {
      double spread_prev = FXAI_GetSpreadAtIndex(i + 1, spread_m1, spread_i);
      double ctx_mean_prev = FXAI_GetArrayValue(ctx_mean_arr, i + 1, ctx_mean_i);
      double ctx_std_prev = FXAI_GetArrayValue(ctx_std_arr, i + 1, ctx_std_i);
      double ctx_up_prev = FXAI_GetArrayValue(ctx_up_arr, i + 1, ctx_up_i);

      has_prev_feat = FXAI_ComputeFeatureVector(i + 1,
                                               spread_prev,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
                                               spread_m1,
                                               time_m5,
                                               close_m5,
                                               map_m5,
                                               time_m15,
                                               close_m15,
                                               map_m15,
                                               time_m30,
                                               close_m30,
                                               map_m30,
                                               time_h1,
                                               close_h1,
                                               map_h1,
                                               ctx_mean_prev,
                                               ctx_std_prev,
                                               ctx_up_prev,
                                               ctx_extra_arr,
                                               norm_method,
                                               feat_prev);
   }

   double feat_norm[FXAI_AI_FEATURES];
   sample.sample_time = ((i >= 0 && i < ArraySize(time_arr)) ? time_arr[i] : 0);
   FXAI_ApplyFeatureNormalization(norm_method, feat, feat_prev, has_prev_feat, sample.sample_time, feat_norm);

   if(!has_label)
      return false;

   sample.label_class = label_class;
   sample.horizon_minutes = FXAI_ClampHorizon(H);
   sample.horizon_slot = FXAI_GetHorizonSlot(sample.horizon_minutes);
   sample.move_points = move_points;
   sample.min_move_points = min_move_i;
   sample.cost_points = min_move_i;
   sample.mfe_points = mfe_points;
   sample.mae_points = mae_points;
   sample.time_to_hit_frac = time_to_hit_frac;
   sample.path_flags = path_flags;
   double spread_ref = FXAI_GetIntArrayMean(spread_m1, i, 64, snapshot.spread_points);
   double vol_ref = FXAI_RollingAbsReturn(close_arr, i, 64);
   if(vol_ref < 1e-6) vol_ref = MathAbs(feat[5]);
   sample.regime_id = FXAI_GetStaticRegimeId(sample.sample_time, spread_i, spread_ref, MathAbs(feat[5]), vol_ref);
   double edge = MathMax(MathAbs(move_points) - min_move_i, 0.0);
   double spread_peak = spread_i;
   int spread_n = 0;
   double spread_sum = 0.0;
   int max_step = H;
   if(i - max_step < 0) max_step = i;
   for(int k=0; k<=max_step; k++)
   {
      int idx_sp = i - k;
      if(idx_sp < 0 || idx_sp >= ArraySize(spread_m1)) break;
      double sp = FXAI_GetSpreadAtIndex(idx_sp, spread_m1, spread_i);
      if(sp > spread_peak) spread_peak = sp;
      spread_sum += sp;
      spread_n++;
   }
   double spread_avg = (spread_n > 0 ? spread_sum / (double)spread_n : spread_i);
   double spread_stress = FXAI_Clamp((spread_peak - spread_avg) / MathMax(min_move_i, 0.50), 0.0, 3.0);
   if(spread_stress > 0.35) sample.path_flags |= FXAI_PATHFLAG_SPREAD_STRESS;
   sample.spread_stress = spread_stress;

   double masked_step_target = 0.0;
   if(i - 1 >= 0 && i - 1 < ArraySize(close_arr))
      masked_step_target = FXAI_MovePoints(close_arr[i], close_arr[i - 1], snapshot.point);
   sample.masked_step_target = masked_step_target;

   int aux_h = H;
   if(aux_h > 8) aux_h = 8;
   if(aux_h < 1) aux_h = 1;
   double vol_sum = 0.0;
   int vol_n = 0;
   for(int step=1; step<=aux_h; step++)
   {
      int idx_aux = i - step;
      if(idx_aux < 0 || idx_aux >= ArraySize(close_arr)) break;
      vol_sum += MathAbs(FXAI_MovePoints(close_arr[i], close_arr[idx_aux], snapshot.point));
      vol_n++;
   }
   sample.next_vol_target = (vol_n > 0 ? vol_sum / (double)vol_n : MathAbs(move_points));

   int future_idx = i - aux_h;
   int future_regime = sample.regime_id;
   if(future_idx >= 0)
   {
      double spread_f = FXAI_GetSpreadAtIndex(future_idx, spread_m1, spread_i);
      double vol_f = FXAI_RollingAbsReturn(close_arr, future_idx, 32);
      if(vol_f < 1e-6) vol_f = vol_ref;
      future_regime = FXAI_GetStaticRegimeId((future_idx < ArraySize(time_arr) ? time_arr[future_idx] : sample.sample_time),
                                             spread_f,
                                             spread_ref,
                                             vol_f,
                                             vol_f);
   }
   sample.regime_shift_target = (future_regime == sample.regime_id ? 0.0 : 1.0);

   double ctx_signal = 0.0;
   if(ctx_std_i > 1e-6)
      ctx_signal = ctx_mean_i / ctx_std_i;
   else
      ctx_signal = ctx_mean_i;
   sample.context_lead_target = FXAI_Clamp(0.5 + 0.5 * FXAI_Sign(ctx_signal) * FXAI_Sign(move_points), 0.0, 1.0);

   double quality = 1.0;
   if(label_class == (int)FXAI_LABEL_SKIP)
   {
      quality = 0.75 - (0.10 * spread_stress);
   }
   else
   {
      double mfe_ratio = mfe_points / MathMax(min_move_i, 0.50);
      double adverse_ratio = mae_points / MathMax(mfe_points, min_move_i);
      double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
      quality = 0.85 +
                0.20 * FXAI_Clamp(mfe_ratio, 0.0, 4.0) +
                0.20 * speed_bonus -
                0.15 * FXAI_Clamp(adverse_ratio, 0.0, 3.0) -
                0.10 * spread_stress;
      if((sample.path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) quality -= 0.12;
      if((sample.path_flags & FXAI_PATHFLAG_KILLED_EARLY) != 0) quality -= 0.10;
   }
   sample.quality_score = FXAI_Clamp(quality, 0.35, 2.20);

   double dir_bias = (label_class == (int)FXAI_LABEL_SKIP ? 0.85 : 1.20);
   double trade_quality_weight = sample.quality_score;
   sample.sample_weight = FXAI_Clamp(dir_bias *
                                     trade_quality_weight *
                                     (0.75 + edge / MathMax(min_move_i, 0.50)),
                                     0.25,
                                     7.50);
   FXAI_BuildInputVector(feat_norm, sample.x);
   sample.valid = true;
   return true;
}


#endif // __FXAI_ENGINE_SAMPLES_MQH__
