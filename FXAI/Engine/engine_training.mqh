#ifndef __FXAI_ENGINE_TRAINING_MQH__
#define __FXAI_ENGINE_TRAINING_MQH__

void FXAI_PrecomputeTrainingSamples(const int i_start,
                                   const int i_end,
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
                                   FXAIPreparedSample &samples[])
{
   if(i_end < i_start) return;
   if(i_end < 0) return;

   int need_size = i_end + 1;
   if(ArraySize(samples) < need_size)
      ArrayResize(samples, need_size);

   // Build samples oldest -> newest (as-series: larger index is older) so
   // any stateful normalizer sees a causal timeline.
   for(int i=i_end; i>=i_start; i--)
   {
      if(i < 0 || i >= ArraySize(samples)) continue;
      FXAI_PrepareTrainingSample(i,
                                H,
                                commission_points,
                                cost_buffer_points,
                                ev_threshold_points,
                                snapshot,
                                spread_m1,
                                time_arr,
                                open_arr,
                                high_arr,
                                low_arr,
                                close_arr,
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
                                ctx_mean_arr,
                                ctx_std_arr,
                                ctx_up_arr,
                                ctx_extra_arr,
                                norm_method_override,
                                samples[i]);
   }
}

int FXAI_FindNormSampleCache(const int method_id,
                             FXAINormSampleCache &caches[])
{
   for(int i=0; i<ArraySize(caches); i++)
   {
      if(caches[i].ready && caches[i].method_id == method_id)
         return i;
   }
   return -1;
}

int FXAI_EnsureNormSampleCache(const int method_id,
                               const int i_start,
                               const int i_end,
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
                               FXAINormSampleCache &caches[])
{
   int idx = FXAI_FindNormSampleCache(method_id, caches);
   if(idx >= 0) return idx;

   int sz = ArraySize(caches);
   ArrayResize(caches, sz + 1);
   caches[sz].method_id = method_id;
   caches[sz].ready = true;
   FXAI_PrecomputeTrainingSamples(i_start,
                                  i_end,
                                  H,
                                  commission_points,
                                  cost_buffer_points,
                                  ev_threshold_points,
                                  snapshot,
                                  spread_m1,
                                  time_arr,
                                  open_arr,
                                  high_arr,
                                  low_arr,
                                  close_arr,
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
                                  ctx_mean_arr,
                                  ctx_std_arr,
                                  ctx_up_arr,
                                  ctx_extra_arr,
                                  method_id,
                                  caches[sz].samples);
   return sz;
}

void FXAI_EnsureRoutedNormCachesForSamples(const int ai_idx,
                                           const int i_start,
                                           const int i_end,
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
                                           const FXAIPreparedSample &samples[],
                                           FXAINormSampleCache &caches[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   bool needed_method[FXAI_NORM_METHOD_COUNT];
   for(int m=0; m<FXAI_NORM_METHOD_COUNT; m++)
      needed_method[m] = false;

   for(int i=end; i>=start; i--)
   {
      if(i < 0 || i >= n) continue;
      if(!samples[i].valid) continue;
      int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                         samples[i].regime_id,
                                                         samples[i].horizon_minutes);
      if(method_id < 0 || method_id >= FXAI_NORM_METHOD_COUNT) continue;
      if(needed_method[method_id]) continue;
      needed_method[method_id] = true;

      FXAI_EnsureNormSampleCache(method_id,
                                 i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
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
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 ctx_extra_arr,
                                 caches);
   }

   int default_method = (int)FXAI_GetModelNormMethodRouted(ai_idx, 0, H);
   if(default_method >= 0 && default_method < FXAI_NORM_METHOD_COUNT && !needed_method[default_method])
   {
      FXAI_EnsureNormSampleCache(default_method,
                                 i_start,
                                 i_end,
                                 H,
                                 commission_points,
                                 cost_buffer_points,
                                 ev_threshold_points,
                                 snapshot,
                                 spread_m1,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
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
                                 ctx_mean_arr,
                                 ctx_std_arr,
                                 ctx_up_arr,
                                 ctx_extra_arr,
                                 caches);
   }
}

void FXAI_GetCachedPreparedSample(const int ai_idx,
                                  const FXAIPreparedSample &reference_sample,
                                  const int sample_index,
                                  FXAINormSampleCache &caches[],
                                  FXAIPreparedSample &out_sample)
{
   out_sample = reference_sample;
   int method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                      reference_sample.regime_id,
                                                      reference_sample.horizon_minutes);
   int cache_idx = FXAI_FindNormSampleCache(method_id, caches);
   if(cache_idx < 0) return;
   if(sample_index < 0 || sample_index >= ArraySize(caches[cache_idx].samples))
      return;
   out_sample = caches[cache_idx].samples[sample_index];
}

int FXAI_FindNormInputCache(const int method_id,
                            FXAINormInputCache &caches[])
{
   for(int i=0; i<ArraySize(caches); i++)
   {
      if(caches[i].ready && caches[i].method_id == method_id)
         return i;
   }
   return -1;
}

int FXAI_EnsureNormInputCache(const int method_id,
                              const double spread_pred,
                              const int &spread_m1[],
                              const FXAIDataSnapshot &snapshot,
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
                              FXAINormInputCache &caches[])
{
   int idx = FXAI_FindNormInputCache(method_id, caches);
   if(idx >= 0) return idx;

   ENUM_FXAI_FEATURE_NORMALIZATION norm_method = FXAI_SanitizeNormMethod(method_id);
   double ctx_mean_pred = FXAI_GetArrayValue(ctx_mean_arr, 0, 0.0);
   double ctx_std_pred = FXAI_GetArrayValue(ctx_std_arr, 0, 0.0);
   double ctx_up_pred = FXAI_GetArrayValue(ctx_up_arr, 0, 0.5);
   double feat_pred[FXAI_AI_FEATURES];
   if(!FXAI_ComputeFeatureVector(0,
                                 spread_pred,
                                 time_arr,
                                 open_arr,
                                 high_arr,
                                 low_arr,
                                 close_arr,
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
                                 ctx_mean_pred,
                                 ctx_std_pred,
                                 ctx_up_pred,
                                 ctx_extra_arr,
                                 norm_method,
                                 feat_pred))
      return -1;

   bool need_prev = FXAI_FeatureNormNeedsPrevious(norm_method);
   bool has_prev_feat = false;
   double feat_prev[FXAI_AI_FEATURES];
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      feat_prev[f] = 0.0;

   if(need_prev && ArraySize(close_arr) > 1)
   {
      double spread_prev = FXAI_GetSpreadAtIndex(1, spread_m1, spread_pred);
      double ctx_mean_prev = FXAI_GetArrayValue(ctx_mean_arr, 1, ctx_mean_pred);
      double ctx_std_prev = FXAI_GetArrayValue(ctx_std_arr, 1, ctx_std_pred);
      double ctx_up_prev = FXAI_GetArrayValue(ctx_up_arr, 1, ctx_up_pred);
      has_prev_feat = FXAI_ComputeFeatureVector(1,
                                               spread_prev,
                                               time_arr,
                                               open_arr,
                                               high_arr,
                                               low_arr,
                                               close_arr,
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
   FXAI_ApplyFeatureNormalization(norm_method,
                                  feat_pred,
                                  feat_prev,
                                  has_prev_feat,
                                  snapshot.bar_time,
                                  feat_norm);

   int sz = ArraySize(caches);
   ArrayResize(caches, sz + 1);
   caches[sz].method_id = method_id;
   caches[sz].ready = true;
   FXAI_BuildInputVector(feat_norm, caches[sz].x);
   return sz;
}

void FXAI_ApplyPreparedSampleToModel(const int ai_idx,
                                    CFXAIAIPlugin &plugin,
                                    const FXAIPreparedSample &sample,
                                    const FXAIAIHyperParams &hp)
{
   if(!sample.valid) return;

   FXAIAITrainRequestV4 s3;
   s3.valid = sample.valid;
   s3.ctx.api_version = FXAI_API_VERSION_V4;
   s3.ctx.regime_id = sample.regime_id;
   s3.ctx.session_bucket = FXAI_DeriveSessionBucket(sample.sample_time);
   s3.ctx.horizon_minutes = sample.horizon_minutes;
   s3.ctx.feature_schema_id = 1;
   s3.ctx.normalization_method_id = (int)FXAI_GetModelNormMethodRouted(ai_idx,
                                                                       sample.regime_id,
                                                                       sample.horizon_minutes);
   s3.ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, sample.horizon_minutes);
   s3.ctx.cost_points = sample.cost_points;
   s3.ctx.min_move_points = sample.min_move_points;
   s3.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
   s3.ctx.sample_time = sample.sample_time;
   s3.label_class = sample.label_class;
   s3.move_points = sample.move_points;
   s3.sample_weight = sample.sample_weight;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      s3.x[k] = sample.x[k];

   FXAI_TrainViaV4(plugin, s3, hp);
   FXAI_UpdateModelMoveStats(ai_idx, sample.move_points);
}

void FXAI_TrainModelWindowPrepared(const int ai_idx,
                                  CFXAIAIPlugin &plugin,
                                  const int i_start,
                                  const int i_end,
                                  const int epochs,
                                  const FXAIAIHyperParams &hp,
                                  const FXAIPreparedSample &samples[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
         FXAI_ApplyPreparedSampleToModel(ai_idx, plugin, samples[i], hp);
   }
}

void FXAI_ApplyPreparedSampleToModelRouted(const int ai_idx,
                                           CFXAIAIPlugin &plugin,
                                           const FXAIPreparedSample &sample)
{
   if(!sample.valid) return;
   FXAIAIHyperParams hp_sample;
   FXAI_GetModelHyperParamsRouted(ai_idx, sample.regime_id, sample.horizon_minutes, hp_sample);
   FXAI_ApplyPreparedSampleToModel(ai_idx, plugin, sample, hp_sample);
}

void FXAI_TrainModelWindowPreparedRouted(const int ai_idx,
                                         CFXAIAIPlugin &plugin,
                                         const int i_start,
                                         const int i_end,
                                         const int epochs,
                                         const FXAIPreparedSample &samples[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, samples[i]);
   }
}

void FXAI_TrainModelWindowPreparedRoutedCached(const int ai_idx,
                                               CFXAIAIPlugin &plugin,
                                               const int i_start,
                                               const int i_end,
                                               const int epochs,
                                               const FXAIPreparedSample &samples[],
                                               FXAINormSampleCache &caches[])
{
   if(i_end < i_start || epochs <= 0) return;
   int n = ArraySize(samples);
   if(n <= 0) return;

   int start = i_start;
   int end = i_end;
   if(start < 0) start = 0;
   if(end >= n) end = n - 1;
   if(end < start) return;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int i=end; i>=start; i--)
      {
         if(i < 0 || i >= ArraySize(samples)) continue;
         if(!samples[i].valid) continue;
         FXAIPreparedSample train_sample;
         FXAI_GetCachedPreparedSample(ai_idx, samples[i], i, caches, train_sample);
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, train_sample);
      }
   }
}

double FXAI_CalcReplayPriority(const FXAIPreparedSample &sample)
{
   double p = sample.sample_weight;
   p += 0.35 * sample.quality_score;
   p += 0.10 * FXAI_Clamp(sample.spread_stress, 0.0, 3.0);
   if(sample.label_class != (int)FXAI_LABEL_SKIP) p += 0.20;
   if((sample.path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) p += 0.30;
   if((sample.path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0) p += 0.25;
   if((sample.path_flags & FXAI_PATHFLAG_SLOW_HIT) != 0) p += 0.10;
   return FXAI_Clamp(p, 0.25, 12.0);
}

void FXAI_AddReplaySample(const FXAIPreparedSample &sample)
{
   if(!sample.valid) return;

   int regime_id = sample.regime_id;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) regime_id = 0;
   int hslot = sample.horizon_slot;
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS)
      hslot = FXAI_GetHorizonSlot(sample.horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;

   if(sample.sample_time > 0 && g_replay_last_sample_time[hslot] == sample.sample_time)
      return;

   int slot = -1;
   if(g_replay_count < FXAI_REPLAY_CAPACITY)
   {
      for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
      {
         if(!g_replay_used[i])
         {
            slot = i;
            break;
         }
      }
   }
   else
   {
      double new_bucket = (double)g_replay_bucket_count[regime_id][hslot];
      double best_evict = -1e18;
      for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
      {
         if(!g_replay_used[i]) continue;
         int er = g_replay_samples[i].regime_id;
         int eh = g_replay_samples[i].horizon_slot;
         if(er < 0 || er >= FXAI_REGIME_COUNT) er = 0;
         if(eh < 0 || eh >= FXAI_MAX_HORIZONS) eh = 0;
         double old_bucket = (double)g_replay_bucket_count[er][eh];
         double evict_score = old_bucket - (0.25 * g_replay_priority[i]);
         if(g_replay_samples[i].label_class == (int)FXAI_LABEL_SKIP)
            evict_score += 0.10;
         if(old_bucket > new_bucket) evict_score += 0.50;
         if(evict_score > best_evict)
         {
            best_evict = evict_score;
            slot = i;
         }
      }
   }

   if(slot < 0) return;
   if(g_replay_used[slot])
   {
      int old_r = g_replay_samples[slot].regime_id;
      int old_h = g_replay_samples[slot].horizon_slot;
      if(old_r >= 0 && old_r < FXAI_REGIME_COUNT &&
         old_h >= 0 && old_h < FXAI_MAX_HORIZONS &&
         g_replay_bucket_count[old_r][old_h] > 0)
      {
         g_replay_bucket_count[old_r][old_h]--;
      }
   }
   else
   {
      g_replay_count++;
   }

   g_replay_samples[slot] = sample;
   g_replay_samples[slot].regime_id = regime_id;
   g_replay_samples[slot].horizon_slot = hslot;
   g_replay_priority[slot] = FXAI_CalcReplayPriority(sample);
   g_replay_flags[slot] = sample.path_flags;
   g_replay_used[slot] = true;
   g_replay_bucket_count[regime_id][hslot]++;
   if(sample.sample_time > 0) g_replay_last_sample_time[hslot] = sample.sample_time;
}

void FXAI_BoostReplayPriorityByOutcome(const datetime sample_time,
                                       const int horizon_minutes,
                                       const int regime_id,
                                       const int label_class,
                                       const int signal,
                                       const double move_points,
                                       const double min_move_points)
{
   if(sample_time <= 0) return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;

   double min_mv = MathMax(min_move_points, 0.50);
   double edge = MathMax(MathAbs(move_points) - min_mv, 0.0);
   double edge_ratio = FXAI_Clamp(edge / min_mv, 0.0, 4.0);

   bool false_positive = ((signal == 0 || signal == 1) && label_class == (int)FXAI_LABEL_SKIP);
   bool wrong_direction = ((signal == 1 && label_class == (int)FXAI_LABEL_SELL) ||
                           (signal == 0 && label_class == (int)FXAI_LABEL_BUY));
   bool missed_move = (signal == -1 && label_class != (int)FXAI_LABEL_SKIP && edge > 0.0);

   double base_boost = 0.0;
   int add_flags = 0;
   if(false_positive)
   {
      base_boost += 1.10 + 0.35 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_FALSE_POS;
   }
   if(wrong_direction)
   {
      base_boost += 1.35 + 0.45 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_WRONG_DIR;
   }
   if(missed_move)
   {
      base_boost += 1.00 + 0.50 * edge_ratio;
      add_flags |= FXAI_REPLAYFLAG_MISSED_MOVE;
   }
   if(base_boost <= 0.0) return;

   for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
   {
      if(!g_replay_used[i]) continue;
      if(g_replay_samples[i].sample_time != sample_time) continue;
      if(g_replay_samples[i].horizon_slot != hslot) continue;

      double boost = base_boost;
      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_replay_samples[i].regime_id == regime_id)
         boost += 0.20;
      if((g_replay_flags[i] & FXAI_PATHFLAG_DUAL_HIT) != 0)
         boost += 0.10;
      if((g_replay_flags[i] & FXAI_PATHFLAG_SPREAD_STRESS) != 0)
         boost += 0.10;

      g_replay_priority[i] = FXAI_Clamp(g_replay_priority[i] + boost, 0.25, 20.0);
      g_replay_flags[i] |= add_flags;
      break;
   }
}

void FXAI_TrainModelReplay(const int ai_idx,
                           CFXAIAIPlugin &plugin,
                           const int regime_id,
                           const int horizon_minutes,
                           const int epochs)
{
   if(epochs <= 0 || g_replay_count <= 0) return;

   FXAIAIManifestV4 manifest;
   plugin.Describe(manifest);
   if(!FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY))
      return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) hslot = 0;
   int prefer_regime = regime_id;
   if(prefer_regime < 0 || prefer_regime >= FXAI_REGIME_COUNT) prefer_regime = 0;

   int probe_limit = g_replay_count;
   if(probe_limit > 64) probe_limit = 64;

   for(int epoch=0; epoch<epochs; epoch++)
   {
      for(int draw=0; draw<FXAI_REPLAY_DRAWS; draw++)
      {
         int best_idx = -1;
         double best_score = -1e18;
         int start = (g_replay_cursor + draw * 7 + epoch * 13) % FXAI_REPLAY_CAPACITY;
         for(int p=0; p<probe_limit; p++)
         {
            int idx = (start + p) % FXAI_REPLAY_CAPACITY;
            if(!g_replay_used[idx]) continue;

            FXAIPreparedSample sample = g_replay_samples[idx];
            double score = g_replay_priority[idx];
            if(sample.regime_id == prefer_regime) score += 1.00;
            if(sample.horizon_slot == hslot) score += 0.75;
            if(sample.label_class != (int)FXAI_LABEL_SKIP) score += 0.20;
            if((g_replay_flags[idx] & FXAI_PATHFLAG_DUAL_HIT) != 0) score += 0.20;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_FALSE_POS) != 0) score += 0.35;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_MISSED_MOVE) != 0) score += 0.45;
            if((g_replay_flags[idx] & FXAI_REPLAYFLAG_WRONG_DIR) != 0) score += 0.55;
            double recency_penalty = 0.05 * MathAbs((double)(g_replay_cursor - idx));
            score -= recency_penalty;
            if(score > best_score)
            {
               best_score = score;
               best_idx = idx;
            }
         }

         if(best_idx < 0) continue;
         FXAI_ApplyPreparedSampleToModelRouted(ai_idx, plugin, g_replay_samples[best_idx]);
         g_replay_cursor = (best_idx + 1) % FXAI_REPLAY_CAPACITY;
      }
   }
}

void FXAI_BuildHyperParams(FXAIAIHyperParams &hp)
{
   hp.lr = FXAI_Clamp(AI_LearningRate, 0.001, 0.200);
   hp.l2 = FXAI_Clamp(AI_L2, 0.0, 0.100);

   hp.ftrl_alpha = FXAI_Clamp(FTRL_Alpha, 0.001, 1.000);
   hp.ftrl_beta  = FXAI_Clamp(FTRL_Beta,  0.000, 5.000);
   hp.ftrl_l1    = FXAI_Clamp(FTRL_L1,    0.000, 0.100);
   hp.ftrl_l2    = FXAI_Clamp(FTRL_L2,    0.000, 1.000);

   hp.pa_c      = FXAI_Clamp(PA_C,      0.010, 10.000);
   hp.pa_margin = FXAI_Clamp(PA_Margin, 0.100, 2.000);

   hp.xgb_lr    = FXAI_Clamp(XGB_FastLearningRate, 0.001, 0.300);
   hp.xgb_l2    = FXAI_Clamp(XGB_FastL2,           0.000, 10.000);
   hp.xgb_split = FXAI_Clamp(XGB_SplitThreshold,  -2.000, 2.000);

   hp.mlp_lr   = FXAI_Clamp(MLP_LearningRate, 0.0005, 0.0500);
   hp.mlp_l2   = FXAI_Clamp(MLP_L2,           0.0000, 0.0500);
   hp.mlp_init = FXAI_Clamp(MLP_InitScale,    0.0100, 0.5000);

   hp.tcn_layers = (double)((int)FXAI_Clamp((double)TCN_Layers, 2.0, 8.0));
   hp.tcn_kernel = (double)((int)FXAI_Clamp((double)TCN_KernelSize, 2.0, 5.0));
   hp.tcn_dilation_base = (double)((int)FXAI_Clamp((double)TCN_DilationBase, 1.0, 3.0));

   hp.quantile_lr = FXAI_Clamp(Quantile_LearningRate, 0.0001, 0.1000);
   hp.quantile_l2 = FXAI_Clamp(Quantile_L2,           0.0000, 0.1000);

   hp.enhash_lr = FXAI_Clamp(ENHash_LearningRate, 0.0005, 0.1000);
   hp.enhash_l1 = FXAI_Clamp(ENHash_L1,           0.0000, 0.1000);
   hp.enhash_l2 = FXAI_Clamp(ENHash_L2,           0.0000, 0.1000);
}

double FXAI_RandRange(const double lo, const double hi)
{
   if(hi <= lo) return lo;
   double u = (double)MathRand() / 32767.0;
   return lo + (hi - lo) * FXAI_Clamp(u, 0.0, 1.0);
}

void FXAI_ResetModelHyperParams()
{
   FXAIAIHyperParams base;
   FXAI_BuildHyperParams(base);
   double base_buy = AI_BuyThreshold;
   double base_sell = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(base_buy, base_sell);

   for(int i=0; i<FXAI_AI_COUNT; i++)
   {
      g_model_hp[i] = base;
      g_model_hp_ready[i] = false;
      g_model_norm_method[i] = (int)FXAI_GetFeatureNormalizationMethod();
      g_model_norm_ready[i] = false;
      g_model_buy_thr[i] = base_buy;
      g_model_sell_thr[i] = base_sell;
      g_model_thr_ready[i] = false;
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_model_norm_method_bank[i][r][h] = (int)FXAI_GetFeatureNormalizationMethod();
            g_model_norm_bank_ready[i][r][h] = false;
         }
         g_model_buy_thr_regime[i][r] = base_buy;
         g_model_sell_thr_regime[i][r] = base_sell;
         g_model_thr_regime_ready[i][r] = false;
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_model_hp_bank[i][r][h] = base;
            g_model_hp_bank_ready[i][r][h] = false;
            g_model_buy_thr_bank[i][r][h] = base_buy;
            g_model_sell_thr_bank[i][r][h] = base_sell;
            g_model_thr_bank_ready[i][r][h] = false;
         }
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_model_hp_horizon[i][h] = base;
         g_model_hp_horizon_ready[i][h] = false;
         g_model_norm_method_horizon[i][h] = (int)FXAI_GetFeatureNormalizationMethod();
         g_model_norm_horizon_ready[i][h] = false;
         g_model_buy_thr_horizon[i][h] = base_buy;
         g_model_sell_thr_horizon[i][h] = base_sell;
         g_model_thr_horizon_ready[i][h] = false;
         g_model_horizon_edge_ema[i][h] = 0.0;
         g_model_horizon_edge_ready[i][h] = false;
         g_model_horizon_obs[i][h] = 0;
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
         g_stack_b1[r][h] = 0.0;
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            g_stack_w1[r][h][k] = 0.0;
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

void FXAI_ResetReplayReservoir()
{
   g_replay_count = 0;
   g_replay_cursor = 0;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         g_replay_bucket_count[r][h] = 0;
   }
   for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      g_replay_last_sample_time[h] = 0;
   for(int i=0; i<FXAI_REPLAY_CAPACITY; i++)
   {
      g_replay_used[i] = false;
      g_replay_priority[i] = 0.0;
      g_replay_flags[i] = 0;
      FXAI_ResetPreparedSample(g_replay_samples[i]);
   }
}

void FXAI_GetModelHyperParams(const int ai_idx, FXAIAIHyperParams &hp)
{
   if(ai_idx >= 0 && ai_idx < FXAI_AI_COUNT && g_model_hp_ready[ai_idx])
   {
      hp = g_model_hp[ai_idx];
      return;
   }
   FXAI_BuildHyperParams(hp);
   // Recommended GeodesicAttention starting defaults.
   if(ai_idx == (int)AI_GEODESICATTENTION)
   {
      hp.lr = 0.0060;
      hp.l2 = 0.0030;
   }
   // Recommended LSTM starting defaults.
   if(ai_idx == (int)AI_LSTM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0040;
   }
   // Recommended LightGBM starting defaults.
   if(ai_idx == (int)AI_LIGHTGBM)
   {
      hp.xgb_lr = 0.0300;
      hp.xgb_l2 = 4.0000;
      hp.xgb_split = 0.0000;
   }
   // Recommended PA_LINEAR starting defaults.
   if(ai_idx == (int)AI_PA_LINEAR)
   {
      hp.lr = 0.0600;
      hp.l2 = 0.0030;
      hp.pa_c = 4.0000;
      hp.pa_margin = 1.2000;
   }
   // Recommended CFX_WORLD starting defaults.
   if(ai_idx == (int)AI_CFX_WORLD)
   {
      hp.lr = 0.0100;
      hp.l2 = 0.0020;
   }
   // Recommended LOFFM starting defaults.
   if(ai_idx == (int)AI_LOFFM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0030;
   }
   // Recommended TRR starting defaults.
   if(ai_idx == (int)AI_TRR)
   {
      hp.lr = 0.0090;
      hp.l2 = 0.0025;
   }
   // Recommended GRAPHWM starting defaults.
   if(ai_idx == (int)AI_GRAPHWM)
   {
      hp.lr = 0.0080;
      hp.l2 = 0.0020;
   }
   // Recommended MOE_CONFORMAL starting defaults.
   if(ai_idx == (int)AI_MOE_CONFORMAL)
   {
      hp.lr = 0.0060;
      hp.l2 = 0.0030;
   }
   // Recommended M1SYNC starting defaults.
   if(ai_idx == (int)AI_M1SYNC)
   {
      hp.lr = 0.0;
      hp.l2 = 0.0;
   }
   if(ai_idx == (int)AI_BUY_ONLY || ai_idx == (int)AI_SELL_ONLY || ai_idx == (int)AI_RANDOM_NOSKIP)
   {
      hp.lr = 0.0;
      hp.l2 = 0.0;
   }
}

void FXAI_GetModelHyperParamsRouted(const int ai_idx,
                                    const int regime_id,
                                    const int horizon_minutes,
                                    FXAIAIHyperParams &hp)
{
   FXAI_GetModelHyperParams(ai_idx, hp);
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_hp_horizon_ready[ai_idx][hslot])
      hp = g_model_hp_horizon[ai_idx][hslot];

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_hp_bank_ready[ai_idx][regime_id][hslot])
   {
      hp = g_model_hp_bank[ai_idx][regime_id][hslot];
   }
}

void FXAI_GetModelThresholds(const int ai_idx,
                            const int regime_id,
                            const int horizon_minutes,
                            const double base_buy,
                            const double base_sell,
                            double &buy_thr,
                            double &sell_thr)
{
   buy_thr = base_buy;
   sell_thr = base_sell;
   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);

   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(g_model_thr_ready[ai_idx])
   {
      buy_thr = g_model_buy_thr[ai_idx];
      sell_thr = g_model_sell_thr[ai_idx];
   }

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_thr_horizon_ready[ai_idx][hslot])
   {
      buy_thr = 0.55 * buy_thr + 0.45 * g_model_buy_thr_horizon[ai_idx][hslot];
      sell_thr = 0.55 * sell_thr + 0.45 * g_model_sell_thr_horizon[ai_idx][hslot];
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_model_thr_regime_ready[ai_idx][regime_id])
   {
      buy_thr = 0.65 * buy_thr + 0.35 * g_model_buy_thr_regime[ai_idx][regime_id];
      sell_thr = 0.65 * sell_thr + 0.35 * g_model_sell_thr_regime[ai_idx][regime_id];
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_model_thr_bank_ready[ai_idx][regime_id][hslot])
   {
      buy_thr = 0.35 * buy_thr + 0.65 * g_model_buy_thr_bank[ai_idx][regime_id][hslot];
      sell_thr = 0.35 * sell_thr + 0.65 * g_model_sell_thr_bank[ai_idx][regime_id][hslot];
   }

   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS && g_model_horizon_edge_ready[ai_idx][hslot])
   {
      double edge = g_model_horizon_edge_ema[ai_idx][hslot];
      double adj = FXAI_Clamp(edge / MathMax(0.50, MathAbs(g_model_global_edge_ema[ai_idx]) + 0.50), -0.08, 0.08);
      buy_thr = FXAI_Clamp(buy_thr - (0.35 * adj), 0.50, 0.95);
      sell_thr = FXAI_Clamp(sell_thr + (0.35 * adj), 0.05, 0.50);
   }

   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FXAI_SampleThresholdPair(const double base_buy,
                             const double base_sell,
                             double &buy_thr,
                             double &sell_thr)
{
   double b0 = base_buy;
   double s0 = base_sell;
   FXAI_SanitizeThresholdPair(b0, s0);

   buy_thr = FXAI_Clamp(FXAI_RandRange(MathMax(0.52, b0 - 0.08), MathMin(0.90, b0 + 0.08)), 0.50, 0.95);
   sell_thr = FXAI_Clamp(FXAI_RandRange(MathMax(0.08, s0 - 0.08), MathMin(0.48, s0 + 0.08)), 0.05, 0.50);
   FXAI_SanitizeThresholdPair(buy_thr, sell_thr);
}

void FXAI_SampleModelHyperParams(const int ai_idx,
                                const FXAIAIHyperParams &base,
                                FXAIAIHyperParams &hp)
{
   hp = base;

      switch(ai_idx)
      {
         case (int)AI_SGD_LOGIT:
      case (int)AI_LSTMG:
      case (int)AI_S4:
      case (int)AI_TFT:
      case (int)AI_AUTOFORMER:
      case (int)AI_STMN:
      case (int)AI_TST:
      case (int)AI_PATCHTST:
         case (int)AI_CHRONOS:
         case (int)AI_TIMESFM:
         case (int)AI_CFX_WORLD:
         case (int)AI_LOFFM:
         case (int)AI_TRR:
         case (int)AI_GRAPHWM:
         case (int)AI_MOE_CONFORMAL:
         case (int)AI_RETRDIFF:
         hp.lr = FXAI_RandRange(0.0030, 0.0600);
         hp.l2 = FXAI_RandRange(0.0000, 0.0300);
         break;

      case (int)AI_M1SYNC:
      case (int)AI_BUY_ONLY:
      case (int)AI_SELL_ONLY:
      case (int)AI_RANDOM_NOSKIP:
         break;

      case (int)AI_LSTM:
         hp.lr = FXAI_RandRange(0.0040, 0.0200);
         hp.l2 = FXAI_RandRange(0.0010, 0.0100);
         break;

      case (int)AI_GEODESICATTENTION:
         hp.lr = FXAI_RandRange(0.0030, 0.0150);
         hp.l2 = FXAI_RandRange(0.0010, 0.0080);
         break;

      case (int)AI_TCN:
         hp.lr = FXAI_RandRange(0.0030, 0.0500);
         hp.l2 = FXAI_RandRange(0.0000, 0.0200);
         hp.tcn_layers = (double)((int)MathRound(FXAI_RandRange(3.0, 6.0)));
         hp.tcn_kernel = (double)((int)MathRound(FXAI_RandRange(2.0, 4.0)));
         hp.tcn_dilation_base = (double)((int)MathRound(FXAI_RandRange(1.0, 3.0)));
         break;

      case (int)AI_FTRL_LOGIT:
         hp.ftrl_alpha = FXAI_RandRange(0.0100, 0.2500);
         hp.ftrl_beta = FXAI_RandRange(0.1000, 2.5000);
         hp.ftrl_l1 = FXAI_RandRange(0.0000, 0.0100);
         hp.ftrl_l2 = FXAI_RandRange(0.0000, 0.1000);
         break;

      case (int)AI_PA_LINEAR:
         hp.lr = FXAI_RandRange(0.0200, 0.0800);
         hp.l2 = FXAI_RandRange(0.0010, 0.0100);
         hp.pa_c = FXAI_RandRange(0.5000, 6.0000);
         hp.pa_margin = FXAI_RandRange(0.6000, 1.9000);
         break;

      case (int)AI_XGB_FAST:
      case (int)AI_XGBOOST:
         hp.xgb_lr = FXAI_RandRange(0.0050, 0.1200);
         hp.xgb_l2 = FXAI_RandRange(0.0000, 0.0300);
         hp.xgb_split = FXAI_RandRange(-0.8000, 0.8000);
         break;

      case (int)AI_LIGHTGBM:
         hp.xgb_lr = FXAI_RandRange(0.0200, 0.0400);
         hp.xgb_l2 = FXAI_RandRange(2.0000, 6.0000);
         hp.xgb_split = FXAI_RandRange(-0.2000, 0.2000);
         break;

      case (int)AI_CATBOOST:
         hp.xgb_lr = FXAI_RandRange(0.0200, 0.0500);
         hp.xgb_l2 = FXAI_RandRange(3.0000, 8.0000);
         hp.xgb_split = FXAI_RandRange(-0.2000, 0.2000);
         break;

      case (int)AI_MLP_TINY:
         hp.mlp_lr = FXAI_RandRange(0.0010, 0.0300);
         hp.mlp_l2 = FXAI_RandRange(0.0000, 0.0200);
         hp.mlp_init = FXAI_RandRange(0.0300, 0.2500);
         break;

      case (int)AI_QUANTILE:
         hp.quantile_lr = FXAI_RandRange(0.0010, 0.0500);
         hp.quantile_l2 = FXAI_RandRange(0.0000, 0.0200);
         break;

      case (int)AI_ENHASH:
         hp.enhash_lr = FXAI_RandRange(0.0020, 0.0500);
         hp.enhash_l1 = FXAI_RandRange(0.0000, 0.0100);
         hp.enhash_l2 = FXAI_RandRange(0.0000, 0.0200);
         break;

      default:
         hp.lr = FXAI_RandRange(0.0030, 0.0600);
         hp.l2 = FXAI_RandRange(0.0000, 0.0300);
         break;
   }
}


#endif // __FXAI_ENGINE_TRAINING_MQH__
