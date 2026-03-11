#ifndef __FXAI_ENGINE_LIFECYCLE_MQH__
#define __FXAI_ENGINE_LIFECYCLE_MQH__

void FXAI_ParseContextSymbols(const string raw, string &symbols[])
{
   ArrayResize(symbols, 0);

   string clean = raw;
   StringReplace(clean, "{", "");
   StringReplace(clean, "}", "");
   StringReplace(clean, ";", ",");
   StringReplace(clean, "|", ",");

   string parts[];
   int n = StringSplit(clean, ',', parts);
   if(n <= 0) return;

   for(int i=0; i<n; i++)
   {
      string sym = parts[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;

      bool exists = false;
      for(int j=0; j<ArraySize(symbols); j++)
      {
         if(StringCompare(symbols[j], sym, false) == 0)
         {
            exists = true;
            break;
         }
      }
      if(exists) continue;

      int sz = ArraySize(symbols);
      ArrayResize(symbols, sz + 1);
      symbols[sz] = sym;
      if(ArraySize(symbols) >= FXAI_MAX_CONTEXT_SYMBOLS)
         break;
   }
}

void FXAI_FilterContextSymbols(const string main_symbol, string &symbols[])
{
   int n = ArraySize(symbols);
   if(n <= 0) return;

   int w = 0;
   for(int i=0; i<n; i++)
   {
      string sym = symbols[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(StringLen(sym) <= 0) continue;
      if(StringCompare(sym, main_symbol, false) == 0) continue;
      if(!SymbolSelect(sym, true)) continue;

      symbols[w] = sym;
      w++;
   }

   if(w < n)
      ArrayResize(symbols, w);
}

double FXAI_ContextAlignedCorr(const double &main_close[],
                               const int main_i,
                               const FXAIContextSeries &ctx,
                               const int window)
{
   int n_main = ArraySize(main_close);
   int n_map = ArraySize(ctx.aligned_idx);
   if(window < 4 || main_i < 0 || main_i >= n_main || main_i >= n_map)
      return 0.0;

   double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
   int used = 0;
   for(int k=0; k<window; k++)
   {
      int im = main_i + k;
      if(im + 1 >= n_main || im >= n_map) break;
      int ic = ctx.aligned_idx[im];
      if(ic < 0 || ic + 1 >= ArraySize(ctx.close)) continue;

      double xr = FXAI_SafeReturn(main_close, im, im + 1);
      double yr = FXAI_SafeReturn(ctx.close, ic, ic + 1);
      sx += xr;
      sy += yr;
      sxx += xr * xr;
      syy += yr * yr;
      sxy += xr * yr;
      used++;
   }

   if(used < 4) return 0.0;
   double cov = sxy - (sx * sy) / (double)used;
   double vx = sxx - (sx * sx) / (double)used;
   double vy = syy - (sy * sy) / (double)used;
   if(vx <= 1e-12 || vy <= 1e-12) return 0.0;
   return FXAI_Clamp(cov / MathSqrt(vx * vy), -1.0, 1.0);
}

void FXAI_PrecomputeContextAggregates(const datetime &main_time[],
                                     const double &main_close[],
                                     FXAIContextSeries &ctx_series[],
                                     const int ctx_count,
                                     const int upto_index,
                                     double &ctx_mean_arr[],
                                     double &ctx_std_arr[],
                                     double &ctx_up_arr[],
                                     double &ctx_extra_arr[])
{
   int n = ArraySize(main_time);
   if(ArraySize(main_close) != n) return;
   ArrayResize(ctx_mean_arr, n);
   ArrayResize(ctx_std_arr, n);
   ArrayResize(ctx_up_arr, n);
   ArrayResize(ctx_extra_arr, n * FXAI_CONTEXT_EXTRA_FEATS);
   if(n <= 0) return;

   int lag_m1 = 2 * PeriodSeconds(PERIOD_M1);
   if(lag_m1 <= 0) lag_m1 = 120;

   int upto = upto_index;
   if(upto < 0) upto = 0;
   if(upto >= n) upto = n - 1;
   // Keep one extra index ready for normalizers that need previous-bar features (i+1).
   int upto_fill = upto;
   if(upto_fill < n - 1) upto_fill++;

   for(int i=0; i<n; i++)
   {
      ctx_mean_arr[i] = 0.0;
      ctx_std_arr[i] = 0.0;
      ctx_up_arr[i] = 0.5;
   }
   for(int i=0; i<ArraySize(ctx_extra_arr); i++)
      ctx_extra_arr[i] = 0.0;

   for(int s=0; s<ctx_count; s++)
   {
      if(!ctx_series[s].loaded)
      {
         ArrayResize(ctx_series[s].aligned_idx, 0);
         continue;
      }
      FXAI_BuildAlignedIndexMapRange(main_time,
                                    ctx_series[s].time,
                                    lag_m1,
                                    upto_fill,
                                    ctx_series[s].aligned_idx);
   }

   for(int i=0; i<=upto_fill; i++)
   {
      datetime t_ref = main_time[i];
      if(t_ref <= 0 || ctx_count <= 0) continue;

      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double main_vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(main_vol < 1e-6) main_vol = MathAbs(main_ret);
      if(main_vol < 1e-6) main_vol = 1e-4;

      double weighted_sum = 0.0;
      double weighted_sum2 = 0.0;
      double weight_total = 0.0;
      double up_weight = 0.0;
      int valid = 0;
      double top_score[FXAI_CONTEXT_TOP_SYMBOLS];
      int top_symbol_idx[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_ret[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_lag[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_rel[FXAI_CONTEXT_TOP_SYMBOLS];
      double top_ctx_corr[FXAI_CONTEXT_TOP_SYMBOLS];
      for(int t=0; t<FXAI_CONTEXT_TOP_SYMBOLS; t++)
      {
         top_score[t] = -1e18;
         top_symbol_idx[t] = -1;
         top_ctx_ret[t] = 0.0;
         top_ctx_lag[t] = 0.0;
         top_ctx_rel[t] = 0.0;
         top_ctx_corr[t] = 0.0;
      }

      for(int s=0; s<ctx_count; s++)
      {
         if(!ctx_series[s].loaded) continue;
         int idx = -1;
         if(i >= 0 && i < ArraySize(ctx_series[s].aligned_idx))
            idx = ctx_series[s].aligned_idx[i];
         if(idx < 0) continue;

         double freshness = FXAI_AlignedFreshnessWeight(ctx_series[s].time, idx, t_ref, lag_m1);
         double ctx_ret_raw = FXAI_SafeReturn(ctx_series[s].close, idx, idx + 1);
         double ctx_lag_raw = FXAI_SafeReturn(ctx_series[s].close, idx + 1, idx + 2);
         double ctx_rel_raw = ctx_ret_raw - main_ret;
         double ctx_corr_raw = FXAI_ContextAlignedCorr(main_close, i, ctx_series[s], 20);
         double rel_edge = FXAI_Clamp(MathAbs(ctx_rel_raw) / main_vol, 0.0, 4.0);
         double ret_edge = FXAI_Clamp(MathAbs(ctx_ret_raw) / main_vol, 0.0, 4.0);
         double lag_edge = FXAI_Clamp(MathAbs(ctx_lag_raw) / main_vol, 0.0, 4.0);
         double corr_edge = MathAbs(ctx_corr_raw);
         double symbol_score = freshness * ((0.40 * corr_edge) +
                                            (0.30 * rel_edge) +
                                            (0.20 * lag_edge) +
                                            (0.10 * ret_edge));

         double w = 0.20 + symbol_score;
         weighted_sum += w * ctx_ret_raw;
         weighted_sum2 += w * ctx_ret_raw * ctx_ret_raw;
         weight_total += w;
         if(ctx_ret_raw > 0.0) up_weight += w;
         valid++;

         for(int slot=0; slot<FXAI_CONTEXT_TOP_SYMBOLS; slot++)
         {
            if(symbol_score <= top_score[slot]) continue;
            for(int shift=FXAI_CONTEXT_TOP_SYMBOLS - 1; shift>slot; shift--)
            {
               top_score[shift] = top_score[shift - 1];
               top_symbol_idx[shift] = top_symbol_idx[shift - 1];
               top_ctx_ret[shift] = top_ctx_ret[shift - 1];
               top_ctx_lag[shift] = top_ctx_lag[shift - 1];
               top_ctx_rel[shift] = top_ctx_rel[shift - 1];
               top_ctx_corr[shift] = top_ctx_corr[shift - 1];
            }
            top_score[slot] = symbol_score;
            top_symbol_idx[slot] = s;
            top_ctx_ret[slot] = ctx_ret_raw * freshness;
            top_ctx_lag[slot] = ctx_lag_raw * freshness;
            top_ctx_rel[slot] = ctx_rel_raw * freshness;
            top_ctx_corr[slot] = ctx_corr_raw * freshness;
            break;
         }
      }

      if(valid <= 0 || weight_total <= 0.0) continue;

      double mean = weighted_sum / weight_total;
      double var = (weighted_sum2 / weight_total) - (mean * mean);
      if(var < 0.0) var = 0.0;
      double up_ratio = up_weight / weight_total;

      double coverage = (ctx_count > 0 ? ((double)valid / (double)ctx_count) : 0.0);
      coverage = FXAI_Clamp(coverage, 0.0, 1.0);
      double conf = 0.30 + (0.70 * coverage);

      ctx_mean_arr[i] = mean * coverage;
      ctx_std_arr[i] = MathSqrt(var) * conf;
      ctx_up_arr[i] = 0.5 + ((up_ratio - 0.5) * coverage);

      for(int top_slot=0; top_slot<FXAI_CONTEXT_TOP_SYMBOLS; top_slot++)
      {
         if(top_symbol_idx[top_slot] < 0) continue;
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 0, top_ctx_ret[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 1, top_ctx_lag[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 2, top_ctx_rel[top_slot]);
         FXAI_SetContextExtraValue(ctx_extra_arr, i, top_slot * 4 + 3, top_ctx_corr[top_slot]);
      }
   }
}

double FXAI_ContextSeriesUtilityNow(const datetime &main_time[],
                                    const double &main_close[],
                                    FXAIContextSeries &series)
{
   if(!series.loaded) return -1e9;
   if(ArraySize(main_time) <= 4 || ArraySize(main_close) <= 4) return -1e9;
   if(ArraySize(series.aligned_idx) <= 4) return -1e9;

   double sum_score = 0.0;
   int used = 0;
   for(int i=0; i<16; i++)
   {
      if(i >= ArraySize(main_time) || i >= ArraySize(series.aligned_idx)) break;
      int idx = series.aligned_idx[i];
      if(idx < 0) continue;

      double lag = 2.0 * PeriodSeconds(PERIOD_M1);
      if(lag <= 0.0) lag = 120.0;
      double fresh = FXAI_AlignedFreshnessWeight(series.time, idx, main_time[i], (int)lag);
      double main_ret = FXAI_SafeReturn(main_close, i, i + 1);
      double ctx_ret = FXAI_SafeReturn(series.close, idx, idx + 1);
      double ctx_lag = FXAI_SafeReturn(series.close, idx + 1, idx + 2);
      double corr = FXAI_ContextAlignedCorr(main_close, i, series, 20);
      double vol = FXAI_RollingAbsReturn(main_close, i, 20);
      if(vol < 1e-6) vol = MathAbs(main_ret);
      if(vol < 1e-6) vol = 1e-4;

      double rel = FXAI_Clamp(MathAbs(ctx_ret - main_ret) / vol, 0.0, 4.0);
      double lead = FXAI_Clamp(MathAbs(ctx_lag) / vol, 0.0, 4.0);
      double mag = FXAI_Clamp(MathAbs(ctx_ret) / vol, 0.0, 4.0);
      double score = fresh * ((0.45 * MathAbs(corr)) +
                              (0.30 * rel) +
                              (0.15 * lead) +
                              (0.10 * mag));
      sum_score += score;
      used++;
   }

   if(used <= 0) return -1e9;
   return sum_score / (double)used;
}

void FXAI_SelectDynamicContextIndices(const datetime &main_time[],
                                      const double &main_close[],
                                      FXAIContextSeries &ctx_series[],
                                      const int ctx_count,
                                      int &selected_idx[])
{
   ArrayResize(selected_idx, 0);
   if(ctx_count <= 0) return;

   double slot_score[FXAI_MAX_CONTEXT_SYMBOLS];
   int slot_idx[FXAI_MAX_CONTEXT_SYMBOLS];
   int keep_n = ctx_count;
   if(keep_n > FXAI_CONTEXT_DYNAMIC_POOL) keep_n = FXAI_CONTEXT_DYNAMIC_POOL;

   for(int j=0; j<FXAI_MAX_CONTEXT_SYMBOLS; j++)
   {
      slot_score[j] = -1e18;
      slot_idx[j] = -1;
   }

   for(int s=0; s<ctx_count && s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      double inst = FXAI_ContextSeriesUtilityNow(main_time, main_close, ctx_series[s]);
      if(inst <= -1e8) continue;

      if(!g_context_symbol_utility_ready[s])
      {
         g_context_symbol_utility[s] = inst;
         g_context_symbol_utility_ready[s] = true;
      }
      else
      {
         g_context_symbol_utility[s] = 0.85 * g_context_symbol_utility[s] + 0.15 * inst;
      }

      double score = g_context_symbol_utility[s];
      for(int slot=0; slot<keep_n; slot++)
      {
         if(score <= slot_score[slot]) continue;
         for(int shift=keep_n - 1; shift>slot; shift--)
         {
            slot_score[shift] = slot_score[shift - 1];
            slot_idx[shift] = slot_idx[shift - 1];
         }
         slot_score[slot] = score;
         slot_idx[slot] = s;
         break;
      }
   }

   for(int slot=0; slot<keep_n; slot++)
   {
      if(slot_idx[slot] < 0) continue;
      int sz = ArraySize(selected_idx);
      ArrayResize(selected_idx, sz + 1);
      selected_idx[sz] = slot_idx[slot];
   }

   if(ArraySize(selected_idx) <= 0)
   {
      for(int s=0; s<ctx_count && s<FXAI_CONTEXT_DYNAMIC_POOL; s++)
      {
         if(!ctx_series[s].loaded) continue;
         int sz = ArraySize(selected_idx);
         ArrayResize(selected_idx, sz + 1);
         selected_idx[sz] = s;
      }
   }
}

void FXAI_PrecomputeDynamicContextAggregates(const datetime &main_time[],
                                             const double &main_close[],
                                             FXAIContextSeries &ctx_series[],
                                             const int ctx_count,
                                             const int upto_index,
                                             double &ctx_mean_arr[],
                                             double &ctx_std_arr[],
                                             double &ctx_up_arr[],
                                             double &ctx_extra_arr[])
{
   int selected_idx[];
   FXAI_SelectDynamicContextIndices(main_time, main_close, ctx_series, ctx_count, selected_idx);
   if(ArraySize(selected_idx) <= 0)
   {
      FXAI_PrecomputeContextAggregates(main_time,
                                       main_close,
                                       ctx_series,
                                       ctx_count,
                                       upto_index,
                                       ctx_mean_arr,
                                       ctx_std_arr,
                                       ctx_up_arr,
                                       ctx_extra_arr);
      return;
   }

   FXAIContextSeries selected[];
   ArrayResize(selected, ArraySize(selected_idx));
   for(int i=0; i<ArraySize(selected_idx); i++)
      selected[i] = ctx_series[selected_idx[i]];

   FXAI_PrecomputeContextAggregates(main_time,
                                    main_close,
                                    selected,
                                    ArraySize(selected),
                                    upto_index,
                                    ctx_mean_arr,
                                    ctx_std_arr,
                                    ctx_up_arr,
                                    ctx_extra_arr);
}

//--------------------------- INIT -----------------------------------
void ResetAIState(const string symbol)
{
   g_ai_last_symbol = symbol;
   g_ai_last_signal_bar = 0;
   g_ai_last_signal = -1;
   g_ai_last_signal_key = -1;
   g_ai_warmup_done = (!AI_Warmup);
   FXAI_ParseHorizonList(AI_Horizons, PredictionTargetMinutes, g_horizon_minutes);
   FXAI_ResetModelHyperParams();
   FXAI_ResetReliabilityPending();
   FXAI_ResetHorizonPolicyPending();
   FXAI_ResetStackPending();
   FXAI_ResetAdaptiveRoutingState();
   FXAI_ResetRegimeCalibration();
   FXAI_ResetReplayReservoir();
   for(int s=0; s<FXAI_MAX_CONTEXT_SYMBOLS; s++)
   {
      g_context_symbol_utility[s] = 0.0;
      g_context_symbol_utility_ready[s] = false;
   }

   if(!g_norm_windows_ready)
   {
      int windows[];
      int default_w = FXAI_GetNormDefaultWindow();
      FXAI_BuildNormWindowsFromGroups(default_w, default_w, default_w, default_w, windows);
      FXAI_ApplyNormWindows(windows, default_w);
   }
   else
   {
      FXAI_ApplyNormWindows(g_norm_feature_windows, g_norm_default_window);
   }

   for(int i=0; i<FXAI_AI_COUNT; i++)
   {
      g_ai_trained[i] = false;
      g_ai_last_train_bar[i] = 0;
      FXAI_ResetModelAuxState(i);
   }

   if(g_plugins_ready)
      g_plugins.ResetAll();
}

bool FXAI_ValidateNativePluginAPI()
{
   double x_dummy[FXAI_AI_WEIGHTS];
   for(int k=0; k<FXAI_AI_WEIGHTS; k++) x_dummy[k] = 0.0;
   x_dummy[0] = 1.0;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.Get(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI error: API v4 plugin missing at id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.EnsureInitialized(hp);

      FXAIAIManifestV4 manifest;
      FXAI_GetPluginManifest(*plugin, manifest);
      string reason = "";
      if(!FXAI_ValidateManifestV4(manifest, reason))
      {
         Print("FXAI error: API v4 manifest invalid. model=", plugin.AIName(),
               " id=", ai_idx,
               " reason=", reason);
         return false;
      }

      FXAIAIPredictRequestV4 req_v4;
      req_v4.valid = true;
      req_v4.ctx.api_version = FXAI_API_VERSION_V4;
      req_v4.ctx.regime_id = 0;
      req_v4.ctx.session_bucket = FXAI_DeriveSessionBucket(TimeCurrent());
      req_v4.ctx.horizon_minutes = 5;
      req_v4.ctx.feature_schema_id = manifest.feature_schema_id;
      req_v4.ctx.normalization_method_id = (int)AI_FeatureNormalization;
      req_v4.ctx.sequence_bars = FXAI_GetPluginSequenceBars(*plugin, req_v4.ctx.horizon_minutes);
      req_v4.ctx.cost_points = 0.5;
      req_v4.ctx.min_move_points = 0.8;
      req_v4.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req_v4.ctx.sample_time = TimeCurrent();
      for(int kk=0; kk<FXAI_AI_WEIGHTS; kk++)
         req_v4.x[kk] = x_dummy[kk];
      FXAI_ApplyFeatureSchemaToInput(manifest.feature_schema_id,
                                     manifest.feature_groups_mask,
                                     req_v4.x);

      FXAIAIPredictionV4 pred_v4;
      if(!plugin.Predict(req_v4, hp, pred_v4))
      {
         Print("FXAI error: API v4 predict failed. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }

      if(!FXAI_ValidatePredictionV4(pred_v4, reason))
      {
         Print("FXAI error: API v4 prediction invalid. model=", plugin.AIName(),
               " id=", ai_idx,
               " reason=", reason);
         return false;
      }

      if(!plugin.SelfTest())
      {
         Print("FXAI error: API v4 self-test failed. model=", plugin.AIName(),
               " id=", ai_idx);
         return false;
      }
   }

   return true;
}

void FXAI_FillComplianceContext(CFXAIAIPlugin &plugin,
                                FXAIAIContextV4 &ctx,
                                const double cost_points,
                                const datetime sample_time,
                                const int regime_id,
                                const int horizon_minutes)
{
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = regime_id;
   ctx.session_bucket = FXAI_DeriveSessionBucket(sample_time);
   ctx.horizon_minutes = horizon_minutes;
   ctx.feature_schema_id = manifest.feature_schema_id;
   ctx.normalization_method_id = (int)AI_FeatureNormalization;
   ctx.sequence_bars = FXAI_GetPluginSequenceBars(plugin, horizon_minutes);
   ctx.min_move_points = MathMax(cost_points + 0.30, 0.50);
   ctx.cost_points = MathMax(cost_points, 0.0);
   ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
   ctx.sample_time = sample_time;
}

void FXAI_FillComplianceTrainRequest(CFXAIAIPlugin &plugin,
                                     FXAIAITrainRequestV4 &req,
                                     const int label_class,
                                     const double move_points,
                                     const double cost_points,
                                     const double v1,
                                     const double v2,
                                     const double v3,
                                     const datetime sample_time,
                                     const int regime_id,
                                     const int horizon_minutes)
{
   req.valid = true;
   FXAI_FillComplianceContext(plugin, req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
   req.label_class = label_class;
   req.move_points = move_points;
   req.sample_weight = 1.0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   req.x[0] = 1.0;
   req.x[1] = v1;
   req.x[2] = v2;
   req.x[3] = v3;
   req.x[4] = 0.35 * v2;
   req.x[5] = 0.25 * v3;
   req.x[6] = 0.15 * v1;
   req.x[7] = req.ctx.cost_points;
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   FXAI_ApplyFeatureSchemaToInput(manifest.feature_schema_id,
                                  manifest.feature_groups_mask,
                                  req.x);
}

void FXAI_FillCompliancePredictRequest(CFXAIAIPlugin &plugin,
                                       FXAIAIPredictRequestV4 &req,
                                       const double cost_points,
                                       const double v1,
                                       const double v2,
                                       const double v3,
                                       const datetime sample_time,
                                       const int regime_id,
                                       const int horizon_minutes)
{
   req.valid = true;
   FXAI_FillComplianceContext(plugin, req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = 0.0;
   req.x[0] = 1.0;
   req.x[1] = v1;
   req.x[2] = v2;
   req.x[3] = v3;
   req.x[4] = 0.35 * v2;
   req.x[5] = 0.25 * v3;
   req.x[6] = 0.15 * v1;
   req.x[7] = req.ctx.cost_points;
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   FXAI_ApplyFeatureSchemaToInput(manifest.feature_schema_id,
                                  manifest.feature_groups_mask,
                                  req.x);
}

bool FXAI_ValidatePredictionOutput(const CFXAIAIPlugin &plugin,
                                   const FXAIAIPredictionV4 &pred,
                                   const string tag)
{
   string reason = "";
   if(!FXAI_ValidatePredictionV4(pred, reason))
   {
      Print("FXAI compliance error: invalid prediction. model=", plugin.AIName(),
            " tag=", tag, " reason=", reason);
      return false;
   }

   double s = pred.class_probs[(int)FXAI_LABEL_SELL]
            + pred.class_probs[(int)FXAI_LABEL_BUY]
            + pred.class_probs[(int)FXAI_LABEL_SKIP];
   for(int c=0; c<3; c++)
   {
      if(!MathIsValidNumber(pred.class_probs[c]) || pred.class_probs[c] < 0.0 || pred.class_probs[c] > 1.0)
      {
         Print("FXAI compliance error: probability range invalid. model=", plugin.AIName(),
               " tag=", tag, " class=", c, " value=", DoubleToString(pred.class_probs[c], 6));
         return false;
      }
   }

   if(!MathIsValidNumber(pred.move_mean_points) || pred.move_mean_points <= 0.0)
   {
      Print("FXAI compliance error: expected move invalid. model=", plugin.AIName(),
            " tag=", tag, " ev=", DoubleToString(pred.move_mean_points, 6));
      return false;
   }

   return true;
}

int FXAI_ComplianceSequenceBars(const FXAIAIManifestV4 &manifest)
{
   if(manifest.max_sequence_bars <= 1)
      return 1;
   int seq = manifest.max_sequence_bars;
   if(seq < manifest.min_sequence_bars)
      seq = manifest.min_sequence_bars;
   if(seq < 2) seq = 2;
   if(seq > 16) seq = 16;
   return seq;
}

int FXAI_ComplianceHorizon(const FXAIAIManifestV4 &manifest,
                           const int desired_horizon)
{
   int h = desired_horizon;
   if(h < manifest.min_horizon_minutes) h = manifest.min_horizon_minutes;
   if(h > manifest.max_horizon_minutes) h = manifest.max_horizon_minutes;
   if(h < 1) h = 1;
   return h;
}

double FXAI_PredictionDistance(const FXAIAIPredictionV4 &a,
                               const FXAIAIPredictionV4 &b)
{
   double d = 0.0;
   for(int c=0; c<3; c++)
      d += MathAbs(a.class_probs[c] - b.class_probs[c]);
   d += 0.05 * MathAbs(a.move_mean_points - b.move_mean_points);
   d += 0.02 * MathAbs(a.move_q25_points - b.move_q25_points);
   d += 0.02 * MathAbs(a.move_q50_points - b.move_q50_points);
   d += 0.02 * MathAbs(a.move_q75_points - b.move_q75_points);
   d += 0.10 * MathAbs(a.confidence - b.confidence);
   d += 0.10 * MathAbs(a.reliability - b.reliability);
   return d;
}

bool FXAI_RunStateResetCompliance(CFXAIAIPlugin &plugin,
                                  const FXAIAIManifestV4 &manifest,
                                  const FXAIAIHyperParams &hp,
                                  const datetime now_t)
{
   if(!FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      return true;

   int seq = FXAI_ComplianceSequenceBars(manifest);
   int horizon = FXAI_ComplianceHorizon(manifest, 5);

   FXAIAITrainRequestV4 train_req;
   FXAI_FillComplianceTrainRequest(plugin, train_req, (int)FXAI_LABEL_BUY, 5.5, 0.8,
                                   0.80, 0.45, 0.20, now_t - 90, 2, horizon);
   train_req.ctx.sequence_bars = seq;

   FXAIAIPredictRequestV4 pred_req;
   FXAI_FillCompliancePredictRequest(plugin, pred_req, 0.8, 0.80, 0.45, 0.20, now_t - 30, 2, horizon);
   pred_req.ctx.sequence_bars = seq;

   for(int i=0; i<6; i++)
      FXAI_TrainViaV4(plugin, train_req, hp);

   FXAIAIPredictionV4 pred_before, pred_after_reset_a, pred_after_reset_b;
   FXAI_PredictViaV4(plugin, pred_req, hp, pred_before);

   plugin.ResetState((int)FXAI_RESET_SESSION_CHANGE, now_t);
   if(!FXAI_PredictViaV4(plugin, pred_req, hp, pred_after_reset_a) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_after_reset_a, "reset_a"))
      return false;

   plugin.ResetState((int)FXAI_RESET_SESSION_CHANGE, now_t + 60);
   if(!FXAI_PredictViaV4(plugin, pred_req, hp, pred_after_reset_b) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_after_reset_b, "reset_b"))
      return false;

   double drift = FXAI_PredictionDistance(pred_after_reset_a, pred_after_reset_b);
   if(drift > 1e-4)
   {
      Print("FXAI compliance error: reset not idempotent. model=", plugin.AIName(),
            " drift=", DoubleToString(drift, 8));
      return false;
   }

   return true;
}

bool FXAI_RunSequenceWindowCompliance(CFXAIAIPlugin &plugin,
                                      const FXAIAIManifestV4 &manifest,
                                      const FXAIAIHyperParams &hp,
                                      const datetime now_t)
{
   bool wants_window = FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT);
   if(!wants_window)
      return (manifest.min_sequence_bars == 1 && manifest.max_sequence_bars == 1);

   int seq = FXAI_ComplianceSequenceBars(manifest);
   int horizon = FXAI_ComplianceHorizon(manifest, 13);

   FXAIAITrainRequestV4 train_req;
   FXAI_FillComplianceTrainRequest(plugin, train_req, (int)FXAI_LABEL_SELL, -5.0, 0.9,
                                   -0.75, -0.42, -0.18, now_t - 150, 3, horizon);
   train_req.ctx.sequence_bars = seq;

   FXAIAIPredictRequestV4 pred_req_seq, pred_req_one;
   FXAI_FillCompliancePredictRequest(plugin, pred_req_seq, 0.9, -0.75, -0.42, -0.18, now_t - 10, 3, horizon);
   FXAI_FillCompliancePredictRequest(plugin, pred_req_one, 0.9, -0.75, -0.42, -0.18, now_t - 10, 3, horizon);
   pred_req_seq.ctx.sequence_bars = seq;
   pred_req_one.ctx.sequence_bars = 1;

   for(int i=0; i<4; i++)
      FXAI_TrainViaV4(plugin, train_req, hp);

   FXAIAIPredictionV4 pred_seq, pred_one;
   if(!FXAI_PredictViaV4(plugin, pred_req_seq, hp, pred_seq) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_seq, "sequence"))
      return false;

   if(!FXAI_PredictViaV4(plugin, pred_req_one, hp, pred_one) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_one, "sequence_one"))
      return false;

   plugin.ResetState((int)FXAI_RESET_REGIME_CHANGE, now_t + 120);
   if(!FXAI_PredictViaV4(plugin, pred_req_seq, hp, pred_seq) ||
      !FXAI_ValidatePredictionOutput(plugin, pred_seq, "sequence_reset"))
      return false;

   return true;
}

bool FXAI_RunCalibrationDriftCompliance(CFXAIAIPlugin &plugin,
                                        const FXAIAIManifestV4 &manifest,
                                        const FXAIAIHyperParams &hp,
                                        const datetime now_t)
{
   int seq = FXAI_ComplianceSequenceBars(manifest);
   int base_h = FXAI_ComplianceHorizon(manifest, 5);
   int alt_h = FXAI_ComplianceHorizon(manifest, 13);
   int long_h = FXAI_ComplianceHorizon(manifest, 34);

   for(int step=0; step<180; step++)
   {
      int cls = (step % 3 == 0 ? (int)FXAI_LABEL_BUY :
                 (step % 3 == 1 ? (int)FXAI_LABEL_SELL : (int)FXAI_LABEL_SKIP));
      int regime = step % 4;
      int horizon = (step % 3 == 0 ? base_h : (step % 3 == 1 ? alt_h : long_h));
      double cost = (step % 5 == 0 ? 1.4 : 0.7);
      double v1 = (cls == (int)FXAI_LABEL_BUY ? 0.85 : (cls == (int)FXAI_LABEL_SELL ? -0.85 : 0.03));
      double v2 = 0.55 * v1;
      double v3 = 0.30 * v1;
      double move = (cls == (int)FXAI_LABEL_BUY ? 5.0 + 0.04 * step :
                    (cls == (int)FXAI_LABEL_SELL ? -(5.0 + 0.04 * step) : 0.15 + 0.002 * step));

      FXAIAITrainRequestV4 train_req;
      FXAI_FillComplianceTrainRequest(plugin, train_req, cls, move, cost,
                                      v1, v2, v3,
                                      now_t - (step * 60), regime, horizon);
      train_req.ctx.sequence_bars = seq;
      FXAI_TrainViaV4(plugin, train_req, hp);

      if((step % 24) != 23)
         continue;

      FXAIAIPredictRequestV4 pred_buy, pred_sell, pred_skip;
      FXAI_FillCompliancePredictRequest(plugin, pred_buy, 0.7, 0.82, 0.46, 0.21, now_t + step, regime, base_h);
      FXAI_FillCompliancePredictRequest(plugin, pred_sell, 0.7, -0.82, -0.46, -0.21, now_t + step, regime, alt_h);
      FXAI_FillCompliancePredictRequest(plugin, pred_skip, 1.2, 0.02, 0.01, 0.00, now_t + step, regime, base_h);
      pred_buy.ctx.sequence_bars = seq;
      pred_sell.ctx.sequence_bars = seq;
      pred_skip.ctx.sequence_bars = seq;

      FXAIAIPredictionV4 out_buy, out_sell, out_skip;
      if(!FXAI_PredictViaV4(plugin, pred_buy, hp, out_buy) ||
         !FXAI_PredictViaV4(plugin, pred_sell, hp, out_sell) ||
         !FXAI_PredictViaV4(plugin, pred_skip, hp, out_skip))
         return false;

      if(!FXAI_ValidatePredictionOutput(plugin, out_buy, "drift_buy") ||
         !FXAI_ValidatePredictionOutput(plugin, out_sell, "drift_sell") ||
         !FXAI_ValidatePredictionOutput(plugin, out_skip, "drift_skip"))
         return false;

      if(out_buy.class_probs[(int)FXAI_LABEL_BUY] + 0.03 < out_buy.class_probs[(int)FXAI_LABEL_SELL])
      {
         Print("FXAI compliance error: calibration drift buy ordering failed. model=", plugin.AIName());
         return false;
      }
      if(out_sell.class_probs[(int)FXAI_LABEL_SELL] + 0.03 < out_sell.class_probs[(int)FXAI_LABEL_BUY])
      {
         Print("FXAI compliance error: calibration drift sell ordering failed. model=", plugin.AIName());
         return false;
      }
      if(out_skip.class_probs[(int)FXAI_LABEL_SKIP] < 0.15)
      {
         Print("FXAI compliance error: calibration drift skip weak. model=", plugin.AIName());
         return false;
      }
   }

   return true;
}

bool FXAI_RunPluginComplianceHarness()
{
   datetime now_t = TimeCurrent();

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      CFXAIAIPlugin *plugin = g_plugins.CreateInstance(ai_idx);
      if(plugin == NULL)
      {
         Print("FXAI compliance error: could not create plugin id=", ai_idx);
         return false;
      }

      FXAIAIHyperParams hp;
      FXAI_GetModelHyperParams(ai_idx, hp);
      plugin.Reset();
      plugin.EnsureInitialized(hp);

      FXAIAIManifestV4 manifest;
      plugin.Describe(manifest);
      string reason = "";
      if(!FXAI_ValidateManifestV4(manifest, reason))
      {
         Print("FXAI compliance error: manifest invalid. model=", plugin.AIName(),
               " reason=", reason);
         delete plugin;
         return false;
      }

      FXAIAITrainRequestV4 buy_s, sell_s, skip_s, buy_big_s;
      FXAI_FillComplianceTrainRequest(*plugin, buy_s, (int)FXAI_LABEL_BUY, 4.5, 0.8, 0.75, 0.40, 0.20, now_t - 180, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, sell_s, (int)FXAI_LABEL_SELL, -4.5, 0.8, -0.75, -0.40, -0.20, now_t - 120, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, skip_s, (int)FXAI_LABEL_SKIP, 0.2, 0.8, 0.02, 0.01, 0.00, now_t - 60, 1, 5);
      FXAI_FillComplianceTrainRequest(*plugin, buy_big_s, (int)FXAI_LABEL_BUY, 8.0, 0.8, 1.20, 0.65, 0.35, now_t - 30, 1, 13);

      for(int rep=0; rep<10; rep++)
      {
         FXAI_TrainViaV4(*plugin, buy_s, hp);
         FXAI_TrainViaV4(*plugin, sell_s, hp);
         FXAI_TrainViaV4(*plugin, skip_s, hp);
         FXAI_TrainViaV4(*plugin, buy_big_s, hp);
      }

      FXAIAIPredictRequestV4 req_buy_lo, req_buy_hi, req_sell_lo, req_skip_lo, req_buy_big;
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_lo, 0.8, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_hi, 3.5, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_sell_lo, 0.8, -0.75, -0.40, -0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_skip_lo, 0.8, 0.02, 0.01, 0.00, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(*plugin, req_buy_big, 0.8, 1.20, 0.65, 0.35, now_t, 1, 13);

      FXAIAIPredictionV4 pred_buy_lo, pred_buy_hi, pred_sell_lo, pred_skip_lo, pred_buy_big;
      FXAI_PredictViaV4(*plugin, req_buy_lo, hp, pred_buy_lo);
      FXAI_PredictViaV4(*plugin, req_buy_hi, hp, pred_buy_hi);
      FXAI_PredictViaV4(*plugin, req_sell_lo, hp, pred_sell_lo);
      FXAI_PredictViaV4(*plugin, req_skip_lo, hp, pred_skip_lo);
      FXAI_PredictViaV4(*plugin, req_buy_big, hp, pred_buy_big);

      bool ok = FXAI_ValidatePredictionOutput(*plugin, pred_buy_lo, "buy_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_hi, "buy_hi")
             && FXAI_ValidatePredictionOutput(*plugin, pred_sell_lo, "sell_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_skip_lo, "skip_lo")
             && FXAI_ValidatePredictionOutput(*plugin, pred_buy_big, "buy_big");
      if(!ok)
      {
         delete plugin;
         return false;
      }

      if(pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + 0.05 < pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL])
      {
         Print("FXAI compliance error: buy ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_sell_lo.class_probs[(int)FXAI_LABEL_SELL] + 0.05 < pred_sell_lo.class_probs[(int)FXAI_LABEL_BUY])
      {
         Print("FXAI compliance error: sell ordering failed. model=", plugin.AIName());
         delete plugin;
         return false;
      }
      if(pred_skip_lo.class_probs[(int)FXAI_LABEL_SKIP] < 0.20)
      {
         Print("FXAI compliance error: skip response too weak. model=", plugin.AIName());
         delete plugin;
         return false;
      }

      double actionable_lo = pred_buy_lo.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_lo.class_probs[(int)FXAI_LABEL_SELL];
      double actionable_hi = pred_buy_hi.class_probs[(int)FXAI_LABEL_BUY] + pred_buy_hi.class_probs[(int)FXAI_LABEL_SELL];
      if(actionable_hi > actionable_lo + 0.20)
      {
         Print("FXAI compliance error: cost awareness failed. model=", plugin.AIName(),
               " low=", DoubleToString(actionable_lo, 4),
               " high=", DoubleToString(actionable_hi, 4));
         delete plugin;
         return false;
      }

      if(pred_buy_big.move_mean_points + 0.25 < pred_buy_lo.move_mean_points)
      {
         Print("FXAI compliance error: EV monotonicity failed. model=", plugin.AIName(),
               " big=", DoubleToString(pred_buy_big.move_mean_points, 4),
               " base=", DoubleToString(pred_buy_lo.move_mean_points, 4));
         delete plugin;
         return false;
      }

      if(plugin.CorePredictFailures() > 0)
      {
         Print("FXAI compliance error: core predict failures detected. model=", plugin.AIName(),
               " failures=", plugin.CorePredictFailures());
         delete plugin;
         return false;
      }

      plugin.ResetState((int)FXAI_RESET_FULL, now_t + 180);
      if(!FXAI_RunStateResetCompliance(*plugin, manifest, hp, now_t + 240) ||
         !FXAI_RunSequenceWindowCompliance(*plugin, manifest, hp, now_t + 300) ||
         !FXAI_RunCalibrationDriftCompliance(*plugin, manifest, hp, now_t + 360))
      {
         delete plugin;
         return false;
      }

      delete plugin;
   }

   return true;
}


#endif // __FXAI_ENGINE_LIFECYCLE_MQH__
