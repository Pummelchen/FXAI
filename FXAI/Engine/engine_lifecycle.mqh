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
      plugin.Describe(manifest);
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
      req_v4.ctx.feature_schema_id = 1;
      req_v4.ctx.normalization_method_id = (int)AI_FeatureNormalization;
      req_v4.ctx.sequence_bars = 1;
      req_v4.ctx.cost_points = 0.5;
      req_v4.ctx.min_move_points = 0.8;
      req_v4.ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
      req_v4.ctx.sample_time = TimeCurrent();
      for(int kk=0; kk<FXAI_AI_WEIGHTS; kk++)
         req_v4.x[kk] = x_dummy[kk];

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

void FXAI_FillComplianceContext(FXAIAIContextV4 &ctx,
                                const double cost_points,
                                const datetime sample_time,
                                const int regime_id,
                                const int horizon_minutes)
{
   ctx.api_version = FXAI_API_VERSION_V4;
   ctx.regime_id = regime_id;
   ctx.session_bucket = FXAI_DeriveSessionBucket(sample_time);
   ctx.horizon_minutes = horizon_minutes;
   ctx.feature_schema_id = 1;
   ctx.normalization_method_id = (int)AI_FeatureNormalization;
   ctx.sequence_bars = 1;
   ctx.min_move_points = MathMax(cost_points + 0.30, 0.50);
   ctx.cost_points = MathMax(cost_points, 0.0);
   ctx.point_value = (_Point > 0.0 ? _Point : 1.0);
   ctx.sample_time = sample_time;
}

void FXAI_FillComplianceTrainRequest(FXAIAITrainRequestV4 &req,
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
   FXAI_FillComplianceContext(req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
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
}

void FXAI_FillCompliancePredictRequest(FXAIAIPredictRequestV4 &req,
                                       const double cost_points,
                                       const double v1,
                                       const double v2,
                                       const double v3,
                                       const datetime sample_time,
                                       const int regime_id,
                                       const int horizon_minutes)
{
   req.valid = true;
   FXAI_FillComplianceContext(req.ctx, cost_points, sample_time, regime_id, horizon_minutes);
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
}

bool FXAI_ValidatePredictionOutput(const CFXAIAIPlugin &plugin,
                                   const FXAIAIPredictionV4 &pred,
                                   const string tag)
{
   double s = pred.class_probs[(int)FXAI_LABEL_SELL]
            + pred.class_probs[(int)FXAI_LABEL_BUY]
            + pred.class_probs[(int)FXAI_LABEL_SKIP];
   if(!MathIsValidNumber(s) || MathAbs(s - 1.0) > 1e-3)
   {
      Print("FXAI compliance error: probability sum invalid. model=", plugin.AIName(),
            " tag=", tag, " sum=", DoubleToString(s, 6));
      return false;
   }

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

      FXAIAITrainRequestV4 buy_s, sell_s, skip_s, buy_big_s;
      FXAI_FillComplianceTrainRequest(buy_s, (int)FXAI_LABEL_BUY, 4.5, 0.8, 0.75, 0.40, 0.20, now_t - 180, 1, 5);
      FXAI_FillComplianceTrainRequest(sell_s, (int)FXAI_LABEL_SELL, -4.5, 0.8, -0.75, -0.40, -0.20, now_t - 120, 1, 5);
      FXAI_FillComplianceTrainRequest(skip_s, (int)FXAI_LABEL_SKIP, 0.2, 0.8, 0.02, 0.01, 0.00, now_t - 60, 1, 5);
      FXAI_FillComplianceTrainRequest(buy_big_s, (int)FXAI_LABEL_BUY, 8.0, 0.8, 1.20, 0.65, 0.35, now_t - 30, 1, 13);

      for(int rep=0; rep<10; rep++)
      {
         FXAI_TrainViaV4(*plugin, buy_s, hp);
         FXAI_TrainViaV4(*plugin, sell_s, hp);
         FXAI_TrainViaV4(*plugin, skip_s, hp);
         FXAI_TrainViaV4(*plugin, buy_big_s, hp);
      }

      FXAIAIPredictRequestV4 req_buy_lo, req_buy_hi, req_sell_lo, req_skip_lo, req_buy_big;
      FXAI_FillCompliancePredictRequest(req_buy_lo, 0.8, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(req_buy_hi, 3.5, 0.75, 0.40, 0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(req_sell_lo, 0.8, -0.75, -0.40, -0.20, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(req_skip_lo, 0.8, 0.02, 0.01, 0.00, now_t, 1, 5);
      FXAI_FillCompliancePredictRequest(req_buy_big, 0.8, 1.20, 0.65, 0.35, now_t, 1, 13);

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

      delete plugin;
   }

   return true;
}


#endif // __FXAI_ENGINE_LIFECYCLE_MQH__
