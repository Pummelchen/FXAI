#ifndef __FXAI_CORE_FEATURE_CORE_MQH__
#define __FXAI_CORE_FEATURE_CORE_MQH__

void FXAI_FeatureCoreResetRequest(FXAIFeatureCoreRequest &request)
{
   request.sample_idx = -1;
   request.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   request.norm_method = FXAI_NORM_EXISTING;
}

void FXAI_FeatureCoreResetFrame(FXAIFeatureCoreFrame &frame)
{
   frame.valid = false;
   frame.sample_idx = -1;
   frame.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   frame.norm_method = FXAI_NORM_EXISTING;
   frame.sample_time = 0;
   frame.spread_points = 0.0;
   frame.has_previous = false;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
   {
      frame.raw[f] = 0.0;
      frame.previous[f] = 0.0;
   }
}

bool FXAI_FeatureCoreBuildFrame(const FXAIDataCoreBundle &bundle,
                                const FXAIFeatureCoreRequest &request,
                                FXAIFeatureCoreFrame &frame)
{
   FXAI_FeatureCoreResetFrame(frame);
   if(!bundle.ready)
      return false;
   if(request.sample_idx < 0 || request.sample_idx >= ArraySize(bundle.close_arr))
      return false;

   double spread_points = FXAI_GetSpreadAtIndex(request.sample_idx,
                                                bundle.spread_m1,
                                                bundle.snapshot.spread_points);
   datetime sample_time = bundle.snapshot.bar_time;
   if(request.sample_idx >= 0 && request.sample_idx < ArraySize(bundle.time_arr))
      sample_time = bundle.time_arr[request.sample_idx];
   double ctx_mean = FXAI_GetArrayValue(bundle.ctx_mean_arr, request.sample_idx, 0.0);
   double ctx_std = FXAI_GetArrayValue(bundle.ctx_std_arr, request.sample_idx, 0.0);
   double ctx_up = FXAI_GetArrayValue(bundle.ctx_up_arr, request.sample_idx, 0.5);

   if(!FXAI_ComputeFeatureVector(request.sample_idx,
                                 bundle.snapshot.symbol,
                                 spread_points,
                                 bundle.time_arr,
                                 bundle.open_arr,
                                 bundle.high_arr,
                                 bundle.low_arr,
                                 bundle.close_arr,
                                 bundle.spread_m1,
                                 bundle.time_m5,
                                 bundle.close_m5,
                                 bundle.map_m5,
                                 bundle.time_m15,
                                 bundle.close_m15,
                                 bundle.map_m15,
                                 bundle.time_m30,
                                 bundle.close_m30,
                                 bundle.map_m30,
                                 bundle.time_h1,
                                 bundle.close_h1,
                                 bundle.map_h1,
                                 ctx_mean,
                                 ctx_std,
                                 ctx_up,
                                 bundle.ctx_extra_arr,
                                 request.norm_method,
                                 frame.raw))
      return false;

   frame.valid = true;
   frame.sample_idx = request.sample_idx;
   frame.horizon_minutes = FXAI_ClampHorizon(request.horizon_minutes);
   frame.norm_method = request.norm_method;
   frame.sample_time = sample_time;
   frame.spread_points = spread_points;

   if(FXAI_FeatureNormNeedsPrevious(request.norm_method) &&
      (request.sample_idx + 1) < ArraySize(bundle.close_arr))
   {
      double prev_spread = FXAI_GetSpreadAtIndex(request.sample_idx + 1,
                                                 bundle.spread_m1,
                                                 spread_points);
      double prev_ctx_mean = FXAI_GetArrayValue(bundle.ctx_mean_arr, request.sample_idx + 1, ctx_mean);
      double prev_ctx_std = FXAI_GetArrayValue(bundle.ctx_std_arr, request.sample_idx + 1, ctx_std);
      double prev_ctx_up = FXAI_GetArrayValue(bundle.ctx_up_arr, request.sample_idx + 1, ctx_up);
      frame.has_previous = FXAI_ComputeFeatureVector(request.sample_idx + 1,
                                                     bundle.snapshot.symbol,
                                                     prev_spread,
                                                     bundle.time_arr,
                                                     bundle.open_arr,
                                                     bundle.high_arr,
                                                     bundle.low_arr,
                                                     bundle.close_arr,
                                                     bundle.spread_m1,
                                                     bundle.time_m5,
                                                     bundle.close_m5,
                                                     bundle.map_m5,
                                                     bundle.time_m15,
                                                     bundle.close_m15,
                                                     bundle.map_m15,
                                                     bundle.time_m30,
                                                     bundle.close_m30,
                                                     bundle.map_m30,
                                                     bundle.time_h1,
                                                     bundle.close_h1,
                                                     bundle.map_h1,
                                                     prev_ctx_mean,
                                                     prev_ctx_std,
                                                     prev_ctx_up,
                                                     bundle.ctx_extra_arr,
                                                     request.norm_method,
                                                     frame.previous);
   }

   return true;
}

bool FXAI_FeatureCoreBuildFrameFromBundle(const FXAIDataCoreBundle &bundle,
                                          const int sample_idx,
                                          const int horizon_minutes,
                                          const ENUM_FXAI_FEATURE_NORMALIZATION norm_method,
                                          FXAIFeatureCoreFrame &frame)
{
   FXAIFeatureCoreRequest request;
   FXAI_FeatureCoreResetRequest(request);
   request.sample_idx = sample_idx;
   request.horizon_minutes = horizon_minutes;
   request.norm_method = norm_method;
   return FXAI_FeatureCoreBuildFrame(bundle, request, frame);
}

#endif // __FXAI_CORE_FEATURE_CORE_MQH__
