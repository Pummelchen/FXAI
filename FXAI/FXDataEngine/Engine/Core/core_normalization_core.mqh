#ifndef __FXAI_CORE_NORMALIZATION_CORE_MQH__
#define __FXAI_CORE_NORMALIZATION_CORE_MQH__

ulong FXAI_NormalizationCoreDefaultGroupsMask(void)
{
   ulong mask = 0;
   for(int group_id=0; group_id<=(int)FXAI_FEAT_GROUP_FILTERS; group_id++)
      mask |= FXAI_FeatureGroupBit(group_id);
   return mask;
}

void FXAI_NormalizationCoreResetPayloadRequest(FXAINormalizationPayloadRequest &request)
{
   request.valid = false;
   request.feature_schema_id = (int)FXAI_SCHEMA_FULL;
   request.feature_groups_mask = FXAI_NormalizationCoreDefaultGroupsMask();
   request.normalization_method_id = (int)FXAI_NORM_EXISTING;
   request.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   request.sequence_bars = 1;
   request.sample_time = 0;
   request.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      request.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         request.x_window[b][k] = 0.0;
}

void FXAI_NormalizationCoreResetPayloadFrame(FXAINormalizationPayloadFrame &frame)
{
   frame.valid = false;
   frame.feature_schema_id = (int)FXAI_SCHEMA_FULL;
   frame.feature_groups_mask = FXAI_NormalizationCoreDefaultGroupsMask();
   frame.normalization_method_id = (int)FXAI_NORM_EXISTING;
   frame.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   frame.sequence_bars = 1;
   frame.sample_time = 0;
   frame.window_size = 0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      frame.x[k] = 0.0;
   for(int b=0; b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         frame.x_window[b][k] = 0.0;
}

void FXAI_NormalizationCoreResetFrame(FXAINormalizationCoreFrame &frame)
{
   frame.valid = false;
   frame.horizon_minutes = FXAI_ClampHorizon(PredictionTargetMinutes);
   frame.norm_method = FXAI_NORM_EXISTING;
   frame.sample_time = 0;
   for(int f=0; f<FXAI_AI_FEATURES; f++)
      frame.normalized[f] = 0.0;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      frame.model_input[k] = 0.0;
}

bool FXAI_NormalizationCoreBuildInputFrameFromFeatureFrame(const FXAIFeatureCoreFrame &feature_frame,
                                                           FXAINormalizationCoreFrame &frame)
{
   FXAI_NormalizationCoreResetFrame(frame);
   if(!feature_frame.valid)
      return false;

   frame.horizon_minutes = feature_frame.horizon_minutes;
   frame.norm_method = feature_frame.norm_method;
   frame.sample_time = feature_frame.sample_time;
   FXAI_ApplyFeatureNormalizationEx(feature_frame.norm_method,
                                    feature_frame.horizon_minutes,
                                    feature_frame.raw,
                                    feature_frame.previous,
                                    feature_frame.has_previous,
                                    feature_frame.sample_time,
                                    frame.normalized);
   FXAI_BuildInputVector(frame.normalized, frame.model_input);
   frame.valid = true;
   return true;
}

bool FXAI_NormalizationCoreBuildPayloadFrame(const FXAINormalizationPayloadRequest &request,
                                             FXAINormalizationPayloadFrame &frame)
{
   FXAI_NormalizationCoreResetPayloadFrame(frame);
   if(!request.valid)
      return false;

   frame.feature_schema_id = request.feature_schema_id;
   frame.feature_groups_mask = request.feature_groups_mask;
   frame.normalization_method_id = request.normalization_method_id;
   frame.horizon_minutes = FXAI_ClampHorizon(request.horizon_minutes);
   frame.sequence_bars = request.sequence_bars;
   frame.sample_time = request.sample_time;
   frame.window_size = request.window_size;
   if(frame.window_size < 0)
      frame.window_size = 0;
   if(frame.window_size > FXAI_MAX_SEQUENCE_BARS)
      frame.window_size = FXAI_MAX_SEQUENCE_BARS;

   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      frame.x[k] = request.x[k];
   for(int b=0; b<frame.window_size; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         frame.x_window[b][k] = request.x_window[b][k];

   FXAI_ApplyPayloadTransformPipelineEx(frame.feature_schema_id,
                                        frame.feature_groups_mask,
                                        frame.normalization_method_id,
                                        frame.horizon_minutes,
                                        frame.sequence_bars,
                                        frame.x_window,
                                        frame.window_size,
                                        frame.x);
   frame.valid = true;
   return true;
}

bool FXAI_NormalizationCoreFinalizePredictRequest(const FXAIAIManifestV4 &manifest,
                                                  FXAIAIPredictRequestV4 &req)
{
   FXAINormalizationPayloadRequest payload_request;
   FXAI_NormalizationCoreResetPayloadRequest(payload_request);
   payload_request.valid = req.valid;
   payload_request.feature_schema_id = manifest.feature_schema_id;
   payload_request.feature_groups_mask = manifest.feature_groups_mask;
   payload_request.normalization_method_id = req.ctx.normalization_method_id;
   payload_request.horizon_minutes = req.ctx.horizon_minutes;
   payload_request.sequence_bars = req.ctx.sequence_bars;
   payload_request.sample_time = req.ctx.sample_time;
   payload_request.window_size = req.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      payload_request.x[k] = req.x[k];
   for(int b=0; b<payload_request.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         payload_request.x_window[b][k] = req.x_window[b][k];

   FXAINormalizationPayloadFrame payload_frame;
   if(!FXAI_NormalizationCoreBuildPayloadFrame(payload_request, payload_frame))
      return false;

   req.window_size = payload_frame.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = payload_frame.x[k];
   for(int b=0; b<req.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = payload_frame.x_window[b][k];
   return true;
}

bool FXAI_NormalizationCoreFinalizePredictPayload(const int feature_schema_id,
                                                  const ulong feature_groups_mask,
                                                  FXAIAIPredictRequestV4 &req)
{
   FXAINormalizationPayloadRequest payload_request;
   FXAI_NormalizationCoreResetPayloadRequest(payload_request);
   payload_request.valid = req.valid;
   payload_request.feature_schema_id = feature_schema_id;
   payload_request.feature_groups_mask = feature_groups_mask;
   payload_request.normalization_method_id = req.ctx.normalization_method_id;
   payload_request.horizon_minutes = req.ctx.horizon_minutes;
   payload_request.sequence_bars = req.ctx.sequence_bars;
   payload_request.sample_time = req.ctx.sample_time;
   payload_request.window_size = req.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      payload_request.x[k] = req.x[k];
   for(int b=0; b<payload_request.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         payload_request.x_window[b][k] = req.x_window[b][k];

   FXAINormalizationPayloadFrame payload_frame;
   if(!FXAI_NormalizationCoreBuildPayloadFrame(payload_request, payload_frame))
      return false;

   req.window_size = payload_frame.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = payload_frame.x[k];
   for(int b=0; b<req.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = payload_frame.x_window[b][k];
   return true;
}

bool FXAI_NormalizationCoreFinalizeTrainRequest(const FXAIAIManifestV4 &manifest,
                                                FXAIAITrainRequestV4 &req)
{
   FXAINormalizationPayloadRequest payload_request;
   FXAI_NormalizationCoreResetPayloadRequest(payload_request);
   payload_request.valid = req.valid;
   payload_request.feature_schema_id = manifest.feature_schema_id;
   payload_request.feature_groups_mask = manifest.feature_groups_mask;
   payload_request.normalization_method_id = req.ctx.normalization_method_id;
   payload_request.horizon_minutes = req.ctx.horizon_minutes;
   payload_request.sequence_bars = req.ctx.sequence_bars;
   payload_request.sample_time = req.ctx.sample_time;
   payload_request.window_size = req.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      payload_request.x[k] = req.x[k];
   for(int b=0; b<payload_request.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         payload_request.x_window[b][k] = req.x_window[b][k];

   FXAINormalizationPayloadFrame payload_frame;
   if(!FXAI_NormalizationCoreBuildPayloadFrame(payload_request, payload_frame))
      return false;

   req.window_size = payload_frame.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = payload_frame.x[k];
   for(int b=0; b<req.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = payload_frame.x_window[b][k];
   return true;
}

bool FXAI_NormalizationCoreFinalizeTrainPayload(const int feature_schema_id,
                                                const ulong feature_groups_mask,
                                                FXAIAITrainRequestV4 &req)
{
   FXAINormalizationPayloadRequest payload_request;
   FXAI_NormalizationCoreResetPayloadRequest(payload_request);
   payload_request.valid = req.valid;
   payload_request.feature_schema_id = feature_schema_id;
   payload_request.feature_groups_mask = feature_groups_mask;
   payload_request.normalization_method_id = req.ctx.normalization_method_id;
   payload_request.horizon_minutes = req.ctx.horizon_minutes;
   payload_request.sequence_bars = req.ctx.sequence_bars;
   payload_request.sample_time = req.ctx.sample_time;
   payload_request.window_size = req.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      payload_request.x[k] = req.x[k];
   for(int b=0; b<payload_request.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         payload_request.x_window[b][k] = req.x_window[b][k];

   FXAINormalizationPayloadFrame payload_frame;
   if(!FXAI_NormalizationCoreBuildPayloadFrame(payload_request, payload_frame))
      return false;

   req.window_size = payload_frame.window_size;
   for(int k=0; k<FXAI_AI_WEIGHTS; k++)
      req.x[k] = payload_frame.x[k];
   for(int b=0; b<req.window_size && b<FXAI_MAX_SEQUENCE_BARS; b++)
      for(int k=0; k<FXAI_AI_WEIGHTS; k++)
         req.x_window[b][k] = payload_frame.x_window[b][k];
   return true;
}

#endif // __FXAI_CORE_NORMALIZATION_CORE_MQH__
