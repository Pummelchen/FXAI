#ifndef __FXAI_CORE_NORMALIZATION_CORE_MQH__
#define __FXAI_CORE_NORMALIZATION_CORE_MQH__

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

#endif // __FXAI_CORE_NORMALIZATION_CORE_MQH__
