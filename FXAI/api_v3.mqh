#ifndef __FXAI_API_V3_MQH__
#define __FXAI_API_V3_MQH__

#include "plugin_base.mqh"

bool FXAI_ValidateManifestV3(const FXAIAIManifestV3 &manifest,
                             string &reason)
{
   if(manifest.api_version != FXAI_API_VERSION_V3)
   {
      reason = "api_version";
      return false;
   }
   if(manifest.ai_id < 0 || manifest.ai_id >= FXAI_AI_COUNT)
   {
      reason = "ai_id";
      return false;
   }
   if(StringLen(manifest.ai_name) <= 0)
   {
      reason = "ai_name";
      return false;
   }
   if(!manifest.supports_native_3class)
   {
      reason = "native_3class";
      return false;
   }
   if(manifest.min_horizon_minutes <= 0 || manifest.max_horizon_minutes < manifest.min_horizon_minutes)
   {
      reason = "horizon_range";
      return false;
   }
   reason = "";
   return true;
}

bool FXAI_ValidatePredictionV3(const FXAIAIPredictionV3 &pred,
                               string &reason)
{
   double sum = 0.0;
   for(int c=0; c<3; c++)
   {
      if(!MathIsValidNumber(pred.class_probs[c]) || pred.class_probs[c] < 0.0)
      {
         reason = "class_probs";
         return false;
      }
      sum += pred.class_probs[c];
   }
   if(!MathIsValidNumber(sum) || sum <= 0.0)
   {
      reason = "class_sum";
      return false;
   }
   if(!MathIsValidNumber(pred.move_mean_points) || pred.move_mean_points < 0.0)
   {
      reason = "move_mean";
      return false;
   }
   if(!MathIsValidNumber(pred.move_q25_points) || !MathIsValidNumber(pred.move_q75_points) ||
      pred.move_q25_points < 0.0 || pred.move_q75_points < pred.move_q25_points)
   {
      reason = "move_quantiles";
      return false;
   }
   if(!MathIsValidNumber(pred.confidence) || pred.confidence < 0.0 || pred.confidence > 1.0)
   {
      reason = "confidence";
      return false;
   }
   if(!MathIsValidNumber(pred.calibration_confidence) || pred.calibration_confidence < 0.0 ||
      pred.calibration_confidence > 1.0)
   {
      reason = "calibration_confidence";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_TrainViaV3(CFXAIAIPlugin &plugin,
                     const FXAIAISampleV2 &sample,
                     const FXAIAIHyperParams &hp)
{
   FXAIAITrainRequestV3 req;
   req.ctx.api_version = FXAI_API_VERSION_V3;
   req.ctx.regime_id = sample.regime_id;
   req.ctx.session_bucket = 0;
   req.ctx.horizon_minutes = sample.horizon_minutes;
   req.ctx.feature_schema_id = 1;
   req.ctx.normalization_method_id = 0;
   req.ctx.sequence_bars = 1;
   req.ctx.cost_points = sample.cost_points;
   req.ctx.min_move_points = sample.min_move_points;
   req.ctx.point_value = 0.0;
   req.ctx.sample_time = sample.sample_time;
   req.label_class = sample.label_class;
   req.move_points = sample.move_points;
   req.sample_weight = 1.0;
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      req.x[i] = sample.x[i];
   plugin.TrainV3(req, hp);
}

void FXAI_PredictViaV3(CFXAIAIPlugin &plugin,
                       const FXAIAIPredictV2 &req_v2,
                       const FXAIAIHyperParams &hp,
                       FXAIAIPredictionV2 &out_v2)
{
   FXAIAIPredictRequestV3 req_v3;
   req_v3.ctx.api_version = FXAI_API_VERSION_V3;
   req_v3.ctx.regime_id = req_v2.regime_id;
   req_v3.ctx.session_bucket = 0;
   req_v3.ctx.horizon_minutes = req_v2.horizon_minutes;
   req_v3.ctx.feature_schema_id = 1;
   req_v3.ctx.normalization_method_id = 0;
   req_v3.ctx.sequence_bars = 1;
   req_v3.ctx.cost_points = req_v2.cost_points;
   req_v3.ctx.min_move_points = req_v2.min_move_points;
   req_v3.ctx.point_value = 0.0;
   req_v3.ctx.sample_time = req_v2.sample_time;
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      req_v3.x[i] = req_v2.x[i];

   FXAIAIPredictionV3 pred_v3;
   plugin.PredictV3(req_v3, hp, pred_v3);
   for(int c=0; c<3; c++)
      out_v2.class_probs[c] = pred_v3.class_probs[c];
   out_v2.p_up = out_v2.class_probs[(int)FXAI_LABEL_BUY];
   out_v2.expected_move_points = pred_v3.move_mean_points;
}

#endif // __FXAI_API_V3_MQH__
