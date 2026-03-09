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
                     const FXAIAITrainRequestV3 &req,
                     const FXAIAIHyperParams &hp)
{
   plugin.TrainV3(req, hp);
}

bool FXAI_PredictViaV3(CFXAIAIPlugin &plugin,
                       const FXAIAIPredictRequestV3 &req,
                       const FXAIAIHyperParams &hp,
                       FXAIAIPredictionV3 &out)
{
   return plugin.PredictV3(req, hp, out);
}

#endif // __FXAI_API_V3_MQH__
