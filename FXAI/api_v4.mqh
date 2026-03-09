#ifndef __FXAI_API_V4_MQH__
#define __FXAI_API_V4_MQH__

#include "plugin_base.mqh"

bool FXAI_ValidateManifestV4(const FXAIAIManifestV4 &manifest,
                             string &reason)
{
   if(manifest.api_version != FXAI_API_VERSION_V4)
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
   if((manifest.capability_mask & (ulong)FXAI_CAP_ONLINE_LEARNING) == 0)
   {
      reason = "capability_mask";
      return false;
   }
   if(manifest.min_horizon_minutes <= 0 || manifest.max_horizon_minutes < manifest.min_horizon_minutes)
   {
      reason = "horizon_range";
      return false;
   }
   if(manifest.min_sequence_bars <= 0 || manifest.max_sequence_bars < manifest.min_sequence_bars)
   {
      reason = "sequence_range";
      return false;
   }
   reason = "";
   return true;
}

bool FXAI_ValidatePredictionV4(const FXAIAIPredictionV4 &pred,
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
   if(!MathIsValidNumber(pred.move_q25_points) || !MathIsValidNumber(pred.move_q50_points) ||
      !MathIsValidNumber(pred.move_q75_points) ||
      pred.move_q25_points < 0.0 || pred.move_q50_points < pred.move_q25_points ||
      pred.move_q75_points < pred.move_q50_points)
   {
      reason = "move_quantiles";
      return false;
   }
   if(!MathIsValidNumber(pred.confidence) || pred.confidence < 0.0 || pred.confidence > 1.0)
   {
      reason = "confidence";
      return false;
   }
   if(!MathIsValidNumber(pred.reliability) || pred.reliability < 0.0 ||
      pred.reliability > 1.0)
   {
      reason = "reliability";
      return false;
   }
   reason = "";
   return true;
}

void FXAI_TrainViaV4(CFXAIAIPlugin &plugin,
                     const FXAIAITrainRequestV4 &req,
                     const FXAIAIHyperParams &hp)
{
   plugin.Train(req, hp);
}

bool FXAI_PredictViaV4(CFXAIAIPlugin &plugin,
                       const FXAIAIPredictRequestV4 &req,
                       const FXAIAIHyperParams &hp,
                       FXAIAIPredictionV4 &out)
{
   return plugin.Predict(req, hp, out);
}

#endif // __FXAI_API_V4_MQH__
