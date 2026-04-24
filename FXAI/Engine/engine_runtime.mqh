#ifndef __FXAI_ENGINE_RUNTIME_MQH__
#define __FXAI_ENGINE_RUNTIME_MQH__

#include "Runtime\\runtime_signal_state.mqh"
#include "Runtime\\runtime_router_stage.mqh"
#include "Runtime\\runtime_adaptive_router_stage.mqh"
#include "Runtime\\runtime_dynamic_ensemble_stage.mqh"
#include "Runtime\\runtime_execution_quality_stage.mqh"
#include "Runtime\\runtime_prob_calibration_stage.mqh"
#include "Runtime\\runtime_signal_finalize.mqh"

int SpecialDirectionAI(const string symbol)
{
   ulong runtime_total_t0 = GetMicrosecondCount();
   ulong feature_stage_t0 = runtime_total_t0;
   g_ai_last_reason = "start";
   FXAIRuntimeSignalCache signal_cache;
   FXAI_RuntimeCaptureSignalCache(signal_cache);
   FXAI_RuntimeResetSignalState();
   if(!g_plugins_ready)
   {
      g_ai_last_reason = "plugins_not_ready";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int base_h = FXAI_ClampHorizon(PredictionTargetMinutes);

   int base = AI_Window;
   if(base < 50) base = 50;
   if(base > 500) base = 500;

   int K = AI_OnlineSamples;
   if(K < 5) K = 5;
   if(K > 200) K = 200;

   int onlineEpochs = AI_OnlineEpochs;
   if(onlineEpochs < 1) onlineEpochs = 1;
   if(onlineEpochs > 5) onlineEpochs = 5;

   int trainEpochs = AI_Epochs;
   if(trainEpochs < 1) trainEpochs = 1;
   if(trainEpochs > 20) trainEpochs = 20;

   int aiType = (int)AI_Type;
   if(aiType < 0 || aiType >= FXAI_AI_COUNT)
      aiType = (int)AI_SGD_LOGIT;

   bool ensembleMode = (bool)AI_Ensemble;
   double agreePct = FXAI_Clamp(Ensemble_AgreePct, 50.0, 100.0);

   double buyThr = AI_BuyThreshold;
   double sellThr = AI_SellThreshold;
   FXAI_SanitizeThresholdPair(buyThr, sellThr);

   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int evLookback = AI_EVLookbackSamples;
   if(evLookback < 20) evLookback = 20;
   if(evLookback > 400) evLookback = 400;

   if(g_ai_last_symbol != symbol)
      ResetAIState(symbol);

   if(AI_Warmup && !g_ai_warmup_done)
   {
      if(!FXAI_WarmupTrainAndTune(symbol))
      {
         g_ai_last_reason = "warmup_pending";
         FXAI_RuntimePublishIdleSnapshot(symbol);
         return -1;
      }
   }

   datetime signal_bar = 0;
   if(!FXAI_MarketDataBarTime(symbol, PERIOD_M1, 1, signal_bar))
      signal_bar = 0;
   if(signal_bar == 0)
   {
      g_ai_last_reason = "bar_time_failed";
      FXAI_RuntimePublishIdleSnapshot(symbol);
      return -1;
   }

   int pctKey = (int)MathRound(agreePct * 10.0);
   int decisionKey = (ensembleMode == 1 ? (100000 + pctKey) : aiType);
   if(g_ai_last_signal_bar == signal_bar && g_ai_last_signal_key == decisionKey)
   {
      FXAI_RuntimeRestoreSignalCache(signal_cache);
      g_ai_last_reason = "signal_cache_hit";
      return g_ai_last_signal;
   }

#include "Runtime\runtime_feature_pipeline_block.mqh"

#include "Runtime\runtime_transfer_stage_block.mqh"

#include "Runtime\runtime_model_stage_block.mqh"

#include "Runtime\runtime_policy_stage_block.mqh"

   return FXAI_RuntimeFinalizeDecision(symbol,
                                       decision,
                                       signal_bar,
                                       decisionKey,
                                       (ensembleMode != 0),
                                       aiType,
                                       singleNoTradeReason,
                                       ensemble_meta_total,
                                       macro_profile_shortfall,
                                       regime_transition_penalty,
                                       runtime_total_t0);
}


#endif // __FXAI_ENGINE_RUNTIME_MQH__
