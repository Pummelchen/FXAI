#ifndef __FXAI_API_MQH__
#define __FXAI_API_MQH__

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
   if(manifest.family < (int)FXAI_FAMILY_LINEAR || manifest.family > (int)FXAI_FAMILY_OTHER)
   {
      reason = "family";
      return false;
   }
   if(manifest.feature_schema_id <= 0)
   {
      reason = "feature_schema_id";
      return false;
   }
   if(manifest.feature_groups_mask == 0)
   {
      reason = "feature_groups_mask";
      return false;
   }
   if((manifest.capability_mask & (ulong)FXAI_CAP_SELF_TEST) == 0)
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
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_REPLAY) &&
      !FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_ONLINE_LEARNING))
   {
      reason = "replay_requires_online_learning";
      return false;
   }
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT) &&
      manifest.max_sequence_bars <= 1)
   {
      reason = "window_context_sequence_range";
      return false;
   }
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL) &&
      manifest.max_sequence_bars <= 1)
   {
      reason = "stateful_sequence_range";
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
   if(MathAbs(sum - 1.0) > 0.02)
   {
      reason = "class_sum_norm";
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

int FXAI_ResolveManifestSequenceBars(const FXAIAIManifestV4 &manifest,
                                     const int horizon_minutes)
{
   if(!FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT) &&
      !FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      return 1;

   int min_seq = manifest.min_sequence_bars;
   int max_seq = manifest.max_sequence_bars;
   if(min_seq < 1) min_seq = 1;
   if(max_seq < min_seq) max_seq = min_seq;

   int h = (horizon_minutes > 0 ? horizon_minutes : 1);
   int seq = h * 8;
   if(seq < min_seq) seq = min_seq;
   if(seq > max_seq) seq = max_seq;
   return seq;
}

int FXAI_GetPluginSequenceBars(CFXAIAIPlugin &plugin,
                               const int horizon_minutes)
{
   FXAIAIManifestV4 manifest;
   plugin.Describe(manifest);
   return FXAI_ResolveManifestSequenceBars(manifest, horizon_minutes);
}

void FXAI_GetPluginManifest(CFXAIAIPlugin &plugin,
                            FXAIAIManifestV4 &manifest)
{
   plugin.Describe(manifest);
   if(manifest.feature_groups_mask == 0)
      manifest.feature_groups_mask = FXAI_DefaultFeatureGroupsForFamily(manifest.family);
   if(manifest.feature_schema_id <= 0)
      manifest.feature_schema_id = FXAI_DefaultFeatureSchemaForFamily(manifest.family);
}

int FXAI_GetPluginFeatureSchema(CFXAIAIPlugin &plugin)
{
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   return manifest.feature_schema_id;
}

ulong FXAI_GetPluginFeatureGroupsMask(CFXAIAIPlugin &plugin)
{
   FXAIAIManifestV4 manifest;
   FXAI_GetPluginManifest(plugin, manifest);
   return manifest.feature_groups_mask;
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

#include "..\Plugins\lin_sgd.mqh"
#include "..\Plugins\lin_ftrl.mqh"
#include "..\Plugins\lin_pa.mqh"
#include "..\Plugins\tree_xgb_fast.mqh"
#include "..\Plugins\ai_mlp.mqh"
#include "..\Plugins\ai_lstm.mqh"
#include "..\Plugins\ai_lstmg.mqh"
#include "..\Plugins\tree_lgbm.mqh"
#include "..\Plugins\ai_s4.mqh"
#include "..\Plugins\ai_tcn.mqh"
#include "..\Plugins\ai_tft.mqh"
#include "..\Plugins\ai_autoformer.mqh"
#include "..\Plugins\ai_stmn.mqh"
#include "..\Plugins\ai_tst.mqh"
#include "..\Plugins\ai_geodesic.mqh"
#include "..\Plugins\tree_xgb.mqh"
#include "..\Plugins\dist_quantile.mqh"
#include "..\Plugins\lin_enhash.mqh"
#include "..\Plugins\tree_catboost.mqh"
#include "..\Plugins\ai_patchtst.mqh"
#include "..\Plugins\ai_chronos.mqh"
#include "..\Plugins\ai_timesfm.mqh"
#include "..\Plugins\wm_cfx.mqh"
#include "..\Plugins\mix_loffm.mqh"
#include "..\Plugins\ai_trr.mqh"
#include "..\Plugins\wm_graph.mqh"
#include "..\Plugins\mix_moe_conformal.mqh"
#include "..\Plugins\mem_retrdiff.mqh"
#include "..\Plugins\rule_m1sync.mqh"
#include "..\Plugins\rule_buyonly.mqh"
#include "..\Plugins\rule_sellonly.mqh"
#include "..\Plugins\rule_random.mqh"

class CFXAIAIRegistry
{
private:
   CFXAIAIPlugin *m_plugins[FXAI_AI_COUNT];

public:
   CFXAIAIRegistry(void)
   {
      for(int i=0; i<FXAI_AI_COUNT; i++)
         m_plugins[i] = NULL;
   }

   ~CFXAIAIRegistry(void)
   {
      Release();
   }

   void Release(void)
   {
      for(int i=0; i<FXAI_AI_COUNT; i++)
      {
         if(m_plugins[i] != NULL)
         {
            delete m_plugins[i];
            m_plugins[i] = NULL;
         }
      }
   }

   bool Initialize(void)
   {
      Release();

      for(int i=0; i<FXAI_AI_COUNT; i++)
         m_plugins[i] = CreateInstance(i);

      for(int i=0; i<FXAI_AI_COUNT; i++)
      {
         if(m_plugins[i] == NULL)
            return false;
      }

      return true;
   }

   void ResetAll(void)
   {
      for(int i=0; i<FXAI_AI_COUNT; i++)
      {
         if(m_plugins[i] != NULL)
            m_plugins[i].Reset();
      }
   }

   CFXAIAIPlugin *Get(const int ai_id)
   {
      if(ai_id < 0 || ai_id >= FXAI_AI_COUNT)
         return NULL;
      return m_plugins[ai_id];
   }

   CFXAIAIPlugin *CreateInstance(const int ai_id) const
   {
      switch(ai_id)
      {
         case (int)AI_SGD_LOGIT: return new CFXAIAISGD();
         case (int)AI_FTRL_LOGIT: return new CFXAIAIFTRL();
         case (int)AI_PA_LINEAR: return new CFXAIAIPA();
         case (int)AI_XGB_FAST: return new CFXAIAIXGBFast();
         case (int)AI_MLP_TINY: return new CFXAIAIMLPTiny();
         case (int)AI_LSTM: return new CFXAIAILSTM();
         case (int)AI_LSTMG: return new CFXAIAILSTMG();
         case (int)AI_LIGHTGBM: return new CFXAIAILightGBM();
         case (int)AI_S4: return new CFXAIAIS4();
         case (int)AI_TCN: return new CFXAIAITCN();
         case (int)AI_TFT: return new CFXAIAITFT();
         case (int)AI_AUTOFORMER: return new CFXAIAIAutoformer();
         case (int)AI_STMN: return new CFXAIAISTMN();
         case (int)AI_TST: return new CFXAIAITST();
         case (int)AI_GEODESICATTENTION: return new CFXAIAIGeodesicAttention();
         case (int)AI_CATBOOST: return new CFXAIAICatBoost();
         case (int)AI_PATCHTST: return new CFXAIAIPatchTST();
         case (int)AI_CHRONOS: return new CFXAIAIChronos();
         case (int)AI_TIMESFM: return new CFXAIAITimesFM();
         case (int)AI_XGBOOST: return new CFXAIAIXGBoost();
         case (int)AI_QUANTILE: return new CFXAIAIQuantile();
         case (int)AI_ENHASH: return new CFXAIAIENHash();
         case (int)AI_CFX_WORLD: return new CFXAIAICFXWorld();
         case (int)AI_LOFFM: return new CFXAIAILOFFM();
         case (int)AI_TRR: return new CFXAIAITRR();
         case (int)AI_GRAPHWM: return new CFXAIAIGraphWM();
         case (int)AI_MOE_CONFORMAL: return new CFXAIAIMoEConformal();
         case (int)AI_RETRDIFF: return new CFXAIAIRetrDiff();
         case (int)AI_M1SYNC: return new CFXAIAIM1Sync();
         case (int)AI_BUY_ONLY: return new CFXAIAIRuleBuyOnly();
         case (int)AI_SELL_ONLY: return new CFXAIAIRuleSellOnly();
         case (int)AI_RANDOM_NOSKIP: return new CFXAIAIRuleRandom();
         default: return NULL;
      }
   }
};

#endif // __FXAI_API_MQH__
