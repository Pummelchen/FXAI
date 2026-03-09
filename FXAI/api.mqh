// FXAI v2
// FXAI v1
#ifndef __FXAI_API_MQH__
#define __FXAI_API_MQH__

#include "plugin_base.mqh"

#include "Plugins\ai_sgd.mqh"
#include "Plugins\ai_ftrl.mqh"
#include "Plugins\ai_pa.mqh"
#include "Plugins\ai_xgb_fast.mqh"
#include "Plugins\ai_mlp_tiny.mqh"
#include "Plugins\ai_lstm.mqh"
#include "Plugins\ai_lstmg.mqh"
#include "Plugins\ai_lightgbm.mqh"
#include "Plugins\ai_s4.mqh"
#include "Plugins\ai_tcn.mqh"
#include "Plugins\ai_tft.mqh"
#include "Plugins\ai_autoformer.mqh"
#include "Plugins\ai_stmn.mqh"
#include "Plugins\ai_tst.mqh"
#include "Plugins\ai_geodesicattention.mqh"
#include "Plugins\ai_xgboost.mqh"
#include "Plugins\ai_quantile.mqh"
#include "Plugins\ai_enhash.mqh"
#include "Plugins\ai_catboost.mqh"
#include "Plugins\ai_patchtst.mqh"
#include "Plugins\ai_chronos.mqh"
#include "Plugins\ai_timesfm.mqh"
#include "Plugins\ai_cfx_world.mqh"
#include "Plugins\ai_loffm.mqh"
#include "Plugins\ai_trr.mqh"
#include "Plugins\ai_graphwm.mqh"
#include "Plugins\ai_moe_conformal.mqh"
#include "Plugins\ai_retrdiff.mqh"

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
         default: return NULL;
      }
   }
};

#endif // __FXAI_API_MQH__
