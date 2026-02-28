// FXAI v1
#ifndef __FX6_API_MQH__
#define __FX6_API_MQH__

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

class CFX6AIRegistry
{
private:
   CFX6AIPlugin *m_plugins[FX6_AI_COUNT];

public:
   CFX6AIRegistry(void)
   {
      for(int i=0; i<FX6_AI_COUNT; i++)
         m_plugins[i] = NULL;
   }

   ~CFX6AIRegistry(void)
   {
      Release();
   }

   void Release(void)
   {
      for(int i=0; i<FX6_AI_COUNT; i++)
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

      for(int i=0; i<FX6_AI_COUNT; i++)
         m_plugins[i] = CreateInstance(i);

      for(int i=0; i<FX6_AI_COUNT; i++)
      {
         if(m_plugins[i] == NULL)
            return false;
      }

      return true;
   }

   void ResetAll(void)
   {
      for(int i=0; i<FX6_AI_COUNT; i++)
      {
         if(m_plugins[i] != NULL)
            m_plugins[i].Reset();
      }
   }

   CFX6AIPlugin *Get(const int ai_id)
   {
      if(ai_id < 0 || ai_id >= FX6_AI_COUNT)
         return NULL;
      return m_plugins[ai_id];
   }

   CFX6AIPlugin *CreateInstance(const int ai_id) const
   {
      switch(ai_id)
      {
         case (int)AI_TYPE_SGD_LOGIT: return new CFX6AISGD();
         case (int)AI_TYPE_FTRL_LOGIT: return new CFX6AIFTRL();
         case (int)AI_TYPE_PA_LINEAR: return new CFX6AIPA();
         case (int)AI_TYPE_XGB_FAST: return new CFX6AIXGBFast();
         case (int)AI_TYPE_MLP_TINY: return new CFX6AIMLPTiny();
         case (int)AI_TYPE_LSTM: return new CFX6AILSTM();
         case (int)AI_TYPE_LSTMG: return new CFX6AILSTMG();
         case (int)AI_TYPE_LIGHTGBM: return new CFX6AILightGBM();
         case (int)AI_TYPE_S4: return new CFX6AIS4();
         case (int)AI_TYPE_TCN: return new CFX6AITCN();
         case (int)AI_TYPE_TFT: return new CFX6AITFT();
         case (int)AI_TYPE_AUTOFORMER: return new CFX6AIAutoformer();
         case (int)AI_TYPE_STMN: return new CFX6AISTMN();
         case (int)AI_TYPE_TST: return new CFX6AITST();
         case (int)AI_TYPE_GEODESICATTENTION: return new CFX6AIGeodesicAttention();
         case (int)AI_TYPE_CATBOOST: return new CFX6AICatBoost();
         case (int)AI_TYPE_PATCHTST: return new CFX6AIPatchTST();
         case (int)AI_TYPE_CHRONOS: return new CFX6AIChronos();
         case (int)AI_TYPE_TIMESFM: return new CFX6AITimesFM();
         case (int)AI_TYPE_XGBOOST: return new CFX6AIXGBoost();
         case (int)AI_TYPE_QUANTILE: return new CFX6AIQuantile();
         case (int)AI_TYPE_ENHASH: return new CFX6AIENHash();
         default: return NULL;
      }
   }
};

#endif // __FX6_API_MQH__
