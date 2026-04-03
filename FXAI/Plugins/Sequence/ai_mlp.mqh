#ifndef __FXAI_AI_MLP_TINY_MQH__
#define __FXAI_AI_MLP_TINY_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_MLP_H1 16
#define FXAI_MLP_H2 16
#define FXAI_MLP_CTX 6
#define FXAI_MLP_CLASSES 3
#define FXAI_MLP_HIST 8
#define FXAI_MLP_REPLAY 384
#define FXAI_MLP_CAL_BINS 12
#define FXAI_MLP_ECE_BINS 12

class CFXAIAIMLPTiny : public CFXAIAIPlugin
{
private:
#include "ai_mlp\ai_mlp_state.mqh"

public:
#include "ai_mlp\ai_mlp_public.mqh"

};


#endif // __FXAI_AI_MLP_TINY_MQH__
