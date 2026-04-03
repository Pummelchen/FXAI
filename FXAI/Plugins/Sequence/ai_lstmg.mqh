#ifndef __FXAI_AI_LSTMG_MQH__
#define __FXAI_AI_LSTMG_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_LSTMG_TBPTT 32
#define FXAI_LSTMG_CLASS_COUNT 3
#define FXAI_LSTMG_CAL_BINS 12
#define FXAI_LSTMG_LN_EPS 0.00001
#define FXAI_LSTMG_DROP_RATE 0.10
#define FXAI_LSTMG_ZONEOUT 0.06
#define FXAI_LSTMG_OPT_GROUPS 6

class CFXAIAILSTMG : public CFXAIAIPlugin
{
private:
#include "ai_lstmg\ai_lstmg_state.mqh"

public:
#include "ai_lstmg\ai_lstmg_public.mqh"

};


#endif // __FXAI_AI_LSTMG_MQH__
