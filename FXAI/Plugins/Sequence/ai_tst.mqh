#ifndef __FXAI_AI_TST_MQH__
#define __FXAI_AI_TST_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_TST_SEQ 80
#define FXAI_TST_TBPTT 32
#define FXAI_TST_HEADS 3
#define FXAI_TST_D_HEAD (FXAI_AI_MLP_HIDDEN / FXAI_TST_HEADS)
#define FXAI_TST_CLASS_COUNT 3
#define FXAI_TST_MAX_STACK 4
#define FXAI_TST_HORIZONS 3
#define FXAI_TST_REPLAY 128
#define FXAI_TST_SESSIONS 4

#define FXAI_TST_SELL 0
#define FXAI_TST_BUY  1
#define FXAI_TST_SKIP 2

class CFXAIAITST : public CFXAIAIPlugin
{
private:
#include "ai_tst\ai_tst_state.mqh"

public:
#include "ai_tst\ai_tst_public.mqh"

};


#endif // __FXAI_AI_TST_MQH__
