#ifndef __FXAI_AI_TCN_MQH__
#define __FXAI_AI_TCN_MQH__

#include "..\\..\\API\\plugin_base.mqh"

#define FXAI_TCN_MAX_LAYERS 10
#define FXAI_TCN_MAX_KERNEL 5
#define FXAI_TCN_HIST 448
#define FXAI_TCN_TBPTT 32
#define FXAI_TCN_CLASS_COUNT 3

#define FXAI_TCN_SELL 0
#define FXAI_TCN_BUY  1
#define FXAI_TCN_SKIP 2

class CFXAIAITCN : public CFXAIAIPlugin
{
private:
   #include "ai_tcn\\ai_tcn_private.mqh"
public:
   #include "ai_tcn\\ai_tcn_public.mqh"
};

#endif // __FXAI_AI_TCN_MQH__
