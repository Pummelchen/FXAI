#ifndef __FXAI_AI_S4_MQH__
#define __FXAI_AI_S4_MQH__

#include "..\\..\\API\\plugin_base.mqh"

#define FXAI_S4_RANK 6
#define FXAI_S4_TBPTT 28
#define FXAI_S4_HORIZONS 3

class CFXAIAIS4 : public CFXAIAIPlugin
{
private:
   #include "ai_s4\\ai_s4_private.mqh"
public:
   #include "ai_s4\\ai_s4_public.mqh"
};

#endif // __FXAI_AI_S4_MQH__
