#ifndef __FXAI_AI_STMN_MQH__
#define __FXAI_AI_STMN_MQH__

#include "..\\..\\API\\plugin_base.mqh"

#define FXAI_STMN_NODES 6
#define FXAI_STMN_CLASS_COUNT 3
#define FXAI_STMN_SEQ 128
#define FXAI_STMN_TBPTT 16

#define FXAI_STMN_SELL 0
#define FXAI_STMN_BUY  1
#define FXAI_STMN_SKIP 2

class CFXAIAISTMN : public CFXAIAIPlugin
{
private:
   #include "ai_stmn\\ai_stmn_private.mqh"
public:
   #include "ai_stmn\\ai_stmn_public.mqh"
};

#endif // __FXAI_AI_STMN_MQH__
