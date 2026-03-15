#ifndef __FXAI_AI_TFT_MQH__
#define __FXAI_AI_TFT_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_TFT_SEQ 64
#define FXAI_TFT_TBPTT 24
#define FXAI_TFT_HEADS 3
#define FXAI_TFT_D_HEAD (FXAI_AI_MLP_HIDDEN / FXAI_TFT_HEADS)
#define FXAI_TFT_CLASS_COUNT 3
#define FXAI_TFT_WF 256
#define FXAI_TFT_SESSIONS 4

#define FXAI_TFT_SELL 0
#define FXAI_TFT_BUY  1
#define FXAI_TFT_SKIP 2

class CFXAIAITFT : public CFXAIAIPlugin
{
private:
#include "ai_tft\ai_tft_state.mqh"
#include "ai_tft\ai_tft_forward.mqh"

public:
#include "ai_tft\ai_tft_public.mqh"

protected:
#include "ai_tft\ai_tft_training.mqh"
};

#endif // __FXAI_AI_TFT_MQH__
