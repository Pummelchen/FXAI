#ifndef __FXAI_AI_AUTOFORMER_MQH__
#define __FXAI_AI_AUTOFORMER_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_AF_SEQ 128
#define FXAI_AF_TBPTT 32
#define FXAI_AF_HEADS 4
#define FXAI_AF_D_HEAD (FXAI_AI_MLP_HIDDEN / FXAI_AF_HEADS)
#define FXAI_AF_TOPK_LAGS 8
#define FXAI_AF_CLASS_COUNT 3
#define FXAI_AF_BLOCKS 2
#define FXAI_AF_MA_KERNELS 4
#define FXAI_AF_HORIZONS 3
#define FXAI_AF_CAL_BINS 12
#define FXAI_AF_SESSIONS 4
#define FXAI_AF_COS_CYCLE 4096
#define FXAI_AF_PI 3.14159265358979323846

#define FXAI_AF_SELL 0
#define FXAI_AF_BUY  1
#define FXAI_AF_SKIP 2

class CFXAIAIAutoformer : public CFXAIAIPlugin
{
private:
#include "ai_autoformer\ai_autoformer_state.mqh"
#include "ai_autoformer\ai_autoformer_forward.mqh"

public:
#include "ai_autoformer\ai_autoformer_public.mqh"
#include "ai_autoformer\ai_autoformer_training.mqh"
};

#endif // __FXAI_AI_AUTOFORMER_MQH__
