#ifndef __FXAI_AI_TIMESFM_MQH__
#define __FXAI_AI_TIMESFM_MQH__

#include "..\..\API\plugin_base.mqh"

// TimesFM foundation-model plugin for FXAI.
// Design: tokenized multivariate context -> encoder stack -> memory retrieval
// -> 3-class probabilities + move-distribution heads (mu/logvar/q25/q75).
#define FXAI_TFM_CLASS_COUNT 3
#define FXAI_TFM_SEQ 96
#define FXAI_TFM_PATCH_LEN 8
#define FXAI_TFM_STRIDE 4
#define FXAI_TFM_MAX_PATCHES 24
#define FXAI_TFM_LAYERS 2
#define FXAI_TFM_HEADS 2
#define FXAI_TFM_D_MODEL FXAI_AI_MLP_HIDDEN
#define FXAI_TFM_D_HEAD (FXAI_TFM_D_MODEL / FXAI_TFM_HEADS)
#define FXAI_TFM_D_FF 16
#define FXAI_TFM_CAL_BINS 12
#define FXAI_TFM_VALUE_BINS 16
#define FXAI_TFM_CODEBOOK 64
#define FXAI_TFM_MEMORY 16
#define FXAI_TFM_HORIZONS 3

class CFXAIAITimesFM : public CFXAIAIPlugin
{
private:
#include "ai_timesfm\ai_timesfm_state.mqh"

public:
#include "ai_timesfm\ai_timesfm_public.mqh"

protected:
#include "ai_timesfm\ai_timesfm_training.mqh"

};


#endif // __FXAI_AI_TIMESFM_MQH__
