#ifndef __FXAI_AI_CHRONOS_MQH__
#define __FXAI_AI_CHRONOS_MQH__

#include "..\..\API\plugin_base.mqh"

// Chronos foundation-model plugin for FXAI.
// Design: tokenized multivariate context -> encoder stack -> memory retrieval
// -> 3-class probabilities + move-distribution heads (mu/logvar/q25/q75).
#define FXAI_CHR_CLASS_COUNT 3
#define FXAI_CHR_SEQ 128
#define FXAI_CHR_PATCH_LEN 8
#define FXAI_CHR_STRIDE 4
#define FXAI_CHR_MAX_PATCHES 32
#define FXAI_CHR_LAYERS 4
#define FXAI_CHR_HEADS 4
#define FXAI_CHR_D_MODEL 32
#define FXAI_CHR_D_HEAD (FXAI_CHR_D_MODEL / FXAI_CHR_HEADS)
#define FXAI_CHR_D_FF 128
#define FXAI_CHR_CAL_BINS 16
#define FXAI_CHR_VALUE_BINS 32
#define FXAI_CHR_CODEBOOK 128
#define FXAI_CHR_MEMORY 32
#define FXAI_CHR_HORIZONS 4
#define FXAI_CHR_QUANTILES 7
#define FXAI_CHR_REPLAY 256
#define FXAI_CHR_ECE_BINS 12

class CFXAIAIChronos : public CFXAIAIPlugin
{
private:
#include "ai_chronos\ai_chronos_state.mqh"
#include "ai_chronos\ai_chronos_forward.mqh"

public:
#include "ai_chronos\ai_chronos_public.mqh"

protected:
#include "ai_chronos\ai_chronos_training.mqh"
};

#endif // __FXAI_AI_CHRONOS_MQH__
