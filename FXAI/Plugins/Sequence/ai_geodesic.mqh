#ifndef __FXAI_AI_GEODESICATTENTION_MQH__
#define __FXAI_AI_GEODESICATTENTION_MQH__

#include "..\..\API\plugin_base.mqh"

// Reference-grade geodesic attention plugin.
#define FXAI_GA_SEQ 96
#define FXAI_GA_TBPTT 24
#define FXAI_GA_HEADS 4
#define FXAI_GA_D_MODEL 32
#define FXAI_GA_D_HEAD (FXAI_GA_D_MODEL / FXAI_GA_HEADS)
#define FXAI_GA_CLASS_COUNT 3
#define FXAI_GA_BLOCKS 2
#define FXAI_GA_QUANTILES 7
#define FXAI_GA_CAL_BINS 16
#define FXAI_GA_ECE_BINS 12
#define FXAI_GA_REPLAY 256
#define FXAI_GA_SELL 0
#define FXAI_GA_BUY  1
#define FXAI_GA_SKIP 2

class CFXAIAIGeodesicAttention : public CFXAIAIPlugin
{
private:
#include "ai_geodesic\ai_geodesic_state.mqh"

public:
#include "ai_geodesic\ai_geodesic_public.mqh"

};


#endif // __FXAI_AI_GEODESICATTENTION_MQH__
