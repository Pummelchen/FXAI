#ifndef __FXAI_AI_PA_MQH__
#define __FXAI_AI_PA_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_PA_CLASS_COUNT 3
#define FXAI_PA_HASH2_BUCKETS 97
#define FXAI_PA_CAL_BINS 10
#define FXAI_PA_REPLAY 192
#define FXAI_PA_TOP_RIVALS 2

class CFXAIAIPA : public CFXAIAIPlugin
{
private:
#include "lin_pa\lin_pa_private.mqh"

public:
#include "lin_pa\lin_pa_public.mqh"

protected:
#include "lin_pa\lin_pa_training.mqh"
};

#endif // __FXAI_AI_PA_MQH__
