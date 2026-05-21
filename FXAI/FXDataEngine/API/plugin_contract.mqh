#ifndef __FXAI_PLUGIN_CONTRACT_MQH__
#define __FXAI_PLUGIN_CONTRACT_MQH__

#include "..\Engine\core.mqh"
#include "..\TensorCore\TensorCore.mqh"

#define FXAI_PLUGIN_STATE_ARTIFACT_DIR "FXAI\\Runtime\\Plugins"
#define FXAI_PLUGIN_STATE_ARTIFACT_VERSION 12

#include "Contract\plugin_contract_support.mqh"

class CFXAIAIPlugin
{
protected:
#include "plugin_context.mqh"
#include "plugin_tensor_bridge.mqh"
#include "plugin_quality_heads.mqh"

public:
#include "Contract\plugin_contract_public.mqh"

protected:
#include "Contract\plugin_contract_persistence.mqh"
};

#endif // __FXAI_PLUGIN_CONTRACT_MQH__
