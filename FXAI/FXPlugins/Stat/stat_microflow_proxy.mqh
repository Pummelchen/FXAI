#ifndef __FXAI_STAT_MICROFLOW_PROXY_MQH__
#define __FXAI_STAT_MICROFLOW_PROXY_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatMicroflowProxy : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_MICROFLOW; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_MICROFLOW_PROXY; }
   virtual string AIName(void) const { return "stat_microflow_proxy"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_MICROFLOW_PROXY_MQH__
