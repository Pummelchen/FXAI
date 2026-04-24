#ifndef __FXAI_STAT_XRATE_CONSISTENCY_MQH__
#define __FXAI_STAT_XRATE_CONSISTENCY_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatXRateConsistency : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_XRATE_CONSISTENCY; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_XRATE_CONSISTENCY; }
   virtual string AIName(void) const { return "stat_xrate_consistency"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_XRATE_CONSISTENCY_MQH__
