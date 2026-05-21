#ifndef __FXAI_STAT_ARIMAX_GARCH_MQH__
#define __FXAI_STAT_ARIMAX_GARCH_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatARIMAXGARCH : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_ARIMAX_GARCH; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_ARIMAX_GARCH; }
   virtual string AIName(void) const { return "stat_arimax_garch"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_ARIMAX_GARCH_MQH__
