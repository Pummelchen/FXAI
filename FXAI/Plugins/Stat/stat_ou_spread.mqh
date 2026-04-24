#ifndef __FXAI_STAT_OU_SPREAD_MQH__
#define __FXAI_STAT_OU_SPREAD_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatOUSpread : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_OU_SPREAD; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_OU_SPREAD; }
   virtual string AIName(void) const { return "stat_ou_spread"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_OU_SPREAD_MQH__
