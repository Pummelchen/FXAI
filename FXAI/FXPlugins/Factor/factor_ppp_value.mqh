#ifndef __FXAI_FACTOR_PPP_VALUE_MQH__
#define __FXAI_FACTOR_PPP_VALUE_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIFactorPPPValue : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_PPP_VALUE; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_OTHER; }

public:
   virtual int AIId(void) const { return (int)AI_FACTOR_PPP_VALUE; }
   virtual string AIName(void) const { return "factor_ppp_value"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_FACTOR_PPP_VALUE_MQH__
