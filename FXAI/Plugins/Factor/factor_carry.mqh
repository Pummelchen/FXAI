#ifndef __FXAI_FACTOR_CARRY_MQH__
#define __FXAI_FACTOR_CARRY_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIFactorCarry : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_CARRY; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_OTHER; }

public:
   virtual int AIId(void) const { return (int)AI_FACTOR_CARRY; }
   virtual string AIName(void) const { return "factor_carry"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_FACTOR_CARRY_MQH__
