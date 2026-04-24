#ifndef __FXAI_LIN_PROFIT_LOGIT_MQH__
#define __FXAI_LIN_PROFIT_LOGIT_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAILinProfitLogit : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_PROFIT_LOGIT; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_LINEAR; }

public:
   virtual int AIId(void) const { return (int)AI_LIN_PROFIT_LOGIT; }
   virtual string AIName(void) const { return "lin_profit_logit"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_LIN_PROFIT_LOGIT_MQH__
