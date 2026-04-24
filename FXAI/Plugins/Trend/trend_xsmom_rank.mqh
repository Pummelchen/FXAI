#ifndef __FXAI_TREND_XSMOM_RANK_MQH__
#define __FXAI_TREND_XSMOM_RANK_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAITrendXSMOMRank : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_XSMOM_RANK; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_OTHER; }

public:
   virtual int AIId(void) const { return (int)AI_TREND_XSMOM_RANK; }
   virtual string AIName(void) const { return "trend_xsmom_rank"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_TREND_XSMOM_RANK_MQH__
