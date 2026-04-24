#ifndef __FXAI_STAT_HMM_REGIME_MQH__
#define __FXAI_STAT_HMM_REGIME_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatHMMRegime : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_HMM_REGIME; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_HMM_REGIME; }
   virtual string AIName(void) const { return "stat_hmm_regime"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_HMM_REGIME_MQH__
