#ifndef __FXAI_STAT_EMD_HHT_MQH__
#define __FXAI_STAT_EMD_HHT_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatEMDHHT : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_EMD_HHT; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_EMD_HHT; }
   virtual string AIName(void) const { return "stat_emd_hht"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_EMD_HHT_MQH__
