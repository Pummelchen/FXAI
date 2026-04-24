#ifndef __FXAI_STAT_VMD_MQH__
#define __FXAI_STAT_VMD_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIStatVMD : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_VMD; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_STAT_VMD; }
   virtual string AIName(void) const { return "stat_vmd"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_STAT_VMD_MQH__
