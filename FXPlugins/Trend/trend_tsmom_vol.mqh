#ifndef __FXAI_TREND_TSMOM_VOL_MQH__
#define __FXAI_TREND_TSMOM_VOL_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAITrendTSMOMVol : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_TSMOM_VOL; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_CONVOLUTIONAL; }

public:
   virtual int AIId(void) const { return (int)AI_TREND_TSMOM_VOL; }
   virtual string AIName(void) const { return "trend_tsmom_vol"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_TREND_TSMOM_VOL_MQH__
