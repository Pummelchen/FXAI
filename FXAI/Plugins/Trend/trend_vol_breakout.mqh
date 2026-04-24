#ifndef __FXAI_TREND_VOL_BREAKOUT_MQH__
#define __FXAI_TREND_VOL_BREAKOUT_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAITrendVolBreakout : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_VOL_BREAKOUT; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_CONVOLUTIONAL; }

public:
   virtual int AIId(void) const { return (int)AI_TREND_VOL_BREAKOUT; }
   virtual string AIName(void) const { return "trend_vol_breakout"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_TREND_VOL_BREAKOUT_MQH__
