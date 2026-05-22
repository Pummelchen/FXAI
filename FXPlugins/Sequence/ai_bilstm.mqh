#ifndef __FXAI_AI_BILSTM_MQH__
#define __FXAI_AI_BILSTM_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIBiLSTM : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_BILSTM; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_RECURRENT; }

public:
   virtual int AIId(void) const { return (int)AI_BILSTM; }
   virtual string AIName(void) const { return "ai_bilstm"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_AI_BILSTM_MQH__
