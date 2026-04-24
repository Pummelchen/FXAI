#ifndef __FXAI_AI_ATTN_CNN_BILSTM_MQH__
#define __FXAI_AI_ATTN_CNN_BILSTM_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIAttnCNNBiLSTM : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_ATTN_CNN_BILSTM; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_RECURRENT; }

public:
   virtual int AIId(void) const { return (int)AI_ATTN_CNN_BILSTM; }
   virtual string AIName(void) const { return "ai_attn_cnn_bilstm"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_AI_ATTN_CNN_BILSTM_MQH__
