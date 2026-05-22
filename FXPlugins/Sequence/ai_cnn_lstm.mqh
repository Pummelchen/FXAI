#ifndef __FXAI_AI_CNN_LSTM_MQH__
#define __FXAI_AI_CNN_LSTM_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAICNNLSTM : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_CNN_LSTM; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_RECURRENT; }

public:
   virtual int AIId(void) const { return (int)AI_CNN_LSTM; }
   virtual string AIName(void) const { return "ai_cnn_lstm"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_AI_CNN_LSTM_MQH__
