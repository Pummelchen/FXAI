#ifndef __FXAI_AI_LSTM_TCN_MQH__
#define __FXAI_AI_LSTM_TCN_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAILSTMTCN : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_LSTM_TCN; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_RECURRENT; }

public:
   virtual int AIId(void) const { return (int)AI_LSTM_TCN; }
   virtual string AIName(void) const { return "ai_lstm_tcn"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_AI_LSTM_TCN_MQH__
