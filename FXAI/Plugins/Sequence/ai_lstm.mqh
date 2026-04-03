#ifndef __FXAI_AI_LSTM_MQH__
#define __FXAI_AI_LSTM_MQH__

#include "..\..\API\plugin_base.mqh"

#define FXAI_LSTM_TBPTT 32
#define FXAI_LSTM_CLASS_COUNT 3
#define FXAI_LSTM_CAL_BINS 12
#define FXAI_LSTM_DROP_RATE 0.08
#define FXAI_LSTM_ZONEOUT 0.05
#define FXAI_LSTM_LN_EPS 0.00001
#define FXAI_LSTM_REPLAY 512
#define FXAI_LSTM_ECE_BINS 12

class CFXAIAILSTM : public CFXAIAIPlugin
{
private:
#include "ai_lstm\ai_lstm_state.mqh"

public:
#include "ai_lstm\ai_lstm_public.mqh"

};


#endif // __FXAI_AI_LSTM_MQH__
