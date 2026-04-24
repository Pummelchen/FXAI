#ifndef __FXAI_RL_PPO_MQH__
#define __FXAI_RL_PPO_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIRLPPO : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_PPO; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_OTHER; }

public:
   virtual int AIId(void) const { return (int)AI_RL_PPO; }
   virtual string AIName(void) const { return "rl_ppo"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_RL_PPO_MQH__
