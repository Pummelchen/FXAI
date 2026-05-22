#ifndef __FXAI_TREE_RF_MQH__
#define __FXAI_TREE_RF_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAITreeRF : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_RANDOM_FOREST; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_TREE; }

public:
   virtual int AIId(void) const { return (int)AI_TREE_RF; }
   virtual string AIName(void) const { return "tree_rf"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_TREE_RF_MQH__
