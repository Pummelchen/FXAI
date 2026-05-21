#ifndef __FXAI_FACTOR_PCA_PANEL_MQH__
#define __FXAI_FACTOR_PCA_PANEL_MQH__

#include "..\Common\fxai_framework_model.mqh"

class CFXAIAIFactorPCAPanel : public CFXAIFrameworkModelPlugin
{
protected:
   virtual int FrameworkKind(void) const { return FXAI_FW_KIND_PCA_PANEL; }
   virtual int FrameworkFamily(void) const { return (int)FXAI_FAMILY_STATE_SPACE; }

public:
   virtual int AIId(void) const { return (int)AI_FACTOR_PCA_PANEL; }
   virtual string AIName(void) const { return "factor_pca_panel"; }
   virtual void Describe(FXAIAIManifestV4 &out) const { CFXAIFrameworkModelPlugin::Describe(out); }
};

#endif // __FXAI_FACTOR_PCA_PANEL_MQH__
