#ifndef __FXAI_RULE_BUYONLY_MQH__
#define __FXAI_RULE_BUYONLY_MQH__

#include "..\API\plugin_base.mqh"

class CFXAIAIRuleBuyOnly : public CFXAIAIPlugin
{
public:
   CFXAIAIRuleBuyOnly(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return (int)AI_BUY_ONLY; }
   virtual string AIName(void) const { return "rule_buyonly"; }

   virtual void Describe(FXAIAIManifestV4 &out) const
   {
      const ulong caps = (ulong)(FXAI_CAP_MULTI_HORIZON | FXAI_CAP_SELF_TEST);
      FillManifest(out, (int)FXAI_FAMILY_RULE_BASED, caps, 1, 1);
   }

   virtual void Reset(void)
   {
      CFXAIAIPlugin::Reset();
   }

   virtual void EnsureInitialized(const FXAIAIHyperParams &hp) {}

protected:
   virtual bool PredictModelCore(const double &x[],
                                 const FXAIAIHyperParams &hp,
                                 double &class_probs[],
                                 double &expected_move_points)
   {
      class_probs[(int)FXAI_LABEL_SELL] = 0.001;
      class_probs[(int)FXAI_LABEL_BUY]  = 0.999;
      class_probs[(int)FXAI_LABEL_SKIP] = 0.0;

      double mm = ResolveMinMovePoints();
      if(mm <= 0.0) mm = MathMax(ResolveCostPoints(x), 0.10);
      expected_move_points = MathMax(1.0, 3.0 * mm + 0.25);
      return true;
   }

   virtual void TrainModelCore(const int y,
                               const double &x[],
                               const FXAIAIHyperParams &hp,
                               const double move_points)
   {
   }
};

#endif // __FXAI_RULE_BUYONLY_MQH__
