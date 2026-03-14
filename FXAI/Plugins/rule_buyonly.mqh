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

   virtual bool PredictDistributionCore(const double &x[],
                                        const FXAIAIHyperParams &hp,
                                        FXAIAIModelOutputV4 &out)
   {
      ResetModelOutput(out);
      if(!PredictModelCore(x, hp, out.class_probs, out.move_mean_points))
         return false;
      double sigma = MathMax(0.10, 0.30 * out.move_mean_points);
      out.move_q25_points = MathMax(0.0, out.move_mean_points - 0.50 * sigma);
      out.move_q50_points = MathMax(out.move_q25_points, out.move_mean_points);
      out.move_q75_points = MathMax(out.move_q50_points, out.move_mean_points + 0.50 * sigma);
      out.confidence = out.class_probs[(int)FXAI_LABEL_BUY];
      out.reliability = 0.55;
      out.has_quantiles = true;
      out.has_confidence = true;
      PopulatePathQualityHeads(out, x, FXAI_Clamp(1.0 - out.class_probs[(int)FXAI_LABEL_SKIP], 0.0, 1.0), out.reliability, out.confidence);
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
