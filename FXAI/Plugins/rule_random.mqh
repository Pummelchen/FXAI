#ifndef __FXAI_RULE_RANDOM_MQH__
#define __FXAI_RULE_RANDOM_MQH__

#include "..\API\plugin_base.mqh"

class CFXAIAIRuleRandom : public CFXAIAIPlugin
{
private:
   uint BuildDeterministicHash(const double &x[]) const
   {
      datetime t = ResolveContextTime();
      if(t < 0) t = -t;
      double acc = (double)t;

      int n = ArraySize(x);
      int lim = (n < 8 ? n : 8);
      for(int i=0; i<lim; i++)
      {
         acc = acc * 1.61803398875 + MathAbs(x[i]) * 1000.0 + 17.0 * (i + 1);
         if(acc > 2147483000.0)
            acc -= 2147483000.0 * MathFloor(acc / 2147483000.0);
      }
      long h = (long)MathRound(acc);
      if(h < 0) h = -h;
      return (uint)h;
   }

public:
   CFXAIAIRuleRandom(void) : CFXAIAIPlugin()
   {
      Reset();
   }

   virtual int AIId(void) const { return (int)AI_RANDOM_NOSKIP; }
   virtual string AIName(void) const { return "rule_random"; }

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
      uint h = BuildDeterministicHash(x);
      bool buy_side = ((h % 2) == 0);

      class_probs[(int)FXAI_LABEL_SELL] = (buy_side ? 0.005 : 0.995);
      class_probs[(int)FXAI_LABEL_BUY]  = (buy_side ? 0.995 : 0.005);
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

#endif // __FXAI_RULE_RANDOM_MQH__
