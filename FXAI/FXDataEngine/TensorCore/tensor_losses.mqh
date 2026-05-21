#ifndef __FXAI_NN_LOSSES_MQH__
#define __FXAI_NN_LOSSES_MQH__

#include "..\Engine\core.mqh"

double FXAI_LossMSE(const double pred,
                    const double target)
{
   double d = pred - target;
   return d * d;
}

double FXAI_LossMSEGrad(const double pred,
                        const double target)
{
   return 2.0 * (pred - target);
}

double FXAI_LossVectorMSE(const double &pred[],
                          const double &target[],
                          const int n)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   int pn = ArraySize(pred);
   int tn = ArraySize(target);
   if(use_n > pn) use_n = pn;
   if(use_n > tn) use_n = tn;
   if(use_n <= 0)
      return 0.0;

   double acc = 0.0;
   for(int i=0; i<use_n; i++)
   {
      double d = pred[i] - target[i];
      acc += d * d;
   }
   return acc / (double)use_n;
}

double FXAI_LossHuber(const double pred,
                      const double target,
                      const double delta)
{
   double d = pred - target;
   double a = MathAbs(d);
   double k = (delta > 1e-9 ? delta : 1.0);
   if(a <= k)
      return 0.5 * d * d;
   return k * (a - 0.5 * k);
}

double FXAI_LossHuberGrad(const double pred,
                          const double target,
                          const double delta)
{
   double d = pred - target;
   double k = (delta > 1e-9 ? delta : 1.0);
   if(d > k) return k;
   if(d < -k) return -k;
   return d;
}

double FXAI_LossCrossEntropy3(const double &probs[],
                              const int cls)
{
   if(ArraySize(probs) < 3)
      return 0.0;
   int y = cls;
   if(y < 0) y = 0;
   if(y > 2) y = 2;
   return -MathLog(FXAI_Clamp(probs[y], 1e-9, 1.0));
}

void FXAI_LossCrossEntropyGrad3(const double &probs[],
                                const int cls,
                                double &grad[])
{
   ArrayResize(grad, 3);
   if(ArraySize(probs) < 3)
   {
      for(int i=0; i<3; i++) grad[i] = 0.0;
      return;
   }

   int y = cls;
   if(y < 0) y = 0;
   if(y > 2) y = 2;
   for(int i=0; i<3; i++)
      grad[i] = probs[i] - (i == y ? 1.0 : 0.0);
}

#endif // __FXAI_NN_LOSSES_MQH__
