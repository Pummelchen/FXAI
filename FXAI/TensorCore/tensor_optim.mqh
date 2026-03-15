#ifndef __FXAI_NN_OPTIM_MQH__
#define __FXAI_NN_OPTIM_MQH__

#include "..\Engine\core.mqh"

struct FXAIBlockBatchPlan
{
   int start;
   int end;
   int replay_budget;
   double lr_scale;
};

bool FXAI_OptimizerFiniteCandidate(const double v)
{
   return MathIsValidNumber(v) && MathAbs(v) < 1e12;
}

void FXAI_OptimizerCheckpointAssign(double &param,
                                    const double candidate,
                                    const double fallback = 0.0)
{
   if(FXAI_OptimizerFiniteCandidate(candidate))
      param = candidate;
   else if(FXAI_OptimizerFiniteCandidate(param))
      return;
   else
      param = fallback;
}

double FXAI_LRScheduleStepDecay(const double base_lr,
                                const int step,
                                const int decay_steps,
                                const double decay_rate,
                                const double floor_lr = 1e-6)
{
   double lr0 = MathMax(base_lr, 0.0);
   int ds = MathMax(decay_steps, 1);
   double rate = FXAI_Clamp(decay_rate, 0.01, 1.0);
   double level = MathFloor((double)MathMax(step, 0) / (double)ds);
   double lr = lr0 * MathPow(rate, level);
   return MathMax(lr, floor_lr);
}

double FXAI_LRScheduleCosineWarm(const double base_lr,
                                 const int step,
                                 const int warmup_steps,
                                 const int cycle_steps,
                                 const double floor_ratio = 0.10)
{
   double lr0 = MathMax(base_lr, 0.0);
   int st = MathMax(step, 1);
   int warm = MathMax(warmup_steps, 1);
   int cyc = MathMax(cycle_steps, warm + 1);
   double warm_scale = FXAI_Clamp((double)st / (double)warm, 0.05, 1.0);
   double phase = (double)(st % cyc) / (double)cyc;
   double floor = FXAI_Clamp(floor_ratio, 0.0, 0.95);
   double cosine = floor + (1.0 - floor) * 0.5 * (1.0 + MathCos(3.141592653589793 * phase));
   return lr0 * warm_scale * cosine;
}

double FXAI_LRScheduleInvSqrt(const double base_lr,
                              const int step,
                              const int warmup_steps,
                              const double decay_factor = 0.001)
{
   double lr0 = MathMax(base_lr, 0.0);
   int st = MathMax(step, 1);
   int warm = MathMax(warmup_steps, 1);
   double warm_scale = FXAI_Clamp((double)st / (double)warm, 0.05, 1.0);
   double inv = 1.0 / MathSqrt(1.0 + MathMax(0.0, (double)st - (double)warm) * MathMax(decay_factor, 1e-6));
   return lr0 * warm_scale * inv;
}

double FXAI_VectorNorm(const double &v[],
                       const int n)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   if(use_n > ArraySize(v)) use_n = ArraySize(v);
   double s = 0.0;
   for(int i=0; i<use_n; i++)
      s += v[i] * v[i];
   return MathSqrt(s);
}

double FXAI_GradientScaleFromNorm(const double grad_norm,
                                  const double clip_norm)
{
   double clip = MathMax(clip_norm, 1e-9);
   if(grad_norm <= clip || grad_norm <= 1e-12)
      return 1.0;
   return clip / grad_norm;
}

void FXAI_ClipVectorInPlace(double &v[],
                            const int n,
                            const double clip_norm)
{
   int use_n = n;
   if(use_n < 0) use_n = 0;
   if(use_n > ArraySize(v)) use_n = ArraySize(v);
   if(use_n <= 0) return;
   double norm = FXAI_VectorNorm(v, use_n);
   double scale = FXAI_GradientScaleFromNorm(norm, clip_norm);
   if(scale >= 0.999999)
      return;
   for(int i=0; i<use_n; i++)
      v[i] *= scale;
}

void FXAI_OptSGDStep(double &param,
                     double &velocity,
                     const double grad,
                     const double lr,
                     const double momentum = 0.0,
                     const double wd = 0.0)
{
   double mom = FXAI_Clamp(momentum, 0.0, 0.9999);
   double g = FXAI_ClipSym(grad, 32.0);
   if(wd > 0.0)
      g += wd * param;
   velocity = mom * velocity + g;
   double candidate = param - MathMax(lr, 0.0) * velocity;
   FXAI_OptimizerCheckpointAssign(param, candidate);
}

void FXAI_OptAdamWStep(double &param,
                       double &m,
                       double &v,
                       const double grad,
                       const double lr,
                       const double beta1,
                       const double beta2,
                       const double wd,
                       const int tstep)
{
   double b1 = FXAI_Clamp(beta1, 0.0, 0.9999);
   double b2 = FXAI_Clamp(beta2, 0.0, 0.999999);
   double g = FXAI_ClipSym(grad, 32.0);
   m = b1 * m + (1.0 - b1) * g;
   v = b2 * v + (1.0 - b2) * g * g;

   double t = (double)MathMax(tstep, 1);
   double mhat = m / MathMax(1.0 - MathPow(b1, t), 1e-12);
   double vhat = v / MathMax(1.0 - MathPow(b2, t), 1e-12);
   double upd = mhat / (MathSqrt(vhat) + 1e-8);
   double candidate = param - MathMax(lr, 0.0) * (upd + MathMax(wd, 0.0) * param);
   FXAI_OptimizerCheckpointAssign(param, candidate);
}

void FXAI_OptRMSPropStep(double &param,
                         double &cache,
                         const double grad,
                         const double lr,
                         const double decay,
                         const double wd = 0.0)
{
   double rho = FXAI_Clamp(decay, 0.0, 0.9999);
   double g = FXAI_ClipSym(grad, 32.0);
   if(wd > 0.0)
      g += wd * param;
   cache = rho * cache + (1.0 - rho) * g * g;
   double upd = g / (MathSqrt(MathMax(cache, 0.0)) + 1e-8);
   double candidate = param - MathMax(lr, 0.0) * upd;
   FXAI_OptimizerCheckpointAssign(param, candidate);
}

FXAIBlockBatchPlan FXAI_MakeBlockBatchPlan(const int start_idx,
                                           const int current_end,
                                           const int block_span,
                                           const bool replay_enabled,
                                           const double lr_scale = 1.0)
{
   FXAIBlockBatchPlan plan;
   int span = MathMax(block_span, 1);
   plan.start = MathMax(start_idx, current_end - span + 1);
   plan.end = current_end;
   plan.replay_budget = (replay_enabled ? MathMin(2, span) : 0);
   plan.lr_scale = FXAI_Clamp(lr_scale, 0.10, 4.0);
   return plan;
}

#endif // __FXAI_NN_OPTIM_MQH__
