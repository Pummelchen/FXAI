#ifndef __FXAI_RUNTIME_ROUTER_STAGE_MQH__
#define __FXAI_RUNTIME_ROUTER_STAGE_MQH__

void FXAI_RuntimeApplyPerformanceModelCap(int &active_ai_ids[],
                                          const FXAILiveDeploymentProfile &deploy_profile)
{
   int runtime_model_cap = deploy_profile.max_runtime_models;
   if(runtime_model_cap <= 0)
      runtime_model_cap = FXAI_AI_COUNT;

   double perf_budget_pressure = FXAI_RuntimeBudgetPressure(deploy_profile.performance_budget_ms);
   if(perf_budget_pressure > 0.05 && runtime_model_cap > 1)
   {
      double pressure_scale = FXAI_Clamp(1.0 - 0.55 * MathMin(perf_budget_pressure, 1.0),
                                         0.25,
                                         1.0);
      runtime_model_cap = (int)MathMax(1.0,
                                       MathFloor((double)runtime_model_cap * pressure_scale));
   }

   if(ArraySize(active_ai_ids) > runtime_model_cap)
      ArrayResize(active_ai_ids, runtime_model_cap);
   g_runtime_last_active_models = ArraySize(active_ai_ids);
}

#endif // __FXAI_RUNTIME_ROUTER_STAGE_MQH__
