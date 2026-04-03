#ifndef __FXAI_CORE_RUNTIME_PERF_MQH__
#define __FXAI_CORE_RUNTIME_PERF_MQH__

enum ENUM_FXAI_RUNTIME_STAGE
{
   FXAI_RUNTIME_STAGE_TOTAL = 0,
   FXAI_RUNTIME_STAGE_FEATURE_PIPELINE,
   FXAI_RUNTIME_STAGE_TRANSFER,
   FXAI_RUNTIME_STAGE_ROUTER,
   FXAI_RUNTIME_STAGE_POLICY,
   FXAI_RUNTIME_STAGE_SHADOW,
   FXAI_RUNTIME_STAGE_CONTROL,
   FXAI_RUNTIME_STAGE_COUNT
};

bool   g_runtime_perf_ready = false;
double g_runtime_stage_mean_ms[FXAI_RUNTIME_STAGE_COUNT];
double g_runtime_stage_max_ms[FXAI_RUNTIME_STAGE_COUNT];
int    g_runtime_stage_obs[FXAI_RUNTIME_STAGE_COUNT];
double g_runtime_plugin_predict_mean_ms[FXAI_AI_COUNT];
double g_runtime_plugin_predict_max_ms[FXAI_AI_COUNT];
int    g_runtime_plugin_predict_obs[FXAI_AI_COUNT];
double g_runtime_plugin_update_mean_ms[FXAI_AI_COUNT];
double g_runtime_plugin_update_max_ms[FXAI_AI_COUNT];
int    g_runtime_plugin_update_obs[FXAI_AI_COUNT];
double g_runtime_plugin_working_set_kb[FXAI_AI_COUNT];
int    g_runtime_last_active_models = 0;
datetime g_runtime_perf_last_time = 0;

string FXAI_RuntimeStageName(const int stage_id)
{
   switch(stage_id)
   {
      case FXAI_RUNTIME_STAGE_TOTAL: return "total";
      case FXAI_RUNTIME_STAGE_FEATURE_PIPELINE: return "feature_pipeline";
      case FXAI_RUNTIME_STAGE_TRANSFER: return "transfer";
      case FXAI_RUNTIME_STAGE_ROUTER: return "router";
      case FXAI_RUNTIME_STAGE_POLICY: return "policy";
      case FXAI_RUNTIME_STAGE_SHADOW: return "shadow";
      case FXAI_RUNTIME_STAGE_CONTROL: return "control_plane";
      default: return "unknown";
   }
}

void FXAI_ResetRuntimePerformanceState(void)
{
   g_runtime_perf_ready = false;
   g_runtime_last_active_models = 0;
   g_runtime_perf_last_time = 0;
   for(int i=0; i<FXAI_RUNTIME_STAGE_COUNT; i++)
   {
      g_runtime_stage_mean_ms[i] = 0.0;
      g_runtime_stage_max_ms[i] = 0.0;
      g_runtime_stage_obs[i] = 0;
   }
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_runtime_plugin_predict_mean_ms[ai] = 0.0;
      g_runtime_plugin_predict_max_ms[ai] = 0.0;
      g_runtime_plugin_predict_obs[ai] = 0;
      g_runtime_plugin_update_mean_ms[ai] = 0.0;
      g_runtime_plugin_update_max_ms[ai] = 0.0;
      g_runtime_plugin_update_obs[ai] = 0;
      g_runtime_plugin_working_set_kb[ai] = 0.0;
   }
}

double FXAI_RuntimePerfBlend(const double prev,
                             const double value,
                             const int obs)
{
   double alpha = (obs <= 0 ? 1.0 : 0.12);
   return (obs <= 0 ? value : ((1.0 - alpha) * prev + alpha * value));
}

void FXAI_RecordRuntimeStageMs(const int stage_id,
                               const double elapsed_ms)
{
   if(stage_id < 0 || stage_id >= FXAI_RUNTIME_STAGE_COUNT)
      return;
   double ms = MathMax(elapsed_ms, 0.0);
   int obs = g_runtime_stage_obs[stage_id];
   g_runtime_stage_mean_ms[stage_id] = FXAI_RuntimePerfBlend(g_runtime_stage_mean_ms[stage_id], ms, obs);
   if(ms > g_runtime_stage_max_ms[stage_id])
      g_runtime_stage_max_ms[stage_id] = ms;
   g_runtime_stage_obs[stage_id] = obs + 1;
   g_runtime_perf_ready = true;
   g_runtime_perf_last_time = TimeCurrent();
}

void FXAI_RecordPluginPredictMs(const int ai_id,
                                const double elapsed_ms)
{
   if(ai_id < 0 || ai_id >= FXAI_AI_COUNT)
      return;
   double ms = MathMax(elapsed_ms, 0.0);
   int obs = g_runtime_plugin_predict_obs[ai_id];
   g_runtime_plugin_predict_mean_ms[ai_id] = FXAI_RuntimePerfBlend(g_runtime_plugin_predict_mean_ms[ai_id], ms, obs);
   if(ms > g_runtime_plugin_predict_max_ms[ai_id])
      g_runtime_plugin_predict_max_ms[ai_id] = ms;
   g_runtime_plugin_predict_obs[ai_id] = obs + 1;
   g_runtime_perf_ready = true;
   g_runtime_perf_last_time = TimeCurrent();
}

void FXAI_RecordPluginUpdateMs(const int ai_id,
                               const double elapsed_ms)
{
   if(ai_id < 0 || ai_id >= FXAI_AI_COUNT)
      return;
   double ms = MathMax(elapsed_ms, 0.0);
   int obs = g_runtime_plugin_update_obs[ai_id];
   g_runtime_plugin_update_mean_ms[ai_id] = FXAI_RuntimePerfBlend(g_runtime_plugin_update_mean_ms[ai_id], ms, obs);
   if(ms > g_runtime_plugin_update_max_ms[ai_id])
      g_runtime_plugin_update_max_ms[ai_id] = ms;
   g_runtime_plugin_update_obs[ai_id] = obs + 1;
   g_runtime_perf_ready = true;
   g_runtime_perf_last_time = TimeCurrent();
}

void FXAI_SetPluginWorkingSetKB(const int ai_id,
                                const double working_set_kb)
{
   if(ai_id < 0 || ai_id >= FXAI_AI_COUNT)
      return;
   double kb = MathMax(working_set_kb, 0.0);
   if(kb > g_runtime_plugin_working_set_kb[ai_id])
      g_runtime_plugin_working_set_kb[ai_id] = kb;
}

double FXAI_EstimatePluginWorkingSetKB(const FXAIAIManifestV4 &manifest,
                                       const int sequence_bars)
{
   int seq = sequence_bars;
   if(seq < 1)
      seq = 1;
   if(seq > FXAI_MAX_SEQUENCE_BARS)
      seq = FXAI_MAX_SEQUENCE_BARS;
   double payload_bytes = (double)FXAI_AI_WEIGHTS * 8.0;
   double context_bytes = payload_bytes * (double)seq;
   double state_mult = 1.0;
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_STATEFUL))
      state_mult += 0.75;
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_WINDOW_CONTEXT))
      state_mult += 0.35;
   if(FXAI_HasCapability(manifest.capability_mask, FXAI_CAP_MULTI_HORIZON))
      state_mult += 0.20;
   return MathMax((payload_bytes + context_bytes) * state_mult / 1024.0, 1.0);
}

double FXAI_RuntimeBudgetPressure(const double budget_ms)
{
   double budget = MathMax(budget_ms, 0.0);
   if(budget <= 0.0 || g_runtime_stage_obs[FXAI_RUNTIME_STAGE_TOTAL] <= 0)
      return 0.0;
   return FXAI_Clamp((g_runtime_stage_mean_ms[FXAI_RUNTIME_STAGE_TOTAL] - budget) /
                     MathMax(budget, 1e-6),
                     0.0,
                     2.0);
}

#endif // __FXAI_CORE_RUNTIME_PERF_MQH__
