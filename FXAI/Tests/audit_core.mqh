#ifndef __FXAI_AUDIT_CORE_MQH__
#define __FXAI_AUDIT_CORE_MQH__

#include "..\Engine\core.mqh"
#include "..\Engine\data_pipeline.mqh"
#include "..\API\api.mqh"

#include "audit_defs.mqh"
#include "audit_utils.mqh"
#include "audit_scenarios.mqh"
#include "audit_samples.mqh"
#include "audit_scoring.mqh"
#include "audit_report.mqh"
#include "audit_tensor.mqh"

// Audit Lab exercises plugins in isolation and does not persist live runtime artifacts.
void FXAI_ApplyConformalPredictionAdjustment(const int ai_idx,
                                             const int regime_id,
                                             const int horizon_minutes,
                                             const double min_move_points,
                                             FXAIAIPredictionV4 &pred)
{
}

void FXAI_ResetConformalState(void)
{
}

void FXAI_EnqueueConformalPending(const int ai_idx,
                                  const int signal_seq,
                                  const int regime_id,
                                  const int horizon_minutes,
                                  const FXAIAIPredictionV4 &pred)
{
}

void FXAI_UpdateConformalFromPending(const int ai_idx,
                                     const long signal_seq,
                                     const int regime_id,
                                     const int horizon_minutes,
                                     const int label_class,
                                     const double realized_move_points,
                                     const double mfe_points,
                                     const double mae_points,
                                     const double time_to_hit_frac,
                                     const int path_flags,
                                     const double spread_stress,
                                     const double commission_points,
                                     const double cost_buffer_points,
                                     const double min_move_points)
{
}

void FXAI_MarkRuntimeArtifactsDirty(void)
{
}

bool FXAI_SaveRuntimeArtifacts(const string symbol)
{
   return true;
}

bool FXAI_LoadRuntimeArtifacts(const string symbol)
{
   return true;
}

void FXAI_MaybeSaveRuntimeArtifacts(const string symbol,
                                    const datetime bar_time)
{
}

#endif // __FXAI_AUDIT_CORE_MQH__
