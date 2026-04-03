#ifndef __FXAI_RUNTIME_SIGNAL_FINALIZE_MQH__
#define __FXAI_RUNTIME_SIGNAL_FINALIZE_MQH__

int FXAI_RuntimeFinalizeDecision(const string symbol,
                                 const int decision,
                                 const datetime signal_bar,
                                 const int decision_key,
                                 const bool ensemble_mode,
                                 const int ai_type,
                                 const string single_no_trade_reason,
                                 const double ensemble_meta_total,
                                 const double macro_profile_shortfall,
                                 const double regime_transition_penalty,
                                 const ulong runtime_total_t0)
{
   g_ai_last_signal_bar = signal_bar;
   g_ai_last_signal_key = decision_key;
   g_ai_last_signal = decision;
   if(decision == 1)
      g_ai_last_reason = "buy";
   else if(decision == 0)
      g_ai_last_reason = "sell";
   else if(!ensemble_mode && ai_type == (int)AI_M1SYNC && StringLen(single_no_trade_reason) > 0)
      g_ai_last_reason = single_no_trade_reason;
   else if(ensemble_mode && ensemble_meta_total <= 0.0)
      g_ai_last_reason = "no_meta_weight";
   else
      g_ai_last_reason = "no_consensus_or_ev";

   double signal_intensity = FXAI_Clamp((0.55 * g_ai_last_trade_gate +
                                         0.25 * g_policy_last_trade_prob +
                                         0.20 * g_policy_last_confidence) *
                                        FXAI_Clamp(g_policy_last_size_mult, 0.25, 1.60) *
                                        FXAI_Clamp(1.0 - 0.35 * macro_profile_shortfall -
                                                   0.20 * regime_transition_penalty,
                                                   0.20,
                                                   1.0),
                                        0.0,
                                        4.0);
   if(decision < 0)
      signal_intensity = 0.0;

   ulong control_stage_t0 = GetMicrosecondCount();
   FXAI_WriteControlPlaneLocalSnapshot(symbol, decision, signal_intensity);
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_CONTROL,
                             (double)(GetMicrosecondCount() - control_stage_t0) / 1000.0);
   FXAI_RecordRuntimeStageMs(FXAI_RUNTIME_STAGE_TOTAL,
                             (double)(GetMicrosecondCount() - runtime_total_t0) / 1000.0);
   return decision;
}

#endif // __FXAI_RUNTIME_SIGNAL_FINALIZE_MQH__
