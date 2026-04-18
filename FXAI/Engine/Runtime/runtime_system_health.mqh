#ifndef __FXAI_RUNTIME_SYSTEM_HEALTH_MQH__
#define __FXAI_RUNTIME_SYSTEM_HEALTH_MQH__

struct FXAISystemHealthState
{
   bool     ready;
   datetime generated_at;
   double   health_score;
   int      degraded_count;
   bool     news_ready;
   bool     news_stale;
   bool     rates_ready;
   bool     rates_stale;
   bool     cross_ready;
   bool     cross_stale;
   bool     micro_ready;
   bool     micro_stale;
   bool     exec_ready;
   bool     exec_stale;
   bool     calendar_ready;
   bool     calendar_stale;
   bool     factor_ready;
   bool     factor_stale;
   string   posture;
   string   reasons_csv;
};

FXAISystemHealthState g_system_health_last_state;
bool                  g_system_health_last_ready = false;

void FXAI_ResetSystemHealthState(FXAISystemHealthState &out)
{
   out.ready = false;
   out.generated_at = 0;
   out.health_score = 0.0;
   out.degraded_count = 0;
   out.news_ready = false;
   out.news_stale = true;
   out.rates_ready = false;
   out.rates_stale = true;
   out.cross_ready = false;
   out.cross_stale = true;
   out.micro_ready = false;
   out.micro_stale = true;
   out.exec_ready = false;
   out.exec_stale = true;
   out.calendar_ready = false;
   out.calendar_stale = true;
   out.factor_ready = false;
   out.factor_stale = true;
   out.posture = "UNKNOWN";
   out.reasons_csv = "";
}

void FXAI_SystemHealthAppendReason(string &csv,
                                   const string reason)
{
   if(StringLen(reason) <= 0)
      return;
   if(StringFind(csv, reason) >= 0)
      return;
   if(StringLen(csv) > 0)
      csv += "; ";
   csv += reason;
}

bool FXAI_RefreshSystemHealth(const string symbol,
                              FXAISystemHealthState &out)
{
   FXAI_ResetSystemHealthState(out);
   out.generated_at = FXAI_ServerNow();

   double score_sum = 0.0;
   double weight_sum = 0.0;

   FXAINewsPulsePairState news_state;
   if(FXAI_ReadNewsPulsePairState(symbol, news_state))
   {
      out.news_ready = news_state.ready;
      out.news_stale = news_state.stale;
      score_sum += (news_state.ready ? (news_state.stale ? 0.45 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(news_state.stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "newspulse_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "newspulse_unavailable");
   }

   FXAIRatesEnginePairState rates_state;
   FXAI_ResetRatesEnginePairState(rates_state);
   if(FXAI_ReadRatesEnginePairState(symbol, rates_state))
   {
      out.rates_ready = rates_state.ready;
      out.rates_stale = rates_state.stale;
      score_sum += (rates_state.ready ? (rates_state.stale ? 0.45 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(rates_state.stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "rates_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "rates_unavailable");
   }

   FXAICrossAssetPairState cross_state;
   FXAI_ResetCrossAssetPairState(cross_state);
   if(FXAI_ReadCrossAssetPairState(symbol, cross_state))
   {
      out.cross_ready = cross_state.ready;
      out.cross_stale = cross_state.stale;
      score_sum += (cross_state.ready ? (cross_state.stale ? 0.45 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(cross_state.stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "cross_asset_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "cross_asset_unavailable");
   }

   FXAIMicrostructurePairState micro_state;
   FXAI_ResetMicrostructurePairState(micro_state);
   if(FXAI_ReadMicrostructurePairState(symbol, micro_state))
   {
      out.micro_ready = micro_state.ready;
      out.micro_stale = micro_state.stale;
      score_sum += (micro_state.ready ? (micro_state.stale ? 0.40 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(micro_state.stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "microstructure_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "microstructure_unavailable");
   }

   FXAIExecutionQualityPairState exec_state;
   FXAI_ResetExecutionQualityPairState(exec_state);
   if(FXAI_ReadExecutionQualityPairState(symbol, exec_state))
   {
      out.exec_ready = exec_state.ready;
      out.exec_stale = (exec_state.stale || exec_state.data_stale);
      score_sum += (exec_state.ready ? (out.exec_stale ? 0.45 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(out.exec_stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "execution_quality_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "execution_quality_unavailable");
   }

   FXAICalendarCachePairState calendar_state;
   if(FXAI_ReadCalendarCachePairState(symbol, calendar_state))
   {
      out.calendar_ready = calendar_state.ready;
      out.calendar_stale = calendar_state.stale;
      score_sum += (calendar_state.ready ? (calendar_state.stale ? 0.45 : 1.0) : 0.0);
      weight_sum += 1.0;
      if(calendar_state.stale)
      {
         out.degraded_count++;
         FXAI_SystemHealthAppendReason(out.reasons_csv, "calendar_cache_stale");
      }
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "calendar_cache_unavailable");
   }

   FXAIPairFactorContext factor_state;
   if(FXAI_RefreshFactorContext(symbol, factor_state))
   {
      out.factor_ready = factor_state.ready;
      out.factor_stale = factor_state.stale;
      score_sum += (factor_state.ready ? 1.0 : 0.0);
      weight_sum += 1.0;
   }
   else
   {
      out.degraded_count++;
      weight_sum += 1.0;
      FXAI_SystemHealthAppendReason(out.reasons_csv, "factor_context_unavailable");
   }

   out.health_score = (weight_sum > 1e-9 ? score_sum / weight_sum : 0.0);
   if(out.health_score >= 0.85 && out.degraded_count <= 1)
      out.posture = "HEALTHY";
   else if(out.health_score >= 0.60)
      out.posture = "CAUTION";
   else
      out.posture = "DEGRADED";

   out.ready = true;
   g_system_health_last_state = out;
   g_system_health_last_ready = true;
   return true;
}

bool FXAI_CheckRuntimeInvariants(const string symbol,
                                 string &reason)
{
   reason = "ok";

   int symbol_total = FXAI_ManagedOrdersTotal(symbol) + FXAI_ManagedPositionsTotal(symbol);
   if(symbol_total <= 0)
   {
      if(CycleActive)
      {
         ResetCycleState();
         reason = "cycle_reset_no_exposure";
      }
      if(g_last_order_request_pending)
      {
         FXAI_ClearLastOrderRequestState();
         if(reason == "ok")
            reason = "cleared_stale_order_request";
      }
      return true;
   }

   if(!CycleActive)
   {
      FXAI_RecoverManagedCycleState(symbol);
      reason = "cycle_recovered";
   }

   if(g_last_order_request_pending &&
      g_last_order_request_uses_pending_order &&
      StringLen(g_last_order_request_symbol) > 0 &&
      FXAI_ManagedOrdersTotal(g_last_order_request_symbol) <= 0)
   {
      FXAI_ClearLastOrderRequestState();
      if(reason == "ok")
         reason = "cleared_completed_pending_request";
   }

   if(g_last_order_request_pending && g_last_order_request_time > 0)
   {
      datetime now_time = FXAI_ServerNow();
      if(now_time > g_last_order_request_time && (now_time - g_last_order_request_time) > 7200)
      {
         FXAI_ClearLastOrderRequestState();
         reason = "expired_order_request_state";
      }
   }

   return true;
}

bool FXAI_RunRuntimeMaintenance(const string symbol,
                                const bool emit_debug = false)
{
   FXAI_RefreshTimeContext();

   string invariant_reason = "ok";
   FXAI_CheckRuntimeInvariants(symbol, invariant_reason);

   FXAISystemHealthState health;
   FXAI_RefreshSystemHealth(symbol, health);

   if(emit_debug && StringLen(invariant_reason) > 0 && invariant_reason != "ok")
      Print("FXAI debug: runtime maintenance action. reason=", invariant_reason);
   return health.ready;
}

#endif // __FXAI_RUNTIME_SYSTEM_HEALTH_MQH__
