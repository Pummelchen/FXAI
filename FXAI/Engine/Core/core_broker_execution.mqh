int FXAI_BrokerExecutionHorizonBucket(const int horizon_minutes)
{
   int h = horizon_minutes;
   if(h < 1) h = 1;
   if(h <= 2) return 0;
   if(h <= 5) return 1;
   if(h <= 15) return 2;
   if(h <= 30) return 3;
   if(h <= 60) return 4;
   if(h <= 240) return 5;
   if(h <= 720) return 6;
   return FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
}

int FXAI_BrokerExecutionSymbolBucket(const string raw_symbol)
{
   string symbol = raw_symbol;
   if(StringLen(symbol) <= 0)
      symbol = _Symbol;
   double h = FXAI_SymbolHash01(symbol);
   int bucket = (int)MathFloor(FXAI_Clamp(h, 0.0, 1.0 - 1e-9) * (double)FXAI_BROKER_EXEC_SYMBOL_BUCKETS);
   if(bucket < 0) bucket = 0;
   if(bucket >= FXAI_BROKER_EXEC_SYMBOL_BUCKETS) bucket = FXAI_BROKER_EXEC_SYMBOL_BUCKETS - 1;
   return bucket;
}

int FXAI_NormalizeBrokerExecutionSide(const int order_side)
{
   if(order_side > 0) return 1;
   if(order_side < 0) return -1;
   return 0;
}

int FXAI_BrokerExecutionSideIndex(const int order_side)
{
   int side = FXAI_NormalizeBrokerExecutionSide(order_side);
   if(side < 0) return 0;
   if(side > 0) return 2;
   return 1;
}

int FXAI_NormalizeBrokerExecutionOrderType(const int order_type_bucket)
{
   int t = order_type_bucket;
   if(t < 0) t = 0;
   if(t > 4) t = 4;
   return t;
}

int FXAI_BrokerExecutionEventKindIndex(const int event_kind)
{
   int idx = event_kind;
   if(idx < 0) idx = 0;
   if(idx >= FXAI_BROKER_EXEC_EVENT_KIND_COUNT)
      idx = FXAI_BROKER_EXEC_EVENT_KIND_COUNT - 1;
   return idx;
}

int FXAI_BrokerExecutionLibraryIndex(const int symbol_bucket,
                                     const int session_bucket,
                                     const int horizon_bucket,
                                     const int side_idx,
                                     const int type_idx)
{
   int sb = symbol_bucket;
   int s = session_bucket;
   int h = horizon_bucket;
   int side = side_idx;
   int type = type_idx;
   if(sb < 0) sb = 0;
   if(sb >= FXAI_BROKER_EXEC_SYMBOL_BUCKETS) sb = FXAI_BROKER_EXEC_SYMBOL_BUCKETS - 1;
   if(s < 0) s = 0;
   if(s >= FXAI_PLUGIN_SESSION_BUCKETS) s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   if(h < 0) h = 0;
   if(h >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) h = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;
   if(side < 0) side = 0;
   if(side >= FXAI_BROKER_EXEC_SIDE_COUNT) side = FXAI_BROKER_EXEC_SIDE_COUNT - 1;
   if(type < 0) type = 0;
   if(type >= FXAI_BROKER_EXEC_ORDER_TYPE_COUNT) type = FXAI_BROKER_EXEC_ORDER_TYPE_COUNT - 1;

   int idx = (((sb * FXAI_PLUGIN_SESSION_BUCKETS + s) * FXAI_SHARED_TRANSFER_HORIZON_BUCKETS + h) * FXAI_BROKER_EXEC_SIDE_COUNT + side) * FXAI_BROKER_EXEC_ORDER_TYPE_COUNT + type;
   return idx;
}

int FXAI_BrokerExecutionLibraryEventIndex(const int symbol_bucket,
                                          const int session_bucket,
                                          const int horizon_bucket,
                                          const int side_idx,
                                          const int type_idx,
                                          const int event_idx)
{
   int base = FXAI_BrokerExecutionLibraryIndex(symbol_bucket, session_bucket, horizon_bucket, side_idx, type_idx);
   int ev = FXAI_BrokerExecutionEventKindIndex(event_idx);
   return base * FXAI_BROKER_EXEC_EVENT_KIND_COUNT + ev;
}

void FXAI_ResetBrokerExecutionReplayStats(void)
{
   g_broker_execution_ready = false;
   g_broker_execution_trace_head = 0;
   g_broker_execution_trace_size = 0;
   for(int s=0; s<FXAI_PLUGIN_SESSION_BUCKETS; s++)
   {
      for(int h=0; h<FXAI_SHARED_TRANSFER_HORIZON_BUCKETS; h++)
      {
         g_broker_execution_obs[s][h] = 0.0;
         g_broker_execution_slippage_ema[s][h] = 0.0;
         g_broker_execution_latency_ema[s][h] = 0.0;
         g_broker_execution_reject_ema[s][h] = 0.0;
         g_broker_execution_partial_ema[s][h] = 0.0;
      }
   }
   for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_CELLS; idx++)
   {
      g_broker_execution_library_obs[idx] = 0.0;
      g_broker_execution_library_slippage[idx] = 0.0;
      g_broker_execution_library_latency[idx] = 0.0;
      g_broker_execution_library_reject[idx] = 0.0;
      g_broker_execution_library_partial[idx] = 0.0;
      g_broker_execution_library_fill_ratio[idx] = 0.0;
   }
   for(int idx=0; idx<FXAI_BROKER_EXEC_LIBRARY_EVENT_CELLS; idx++)
      g_broker_execution_library_event_mass[idx] = 0.0;
   for(int i=0; i<FXAI_BROKER_EXEC_TRACE_CAP; i++)
   {
      g_broker_execution_trace_time[i] = 0;
      g_broker_execution_trace_session[i] = 0;
      g_broker_execution_trace_horizon[i] = 0;
      g_broker_execution_trace_symbol_bucket[i] = 0;
      g_broker_execution_trace_side[i] = 0;
      g_broker_execution_trace_order_type[i] = 0;
      g_broker_execution_trace_event_kind[i] = 0;
      g_broker_execution_trace_slippage[i] = 0.0;
      g_broker_execution_trace_latency[i] = 0.0;
      g_broker_execution_trace_reject[i] = 0.0;
      g_broker_execution_trace_partial[i] = 0.0;
      g_broker_execution_trace_fill_ratio[i] = 0.0;
   }
}

void FXAI_AppendBrokerExecutionTrace(const datetime sample_time,
                                     const int symbol_bucket,
                                     const int session_bucket,
                                     const int horizon_bucket,
                                     const int order_side,
                                     const int order_type_bucket,
                                     const int event_kind,
                                     const double slippage_points,
                                     const double latency_points,
                                     const double reject_prob,
                                     const double partial_fill_prob,
                                     const double fill_ratio)
{
   int idx = g_broker_execution_trace_head;
   if(idx < 0 || idx >= FXAI_BROKER_EXEC_TRACE_CAP)
      idx = 0;
   g_broker_execution_trace_time[idx] = sample_time;
    g_broker_execution_trace_symbol_bucket[idx] = symbol_bucket;
   g_broker_execution_trace_session[idx] = session_bucket;
   g_broker_execution_trace_horizon[idx] = horizon_bucket;
   g_broker_execution_trace_side[idx] = FXAI_NormalizeBrokerExecutionSide(order_side);
   g_broker_execution_trace_order_type[idx] = FXAI_NormalizeBrokerExecutionOrderType(order_type_bucket);
   g_broker_execution_trace_event_kind[idx] = event_kind;
   g_broker_execution_trace_slippage[idx] = MathMax(slippage_points, 0.0);
   g_broker_execution_trace_latency[idx] = MathMax(latency_points, 0.0);
   g_broker_execution_trace_reject[idx] = FXAI_Clamp(reject_prob, 0.0, 1.0);
   g_broker_execution_trace_partial[idx] = FXAI_Clamp(partial_fill_prob, 0.0, 1.0);
   g_broker_execution_trace_fill_ratio[idx] = FXAI_Clamp(fill_ratio, 0.0, 1.0);
   g_broker_execution_trace_head = (idx + 1) % FXAI_BROKER_EXEC_TRACE_CAP;
   if(g_broker_execution_trace_size < FXAI_BROKER_EXEC_TRACE_CAP)
      g_broker_execution_trace_size++;
}

void FXAI_RecordBrokerExecutionEventEx(const datetime sample_time,
                                       const string symbol,
                                       const int horizon_minutes,
                                       const int order_side,
                                       const int order_type_bucket,
                                       const int event_kind,
                                       const double slippage_points,
                                       const double latency_points,
                                       const bool rejected,
                                       const bool partial_fill,
                                       const double fill_ratio)
{
   int s = FXAI_DeriveSessionBucket(sample_time);
   int h = FXAI_BrokerExecutionHorizonBucket(horizon_minutes);
   int symbol_bucket = FXAI_BrokerExecutionSymbolBucket(symbol);
   int side_idx = FXAI_BrokerExecutionSideIndex(order_side);
   int type_idx = FXAI_NormalizeBrokerExecutionOrderType(order_type_bucket);
   int event_idx = FXAI_BrokerExecutionEventKindIndex(event_kind);
   if(s < 0) s = 0;
   if(s >= FXAI_PLUGIN_SESSION_BUCKETS) s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   if(h < 0) h = 0;
   if(h >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) h = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;

   double obs = g_broker_execution_obs[s][h];
   double alpha = FXAI_Clamp(0.18 / MathSqrt(1.0 + 0.05 * obs), 0.02, 0.18);
   double slip = MathMax(slippage_points, 0.0);
   double lat = MathMax(latency_points, 0.0);
   double rej = (rejected ? 1.0 : 0.0);
   double part = (partial_fill ? 1.0 : 0.0);

   if(obs <= 0.0)
   {
      g_broker_execution_slippage_ema[s][h] = slip;
      g_broker_execution_latency_ema[s][h] = lat;
      g_broker_execution_reject_ema[s][h] = rej;
      g_broker_execution_partial_ema[s][h] = part;
   }
   else
   {
      g_broker_execution_slippage_ema[s][h] =
         (1.0 - alpha) * g_broker_execution_slippage_ema[s][h] + alpha * slip;
      g_broker_execution_latency_ema[s][h] =
         (1.0 - alpha) * g_broker_execution_latency_ema[s][h] + alpha * lat;
      g_broker_execution_reject_ema[s][h] =
         (1.0 - alpha) * g_broker_execution_reject_ema[s][h] + alpha * rej;
      g_broker_execution_partial_ema[s][h] =
         (1.0 - alpha) * g_broker_execution_partial_ema[s][h] + alpha * part;
   }

   g_broker_execution_obs[s][h] = MathMin(obs + 1.0, 50000.0);
   int lib_idx = FXAI_BrokerExecutionLibraryIndex(symbol_bucket, s, h, side_idx, type_idx);
   double lib_obs = g_broker_execution_library_obs[lib_idx];
   double lib_alpha = FXAI_Clamp(0.20 / MathSqrt(1.0 + 0.04 * lib_obs), 0.015, 0.20);
   double fill = FXAI_Clamp(fill_ratio, 0.0, 1.0);
   if(lib_obs <= 0.0)
   {
      g_broker_execution_library_slippage[lib_idx] = slip;
      g_broker_execution_library_latency[lib_idx] = lat;
      g_broker_execution_library_reject[lib_idx] = rej;
      g_broker_execution_library_partial[lib_idx] = part;
      g_broker_execution_library_fill_ratio[lib_idx] = fill;
   }
   else
   {
      g_broker_execution_library_slippage[lib_idx] =
         (1.0 - lib_alpha) * g_broker_execution_library_slippage[lib_idx] + lib_alpha * slip;
      g_broker_execution_library_latency[lib_idx] =
         (1.0 - lib_alpha) * g_broker_execution_library_latency[lib_idx] + lib_alpha * lat;
      g_broker_execution_library_reject[lib_idx] =
         (1.0 - lib_alpha) * g_broker_execution_library_reject[lib_idx] + lib_alpha * rej;
      g_broker_execution_library_partial[lib_idx] =
         (1.0 - lib_alpha) * g_broker_execution_library_partial[lib_idx] + lib_alpha * part;
      g_broker_execution_library_fill_ratio[lib_idx] =
         (1.0 - lib_alpha) * g_broker_execution_library_fill_ratio[lib_idx] + lib_alpha * fill;
   }
   g_broker_execution_library_obs[lib_idx] = MathMin(lib_obs + 1.0, 50000.0);
   for(int ev=0; ev<FXAI_BROKER_EXEC_EVENT_KIND_COUNT; ev++)
      g_broker_execution_library_event_mass[FXAI_BrokerExecutionLibraryEventIndex(symbol_bucket, s, h, side_idx, type_idx, ev)] *= (1.0 - lib_alpha);
   int event_mass_idx = FXAI_BrokerExecutionLibraryEventIndex(symbol_bucket, s, h, side_idx, type_idx, event_idx);
   g_broker_execution_library_event_mass[event_mass_idx] =
      MathMin(g_broker_execution_library_event_mass[event_mass_idx] + lib_alpha, 10.0);
   FXAI_AppendBrokerExecutionTrace(sample_time,
                                   symbol_bucket,
                                   s,
                                   h,
                                   order_side,
                                   order_type_bucket,
                                   event_kind,
                                   slip,
                                   lat,
                                   rej,
                                   part,
                                   fill_ratio);
   g_broker_execution_ready = true;
}

void FXAI_RecordBrokerExecutionEvent(const datetime sample_time,
                                     const int horizon_minutes,
                                     const double slippage_points,
                                     const double latency_points,
                                     const bool rejected,
                                     const bool partial_fill)
{
   FXAI_RecordBrokerExecutionEventEx(sample_time,
                                     _Symbol,
                                     horizon_minutes,
                                     0,
                                     0,
                                     (rejected ? 0 : (partial_fill ? 1 : 2)),
                                     slippage_points,
                                     latency_points,
                                     rejected,
                                     partial_fill,
                                     (partial_fill ? 0.5 : (rejected ? 0.0 : 1.0)));
}

void FXAI_GetBrokerExecutionTraceStressEx(const datetime sample_time,
                                          const string symbol,
                                          const int horizon_minutes,
                                          const int order_side,
                                          const int order_type_bucket,
                                          FXAIBrokerExecutionStats &stats)
{
   stats.coverage = 0.0;
   stats.slippage_points = 0.0;
   stats.latency_points = 0.0;
   stats.reject_prob = 0.0;
   stats.partial_fill_prob = 0.0;
   stats.trace_coverage = 0.0;
   stats.library_coverage = 0.0;
   stats.fill_ratio_mean = 1.0;
   stats.event_burst_penalty = 0.0;

   if(g_broker_execution_trace_size <= 0)
      return;

   int target_s = FXAI_DeriveSessionBucket(sample_time);
   int target_h = FXAI_BrokerExecutionHorizonBucket(horizon_minutes);
   int target_symbol_bucket = FXAI_BrokerExecutionSymbolBucket(symbol);
   int target_side = FXAI_NormalizeBrokerExecutionSide(order_side);
   int target_type = FXAI_NormalizeBrokerExecutionOrderType(order_type_bucket);
   if(target_s < 0) target_s = 0;
   if(target_s >= FXAI_PLUGIN_SESSION_BUCKETS) target_s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   if(target_h < 0) target_h = 0;
   if(target_h >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) target_h = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;

   double w_sum = 0.0;
   double slip_sum = 0.0;
   double lat_sum = 0.0;
   double rej_sum = 0.0;
   double part_sum = 0.0;
   double exact_w_sum = 0.0;
   double exact_slip_sum = 0.0;
   double exact_lat_sum = 0.0;
   double exact_rej_sum = 0.0;
   double exact_part_sum = 0.0;
   int lookback = MathMin(g_broker_execution_trace_size, FXAI_BROKER_EXEC_TRACE_CAP);
   for(int n=0; n<lookback; n++)
   {
      int idx = g_broker_execution_trace_head - 1 - n;
      while(idx < 0) idx += FXAI_BROKER_EXEC_TRACE_CAP;
      if(idx >= FXAI_BROKER_EXEC_TRACE_CAP) idx %= FXAI_BROKER_EXEC_TRACE_CAP;

      double age_w = 1.0 / (1.0 + 0.08 * (double)n);
      double sess_w = (g_broker_execution_trace_session[idx] == target_s ? 1.0 : 0.35);
      int h_delta = MathAbs(g_broker_execution_trace_horizon[idx] - target_h);
      double horizon_w = 1.0 / (1.0 + 0.75 * (double)h_delta);
      double symbol_w = (g_broker_execution_trace_symbol_bucket[idx] == target_symbol_bucket ? 1.0 : 0.55);
      int trace_side = FXAI_NormalizeBrokerExecutionSide(g_broker_execution_trace_side[idx]);
      double side_w = (target_side == 0 || trace_side == 0 ? 0.85 : (trace_side == target_side ? 1.0 : 0.55));
      int trace_type = FXAI_NormalizeBrokerExecutionOrderType(g_broker_execution_trace_order_type[idx]);
      double type_w = (target_type == 0 || trace_type == 0 ? 0.88 : (trace_type == target_type ? 1.0 : 0.65));
      double time_w = 1.0;
      datetime ev_time = g_broker_execution_trace_time[idx];
      if(sample_time > 0 && ev_time > 0)
      {
         double delta_hours = MathAbs((double)(sample_time - ev_time)) / 3600.0;
         time_w = 1.0 / (1.0 + delta_hours / 72.0);
      }
      double severity = 1.0 +
                        0.20 * g_broker_execution_trace_reject[idx] +
                        0.15 * g_broker_execution_trace_partial[idx] +
                        0.10 * FXAI_Clamp(g_broker_execution_trace_latency[idx], 0.0, 4.0);
      double w = age_w * sess_w * horizon_w * symbol_w * time_w * severity;
      if(w <= 1e-6)
         continue;

      w_sum += w;
      slip_sum += w * g_broker_execution_trace_slippage[idx];
      lat_sum += w * g_broker_execution_trace_latency[idx];
      rej_sum += w * g_broker_execution_trace_reject[idx];
      double fill_shortfall = FXAI_Clamp(1.0 - g_broker_execution_trace_fill_ratio[idx], 0.0, 1.0);
      part_sum += w * MathMax(g_broker_execution_trace_partial[idx], fill_shortfall);

      double exact_w = w * side_w * type_w;
      if(exact_w > 1e-6)
      {
         exact_w_sum += exact_w;
         exact_slip_sum += exact_w * g_broker_execution_trace_slippage[idx];
         exact_lat_sum += exact_w * g_broker_execution_trace_latency[idx];
         exact_rej_sum += exact_w * g_broker_execution_trace_reject[idx];
         double exact_fill_shortfall = FXAI_Clamp(1.0 - g_broker_execution_trace_fill_ratio[idx], 0.0, 1.0);
         exact_part_sum += exact_w * MathMax(g_broker_execution_trace_partial[idx], exact_fill_shortfall);
      }
   }

   if(w_sum <= 1e-6)
      return;

   double general_cov = FXAI_Clamp(w_sum / 16.0, 0.0, 1.0);
   double exact_cov = FXAI_Clamp(exact_w_sum / 10.0, 0.0, 1.0);
   double exact_blend = FXAI_Clamp(0.20 + 0.70 * exact_cov, 0.0, 0.85);
   if(exact_w_sum <= 1e-6)
      exact_blend = 0.0;

   double slip_general = MathMax(slip_sum / w_sum, 0.0);
   double lat_general = MathMax(lat_sum / w_sum, 0.0);
   double rej_general = FXAI_Clamp(rej_sum / w_sum, 0.0, 1.0);
   double part_general = FXAI_Clamp(part_sum / w_sum, 0.0, 1.0);
   double slip_exact = (exact_w_sum > 1e-6 ? MathMax(exact_slip_sum / exact_w_sum, 0.0) : slip_general);
   double lat_exact = (exact_w_sum > 1e-6 ? MathMax(exact_lat_sum / exact_w_sum, 0.0) : lat_general);
   double rej_exact = (exact_w_sum > 1e-6 ? FXAI_Clamp(exact_rej_sum / exact_w_sum, 0.0, 1.0) : rej_general);
   double part_exact = (exact_w_sum > 1e-6 ? FXAI_Clamp(exact_part_sum / exact_w_sum, 0.0, 1.0) : part_general);

   stats.coverage = FXAI_Clamp(0.55 * general_cov + 0.45 * exact_cov, 0.0, 1.0);
   stats.trace_coverage = exact_cov;
   stats.slippage_points = (1.0 - exact_blend) * slip_general + exact_blend * MathMax(slip_general, slip_exact);
   stats.latency_points = (1.0 - exact_blend) * lat_general + exact_blend * MathMax(lat_general, lat_exact);
   stats.reject_prob = FXAI_Clamp((1.0 - exact_blend) * rej_general + exact_blend * MathMax(rej_general, rej_exact), 0.0, 1.0);
   stats.partial_fill_prob = FXAI_Clamp((1.0 - exact_blend) * part_general + exact_blend * MathMax(part_general, part_exact), 0.0, 1.0);
   stats.fill_ratio_mean = FXAI_Clamp(1.0 - stats.partial_fill_prob, 0.0, 1.0);
   stats.event_burst_penalty = FXAI_Clamp(0.60 * stats.reject_prob + 0.40 * stats.partial_fill_prob, 0.0, 1.0);
}

void FXAI_GetBrokerExecutionTraceStress(const datetime sample_time,
                                        const int horizon_minutes,
                                        FXAIBrokerExecutionStats &stats)
{
   FXAI_GetBrokerExecutionTraceStressEx(sample_time, _Symbol, horizon_minutes, 0, 0, stats);
}

void FXAI_GetBrokerExecutionLibraryStressEx(const datetime sample_time,
                                            const string symbol,
                                            const int horizon_minutes,
                                            const int order_side,
                                            const int order_type_bucket,
                                            FXAIBrokerExecutionStats &stats)
{
   stats.coverage = 0.0;
   stats.slippage_points = 0.0;
   stats.latency_points = 0.0;
   stats.reject_prob = 0.0;
   stats.partial_fill_prob = 0.0;
   stats.trace_coverage = 0.0;
   stats.library_coverage = 0.0;
   stats.fill_ratio_mean = 1.0;
   stats.event_burst_penalty = 0.0;

   int target_s = FXAI_DeriveSessionBucket(sample_time);
   int target_h = FXAI_BrokerExecutionHorizonBucket(horizon_minutes);
   int target_symbol_bucket = FXAI_BrokerExecutionSymbolBucket(symbol);
   int target_side = FXAI_BrokerExecutionSideIndex(order_side);
   int target_type = FXAI_NormalizeBrokerExecutionOrderType(order_type_bucket);
   if(target_s < 0) target_s = 0;
   if(target_s >= FXAI_PLUGIN_SESSION_BUCKETS) target_s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   if(target_h < 0) target_h = 0;
   if(target_h >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) target_h = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;

   double w_sum = 0.0;
   double slip_sum = 0.0;
   double lat_sum = 0.0;
   double rej_sum = 0.0;
   double part_sum = 0.0;
   double fill_sum = 0.0;
   double burst_sum = 0.0;
   for(int side=0; side<FXAI_BROKER_EXEC_SIDE_COUNT; side++)
   {
      for(int type=0; type<FXAI_BROKER_EXEC_ORDER_TYPE_COUNT; type++)
      {
         int lib_idx = FXAI_BrokerExecutionLibraryIndex(target_symbol_bucket, target_s, target_h, side, type);
         double obs = g_broker_execution_library_obs[lib_idx];
         if(obs <= 1e-6)
            continue;
         double side_w = (side == target_side ? 1.0 : (target_side == 1 || side == 1 ? 0.72 : 0.48));
         double type_w = (type == target_type ? 1.0 : (target_type == 0 || type == 0 ? 0.80 : 0.58));
         double cov_w = FXAI_Clamp(obs / 12.0, 0.08, 1.0);
         double w = side_w * type_w * cov_w;
         if(w <= 1e-6)
            continue;

         double event_mass_sum = 0.0;
         double reject_event = 0.0;
         double partial_event = 0.0;
         for(int ev=0; ev<FXAI_BROKER_EXEC_EVENT_KIND_COUNT; ev++)
         {
            double mass = g_broker_execution_library_event_mass[FXAI_BrokerExecutionLibraryEventIndex(target_symbol_bucket, target_s, target_h, side, type, ev)];
            event_mass_sum += mass;
            if(ev == 0)
               reject_event += mass;
            else if(ev == 1)
               partial_event += mass;
         }
         double burst_penalty = 0.0;
         if(event_mass_sum > 1e-6)
            burst_penalty = FXAI_Clamp((reject_event + 0.65 * partial_event) / event_mass_sum, 0.0, 1.0);

         w_sum += w;
         slip_sum += w * g_broker_execution_library_slippage[lib_idx];
         lat_sum += w * g_broker_execution_library_latency[lib_idx];
         rej_sum += w * g_broker_execution_library_reject[lib_idx];
         part_sum += w * g_broker_execution_library_partial[lib_idx];
         fill_sum += w * g_broker_execution_library_fill_ratio[lib_idx];
         burst_sum += w * burst_penalty;
      }
   }

   if(w_sum <= 1e-6)
      return;
   stats.coverage = FXAI_Clamp(w_sum / 3.5, 0.0, 1.0);
   stats.library_coverage = stats.coverage;
   stats.slippage_points = MathMax(slip_sum / w_sum, 0.0);
   stats.latency_points = MathMax(lat_sum / w_sum, 0.0);
   stats.reject_prob = FXAI_Clamp(rej_sum / w_sum, 0.0, 1.0);
   stats.partial_fill_prob = FXAI_Clamp(part_sum / w_sum, 0.0, 1.0);
   stats.fill_ratio_mean = FXAI_Clamp(fill_sum / w_sum, 0.0, 1.0);
   stats.event_burst_penalty = FXAI_Clamp(burst_sum / w_sum, 0.0, 1.0);
}

void FXAI_GetBrokerExecutionStressEx(const datetime sample_time,
                                     const string symbol,
                                     const int horizon_minutes,
                                     const int order_side,
                                     const int order_type_bucket,
                                     FXAIBrokerExecutionStats &stats)
{
   stats.coverage = 0.0;
   stats.slippage_points = 0.0;
   stats.latency_points = 0.0;
   stats.reject_prob = 0.0;
   stats.partial_fill_prob = 0.0;
   stats.trace_coverage = 0.0;
   stats.library_coverage = 0.0;
   stats.fill_ratio_mean = 1.0;
   stats.event_burst_penalty = 0.0;

   if(!g_broker_execution_ready)
   {
      FXAI_GetBrokerExecutionLibraryStressEx(sample_time, symbol, horizon_minutes, order_side, order_type_bucket, stats);
      FXAIBrokerExecutionStats trace_only;
      FXAI_GetBrokerExecutionTraceStressEx(sample_time, symbol, horizon_minutes, order_side, order_type_bucket, trace_only);
      if(trace_only.coverage > 1e-6)
      {
         stats.coverage = trace_only.coverage;
         stats.slippage_points = trace_only.slippage_points;
         stats.latency_points = trace_only.latency_points;
         stats.reject_prob = trace_only.reject_prob;
         stats.partial_fill_prob = trace_only.partial_fill_prob;
         stats.trace_coverage = trace_only.trace_coverage;
         stats.library_coverage = trace_only.library_coverage;
         stats.fill_ratio_mean = trace_only.fill_ratio_mean;
         stats.event_burst_penalty = trace_only.event_burst_penalty;
      }
      return;
   }

   int s = FXAI_DeriveSessionBucket(sample_time);
   int h = FXAI_BrokerExecutionHorizonBucket(horizon_minutes);
   if(s < 0) s = 0;
   if(s >= FXAI_PLUGIN_SESSION_BUCKETS) s = FXAI_PLUGIN_SESSION_BUCKETS - 1;
   if(h < 0) h = 0;
   if(h >= FXAI_SHARED_TRANSFER_HORIZON_BUCKETS) h = FXAI_SHARED_TRANSFER_HORIZON_BUCKETS - 1;

   double obs = g_broker_execution_obs[s][h];
   if(obs <= 0.0)
      return;

   stats.coverage = FXAI_Clamp(obs / 64.0, 0.0, 1.0);
   stats.slippage_points = MathMax(g_broker_execution_slippage_ema[s][h], 0.0);
   stats.latency_points = MathMax(g_broker_execution_latency_ema[s][h], 0.0);
   stats.reject_prob = FXAI_Clamp(g_broker_execution_reject_ema[s][h], 0.0, 1.0);
   stats.partial_fill_prob = FXAI_Clamp(g_broker_execution_partial_ema[s][h], 0.0, 1.0);
   stats.fill_ratio_mean = FXAI_Clamp(1.0 - stats.partial_fill_prob, 0.0, 1.0);
   stats.event_burst_penalty = FXAI_Clamp(0.55 * stats.reject_prob + 0.35 * stats.partial_fill_prob, 0.0, 1.0);

   FXAIBrokerExecutionStats library_stats;
   FXAI_GetBrokerExecutionLibraryStressEx(sample_time, symbol, horizon_minutes, order_side, order_type_bucket, library_stats);
   if(library_stats.coverage > 1e-6)
   {
      double blend = FXAI_Clamp(0.25 + 0.60 * library_stats.coverage, 0.0, 0.82);
      stats.slippage_points = (1.0 - blend) * stats.slippage_points + blend * MathMax(stats.slippage_points, library_stats.slippage_points);
      stats.latency_points = (1.0 - blend) * stats.latency_points + blend * MathMax(stats.latency_points, library_stats.latency_points);
      stats.reject_prob = FXAI_Clamp((1.0 - blend) * stats.reject_prob + blend * MathMax(stats.reject_prob, library_stats.reject_prob), 0.0, 1.0);
      stats.partial_fill_prob = FXAI_Clamp((1.0 - blend) * stats.partial_fill_prob + blend * MathMax(stats.partial_fill_prob, library_stats.partial_fill_prob), 0.0, 1.0);
      stats.fill_ratio_mean = FXAI_Clamp((1.0 - blend) * stats.fill_ratio_mean + blend * library_stats.fill_ratio_mean, 0.0, 1.0);
      stats.event_burst_penalty = FXAI_Clamp((1.0 - blend) * stats.event_burst_penalty + blend * library_stats.event_burst_penalty, 0.0, 1.0);
      stats.coverage = FXAI_Clamp(0.60 * stats.coverage + 0.40 * library_stats.coverage, 0.0, 1.0);
      stats.library_coverage = library_stats.coverage;
   }

   FXAIBrokerExecutionStats trace_stats;
   FXAI_GetBrokerExecutionTraceStressEx(sample_time, symbol, horizon_minutes, order_side, order_type_bucket, trace_stats);
   if(trace_stats.coverage > 1e-6)
   {
      double blend = FXAI_Clamp(0.30 + 0.50 * trace_stats.coverage, 0.0, 0.80);
      stats.slippage_points = (1.0 - blend) * stats.slippage_points + blend * MathMax(stats.slippage_points, trace_stats.slippage_points);
      stats.latency_points = (1.0 - blend) * stats.latency_points + blend * MathMax(stats.latency_points, trace_stats.latency_points);
      stats.reject_prob = FXAI_Clamp((1.0 - blend) * stats.reject_prob + blend * MathMax(stats.reject_prob, trace_stats.reject_prob), 0.0, 1.0);
      stats.partial_fill_prob = FXAI_Clamp((1.0 - blend) * stats.partial_fill_prob + blend * MathMax(stats.partial_fill_prob, trace_stats.partial_fill_prob), 0.0, 1.0);
      stats.coverage = FXAI_Clamp(0.60 * stats.coverage + 0.40 * trace_stats.coverage, 0.0, 1.0);
      stats.trace_coverage = trace_stats.coverage;
      stats.fill_ratio_mean = FXAI_Clamp((1.0 - blend) * stats.fill_ratio_mean + blend * trace_stats.fill_ratio_mean, 0.0, 1.0);
      stats.event_burst_penalty = FXAI_Clamp((1.0 - blend) * stats.event_burst_penalty + blend * trace_stats.event_burst_penalty, 0.0, 1.0);
   }
}

void FXAI_GetBrokerExecutionStress(const datetime sample_time,
                                   const int horizon_minutes,
                                   FXAIBrokerExecutionStats &stats)
{
   FXAI_GetBrokerExecutionStressEx(sample_time, _Symbol, horizon_minutes, 0, 0, stats);
}

double FXAI_Tanh(const double z)
{
   if(z > 18.0) return 1.0;
   if(z < -18.0) return -1.0;
   double e2 = MathExp(2.0 * z);
   return (e2 - 1.0) / (e2 + 1.0);
}

double FXAI_DotLinear(const double &w[], const double &x[])
{
   double z = 0.0;
   for(int i=0; i<FXAI_AI_WEIGHTS; i++)
      z += w[i] * x[i];
   return z;
}

double FXAI_Sign(const double v)
{
   if(v > 0.0) return 1.0;
   if(v < 0.0) return -1.0;
   return 0.0;
}

double FXAI_ClipSym(const double v, const double limit_abs)
{
   double lim = (limit_abs > 0.0 ? limit_abs : 0.0);
   if(lim <= 0.0) return v;
   if(v > lim) return lim;
   if(v < -lim) return -lim;
   return v;
}

double FXAI_MoveWeight(const double move_points)
{
   double a = MathAbs(move_points);
   // Keep weighting lightweight and bounded for stable online updates.
   return FXAI_Clamp(1.0 + (0.05 * a), 0.80, 3.00);
}

double FXAI_MoveEdgeWeight(const double move_points, const double cost_points)
{
   double mv = MathAbs(move_points);
   double c = (cost_points > 0.0 ? cost_points : 0.0);
   double edge = mv - c;
   double denom = MathMax(c, 1.0);
   return FXAI_Clamp(0.50 + (edge / denom), 0.25, 4.00);
}

double FXAI_PathRiskFromTargets(const double mfe_points,
                                const double mae_points,
                                const double min_move_points,
                                const double time_to_hit_frac,
                                const int path_flags)
{
   double mfe = MathMax(MathAbs(mfe_points), MathMax(min_move_points, 0.10));
   double mae = MathMax(MathAbs(mae_points), 0.0);
   double adverse_ratio = FXAI_Clamp(mae / mfe, 0.0, 3.0);
   double slow = FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
   double risk = 0.45 * adverse_ratio + 0.30 * slow;
   if((path_flags & 1) != 0)
      risk += 0.15;
   if((path_flags & 2) != 0)
      risk += 0.10;
   if((path_flags & 8) != 0)
      risk += 0.08;
   return FXAI_Clamp(risk, 0.0, 1.0);
}

double FXAI_FillRiskFromTargets(const double spread_stress_points,
                                const double min_move_points,
                                const double cost_points)
{
   double denom = MathMax(min_move_points + MathMax(cost_points, 0.0), 0.25);
   return FXAI_Clamp(MathAbs(spread_stress_points) / denom, 0.0, 1.0);
}

void FXAI_ClearExecutionProfile(FXAIExecutionProfile &profile)
{
   profile.profile_id = (int)FXAI_EXEC_DEFAULT;
   profile.commission_per_lot_side = 0.0;
   profile.cost_buffer_points = 2.0;
   profile.slippage_points = 0.0;
   profile.fill_penalty_points = 0.0;
   profile.slippage_cost_weight = 0.04;
   profile.slippage_stress_weight = 0.18;
   profile.slippage_horizon_weight = 0.02;
   profile.dual_hit_penalty = 0.12;
   profile.slow_hit_penalty = 0.10;
   profile.spread_shock_penalty = 0.25;
   profile.partial_fill_penalty = 0.0;
   profile.latency_penalty_points = 0.0;
   profile.allowed_deviation_points = 2.0;
}

void FXAI_ClearExecutionReplayFrame(FXAIExecutionReplayFrame &frame)
{
   frame.slippage_mult = 1.0;
   frame.fill_mult = 1.0;
   frame.latency_add_points = 0.0;
   frame.reject_prob = 0.0;
   frame.partial_fill_prob = 0.0;
   frame.drift_penalty_points = 0.0;
   frame.event_flags = 0;
}

void FXAI_ClearExecutionTraceStats(FXAIExecutionTraceStats &trace)
{
   trace.spread_mean_ratio = 1.0;
   trace.spread_peak_ratio = 1.0;
   trace.range_mean_ratio = 1.0;
   trace.body_efficiency = 0.5;
   trace.gap_ratio = 0.0;
   trace.reversal_ratio = 0.0;
   trace.session_transition_exposure = 0.0;
   trace.rollover_exposure = 0.0;
}

void FXAI_BuildExecutionTraceStats(const int i,
                                   const int horizon_minutes,
                                   const double point_value,
                                   const datetime &time_arr[],
                                   const double &open_arr[],
                                   const double &high_arr[],
                                   const double &low_arr[],
                                   const double &close_arr[],
                                   const int &spread_arr[],
                                   FXAIExecutionTraceStats &trace)
{
   FXAI_ClearExecutionTraceStats(trace);
   int n = ArraySize(close_arr);
   if(i < 0 || i >= n || point_value <= 0.0)
      return;

   int steps = horizon_minutes;
   if(steps < 1)
      steps = 1;
   if(steps > 1440)
      steps = 1440;
   if(steps > FXAI_EXEC_TRACE_BARS)
      steps = FXAI_EXEC_TRACE_BARS;
   if(i < steps)
      steps = i;
   if(steps < 1)
      steps = 1;

   double entry_spread = MathMax(FXAI_GetSpreadAtIndex(i, spread_arr, 1.0), 0.25);
   double entry_range = MathMax((high_arr[i] - low_arr[i]) / point_value, 0.25);
   double spread_sum = 0.0;
   double spread_peak = entry_spread;
   double range_sum = 0.0;
   double body_sum = 0.0;
   double gap_sum = 0.0;
   int reversal_count = 0;
   double prev_dir = 0.0;
   double session_sum = 0.0;
   double rollover_sum = 0.0;

   for(int step=0; step<=steps; step++)
   {
      int idx = i - step;
      if(idx < 0 || idx >= n)
         break;

      double spread = MathMax(FXAI_GetSpreadAtIndex(idx, spread_arr, entry_spread), 0.0);
      if(spread > spread_peak)
         spread_peak = spread;
      spread_sum += spread;

      double range_points = MathMax((high_arr[idx] - low_arr[idx]) / point_value, 0.0);
      range_sum += range_points;
      double body_eff = MathAbs(close_arr[idx] - open_arr[idx]) / MathMax(high_arr[idx] - low_arr[idx], point_value);
      body_sum += FXAI_Clamp(body_eff, 0.0, 1.0);

      if(idx + 1 < n)
      {
         double gap_points = MathAbs(open_arr[idx] - close_arr[idx + 1]) / point_value;
         gap_sum += gap_points;
      }

      double bar_dir = FXAI_Sign(close_arr[idx] - open_arr[idx]);
      if(step > 0 && MathAbs(bar_dir) > 1e-9 && MathAbs(prev_dir) > 1e-9 && bar_dir != prev_dir)
         reversal_count++;
      if(MathAbs(bar_dir) > 1e-9)
         prev_dir = bar_dir;

      datetime t = (idx < ArraySize(time_arr) ? time_arr[idx] : 0);
      MqlDateTime dt;
      TimeToStruct(t, dt);
      double hour_value = (double)dt.hour + (double)dt.min / 60.0;
      double asia_to_eu = FXAI_Clamp(1.0 - MathAbs(hour_value - 7.0) / 2.5, 0.0, 1.0);
      double eu_to_us = FXAI_Clamp(1.0 - MathAbs(hour_value - 13.0) / 2.5, 0.0, 1.0);
      double us_to_roll = FXAI_Clamp(1.0 - MathAbs(hour_value - 21.0) / 2.5, 0.0, 1.0);
      session_sum += FXAI_Clamp(0.60 * asia_to_eu + 0.80 * eu_to_us - 0.70 * us_to_roll, -1.0, 1.0);
      double roll_d = MathAbs(hour_value - 23.0);
      if(roll_d > 12.0)
         roll_d = 24.0 - roll_d;
      rollover_sum += FXAI_Clamp(1.0 - roll_d / 3.0, 0.0, 1.0);
   }

   double used = (double)(steps + 1);
   trace.spread_mean_ratio = FXAI_Clamp((spread_sum / used) / entry_spread, 0.5, 6.0);
   trace.spread_peak_ratio = FXAI_Clamp(spread_peak / entry_spread, 1.0, 10.0);
   trace.range_mean_ratio = FXAI_Clamp((range_sum / used) / entry_range, 0.25, 8.0);
   trace.body_efficiency = FXAI_Clamp(body_sum / used, 0.0, 1.0);
   trace.gap_ratio = FXAI_Clamp((gap_sum / used) / entry_spread, 0.0, 8.0);
   trace.reversal_ratio = FXAI_Clamp((double)reversal_count / MathMax((double)steps, 1.0), 0.0, 1.0);
   trace.session_transition_exposure = FXAI_Clamp(0.5 + 0.5 * (session_sum / used), 0.0, 1.0);
   trace.rollover_exposure = FXAI_Clamp(rollover_sum / used, 0.0, 1.0);
}

void FXAI_BuildExecutionReplayFrame(const FXAIExecutionProfile &profile,
                                    const datetime sample_time,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags,
                                    const int scenario_id,
                                    const FXAIExecutionTraceStats &trace,
                                    FXAIExecutionReplayFrame &frame)
{
   FXAI_ClearExecutionReplayFrame(frame);

   MqlDateTime dt;
   TimeToStruct((sample_time > 0 ? sample_time : TimeCurrent()), dt);
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double horizon_scale = MathSqrt((double)MathMax(horizon_minutes, 1));
   double mm_norm = ((double)dt.min + (double)dt.sec / 60.0) / 60.0;
   double session_edge = 1.0 - MathMin(MathAbs((double)dt.hour - 8.0), MathAbs((double)dt.hour - 16.0)) / 8.0;
   session_edge = FXAI_Clamp(session_edge, 0.0, 1.0);
   double rollover_edge = 1.0 - MathMin(MathAbs((double)dt.hour - 23.0), 6.0) / 6.0;
   rollover_edge = FXAI_Clamp(rollover_edge, 0.0, 1.0);
   double pulse = 0.5 + 0.5 * MathSin(6.283185307179586 * mm_norm);
   double trace_spread = FXAI_Clamp(0.55 * (trace.spread_mean_ratio - 1.0) +
                                    0.45 * (trace.spread_peak_ratio - 1.0),
                                    0.0,
                                    6.0);
   double trace_range = FXAI_Clamp(trace.range_mean_ratio - 1.0, -0.5, 4.0);
   double trace_gap = FXAI_Clamp(trace.gap_ratio, 0.0, 6.0);
   double trace_reversal = FXAI_Clamp(trace.reversal_ratio, 0.0, 1.0);
   double trace_session = FXAI_Clamp(0.55 * session_edge + 0.45 * trace.session_transition_exposure, 0.0, 1.0);
   double trace_roll = FXAI_Clamp(0.55 * rollover_edge + 0.45 * trace.rollover_exposure, 0.0, 1.0);
   double body_penalty = FXAI_Clamp(1.0 - trace.body_efficiency, 0.0, 1.0);

   frame.slippage_mult = 1.0 +
                         0.04 * stress +
                         0.05 * trace_spread +
                         0.03 * MathMax(trace_range, 0.0) +
                         0.04 * trace_gap +
                         0.04 * trace_session +
                         0.03 * trace_roll +
                         0.015 * horizon_scale +
                         0.02 * pulse;
   frame.fill_mult = 1.0 +
                     0.03 * stress +
                     0.05 * trace_reversal +
                     0.03 * body_penalty +
                     0.03 * trace_session +
                     0.02 * trace_roll;
   frame.latency_add_points = MathMax(profile.latency_penalty_points, 0.0) *
                              (0.32 + 0.22 * trace_spread + 0.18 * trace_gap + 0.16 * trace_session + 0.12 * trace_roll + 0.08 * stress);
   frame.reject_prob = FXAI_Clamp(0.002 +
                                  0.008 * stress +
                                  0.012 * trace_spread +
                                  0.010 * trace_gap +
                                  0.009 * trace_session +
                                  0.006 * trace_roll,
                                  0.0,
                                  0.35);
   frame.partial_fill_prob = FXAI_Clamp(0.01 +
                                        0.030 * stress +
                                        0.040 * trace_spread +
                                        0.030 * trace_reversal +
                                        0.020 * body_penalty +
                                        0.020 * trace_session,
                                        0.0,
                                        0.95);
   frame.drift_penalty_points = 0.05 * MathMax(profile.cost_buffer_points, 0.0) *
                                (trace_session + trace_roll + 0.35 * trace_spread);

   FXAIBrokerExecutionStats broker_stats;
   FXAI_GetBrokerExecutionStress(sample_time, horizon_minutes, broker_stats);
   if(broker_stats.coverage > 1e-6)
   {
      double slip_ref = MathMax(profile.slippage_points + 0.25, 0.25);
      double fill_shortfall = FXAI_Clamp(1.0 - broker_stats.fill_ratio_mean, 0.0, 1.0);
      double burst_penalty = FXAI_Clamp(broker_stats.event_burst_penalty, 0.0, 1.0);
      double broker_slip_mult = 1.0 + 0.35 * broker_stats.coverage *
                                FXAI_Clamp(broker_stats.slippage_points / slip_ref, 0.0, 3.0);
      frame.slippage_mult *= broker_slip_mult;
      frame.fill_mult *= 1.0 + broker_stats.coverage *
                         (0.18 * broker_stats.partial_fill_prob + 0.12 * fill_shortfall + 0.08 * burst_penalty);
      frame.latency_add_points += broker_stats.coverage * broker_stats.latency_points;
      frame.reject_prob = FXAI_Clamp((1.0 - 0.55 * broker_stats.coverage) * frame.reject_prob +
                                     0.55 * broker_stats.coverage * MathMax(frame.reject_prob,
                                                                            FXAI_Clamp(broker_stats.reject_prob + 0.20 * burst_penalty,
                                                                                       0.0,
                                                                                       1.0)),
                                     0.0,
                                     0.75);
      frame.partial_fill_prob = FXAI_Clamp((1.0 - 0.55 * broker_stats.coverage) * frame.partial_fill_prob +
                                           0.55 * broker_stats.coverage * MathMax(frame.partial_fill_prob,
                                                                                  FXAI_Clamp(broker_stats.partial_fill_prob + 0.35 * fill_shortfall,
                                                                                             0.0,
                                                                                             1.0)),
                                           0.0,
                                           0.99);
      frame.drift_penalty_points += 0.20 * broker_stats.coverage * broker_stats.slippage_points +
                                    0.12 * broker_stats.coverage * burst_penalty * MathMax(profile.cost_buffer_points, 1.0);
   }

   if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0)
      frame.event_flags |= FXAI_PATHFLAG_DUAL_HIT;
   if((path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0 || trace.spread_peak_ratio > 1.35)
      frame.event_flags |= FXAI_PATHFLAG_SPREAD_STRESS;

   if(scenario_id == 11) // market_session_edges
   {
      frame.slippage_mult += 0.10;
      frame.fill_mult += 0.08;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.05, 0.0, 0.45);
      frame.partial_fill_prob = FXAI_Clamp(frame.partial_fill_prob + 0.10, 0.0, 0.98);
      frame.event_flags |= FXAI_PATHFLAG_SLOW_HIT;
   }
   else if(scenario_id == 12) // market_spread_shock
   {
      frame.slippage_mult += 0.12 + 0.03 * trace_spread;
      frame.fill_mult += 0.10 + 0.02 * trace_spread;
      frame.latency_add_points += 0.20 + 0.10 * stress + 0.06 * trace_gap;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.08, 0.0, 0.50);
      frame.partial_fill_prob = FXAI_Clamp(frame.partial_fill_prob + 0.14, 0.0, 0.99);
      frame.event_flags |= FXAI_PATHFLAG_SPREAD_STRESS;
   }
   else if(scenario_id == 13) // market_walkforward
   {
      frame.slippage_mult += 0.04 + 0.02 * trace_reversal;
      frame.fill_mult += 0.03 + 0.02 * body_penalty;
      frame.reject_prob = FXAI_Clamp(frame.reject_prob + 0.03, 0.0, 0.40);
   }
}

void FXAI_BuildExecutionReplayFrame(const FXAIExecutionProfile &profile,
                                    const datetime sample_time,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags,
                                    const int scenario_id,
                                    FXAIExecutionReplayFrame &frame)
{
   FXAIExecutionTraceStats trace;
   FXAI_ClearExecutionTraceStats(trace);
   FXAI_BuildExecutionReplayFrame(profile,
                                  sample_time,
                                  horizon_minutes,
                                  spread_stress,
                                  path_flags,
                                  scenario_id,
                                  trace,
                                  frame);
}

void FXAI_SetExecutionProfilePreset(const int profile_id,
                                    FXAIExecutionProfile &profile)
{
   FXAI_ClearExecutionProfile(profile);
   profile.profile_id = profile_id;

   if(profile_id == (int)FXAI_EXEC_TIGHT_FX)
   {
      profile.cost_buffer_points = 1.5;
      profile.slippage_points = 0.10;
      profile.fill_penalty_points = 0.10;
      profile.allowed_deviation_points = 2.0;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_PRIME_ECN)
   {
      profile.commission_per_lot_side = 3.5;
      profile.cost_buffer_points = 1.5;
      profile.slippage_points = 0.20;
      profile.fill_penalty_points = 0.15;
      profile.allowed_deviation_points = 2.5;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_RETAIL_FX)
   {
      profile.cost_buffer_points = 2.5;
      profile.slippage_points = 0.40;
      profile.fill_penalty_points = 0.25;
      profile.slippage_cost_weight = 0.05;
      profile.allowed_deviation_points = 4.0;
      return;
   }
   if(profile_id == (int)FXAI_EXEC_STRESS)
   {
      profile.commission_per_lot_side = 5.0;
      profile.cost_buffer_points = 3.5;
      profile.slippage_points = 1.0;
      profile.fill_penalty_points = 0.50;
      profile.slippage_cost_weight = 0.06;
      profile.slippage_stress_weight = 0.25;
      profile.slippage_horizon_weight = 0.03;
      profile.dual_hit_penalty = 0.20;
      profile.slow_hit_penalty = 0.16;
      profile.spread_shock_penalty = 0.55;
      profile.partial_fill_penalty = 0.25;
      profile.latency_penalty_points = 0.20;
      profile.allowed_deviation_points = 8.0;
      return;
   }
}

double FXAI_ExecutionEntryCostPoints(const double spread_points,
                                     const double commission_points,
                                     const double base_cost_buffer_points,
                                     const FXAIExecutionProfile &profile)
{
   double cost = MathMax(spread_points, 0.0) +
                 MathMax(commission_points, 0.0) +
                 MathMax(base_cost_buffer_points, 0.0) +
                 MathMax(profile.cost_buffer_points, 0.0) +
                 MathMax(profile.slippage_points, 0.0) +
                 MathMax(profile.fill_penalty_points, 0.0) +
                 MathMax(profile.latency_penalty_points, 0.0);
   if(cost < 0.0) cost = 0.0;
   return cost;
}

double FXAI_ExecutionSlippagePoints(const FXAIExecutionProfile &profile,
                                    const double roundtrip_cost_points,
                                    const int horizon_minutes,
                                    const double spread_stress,
                                    const int path_flags)
{
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double slippage_points = MathMax(profile.slippage_points, 0.0) +
                            profile.slippage_cost_weight * MathMax(roundtrip_cost_points, 0.0) +
                            profile.slippage_stress_weight * stress +
                            profile.slippage_horizon_weight * MathSqrt((double)MathMax(horizon_minutes, 1)) +
                            MathMax(profile.latency_penalty_points, 0.0);
   if((path_flags & 1) != 0)
      slippage_points += MathMax(profile.dual_hit_penalty, 0.0) +
                         0.12 * MathMax(roundtrip_cost_points, 0.0) +
                         0.12 * stress;
   if((path_flags & 8) != 0)
      slippage_points += MathMax(profile.slow_hit_penalty, 0.0);
   if((path_flags & 4) != 0)
      slippage_points += MathMax(profile.spread_shock_penalty, 0.0);
   if(slippage_points > 12.0) slippage_points = 12.0;
   return slippage_points;
}

double FXAI_ExecutionSlippagePointsReplay(const FXAIExecutionProfile &profile,
                                          const FXAIExecutionReplayFrame &frame,
                                          const double roundtrip_cost_points,
                                          const int horizon_minutes,
                                          const double spread_stress,
                                          const int path_flags)
{
   double slippage_points = FXAI_ExecutionSlippagePoints(profile,
                                                         roundtrip_cost_points,
                                                         horizon_minutes,
                                                         spread_stress,
                                                         path_flags | frame.event_flags);
   slippage_points = slippage_points * FXAI_Clamp(frame.slippage_mult, 0.50, 3.00) +
                     MathMax(frame.latency_add_points, 0.0) +
                     MathMax(frame.drift_penalty_points, 0.0);
   if(slippage_points > 18.0) slippage_points = 18.0;
   return slippage_points;
}

double FXAI_ExecutionFillPenaltyPoints(const FXAIExecutionProfile &profile,
                                       const double roundtrip_cost_points,
                                       const double spread_stress,
                                       const int path_flags)
{
   double stress = FXAI_Clamp(spread_stress, 0.0, 4.0);
   double fill_penalty = MathMax(profile.fill_penalty_points, 0.0) +
                         MathMax(profile.partial_fill_penalty, 0.0) * (0.20 + 0.20 * stress);
   if((path_flags & 4) != 0)
      fill_penalty += 0.10 * MathMax(roundtrip_cost_points, 0.0) + 0.12 * stress;
   if((path_flags & 1) != 0)
      fill_penalty += 0.08 * MathMax(roundtrip_cost_points, 0.0);
   if(fill_penalty > 10.0) fill_penalty = 10.0;
   return fill_penalty;
}

double FXAI_ExecutionFillPenaltyPointsReplay(const FXAIExecutionProfile &profile,
                                             const FXAIExecutionReplayFrame &frame,
                                             const double roundtrip_cost_points,
                                             const double spread_stress,
                                             const int path_flags)
{
   double fill_penalty = FXAI_ExecutionFillPenaltyPoints(profile,
                                                         roundtrip_cost_points,
                                                         spread_stress,
                                                         path_flags | frame.event_flags);
   fill_penalty *= FXAI_Clamp(frame.fill_mult, 0.50, 3.00);
   fill_penalty += MathMax(profile.partial_fill_penalty, 0.0) * FXAI_Clamp(frame.partial_fill_prob, 0.0, 1.0);
   fill_penalty += 0.50 * MathMax(frame.drift_penalty_points, 0.0);
   if(fill_penalty > 15.0) fill_penalty = 15.0;
   return fill_penalty;
}

double FXAI_ExecutionAllowedDeviationPoints(const FXAIExecutionProfile &profile,
                                            const double path_risk,
                                            const double fill_risk)
{
   double dev = MathMax(profile.allowed_deviation_points, 0.0) +
                2.5 * FXAI_Clamp(path_risk, 0.0, 1.0) +
                3.0 * FXAI_Clamp(fill_risk, 0.0, 1.0);
   if(dev > 25.0) dev = 25.0;
   return dev;
}

