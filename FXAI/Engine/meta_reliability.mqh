#ifndef __FXAI_META_RELIABILITY_MQH__
#define __FXAI_META_RELIABILITY_MQH__

void FXAI_UpdateModelReliability(const int ai_idx,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   int best = 0;
   for(int c=1; c<3; c++)
      if(probs[c] > probs[best]) best = c;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double min_mv = MathMax(min_move_points, 0.10);
   double target = 1.0;

   if(signal == 1 || signal == 0)
   {
      double net_points = (signal == 1 ? realized_move_points : -realized_move_points) - min_mv;
      double edge_norm = FXAI_Clamp(net_points / min_mv, -2.5, 2.5);
      int pred_class = (signal == 1 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
      double cls_bonus = (pred_class == label_class ? 0.20 : -0.20);
      if(label_class == (int)FXAI_LABEL_SKIP) cls_bonus -= 0.15;
      double exp_mv = MathMax(expected_move_points, min_mv);
      double exp_fit = 1.0 - FXAI_Clamp(MathAbs(MathAbs(realized_move_points) - exp_mv) / MathMax(exp_mv, 0.10), 0.0, 1.5);
      target = 1.0 + (0.35 * edge_norm) + cls_bonus + (0.10 * (p_true - 0.5) * 2.0) + (0.08 * exp_fit);
   }
   else
   {
      // Abstention-aware: reward correct skips, penalize missed opportunities.
      if(label_class == (int)FXAI_LABEL_SKIP)
      {
         target = 1.10 + (0.10 * p_true);
      }
      else
      {
         double opportunity = FXAI_Clamp((MathAbs(realized_move_points) - min_mv) / min_mv, 0.0, 3.0);
         target = 0.95 - (0.20 * opportunity);
      }
   }

   if(best == label_class) target += 0.05;
   target = FXAI_Clamp(target, 0.20, 2.80);
   g_model_reliability[ai_idx] = FXAI_Clamp((0.97 * g_model_reliability[ai_idx]) + (0.03 * target), 0.20, 3.00);
}

void FXAI_ResetReliabilityPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_rel_pending_head[ai] = 0;
      g_rel_pending_tail[ai] = 0;
      for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
      {
         g_rel_pending_seq[ai][k] = -1;
         g_rel_pending_signal[ai][k] = -1;
         g_rel_pending_regime[ai][k] = -1;
         g_rel_pending_expected_move[ai][k] = 0.0;
         g_rel_pending_horizon[ai][k] = default_h;
      }
   }
   g_rel_clock_bar_time = 0;
   g_rel_clock_seq = 0;
}

void FXAI_AdvanceReliabilityClock(const datetime signal_bar)
{
   if(signal_bar <= 0) return;
   if(g_rel_clock_bar_time == 0)
   {
      g_rel_clock_bar_time = signal_bar;
      g_rel_clock_seq = 0;
      return;
   }

   if(signal_bar != g_rel_clock_bar_time)
   {
      g_rel_clock_seq++;
      g_rel_clock_bar_time = signal_bar;
   }
}

void FXAI_EnqueueReliabilityPending(const int ai_idx,
                                    const int signal_seq,
                                    const int signal,
                                    const int regime_id,
                                    const double expected_move_points,
                                    const int horizon_minutes,
                                    const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;
   if(head != tail && g_rel_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_rel_pending_prob[ai_idx][prev][0] = probs[0];
      g_rel_pending_prob[ai_idx][prev][1] = probs[1];
      g_rel_pending_prob[ai_idx][prev][2] = probs[2];
      g_rel_pending_signal[ai_idx][prev] = signal;
      g_rel_pending_regime[ai_idx][prev] = regime_id;
      g_rel_pending_expected_move[ai_idx][prev] = expected_move_points;
      g_rel_pending_horizon[ai_idx][prev] = h;
      return;
   }

   g_rel_pending_seq[ai_idx][tail] = signal_seq;
   g_rel_pending_prob[ai_idx][tail][0] = probs[0];
   g_rel_pending_prob[ai_idx][tail][1] = probs[1];
   g_rel_pending_prob[ai_idx][tail][2] = probs[2];
   g_rel_pending_signal[ai_idx][tail] = signal;
   g_rel_pending_regime[ai_idx][tail] = regime_id;
   g_rel_pending_expected_move[ai_idx][tail] = expected_move_points;
   g_rel_pending_horizon[ai_idx][tail] = h;

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_rel_pending_head[ai_idx] = head;
   }
   g_rel_pending_tail[ai_idx] = next_tail;
}

void FXAI_UpdateReliabilityFromPending(const int ai_idx,
                                      const int current_signal_seq,
                                      const FXAIDataSnapshot &snapshot,
                                      const int &spread_m1[],
                                      const datetime &time_arr[],
                                      const double &high_arr[],
                                      const double &low_arr[],
                                      const double &close_arr[],
                                      const double commission_points,
                                      const double cost_buffer_points,
                                      const double ev_threshold_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(current_signal_seq < 0) return;

   int head = g_rel_pending_head[ai_idx];
   int tail = g_rel_pending_tail[ai_idx];
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_rel_pending_seq[ai_idx][idx];
      int pending_signal = g_rel_pending_signal[ai_idx][idx];
      int pending_regime = g_rel_pending_regime[ai_idx][idx];
      double pending_expected_move = g_rel_pending_expected_move[ai_idx][idx];
      int pending_h = FXAI_ClampHorizon(g_rel_pending_horizon[ai_idx][idx]);

      double p0 = g_rel_pending_prob[ai_idx][idx][0];
      double p1 = g_rel_pending_prob[ai_idx][idx][1];
      double p2 = g_rel_pending_prob[ai_idx][idx][2];

      bool consumed = false;
      if(seq_pred < 0)
      {
         consumed = true;
      }
      else
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            int idx_future = age - pending_h;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(time_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr) &&
               idx_future >= 0 && idx_future < ArraySize(close_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               int label_class = FXAI_BuildTripleBarrierLabel(idx_pred,
                                                              pending_h,
                                                              min_move_i,
                                                              ev_threshold_points,
                                                              snapshot,
                                                              high_arr,
                                                              low_arr,
                                                              close_arr,
                                                              move_points);

               double probs_eval[3];
               probs_eval[0] = p0;
               probs_eval[1] = p1;
               probs_eval[2] = p2;

               FXAI_UpdateModelReliability(ai_idx,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_UpdateRegimeCalibration(ai_idx, pending_regime, label_class, probs_eval);
               FXAI_UpdateModelPerformance(ai_idx,
                                           pending_regime,
                                           label_class,
                                           pending_signal,
                                           move_points,
                                           min_move_i,
                                           pending_h,
                                           pending_expected_move,
                                           probs_eval);
               FXAI_BoostReplayPriorityByOutcome(time_arr[idx_pred],
                                                pending_h,
                                                pending_regime,
                                                label_class,
                                                pending_signal,
                                                move_points,
                                                min_move_i);
            }
            consumed = true;
         }
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_signal, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_expected, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);

            keep_seq[ks] = seq_pred;
            keep_signal[ks] = pending_signal;
            keep_regime[ks] = pending_regime;
            keep_horizon[ks] = pending_h;
            keep_expected[ks] = pending_expected_move;
            keep_prob0[ks] = p0;
            keep_prob1[ks] = p1;
            keep_prob2[ks] = p2;
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   int keep_n = ArraySize(keep_seq);
   int queue_cap = FXAI_REL_MAX_PENDING - 1;
   if(queue_cap < 0) queue_cap = 0;
   if(keep_n > queue_cap) keep_n = queue_cap;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_rel_pending_seq[ai_idx][k] = -1;
      g_rel_pending_signal[ai_idx][k] = -1;
      g_rel_pending_regime[ai_idx][k] = -1;
      g_rel_pending_expected_move[ai_idx][k] = 0.0;
      g_rel_pending_horizon[ai_idx][k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_rel_pending_prob[ai_idx][k][0] = 0.0;
      g_rel_pending_prob[ai_idx][k][1] = 0.0;
      g_rel_pending_prob[ai_idx][k][2] = 0.0;
   }

   for(int k=0; k<keep_n; k++)
   {
      g_rel_pending_seq[ai_idx][k] = keep_seq[k];
      g_rel_pending_signal[ai_idx][k] = keep_signal[k];
      g_rel_pending_regime[ai_idx][k] = keep_regime[k];
      g_rel_pending_expected_move[ai_idx][k] = keep_expected[k];
      g_rel_pending_horizon[ai_idx][k] = keep_horizon[k];
      g_rel_pending_prob[ai_idx][k][0] = keep_prob0[k];
      g_rel_pending_prob[ai_idx][k][1] = keep_prob1[k];
      g_rel_pending_prob[ai_idx][k][2] = keep_prob2[k];
   }

   g_rel_pending_head[ai_idx] = 0;
   g_rel_pending_tail[ai_idx] = keep_n;
}

int FXAI_GetMaxPendingHorizon(const int fallback_h)
{
   int hmax = FXAI_ClampHorizon(fallback_h);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      int head = g_rel_pending_head[ai];
      int tail = g_rel_pending_tail[ai];
      int idx = head;
      while(idx != tail)
      {
         int seq_pred = g_rel_pending_seq[ai][idx];
         if(seq_pred >= 0)
         {
            int h = FXAI_ClampHorizon(g_rel_pending_horizon[ai][idx]);
            if(h > hmax) hmax = h;
         }
         idx++;
         if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
      }
   }

   int idx = g_stack_pending_head;
   while(idx != g_stack_pending_tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   idx = g_hpolicy_pending_head;
   while(idx != g_hpolicy_pending_tail)
   {
      int seq_pred = g_hpolicy_pending_seq[idx];
      if(seq_pred >= 0)
      {
         int h = FXAI_ClampHorizon(g_hpolicy_pending_horizon[idx]);
         if(h > hmax) hmax = h;
      }
      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }
   return hmax;
}

void FXAI_ProcessReliabilityBar(const string symbol)
{
   if(StringLen(symbol) <= 0) return;

   int H = FXAI_ClampHorizon(PredictionTargetMinutes);
   H = FXAI_GetMaxConfiguredHorizon(H);
   H = FXAI_GetMaxPendingHorizon(H);

   datetime signal_bar = iTime(symbol, PERIOD_M1, 1);
   if(signal_bar <= 0) return;

   static string rel_symbol = "";
   static datetime rel_last_processed_bar = 0;
   static datetime rel_last_rates_bar = 0;
   static MqlRates rel_rates_m1[];
   static double rel_open_arr[];
   static double rel_high_arr[];
   static double rel_low_arr[];
   static double rel_close_arr[];
   static datetime rel_time_arr[];
   static int rel_spread_arr[];

   if(rel_symbol != symbol)
   {
      rel_symbol = symbol;
      rel_last_processed_bar = 0;
      rel_last_rates_bar = 0;
      ArrayResize(rel_rates_m1, 0);
      ArrayResize(rel_open_arr, 0);
      ArrayResize(rel_high_arr, 0);
      ArrayResize(rel_low_arr, 0);
      ArrayResize(rel_close_arr, 0);
      ArrayResize(rel_time_arr, 0);
      ArrayResize(rel_spread_arr, 0);
   }

   FXAI_AdvanceReliabilityClock(signal_bar);
   if(signal_bar == rel_last_processed_bar) return;
   rel_last_processed_bar = signal_bar;

   int needed = H + 64;
   if(needed < 128) needed = 128;
   if(needed > 1500) needed = 1500;

   if(!FXAI_UpdateRatesRolling(symbol, PERIOD_M1, needed, rel_last_rates_bar, rel_rates_m1))
      return;

   FXAI_ExtractRatesCloseTimeSpread(rel_rates_m1, rel_close_arr, rel_time_arr, rel_spread_arr);
   FXAI_ExtractRatesOHLC(rel_rates_m1, rel_open_arr, rel_high_arr, rel_low_arr, rel_close_arr);
   if(ArraySize(rel_close_arr) <= H || ArraySize(rel_spread_arr) <= H)
      return;

   FXAIDataSnapshot snapshot;
   if(!FXAI_ExportDataSnapshot(symbol, AI_CommissionPerLotSide, AI_CostBufferPoints, snapshot))
      return;
   snapshot.bar_time = signal_bar;

   double cost_buffer_points = (AI_CostBufferPoints < 0.0 ? 0.0 : AI_CostBufferPoints);
   double commission_points = snapshot.commission_points;
   double evThresholdPoints = FXAI_Clamp(AI_EVThresholdPoints, 0.0, 100.0);
   int signal_seq = g_rel_clock_seq;

   for(int ai_idx=0; ai_idx<FXAI_AI_COUNT; ai_idx++)
   {
      FXAI_UpdateReliabilityFromPending(ai_idx,
                                       signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_time_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
   }

   FXAI_UpdateStackFromPending(signal_seq,
                               snapshot,
                               rel_spread_arr,
                               rel_high_arr,
                               rel_low_arr,
                               rel_close_arr,
                               commission_points,
                               cost_buffer_points,
                               evThresholdPoints);
   FXAI_UpdateHorizonPolicyFromPending(signal_seq,
                                       snapshot,
                                       rel_spread_arr,
                                       rel_high_arr,
                                       rel_low_arr,
                                       rel_close_arr,
                                       commission_points,
                                       cost_buffer_points,
                                       evThresholdPoints);
   FXAI_MaybeSaveMetaArtifacts(symbol, signal_bar);
}

double FXAI_GetModelVoteWeight(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 1.0;
   return FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
}


#endif // __FXAI_META_RELIABILITY_MQH__
