#ifndef __FXAI_META_CONFORMAL_MQH__
#define __FXAI_META_CONFORMAL_MQH__

double FXAI_ConformalQuantile(const int ai_idx,
                              const int regime_id,
                              const int hslot,
                              const int score_kind,
                              const double fallback)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return fallback;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return fallback;
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return fallback;

   int count = g_conf_count[ai_idx][regime_id][hslot];
   if(count <= 0) return fallback;
   if(count > FXAI_CONFORMAL_DEPTH) count = FXAI_CONFORMAL_DEPTH;

   double tmp[];
   ArrayResize(tmp, count);
   for(int i=0; i<count; i++)
   {
      if(score_kind == 0)
         tmp[i] = g_conf_class_score[ai_idx][regime_id][hslot][i];
      else if(score_kind == 1)
         tmp[i] = g_conf_move_score[ai_idx][regime_id][hslot][i];
      else
         tmp[i] = g_conf_path_score[ai_idx][regime_id][hslot][i];
   }
   ArraySort(tmp);

   int qi = (int)MathFloor(0.90 * (double)(count - 1));
   if(qi < 0) qi = 0;
   if(qi >= count) qi = count - 1;
   return tmp[qi];
}

void FXAI_ResetConformalState(void)
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      g_conf_pending_head[ai] = 0;
      g_conf_pending_tail[ai] = 0;
      for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
      {
         g_conf_pending_seq[ai][k] = -1;
         g_conf_pending_regime[ai][k] = 0;
         g_conf_pending_horizon[ai][k] = default_h;
         g_conf_pending_prob[ai][k][0] = 0.0;
         g_conf_pending_prob[ai][k][1] = 0.0;
         g_conf_pending_prob[ai][k][2] = 1.0;
         g_conf_pending_move_q25[ai][k] = 0.0;
         g_conf_pending_move_q50[ai][k] = 0.0;
         g_conf_pending_move_q75[ai][k] = 0.0;
         g_conf_pending_path_risk[ai][k] = 0.5;
      }

      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         for(int h=0; h<FXAI_MAX_HORIZONS; h++)
         {
            g_conf_count[ai][r][h] = 0;
            g_conf_head[ai][r][h] = 0;
            for(int i=0; i<FXAI_CONFORMAL_DEPTH; i++)
            {
               g_conf_class_score[ai][r][h][i] = 0.35;
               g_conf_move_score[ai][r][h][i] = 0.20;
               g_conf_path_score[ai][r][h][i] = 0.10;
            }
         }
      }
   }
}

void FXAI_ConformalPushScore(const int ai_idx,
                             const int regime_id,
                             const int hslot,
                             const double class_score,
                             const double move_score,
                             const double path_score)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   int r = regime_id;
   if(r < 0) r = 0;
   if(r >= FXAI_REGIME_COUNT) r = FXAI_REGIME_COUNT - 1;
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;

   int head = g_conf_head[ai_idx][r][hslot];
   if(head < 0 || head >= FXAI_CONFORMAL_DEPTH) head = 0;
   g_conf_class_score[ai_idx][r][hslot][head] = FXAI_Clamp(class_score, 0.0, 1.0);
   g_conf_move_score[ai_idx][r][hslot][head] = FXAI_Clamp(move_score, 0.0, 6.0);
   g_conf_path_score[ai_idx][r][hslot][head] = FXAI_Clamp(path_score, 0.0, 1.0);
   head++;
   if(head >= FXAI_CONFORMAL_DEPTH) head = 0;
   g_conf_head[ai_idx][r][hslot] = head;
   if(g_conf_count[ai_idx][r][hslot] < FXAI_CONFORMAL_DEPTH)
      g_conf_count[ai_idx][r][hslot]++;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_EnqueueConformalPending(const int ai_idx,
                                  const int signal_seq,
                                  const int regime_id,
                                  const int horizon_minutes,
                                  const FXAIAIPredictionV4 &pred)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(signal_seq < 0) return;

   int h = FXAI_ClampHorizon(horizon_minutes);
   int head = g_conf_pending_head[ai_idx];
   int tail = g_conf_pending_tail[ai_idx];

   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;
   if(head != tail && g_conf_pending_seq[ai_idx][prev] == signal_seq)
   {
      g_conf_pending_regime[ai_idx][prev] = regime_id;
      g_conf_pending_horizon[ai_idx][prev] = h;
      g_conf_pending_prob[ai_idx][prev][0] = pred.class_probs[0];
      g_conf_pending_prob[ai_idx][prev][1] = pred.class_probs[1];
      g_conf_pending_prob[ai_idx][prev][2] = pred.class_probs[2];
      g_conf_pending_move_q25[ai_idx][prev] = pred.move_q25_points;
      g_conf_pending_move_q50[ai_idx][prev] = pred.move_q50_points;
      g_conf_pending_move_q75[ai_idx][prev] = pred.move_q75_points;
      g_conf_pending_path_risk[ai_idx][prev] = pred.path_risk;
      FXAI_MarkRuntimeArtifactsDirty();
      return;
   }

   g_conf_pending_seq[ai_idx][tail] = signal_seq;
   g_conf_pending_regime[ai_idx][tail] = regime_id;
   g_conf_pending_horizon[ai_idx][tail] = h;
   g_conf_pending_prob[ai_idx][tail][0] = pred.class_probs[0];
   g_conf_pending_prob[ai_idx][tail][1] = pred.class_probs[1];
   g_conf_pending_prob[ai_idx][tail][2] = pred.class_probs[2];
   g_conf_pending_move_q25[ai_idx][tail] = pred.move_q25_points;
   g_conf_pending_move_q50[ai_idx][tail] = pred.move_q50_points;
   g_conf_pending_move_q75[ai_idx][tail] = pred.move_q75_points;
   g_conf_pending_path_risk[ai_idx][tail] = pred.path_risk;

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_conf_pending_head[ai_idx] = head;
   }
   g_conf_pending_tail[ai_idx] = next_tail;
   FXAI_MarkRuntimeArtifactsDirty();
}

void FXAI_UpdateConformalFromPending(const int ai_idx,
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

   int head = g_conf_pending_head[ai_idx];
   int tail = g_conf_pending_tail[ai_idx];
   if(head == tail) return;

   int keep_seq[];
   int keep_regime[];
   int keep_horizon[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   double keep_q25[];
   double keep_q50[];
   double keep_q75[];
   double keep_path[];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);
   ArrayResize(keep_q25, 0);
   ArrayResize(keep_q50, 0);
   ArrayResize(keep_q75, 0);
   ArrayResize(keep_path, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_conf_pending_seq[ai_idx][idx];
      int regime_id = g_conf_pending_regime[ai_idx][idx];
      int pending_h = FXAI_ClampHorizon(g_conf_pending_horizon[ai_idx][idx]);

      double p0 = g_conf_pending_prob[ai_idx][idx][0];
      double p1 = g_conf_pending_prob[ai_idx][idx][1];
      double p2 = g_conf_pending_prob[ai_idx][idx][2];
      double q25 = g_conf_pending_move_q25[ai_idx][idx];
      double q50 = g_conf_pending_move_q50[ai_idx][idx];
      double q75 = g_conf_pending_move_q75[ai_idx][idx];
      double path_pred = g_conf_pending_path_risk[ai_idx][idx];

      bool consumed = false;
      if(seq_pred >= 0)
      {
         int age = current_signal_seq - seq_pred;
         if(age >= pending_h)
         {
            int idx_pred = age;
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(time_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = spread_i + commission_points + cost_buffer_points;
               if(min_move_i < 0.0) min_move_i = 0.0;

               double move_points = 0.0;
               double mfe_points = 0.0;
               double mae_points = 0.0;
               double time_to_hit_frac = 1.0;
               int path_flags = 0;
               int label_class = FXAI_BuildTripleBarrierLabelEx(idx_pred,
                                                                pending_h,
                                                                min_move_i,
                                                                ev_threshold_points,
                                                                snapshot,
                                                                high_arr,
                                                                low_arr,
                                                                close_arr,
                                                                move_points,
                                                                mfe_points,
                                                                mae_points,
                                                                time_to_hit_frac,
                                                                path_flags);
               double probs_eval[3];
               probs_eval[0] = p0;
               probs_eval[1] = p1;
               probs_eval[2] = p2;
               double p_true = probs_eval[label_class];
               if(label_class < 0 || label_class > 2)
                  p_true = MathMax(probs_eval[(int)FXAI_LABEL_BUY], probs_eval[(int)FXAI_LABEL_SELL]);

               double realized_abs = MathAbs(move_points);
               double width = MathMax(q75 - q25, MathMax(min_move_i, 0.25));
               double move_score = MathAbs(realized_abs - MathMax(q50, 0.0)) / width;
               double spread_stress = FXAI_Clamp(spread_i / MathMax(min_move_i, 0.10), 0.0, 4.0);
               double path_actual = FXAI_PathRiskFromTargets(mfe_points,
                                                            mae_points,
                                                            min_move_i,
                                                            time_to_hit_frac,
                                                            path_flags);
               double fill_actual = FXAI_FillRiskFromTargets(spread_stress,
                                                             min_move_i,
                                                             commission_points + cost_buffer_points);
               double path_score = 0.70 * MathAbs(path_actual - path_pred) +
                                   0.30 * MathAbs(fill_actual - path_pred);
               FXAI_ConformalPushScore(ai_idx,
                                       regime_id,
                                       FXAI_GetHorizonSlot(pending_h),
                                       1.0 - FXAI_Clamp(p_true, 0.0, 1.0),
                                       FXAI_Clamp(move_score, 0.0, 6.0),
                                       FXAI_Clamp(path_score, 0.0, 1.0));
            }
            consumed = true;
         }
      }
      else
      {
         consumed = true;
      }

      if(!consumed)
      {
         int ks = ArraySize(keep_seq);
         if(ks < FXAI_REL_MAX_PENDING)
         {
            ArrayResize(keep_seq, ks + 1);
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_prob0, ks + 1);
            ArrayResize(keep_prob1, ks + 1);
            ArrayResize(keep_prob2, ks + 1);
            ArrayResize(keep_q25, ks + 1);
            ArrayResize(keep_q50, ks + 1);
            ArrayResize(keep_q75, ks + 1);
            ArrayResize(keep_path, ks + 1);
            keep_seq[ks] = seq_pred;
            keep_regime[ks] = regime_id;
            keep_horizon[ks] = pending_h;
            keep_prob0[ks] = p0;
            keep_prob1[ks] = p1;
            keep_prob2[ks] = p2;
            keep_q25[ks] = q25;
            keep_q50[ks] = q50;
            keep_q75[ks] = q75;
            keep_path[ks] = path_pred;
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_conf_pending_seq[ai_idx][k] = -1;
      g_conf_pending_regime[ai_idx][k] = 0;
      g_conf_pending_horizon[ai_idx][k] = FXAI_ClampHorizon(PredictionTargetMinutes);
      g_conf_pending_prob[ai_idx][k][0] = 0.0;
      g_conf_pending_prob[ai_idx][k][1] = 0.0;
      g_conf_pending_prob[ai_idx][k][2] = 1.0;
      g_conf_pending_move_q25[ai_idx][k] = 0.0;
      g_conf_pending_move_q50[ai_idx][k] = 0.0;
      g_conf_pending_move_q75[ai_idx][k] = 0.0;
      g_conf_pending_path_risk[ai_idx][k] = 0.5;
   }

   int keep_n = ArraySize(keep_seq);
   for(int k=0; k<keep_n; k++)
   {
      g_conf_pending_seq[ai_idx][k] = keep_seq[k];
      g_conf_pending_regime[ai_idx][k] = keep_regime[k];
      g_conf_pending_horizon[ai_idx][k] = keep_horizon[k];
      g_conf_pending_prob[ai_idx][k][0] = keep_prob0[k];
      g_conf_pending_prob[ai_idx][k][1] = keep_prob1[k];
      g_conf_pending_prob[ai_idx][k][2] = keep_prob2[k];
      g_conf_pending_move_q25[ai_idx][k] = keep_q25[k];
      g_conf_pending_move_q50[ai_idx][k] = keep_q50[k];
      g_conf_pending_move_q75[ai_idx][k] = keep_q75[k];
      g_conf_pending_path_risk[ai_idx][k] = keep_path[k];
   }
   g_conf_pending_head[ai_idx] = 0;
   g_conf_pending_tail[ai_idx] = keep_n;
}

void FXAI_ApplyConformalPredictionAdjustment(const int ai_idx,
                                             const int regime_id,
                                             const int horizon_minutes,
                                             const double min_move_points,
                                             FXAIAIPredictionV4 &pred)
{
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot < 0 || hslot >= FXAI_MAX_HORIZONS) return;
   int r = regime_id;
   if(r < 0) r = 0;
   if(r >= FXAI_REGIME_COUNT) r = FXAI_REGIME_COUNT - 1;

   double q_class = FXAI_ConformalQuantile(ai_idx, r, hslot, 0, 0.35);
   double q_move = FXAI_ConformalQuantile(ai_idx, r, hslot, 1, 0.20);
   double q_path = FXAI_ConformalQuantile(ai_idx, r, hslot, 2, 0.10);

   double uncertainty = FXAI_Clamp(q_class, 0.0, 0.55);
   double skip_boost = FXAI_Clamp(0.32 * uncertainty + 0.14 * q_path, 0.0, 0.45);
   double sell = pred.class_probs[(int)FXAI_LABEL_SELL] * (1.0 - skip_boost);
   double buy = pred.class_probs[(int)FXAI_LABEL_BUY] * (1.0 - skip_boost);
   double skip = pred.class_probs[(int)FXAI_LABEL_SKIP] + skip_boost * (1.0 - pred.class_probs[(int)FXAI_LABEL_SKIP]);
   double s = sell + buy + skip;
   if(s <= 0.0) s = 1.0;
   pred.class_probs[(int)FXAI_LABEL_SELL] = sell / s;
   pred.class_probs[(int)FXAI_LABEL_BUY] = buy / s;
   pred.class_probs[(int)FXAI_LABEL_SKIP] = skip / s;

   double move_width = MathMax(pred.move_q75_points - pred.move_q25_points,
                               MathMax(min_move_points, 0.25));
   double extra = FXAI_Clamp(q_move, 0.0, 3.0) * MathMax(0.50 * move_width, min_move_points);
   pred.move_mean_points = MathMax(0.0, pred.move_mean_points * (1.0 - 0.12 * uncertainty));
   pred.move_q25_points = MathMax(0.0, pred.move_q25_points - 0.50 * extra);
   pred.move_q50_points = MathMax(pred.move_q25_points, pred.move_q50_points);
   pred.move_q75_points = MathMax(pred.move_q50_points, pred.move_q75_points + 0.50 * extra);
   pred.path_risk = FXAI_Clamp(pred.path_risk + 0.28 * q_path + 0.10 * uncertainty, 0.0, 1.0);
   pred.fill_risk = FXAI_Clamp(pred.fill_risk + 0.18 * q_path + 0.12 * uncertainty, 0.0, 1.0);
   pred.confidence = FXAI_Clamp(MathMax(pred.class_probs[(int)FXAI_LABEL_BUY],
                                        pred.class_probs[(int)FXAI_LABEL_SELL]), 0.0, 1.0);
   pred.reliability = FXAI_Clamp(pred.reliability * (1.0 - 0.35 * uncertainty), 0.0, 1.0);
}

#endif // __FXAI_META_CONFORMAL_MQH__
