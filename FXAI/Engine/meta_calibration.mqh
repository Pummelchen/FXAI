#ifndef __FXAI_META_CALIBRATION_MQH__
#define __FXAI_META_CALIBRATION_MQH__

void FXAI_ResetRegimeCalibration()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SELL] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_BUY] = 1.0;
         g_regime_class_mass[ai][r][(int)FXAI_LABEL_SKIP] = 1.2;
         g_regime_total[ai][r] = 3.2;
      }
   }
   g_regime_spread_ema = 0.0;
   g_regime_vol_ema = 0.0;
   g_regime_ema_ready = false;
}

void FXAI_UpdateRegimeEMAs(const double spread_points, const double vol_proxy_abs)
{
   double sp = MathMax(0.0, spread_points);
   double vp = MathMax(0.0, vol_proxy_abs);
   if(!g_regime_ema_ready)
   {
      g_regime_spread_ema = sp;
      g_regime_vol_ema = vp;
      g_regime_ema_ready = true;
      return;
   }

   g_regime_spread_ema = (0.98 * g_regime_spread_ema) + (0.02 * sp);
   g_regime_vol_ema = (0.98 * g_regime_vol_ema) + (0.02 * vp);
}

int FXAI_GetRegimeId(const datetime sample_time,
                     const double spread_points,
                     const double vol_proxy_abs)
{
   MqlDateTime dt;
   TimeToStruct(sample_time > 0 ? sample_time : TimeCurrent(), dt);
   int hour = dt.hour;
   if(hour < 0) hour = 0;
   if(hour > 23) hour = 23;

   int sess = 0;
   if(hour < 8) sess = 0;
   else if(hour < 16) sess = 1;
   else sess = 2;

   double sp_ref = (g_regime_ema_ready ? MathMax(g_regime_spread_ema, 0.10) : MathMax(spread_points, 0.10));
   double vp_ref = (g_regime_ema_ready ? MathMax(g_regime_vol_ema, 1e-6) : MathMax(MathAbs(vol_proxy_abs), 1e-6));

   int spread_hi = (spread_points > (1.15 * sp_ref + 0.10) ? 1 : 0);
   int vol_hi = (MathAbs(vol_proxy_abs) > (1.15 * vp_ref + 0.02) ? 1 : 0);

   int regime = sess * 4 + vol_hi * 2 + spread_hi;
   if(regime < 0) regime = 0;
   if(regime >= FXAI_REGIME_COUNT) regime = FXAI_REGIME_COUNT - 1;
   return regime;
}

void FXAI_ApplyRegimeCalibration(const int ai_idx, const int regime_id, double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;

   double total = g_regime_total[ai_idx][regime_id];
   if(total < 8.0) return;

   double prior[3];
   double prior_total = 0.0;
   for(int c=0; c<3; c++)
      prior_total += g_regime_class_mass[ai_idx][regime_id][c];
   if(prior_total <= 1e-9) return;
   for(int c=0; c<3; c++)
      prior[c] = g_regime_class_mass[ai_idx][regime_id][c] / prior_total;

   double strength = FXAI_Clamp((total - 8.0) / 220.0, 0.0, 0.45);
   double s = 0.0;
   for(int c=0; c<3; c++)
   {
      probs[c] = FXAI_Clamp(((1.0 - strength) * probs[c]) + (strength * prior[c]), 0.0005, 0.9990);
      s += probs[c];
   }
   if(s <= 0.0) s = 1.0;
   for(int c=0; c<3; c++) probs[c] /= s;
}

void FXAI_UpdateRegimeCalibration(const int ai_idx,
                                  const int regime_id,
                                  const int label_class,
                                  const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(regime_id < 0 || regime_id >= FXAI_REGIME_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double p_true = FXAI_Clamp(probs[label_class], 0.0, 1.0);
   double w = 1.0 + (0.5 * p_true);
   g_regime_class_mass[ai_idx][regime_id][label_class] += w;
   g_regime_total[ai_idx][regime_id] += w;

   if(g_regime_total[ai_idx][regime_id] > 20000.0)
   {
      for(int c=0; c<3; c++)
         g_regime_class_mass[ai_idx][regime_id][c] *= 0.5;
      g_regime_total[ai_idx][regime_id] *= 0.5;
   }
}

void FXAI_ResetModelPerformanceState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   g_model_meta_weight[ai_idx] = 1.0;
   g_model_global_edge_ema[ai_idx] = 0.0;
   g_model_global_edge_ready[ai_idx] = false;
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_model_regime_edge_ema[ai_idx][r] = 0.0;
      g_model_regime_edge_ready[ai_idx][r] = false;
      g_model_regime_obs[ai_idx][r] = 0;
   }
}

double FXAI_GetModelRegimeEdge(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT && g_model_regime_edge_ready[ai_idx][regime_id])
      return g_model_regime_edge_ema[ai_idx][regime_id];
   if(g_model_global_edge_ready[ai_idx]) return g_model_global_edge_ema[ai_idx];
   return 0.0;
}

double FXAI_GetHorizonRegimeEdge(const int regime_id, const int horizon_minutes)
{
   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT &&
      hslot >= 0 && hslot < FXAI_MAX_HORIZONS &&
      g_horizon_regime_edge_ready[regime_id][hslot])
   {
      return g_horizon_regime_edge_ema[regime_id][hslot];
   }
   return 0.0;
}

bool FXAI_IsModelPruned(const int ai_idx, const int regime_id)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return true;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.0, 3.0);
   if(rel < 0.30) return true;

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      int obs = g_model_regime_obs[ai_idx][regime_id];
      if(obs >= 24)
      {
         double edge = g_model_regime_edge_ema[ai_idx][regime_id];
         if(edge < -0.35) return true;
      }
   }

   if(g_model_global_edge_ready[ai_idx] && g_model_global_edge_ema[ai_idx] < -0.45)
      return true;
   return false;
}

double FXAI_GetModelMetaScore(const int ai_idx,
                              const int regime_id,
                              const double min_move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return 0.0;
   double rel = FXAI_Clamp(g_model_reliability[ai_idx], 0.20, 3.00);
   double meta = FXAI_Clamp(g_model_meta_weight[ai_idx], 0.20, 3.00);
   double edge = FXAI_GetModelRegimeEdge(ai_idx, regime_id);
   double edge_scale = 1.0 + FXAI_Clamp(edge / MathMax(min_move_points, 0.50), -0.70, 1.20);
   if(edge_scale < 0.15) edge_scale = 0.15;
   return rel * meta * edge_scale;
}

void FXAI_UpdateModelPerformance(const int ai_idx,
                                 const int regime_id,
                                 const int label_class,
                                 const int signal,
                                 const double realized_move_points,
                                 const double min_move_points,
                                 const int horizon_minutes,
                                 const double spread_stress,
                                 const int path_flags,
                                 const double expected_move_points,
                                 const double &probs[])
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;
   if(label_class < 0 || label_class > 2) return;

   double min_mv = MathMax(min_move_points, 0.10);
   double realized_net = 0.0;
   double opportunity_penalty = 0.0;
   if(signal == 0 || signal == 1)
   {
      realized_net = FXAI_RealizedNetPointsForSignal(signal,
                                                     realized_move_points,
                                                     min_mv,
                                                     horizon_minutes,
                                                     spread_stress,
                                                     path_flags);
   }
   else
   {
      if(label_class == (int)FXAI_LABEL_SKIP) realized_net = 0.05 * min_mv;
      else
      {
         opportunity_penalty = FXAI_Clamp((MathAbs(realized_move_points) - min_mv), 0.0, 8.0 * min_mv);
         realized_net = -0.25 * opportunity_penalty;
      }
   }

   double pred_edge = 0.0;
   if(signal == 1)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_BUY]) - 1.0) * expected_move_points - min_mv;
   else if(signal == 0)
      pred_edge = ((2.0 * probs[(int)FXAI_LABEL_SELL]) - 1.0) * expected_move_points - min_mv;

   double err = FXAI_Clamp(realized_net - pred_edge, -8.0 * min_mv, 8.0 * min_mv);
   double alpha = 0.04;

   if(!g_model_global_edge_ready[ai_idx])
   {
      g_model_global_edge_ema[ai_idx] = realized_net;
      g_model_global_edge_ready[ai_idx] = true;
   }
   else
   {
      g_model_global_edge_ema[ai_idx] = (1.0 - alpha) * g_model_global_edge_ema[ai_idx] + alpha * realized_net;
   }

   if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
   {
      if(!g_model_regime_edge_ready[ai_idx][regime_id])
      {
         g_model_regime_edge_ema[ai_idx][regime_id] = realized_net;
         g_model_regime_edge_ready[ai_idx][regime_id] = true;
      }
      else
      {
         g_model_regime_edge_ema[ai_idx][regime_id] =
            (1.0 - alpha) * g_model_regime_edge_ema[ai_idx][regime_id] + alpha * realized_net;
      }
      g_model_regime_obs[ai_idx][regime_id]++;
      if(g_model_regime_obs[ai_idx][regime_id] > 200000)
         g_model_regime_obs[ai_idx][regime_id] = 200000;
   }

   int hslot = FXAI_GetHorizonSlot(horizon_minutes);
   if(hslot >= 0 && hslot < FXAI_MAX_HORIZONS)
   {
      if(!g_model_horizon_edge_ready[ai_idx][hslot])
      {
         g_model_horizon_edge_ema[ai_idx][hslot] = realized_net;
         g_model_horizon_edge_ready[ai_idx][hslot] = true;
      }
      else
      {
         g_model_horizon_edge_ema[ai_idx][hslot] =
            (1.0 - alpha) * g_model_horizon_edge_ema[ai_idx][hslot] + alpha * realized_net;
      }
      g_model_horizon_obs[ai_idx][hslot]++;
      if(g_model_horizon_obs[ai_idx][hslot] > 200000)
         g_model_horizon_obs[ai_idx][hslot] = 200000;

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         if(!g_horizon_regime_edge_ready[regime_id][hslot])
         {
            g_horizon_regime_edge_ema[regime_id][hslot] = realized_net;
            g_horizon_regime_edge_ready[regime_id][hslot] = true;
         }
         else
         {
            g_horizon_regime_edge_ema[regime_id][hslot] =
               (1.0 - alpha) * g_horizon_regime_edge_ema[regime_id][hslot] + alpha * realized_net;
         }
         g_horizon_regime_obs[regime_id][hslot]++;
         if(g_horizon_regime_obs[regime_id][hslot] > 200000)
            g_horizon_regime_obs[regime_id][hslot] = 200000;
         g_horizon_regime_total_obs[regime_id] += 1.0;
         if(g_horizon_regime_total_obs[regime_id] > 1e9)
            g_horizon_regime_total_obs[regime_id] = 1e9;
      }
   }

   // Online utility-driven threshold adaptation (model + regime + horizon aware).
   if(signal == 1 || signal == 0)
   {
      double utility = FXAI_Clamp(realized_net / min_mv, -2.5, 2.5);
      double mag = MathAbs(utility);
      double step = 0.004 + (0.004 * mag);
      step = FXAI_Clamp(step, 0.002, 0.02);

      double buy_u = g_model_buy_thr[ai_idx];
      double sell_u = g_model_sell_thr[ai_idx];
      if(signal == 1)
      {
         if(utility >= 0.0) buy_u -= step * MathMin(utility, 1.5) * 0.8;
         else               buy_u += step * mag;
      }
      else
      {
         // sell_thr is inverse-coded: higher sell_thr means looser sell gate.
         if(utility >= 0.0) sell_u += step * MathMin(utility, 1.5) * 0.8;
         else               sell_u -= step * mag;
      }
      g_model_buy_thr[ai_idx] = FXAI_Clamp(buy_u, 0.50, 0.95);
      g_model_sell_thr[ai_idx] = FXAI_Clamp(sell_u, 0.05, 0.50);
      FXAI_SanitizeThresholdPair(g_model_buy_thr[ai_idx], g_model_sell_thr[ai_idx]);
      g_model_thr_ready[ai_idx] = true;

      int hslot_thr = FXAI_GetHorizonSlot(horizon_minutes);
      if(hslot_thr >= 0 && hslot_thr < FXAI_MAX_HORIZONS)
      {
         double bh = g_model_buy_thr_horizon[ai_idx][hslot_thr];
         double sh = g_model_sell_thr_horizon[ai_idx][hslot_thr];
         if(!g_model_thr_horizon_ready[ai_idx][hslot_thr])
         {
            bh = g_model_buy_thr[ai_idx];
            sh = g_model_sell_thr[ai_idx];
            g_model_thr_horizon_ready[ai_idx][hslot_thr] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) bh -= step * MathMin(utility, 1.5);
            else               bh += step * mag;
         }
         else
         {
            if(utility >= 0.0) sh += step * MathMin(utility, 1.5);
            else               sh -= step * mag;
         }
         g_model_buy_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(bh, 0.50, 0.95);
         g_model_sell_thr_horizon[ai_idx][hslot_thr] = FXAI_Clamp(sh, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_horizon[ai_idx][hslot_thr],
                                    g_model_sell_thr_horizon[ai_idx][hslot_thr]);
      }

      if(regime_id >= 0 && regime_id < FXAI_REGIME_COUNT)
      {
         double br = g_model_buy_thr_regime[ai_idx][regime_id];
         double sr = g_model_sell_thr_regime[ai_idx][regime_id];
         if(!g_model_thr_regime_ready[ai_idx][regime_id])
         {
            br = g_model_buy_thr[ai_idx];
            sr = g_model_sell_thr[ai_idx];
            g_model_thr_regime_ready[ai_idx][regime_id] = true;
         }
         if(signal == 1)
         {
            if(utility >= 0.0) br -= step * MathMin(utility, 1.5);
            else               br += step * mag;
         }
         else
         {
            if(utility >= 0.0) sr += step * MathMin(utility, 1.5);
            else               sr -= step * mag;
         }
         g_model_buy_thr_regime[ai_idx][regime_id] = FXAI_Clamp(br, 0.50, 0.95);
         g_model_sell_thr_regime[ai_idx][regime_id] = FXAI_Clamp(sr, 0.05, 0.50);
         FXAI_SanitizeThresholdPair(g_model_buy_thr_regime[ai_idx][regime_id],
                                    g_model_sell_thr_regime[ai_idx][regime_id]);

         int hslot_bank = FXAI_GetHorizonSlot(horizon_minutes);
         if(hslot_bank >= 0 && hslot_bank < FXAI_MAX_HORIZONS)
         {
            double bb = g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank];
            double sb = g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank];
            if(!g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank])
            {
               bb = g_model_buy_thr_regime[ai_idx][regime_id];
               sb = g_model_sell_thr_regime[ai_idx][regime_id];
               g_model_thr_bank_ready[ai_idx][regime_id][hslot_bank] = true;
            }
            if(signal == 1)
            {
               if(utility >= 0.0) bb -= step * MathMin(utility, 1.5);
               else               bb += step * mag;
            }
            else
            {
               if(utility >= 0.0) sb += step * MathMin(utility, 1.5);
               else               sb -= step * mag;
            }
            g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(bb, 0.50, 0.95);
            g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank] = FXAI_Clamp(sb, 0.05, 0.50);
            FXAI_SanitizeThresholdPair(g_model_buy_thr_bank[ai_idx][regime_id][hslot_bank],
                                       g_model_sell_thr_bank[ai_idx][regime_id][hslot_bank]);
         }
      }
   }

   double grad = FXAI_Clamp(err / min_mv, -2.0, 2.0);
   double lr = 0.015;
   double meta_new = g_model_meta_weight[ai_idx] + lr * grad;
   g_model_meta_weight[ai_idx] = FXAI_Clamp(meta_new, 0.20, 3.00);
}

void FXAI_ResetModelAuxState(const int ai_idx)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   g_model_reliability[ai_idx] = 1.0;
   g_model_abs_move_ema[ai_idx] = 0.0;
   g_model_abs_move_ready[ai_idx] = false;
   FXAI_ResetModelPerformanceState(ai_idx);
}

void FXAI_UpdateModelMoveStats(const int ai_idx, const double move_points)
{
   if(ai_idx < 0 || ai_idx >= FXAI_AI_COUNT) return;

   double abs_move = MathAbs(move_points);
   if(!MathIsValidNumber(abs_move) || abs_move <= 0.0) return;

   if(!g_model_abs_move_ready[ai_idx])
   {
      g_model_abs_move_ema[ai_idx] = abs_move;
      g_model_abs_move_ready[ai_idx] = true;
      return;
   }

   g_model_abs_move_ema[ai_idx] = (0.95 * g_model_abs_move_ema[ai_idx]) + (0.05 * abs_move);
}

double FXAI_GetModelExpectedMove(const int ai_idx, const double fallback_move)
{
   if(ai_idx >= 0 && ai_idx < FXAI_AI_COUNT && g_model_abs_move_ready[ai_idx] && g_model_abs_move_ema[ai_idx] > 0.0)
      return g_model_abs_move_ema[ai_idx];
   return fallback_move;
}


#endif // __FXAI_META_CALIBRATION_MQH__
