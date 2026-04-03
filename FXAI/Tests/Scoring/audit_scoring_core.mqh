void FXAI_AuditComparePredictions(const FXAIAIPredictionV4 &a,
                                  const FXAIAIPredictionV4 &b,
                                  double &delta_out)
{
   delta_out = 0.0;
   for(int c=0; c<3; c++)
      delta_out += MathAbs(a.class_probs[c] - b.class_probs[c]);
   delta_out += 0.10 * MathAbs(a.move_mean_points - b.move_mean_points);
   delta_out += 0.04 * MathAbs(a.mfe_mean_points - b.mfe_mean_points);
   delta_out += 0.04 * MathAbs(a.mae_mean_points - b.mae_mean_points);
   delta_out += 0.05 * MathAbs(a.hit_time_frac - b.hit_time_frac);
   delta_out += 0.05 * MathAbs(a.path_risk - b.path_risk);
   delta_out += 0.05 * MathAbs(a.fill_risk - b.fill_risk);
   delta_out += 0.05 * MathAbs(a.confidence - b.confidence);
   delta_out += 0.05 * MathAbs(a.reliability - b.reliability);
}

void FXAI_AuditResetFoldMetrics(FXAIAuditFoldMetrics &m)
{
   m.samples_total = 0;
   m.valid_preds = 0;
   m.invalid_preds = 0;
   m.buy_count = 0;
   m.sell_count = 0;
   m.skip_count = 0;
   m.directional_eval_count = 0;
   m.directional_correct_count = 0;
   m.conf_sum = 0.0;
   m.rel_sum = 0.0;
   m.move_sum = 0.0;
   m.brier_sum = 0.0;
   m.calibration_abs_sum = 0.0;
   m.path_quality_abs_sum = 0.0;
   m.path_quality_count = 0;
   m.net_sum = 0.0;
}

void FXAI_AuditFoldInvalid(FXAIAuditFoldMetrics &m)
{
   m.samples_total++;
   m.invalid_preds++;
}

void FXAI_AuditFoldValid(FXAIAuditFoldMetrics &m,
                         const int decision,
                         const FXAIAIPredictionV4 &pred,
                         const double brier,
                         const double net_points,
                         const bool directional_eval,
                         const bool directional_ok,
                         const double calibration_abs,
                         const double path_quality)
{
   m.samples_total++;
   m.valid_preds++;
   if(decision == (int)FXAI_LABEL_BUY) m.buy_count++;
   else if(decision == (int)FXAI_LABEL_SELL) m.sell_count++;
   else m.skip_count++;

   m.conf_sum += pred.confidence;
   m.rel_sum += pred.reliability;
   m.move_sum += pred.move_mean_points;
   m.brier_sum += brier;
   m.net_sum += net_points;
   if(directional_eval)
   {
      m.directional_eval_count++;
      if(directional_ok) m.directional_correct_count++;
      m.calibration_abs_sum += calibration_abs;
   }
   if(path_quality >= 0.0)
   {
      m.path_quality_abs_sum += path_quality;
      m.path_quality_count++;
   }
}

double FXAI_AuditSessionEdgePressure(const datetime sample_time)
{
   MqlDateTime dt;
   TimeToStruct(sample_time, dt);
   double hour = (double)dt.hour + ((double)dt.min / 60.0);
   double dist_tokyo = MathMin(MathAbs(hour - 0.0), MathAbs(hour - 24.0));
   double dist_london = MathAbs(hour - 8.0);
   double dist_newyork = MathAbs(hour - 16.0);
   double best = MathMin(dist_tokyo, MathMin(dist_london, dist_newyork));
   return FXAI_Clamp(1.0 - best / 4.0, 0.0, 1.0);
}

double FXAI_AuditAdversarialWeaknessScore(const int label_class,
                                          const double move_points,
                                          const double min_move_points,
                                          const double mfe_points,
                                          const double mae_points,
                                          const double time_to_hit_frac,
                                          const int path_flags,
                                          const double spread_stress,
                                          const double macro_activity,
                                          const datetime sample_time,
                                          const FXAIAIPredictionV4 &pred)
{
   double target_probs[3] = {0.0, 0.0, 0.0};
   int cls_idx = label_class;
   if(cls_idx < (int)FXAI_LABEL_SELL || cls_idx > (int)FXAI_LABEL_SKIP)
      cls_idx = (move_points >= 0.0 ? (int)FXAI_LABEL_BUY : (int)FXAI_LABEL_SELL);
   target_probs[cls_idx] = 1.0;

   double brier = 0.0;
   for(int c=0; c<3; c++)
   {
      double d = pred.class_probs[c] - target_probs[c];
      brier += d * d;
   }

   int decision = FXAI_AuditDecisionFromPred(pred);
   bool directional_eval = (decision != (int)FXAI_LABEL_SKIP);
   bool dir_ok = ((decision == (int)FXAI_LABEL_BUY && cls_idx == (int)FXAI_LABEL_BUY) ||
                  (decision == (int)FXAI_LABEL_SELL && cls_idx == (int)FXAI_LABEL_SELL));
   double dir_conf = MathMax(pred.class_probs[(int)FXAI_LABEL_BUY], pred.class_probs[(int)FXAI_LABEL_SELL]);
   double calibration_abs = (directional_eval ? MathAbs(dir_conf - (dir_ok ? 1.0 : 0.0)) : MathAbs(pred.class_probs[(int)FXAI_LABEL_SKIP] - target_probs[(int)FXAI_LABEL_SKIP]));
   double move_scale = MathMax(MathAbs(move_points), MathMax(MathAbs(pred.move_mean_points), MathMax(min_move_points, 0.50)));
   double path_quality = 0.25 * FXAI_Clamp(MathAbs(pred.mfe_mean_points - mfe_points) / move_scale, 0.0, 3.0) +
                         0.20 * FXAI_Clamp(MathAbs(pred.mae_mean_points - mae_points) / move_scale, 0.0, 3.0) +
                         0.20 * MathAbs(pred.hit_time_frac - time_to_hit_frac) +
                         0.20 * MathAbs(pred.path_risk - spread_stress) +
                         0.15 * MathAbs(pred.fill_risk - FXAI_Clamp(spread_stress + (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 0.25 : 0.0), 0.0, 1.0));

   double wrong_dir = (directional_eval && !dir_ok ? 1.0 : 0.0);
   double noise_overtrade = ((cls_idx == (int)FXAI_LABEL_SKIP) && directional_eval ? dir_conf : 0.0);
   double missed_trade = ((cls_idx != (int)FXAI_LABEL_SKIP) && decision == (int)FXAI_LABEL_SKIP ? 1.0 - pred.class_probs[(int)FXAI_LABEL_SKIP] : 0.0);
   double stress = 0.18 * FXAI_Clamp(spread_stress, 0.0, 4.0) +
                   0.10 * (((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) ? 1.0 : 0.0) +
                   0.08 * (((path_flags & FXAI_PATHFLAG_SLOW_HIT) != 0) ? 1.0 : 0.0) +
                   0.08 * FXAI_Clamp(macro_activity, 0.0, 1.0) +
                   0.08 * FXAI_AuditSessionEdgePressure(sample_time) +
                   0.10 * FXAI_Clamp(MathAbs(move_points) / MathMax(min_move_points, 0.50), 0.0, 4.0);

   return 0.55 * brier +
          0.40 * calibration_abs +
          0.32 * path_quality +
          0.38 * wrong_dir * dir_conf +
          0.24 * noise_overtrade +
          0.18 * missed_trade +
          stress;
}

