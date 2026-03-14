#ifndef __FXAI_META_THRESHOLDS_MQH__
#define __FXAI_META_THRESHOLDS_MQH__

void FXAI_DeriveAdaptiveThresholds(const double base_buy_threshold,
                                  const double base_sell_threshold,
                                  const double min_move_points,
                                  const double expected_move_points,
                                  const double vol_proxy,
                                  double &buy_min_prob,
                                  double &sell_min_prob,
                                  double &skip_min_prob)
{
   double buy_base = FXAI_Clamp(base_buy_threshold, 0.50, 0.95);
   double sell_base = FXAI_Clamp(1.0 - base_sell_threshold, 0.50, 0.95);

   double em = MathMax(expected_move_points, min_move_points + 0.10);
   double cost_ratio = FXAI_Clamp(min_move_points / em, 0.0, 2.0);
   double vol_ratio = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);

   double tighten = FXAI_Clamp(((cost_ratio - 0.35) * 0.35) + (0.10 * vol_ratio), 0.0, 0.25);

   buy_min_prob = FXAI_Clamp(buy_base + tighten, 0.50, 0.96);
   sell_min_prob = FXAI_Clamp(sell_base + tighten, 0.50, 0.96);
   skip_min_prob = FXAI_Clamp(0.45 + (0.20 * cost_ratio) + (0.10 * vol_ratio), 0.35, 0.85);
}

int FXAI_ClassSignalFromEV(const double &probs[],
                          const double buy_min_prob,
                          const double sell_min_prob,
                          const double skip_min_prob,
                          const double expected_move_points,
                          const double min_move_points,
                          const double ev_threshold_points)
{
   if(expected_move_points <= 0.0) return -1;

   double p_sell = probs[(int)FXAI_LABEL_SELL];
   double p_buy = probs[(int)FXAI_LABEL_BUY];
   double p_skip = probs[(int)FXAI_LABEL_SKIP];

   if(p_skip >= skip_min_prob) return -1;

   double buy_ev = ((2.0 * p_buy) - 1.0) * expected_move_points - min_move_points;
   double sell_ev = ((2.0 * p_sell) - 1.0) * expected_move_points - min_move_points;

   if(p_buy >= buy_min_prob && buy_ev >= ev_threshold_points && buy_ev > sell_ev)
      return 1;
   if(p_sell >= sell_min_prob && sell_ev >= ev_threshold_points && sell_ev > buy_ev)
      return 0;

   return -1;
}


#endif // __FXAI_META_THRESHOLDS_MQH__
