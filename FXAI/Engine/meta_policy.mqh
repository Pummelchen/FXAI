#ifndef __FXAI_META_POLICY_MQH__
#define __FXAI_META_POLICY_MQH__

#define FXAI_POLICY_FEATS 32
#define FXAI_POLICY_HIDDEN 16
#define FXAI_POLICY_ACTION_NO_TRADE 0
#define FXAI_POLICY_ACTION_ENTER 1
#define FXAI_POLICY_ACTION_HOLD 2
#define FXAI_POLICY_ACTION_EXIT 3

struct FXAIPolicyDecision
{
   double trade_prob;
   double no_trade_prob;
   double enter_prob;
   double exit_prob;
   double direction_bias;
   double size_mult;
   double hold_quality;
   double expected_utility;
   double confidence;
   double portfolio_fit;
   double capital_efficiency;
   int    action_code;
};

double g_policy_w1[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN][FXAI_POLICY_FEATS];
double g_policy_b1[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN];
double g_policy_trade_w2[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN];
double g_policy_dir_w2[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN];
double g_policy_size_w2[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN];
double g_policy_hold_w2[FXAI_REGIME_COUNT][FXAI_POLICY_HIDDEN];
double g_policy_trade_b2[FXAI_REGIME_COUNT];
double g_policy_dir_b2[FXAI_REGIME_COUNT];
double g_policy_size_b2[FXAI_REGIME_COUNT];
double g_policy_hold_b2[FXAI_REGIME_COUNT];
bool   g_policy_ready[FXAI_REGIME_COUNT];
int    g_policy_obs[FXAI_REGIME_COUNT];

int    g_policy_pending_seq[FXAI_REL_MAX_PENDING];
int    g_policy_pending_regime[FXAI_REL_MAX_PENDING];
int    g_policy_pending_horizon[FXAI_REL_MAX_PENDING];
double g_policy_pending_min_move[FXAI_REL_MAX_PENDING];
double g_policy_pending_feat[FXAI_REL_MAX_PENDING][FXAI_POLICY_FEATS];
int    g_policy_pending_head = 0;
int    g_policy_pending_tail = 0;

double g_policy_last_trade_prob = 0.0;
double g_policy_last_no_trade_prob = 1.0;
double g_policy_last_enter_prob = 0.0;
double g_policy_last_exit_prob = 0.0;
double g_policy_last_direction_bias = 0.0;
double g_policy_last_size_mult = 1.0;
double g_policy_last_hold_quality = 0.0;
double g_policy_last_expected_utility = 0.0;
double g_policy_last_confidence = 0.0;
double g_policy_last_portfolio_fit = 0.0;
double g_policy_last_capital_efficiency = 0.0;
int    g_policy_last_action = FXAI_POLICY_ACTION_NO_TRADE;

void FXAI_ClearPolicyDecision(FXAIPolicyDecision &out)
{
   out.trade_prob = 0.0;
   out.no_trade_prob = 1.0;
   out.enter_prob = 0.0;
   out.exit_prob = 0.0;
   out.direction_bias = 0.0;
   out.size_mult = 1.0;
   out.hold_quality = 0.0;
   out.expected_utility = 0.0;
   out.confidence = 0.0;
   out.portfolio_fit = 0.0;
   out.capital_efficiency = 0.0;
   out.action_code = FXAI_POLICY_ACTION_NO_TRADE;
}

void FXAI_ResetPolicyPending(void)
{
   g_policy_pending_head = 0;
   g_policy_pending_tail = 0;
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   for(int i=0; i<FXAI_REL_MAX_PENDING; i++)
   {
      g_policy_pending_seq[i] = -1;
      g_policy_pending_regime[i] = 0;
      g_policy_pending_horizon[i] = default_h;
      g_policy_pending_min_move[i] = 0.0;
      for(int j=0; j<FXAI_POLICY_FEATS; j++)
         g_policy_pending_feat[i][j] = 0.0;
   }
}

void FXAI_ResetPolicyState(void)
{
   g_policy_last_trade_prob = 0.0;
   g_policy_last_no_trade_prob = 1.0;
   g_policy_last_enter_prob = 0.0;
   g_policy_last_exit_prob = 0.0;
   g_policy_last_direction_bias = 0.0;
   g_policy_last_size_mult = 1.0;
   g_policy_last_hold_quality = 0.0;
   g_policy_last_expected_utility = 0.0;
   g_policy_last_confidence = 0.0;
   g_policy_last_portfolio_fit = 0.0;
   g_policy_last_capital_efficiency = 0.0;
   g_policy_last_action = FXAI_POLICY_ACTION_NO_TRADE;
   FXAI_ResetPolicyPending();
   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_policy_ready[r] = false;
      g_policy_obs[r] = 0;
      g_policy_trade_b2[r] = 0.0;
      g_policy_dir_b2[r] = 0.0;
      g_policy_size_b2[r] = 0.0;
      g_policy_hold_b2[r] = 0.0;
      for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
      {
         g_policy_b1[r][h] = 0.0;
         g_policy_trade_w2[r][h] = 0.0;
         g_policy_dir_w2[r][h] = 0.0;
         g_policy_size_w2[r][h] = 0.0;
         g_policy_hold_w2[r][h] = 0.0;
         for(int k=0; k<FXAI_POLICY_FEATS; k++)
            g_policy_w1[r][h][k] = 0.0;
      }
   }
}

void FXAI_BuildPolicyFeatures(const double &stack_feat[],
                              const double trade_gate,
                              const double trade_edge_points,
                              const double expected_move_points,
                              const double min_move_points,
                              const double macro_quality,
                              const double context_quality,
                              const double context_strength,
                              const double foundation_trust,
                              const double foundation_direction_bias,
                              const double student_trust,
                              const double analog_similarity,
                              const double analog_quality,
                              const FXAIRegimeGraphQuery &regime_q,
                              const FXAILiveDeploymentProfile &deploy,
                              const double portfolio_pressure_hint,
                              double &feat[])
{
   ArrayResize(feat, FXAI_POLICY_FEATS);
   for(int i=0; i<FXAI_POLICY_FEATS; i++)
      feat[i] = 0.0;

   double mm = MathMax(min_move_points, 0.10);
   double analog_weight = FXAI_Clamp(deploy.analog_weight, 0.0, 0.80);
   double transition_weight = FXAI_Clamp(deploy.regime_transition_weight, 0.0, 1.0);
   double macro_floor = FXAI_Clamp(deploy.macro_quality_floor, 0.0, 1.0);
   feat[0] = 1.0;
   feat[1] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetArrayValue(stack_feat, 1, 0.0), 0.0, 1.0);
   feat[2] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetArrayValue(stack_feat, 2, 0.0), 0.0, 1.0);
   feat[3] = FXAI_Clamp(0.5 + 0.5 * FXAI_GetArrayValue(stack_feat, 3, 0.0), 0.0, 1.0);
   feat[4] = FXAI_Clamp(expected_move_points / mm, 0.0, 8.0) / 4.0 - 0.5;
   feat[5] = FXAI_Clamp(trade_edge_points / mm, -4.0, 4.0) / 4.0;
   feat[6] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 20, 0.0), 0.0, 1.0);
   feat[7] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 21, 0.0), 0.0, 1.0);
   feat[8] = FXAI_Clamp(trade_gate, 0.0, 1.0);
   feat[9] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 80, 0.0), 0.0, 1.0);
   feat[10] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 81, 0.0), 0.0, 1.0);
   feat[11] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 82, 0.0), 0.0, 1.0);
   feat[12] = FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 83, 0.0), 0.0, 1.0);
   feat[13] = 1.0 - FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 49, 1.0), 0.0, 1.0);
   feat[14] = 1.0 - FXAI_Clamp(FXAI_GetArrayValue(stack_feat, 50, 1.0), 0.0, 1.0);
   feat[15] = FXAI_Clamp(macro_quality, 0.0, 1.0);
   feat[16] = FXAI_Clamp(0.5 + 0.5 * context_quality, 0.0, 1.0);
   feat[17] = FXAI_Clamp(context_strength / 3.0, 0.0, 1.0);
   feat[18] = FXAI_Clamp(foundation_trust * (0.82 + 0.50 * FXAI_Clamp(deploy.foundation_weight, 0.0, 0.90)),
                         0.0,
                         1.0);
   feat[19] = FXAI_Clamp(foundation_direction_bias, -1.0, 1.0);
   feat[20] = FXAI_Clamp(student_trust, 0.0, 1.0);
   feat[21] = FXAI_Clamp(analog_similarity * (0.80 + 0.60 * analog_weight), 0.0, 1.0);
   feat[22] = FXAI_Clamp(analog_quality * (0.80 + 0.60 * analog_weight), 0.0, 1.0);
   feat[23] = FXAI_Clamp(regime_q.persistence, 0.0, 1.0);
   feat[24] = FXAI_Clamp(regime_q.transition_confidence * (0.65 + 0.35 * transition_weight), 0.0, 1.0);
   feat[25] = FXAI_Clamp(regime_q.instability * (0.70 + 0.30 * transition_weight), 0.0, 1.0);
   feat[26] = FXAI_Clamp(regime_q.edge_bias, -1.0, 1.0);
   feat[27] = FXAI_Clamp(regime_q.quality_bias, 0.0, 1.0);
   feat[28] = FXAI_Clamp(0.60 * regime_q.macro_alignment +
                         0.25 * FXAI_Clamp(macro_quality - macro_floor + 0.50, 0.0, 1.0) +
                         0.15 * (1.0 - FXAI_Clamp(MathMax(macro_floor - macro_quality, 0.0), 0.0, 1.0)),
                         0.0,
                         1.0);
   feat[29] = FXAI_Clamp(deploy.teacher_weight - 0.50, -0.50, 0.50);
   feat[30] = FXAI_Clamp(0.65 * deploy.policy_trade_floor + 0.35 * macro_floor, 0.0, 1.0);
   feat[31] = FXAI_Clamp(portfolio_pressure_hint, 0.0, 1.5) / 1.5;
}

void FXAI_PolicyPredict(const int regime_id,
                        const double &feat[],
                        const FXAILiveDeploymentProfile &deploy,
                        FXAIPolicyDecision &out)
{
   FXAI_ClearPolicyDecision(out);
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT)
      r = 0;

   double macro_floor = FXAI_Clamp(deploy.macro_quality_floor, 0.0, 1.0);
   double macro_shortfall = FXAI_Clamp(macro_floor - FXAI_GetArrayValue(feat, 15, 0.0), 0.0, 1.0);
   double transition_pressure = FXAI_Clamp(deploy.regime_transition_weight, 0.0, 1.0) *
                                FXAI_GetArrayValue(feat, 25, 0.0);
   double analog_bonus = 0.5 * (FXAI_GetArrayValue(feat, 21, 0.0) + FXAI_GetArrayValue(feat, 22, 0.0));

   double heuristic_trade = FXAI_Clamp(0.16 +
                                       0.15 * FXAI_GetArrayValue(feat, 1, 0.0) +
                                       0.15 * FXAI_GetArrayValue(feat, 2, 0.0) -
                                       0.12 * FXAI_GetArrayValue(feat, 3, 0.0) +
                                       0.10 * FXAI_GetArrayValue(feat, 5, 0.0) +
                                       0.09 * FXAI_GetArrayValue(feat, 6, 0.0) +
                                       0.09 * FXAI_GetArrayValue(feat, 7, 0.0) +
                                       0.10 * FXAI_GetArrayValue(feat, 8, 0.0) +
                                       0.08 * FXAI_GetArrayValue(feat, 9, 0.0) +
                                       0.06 * FXAI_GetArrayValue(feat, 10, 0.0) +
                                       0.06 * FXAI_GetArrayValue(feat, 11, 0.0) +
                                       0.04 * FXAI_GetArrayValue(feat, 12, 0.0) +
                                       0.06 * FXAI_GetArrayValue(feat, 15, 0.0) +
                                       0.05 * FXAI_GetArrayValue(feat, 18, 0.0) +
                                       0.04 * FXAI_GetArrayValue(feat, 20, 0.0) +
                                       0.05 * analog_bonus +
                                       0.03 * FXAI_GetArrayValue(feat, 23, 0.0) -
                                       0.08 * transition_pressure -
                                       0.14 * macro_shortfall -
                                       0.10 * FXAI_GetArrayValue(feat, 31, 0.0) +
                                       0.08 * FXAI_GetArrayValue(feat, 29, 0.0) +
                                       0.08 * FXAI_GetArrayValue(feat, 30, 0.0),
                                       0.01,
                                       0.99);
   double heuristic_dir = FXAI_Clamp(0.70 * (FXAI_GetArrayValue(feat, 1, 0.5) - FXAI_GetArrayValue(feat, 2, 0.5)) +
                                     0.30 * FXAI_GetArrayValue(feat, 5, 0.0) +
                                     0.12 * FXAI_GetArrayValue(feat, 19, 0.0) +
                                     0.10 * FXAI_GetArrayValue(feat, 26, 0.0),
                                     -1.0,
                                     1.0);
   double heuristic_hold = FXAI_Clamp(0.18 +
                                      0.18 * FXAI_GetArrayValue(feat, 13, 0.0) +
                                      0.18 * FXAI_GetArrayValue(feat, 14, 0.0) +
                                      0.14 * FXAI_GetArrayValue(feat, 9, 0.0) +
                                      0.12 * FXAI_GetArrayValue(feat, 11, 0.0) +
                                      0.10 * FXAI_GetArrayValue(feat, 12, 0.0) +
                                      0.08 * FXAI_GetArrayValue(feat, 27, 0.0) -
                                      0.08 * transition_pressure -
                                      0.12 * macro_shortfall -
                                      0.08 * FXAI_GetArrayValue(feat, 31, 0.0),
                                      0.0,
                                      1.0);
   double heuristic_size = FXAI_Clamp(0.42 +
                                      0.26 * heuristic_trade +
                                      0.18 * MathMax(FXAI_GetArrayValue(feat, 5, 0.0), 0.0) +
                                      0.12 * FXAI_GetArrayValue(feat, 7, 0.0) +
                                      0.10 * FXAI_GetArrayValue(feat, 27, 0.0) -
                                      0.06 * transition_pressure -
                                      0.10 * macro_shortfall -
                                      0.18 * FXAI_GetArrayValue(feat, 31, 0.0),
                                      0.25,
                                      1.50);
   heuristic_size *= FXAI_Clamp(deploy.policy_size_bias, 0.40, 1.60);
   heuristic_size = FXAI_Clamp(heuristic_size, 0.25, 1.60);

   if(!g_policy_ready[r])
   {
      out.trade_prob = heuristic_trade;
      out.no_trade_prob = FXAI_Clamp(0.60 - 0.45 * heuristic_trade +
                                     0.20 * macro_shortfall +
                                     0.18 * transition_pressure +
                                     0.12 * FXAI_GetArrayValue(feat, 31, 0.0),
                                     0.0,
                                     1.0);
      out.direction_bias = heuristic_dir;
      out.hold_quality = heuristic_hold;
      out.size_mult = heuristic_size;
      out.portfolio_fit = FXAI_Clamp(0.62 * (1.0 - FXAI_GetArrayValue(feat, 31, 0.0)) +
                                     0.20 * FXAI_GetArrayValue(feat, 16, 0.0) +
                                     0.18 * FXAI_GetArrayValue(feat, 27, 0.0),
                                     0.0,
                                     1.0);
      out.capital_efficiency = FXAI_Clamp(0.34 +
                                          0.30 * MathMax(FXAI_GetArrayValue(feat, 5, 0.0), 0.0) +
                                          0.22 * FXAI_GetArrayValue(feat, 4, 0.0) +
                                          0.14 * heuristic_hold,
                                          0.0,
                                          1.0);
      out.enter_prob = FXAI_Clamp(out.trade_prob *
                                  (0.52 + 0.24 * out.portfolio_fit + 0.24 * out.capital_efficiency) *
                                  (1.0 - 0.45 * out.no_trade_prob),
                                  0.0,
                                  1.0);
      out.exit_prob = FXAI_Clamp(0.24 +
                                 0.22 * (1.0 - heuristic_hold) +
                                 0.18 * macro_shortfall +
                                 0.16 * transition_pressure +
                                 0.20 * FXAI_GetArrayValue(feat, 31, 0.0),
                                 0.0,
                                 1.0);
      out.expected_utility = FXAI_Clamp(0.55 * FXAI_GetArrayValue(feat, 5, 0.0) +
                                        0.25 * heuristic_trade +
                                        0.20 * heuristic_hold,
                                        -1.0,
                                        1.0);
      out.confidence = FXAI_Clamp(0.45 * heuristic_trade + 0.30 * heuristic_hold + 0.25 * (1.0 - FXAI_GetArrayValue(feat, 31, 0.0)), 0.0, 1.0);
      out.action_code = FXAI_POLICY_ACTION_NO_TRADE;
      if(out.enter_prob > MathMax(out.no_trade_prob, out.exit_prob) && out.enter_prob >= 0.42)
         out.action_code = FXAI_POLICY_ACTION_ENTER;
      else if(out.exit_prob > out.no_trade_prob && out.exit_prob >= 0.55)
         out.action_code = FXAI_POLICY_ACTION_EXIT;
      else if(out.hold_quality >= 0.55 && out.trade_prob >= 0.40)
         out.action_code = FXAI_POLICY_ACTION_HOLD;
      return;
   }

   double hidden[FXAI_POLICY_HIDDEN];
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      double z = g_policy_b1[r][h];
      for(int k=0; k<FXAI_POLICY_FEATS; k++)
         z += g_policy_w1[r][h][k] * FXAI_GetArrayValue(feat, k, 0.0);
      hidden[h] = FXAI_Tanh(z);
   }

   double trade_z = g_policy_trade_b2[r];
   double dir_z = g_policy_dir_b2[r];
   double size_z = g_policy_size_b2[r];
   double hold_z = g_policy_hold_b2[r];
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      trade_z += g_policy_trade_w2[r][h] * hidden[h];
      dir_z += g_policy_dir_w2[r][h] * hidden[h];
      size_z += g_policy_size_w2[r][h] * hidden[h];
      hold_z += g_policy_hold_w2[r][h] * hidden[h];
   }

   double learned_trade = FXAI_Sigmoid(trade_z);
   double learned_dir = FXAI_Tanh(dir_z);
   double learned_hold = FXAI_Sigmoid(hold_z);
   double learned_size = FXAI_Clamp(0.25 + 1.35 * FXAI_Sigmoid(size_z), 0.25, 1.60);
   double mix = FXAI_Clamp((double)g_policy_obs[r] / 160.0, 0.18, 0.82);
   out.trade_prob = FXAI_Clamp((1.0 - mix) * heuristic_trade + mix * learned_trade, 0.0, 1.0);
   out.direction_bias = FXAI_Clamp((1.0 - mix) * heuristic_dir + mix * learned_dir, -1.0, 1.0);
   out.hold_quality = FXAI_Clamp((1.0 - mix) * heuristic_hold + mix * learned_hold, 0.0, 1.0);
   out.size_mult = FXAI_Clamp((1.0 - mix) * heuristic_size + mix * learned_size, 0.25, 1.60);
   out.portfolio_fit = FXAI_Clamp(0.54 * (1.0 - FXAI_GetArrayValue(feat, 31, 0.0)) +
                                  0.18 * FXAI_GetArrayValue(feat, 16, 0.0) +
                                  0.12 * FXAI_GetArrayValue(feat, 17, 0.0) +
                                  0.16 * FXAI_GetArrayValue(feat, 27, 0.0),
                                  0.0,
                                  1.0);
   out.capital_efficiency = FXAI_Clamp(0.28 +
                                       0.26 * MathMax(FXAI_GetArrayValue(feat, 5, 0.0), 0.0) +
                                       0.18 * FXAI_GetArrayValue(feat, 4, 0.0) +
                                       0.14 * out.hold_quality +
                                       0.14 * out.trade_prob,
                                       0.0,
                                       1.0);
   out.expected_utility = FXAI_Clamp(0.60 * FXAI_GetArrayValue(feat, 5, 0.0) +
                                     0.20 * out.trade_prob +
                                     0.20 * out.hold_quality,
                                     -1.0,
                                     1.0);
   out.confidence = FXAI_Clamp(0.35 * out.trade_prob +
                               0.25 * out.hold_quality +
                               0.20 * (1.0 - FXAI_GetArrayValue(feat, 31, 0.0)) +
                               0.20 * FXAI_GetArrayValue(feat, 24, 0.0),
                               0.0,
                               1.0);
   double macro_guard = FXAI_Clamp(1.0 - 0.55 * macro_shortfall, 0.15, 1.0);
   double transition_guard = FXAI_Clamp(1.0 - 0.35 * transition_pressure, 0.35, 1.0);
   out.trade_prob = FXAI_Clamp(out.trade_prob * macro_guard * transition_guard, 0.0, 1.0);
   out.hold_quality = FXAI_Clamp(out.hold_quality * macro_guard * transition_guard, 0.0, 1.0);
   out.size_mult = FXAI_Clamp(out.size_mult * FXAI_Clamp(1.0 - 0.20 * macro_shortfall - 0.15 * transition_pressure, 0.40, 1.0),
                              0.25,
                              1.60);
   out.expected_utility = FXAI_Clamp(out.expected_utility - 0.25 * macro_shortfall - 0.18 * transition_pressure,
                                     -1.0,
                                     1.0);
   out.confidence = FXAI_Clamp(out.confidence * FXAI_Clamp(1.0 - 0.30 * macro_shortfall - 0.20 * transition_pressure, 0.20, 1.0),
                               0.0,
                               1.0);
   out.no_trade_prob = FXAI_Clamp(0.58 * (1.0 - out.trade_prob) +
                                  0.16 * macro_shortfall +
                                  0.14 * transition_pressure +
                                  0.12 * FXAI_GetArrayValue(feat, 31, 0.0),
                                  0.0,
                                  1.0);
   out.enter_prob = FXAI_Clamp(out.trade_prob *
                               (0.46 + 0.28 * out.portfolio_fit + 0.26 * out.capital_efficiency) *
                               (1.0 - 0.40 * out.no_trade_prob),
                               0.0,
                               1.0);
   out.exit_prob = FXAI_Clamp(0.20 +
                              0.18 * (1.0 - out.hold_quality) +
                              0.16 * macro_shortfall +
                              0.14 * transition_pressure +
                              0.14 * FXAI_GetArrayValue(feat, 31, 0.0) +
                              0.18 * MathMax(-FXAI_GetArrayValue(feat, 5, 0.0), 0.0),
                              0.0,
                              1.0);
   out.action_code = FXAI_POLICY_ACTION_NO_TRADE;
   double hold_score = FXAI_Clamp(0.52 * out.hold_quality + 0.28 * out.portfolio_fit + 0.20 * out.confidence, 0.0, 1.0);
   if(out.enter_prob > MathMax(out.no_trade_prob, MathMax(out.exit_prob, hold_score)) && out.enter_prob >= 0.40)
      out.action_code = FXAI_POLICY_ACTION_ENTER;
   else if(out.exit_prob > MathMax(out.no_trade_prob, hold_score) && out.exit_prob >= 0.52)
      out.action_code = FXAI_POLICY_ACTION_EXIT;
   else if(hold_score > out.no_trade_prob && hold_score >= 0.50)
      out.action_code = FXAI_POLICY_ACTION_HOLD;
}

void FXAI_EnqueuePolicyPending(const int signal_seq,
                               const int regime_id,
                               const int horizon_minutes,
                               const double min_move_points,
                               const double &feat[])
{
   if(signal_seq < 0)
      return;

   int head = g_policy_pending_head;
   int tail = g_policy_pending_tail;
   int prev = tail - 1;
   if(prev < 0)
      prev += FXAI_REL_MAX_PENDING;
   if(head != tail && g_policy_pending_seq[prev] == signal_seq)
   {
      g_policy_pending_regime[prev] = regime_id;
      g_policy_pending_horizon[prev] = FXAI_ClampHorizon(horizon_minutes);
      g_policy_pending_min_move[prev] = min_move_points;
      for(int k=0; k<FXAI_POLICY_FEATS; k++)
         g_policy_pending_feat[prev][k] = FXAI_GetArrayValue(feat, k, 0.0);
      return;
   }

   g_policy_pending_seq[tail] = signal_seq;
   g_policy_pending_regime[tail] = regime_id;
   g_policy_pending_horizon[tail] = FXAI_ClampHorizon(horizon_minutes);
   g_policy_pending_min_move[tail] = min_move_points;
   for(int k=0; k<FXAI_POLICY_FEATS; k++)
      g_policy_pending_feat[tail][k] = FXAI_GetArrayValue(feat, k, 0.0);

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING)
      next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING)
         head = 0;
      g_policy_pending_head = head;
   }
   g_policy_pending_tail = next_tail;
}

void FXAI_UpdatePolicyModel(const int regime_id,
                            const double &feat[],
                            const double trade_target,
                            const double dir_target,
                            const double size_target,
                            const double hold_target,
                            const double sample_weight)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT)
      r = 0;

   double hidden[FXAI_POLICY_HIDDEN];
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      double z = g_policy_b1[r][h];
      for(int k=0; k<FXAI_POLICY_FEATS; k++)
         z += g_policy_w1[r][h][k] * FXAI_GetArrayValue(feat, k, 0.0);
      hidden[h] = FXAI_Tanh(z);
   }

   double trade_z = g_policy_trade_b2[r];
   double dir_z = g_policy_dir_b2[r];
   double size_z = g_policy_size_b2[r];
   double hold_z = g_policy_hold_b2[r];
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      trade_z += g_policy_trade_w2[r][h] * hidden[h];
      dir_z += g_policy_dir_w2[r][h] * hidden[h];
      size_z += g_policy_size_w2[r][h] * hidden[h];
      hold_z += g_policy_hold_w2[r][h] * hidden[h];
   }

   double trade_pred = FXAI_Sigmoid(trade_z);
   double dir_pred = FXAI_Tanh(dir_z);
   double size_pred = FXAI_Clamp(0.25 + 1.35 * FXAI_Sigmoid(size_z), 0.25, 1.60);
   double hold_pred = FXAI_Sigmoid(hold_z);
   double sw = FXAI_Clamp(sample_weight, 0.20, 8.0);
   double lr = 0.018 / MathSqrt(1.0 + 0.02 * (double)g_policy_obs[r]);
   lr = FXAI_Clamp(lr, 0.0015, 0.018);

   double trade_err = FXAI_Clamp((trade_target - trade_pred) * sw, -3.0, 3.0);
   double dir_err = FXAI_Clamp((dir_target - dir_pred) * sw, -3.0, 3.0);
   double size_err = FXAI_Clamp(((size_target - size_pred) / 1.35) * sw, -3.0, 3.0);
   double hold_err = FXAI_Clamp((hold_target - hold_pred) * sw, -3.0, 3.0);

   double w_trade_old[FXAI_POLICY_HIDDEN];
   double w_dir_old[FXAI_POLICY_HIDDEN];
   double w_size_old[FXAI_POLICY_HIDDEN];
   double w_hold_old[FXAI_POLICY_HIDDEN];
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      w_trade_old[h] = g_policy_trade_w2[r][h];
      w_dir_old[h] = g_policy_dir_w2[r][h];
      w_size_old[h] = g_policy_size_w2[r][h];
      w_hold_old[h] = g_policy_hold_w2[r][h];
   }

   g_policy_trade_b2[r] += lr * trade_err;
   g_policy_dir_b2[r] += lr * dir_err;
   g_policy_size_b2[r] += lr * size_err;
   g_policy_hold_b2[r] += lr * hold_err;
   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      g_policy_trade_w2[r][h] += lr * (trade_err * hidden[h] - 0.0005 * g_policy_trade_w2[r][h]);
      g_policy_dir_w2[r][h] += lr * (dir_err * hidden[h] - 0.0005 * g_policy_dir_w2[r][h]);
      g_policy_size_w2[r][h] += lr * (size_err * hidden[h] - 0.0005 * g_policy_size_w2[r][h]);
      g_policy_hold_w2[r][h] += lr * (hold_err * hidden[h] - 0.0005 * g_policy_hold_w2[r][h]);
   }

   for(int h=0; h<FXAI_POLICY_HIDDEN; h++)
   {
      double back = w_trade_old[h] * trade_err +
                    w_dir_old[h] * dir_err +
                    w_size_old[h] * size_err +
                    w_hold_old[h] * hold_err;
      double dh = FXAI_Clamp(back * (1.0 - hidden[h] * hidden[h]), -3.0, 3.0);
      g_policy_b1[r][h] += lr * dh;
      for(int k=0; k<FXAI_POLICY_FEATS; k++)
      {
         double reg = (k == 0 ? 0.0 : 0.0004 * g_policy_w1[r][h][k]);
         g_policy_w1[r][h][k] += lr * (dh * FXAI_GetArrayValue(feat, k, 0.0) - reg);
      }
   }

   g_policy_obs[r]++;
   if(g_policy_obs[r] > 200000)
      g_policy_obs[r] = 200000;
   g_policy_ready[r] = true;
   FXAI_MarkMetaArtifactsDirty();
}

void FXAI_UpdatePolicyFromPending(const int current_signal_seq,
                                  const int current_regime_id,
                                  const double current_macro_quality,
                                  const FXAIDataSnapshot &snapshot,
                                  const int &spread_m1[],
                                  const double &high_arr[],
                                  const double &low_arr[],
                                  const double &close_arr[],
                                  const double commission_points,
                                  const double cost_buffer_points,
                                  const double ev_threshold_points)
{
   if(current_signal_seq < 0)
      return;

   int head = g_policy_pending_head;
   int tail = g_policy_pending_tail;
   if(head == tail)
      return;

   int keep_seq[];
   int keep_regime[];
   int keep_horizon[];
   double keep_min_move[];
   double keep_feat[][FXAI_POLICY_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_min_move, 0);
   ArrayResize(keep_feat, 0);

   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_policy_pending_seq[idx];
      int pending_h = FXAI_ClampHorizon(g_policy_pending_horizon[idx]);
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
            if(idx_pred >= 0 &&
               idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = FXAI_ExecutionEntryCostPoints(spread_i,
                                                                 commission_points,
                                                                 cost_buffer_points,
                                                                 exec_profile);
               if(min_move_i < 0.0)
                  min_move_i = 0.0;

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
               double realized_edge = 0.0;
               if(label_class == (int)FXAI_LABEL_BUY)
                  realized_edge = move_points - min_move_i;
               else if(label_class == (int)FXAI_LABEL_SELL)
                  realized_edge = -move_points - min_move_i;
               else
                  realized_edge = -MathMax(MathAbs(move_points) - min_move_i, 0.0);

               double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
               double mae_ratio = FXAI_Clamp(mae_points / MathMax(mfe_points, min_move_i), 0.0, 3.0);
               double quality = 1.0 + 0.28 * speed_bonus - 0.16 * mae_ratio;
               if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0)
                  quality -= 0.10;
               if((path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0)
                  quality -= 0.08;
               quality = FXAI_Clamp(quality, 0.0, 2.0);

               double trade_target = (label_class != (int)FXAI_LABEL_SKIP &&
                                      realized_edge > 0.0 &&
                                      quality > 0.70 ? 1.0 : 0.0);
               double dir_target = 0.0;
               if(label_class == (int)FXAI_LABEL_BUY)
                  dir_target = 1.0;
               else if(label_class == (int)FXAI_LABEL_SELL)
                  dir_target = -1.0;
               double size_target = FXAI_Clamp(0.25 +
                                               0.45 * MathMax(realized_edge / MathMax(min_move_i, 0.10), 0.0) +
                                               0.20 * quality,
                                               0.25,
                                               1.60);
               if(trade_target < 0.5)
                  size_target = 0.25;
               double hold_target = FXAI_Clamp(0.28 +
                                               0.24 * speed_bonus +
                                               0.22 * quality / 2.0 +
                                               0.16 * (1.0 - mae_ratio / 3.0) -
                                               0.12 * ((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0 ? 1.0 : 0.0),
                                               0.0,
                                               1.0);

               double feat_local[];
               ArrayResize(feat_local, FXAI_POLICY_FEATS);
               for(int k=0; k<FXAI_POLICY_FEATS; k++)
                  feat_local[k] = g_policy_pending_feat[idx][k];
               double sw = FXAI_Clamp(0.40 + 0.45 * quality + 0.25 * MathMax(MathAbs(realized_edge) / MathMax(min_move_i, 0.10), 0.0), 0.20, 6.0);
               FXAI_UpdatePolicyModel(g_policy_pending_regime[idx],
                                      feat_local,
                                      trade_target,
                                      dir_target,
                                      size_target,
                                      hold_target,
                                      sw);
               FXAI_UpdateRegimeGraphFeedback(g_policy_pending_regime[idx],
                                             current_regime_id,
                                             realized_edge,
                                             quality,
                                             current_macro_quality);
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
            ArrayResize(keep_regime, ks + 1);
            ArrayResize(keep_horizon, ks + 1);
            ArrayResize(keep_min_move, ks + 1);
            ArrayResize(keep_feat, ks + 1);
            keep_seq[ks] = g_policy_pending_seq[idx];
            keep_regime[ks] = g_policy_pending_regime[idx];
            keep_horizon[ks] = g_policy_pending_horizon[idx];
            keep_min_move[ks] = g_policy_pending_min_move[idx];
            for(int k=0; k<FXAI_POLICY_FEATS; k++)
               keep_feat[ks][k] = g_policy_pending_feat[idx][k];
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING)
         idx = 0;
   }

   FXAI_ResetPolicyPending();
   int keep_n = ArraySize(keep_seq);
   int cap = FXAI_REL_MAX_PENDING - 1;
   if(keep_n > cap)
      keep_n = cap;
   for(int i=0; i<keep_n; i++)
   {
      g_policy_pending_seq[i] = keep_seq[i];
      g_policy_pending_regime[i] = keep_regime[i];
      g_policy_pending_horizon[i] = keep_horizon[i];
      g_policy_pending_min_move[i] = keep_min_move[i];
      for(int k=0; k<FXAI_POLICY_FEATS; k++)
         g_policy_pending_feat[i][k] = keep_feat[i][k];
   }
   g_policy_pending_head = 0;
   g_policy_pending_tail = keep_n;
}

#endif // __FXAI_META_POLICY_MQH__
