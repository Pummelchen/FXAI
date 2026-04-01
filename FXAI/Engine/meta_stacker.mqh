#ifndef __FXAI_META_STACKER_MQH__
#define __FXAI_META_STACKER_MQH__

bool FXAI_IsModelInList(const int ai_idx, const int &ai_list[])
{
   for(int i=0; i<ArraySize(ai_list); i++)
      if(ai_list[i] == ai_idx) return true;
   return false;
}

double FXAI_StackPortfolioObjective(const double &feat[])
{
   return FXAI_Clamp(0.30 * FXAI_GetArrayValue(feat, 61, 0.0) +
                     0.28 * FXAI_GetArrayValue(feat, 62, 0.0) -
                     0.22 * FXAI_GetArrayValue(feat, 63, 0.0) +
                     0.24 * FXAI_GetArrayValue(feat, 64, 0.0) +
                     0.10 * FXAI_GetArrayValue(feat, 70, 0.0) +
                     0.10 * FXAI_GetArrayValue(feat, 71, 0.0),
                     -1.0,
                     1.0);
}

double FXAI_StackRoutingObjective(const double &feat[])
{
   return FXAI_Clamp(0.22 * FXAI_GetArrayValue(feat, 56, 0.0) -
                     0.30 * FXAI_GetArrayValue(feat, 57, 0.0) +
                     0.18 * FXAI_GetArrayValue(feat, 58, 0.0) +
                     0.20 * FXAI_GetArrayValue(feat, 59, 0.0) -
                     0.18 * FXAI_GetArrayValue(feat, 60, 0.0) +
                     0.10 * FXAI_GetArrayValue(feat, 68, 0.0) +
                     0.16 * FXAI_GetArrayValue(feat, 69, 0.0),
                     -1.0,
                     1.0);
}

double FXAI_StackRouterContextTrust(const double &feat[])
{
   return FXAI_Clamp(0.18 +
                     0.22 * FXAI_GetArrayValue(feat, 68, 0.0) +
                     0.16 * FXAI_GetArrayValue(feat, 62, 0.0) +
                     0.12 * FXAI_GetArrayValue(feat, 70, 0.0) +
                     0.10 * FXAI_GetArrayValue(feat, 71, 0.0) -
                     0.14 * FXAI_GetArrayValue(feat, 57, 0.0) -
                     0.10 * FXAI_GetArrayValue(feat, 63, 0.0),
                     0.0,
                     1.0);
}

double FXAI_StackRouterActionUtility(const int action,
                                     const int label_class,
                                     const double realized_edge,
                                     const double quality_score)
{
   double edge_norm = FXAI_Clamp(realized_edge / MathMax(MathAbs(realized_edge), 1.0), -1.0, 1.0);
   double qual = FXAI_Clamp(quality_score, 0.0, 2.0);
   if(action == (int)FXAI_LABEL_SKIP)
   {
      if(label_class == (int)FXAI_LABEL_SKIP)
         return FXAI_Clamp(0.28 + 0.22 * qual - 0.12 * edge_norm, -1.0, 1.0);
      return FXAI_Clamp(-0.28 - 0.46 * MathMax(edge_norm, 0.0) - 0.10 * qual, -1.0, 1.0);
   }

   if(action == label_class)
      return FXAI_Clamp(0.32 + 0.52 * edge_norm + 0.18 * qual, -1.0, 1.0);
   if(label_class == (int)FXAI_LABEL_SKIP)
      return FXAI_Clamp(-0.22 - 0.18 * qual - 0.18 * MathAbs(edge_norm), -1.0, 1.0);
   return FXAI_Clamp(-0.34 - 0.52 * MathMax(edge_norm, 0.0) - 0.12 * qual, -1.0, 1.0);
}

void FXAI_StackRouterBlend(const int regime_id,
                           const int horizon_minutes,
                           const double &feat[],
                           double &probs[])
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;
   int slot = FXAI_GetHorizonSlot(horizon_minutes);
   double context_trust = FXAI_StackRouterContextTrust(feat);
   if(context_trust <= 1e-6)
      return;

   for(int c=0; c<3; c++)
   {
      if(!g_router_action_ready[r][slot][c] || g_router_action_obs[r][slot][c] <= 0)
         continue;
      double obs_trust = FXAI_Clamp((double)g_router_action_obs[r][slot][c] / 48.0, 0.0, 1.0);
      double router_score = 0.70 * g_router_action_value[r][slot][c] +
                            0.35 * g_router_action_counterfactual[r][slot][c] -
                            0.55 * g_router_action_regret[r][slot][c];
      if(c == (int)FXAI_LABEL_BUY)
         router_score += 0.10 * FXAI_StackRoutingObjective(feat) * FXAI_Clamp(FXAI_GetArrayValue(feat, 6, 0.0), -1.0, 1.0);
      else if(c == (int)FXAI_LABEL_SELL)
         router_score -= 0.10 * FXAI_StackRoutingObjective(feat) * FXAI_Clamp(FXAI_GetArrayValue(feat, 6, 0.0), -1.0, 1.0);
      else
         router_score += 0.06 * FXAI_StackPortfolioObjective(feat) * FXAI_Clamp(FXAI_GetArrayValue(feat, 63, 0.0), 0.0, 1.0);

      double mult = MathExp(FXAI_ClipSym(context_trust * obs_trust * router_score, 1.2));
      probs[c] = FXAI_Clamp(probs[c] * mult, 0.0005, 0.9990);
   }

   double den = probs[0] + probs[1] + probs[2];
   if(den <= 0.0) den = 1.0;
   probs[0] /= den;
   probs[1] /= den;
   probs[2] /= den;
}

void FXAI_StackRouterObserve(const int regime_id,
                             const int horizon_minutes,
                             const int label_class,
                             const double realized_edge,
                             const double quality_score,
                             const double &feat[],
                             const double &pred_probs[],
                             const double sample_weight)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;
   int slot = FXAI_GetHorizonSlot(horizon_minutes);
   double baseline = 0.0;
   double utility[3];
   double best_u = -1e9;
   for(int c=0; c<3; c++)
   {
      utility[c] = FXAI_StackRouterActionUtility(c, label_class, realized_edge, quality_score);
      baseline += FXAI_GetArrayValue(pred_probs, c, 0.3333333) * utility[c];
      if(utility[c] > best_u)
         best_u = utility[c];
   }

   double trust = FXAI_Clamp(sample_weight * (0.40 + 0.60 * FXAI_StackRouterContextTrust(feat)), 0.10, 6.0);
   for(int c=0; c<3; c++)
   {
      double obs = (double)g_router_action_obs[r][slot][c];
      double alpha = FXAI_Clamp(0.18 / MathSqrt(1.0 + 0.05 * obs), 0.02, 0.18);
      double u = utility[c];
      double cf = FXAI_ClipSym(u - baseline, 1.0);
      double regret = FXAI_Clamp(best_u - u, 0.0, 1.0);
      if(obs <= 0.0)
      {
         g_router_action_value[r][slot][c] = u;
         g_router_action_counterfactual[r][slot][c] = cf;
         g_router_action_regret[r][slot][c] = regret;
      }
      else
      {
         double blend = FXAI_Clamp(alpha * trust, 0.01, 0.25);
         g_router_action_value[r][slot][c] =
            (1.0 - blend) * g_router_action_value[r][slot][c] + blend * u;
         g_router_action_counterfactual[r][slot][c] =
            (1.0 - blend) * g_router_action_counterfactual[r][slot][c] + blend * cf;
         g_router_action_regret[r][slot][c] =
            (1.0 - blend) * g_router_action_regret[r][slot][c] + blend * regret;
      }
      g_router_action_obs[r][slot][c]++;
      if(g_router_action_obs[r][slot][c] > 200000)
         g_router_action_obs[r][slot][c] = 200000;
      g_router_action_ready[r][slot][c] = true;
   }

   FXAI_MarkMetaArtifactsDirty();
}

void FXAI_StackBuildFeatures(const double buy_pct,
                             const double sell_pct,
                             const double skip_pct,
                             const double avg_buy_ev,
                             const double avg_sell_ev,
                             const double min_move_points,
                             const double expected_move_points,
                             const double vol_proxy,
                             const int horizon_minutes,
                             const double avg_confidence,
                             const double avg_reliability,
                             const double move_dispersion,
                             const double directional_margin,
                             const double active_family_ratio,
                             const double dominant_family_ratio,
                             const double context_strength,
                             const double context_quality,
                             const double avg_hit_time,
                             const double avg_path_risk,
                             const double avg_fill_risk,
                             const double avg_mfe_ratio,
                             const double avg_mae_ratio,
                             const double avg_ctx_edge_norm,
                             const double avg_ctx_regret,
                             const double avg_global_edge_norm,
                             const double best_counterfactual_edge_norm,
                             const double ensemble_vs_best_gap_norm,
                             const double avg_portfolio_edge_norm,
                             const double avg_portfolio_stability,
                             const double avg_portfolio_corr_penalty,
                             const double avg_portfolio_diversification,
                             const double best_model_share,
                             const double best_buy_share,
                             const double best_sell_share,
                             const double avg_context_trust,
                             const double foundation_trust,
                             const double foundation_direction_bias,
                             const double foundation_move_ratio,
                             const double student_trust,
                             const double student_tradability,
                             const double analog_similarity,
                             const double analog_edge_norm,
                             const double analog_quality,
                             const double hierarchy_consistency,
                             const double hierarchy_tradability,
                             const double hierarchy_execution_viability,
                             const double hierarchy_horizon_fit,
                             double &feat[])
{
   for(int k=0; k<FXAI_STACK_FEATS; k++)
      feat[k] = 0.0;

   double mm = MathMax(min_move_points, 0.10);
   double em = MathMax(expected_move_points, mm);
   double pb = FXAI_Clamp(buy_pct / 100.0, 0.0, 1.0);
   double ps = FXAI_Clamp(sell_pct / 100.0, 0.0, 1.0);
   double pk = FXAI_Clamp(skip_pct / 100.0, 0.0, 1.0);
   double sump = pb + ps + pk;
   if(sump <= 0.0) sump = 1.0;
   pb /= sump; ps /= sump; pk /= sump;

   double entropy = 0.0;
   if(pb > 1e-9) entropy -= pb * MathLog(pb);
   if(ps > 1e-9) entropy -= ps * MathLog(ps);
   if(pk > 1e-9) entropy -= pk * MathLog(pk);
   double entropy_norm = 1.0 - FXAI_Clamp(entropy / MathLog(3.0), 0.0, 1.0);

   feat[0] = 1.0;
   feat[1] = FXAI_Clamp((pb - 0.5) * 2.0, -1.0, 1.0);
   feat[2] = FXAI_Clamp((ps - 0.5) * 2.0, -1.0, 1.0);
   feat[3] = FXAI_Clamp((pk - 0.5) * 2.0, -1.0, 1.0);
   feat[4] = FXAI_Clamp(avg_buy_ev / mm, -3.0, 3.0) / 3.0;
   feat[5] = FXAI_Clamp(avg_sell_ev / mm, -3.0, 3.0) / 3.0;
   feat[6] = FXAI_Clamp(pb - ps, -1.0, 1.0);
   feat[7] = FXAI_Clamp(MathMax(pb, ps), 0.0, 1.0);
   feat[8] = FXAI_Clamp(entropy_norm, 0.0, 1.0);
   feat[9] = FXAI_Clamp((avg_buy_ev - avg_sell_ev) / mm, -4.0, 4.0) / 4.0;
   feat[10] = FXAI_Clamp(min_move_points / em, 0.0, 2.0) - 0.5;
   feat[11] = FXAI_Clamp(vol_proxy / 4.0, 0.0, 1.0);
   feat[12] = FXAI_Clamp((double)horizon_minutes / (double)MathMax(FXAI_GetMaxConfiguredHorizon(horizon_minutes), 1), 0.0, 1.0);
   feat[13] = FXAI_Clamp(pk - MathMax(pb, ps), -1.0, 1.0);
   double top = pb;
   if(ps > top) top = ps;
   if(pk > top) top = pk;
   double second = 0.0;
   if(top == pb)
      second = MathMax(ps, pk);
   else if(top == ps)
      second = MathMax(pb, pk);
   else
      second = MathMax(pb, ps);
   feat[14] = FXAI_Clamp(top - second, 0.0, 1.0);
   feat[15] = FXAI_Clamp(MathAbs(pb - ps), 0.0, 1.0);
   feat[16] = FXAI_Clamp(MathMin(pb, ps), 0.0, 0.5) * 2.0;
   feat[17] = FXAI_Clamp(MathMax(avg_buy_ev, avg_sell_ev) / mm, -2.0, 6.0) / 4.0;
   feat[18] = FXAI_Clamp(expected_move_points / mm, 0.0, 8.0) / 4.0;
   feat[19] = FXAI_Clamp((pb + ps) - pk, -1.0, 1.0);
   feat[20] = FXAI_Clamp(avg_confidence, 0.0, 1.0);
   feat[21] = FXAI_Clamp(avg_reliability, 0.0, 1.0);
   feat[22] = FXAI_Clamp(move_dispersion / mm, 0.0, 4.0) / 2.0;
   feat[23] = FXAI_Clamp(directional_margin, 0.0, 1.0);
   feat[24] = FXAI_Clamp(active_family_ratio, 0.0, 1.0);
   feat[25] = FXAI_Clamp(dominant_family_ratio, 0.0, 1.0);
   feat[26] = FXAI_Clamp(context_strength, 0.0, 4.0) / 2.0;
   feat[27] = FXAI_Clamp(context_quality, -1.0, 2.0) / 2.0;
   feat[28] = FXAI_Clamp(pb + ps, 0.0, 1.0);
   feat[29] = FXAI_Clamp(avg_confidence * avg_reliability, 0.0, 1.0);
   feat[30] = FXAI_Clamp((pb - ps) * directional_margin, -1.0, 1.0);
   feat[31] = FXAI_Clamp(move_dispersion / em, 0.0, 4.0) / 2.0;
   feat[32] = FXAI_Clamp((context_strength * MathMax(context_quality, 0.0)) / 4.0, 0.0, 2.0) - 0.5;
   feat[33] = FXAI_Clamp(dominant_family_ratio * avg_reliability, 0.0, 1.0);
   feat[34] = FXAI_Clamp((avg_buy_ev + avg_sell_ev) / (2.0 * mm), -3.0, 3.0) / 3.0;
   feat[35] = FXAI_Clamp(entropy_norm * 0.5 * (avg_confidence + avg_reliability), 0.0, 1.0);
   feat[36] = FXAI_Clamp(MathMax(avg_buy_ev, avg_sell_ev) / MathMax(expected_move_points, mm), -2.0, 6.0) / 4.0;
   feat[37] = FXAI_Clamp(avg_confidence - avg_reliability, -1.0, 1.0);
   feat[38] = FXAI_Clamp(avg_reliability * MathMax(context_quality, 0.0), 0.0, 1.5) - 0.25;
   feat[39] = FXAI_Clamp(directional_margin * context_strength, 0.0, 4.0) / 2.0 - 0.5;
   feat[40] = FXAI_Clamp(active_family_ratio * dominant_family_ratio, 0.0, 1.0);
   feat[41] = FXAI_Clamp(move_dispersion / MathMax(expected_move_points, mm), 0.0, 4.0) / 2.0;
   feat[42] = FXAI_Clamp((pb + ps - pk) * avg_confidence, -1.0, 1.0);
   feat[43] = FXAI_Clamp(((avg_buy_ev - avg_sell_ev) / MathMax(expected_move_points, mm)) * directional_margin, -2.0, 2.0) / 2.0;
   feat[44] = FXAI_Clamp(entropy_norm * dominant_family_ratio, 0.0, 1.0);
   feat[45] = FXAI_Clamp(avg_confidence * directional_margin * (1.0 - pk), 0.0, 1.0);
   feat[46] = FXAI_Clamp(context_strength * dominant_family_ratio * avg_reliability, 0.0, 4.0) / 2.0 - 0.5;
   feat[47] = FXAI_Clamp((avg_buy_ev + avg_sell_ev) / MathMax(expected_move_points + mm, mm), -2.0, 2.0) / 2.0;
   feat[48] = FXAI_Clamp(avg_hit_time, 0.0, 1.0);
   feat[49] = FXAI_Clamp(avg_path_risk, 0.0, 1.0);
   feat[50] = FXAI_Clamp(avg_fill_risk, 0.0, 1.0);
   feat[51] = FXAI_Clamp(avg_mfe_ratio, 0.0, 4.0) / 2.0 - 0.5;
   feat[52] = FXAI_Clamp(avg_mae_ratio, 0.0, 2.0) - 0.5;
   feat[53] = FXAI_Clamp((1.0 - avg_path_risk) * avg_confidence, 0.0, 1.0);
   feat[54] = FXAI_Clamp((1.0 - avg_fill_risk) * dominant_family_ratio, 0.0, 1.0);
   feat[55] = FXAI_Clamp(avg_mfe_ratio * (1.0 - avg_mae_ratio) * MathMax(context_quality + 1.0, 0.0), 0.0, 4.0) / 2.0 - 0.5;
   feat[56] = FXAI_Clamp(avg_ctx_edge_norm, -1.0, 1.0);
   feat[57] = FXAI_Clamp(avg_ctx_regret, 0.0, 1.0);
   feat[58] = FXAI_Clamp(avg_global_edge_norm, -1.0, 1.0);
   feat[59] = FXAI_Clamp(best_counterfactual_edge_norm, -1.0, 1.0);
   feat[60] = FXAI_Clamp(ensemble_vs_best_gap_norm, 0.0, 1.0);
   feat[61] = FXAI_Clamp(avg_portfolio_edge_norm, -1.0, 1.0);
   feat[62] = FXAI_Clamp(avg_portfolio_stability, 0.0, 1.0);
   feat[63] = FXAI_Clamp(avg_portfolio_corr_penalty, 0.0, 1.0);
   feat[64] = FXAI_Clamp(avg_portfolio_diversification, 0.0, 1.0);
   feat[65] = FXAI_Clamp(best_model_share, 0.0, 1.0);
   feat[66] = FXAI_Clamp(best_buy_share, 0.0, 1.0);
   feat[67] = FXAI_Clamp(best_sell_share, 0.0, 1.0);
   feat[68] = FXAI_Clamp(avg_context_trust, 0.0, 1.0);
   feat[69] = FXAI_Clamp(best_counterfactual_edge_norm - avg_ctx_regret, -1.0, 1.0);
   feat[70] = FXAI_Clamp(avg_portfolio_stability * (1.0 - avg_ctx_regret), 0.0, 1.0);
   feat[71] = FXAI_Clamp((avg_ctx_edge_norm + avg_global_edge_norm + avg_portfolio_edge_norm) / 3.0, -1.0, 1.0);
   feat[72] = FXAI_Clamp(foundation_trust, 0.0, 1.0);
   feat[73] = FXAI_Clamp(foundation_direction_bias, -1.0, 1.0);
   feat[74] = FXAI_Clamp((foundation_move_ratio - 1.0) / 1.5, -1.0, 1.0);
   feat[75] = FXAI_Clamp(student_trust, 0.0, 1.0);
   feat[76] = FXAI_Clamp(student_tradability, 0.0, 1.0);
   feat[77] = FXAI_Clamp(analog_similarity, 0.0, 1.0);
   feat[78] = FXAI_Clamp(analog_edge_norm, -1.0, 1.0);
   feat[79] = FXAI_Clamp(analog_quality, 0.0, 1.0);
   feat[80] = FXAI_Clamp(hierarchy_consistency, 0.0, 1.0);
   feat[81] = FXAI_Clamp(hierarchy_tradability, 0.0, 1.0);
   feat[82] = FXAI_Clamp(hierarchy_execution_viability, 0.0, 1.0);
   feat[83] = FXAI_Clamp(hierarchy_horizon_fit, 0.0, 1.0);
}

void FXAI_StackPredict(const int regime_id,
                       const int horizon_minutes,
                       const double &feat[],
                       double &probs[])
{
   if(ArraySize(probs) != 3) ArrayResize(probs, 3);
   probs[0] = 0.3333;
   probs[1] = 0.3333;
   probs[2] = 0.3334;

   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   if(!g_stack_ready[r])
   {
      double p_sell = FXAI_Clamp(0.26 + (0.31 * feat[2]) - (0.12 * feat[3]) + (0.16 * feat[5]) -
                                 (0.07 * feat[10]) + (0.08 * feat[20]) + (0.08 * feat[21]) +
                                 (0.05 * feat[23]) + (0.05 * feat[30]) + (0.04 * feat[34]) +
                                 (0.05 * feat[58]) + (0.04 * feat[59]) + (0.04 * feat[61]) +
                                 (0.04 * feat[62]) - (0.05 * feat[57]) - (0.04 * feat[63]) -
                                 (0.03 * feat[73]) + (0.03 * feat[78]) + (0.03 * feat[80]) +
                                 (0.02 * feat[82]), 0.01, 0.98);
      double p_buy  = FXAI_Clamp(0.26 + (0.31 * feat[1]) - (0.12 * feat[3]) + (0.16 * feat[4]) -
                                 (0.07 * feat[10]) + (0.08 * feat[20]) + (0.08 * feat[21]) +
                                 (0.05 * feat[23]) - (0.05 * feat[30]) + (0.04 * feat[34]) +
                                 (0.05 * feat[58]) + (0.04 * feat[59]) + (0.04 * feat[61]) +
                                 (0.04 * feat[62]) - (0.05 * feat[57]) - (0.04 * feat[63]) +
                                 (0.03 * feat[73]) + (0.03 * feat[78]) + (0.03 * feat[80]) +
                                 (0.02 * feat[82]), 0.01, 0.98);
      double p_skip = FXAI_Clamp(0.18 + (0.32 * feat[3]) + (0.18 * feat[10]) - (0.08 * feat[8]) -
                                 (0.07 * feat[20]) - (0.07 * feat[21]) + (0.08 * feat[31]) -
                                 (0.04 * feat[28]) - (0.03 * feat[32]) + (0.08 * feat[57]) +
                                 (0.07 * feat[60]) + (0.05 * feat[63]) - (0.05 * feat[62]) -
                                 (0.05 * feat[80]) - (0.05 * feat[81]) - (0.04 * feat[82]) -
                                 (0.03 * feat[77]), 0.01, 0.98);
      double portfolio_obj = FXAI_StackPortfolioObjective(feat);
      double routing_obj = FXAI_StackRoutingObjective(feat);
      p_buy = FXAI_Clamp(p_buy + 0.05 * portfolio_obj + 0.06 * routing_obj * FXAI_Clamp(feat[6], -1.0, 1.0), 0.01, 0.98);
      p_sell = FXAI_Clamp(p_sell + 0.05 * portfolio_obj - 0.06 * routing_obj * FXAI_Clamp(feat[6], -1.0, 1.0), 0.01, 0.98);
      p_skip = FXAI_Clamp(p_skip - 0.05 * portfolio_obj - 0.04 * routing_obj * FXAI_Clamp(feat[19], -1.0, 1.0), 0.01, 0.98);
      double s0 = p_sell + p_buy + p_skip;
      if(s0 <= 0.0) s0 = 1.0;
      probs[0] = p_sell / s0;
      probs[1] = p_buy / s0;
      probs[2] = p_skip / s0;
      FXAI_StackRouterBlend(r, horizon_minutes, feat, probs);
      return;
   }

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z[3];
   for(int c=0; c<3; c++)
   {
      z[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z[0];
   if(z[1] > zmax) zmax = z[1];
   if(z[2] > zmax) zmax = z[2];
   double e0 = MathExp(z[0] - zmax);
   double e1 = MathExp(z[1] - zmax);
   double e2 = MathExp(z[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;
   double portfolio_obj = FXAI_StackPortfolioObjective(feat);
   double routing_obj = FXAI_StackRoutingObjective(feat);
   double dir_bias = FXAI_Clamp(0.08 * routing_obj * FXAI_Clamp(feat[6], -1.0, 1.0) +
                                0.05 * portfolio_obj * FXAI_Clamp(feat[71], -1.0, 1.0),
                                -0.12,
                                0.12);
   probs[0] = FXAI_Clamp(probs[0] + MathMax(-dir_bias, 0.0), 0.0005, 0.9990);
   probs[1] = FXAI_Clamp(probs[1] + MathMax(dir_bias, 0.0), 0.0005, 0.9990);
   probs[2] = FXAI_Clamp(probs[2] - 0.04 * portfolio_obj, 0.0005, 0.9990);
   double sn = probs[0] + probs[1] + probs[2];
   if(sn <= 0.0) sn = 1.0;
   probs[0] /= sn;
   probs[1] /= sn;
   probs[2] /= sn;
   FXAI_StackRouterBlend(r, horizon_minutes, feat, probs);
}

void FXAI_StackUpdate(const int regime_id,
                      const int label_class,
                      const double &feat[],
                      const double sample_weight)
{
   if(label_class < 0 || label_class > 2) return;
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double z = g_stack_b1[r][h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
         z += g_stack_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z_out[3];
   double probs[3];
   for(int c=0; c<3; c++)
   {
      z_out[c] = g_stack_b2[r][c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
         z_out[c] += g_stack_w2[r][c][h] * hidden[h];
   }

   double zmax = z_out[0];
   if(z_out[1] > zmax) zmax = z_out[1];
   if(z_out[2] > zmax) zmax = z_out[2];
   double e0 = MathExp(z_out[0] - zmax);
   double e1 = MathExp(z_out[1] - zmax);
   double e2 = MathExp(z_out[2] - zmax);
   double s = e0 + e1 + e2;
   if(s <= 0.0) s = 1.0;
   probs[0] = e0 / s;
   probs[1] = e1 / s;
   probs[2] = e2 / s;

   double lr = 0.025 / MathSqrt(1.0 + 0.02 * (double)g_stack_obs[r]);
   lr = FXAI_Clamp(lr, 0.002, 0.025);
   double portfolio_obj = FXAI_StackPortfolioObjective(feat);
   double routing_obj = FXAI_StackRoutingObjective(feat);
   double sw = FXAI_Clamp(sample_weight *
                          FXAI_Clamp(0.90 + 0.30 * portfolio_obj + 0.20 * routing_obj, 0.45, 1.60),
                          0.20,
                          7.50);

   double delta_out[3];
   for(int c=0; c<3; c++)
   {
      double target = (c == label_class ? 1.0 : 0.0);
      delta_out[c] = FXAI_Clamp((target - probs[c]) * sw, -3.0, 3.0);
   }

   double delta_hidden[FXAI_STACK_HIDDEN];
   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      double back = 0.0;
      for(int c=0; c<3; c++)
         back += delta_out[c] * g_stack_w2[r][c][h];
      delta_hidden[h] = back * (1.0 - hidden[h] * hidden[h]);
      delta_hidden[h] = FXAI_Clamp(delta_hidden[h], -3.0, 3.0);
   }

   for(int c=0; c<3; c++)
   {
      g_stack_b2[r][c] += lr * delta_out[c];
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         double reg = 0.0005 * g_stack_w2[r][c][h];
         g_stack_w2[r][c][h] += lr * (delta_out[c] * hidden[h] - reg);
      }
   }

   for(int h=0; h<FXAI_STACK_HIDDEN; h++)
   {
      g_stack_b1[r][h] += lr * delta_hidden[h];
      for(int k=0; k<FXAI_STACK_FEATS; k++)
      {
         double reg = (k == 0 ? 0.0 : 0.0004 * g_stack_w1[r][h][k]);
         g_stack_w1[r][h][k] += lr * (delta_hidden[h] * feat[k] - reg);
      }
   }

   g_stack_obs[r]++;
   if(g_stack_obs[r] > 200000) g_stack_obs[r] = 200000;
   g_stack_ready[r] = true;
   FXAI_MarkMetaArtifactsDirty();
}

double FXAI_TradeGatePredict(const int regime_id,
                             const int horizon_minutes,
                             const double &feat[])
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;
   int slot = FXAI_GetHorizonSlot(horizon_minutes);

   double heuristic = FXAI_Clamp(0.46 +
                                 0.16 * feat[7] +
                                 0.12 * feat[20] +
                                 0.12 * feat[21] +
                                 0.10 * feat[23] +
                                 0.10 * feat[53] +
                                 0.08 * feat[54] -
                                 0.10 * feat[3] -
                                 0.08 * feat[49] -
                                 0.08 * feat[50] +
                                 0.08 * feat[62] +
                                 0.06 * feat[70] -
                                 0.08 * feat[57] -
                                 0.06 * feat[60] -
                                 0.04 * feat[63] +
                                 0.06 * feat[72] +
                                 0.06 * feat[75] +
                                 0.08 * feat[77] +
                                 0.08 * feat[79] +
                                 0.10 * feat[80] +
                                 0.10 * feat[81] +
                                 0.10 * feat[82] +
                                 0.08 * feat[83],
                                 0.01,
                                 0.99);
   double oof_prior = FXAI_GetOOFTradeGatePrior(r, slot);
   if(oof_prior >= 0.0)
      heuristic = FXAI_Clamp(0.78 * heuristic + 0.22 * oof_prior, 0.01, 0.99);
   if(!g_trade_gate_ready[r])
      return heuristic;

   double hidden[FXAI_TRADE_GATE_HIDDEN];
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
   {
      double z = g_trade_gate_b1[r][h];
      for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
         z += g_trade_gate_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z = g_trade_gate_b2[r];
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
      z += g_trade_gate_w2[r][h] * hidden[h];
   double learned = FXAI_Sigmoid(z);
   double mix = FXAI_Clamp((double)g_trade_gate_obs[r] / 180.0, 0.20, 0.85);
   double pred = FXAI_Clamp((1.0 - mix) * heuristic + mix * learned, 0.0, 1.0);
   if(oof_prior >= 0.0)
   {
      double oof_mix = FXAI_Clamp((double)g_meta_oof_obs[r][slot] / 96.0, 0.08, 0.28);
      pred = FXAI_Clamp((1.0 - oof_mix) * pred + oof_mix * oof_prior, 0.0, 1.0);
   }
   return pred;
}

double FXAI_TradeGatePredict(const int regime_id,
                             const double &feat[])
{
   return FXAI_TradeGatePredict(regime_id, PredictionTargetMinutes, feat);
}

void FXAI_TradeGateUpdate(const int regime_id,
                          const bool trade_target,
                          const double &feat[],
                          const double sample_weight)
{
   int r = regime_id;
   if(r < 0 || r >= FXAI_REGIME_COUNT) r = 0;

   double hidden[FXAI_TRADE_GATE_HIDDEN];
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
   {
      double z = g_trade_gate_b1[r][h];
      for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
         z += g_trade_gate_w1[r][h][k] * feat[k];
      hidden[h] = FXAI_Tanh(z);
   }

   double z = g_trade_gate_b2[r];
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
      z += g_trade_gate_w2[r][h] * hidden[h];
   double p = FXAI_Sigmoid(z);
   double target = (trade_target ? 1.0 : 0.0);
   double err = FXAI_Clamp((target - p) * FXAI_Clamp(sample_weight, 0.20, 8.00), -3.0, 3.0);
   double lr = 0.020 / MathSqrt(1.0 + 0.02 * (double)g_trade_gate_obs[r]);
   lr = FXAI_Clamp(lr, 0.0015, 0.020);

   double w2_old[FXAI_TRADE_GATE_HIDDEN];
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
      w2_old[h] = g_trade_gate_w2[r][h];

   g_trade_gate_b2[r] += lr * err;
   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
   {
      double reg2 = 0.0006 * g_trade_gate_w2[r][h];
      g_trade_gate_w2[r][h] += lr * (err * hidden[h] - reg2);
   }

   for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
   {
      double dh = (1.0 - hidden[h] * hidden[h]) * w2_old[h] * err;
      g_trade_gate_b1[r][h] += lr * dh;
      for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
      {
         double reg1 = (k == 0 ? 0.0 : 0.0004 * g_trade_gate_w1[r][h][k]);
         g_trade_gate_w1[r][h][k] += lr * (dh * feat[k] - reg1);
      }
   }

   g_trade_gate_obs[r]++;
   if(g_trade_gate_obs[r] > 200000) g_trade_gate_obs[r] = 200000;
   g_trade_gate_ready[r] = true;
   FXAI_MarkMetaArtifactsDirty();
}

void FXAI_ResetStackPending()
{
   int default_h = FXAI_ClampHorizon(PredictionTargetMinutes);
   g_stack_pending_head = 0;
   g_stack_pending_tail = 0;
   for(int k=0; k<FXAI_REL_MAX_PENDING; k++)
   {
      g_stack_pending_seq[k] = -1;
      g_stack_pending_signal[k] = -1;
      g_stack_pending_regime[k] = -1;
      g_stack_pending_horizon[k] = default_h;
      g_stack_pending_prob[k][0] = 0.0;
      g_stack_pending_prob[k][1] = 0.0;
      g_stack_pending_prob[k][2] = 0.0;
      g_stack_pending_expected_move[k] = 0.0;
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = 0.0;
   }
}

void FXAI_ResetAdaptiveRoutingState()
{
   for(int ai=0; ai<FXAI_AI_COUNT; ai++)
   {
      for(int r=0; r<FXAI_REGIME_COUNT; r++)
      {
         g_model_thr_regime_ready[ai][r] = false;
         g_model_buy_thr_regime[ai][r] = g_model_buy_thr[ai];
         g_model_sell_thr_regime[ai][r] = g_model_sell_thr[ai];
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_model_horizon_edge_ema[ai][h] = 0.0;
         g_model_horizon_edge_ready[ai][h] = false;
         g_model_horizon_obs[ai][h] = 0;
      }
   }

   for(int r=0; r<FXAI_REGIME_COUNT; r++)
   {
      g_horizon_regime_total_obs[r] = 0.0;
      g_stack_ready[r] = false;
      g_stack_obs[r] = 0;
      g_trade_gate_ready[r] = false;
      g_trade_gate_obs[r] = 0;
      g_hpolicy_ready[r] = false;
      g_hpolicy_obs[r] = 0;
      for(int h=0; h<FXAI_STACK_HIDDEN; h++)
      {
         for(int k=0; k<FXAI_STACK_FEATS; k++)
            g_stack_w1[r][h][k] = 0.0;
         g_stack_b1[r][h] = 0.0;
      }
      for(int c=0; c<3; c++)
      {
         g_stack_b2[r][c] = 0.0;
         for(int h=0; h<FXAI_STACK_HIDDEN; h++)
            g_stack_w2[r][c][h] = 0.0;
      }
      for(int slot=0; slot<FXAI_MAX_HORIZONS; slot++)
      {
         for(int c=0; c<3; c++)
         {
            g_router_action_value[r][slot][c] = 0.0;
            g_router_action_regret[r][slot][c] = 0.0;
            g_router_action_counterfactual[r][slot][c] = 0.0;
            g_router_action_ready[r][slot][c] = false;
            g_router_action_obs[r][slot][c] = 0;
         }
      }
      g_trade_gate_b2[r] = 0.0;
      for(int h=0; h<FXAI_TRADE_GATE_HIDDEN; h++)
      {
         g_trade_gate_b1[r][h] = 0.0;
         g_trade_gate_w2[r][h] = 0.0;
         for(int k=0; k<FXAI_TRADE_GATE_FEATS; k++)
            g_trade_gate_w1[r][h][k] = 0.0;
      }
      g_hpolicy_b2[r] = 0.0;
      for(int h=0; h<FXAI_HPOL_HIDDEN; h++)
      {
         g_hpolicy_b1[r][h] = 0.0;
         g_hpolicy_w2[r][h] = 0.0;
         for(int k=0; k<FXAI_HPOL_FEATS; k++)
            g_hpolicy_w1[r][h][k] = 0.0;
      }
      for(int h=0; h<FXAI_MAX_HORIZONS; h++)
      {
         g_horizon_regime_edge_ema[r][h] = 0.0;
         g_horizon_regime_edge_ready[r][h] = false;
         g_horizon_regime_obs[r][h] = 0;
         g_meta_oof_score_ema[r][h] = 0.0;
         g_meta_oof_edge_ema[r][h] = 0.0;
         g_meta_oof_quality_ema[r][h] = 0.0;
         g_meta_oof_trade_rate_ema[r][h] = 0.0;
         g_meta_oof_ready[r][h] = false;
         g_meta_oof_obs[r][h] = 0;
      }
   }
}

void FXAI_EnqueueStackPending(const int signal_seq,
                              const int signal,
                              const int regime_id,
                              const int horizon_minutes,
                              const double expected_move_points,
                              const double &probs[],
                              const double &feat[])
{
   if(signal_seq < 0) return;
   int h = FXAI_ClampHorizon(horizon_minutes);
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   int prev = tail - 1;
   if(prev < 0) prev += FXAI_REL_MAX_PENDING;

   if(head != tail && g_stack_pending_seq[prev] == signal_seq)
   {
      g_stack_pending_signal[prev] = signal;
      g_stack_pending_regime[prev] = regime_id;
      g_stack_pending_horizon[prev] = h;
      g_stack_pending_expected_move[prev] = expected_move_points;
      g_stack_pending_prob[prev][0] = probs[0];
      g_stack_pending_prob[prev][1] = probs[1];
      g_stack_pending_prob[prev][2] = probs[2];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[prev][j] = feat[j];
      return;
   }

   g_stack_pending_seq[tail] = signal_seq;
   g_stack_pending_signal[tail] = signal;
   g_stack_pending_regime[tail] = regime_id;
   g_stack_pending_horizon[tail] = h;
   g_stack_pending_expected_move[tail] = expected_move_points;
   g_stack_pending_prob[tail][0] = probs[0];
   g_stack_pending_prob[tail][1] = probs[1];
   g_stack_pending_prob[tail][2] = probs[2];
   for(int j=0; j<FXAI_STACK_FEATS; j++)
      g_stack_pending_feat[tail][j] = feat[j];

   int next_tail = tail + 1;
   if(next_tail >= FXAI_REL_MAX_PENDING) next_tail = 0;
   if(next_tail == head)
   {
      head++;
      if(head >= FXAI_REL_MAX_PENDING) head = 0;
      g_stack_pending_head = head;
   }
   g_stack_pending_tail = next_tail;
}

void FXAI_UpdateStackFromPending(const int current_signal_seq,
                                 const FXAIDataSnapshot &snapshot,
                                 const int &spread_m1[],
                                 const double &high_arr[],
                                 const double &low_arr[],
                                 const double &close_arr[],
                                 const double commission_points,
                                 const double cost_buffer_points,
                                 const double ev_threshold_points)
{
   if(current_signal_seq < 0) return;
   FXAIExecutionProfile exec_profile;
   FXAI_ResolveExecutionProfile(exec_profile);
   int head = g_stack_pending_head;
   int tail = g_stack_pending_tail;
   if(head == tail) return;

   int keep_seq[];
   int keep_signal[];
   int keep_regime[];
   int keep_horizon[];
   double keep_expected[];
   double keep_prob0[];
   double keep_prob1[];
   double keep_prob2[];
   double keep_feat[][FXAI_STACK_FEATS];
   ArrayResize(keep_seq, 0);
   ArrayResize(keep_signal, 0);
   ArrayResize(keep_regime, 0);
   ArrayResize(keep_horizon, 0);
   ArrayResize(keep_expected, 0);
   ArrayResize(keep_prob0, 0);
   ArrayResize(keep_prob1, 0);
   ArrayResize(keep_prob2, 0);
   ArrayResize(keep_feat, 0);

   int idx = head;
   while(idx != tail)
   {
      int seq_pred = g_stack_pending_seq[idx];
      int pending_signal = g_stack_pending_signal[idx];
      int pending_regime = g_stack_pending_regime[idx];
      int pending_h = FXAI_ClampHorizon(g_stack_pending_horizon[idx]);
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
            if(idx_pred >= 0 && idx_pred < ArraySize(close_arr) &&
               idx_pred < ArraySize(high_arr) &&
               idx_pred < ArraySize(low_arr))
            {
               double spread_i = FXAI_GetSpreadAtIndex(idx_pred, spread_m1, snapshot.spread_points);
               double min_move_i = FXAI_ExecutionEntryCostPoints(spread_i,
                                                                 commission_points,
                                                                 cost_buffer_points,
                                                                 exec_profile);
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
               double feat[FXAI_STACK_FEATS];
               for(int j=0; j<FXAI_STACK_FEATS; j++)
                  feat[j] = g_stack_pending_feat[idx][j];
               double sw = FXAI_MoveEdgeWeight(move_points, min_move_i);
               double speed_bonus = 1.0 - FXAI_Clamp(time_to_hit_frac, 0.0, 1.0);
               double mae_ratio = FXAI_Clamp(mae_points / MathMax(mfe_points, min_move_i), 0.0, 3.0);
               double quality = 1.0 + 0.25 * speed_bonus - 0.15 * mae_ratio;
               if((path_flags & FXAI_PATHFLAG_DUAL_HIT) != 0) quality -= 0.10;
               if((path_flags & FXAI_PATHFLAG_SPREAD_STRESS) != 0) quality -= 0.08;
               sw *= FXAI_Clamp(quality, 0.25, 1.75);
               if(pending_signal == -1 && label_class != (int)FXAI_LABEL_SKIP)
                  sw *= 0.80;
               FXAI_StackUpdate(pending_regime, label_class, feat, sw);

               double realized_edge = 0.0;
               if(label_class == (int)FXAI_LABEL_BUY)
                  realized_edge = move_points - min_move_i;
               else if(label_class == (int)FXAI_LABEL_SELL)
                  realized_edge = -move_points - min_move_i;
               else
                  realized_edge = -MathMax(MathAbs(move_points) - min_move_i, 0.0);
               bool trade_target = (label_class != (int)FXAI_LABEL_SKIP &&
                                    realized_edge > 0.0 &&
                                    quality > 0.70 &&
                                    time_to_hit_frac < 0.95);
               double pred_probs[3];
               pred_probs[0] = g_stack_pending_prob[idx][0];
               pred_probs[1] = g_stack_pending_prob[idx][1];
               pred_probs[2] = g_stack_pending_prob[idx][2];
               FXAI_StackRouterObserve(pending_regime,
                                       pending_h,
                                       label_class,
                                       realized_edge,
                                       quality,
                                       feat,
                                       pred_probs,
                                       sw);
               FXAI_TradeGateUpdate(pending_regime, trade_target, feat, sw);
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
            ArrayResize(keep_feat, ks + 1);

            keep_seq[ks] = g_stack_pending_seq[idx];
            keep_signal[ks] = g_stack_pending_signal[idx];
            keep_regime[ks] = g_stack_pending_regime[idx];
            keep_horizon[ks] = g_stack_pending_horizon[idx];
            keep_expected[ks] = g_stack_pending_expected_move[idx];
            keep_prob0[ks] = g_stack_pending_prob[idx][0];
            keep_prob1[ks] = g_stack_pending_prob[idx][1];
            keep_prob2[ks] = g_stack_pending_prob[idx][2];
            for(int j=0; j<FXAI_STACK_FEATS; j++)
               keep_feat[ks][j] = g_stack_pending_feat[idx][j];
         }
      }

      idx++;
      if(idx >= FXAI_REL_MAX_PENDING) idx = 0;
   }

   FXAI_ResetStackPending();
   int keep_n = ArraySize(keep_seq);
   int queue_cap = FXAI_REL_MAX_PENDING - 1;
   if(queue_cap < 0) queue_cap = 0;
   if(keep_n > queue_cap) keep_n = queue_cap;
   for(int k=0; k<keep_n; k++)
   {
      g_stack_pending_seq[k] = keep_seq[k];
      g_stack_pending_signal[k] = keep_signal[k];
      g_stack_pending_regime[k] = keep_regime[k];
      g_stack_pending_horizon[k] = keep_horizon[k];
      g_stack_pending_expected_move[k] = keep_expected[k];
      g_stack_pending_prob[k][0] = keep_prob0[k];
      g_stack_pending_prob[k][1] = keep_prob1[k];
      g_stack_pending_prob[k][2] = keep_prob2[k];
      for(int j=0; j<FXAI_STACK_FEATS; j++)
         g_stack_pending_feat[k][j] = keep_feat[k][j];
   }
   g_stack_pending_head = 0;
   g_stack_pending_tail = keep_n;
}


#endif // __FXAI_META_STACKER_MQH__
