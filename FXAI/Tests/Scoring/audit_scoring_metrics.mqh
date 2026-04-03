double FXAI_AuditApproxNormalCdf(const double x)
{
   return 1.0 / (1.0 + MathExp(-1.702 * x));
}

double FXAI_AuditMean(const double &values[])
{
   int n = ArraySize(values);
   if(n <= 0) return 0.0;
   double sum = 0.0;
   for(int i=0; i<n; i++)
      sum += values[i];
   return sum / (double)n;
}

double FXAI_AuditStd(const double &values[],
                     const double mean)
{
   int n = ArraySize(values);
   if(n <= 1) return 0.0;
   double var = 0.0;
   for(int i=0; i<n; i++)
   {
      double d = values[i] - mean;
      var += d * d;
   }
   var /= (double)(n - 1);
   if(var < 0.0) var = 0.0;
   return MathSqrt(var);
}

double FXAI_AuditScoreFold(const FXAIAuditFoldMetrics &m)
{
   if(m.samples_total < 12 || m.valid_preds <= 0)
      return -1e9;

   double invalid_rate = (double)m.invalid_preds / (double)m.samples_total;
   double active_ratio = (double)(m.buy_count + m.sell_count) / (double)m.samples_total;
   double skip_ratio = (double)m.skip_count / (double)m.samples_total;
   double hit_rate = (m.directional_eval_count > 0 ?
                      (double)m.directional_correct_count / (double)m.directional_eval_count :
                      0.50);
   double brier = m.brier_sum / (double)m.valid_preds;
   double cal = (m.directional_eval_count > 0 ?
                 m.calibration_abs_sum / (double)m.directional_eval_count :
                 0.35);
   double pq = (m.path_quality_count > 0 ?
                m.path_quality_abs_sum / (double)m.path_quality_count :
                0.50);
   double avg_conf = m.conf_sum / (double)m.valid_preds;
   double avg_rel = m.rel_sum / (double)m.valid_preds;
   double avg_move = m.move_sum / (double)m.valid_preds;
   double avg_net = m.net_sum / (double)m.valid_preds;

   double score = 100.0;
   score -= 42.0 * invalid_rate;
   score -= 28.0 * brier;
   score -= 16.0 * cal;
   score -= 12.0 * pq;
   if(active_ratio > 0.78)
      score -= 18.0 * FXAI_Clamp((active_ratio - 0.78) / 0.22, 0.0, 1.0);
   if(active_ratio < 0.05)
      score -= 10.0 * FXAI_Clamp((0.05 - active_ratio) / 0.05, 0.0, 1.0);
   if(skip_ratio > 0.92)
      score -= 6.0 * FXAI_Clamp((skip_ratio - 0.92) / 0.08, 0.0, 1.0);
   score += 24.0 * FXAI_Clamp(hit_rate - 0.50, -0.50, 0.50);
   score += 8.0 * FXAI_Clamp(avg_rel - 0.50, -0.50, 0.50);
   score += 4.0 * FXAI_Clamp(avg_conf - 0.50, -0.50, 0.50);
   score += 4.0 * FXAI_Clamp(avg_move / 8.0, 0.0, 1.0);
   score += 6.0 * FXAI_Clamp(avg_net / 8.0, -1.0, 1.0);
   if(score < 0.0) score = 0.0;
   if(score > 100.0) score = 100.0;
   return score;
}

double FXAI_AuditDeflatedSharpeProxy(const double &scores[],
                                     const double pbo)
{
   int n = ArraySize(scores);
   if(n <= 1) return 0.0;

   double returns[];
   ArrayResize(returns, n);
   for(int i=0; i<n; i++)
      returns[i] = (scores[i] - 60.0) / 20.0;

   double mean_ret = FXAI_AuditMean(returns);
   double std_ret = FXAI_AuditStd(returns, mean_ret);
   if(std_ret <= 1e-9)
      return (mean_ret > 0.0 ? 1.0 : 0.0);

   double sharpe = mean_ret / std_ret;
   double sample_deflator = MathSqrt((double)(n - 1) / (double)(n + 3));
   double selection_penalty = 0.35 + 0.65 * FXAI_Clamp(pbo, 0.0, 1.0);
   double z = sharpe * sample_deflator - selection_penalty - 0.08 * MathLog((double)n + 1.0);
   return FXAI_Clamp(FXAI_AuditApproxNormalCdf(z), 0.0, 1.0);
}

void FXAI_AuditFinalizeWalkForward(FXAIAuditScenarioMetrics &m,
                                   const FXAIAuditFoldMetrics &train_folds[],
                                   const FXAIAuditFoldMetrics &test_folds[])
{
   int count = MathMin(ArraySize(train_folds), ArraySize(test_folds));
   if(count <= 0)
      return;

   double train_scores[];
   double test_scores[];
   int pass_count = 0;
   int overfit_count = 0;

   for(int f=0; f<count; f++)
   {
      m.wf_train_samples += train_folds[f].samples_total;
      m.wf_test_samples += test_folds[f].samples_total;

      double train_score = FXAI_AuditScoreFold(train_folds[f]);
      double test_score = FXAI_AuditScoreFold(test_folds[f]);
      if(train_score <= -1e8 || test_score <= -1e8)
         continue;

      int train_sz = ArraySize(train_scores);
      ArrayResize(train_scores, train_sz + 1);
      train_scores[train_sz] = train_score;

      int test_sz = ArraySize(test_scores);
      ArrayResize(test_scores, test_sz + 1);
      test_scores[test_sz] = test_score;

      if(test_score + 6.0 < train_score)
         overfit_count++;
      if(test_score >= 68.0 && test_score + 8.0 >= train_score)
         pass_count++;
   }

   m.wf_folds = ArraySize(test_scores);
   if(m.wf_folds <= 0)
      return;

   m.wf_train_score = FXAI_AuditMean(train_scores);
   m.wf_test_score = FXAI_AuditMean(test_scores);
   m.wf_test_score_std = FXAI_AuditStd(test_scores, m.wf_test_score);
   m.wf_gap = m.wf_train_score - m.wf_test_score;
   m.wf_pbo = (double)overfit_count / (double)m.wf_folds;
   m.wf_pass_rate = (double)pass_count / (double)m.wf_folds;
   m.wf_dsr = FXAI_AuditDeflatedSharpeProxy(test_scores, m.wf_pbo);
}

void FXAI_AuditFinalizeMetrics(FXAIAuditScenarioMetrics &m)
{
   if(m.samples_total > 0)
   {
      m.skip_ratio = (double)m.skip_count / (double)m.samples_total;
      m.active_ratio = (double)(m.buy_count + m.sell_count) / (double)m.samples_total;
   }
   int active = m.buy_count + m.sell_count;
   if(active > 0)
      m.bias_abs = MathAbs((double)m.buy_count - (double)m.sell_count) / (double)active;
   if(m.directional_eval_count > 0)
   {
      double avg_conf = m.dir_conf_sum / (double)m.directional_eval_count;
      double avg_hit = m.dir_hit_sum / (double)m.directional_eval_count;
      m.conf_drift = MathAbs(avg_conf - avg_hit);
      m.calibration_error = m.calibration_abs_sum / (double)m.directional_eval_count;
   }
   if(m.valid_preds > 0)
      m.brier_score = m.brier_sum / (double)m.valid_preds;
   if(m.path_quality_count > 0)
      m.path_quality_error = m.path_quality_abs_sum / (double)m.path_quality_count;
   if(m.samples_total > 0)
   {
      double denom = (double)m.samples_total;
      m.macro_event_rate /= denom;
      m.macro_pre_rate /= denom;
      m.macro_post_rate /= denom;
      m.macro_importance_mean /= denom;
      m.macro_surprise_abs_mean /= denom;
      m.macro_data_coverage /= denom;
      m.macro_surprise_z_abs_mean /= denom;
      m.macro_revision_abs_mean /= denom;
      m.macro_currency_relevance_mean /= denom;
      m.macro_provenance_trust_mean /= denom;
      m.macro_rates_rate /= denom;
      m.macro_inflation_rate /= denom;
      m.macro_labor_rate /= denom;
      m.macro_growth_rate /= denom;
   }
   double avg_net = (m.valid_preds > 0 ? m.net_sum / (double)m.valid_preds : 0.0);
   double hit_rate = (m.directional_eval_count > 0 ?
                      (double)m.directional_correct_count / (double)m.directional_eval_count :
                      0.50);
   bool macro_dataset_active = FXAI_HasMacroEventDataset();
   bool macro_dataset_safe = (macro_dataset_active && FXAI_MacroEventLeakageSafe());

   double score = 100.0;
   if(m.invalid_preds > 0) score -= 35.0;
   if(m.skip_ratio < 0.45 && (m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock")) score -= 18.0;
   if(m.active_ratio > 0.80 && (m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock")) score -= 12.0;
   if(m.trend_alignment_count > 0)
   {
      double align = m.trend_alignment_sum / (double)m.trend_alignment_count;
      if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down" || m.scenario == "market_trend" || m.scenario == "market_walkforward") && align < 0.20)
         score -= 18.0;
   }
   if(m.scenario == "market_session_edges" && m.conf_drift > 0.18) score -= 8.0;
   if(m.conf_drift > 0.22) score -= 10.0;
   if(m.brier_score > 0.52) score -= 8.0;
   if(m.calibration_error > 0.28) score -= 8.0;
   if(m.path_quality_error > 0.55) score -= 8.0;
   if(m.scenario == "market_macro_event")
   {
      if(!macro_dataset_active)
      {
         // Keep the scenario neutral when no macro dataset is configured.
      }
      else if(!macro_dataset_safe)
      {
         score -= 22.0;
      }
      else
      {
         if(m.macro_data_coverage < 0.08) score -= 20.0;
         if(m.macro_event_rate < 0.06) score -= 16.0;
         if(m.macro_importance_mean < 0.08) score -= 10.0;
         if(m.macro_currency_relevance_mean < 0.40) score -= 8.0;
         if(m.macro_provenance_trust_mean < 0.45) score -= 8.0;
         if(m.active_ratio < 0.05 && m.macro_event_rate > 0.10) score -= 8.0;
         if(m.active_ratio > 0.88 && m.macro_surprise_abs_mean < 0.20) score -= 8.0;
         if(avg_net < 0.0) score -= 10.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
      }
   }
   if((m.scenario == "market_session_edges" || m.scenario == "market_spread_shock" || m.scenario == "market_walkforward") && avg_net < 0.0)
      score -= 8.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
   if(m.scenario == "market_adversarial")
   {
      if(hit_rate < 0.53) score -= 12.0 * FXAI_Clamp((0.53 - hit_rate) / 0.18, 0.0, 1.0);
      if(m.conf_drift > 0.20) score -= 8.0;
      if(m.calibration_error > 0.26) score -= 10.0;
      if(m.path_quality_error > 0.50) score -= 10.0;
      if(avg_net < 0.0) score -= 12.0 * FXAI_Clamp(-avg_net / 4.0, 0.0, 1.0);
      if(m.active_ratio < 0.03) score -= 6.0;
      if(m.active_ratio > 0.90 && m.brier_score > 0.42) score -= 8.0;
   }
   if(m.reset_delta > 0.30) score -= 12.0;
   if(m.sequence_delta < 0.005 && m.sequence_delta >= 0.0) score -= 6.0;
   if(m.move_sum <= 0.0) score -= 8.0;

   if(m.scenario == "market_walkforward")
   {
      if(m.wf_folds < 3) score -= 18.0;
      if(m.wf_gap > 12.0) score -= 10.0;
      if(m.wf_pbo > 0.45) score -= 12.0;
      if(m.wf_pass_rate < 0.55) score -= 12.0;
      if(m.wf_dsr < 0.35) score -= 10.0;
      if(m.wf_test_score > 0.0 && m.wf_test_score < 68.0) score -= 10.0;
      if(m.wf_test_score_std > 10.0) score -= 6.0;
   }

   if(score < 0.0) score = 0.0;
   m.score = score;

   if(m.invalid_preds > 0) m.issue_flags |= FXAI_AUDIT_ISSUE_INVALID_PRED;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock" || m.scenario == "market_session_edges") && (m.skip_ratio < 0.55 || m.active_ratio > 0.70))
      m.issue_flags |= FXAI_AUDIT_ISSUE_OVERTRADES_NOISE;
   if(m.scenario == "market_macro_event")
   {
      if(macro_dataset_active)
      {
         if(!macro_dataset_safe || m.macro_data_coverage < 0.05)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_DATA_GAP;
         if(macro_dataset_safe && m.active_ratio < 0.05 && m.macro_event_rate > 0.10)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_BLIND;
         if(macro_dataset_safe && m.active_ratio > 0.88 && m.macro_surprise_abs_mean < 0.20)
            m.issue_flags |= FXAI_AUDIT_ISSUE_MACRO_OVERREACT;
      }
   }
   if((m.scenario == "drift_up" || m.scenario == "drift_down" || m.scenario == "monotonic_up" || m.scenario == "monotonic_down" || m.scenario == "market_trend" || m.scenario == "market_walkforward") &&
      m.trend_alignment_count > 0 && (m.trend_alignment_sum / (double)m.trend_alignment_count) < 0.25)
      m.issue_flags |= FXAI_AUDIT_ISSUE_MISSES_TREND;
   if(m.conf_drift > 0.22) m.issue_flags |= FXAI_AUDIT_ISSUE_CALIBRATION_DRIFT;
   if(m.reset_delta > 0.30) m.issue_flags |= FXAI_AUDIT_ISSUE_RESET_DRIFT;
   if(m.sequence_delta >= 0.0 && m.sequence_delta < 0.005) m.issue_flags |= FXAI_AUDIT_ISSUE_SEQUENCE_WEAK;
   if(m.move_sum <= 0.0) m.issue_flags |= FXAI_AUDIT_ISSUE_DEAD_OUTPUT;
   if((m.scenario == "random_walk" || m.scenario == "market_chop" || m.scenario == "market_spread_shock") && m.bias_abs > 0.85 && active > 24)
      m.issue_flags |= FXAI_AUDIT_ISSUE_SIDE_COLLAPSE;
   if(m.scenario == "market_adversarial")
   {
      if(m.score < 68.0 || avg_net < 0.0 || hit_rate < 0.53 || m.calibration_error > 0.26 || m.path_quality_error > 0.50)
         m.issue_flags |= FXAI_AUDIT_ISSUE_ADVERSARIAL_WEAK;
   }

   if(m.scenario == "market_walkforward")
   {
      if(m.wf_pbo > 0.45 || m.wf_gap > 12.0)
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_OVERFIT;
      if(m.wf_folds < 3 || m.wf_pass_rate < 0.55 || m.wf_test_score_std > 10.0)
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_UNSTABLE;
      if(m.wf_dsr < 0.35 || (m.wf_test_score > 0.0 && m.wf_test_score < 68.0))
         m.issue_flags |= FXAI_AUDIT_ISSUE_WF_WEAK_EDGE;
   }
}

