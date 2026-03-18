#ifndef __FXAI_AUDIT_REPORT_MQH__
#define __FXAI_AUDIT_REPORT_MQH__

bool FXAI_AuditWriteHeader(const int handle)
{
   return FileWrite(handle,
                    "ai_id\tai_name\tfamily\tscenario\tbars_total\tsamples_total\tvalid_preds\tinvalid_preds\tbuy_count\tsell_count\tskip_count\ttrue_buy_count\ttrue_sell_count\ttrue_skip_count\texact_match_count\tdirectional_eval_count\tdirectional_correct_count\tskip_ratio\tactive_ratio\tbias_abs\tconf_drift\tbrier_score\tcalibration_error\tpath_quality_error\treset_delta\tsequence_delta\twf_folds\twf_train_samples\twf_test_samples\twf_train_score\twf_test_score\twf_test_score_std\twf_gap\twf_pbo\twf_dsr\twf_pass_rate\tscore\tissue_flags\tavg_conf\tavg_rel\tavg_move\ttrend_align") > 0;
}

bool FXAI_AuditWriteMetrics(const int handle,
                            const FXAIAuditScenarioMetrics &m)
{
   double avg_conf = (m.valid_preds > 0 ? m.conf_sum / (double)m.valid_preds : 0.0);
   double avg_rel = (m.valid_preds > 0 ? m.rel_sum / (double)m.valid_preds : 0.0);
   double avg_move = (m.valid_preds > 0 ? m.move_sum / (double)m.valid_preds : 0.0);
   double trend_align = (m.trend_alignment_count > 0 ? m.trend_alignment_sum / (double)m.trend_alignment_count : 0.0);
   return FileWrite(handle,
                    IntegerToString(m.ai_id) + "\t" +
                    m.ai_name + "\t" +
                    IntegerToString(m.family) + "\t" +
                    m.scenario + "\t" +
                    IntegerToString(m.bars_total) + "\t" +
                    IntegerToString(m.samples_total) + "\t" +
                    IntegerToString(m.valid_preds) + "\t" +
                    IntegerToString(m.invalid_preds) + "\t" +
                    IntegerToString(m.buy_count) + "\t" +
                    IntegerToString(m.sell_count) + "\t" +
                    IntegerToString(m.skip_count) + "\t" +
                    IntegerToString(m.true_buy_count) + "\t" +
                    IntegerToString(m.true_sell_count) + "\t" +
                    IntegerToString(m.true_skip_count) + "\t" +
                    IntegerToString(m.exact_match_count) + "\t" +
                    IntegerToString(m.directional_eval_count) + "\t" +
                    IntegerToString(m.directional_correct_count) + "\t" +
                    DoubleToString(m.skip_ratio, 6) + "\t" +
                    DoubleToString(m.active_ratio, 6) + "\t" +
                    DoubleToString(m.bias_abs, 6) + "\t" +
                    DoubleToString(m.conf_drift, 6) + "\t" +
                    DoubleToString(m.brier_score, 6) + "\t" +
                    DoubleToString(m.calibration_error, 6) + "\t" +
                    DoubleToString(m.path_quality_error, 6) + "\t" +
                    DoubleToString(m.reset_delta, 6) + "\t" +
                    DoubleToString(m.sequence_delta, 6) + "\t" +
                    IntegerToString(m.wf_folds) + "\t" +
                    IntegerToString(m.wf_train_samples) + "\t" +
                    IntegerToString(m.wf_test_samples) + "\t" +
                    DoubleToString(m.wf_train_score, 6) + "\t" +
                    DoubleToString(m.wf_test_score, 6) + "\t" +
                    DoubleToString(m.wf_test_score_std, 6) + "\t" +
                    DoubleToString(m.wf_gap, 6) + "\t" +
                    DoubleToString(m.wf_pbo, 6) + "\t" +
                    DoubleToString(m.wf_dsr, 6) + "\t" +
                    DoubleToString(m.wf_pass_rate, 6) + "\t" +
                    DoubleToString(m.score, 4) + "\t" +
                    IntegerToString(m.issue_flags) + "\t" +
                    DoubleToString(avg_conf, 6) + "\t" +
                    DoubleToString(avg_rel, 6) + "\t" +
                    DoubleToString(avg_move, 6) + "\t" +
                    DoubleToString(trend_align, 6)) > 0;
}


#endif // __FXAI_AUDIT_REPORT_MQH__
