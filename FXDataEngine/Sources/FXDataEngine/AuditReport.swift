import Foundation

public enum AuditReportTools {
    public static let headerColumns = [
        "ai_id",
        "ai_name",
        "family",
        "scenario",
        "bars_total",
        "samples_total",
        "valid_preds",
        "invalid_preds",
        "buy_count",
        "sell_count",
        "skip_count",
        "true_buy_count",
        "true_sell_count",
        "true_skip_count",
        "exact_match_count",
        "directional_eval_count",
        "directional_correct_count",
        "skip_ratio",
        "active_ratio",
        "bias_abs",
        "conf_drift",
        "brier_score",
        "calibration_error",
        "path_quality_error",
        "macro_event_rate",
        "macro_pre_rate",
        "macro_post_rate",
        "macro_importance_mean",
        "macro_surprise_abs_mean",
        "macro_data_coverage",
        "macro_surprise_z_abs_mean",
        "macro_revision_abs_mean",
        "macro_currency_relevance_mean",
        "macro_provenance_trust_mean",
        "macro_rates_rate",
        "macro_inflation_rate",
        "macro_labor_rate",
        "macro_growth_rate",
        "reset_delta",
        "sequence_delta",
        "wf_folds",
        "wf_train_samples",
        "wf_test_samples",
        "wf_train_score",
        "wf_test_score",
        "wf_test_score_std",
        "wf_gap",
        "wf_pbo",
        "wf_dsr",
        "wf_pass_rate",
        "score",
        "issue_flags",
        "avg_conf",
        "avg_rel",
        "avg_move",
        "trend_align"
    ]

    public static var headerLine: String {
        headerColumns.joined(separator: "\t")
    }

    public static func rowColumns(for metrics: AuditScenarioMetrics) -> [String] {
        let valid = Double(max(metrics.validPredictions, 0))
        let averageConfidence = valid > 0.0 ? metrics.confidenceSum / valid : 0.0
        let averageReliability = valid > 0.0 ? metrics.reliabilitySum / valid : 0.0
        let averageMove = valid > 0.0 ? metrics.moveSum / valid : 0.0
        let trendAlignment = metrics.trendAlignmentCount > 0
            ? metrics.trendAlignmentSum / Double(metrics.trendAlignmentCount)
            : 0.0

        return [
            String(metrics.aiID),
            metrics.aiName,
            String(metrics.family),
            metrics.scenario,
            String(metrics.barsTotal),
            String(metrics.samplesTotal),
            String(metrics.validPredictions),
            String(metrics.invalidPredictions),
            String(metrics.buyCount),
            String(metrics.sellCount),
            String(metrics.skipCount),
            String(metrics.trueBuyCount),
            String(metrics.trueSellCount),
            String(metrics.trueSkipCount),
            String(metrics.exactMatchCount),
            String(metrics.directionalEvaluationCount),
            String(metrics.directionalCorrectCount),
            RuntimeArtifactTSV.double(metrics.skipRatio),
            RuntimeArtifactTSV.double(metrics.activeRatio),
            RuntimeArtifactTSV.double(metrics.biasAbs),
            RuntimeArtifactTSV.double(metrics.confidenceDrift),
            RuntimeArtifactTSV.double(metrics.brierScore),
            RuntimeArtifactTSV.double(metrics.calibrationError),
            RuntimeArtifactTSV.double(metrics.pathQualityError),
            RuntimeArtifactTSV.double(metrics.macroEventRate),
            RuntimeArtifactTSV.double(metrics.macroPreRate),
            RuntimeArtifactTSV.double(metrics.macroPostRate),
            RuntimeArtifactTSV.double(metrics.macroImportanceMean),
            RuntimeArtifactTSV.double(metrics.macroSurpriseAbsMean),
            RuntimeArtifactTSV.double(metrics.macroDataCoverage),
            RuntimeArtifactTSV.double(metrics.macroSurpriseZAbsMean),
            RuntimeArtifactTSV.double(metrics.macroRevisionAbsMean),
            RuntimeArtifactTSV.double(metrics.macroCurrencyRelevanceMean),
            RuntimeArtifactTSV.double(metrics.macroProvenanceTrustMean),
            RuntimeArtifactTSV.double(metrics.macroRatesRate),
            RuntimeArtifactTSV.double(metrics.macroInflationRate),
            RuntimeArtifactTSV.double(metrics.macroLaborRate),
            RuntimeArtifactTSV.double(metrics.macroGrowthRate),
            RuntimeArtifactTSV.double(metrics.resetDelta),
            RuntimeArtifactTSV.double(metrics.sequenceDelta),
            String(metrics.walkForwardFolds),
            String(metrics.walkForwardTrainSamples),
            String(metrics.walkForwardTestSamples),
            RuntimeArtifactTSV.double(metrics.walkForwardTrainScore),
            RuntimeArtifactTSV.double(metrics.walkForwardTestScore),
            RuntimeArtifactTSV.double(metrics.walkForwardTestScoreStd),
            RuntimeArtifactTSV.double(metrics.walkForwardGap),
            RuntimeArtifactTSV.double(metrics.walkForwardPBO),
            RuntimeArtifactTSV.double(metrics.walkForwardDSR),
            RuntimeArtifactTSV.double(metrics.walkForwardPassRate),
            RuntimeArtifactTSV.double(metrics.score, decimals: 4),
            String(metrics.issueFlags.rawValue),
            RuntimeArtifactTSV.double(averageConfidence),
            RuntimeArtifactTSV.double(averageReliability),
            RuntimeArtifactTSV.double(averageMove),
            RuntimeArtifactTSV.double(trendAlignment)
        ]
    }

    public static func rowLine(for metrics: AuditScenarioMetrics) -> String {
        rowColumns(for: metrics).map(RuntimeArtifactTSV.field).joined(separator: "\t")
    }

    public static func document(_ metrics: [AuditScenarioMetrics]) -> String {
        RuntimeArtifactTSV.document(header: headerColumns, rows: metrics.map(rowColumns(for:)))
    }
}
