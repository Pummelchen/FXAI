import XCTest
@testable import FXDataEngine

final class AuditReportTests: XCTestCase {
    func testAuditReportHeaderMatchesLegacyColumnOrder() {
        XCTAssertEqual(AuditReportTools.headerColumns.first, "ai_id")
        XCTAssertEqual(AuditReportTools.headerColumns.last, "trend_align")
        XCTAssertEqual(AuditReportTools.headerColumns.count, 56)
        XCTAssertEqual(
            AuditReportTools.headerLine,
            "ai_id\tai_name\tfamily\tscenario\tbars_total\tsamples_total\tvalid_preds\tinvalid_preds\tbuy_count\tsell_count\tskip_count\ttrue_buy_count\ttrue_sell_count\ttrue_skip_count\texact_match_count\tdirectional_eval_count\tdirectional_correct_count\tskip_ratio\tactive_ratio\tbias_abs\tconf_drift\tbrier_score\tcalibration_error\tpath_quality_error\tmacro_event_rate\tmacro_pre_rate\tmacro_post_rate\tmacro_importance_mean\tmacro_surprise_abs_mean\tmacro_data_coverage\tmacro_surprise_z_abs_mean\tmacro_revision_abs_mean\tmacro_currency_relevance_mean\tmacro_provenance_trust_mean\tmacro_rates_rate\tmacro_inflation_rate\tmacro_labor_rate\tmacro_growth_rate\treset_delta\tsequence_delta\twf_folds\twf_train_samples\twf_test_samples\twf_train_score\twf_test_score\twf_test_score_std\twf_gap\twf_pbo\twf_dsr\twf_pass_rate\tscore\tissue_flags\tavg_conf\tavg_rel\tavg_move\ttrend_align"
        )
    }

    func testAuditReportRowFormatsLegacyMetrics() {
        let metrics = AuditScenarioMetrics(
            aiID: 7,
            aiName: "ai_lstm",
            family: AIFamily.recurrent.rawValue,
            scenario: "trend_persistence",
            barsTotal: 2_048,
            samplesTotal: 100,
            validPredictions: 5,
            invalidPredictions: 1,
            buyCount: 2,
            sellCount: 1,
            skipCount: 2,
            trueBuyCount: 3,
            trueSellCount: 2,
            trueSkipCount: 1,
            exactMatchCount: 4,
            directionalEvaluationCount: 3,
            directionalCorrectCount: 2,
            trendAlignmentSum: 1.5,
            trendAlignmentCount: 3,
            confidenceSum: 3.0,
            reliabilitySum: 4.0,
            moveSum: 10.0,
            skipRatio: 0.20,
            activeRatio: 0.60,
            biasAbs: 0.3333333,
            confidenceDrift: 0.125,
            brierScore: 0.2345678,
            calibrationError: 0.3456789,
            pathQualityError: 0.4567891,
            macroEventRate: 0.10,
            macroPreRate: 0.20,
            macroPostRate: 0.30,
            macroImportanceMean: 0.40,
            macroSurpriseAbsMean: 0.50,
            macroDataCoverage: 0.60,
            macroSurpriseZAbsMean: 0.70,
            macroRevisionAbsMean: 0.80,
            macroCurrencyRelevanceMean: 0.90,
            macroProvenanceTrustMean: 1.00,
            macroRatesRate: 0.11,
            macroInflationRate: 0.12,
            macroLaborRate: 0.13,
            macroGrowthRate: 0.14,
            resetDelta: 0.15,
            sequenceDelta: 0.16,
            walkForwardFolds: 2,
            walkForwardTrainSamples: 60,
            walkForwardTestSamples: 40,
            walkForwardTrainScore: 70.1234567,
            walkForwardTestScore: 68.7654321,
            walkForwardTestScoreStd: 1.234567,
            walkForwardGap: 1.358024,
            walkForwardPBO: 0.25,
            walkForwardDSR: 0.75,
            walkForwardPassRate: 0.50,
            score: 88.12346,
            issueFlags: [.invalidPrediction, .macroBlind]
        )

        let columns = AuditReportTools.rowColumns(for: metrics)
        XCTAssertEqual(columns.count, AuditReportTools.headerColumns.count)
        let byHeader = Dictionary(uniqueKeysWithValues: zip(AuditReportTools.headerColumns, columns))
        XCTAssertEqual(byHeader["ai_id"], "7")
        XCTAssertEqual(byHeader["family"], String(AIFamily.recurrent.rawValue))
        XCTAssertEqual(byHeader["bars_total"], "2048")
        XCTAssertEqual(byHeader["true_buy_count"], "3")
        XCTAssertEqual(byHeader["skip_ratio"], "0.200000")
        XCTAssertEqual(byHeader["bias_abs"], "0.333333")
        XCTAssertEqual(byHeader["wf_train_score"], "70.123457")
        XCTAssertEqual(byHeader["score"], "88.1235")
        XCTAssertEqual(byHeader["issue_flags"], String((AuditIssueFlags.invalidPrediction.rawValue | AuditIssueFlags.macroBlind.rawValue)))
        XCTAssertEqual(byHeader["avg_conf"], "0.600000")
        XCTAssertEqual(byHeader["avg_rel"], "0.800000")
        XCTAssertEqual(byHeader["avg_move"], "2.000000")
        XCTAssertEqual(byHeader["trend_align"], "0.500000")
    }

    func testAuditReportDocumentEscapesFieldsAndTerminatesWithCRLF() {
        let metrics = AuditScenarioMetrics(
            aiID: 1,
            aiName: "ai\tname",
            family: 2,
            scenario: "scenario\nname",
            barsTotal: 10,
            samplesTotal: 1
        )
        let document = AuditReportTools.document([metrics])
        XCTAssertTrue(document.hasSuffix("\r\n"))
        XCTAssertTrue(document.contains("1\tai name\t2\tscenario name\t10\t1"))
    }

    func testWalkForwardDiagnosticsJSONContainsPerFoldEvidence() throws {
        let metrics = AuditScenarioMetrics(
            aiID: 4,
            aiName: "ai_mlp",
            family: 2,
            scenario: "market_walkforward",
            walkForwardFolds: 2,
            walkForwardFoldEvidence: [
                AuditWalkForwardFoldEvidence(
                    fold: 1,
                    trainSamples: 128,
                    testSamples: 64,
                    trainScore: 82.5,
                    testScore: 78.0,
                    gap: 4.5,
                    passed: true,
                    overfit: false
                ),
                AuditWalkForwardFoldEvidence(
                    fold: 2,
                    trainSamples: 128,
                    testSamples: 64,
                    trainScore: 84.0,
                    testScore: 66.0,
                    gap: 18.0,
                    passed: false,
                    overfit: true
                )
            ]
        )

        let diagnostics = AuditReportTools.walkForwardDiagnostics([metrics])
        XCTAssertEqual(diagnostics.schemaVersion, 1)
        XCTAssertEqual(diagnostics.plugins.count, 1)
        XCTAssertEqual(diagnostics.plugins[0].windows.count, 2)

        let json = try AuditReportTools.walkForwardDiagnosticsJSON([metrics])
        XCTAssertTrue(json.contains(#""aiName" : "ai_mlp""#))
        XCTAssertTrue(json.contains(#""testScore" : 66"#))
        XCTAssertTrue(json.contains(#""overfit" : true"#))
    }
}
