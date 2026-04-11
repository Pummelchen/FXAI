import FXAIGUICore
import Foundation
import Testing

struct LabelEngineArtifactReaderTests {
    @Test
    func readerParsesReportAndStatusArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-label-engine-reader-\(UUID().uuidString)", isDirectory: true)
        let reportDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/LabelEngine/Reports", isDirectory: true)
        let statusDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/LabelEngine", isDirectory: true)
        try FileManager.default.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: statusDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let reportJSON: [String: Any] = [
            "generated_at": "2026-04-11T12:00:00Z",
            "artifact_count": 1,
            "latest_dataset_key": "continuous:EURUSD:m1:labels",
            "builds": [
                [
                    "dataset_key": "continuous:EURUSD:m1:labels",
                    "profile_name": "continuous",
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "bar_count": 96,
                    "point_size": 0.00001,
                    "execution_profile": "default",
                    "label_version": 1,
                    "generated_at": "2026-04-11T11:58:00Z",
                    "summary_metrics": [
                        "meta_acceptance_rate": 0.43,
                        "long_tradeability_rate": 0.61,
                    ],
                    "meta_summary": [
                        "candidate_mode": "BASELINE_MOMENTUM",
                        "candidate_count": 28,
                    ],
                    "quality_flags": [
                        "path_approximation_used": true,
                        "partial_cost_model": true,
                    ],
                    "artifact_paths": [
                        "bundle_json": "/tmp/label_bundle.json",
                        "labels_ndjson": "/tmp/labels.ndjson",
                    ],
                    "top_reason_codes": [
                        ["reason": "MOVE_TOO_SMALL_AFTER_COSTS", "count": 44],
                    ],
                    "horizon_summaries": [
                        [
                            "horizon_id": "M15",
                            "bars": 15,
                            "sample_count": 56,
                            "long_tradeability_rate": 0.61,
                            "short_tradeability_rate": 0.47,
                            "candidate_count": 8,
                            "candidate_acceptance_rate": 0.43,
                            "mean_cost_adjusted_return_points": 4.8,
                            "median_time_to_favorable_hit_sec": 420,
                        ],
                    ],
                ],
            ],
        ]
        let statusJSON: [String: Any] = [
            "artifact_count": 1,
            "latest_dataset_key": "continuous:EURUSD:m1:labels",
            "profile_name": "continuous",
            "artifacts": [
                "report_json": "/tmp/label_engine_report.json",
                "runtime_summary_json": "/tmp/label_engine_summary.json",
            ],
        ]

        let reportData = try JSONSerialization.data(withJSONObject: reportJSON, options: [.prettyPrinted, .sortedKeys])
        let statusData = try JSONSerialization.data(withJSONObject: statusJSON, options: [.prettyPrinted, .sortedKeys])
        try reportData.write(to: reportDirectory.appendingPathComponent("label_engine_report.json"))
        try statusData.write(to: statusDirectory.appendingPathComponent("label_engine_status.json"))

        let snapshot = try #require(LabelEngineArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.artifactCount == 1)
        #expect(snapshot.latestDatasetKey == "continuous:EURUSD:m1:labels")
        #expect(snapshot.builds.count == 1)
        #expect(snapshot.builds.first?.symbol == "EURUSD")
        #expect(snapshot.builds.first?.horizons.first?.horizonID == "M15")
        #expect(snapshot.builds.first?.topReasons.first?.reason == "MOVE_TOO_SMALL_AFTER_COSTS")
    }

    @Test
    func readerReturnsNilWhenReportIsMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-label-engine-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(LabelEngineArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
