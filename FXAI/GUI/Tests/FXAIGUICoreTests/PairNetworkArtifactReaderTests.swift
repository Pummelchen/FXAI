import FXAIGUICore
import Foundation
import Testing

struct PairNetworkArtifactReaderTests {
    @Test
    func readerParsesRuntimeAndReportArtifacts() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-pair-network-reader-\(UUID().uuidString)", isDirectory: true)
        let runtimeDirectory = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
        let reportDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/PairNetwork/Reports", isDirectory: true)
        let statusDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/PairNetwork", isDirectory: true)
        try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: statusDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let runtimeState = """
        symbol\tNZDUSD
        generated_at\t1775728200
        decision\tPREFER_ALTERNATIVE_EXPRESSION
        fallback_graph_used\t0
        partial_dependency_data\t0
        graph_stale\t0
        conflict_score\t0.790000
        redundancy_score\t0.830000
        contradiction_score\t0.050000
        concentration_score\t0.570000
        currency_concentration\t0.600000
        factor_concentration\t0.620000
        recommended_size_multiplier\t0.000000
        preferred_expression\tAUDUSD
        currency_exposure_csv\tNZD:1.0000; USD:-1.0000
        factor_exposure_csv\tcommodity_fx:1.0600; risk_on:1.0800
        reasons_csv\tBETTER_ALTERNATIVE_EXPRESSION; HIGH_COMMODITY_BLOC_OVERLAP
        """
        try runtimeState.write(
            to: runtimeDirectory.appendingPathComponent("fxai_pair_network_NZDUSD.tsv"),
            atomically: true,
            encoding: .utf8
        )

        let statusJSON: [String: Any] = [
            "ok": true,
            "generated_at": "2026-04-11T09:15:00Z",
            "graph_mode": "STRUCTURAL_PLUS_EMPIRICAL",
            "pair_count": 54,
            "currency_count": 20,
            "edge_count": 420,
            "fallback_graph_used": false,
            "partial_dependency_data": false,
            "graph_stale": false,
            "action_mode": "AUTO_APPLY",
            "config_path": "/tmp/pair_network_config.json",
            "runtime_config_path": "/tmp/pair_network_config.tsv",
            "runtime_status_path": "/tmp/pair_network_status.tsv",
            "report_path": "/tmp/pair_network_report.json",
        ]
        try JSONSerialization.data(withJSONObject: statusJSON, options: [.prettyPrinted, .sortedKeys])
            .write(to: statusDirectory.appendingPathComponent("pair_network_status.json"))

        let reportJSON: [String: Any] = [
            "generated_at": "2026-04-11T09:15:00Z",
            "graph_mode": "STRUCTURAL_PLUS_EMPIRICAL",
            "pair_count": 54,
            "currency_count": 20,
            "edge_count": 420,
            "pairs": [
                [
                    "pair": "NZDUSD",
                    "base_currency": "NZD",
                    "quote_currency": "USD",
                    "factor_signature": [
                        "commodity_fx": 1.06,
                        "risk_on": 1.08,
                    ],
                    "top_dependencies": [
                        [
                            "pair": "AUDUSD",
                            "structural_score": 0.80,
                            "empirical_score": 0.77,
                            "combined_score": 0.85,
                            "correlation": 0.77,
                            "support": 288,
                            "shared_currencies": ["USD"],
                            "relation": "SHARED_CURRENCY",
                        ],
                    ],
                ],
            ],
            "top_edges": [
                [
                    "source_pair": "NZDUSD",
                    "target_pair": "AUDUSD",
                    "pair": "AUDUSD",
                    "structural_score": 0.80,
                    "empirical_score": 0.77,
                    "combined_score": 0.85,
                    "correlation": 0.77,
                    "support": 288,
                    "shared_currencies": ["USD"],
                    "relation": "SHARED_CURRENCY",
                ],
            ],
            "reason_codes": ["STRUCTURAL_PLUS_EMPIRICAL_GRAPH_READY"],
            "quality_flags": [
                "fallback_graph_used": false,
                "partial_dependency_data": false,
                "graph_stale": false,
            ],
        ]
        try JSONSerialization.data(withJSONObject: reportJSON, options: [.prettyPrinted, .sortedKeys])
            .write(to: reportDirectory.appendingPathComponent("pair_network_report.json"))

        let snapshot = try #require(PairNetworkArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.graphMode == "STRUCTURAL_PLUS_EMPIRICAL")
        #expect(snapshot.actionMode == "AUTO_APPLY")
        #expect(snapshot.symbols.count == 1)
        #expect(snapshot.symbols.first?.decision == "PREFER_ALTERNATIVE_EXPRESSION")
        #expect(snapshot.symbols.first?.preferredExpression == "AUDUSD")
        #expect(snapshot.symbols.first?.currencyExposure.first?.key == "NZD")
        #expect(snapshot.topEdges.first?.sourcePair == "NZDUSD")
        #expect(snapshot.topEdges.first?.targetPair == "AUDUSD")
        #expect(snapshot.pairSummaries.first?.topDependencies.first?.targetPair == "AUDUSD")
    }

    @Test
    func readerReturnsNilWhenArtifactsAreMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-pair-network-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(PairNetworkArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
