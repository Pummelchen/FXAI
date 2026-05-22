import FXAIGUICore
import Foundation
import Testing

struct CrossAssetArtifactReaderTests {
    @Test
    func readerParsesRuntimeSnapshotAndStatus() throws {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-cross-asset-reader-\(UUID().uuidString)", isDirectory: true)
        let runtimeDirectory = projectRoot
            .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
        let statusDirectory = projectRoot
            .appendingPathComponent("Tools/OfflineLab/CrossAsset", isDirectory: true)

        try FileManager.default.createDirectory(at: runtimeDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: statusDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        let snapshotJSON: [String: Any] = [
            "generated_at": "2026-04-11T10:00:00Z",
            "source_status": [
                "rates": [
                    "ok": true,
                    "stale": false,
                    "last_update_at": "2026-04-11T09:59:30Z",
                ],
                "equities": [
                    "ok": true,
                    "stale": false,
                    "proxy_symbol": "US500",
                ],
            ],
            "features": [
                "front_end_rate_divergence_z": 1.42,
                "volatility_stress_z": 1.63,
            ],
            "state_scores": [
                "rates_repricing_score": 0.74,
                "volatility_shock_score": 0.77,
            ],
            "state_labels": [
                "macro_state": "RATES_REPRICING",
                "risk_state": "RISK_OFF",
                "liquidity_state": "STRESSED",
            ],
            "selected_proxies": [
                "equities": [
                    "symbol": "US500",
                    "fallback_used": false,
                    "available": true,
                    "change_pct_1d": -1.26,
                    "range_ratio_1d": 1.44,
                ],
            ],
            "pair_states": [
                "EURUSD": [
                    "base_currency": "EUR",
                    "quote_currency": "USD",
                    "macro_state": "RATES_REPRICING",
                    "risk_state": "RISK_OFF",
                    "liquidity_state": "STRESSED",
                    "pair_cross_asset_risk_score": 0.81,
                    "pair_sensitivity": 0.79,
                    "trade_gate": "BLOCK",
                    "stale": false,
                    "reasons": ["FRONT_END_RATES_DIVERGING", "USD_LIQUIDITY_PRESSURE_RISING"],
                ],
            ],
            "recent_transitions": [
                [
                    "type": "pair_gate",
                    "target": "EURUSD",
                    "from": "CAUTION",
                    "to": "BLOCK",
                    "observed_at": "2026-04-11T09:58:00Z",
                ],
            ],
            "reason_codes": [
                "FRONT_END_RATES_DIVERGING",
                "VOLATILITY_STRESS_ELEVATED",
            ],
            "quality_flags": [
                "fallback_proxy_used": true,
                "partial_data": false,
                "data_stale": false,
            ],
        ]
        let statusJSON: [String: Any] = [
            "health": [
                "pair_count": 54,
                "feature_count": 10,
                "snapshot_stale_after_sec": 900,
            ],
            "artifacts": [
                "snapshot_json": "/tmp/cross_asset_snapshot.json",
                "snapshot_flat": "/tmp/cross_asset_snapshot_flat.tsv",
            ],
        ]

        let snapshotData = try JSONSerialization.data(withJSONObject: snapshotJSON, options: [.prettyPrinted, .sortedKeys])
        let statusData = try JSONSerialization.data(withJSONObject: statusJSON, options: [.prettyPrinted, .sortedKeys])
        try snapshotData.write(to: runtimeDirectory.appendingPathComponent("cross_asset_snapshot.json"))
        try statusData.write(to: statusDirectory.appendingPathComponent("cross_asset_status.json"))

        let snapshot = try #require(CrossAssetArtifactReader().read(projectRoot: projectRoot))
        #expect(snapshot.sourceStatuses.count == 2)
        #expect(snapshot.pairs.count == 1)
        #expect(snapshot.pairs.first?.tradeGate == "BLOCK")
        #expect(snapshot.selectedProxies.first?.symbol == "US500")
        #expect(snapshot.recentTransitions.first?.toValue == "BLOCK")
        #expect(snapshot.stateLabels.first(where: { $0.key == "macro_state" })?.value == "RATES_REPRICING")
    }

    @Test
    func readerReturnsNilWhenSnapshotIsMissing() {
        let projectRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("fxai-cross-asset-reader-missing-\(UUID().uuidString)", isDirectory: true)
        defer {
            try? FileManager.default.removeItem(at: projectRoot)
        }

        #expect(CrossAssetArtifactReader().read(projectRoot: projectRoot) == nil)
    }
}
