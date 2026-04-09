import Foundation

public struct MicrostructureArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> MicrostructureSnapshot? {
        guard let artifactURL = statusArtifactURL(projectRoot: projectRoot),
              let document = parseJSON(artifactURL)
        else {
            return nil
        }

        let generatedAt = parseDate(document["generated_at"]) ?? modificationDate(for: artifactURL) ?? Date()
        let serviceStatus = parseServiceStatus(document["service"] as? [String: Any] ?? [:])
        let symbols = parseSymbols(document["symbols"] as? [String: Any] ?? [:])
        guard !symbols.isEmpty else {
            return nil
        }

        return MicrostructureSnapshot(
            generatedAt: generatedAt,
            serviceStatus: serviceStatus,
            symbols: symbols,
            healthSummary: keyValueRecords(from: document["health"] as? [String: Any] ?? [:]),
            artifactPaths: keyValueRecords(from: document["artifacts"] as? [String: Any] ?? [:])
        )
    }

    private func statusArtifactURL(projectRoot: URL) -> URL? {
        let localStatus = projectRoot
            .appendingPathComponent("Tools/OfflineLab/Microstructure", isDirectory: true)
            .appendingPathComponent("microstructure_status.json", isDirectory: false)
        if FileManager.default.fileExists(atPath: localStatus.path) {
            return localStatus
        }

        if let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) {
            let runtimeStatus = runtimeDirectory.appendingPathComponent("microstructure_status.json", isDirectory: false)
            if FileManager.default.fileExists(atPath: runtimeStatus.path) {
                return runtimeStatus
            }
            let runtimeSnapshot = runtimeDirectory.appendingPathComponent("microstructure_snapshot.json", isDirectory: false)
            if FileManager.default.fileExists(atPath: runtimeSnapshot.path) {
                return runtimeSnapshot
            }
        }

        return nil
    }

    private func parseServiceStatus(_ raw: [String: Any]) -> MicrostructureServiceStatus {
        MicrostructureServiceStatus(
            ok: parseBool(raw["ok"]) ?? false,
            stale: parseBool(raw["stale"]) ?? true,
            enabled: parseBool(raw["enabled"]) ?? true,
            pollIntervalMS: parseInt(raw["poll_interval_ms"]) ?? 0,
            symbolRefreshSec: parseInt(raw["symbol_refresh_sec"]) ?? 0,
            snapshotStaleAfterSec: parseInt(raw["snapshot_stale_after_sec"]) ?? 0,
            lastPollAt: parseDate(raw["last_poll_at"]),
            lastSuccessAt: parseDate(raw["last_success_at"]),
            lastSymbolRefreshAt: parseDate(raw["last_symbol_refresh_at"]),
            lastError: raw["last_error"] as? String
        )
    }

    private func parseSymbols(_ raw: [String: Any]) -> [MicrostructureSymbolState] {
        raw.compactMap { symbol, value in
            guard let spec = value as? [String: Any] else { return nil }
            return MicrostructureSymbolState(
                symbol: symbol,
                brokerSymbol: spec["broker_symbol"] as? String ?? "",
                available: parseBool(spec["available"]) ?? true,
                stale: parseBool(spec["stale"]) ?? true,
                generatedAt: parseDate(spec["generated_at"]),
                spreadCurrent: parseDouble(spec["spread_current"]) ?? 0,
                silentGapSecondsCurrent: parseDouble(spec["silent_gap_seconds_current"]) ?? 0,
                sessionTag: spec["session_tag"] as? String ?? "UNKNOWN",
                handoffFlag: parseBool(spec["handoff_flag"]) ?? false,
                minutesSinceSessionOpen: parseInt(spec["minutes_since_session_open"]),
                minutesToSessionClose: parseInt(spec["minutes_to_session_close"]),
                sessionOpenBurstScore: parseDouble(spec["session_open_burst_score"]) ?? 0,
                sessionSpreadBehaviorScore: parseDouble(spec["session_spread_behavior_score"]) ?? 0,
                liquidityStressScore: parseDouble(spec["liquidity_stress_score"]) ?? 0,
                hostileExecutionScore: parseDouble(spec["hostile_execution_score"]) ?? 0,
                microstructureRegime: spec["microstructure_regime"] as? String ?? "UNKNOWN",
                tradeGate: spec["trade_gate"] as? String ?? "UNKNOWN",
                tickImbalance30s: parseDouble(spec["tick_imbalance_30s"]) ?? 0,
                directionalEfficiency60s: parseDouble(spec["directional_efficiency_60s"]) ?? 0,
                spreadZScore60s: parseDouble(spec["spread_zscore_60s"]) ?? 0,
                tickRate60s: parseDouble(spec["tick_rate_60s"]) ?? 0,
                tickRateZScore60s: parseDouble(spec["tick_rate_zscore_60s"]) ?? 0,
                realizedVol5m: parseDouble(spec["realized_vol_5m"]) ?? 0,
                volBurstScore5m: parseDouble(spec["vol_burst_score_5m"]) ?? 0,
                localExtremaBreachScore60s: parseDouble(spec["local_extrema_breach_score_60s"]) ?? 0,
                sweepAndRejectFlag60s: parseBool(spec["sweep_and_reject_flag_60s"]) ?? false,
                breakoutReversalScore60s: parseDouble(spec["breakout_reversal_score_60s"]) ?? 0,
                exhaustionProxy60s: parseDouble(spec["exhaustion_proxy_60s"]) ?? 0,
                reasons: spec["reasons"] as? [String] ?? []
            )
        }
        .sorted { lhs, rhs in
            if lhs.tradeGate == rhs.tradeGate {
                if lhs.hostileExecutionScore == rhs.hostileExecutionScore {
                    return lhs.symbol < rhs.symbol
                }
                return lhs.hostileExecutionScore > rhs.hostileExecutionScore
            }
            return posturePriority(lhs.tradeGate) > posturePriority(rhs.tradeGate)
        }
    }

    private func parseJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func modificationDate(for url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate]) as? Date
    }

    private func parseDate(_ raw: Any?) -> Date? {
        guard let text = raw as? String, !text.isEmpty else { return nil }
        return makeDateFormatter(fractional: true).date(from: text)
            ?? makeDateFormatter(fractional: false).date(from: text)
    }

    private func parseDouble(_ raw: Any?) -> Double? {
        if let value = raw as? Double {
            return value
        }
        if let value = raw as? Int {
            return Double(value)
        }
        if let text = raw as? String {
            return Double(text)
        }
        return nil
    }

    private func parseInt(_ raw: Any?) -> Int? {
        if let value = raw as? Int {
            return value
        }
        if let text = raw as? String {
            return Int(text)
        }
        return nil
    }

    private func parseBool(_ raw: Any?) -> Bool? {
        if let value = raw as? Bool {
            return value
        }
        if let value = raw as? Int {
            return value != 0
        }
        if let text = raw as? String {
            let lowered = text.lowercased()
            if lowered == "true" || lowered == "1" {
                return true
            }
            if lowered == "false" || lowered == "0" {
                return false
            }
        }
        return nil
    }

    private func keyValueRecords(from raw: [String: Any]) -> [KeyValueRecord] {
        raw.map { key, value in
            KeyValueRecord(key: key, value: String(describing: value))
        }
        .sorted { $0.key < $1.key }
    }

    private func posturePriority(_ value: String) -> Int {
        switch value.uppercased() {
        case "BLOCK": 3
        case "CAUTION": 2
        case "ALLOW": 1
        default: 0
        }
    }

    private func makeDateFormatter(fractional: Bool) -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = fractional
            ? [.withInternetDateTime, .withFractionalSeconds]
            : [.withInternetDateTime]
        return formatter
    }
}
