import Foundation

public struct CrossAssetArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> CrossAssetSnapshot? {
        guard let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) else {
            return nil
        }

        let runtimeSnapshotURL = runtimeDirectory.appendingPathComponent("cross_asset_snapshot.json", isDirectory: false)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/CrossAsset", isDirectory: true)
            .appendingPathComponent("cross_asset_status.json", isDirectory: false)

        guard let snapshot = parseJSON(runtimeSnapshotURL) else {
            return nil
        }
        let status = parseJSON(statusURL) ?? [:]

        let sourceStatuses = parseSourceStatuses(snapshot["source_status"] as? [String: Any] ?? [:])
        let pairs = parsePairs(snapshot["pair_states"] as? [String: Any] ?? [:])
        let features = keyValueRecords(from: snapshot["features"] as? [String: Any] ?? [:])
        let stateScores = keyValueRecords(from: snapshot["state_scores"] as? [String: Any] ?? [:])
        let stateLabels = keyValueRecords(from: snapshot["state_labels"] as? [String: Any] ?? [:])
        let selectedProxies = parseSelectedProxies(snapshot["selected_proxies"] as? [String: Any] ?? [:])
        let recentTransitions = parseTransitions(snapshot["recent_transitions"] as? [[String: Any]] ?? [])
        let reasons = snapshot["reason_codes"] as? [String] ?? []
        let qualityFlags = keyValueRecords(from: snapshot["quality_flags"] as? [String: Any] ?? [:])
        let healthSummary = keyValueRecords(from: status["health"] as? [String: Any] ?? [:])
        let artifactPaths = keyValueRecords(from: status["artifacts"] as? [String: Any] ?? [:])
        let generatedAt = parseDate(snapshot["generated_at"]) ?? modificationDate(for: runtimeSnapshotURL) ?? Date()

        guard !pairs.isEmpty || !features.isEmpty || !stateScores.isEmpty else {
            return nil
        }

        return CrossAssetSnapshot(
            generatedAt: generatedAt,
            sourceStatuses: sourceStatuses,
            features: features,
            stateScores: stateScores,
            stateLabels: stateLabels,
            selectedProxies: selectedProxies,
            pairs: pairs,
            recentTransitions: recentTransitions,
            reasons: reasons,
            qualityFlags: qualityFlags,
            healthSummary: healthSummary,
            artifactPaths: artifactPaths
        )
    }

    private func parseSourceStatuses(_ raw: [String: Any]) -> [CrossAssetSourceStatus] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return CrossAssetSourceStatus(
                id: key,
                ok: parseBool(spec["ok"]) ?? false,
                stale: parseBool(spec["stale"]) ?? false,
                lastUpdateAt: parseDate(spec["last_update_at"]),
                proxySymbol: spec["proxy_symbol"] as? String ?? spec["oil_proxy"] as? String ?? spec["gold_proxy"] as? String,
                availableSymbols: parseInt(spec["available_symbols"]),
                configuredSymbols: parseInt(spec["configured_symbols"])
            )
        }
        .sorted { $0.id < $1.id }
    }

    private func parseSelectedProxies(_ raw: [String: Any]) -> [CrossAssetProxySelection] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return CrossAssetProxySelection(
                id: key,
                symbol: spec["symbol"] as? String ?? "",
                fallbackUsed: parseBool(spec["fallback_used"]) ?? false,
                available: parseBool(spec["available"]) ?? false,
                changePct1d: parseDouble(spec["change_pct_1d"]) ?? 0,
                rangeRatio1d: parseDouble(spec["range_ratio_1d"]) ?? 0
            )
        }
        .sorted { $0.id < $1.id }
    }

    private func parsePairs(_ raw: [String: Any]) -> [CrossAssetPairState] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return CrossAssetPairState(
                pair: key,
                baseCurrency: spec["base_currency"] as? String ?? "",
                quoteCurrency: spec["quote_currency"] as? String ?? "",
                macroState: spec["macro_state"] as? String ?? "UNKNOWN",
                riskState: spec["risk_state"] as? String ?? "UNKNOWN",
                liquidityState: spec["liquidity_state"] as? String ?? "UNKNOWN",
                pairCrossAssetRiskScore: parseDouble(spec["pair_cross_asset_risk_score"]) ?? 0,
                pairSensitivity: parseDouble(spec["pair_sensitivity"]) ?? 0,
                tradeGate: spec["trade_gate"] as? String ?? "UNKNOWN",
                stale: parseBool(spec["stale"]) ?? true,
                reasons: spec["reasons"] as? [String] ?? []
            )
        }
        .sorted { lhs, rhs in
            if posturePriority(lhs.tradeGate) == posturePriority(rhs.tradeGate) {
                if lhs.pairCrossAssetRiskScore == rhs.pairCrossAssetRiskScore {
                    return lhs.pair < rhs.pair
                }
                return lhs.pairCrossAssetRiskScore > rhs.pairCrossAssetRiskScore
            }
            return posturePriority(lhs.tradeGate) > posturePriority(rhs.tradeGate)
        }
    }

    private func parseTransitions(_ raw: [[String: Any]]) -> [CrossAssetTransition] {
        raw.map { item in
            CrossAssetTransition(
                type: item["type"] as? String ?? "change",
                target: item["target"] as? String ?? "global",
                fromValue: item["from"] as? String ?? "",
                toValue: item["to"] as? String ?? "",
                observedAt: parseDate(item["observed_at"])
            )
        }
        .sorted { ($0.observedAt ?? .distantPast) > ($1.observedAt ?? .distantPast) }
    }

    private func keyValueRecords(from raw: [String: Any]) -> [KeyValueRecord] {
        raw.map { key, value in
            KeyValueRecord(key: key, value: String(describing: value))
        }
        .sorted { $0.key < $1.key }
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
        if let value = raw as? Double { return value }
        if let value = raw as? Int { return Double(value) }
        if let text = raw as? String { return Double(text) }
        return nil
    }

    private func parseInt(_ raw: Any?) -> Int? {
        if let value = raw as? Int { return value }
        if let text = raw as? String { return Int(text) }
        return nil
    }

    private func parseBool(_ raw: Any?) -> Bool? {
        if let value = raw as? Bool { return value }
        if let value = raw as? Int { return value != 0 }
        if let text = raw as? String {
            let lowered = text.lowercased()
            if lowered == "true" || lowered == "1" { return true }
            if lowered == "false" || lowered == "0" { return false }
        }
        return nil
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
