import Foundation

public struct ExecutionQualityArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> ExecutionQualitySnapshot? {
        guard let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) else {
            return nil
        }

        let replayURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/ExecutionQuality/Reports", isDirectory: true)
            .appendingPathComponent("execution_quality_replay_report.json", isDirectory: false)
        let replayDocument = parseJSON(replayURL)
        let replayBySymbol: [String: [String: Any]] = Dictionary(
            uniqueKeysWithValues: (replayDocument?["symbols"] as? [[String: Any]] ?? []).compactMap { item in
                guard let symbol = item["symbol"] as? String, !symbol.isEmpty else { return nil }
                return (symbol.uppercased(), item)
            }
        )

        let stateFiles = ((try? FileManager.default.contentsOfDirectory(at: runtimeDirectory, includingPropertiesForKeys: nil)) ?? [])
            .filter { url in
                let name = url.lastPathComponent
                return name.hasPrefix("fxai_execution_quality_")
                    && name.hasSuffix(".tsv")
                    && !name.contains("_history_")
            }

        let symbols = stateFiles.compactMap { url in
            let symbol = url.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "fxai_execution_quality_", with: "")
                .uppercased()
            return parseSymbol(url: url, replay: replayBySymbol[symbol] ?? [:])
        }
        .sorted { lhs, rhs in
            if statePriority(lhs.executionState) == statePriority(rhs.executionState) {
                if lhs.executionQualityScore == rhs.executionQualityScore {
                    return lhs.symbol < rhs.symbol
                }
                return lhs.executionQualityScore < rhs.executionQualityScore
            }
            return statePriority(lhs.executionState) > statePriority(rhs.executionState)
        }

        guard !symbols.isEmpty else {
            return nil
        }

        let generatedAt = symbols.compactMap(\.generatedAt).max() ?? modificationDate(for: replayURL) ?? Date()
        return ExecutionQualitySnapshot(
            generatedAt: generatedAt,
            replayHoursBack: replayDocument?["hours_back"] as? Int ?? 72,
            symbols: symbols
        )
    }

    private func parseSymbol(url: URL, replay: [String: Any]) -> ExecutionQualitySymbolSnapshot? {
        let values = parseTSVMap(url) ?? [:]
        let symbol = values["symbol"] ?? url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "fxai_execution_quality_", with: "")
        guard !symbol.isEmpty else { return nil }

        let latest = replay["latest"] as? [String: Any] ?? [:]
        let latestState = latest["state"] as? [String: Any] ?? [:]
        let reasons = splitSemicolonSeparated(values["reasons_csv"]).ifEmpty(latestState["reason_codes"] as? [String] ?? [])

        return ExecutionQualitySymbolSnapshot(
            symbol: symbol,
            generatedAt: parseEpochDate(values["generated_at"]) ?? parseDate(latest["generated_at"]) ?? modificationDate(for: url),
            method: values["method"] ?? (latestState["method"] as? String) ?? "SCORECARD_V1",
            sessionLabel: values["session_label"] ?? (latestState["session_label"] as? String) ?? "UNKNOWN",
            regimeLabel: values["regime_label"] ?? (latestState["regime_label"] as? String) ?? "UNKNOWN",
            tierKind: values["selected_tier_kind"] ?? (latestState["selected_tier_kind"] as? String) ?? "GLOBAL",
            tierKey: values["selected_tier_key"] ?? (latestState["selected_tier_key"] as? String) ?? "GLOBAL|*|*|*",
            support: parseInt(values["selected_support"]) ?? (latestState["selected_support"] as? Int) ?? 0,
            quality: parseDouble(values["selected_quality"]) ?? (latestState["selected_quality"] as? Double) ?? 0,
            fallbackUsed: parseBool(values["fallback_used"]) ?? (latestState["fallback_used"] as? Bool) ?? false,
            memoryStale: parseBool(values["memory_stale"]) ?? (latestState["memory_stale"] as? Bool) ?? true,
            dataStale: parseBool(values["data_stale"]) ?? (latestState["data_stale"] as? Bool) ?? true,
            supportUsable: parseBool(values["support_usable"]) ?? (latestState["support_usable"] as? Bool) ?? false,
            newsWindowActive: parseBool(values["news_window_active"]) ?? (latestState["news_window_active"] as? Bool) ?? false,
            ratesRepricingActive: parseBool(values["rates_repricing_active"]) ?? (latestState["rates_repricing_active"] as? Bool) ?? false,
            brokerCoverage: parseDouble(values["broker_coverage"]) ?? (latestState["broker_coverage"] as? Double) ?? 0,
            brokerRejectProbability: parseDouble(values["broker_reject_prob"]) ?? (latestState["broker_reject_prob"] as? Double) ?? 0,
            brokerPartialFillProbability: parseDouble(values["broker_partial_fill_prob"]) ?? (latestState["broker_partial_fill_prob"] as? Double) ?? 0,
            spreadNowPoints: parseDouble(values["spread_now_points"]) ?? (latestState["spread_now_points"] as? Double) ?? 0,
            spreadExpectedPoints: parseDouble(values["spread_expected_points"]) ?? (latestState["spread_expected_points"] as? Double) ?? 0,
            spreadWideningRisk: parseDouble(values["spread_widening_risk"]) ?? (latestState["spread_widening_risk"] as? Double) ?? 0,
            expectedSlippagePoints: parseDouble(values["expected_slippage_points"]) ?? (latestState["expected_slippage_points"] as? Double) ?? 0,
            slippageRisk: parseDouble(values["slippage_risk"]) ?? (latestState["slippage_risk"] as? Double) ?? 0,
            fillQualityScore: parseDouble(values["fill_quality_score"]) ?? (latestState["fill_quality_score"] as? Double) ?? 0,
            latencySensitivityScore: parseDouble(values["latency_sensitivity_score"]) ?? (latestState["latency_sensitivity_score"] as? Double) ?? 0,
            liquidityFragilityScore: parseDouble(values["liquidity_fragility_score"]) ?? (latestState["liquidity_fragility_score"] as? Double) ?? 0,
            executionQualityScore: parseDouble(values["execution_quality_score"]) ?? (latestState["execution_quality_score"] as? Double) ?? 0,
            allowedDeviationPoints: parseDouble(values["allowed_deviation_points"]) ?? (latestState["allowed_deviation_points"] as? Double) ?? 0,
            cautionLotScale: parseDouble(values["caution_lot_scale"]) ?? (latestState["caution_lot_scale"] as? Double) ?? 1,
            cautionEnterProbBuffer: parseDouble(values["caution_enter_prob_buffer"]) ?? (latestState["caution_enter_prob_buffer"] as? Double) ?? 0,
            executionState: values["execution_state"] ?? (latestState["execution_state"] as? String) ?? "UNKNOWN",
            reasons: reasons,
            replayStateCounts: keyValueRecords(from: replay["state_counts"] as? [String: Any] ?? [:]),
            replayTierCounts: keyValueRecords(from: replay["tier_counts"] as? [String: Any] ?? [:]),
            replayTopReasons: namedCounts(replay["top_reasons"] as? [[String: Any]] ?? [], nameKey: "reason"),
            recentTransitions: parseTransitions(replay["recent_transitions"] as? [[String: Any]] ?? []),
            observationCount: replay["observations"] as? Int ?? 0,
            maxSpreadWideningRisk: replay["max_spread_widening_risk"] as? Double ?? 0,
            maxSlippageRisk: replay["max_slippage_risk"] as? Double ?? 0,
            minExecutionQualityScore: replay["min_execution_quality_score"] as? Double ?? 0
        )
    }

    private func parseTransitions(_ raw: [[String: Any]]) -> [ExecutionQualityTransition] {
        raw.map { item in
            ExecutionQualityTransition(
                type: item["type"] as? String ?? "change",
                fromValue: item["from"] as? String ?? "",
                toValue: item["to"] as? String ?? "",
                observedAt: parseDate(item["at"])
            )
        }
        .sorted { ($0.observedAt ?? .distantPast) > ($1.observedAt ?? .distantPast) }
    }

    private func parseJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func parseTSVMap(_ url: URL) -> [String: String]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var values: [String: String] = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            values[String(parts[0])] = String(parts[1])
        }
        return values
    }

    private func parseDate(_ raw: Any?) -> Date? {
        guard let text = raw as? String, !text.isEmpty else { return nil }
        return makeDateFormatter(fractional: true).date(from: text)
            ?? makeDateFormatter(fractional: false).date(from: text)
    }

    private func parseEpochDate(_ raw: String?) -> Date? {
        guard let raw, let seconds = TimeInterval(raw) else { return nil }
        return Date(timeIntervalSince1970: seconds)
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

    private func keyValueRecords(from raw: [String: Any]) -> [KeyValueRecord] {
        raw.map { key, value in
            KeyValueRecord(key: key, value: String(describing: value))
        }
        .sorted { $0.key < $1.key }
    }

    private func namedCounts(_ raw: [[String: Any]], nameKey: String) -> [KeyValueRecord] {
        raw.compactMap { item in
            guard let key = item[nameKey] as? String else { return nil }
            let count = item["count"] as? Int ?? 0
            return KeyValueRecord(key: key, value: "\(count)")
        }
    }

    private func splitSemicolonSeparated(_ raw: String?) -> [String] {
        guard let raw, !raw.isEmpty else { return [] }
        return raw
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func statePriority(_ value: String) -> Int {
        switch value.uppercased() {
        case "BLOCKED": 3
        case "STRESSED": 2
        case "CAUTION": 1
        default: 0
        }
    }

    private func modificationDate(for url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate]) as? Date
    }

    private func makeDateFormatter(fractional: Bool) -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = fractional
            ? [.withInternetDateTime, .withFractionalSeconds]
            : [.withInternetDateTime]
        return formatter
    }
}

private extension Array {
    func ifEmpty(_ fallback: @autoclosure () -> [Element]) -> [Element] {
        isEmpty ? fallback() : self
    }
}
