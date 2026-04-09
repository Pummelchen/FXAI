import Foundation

public struct DynamicEnsembleArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> DynamicEnsembleSnapshot? {
        guard let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) else {
            return nil
        }

        let replayURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DynamicEnsemble/Reports", isDirectory: true)
            .appendingPathComponent("dynamic_ensemble_replay_report.json", isDirectory: false)
        let replayDocument = parseJSON(replayURL)
        let replayBySymbol: [String: [String: Any]] = Dictionary(
            uniqueKeysWithValues: (replayDocument?["symbols"] as? [[String: Any]] ?? []).compactMap { item in
                guard let symbol = item["symbol"] as? String, !symbol.isEmpty else { return nil }
                return (symbol, item)
            }
        )

        let stateFiles = ((try? FileManager.default.contentsOfDirectory(at: runtimeDirectory, includingPropertiesForKeys: nil)) ?? [])
            .filter { url in
                let name = url.lastPathComponent
                return name.hasPrefix("fxai_dynamic_ensemble_")
                    && name.hasSuffix(".tsv")
                    && !name.contains("_history_")
            }

        let symbols = stateFiles.compactMap { url in
            parseSymbol(
                url: url,
                replay: replayBySymbol[url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "fxai_dynamic_ensemble_", with: "").uppercased()] ?? [:]
            )
        }
        .sorted { lhs, rhs in
            if lhs.tradePosture == rhs.tradePosture {
                if lhs.ensembleQuality == rhs.ensembleQuality {
                    return lhs.symbol < rhs.symbol
                }
                return lhs.ensembleQuality > rhs.ensembleQuality
            }
            return posturePriority(lhs.tradePosture) > posturePriority(rhs.tradePosture)
        }

        guard !symbols.isEmpty else {
            return nil
        }

        let generatedAt = symbols.compactMap(\.generatedAt).max() ?? modificationDate(for: replayURL) ?? Date()
        return DynamicEnsembleSnapshot(
            generatedAt: generatedAt,
            replayHoursBack: replayDocument?["hours_back"] as? Int ?? 72,
            symbols: symbols
        )
    }

    private func parseSymbol(url: URL, replay: [String: Any]) -> DynamicEnsembleSymbolSnapshot? {
        let values = parseTSVMap(url) ?? [:]
        let symbol = values["symbol"] ?? url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "fxai_dynamic_ensemble_", with: "")
        guard !symbol.isEmpty else { return nil }

        let latest = replay["latest"] as? [String: Any] ?? [:]
        let latestEnsemble = latest["ensemble"] as? [String: Any] ?? [:]

        let activePlugins = parsePluginCSV(values["active_plugins_csv"], status: "ACTIVE", latestPlugins: latest["plugins"] as? [[String: Any]] ?? [])
        let downweightedPlugins = parsePluginCSV(values["downweighted_plugins_csv"], status: "DOWNWEIGHTED", latestPlugins: latest["plugins"] as? [[String: Any]] ?? [])
        let suppressedPlugins = parsePluginCSV(values["suppressed_plugins_csv"], status: "SUPPRESSED", latestPlugins: latest["plugins"] as? [[String: Any]] ?? [])

        return DynamicEnsembleSymbolSnapshot(
            symbol: symbol,
            generatedAt: parseEpochDate(values["generated_at"]) ?? parseDate(latest["generated_at"]) ?? modificationDate(for: url),
            topRegime: values["top_regime"] ?? (latestEnsemble["top_regime"] as? String) ?? "UNKNOWN",
            sessionLabel: values["session_label"] ?? (latestEnsemble["session_label"] as? String) ?? "UNKNOWN",
            tradePosture: values["trade_posture"] ?? (latestEnsemble["trade_posture"] as? String) ?? "UNKNOWN",
            ensembleQuality: parseDouble(values["ensemble_quality"]) ?? (latestEnsemble["ensemble_quality"] as? Double) ?? 0,
            abstainBias: parseDouble(values["abstain_bias"]) ?? (latestEnsemble["abstain_bias"] as? Double) ?? 0,
            agreementScore: parseDouble(values["agreement_score"]) ?? (latestEnsemble["agreement_score"] as? Double) ?? 0,
            contextFitScore: parseDouble(values["context_fit_score"]) ?? (latestEnsemble["context_fit_score"] as? Double) ?? 0,
            dominantPluginShare: parseDouble(values["dominant_plugin_share"]) ?? (latestEnsemble["dominant_plugin_share"] as? Double) ?? 0,
            buyProb: parseDouble(values["buy_prob"]) ?? (latestEnsemble["buy_prob"] as? Double) ?? 0,
            sellProb: parseDouble(values["sell_prob"]) ?? (latestEnsemble["sell_prob"] as? Double) ?? 0,
            skipProb: parseDouble(values["skip_prob"]) ?? (latestEnsemble["skip_prob"] as? Double) ?? 0,
            finalScore: parseDouble(values["final_score"]) ?? (latestEnsemble["final_score"] as? Double) ?? 0,
            finalAction: values["final_action"] ?? (latestEnsemble["final_action"] as? String) ?? "SKIP",
            fallbackUsed: parseBool(values["fallback_used"]) ?? false,
            reasons: splitSemicolonSeparated(values["reasons_csv"]).ifEmpty(latestEnsemble["reasons"] as? [String] ?? []),
            activePlugins: activePlugins,
            downweightedPlugins: downweightedPlugins,
            suppressedPlugins: suppressedPlugins,
            replayPostureCounts: keyValueRecords(from: replay["posture_counts"] as? [String: Any] ?? [:]),
            replayActionCounts: keyValueRecords(from: replay["action_counts"] as? [String: Any] ?? [:]),
            replayStatusCounts: keyValueRecords(from: replay["plugin_status_counts"] as? [String: Any] ?? [:]),
            replayTopReasons: namedCounts(replay["top_reasons"] as? [[String: Any]] ?? [], nameKey: "reason"),
            replayTopDominantPlugins: namedCounts(replay["top_dominant_plugins"] as? [[String: Any]] ?? [], nameKey: "plugin"),
            recentTransitions: parseTransitions(replay["recent_transitions"] as? [[String: Any]] ?? []),
            observationCount: replay["observations"] as? Int ?? 0,
            averageQuality: replay["average_quality"] as? Double ?? 0,
            maxAbstainBias: replay["max_abstain_bias"] as? Double ?? 0
        )
    }

    private func parsePluginCSV(_ csv: String?, status: String, latestPlugins: [[String: Any]]) -> [DynamicEnsemblePluginState] {
        let parsed = splitPipe(csv).compactMap { token -> DynamicEnsemblePluginState? in
            let parts = token.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
            guard let name = parts.first, !name.isEmpty else { return nil }
            let weight = parts.count > 1 ? (Double(parts[1]) ?? 0) : 0
            let trust = parts.count > 2 ? (Double(parts[2]) ?? 0) : 0
            return DynamicEnsemblePluginState(
                name: name,
                family: "unknown",
                status: status,
                signal: status == "SUPPRESSED" ? "SKIP" : "MIXED",
                weight: weight,
                trust: trust,
                calibrationShrink: 1.0,
                reasons: [defaultReason(for: status)]
            )
        }
        if !parsed.isEmpty {
            return parsed.sorted(by: pluginOrdering)
        }

        return latestPlugins.compactMap { item in
            let pluginStatus = (item["status"] as? String ?? "ACTIVE").uppercased()
            guard pluginStatus == status else { return nil }
            return DynamicEnsemblePluginState(
                name: item["name"] as? String ?? "unknown",
                family: item["family"] as? String ?? "unknown",
                status: pluginStatus,
                signal: item["signal"] as? String ?? "SKIP",
                weight: item["weight"] as? Double ?? 0,
                trust: item["trust"] as? Double ?? 0,
                calibrationShrink: item["calibration_shrink"] as? Double ?? 1,
                reasons: item["reasons"] as? [String] ?? []
            )
        }
        .sorted(by: pluginOrdering)
    }

    private func parseTransitions(_ raw: [[String: Any]]) -> [DynamicEnsembleTransition] {
        raw.map { item in
            DynamicEnsembleTransition(
                type: item["type"] as? String ?? "change",
                fromValue: item["from"] as? String ?? "",
                toValue: item["to"] as? String ?? "",
                observedAt: parseDate(item["at"])
            )
        }
        .sorted { ($0.observedAt ?? .distantPast) > ($1.observedAt ?? .distantPast) }
    }

    private func defaultReason(for status: String) -> String {
        switch status.uppercased() {
        case "SUPPRESSED":
            return "Suppressed by dynamic ensemble trust filters"
        case "DOWNWEIGHTED":
            return "Downweighted by dynamic ensemble trust filters"
        default:
            return "Active in the current dynamic ensemble"
        }
    }

    private func pluginOrdering(_ lhs: DynamicEnsemblePluginState, _ rhs: DynamicEnsemblePluginState) -> Bool {
        if lhs.weight == rhs.weight {
            return lhs.name < rhs.name
        }
        return lhs.weight > rhs.weight
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

    private func splitPipe(_ raw: String?) -> [String] {
        guard let raw, !raw.isEmpty else { return [] }
        return raw.split(separator: "|").map(String.init)
    }

    private func splitSemicolonSeparated(_ raw: String?) -> [String] {
        guard let raw, !raw.isEmpty else { return [] }
        return raw
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func posturePriority(_ value: String) -> Int {
        switch value.uppercased() {
        case "BLOCK": 3
        case "ABSTAIN_BIAS": 2
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
