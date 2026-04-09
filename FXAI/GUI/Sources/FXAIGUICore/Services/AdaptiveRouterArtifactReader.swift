import Foundation

public struct AdaptiveRouterArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> AdaptiveRouterSnapshot? {
        let researchRoot = projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        guard
            let dashboardURL = latestFile(named: "operator_dashboard.json", under: researchRoot),
            let dashboard = parseJSON(dashboardURL)
        else {
            return nil
        }

        let profileDirectory = dashboardURL.deletingLastPathComponent()
        let profileName = profileDirectory.lastPathComponent
        let replayURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/AdaptiveRouter/Reports", isDirectory: true)
            .appendingPathComponent("adaptive_router_replay_report.json", isDirectory: false)
        let replayDocument = parseJSON(replayURL)
        let replayBySymbol: [String: [String: Any]] = Dictionary(
            uniqueKeysWithValues: (replayDocument?["symbols"] as? [[String: Any]] ?? []).compactMap { item in
                guard let symbol = item["symbol"] as? String, !symbol.isEmpty else { return nil }
                return (symbol, item)
            }
        )

        let symbols = (dashboard["deployments"] as? [[String: Any]] ?? [])
            .compactMap { entry in
                parseSymbol(
                    entry: entry,
                    profileDirectory: profileDirectory,
                    profileName: profileName,
                    replayDocument: replayBySymbol
                )
            }
            .sorted { lhs, rhs in
                if lhs.tradePosture == rhs.tradePosture {
                    if lhs.confidence == rhs.confidence {
                        return lhs.symbol < rhs.symbol
                    }
                    return lhs.confidence > rhs.confidence
                }
                return posturePriority(lhs.tradePosture) > posturePriority(rhs.tradePosture)
            }

        guard !symbols.isEmpty else {
            return nil
        }

        let generatedAt = parseDate(dashboard["generated_at"]) ?? modificationDate(for: dashboardURL)
        return AdaptiveRouterSnapshot(
            generatedAt: generatedAt,
            profileName: profileName,
            replayHoursBack: replayDocument?["hours_back"] as? Int ?? 72,
            symbols: symbols
        )
    }

    private func parseSymbol(
        entry: [String: Any],
        profileDirectory: URL,
        profileName: String,
        replayDocument: [String: [String: Any]]
    ) -> AdaptiveRouterSymbolSnapshot? {
        let symbol = entry["symbol"] as? String ?? ""
        guard !symbol.isEmpty else { return nil }

        let liveState = entry["live_state"] as? [String: Any] ?? [:]
        let runtimeValues = (liveState["adaptive_runtime_tsv"] as? [String: String]).flatMap { $0.isEmpty ? nil : $0 }
            ?? parseTSVMap(
                profileDirectory
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .appendingPathComponent("FILE_COMMON/FXAI/Runtime", isDirectory: true)
                    .appendingPathComponent("fxai_regime_router_\(symbol).tsv", isDirectory: false)
            )
            ?? [:]
        let profileValues = (liveState["adaptive_router_tsv"] as? [String: String]).flatMap { $0.isEmpty ? nil : $0 }
            ?? [:]
        let profileJSON = parseJSON(profileDirectory.appendingPathComponent("adaptive_router_\(symbol).json", isDirectory: false)) ?? [:]
        let replay = replayDocument[symbol] ?? [:]
        let latestReplay = replay["latest"] as? [String: Any] ?? [:]
        let latestReplayRegime = latestReplay["regime"] as? [String: Any] ?? [:]
        let latestReplayRouter = latestReplay["router"] as? [String: Any] ?? [:]

        let runtimeProbabilities = parseProbabilityRecords(
            runtimeValues["probabilities_csv"],
            fallback: latestReplayRegime["probabilities"] as? [String: Any] ?? [:]
        )
        let routedPlugins = parsePluginStates(from: runtimeValues, latestReplay: latestReplay["plugins"] as? [[String: Any]] ?? [])
        let thresholdMetrics = profileValues
            .filter { key, _ in
                [
                    "caution_threshold",
                    "abstain_threshold",
                    "block_threshold",
                    "confidence_floor",
                    "suppression_threshold",
                    "downweight_threshold",
                    "stale_news_abstain_bias",
                    "max_active_weight_share",
                ].contains(key)
            }
            .sortedKeyValues()
        let regimeBiasMetrics = profileValues
            .filter { $0.key.hasPrefix("regime_bias_") }
            .sortedKeyValues()

        let replayRegimeCounts = keyValueRecords(from: replay["regime_counts"] as? [String: Any] ?? [:])
        let replayPostureCounts = keyValueRecords(from: replay["posture_counts"] as? [String: Any] ?? [:])
        let replayTopReasons = namedCounts(
            replay["top_reasons"] as? [[String: Any]] ?? [],
            nameKey: "reason"
        )
        let replayTopPlugins = namedCounts(
            replay["top_plugins"] as? [[String: Any]] ?? [],
            nameKey: "plugin"
        )
        let pairTags = splitCSV(profileValues["pair_tags_csv"])
        let topProfilePlugins = ((profileJSON["summary"] as? [String: Any])?["top_plugins"] as? [String]) ?? []
        let reasons = splitSemicolonSeparated(runtimeValues["reasons_csv"])
            .ifEmpty(latestReplayRouter["reasons"] as? [String] ?? latestReplayRegime["reasons"] as? [String] ?? [])

        return AdaptiveRouterSymbolSnapshot(
            symbol: symbol,
            profileName: profileName,
            routerMode: runtimeValues["router_mode"]
                ?? profileValues["router_mode"]
                ?? (profileJSON["router_mode"] as? String)
                ?? (latestReplayRouter["mode"] as? String)
                ?? "WEIGHTED_ENSEMBLE",
            topRegime: runtimeValues["top_regime_label"]
                ?? (latestReplayRegime["top_label"] as? String)
                ?? "UNKNOWN",
            confidence: parseDouble(runtimeValues["regime_confidence"])
                ?? (latestReplayRegime["confidence"] as? Double)
                ?? 0,
            tradePosture: runtimeValues["trade_posture"]
                ?? (latestReplayRouter["trade_posture"] as? String)
                ?? "UNKNOWN",
            abstainBias: parseDouble(runtimeValues["abstain_bias"])
                ?? (latestReplayRouter["abstain_bias"] as? Double)
                ?? 0,
            sessionLabel: runtimeValues["session_label"] ?? (latestReplayRegime["session"] as? String) ?? "UNKNOWN",
            spreadRegime: runtimeValues["spread_regime"] ?? (latestReplayRegime["spread_regime"] as? String) ?? "UNKNOWN",
            volatilityRegime: runtimeValues["volatility_regime"] ?? (latestReplayRegime["volatility_regime"] as? String) ?? "UNKNOWN",
            newsRiskScore: parseDouble(runtimeValues["news_risk_score"])
                ?? (latestReplayRegime["news_risk_score"] as? Double)
                ?? 0,
            newsPressure: parseDouble(runtimeValues["news_pressure"])
                ?? (latestReplayRegime["news_pressure"] as? Double)
                ?? 0,
            eventETAMin: parseInt(runtimeValues["event_eta_min"])
                ?? (latestReplayRegime["event_eta_min"] as? Int),
            staleNews: parseBool(runtimeValues["stale_news"])
                ?? (latestReplayRegime["stale_news"] as? Bool)
                ?? true,
            liquidityStress: parseDouble(runtimeValues["liquidity_stress"]) ?? 0,
            breakoutPressure: parseDouble(runtimeValues["breakout_pressure"]) ?? 0,
            trendStrength: parseDouble(runtimeValues["trend_strength"]) ?? 0,
            rangePressure: parseDouble(runtimeValues["range_pressure"]) ?? 0,
            macroPressure: parseDouble(runtimeValues["macro_pressure"]) ?? 0,
            generatedAt: parseEpochDate(runtimeValues["generated_at"]) ?? parseDate(latestReplay["generated_at"]),
            profileGeneratedAt: parseDate(((profileJSON["summary"] as? [String: Any])?["generated_at"])),
            reasons: reasons,
            probabilities: runtimeProbabilities,
            activePlugins: routedPlugins.active,
            downweightedPlugins: routedPlugins.downweighted,
            suppressedPlugins: routedPlugins.suppressed,
            pairTags: pairTags,
            topProfilePlugins: topProfilePlugins,
            thresholdMetrics: thresholdMetrics,
            regimeBiasMetrics: regimeBiasMetrics,
            replayRegimeCounts: replayRegimeCounts,
            replayPostureCounts: replayPostureCounts,
            replayTopReasons: replayTopReasons,
            replayTopPlugins: replayTopPlugins,
            recentTransitions: parseTransitions(replay["recent_transitions"] as? [[String: Any]] ?? []),
            observationCount: replay["observations"] as? Int ?? 0
        )
    }

    private func parseProbabilityRecords(_ csv: String?, fallback: [String: Any]) -> [AdaptiveRouterProbabilityRecord] {
        var items: [AdaptiveRouterProbabilityRecord] = []
        for token in splitCSV(csv) {
            let parts = token.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, let probability = Double(parts[1]) else { continue }
            items.append(AdaptiveRouterProbabilityRecord(label: parts[0], probability: probability))
        }
        if items.isEmpty {
            items = fallback
                .compactMap { key, value in
                    guard let probability = value as? Double ?? Double(String(describing: value)) else { return nil }
                    return AdaptiveRouterProbabilityRecord(label: key, probability: probability)
                }
        }
        return items.sorted {
            if $0.probability == $1.probability {
                return $0.label < $1.label
            }
            return $0.probability > $1.probability
        }
    }

    private func parsePluginStates(from runtimeValues: [String: String], latestReplay: [[String: Any]]) -> (active: [AdaptiveRouterPluginState], downweighted: [AdaptiveRouterPluginState], suppressed: [AdaptiveRouterPluginState]) {
        let active = parseRuntimePluginCSV(runtimeValues["active_plugins_csv"], status: "ACTIVE")
        let downweighted = parseRuntimePluginCSV(runtimeValues["downweighted_plugins_csv"], status: "DOWNWEIGHTED")
        let suppressed = parseRuntimePluginCSV(runtimeValues["suppressed_plugins_csv"], status: "SUPPRESSED")
        if !active.isEmpty || !downweighted.isEmpty || !suppressed.isEmpty {
            return (active, downweighted, suppressed)
        }

        var activeFallback: [AdaptiveRouterPluginState] = []
        var downweightedFallback: [AdaptiveRouterPluginState] = []
        var suppressedFallback: [AdaptiveRouterPluginState] = []
        for item in latestReplay {
            let status = (item["status"] as? String ?? "ACTIVE").uppercased()
            let state = AdaptiveRouterPluginState(
                name: item["name"] as? String ?? "unknown",
                weight: item["weight"] as? Double ?? 0,
                suitability: item["suitability"] as? Double ?? 0,
                status: status,
                reasons: item["reasons"] as? [String] ?? []
            )
            switch status {
            case "SUPPRESSED":
                suppressedFallback.append(state)
            case "DOWNWEIGHTED":
                downweightedFallback.append(state)
            default:
                activeFallback.append(state)
            }
        }
        return (
            activeFallback.sorted(by: pluginOrdering),
            downweightedFallback.sorted(by: pluginOrdering),
            suppressedFallback.sorted(by: pluginOrdering)
        )
    }

    private func parseRuntimePluginCSV(_ csv: String?, status: String) -> [AdaptiveRouterPluginState] {
        splitPipe(csv).compactMap { token in
            let parts = token.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
            guard let name = parts.first, !name.isEmpty else { return nil }
            let weight = parts.count > 1 ? (Double(parts[1]) ?? 0) : 0
            let suitability = parts.count > 2 ? (Double(parts[2]) ?? 0) : 0
            return AdaptiveRouterPluginState(
                name: name,
                weight: weight,
                suitability: suitability,
                status: status,
                reasons: [defaultReason(for: status)]
            )
        }
        .sorted(by: pluginOrdering)
    }

    private func parseTransitions(_ raw: [[String: Any]]) -> [AdaptiveRouterTransition] {
        raw.map { item in
            AdaptiveRouterTransition(
                type: item["type"] as? String ?? "transition",
                fromValue: item["from"] as? String ?? "",
                toValue: item["to"] as? String ?? "",
                observedAt: parseDate(item["at"])
            )
        }
    }

    private func namedCounts(_ raw: [[String: Any]], nameKey: String) -> [KeyValueRecord] {
        raw.compactMap { item in
            guard let name = item[nameKey] as? String, !name.isEmpty else { return nil }
            let count = item["count"] as? Int ?? 0
            return KeyValueRecord(key: name, value: "\(count)")
        }
    }

    private func keyValueRecords(from raw: [String: Any]) -> [KeyValueRecord] {
        raw
            .map { KeyValueRecord(key: $0.key, value: stringify($0.value)) }
            .sorted { $0.key < $1.key }
    }

    private func latestFile(named fileName: String, under root: URL) -> URL? {
        recursiveFiles(at: root)
            .filter { $0.lastPathComponent == fileName }
            .max { modificationDate(for: $0) < modificationDate(for: $1) }
    }

    private func recursiveFiles(at root: URL) -> [URL] {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: root.path) else { return [] }
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        var results: [URL] = []
        for case let url as URL in enumerator {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            results.append(url)
        }
        return results
    }

    private func parseTSVMap(_ url: URL?) -> [String: String]? {
        guard let url else { return nil }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var values: [String: String] = [:]
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            values[String(parts[0])] = String(parts[1])
        }
        return values.isEmpty ? nil : values
    }

    private func parseJSON(_ url: URL?) -> [String: Any]? {
        guard let url else { return nil }
        guard
            let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return raw
    }

    private func parseDate(_ raw: Any?) -> Date? {
        guard let text = raw as? String, !text.isEmpty else { return nil }
        return ISO8601DateFormatter().date(from: text)
    }

    private func parseEpochDate(_ raw: String?) -> Date? {
        guard let raw, let epoch = TimeInterval(raw) else { return nil }
        return Date(timeIntervalSince1970: epoch)
    }

    private func parseDouble(_ raw: String?) -> Double? {
        guard let raw else { return nil }
        return Double(raw)
    }

    private func parseInt(_ raw: String?) -> Int? {
        guard let raw else { return nil }
        return Int(raw)
    }

    private func parseBool(_ raw: String?) -> Bool? {
        guard let raw else { return nil }
        switch raw.lowercased() {
        case "1", "true", "yes":
            return true
        case "0", "false", "no":
            return false
        default:
            return nil
        }
    }

    private func splitCSV(_ raw: String?) -> [String] {
        let text = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        return text
            .split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func splitPipe(_ raw: String?) -> [String] {
        let text = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        return text
            .split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func splitSemicolonSeparated(_ raw: String?) -> [String] {
        let text = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        return text
            .split(separator: ";", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func defaultReason(for status: String) -> String {
        switch status {
        case "SUPPRESSED":
            return "Suppressed by current regime fit"
        case "DOWNWEIGHTED":
            return "Downweighted by current regime fit"
        default:
            return "Active in current regime"
        }
    }

    private func posturePriority(_ posture: String) -> Int {
        switch posture.uppercased() {
        case "BLOCK": return 4
        case "ABSTAIN_BIAS": return 3
        case "CAUTION": return 2
        case "NORMAL": return 1
        default: return 0
        }
    }

    private func pluginOrdering(_ lhs: AdaptiveRouterPluginState, _ rhs: AdaptiveRouterPluginState) -> Bool {
        if lhs.weight == rhs.weight {
            if lhs.suitability == rhs.suitability {
                return lhs.name < rhs.name
            }
            return lhs.suitability > rhs.suitability
        }
        return lhs.weight > rhs.weight
    }

    private func modificationDate(for url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private func stringify(_ value: Any) -> String {
        if let string = value as? String { return string }
        if let number = value as? NSNumber { return number.stringValue }
        if let bool = value as? Bool { return bool ? "true" : "false" }
        if let array = value as? [Any] {
            return array.map(stringify).joined(separator: ", ")
        }
        return String(describing: value)
    }
}

private extension Array where Element == String {
    func ifEmpty(_ fallback: [String]) -> [String] {
        isEmpty ? fallback : self
    }
}

private extension Dictionary where Key == String, Value == String {
    func sortedKeyValues() -> [KeyValueRecord] {
        map { KeyValueRecord(key: $0.key, value: $0.value) }
            .sorted { $0.key < $1.key }
    }
}
