import Foundation

public struct PairNetworkArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> PairNetworkSnapshot? {
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/PairNetwork", isDirectory: true)
            .appendingPathComponent("pair_network_status.json", isDirectory: false)
        let reportURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/PairNetwork/Reports", isDirectory: true)
            .appendingPathComponent("pair_network_report.json", isDirectory: false)

        let status = parseJSON(statusURL) ?? [:]
        let report = parseJSON(reportURL) ?? [:]

        var runtimeSymbols: [PairNetworkSymbolSnapshot] = []
        if let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) {
            let stateFiles = ((try? FileManager.default.contentsOfDirectory(at: runtimeDirectory, includingPropertiesForKeys: nil)) ?? [])
                .filter { url in
                    let name = url.lastPathComponent
                    return name.hasPrefix("fxai_pair_network_")
                        && name.hasSuffix(".tsv")
                        && !name.contains("_history_")
                }

            runtimeSymbols = stateFiles.compactMap(parseSymbolState(url:))
                .sorted { lhs, rhs in
                    if decisionPriority(lhs.decision) == decisionPriority(rhs.decision) {
                        if lhs.conflictScore == rhs.conflictScore {
                            return lhs.symbol < rhs.symbol
                        }
                        return lhs.conflictScore > rhs.conflictScore
                    }
                    return decisionPriority(lhs.decision) > decisionPriority(rhs.decision)
                }
        }

        let topEdges = parseEdges(report["top_edges"] as? [[String: Any]] ?? [])
        let pairSummaries = parsePairSummaries(report["pairs"] as? [[String: Any]] ?? [])
        let reasons = report["reason_codes"] as? [String] ?? []
        let qualityFlags = keyValueRecords(from: report["quality_flags"] as? [String: Any] ?? [:])

        let fallbackGraphUsed = parseBool(status["fallback_graph_used"]) ?? parseBool((report["quality_flags"] as? [String: Any])?["fallback_graph_used"]) ?? false
        let partialDependencyData = parseBool(status["partial_dependency_data"]) ?? parseBool((report["quality_flags"] as? [String: Any])?["partial_dependency_data"]) ?? false
        let graphStale = parseBool(status["graph_stale"]) ?? parseBool((report["quality_flags"] as? [String: Any])?["graph_stale"]) ?? true
        let graphMode = status["graph_mode"] as? String ?? report["graph_mode"] as? String ?? "STRUCTURAL_ONLY"
        let actionMode = status["action_mode"] as? String ?? "AUTO_APPLY"
        let pairCount = parseInt(status["pair_count"]) ?? parseInt(report["pair_count"]) ?? pairSummaries.count
        let currencyCount = parseInt(status["currency_count"]) ?? parseInt(report["currency_count"]) ?? 0
        let edgeCount = parseInt(status["edge_count"]) ?? parseInt(report["edge_count"]) ?? topEdges.count
        let generatedAt = parseDate(status["generated_at"]) ?? parseDate(report["generated_at"]) ?? runtimeSymbols.compactMap(\.generatedAt).max() ?? modificationDate(for: reportURL) ?? modificationDate(for: statusURL) ?? Date()

        let statusRecords = keyValueRecords(from: status)
        let artifactPaths = keyValueRecords(from: [
            "config_path": status["config_path"] ?? "",
            "runtime_config_path": status["runtime_config_path"] ?? "",
            "runtime_status_path": status["runtime_status_path"] ?? "",
            "report_path": status["report_path"] ?? "",
        ])

        guard !runtimeSymbols.isEmpty || !topEdges.isEmpty || !pairSummaries.isEmpty else {
            return nil
        }

        return PairNetworkSnapshot(
            generatedAt: generatedAt,
            graphMode: graphMode,
            actionMode: actionMode,
            pairCount: pairCount,
            currencyCount: currencyCount,
            edgeCount: edgeCount,
            fallbackGraphUsed: fallbackGraphUsed,
            partialDependencyData: partialDependencyData,
            graphStale: graphStale,
            symbols: runtimeSymbols,
            topEdges: topEdges,
            pairSummaries: pairSummaries,
            reasons: reasons,
            qualityFlags: qualityFlags,
            statusRecords: statusRecords,
            artifactPaths: artifactPaths
        )
    }

    private func parseSymbolState(url: URL) -> PairNetworkSymbolSnapshot? {
        let values = parseTSVMap(url) ?? [:]
        let symbol = values["symbol"] ?? url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "fxai_pair_network_", with: "")
        guard !symbol.isEmpty else { return nil }

        return PairNetworkSymbolSnapshot(
            symbol: symbol,
            generatedAt: parseEpochDate(values["generated_at"]) ?? modificationDate(for: url),
            decision: values["decision"] ?? "ALLOW",
            fallbackGraphUsed: parseBool(values["fallback_graph_used"]) ?? false,
            partialDependencyData: parseBool(values["partial_dependency_data"]) ?? false,
            graphStale: parseBool(values["graph_stale"]) ?? true,
            conflictScore: parseDouble(values["conflict_score"]) ?? 0,
            redundancyScore: parseDouble(values["redundancy_score"]) ?? 0,
            contradictionScore: parseDouble(values["contradiction_score"]) ?? 0,
            concentrationScore: parseDouble(values["concentration_score"]) ?? 0,
            currencyConcentration: parseDouble(values["currency_concentration"]) ?? 0,
            factorConcentration: parseDouble(values["factor_concentration"]) ?? 0,
            recommendedSizeMultiplier: parseDouble(values["recommended_size_multiplier"]) ?? 1,
            preferredExpression: values["preferred_expression"] ?? "",
            currencyExposure: parseExposureCSV(values["currency_exposure_csv"]),
            factorExposure: parseExposureCSV(values["factor_exposure_csv"]),
            reasons: splitSemicolonSeparated(values["reasons_csv"])
        )
    }

    private func parsePairSummaries(_ raw: [[String: Any]]) -> [PairNetworkPairSummary] {
        raw.compactMap { item in
            guard let pair = item["pair"] as? String, !pair.isEmpty else { return nil }
            return PairNetworkPairSummary(
                pair: pair,
                baseCurrency: item["base_currency"] as? String ?? "",
                quoteCurrency: item["quote_currency"] as? String ?? "",
                factorSignature: keyValueRecords(from: item["factor_signature"] as? [String: Any] ?? [:]),
                topDependencies: parseEdges(item["top_dependencies"] as? [[String: Any]] ?? [], defaultSourcePair: pair)
            )
        }
        .sorted { $0.pair < $1.pair }
    }

    private func parseEdges(_ raw: [[String: Any]], defaultSourcePair: String = "") -> [PairNetworkDependencyEdge] {
        raw.compactMap { item in
            let sourcePair = item["source_pair"] as? String ?? defaultSourcePair
            let targetPair = item["target_pair"] as? String ?? item["pair"] as? String ?? ""
            guard !targetPair.isEmpty else { return nil }
            return PairNetworkDependencyEdge(
                sourcePair: sourcePair,
                targetPair: targetPair,
                combinedScore: parseDouble(item["combined_score"]) ?? 0,
                structuralScore: parseDouble(item["structural_score"]) ?? 0,
                empiricalScore: parseDouble(item["empirical_score"]) ?? 0,
                correlation: parseDouble(item["correlation"]) ?? 0,
                support: parseInt(item["support"]) ?? 0,
                relation: item["relation"] as? String ?? "UNKNOWN",
                sharedCurrencies: item["shared_currencies"] as? [String] ?? []
            )
        }
        .sorted { lhs, rhs in
            if lhs.combinedScore == rhs.combinedScore {
                if lhs.sourcePair == rhs.sourcePair {
                    return lhs.targetPair < rhs.targetPair
                }
                return lhs.sourcePair < rhs.sourcePair
            }
            return lhs.combinedScore > rhs.combinedScore
        }
    }

    private func parseExposureCSV(_ raw: String?) -> [KeyValueRecord] {
        guard let raw, !raw.isEmpty else { return [] }
        return raw
            .split(separator: ";")
            .compactMap { chunk in
                let part = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !part.isEmpty else { return nil }
                let pieces = part.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
                guard pieces.count == 2 else {
                    return KeyValueRecord(key: part, value: "")
                }
                return KeyValueRecord(
                    key: String(pieces[0]).trimmingCharacters(in: .whitespacesAndNewlines),
                    value: String(pieces[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                )
            }
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

    private func splitSemicolonSeparated(_ raw: String?) -> [String] {
        guard let raw, !raw.isEmpty else { return [] }
        return raw
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func decisionPriority(_ value: String) -> Int {
        switch value.uppercased() {
        case "BLOCK_CONTRADICTORY", "BLOCK_CONCENTRATION": 4
        case "SUPPRESS_REDUNDANT", "PREFER_ALTERNATIVE_EXPRESSION": 3
        case "ALLOW_REDUCED": 2
        default: 1
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
