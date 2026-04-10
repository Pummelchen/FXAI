import Foundation

public struct ProbCalibrationArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> ProbCalibrationSnapshot? {
        guard let runtimeDirectory = RuntimeArtifactPathResolver.runtimeDirectory(projectRoot: projectRoot) else {
            return nil
        }

        let replayURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/ProbabilisticCalibration/Reports", isDirectory: true)
            .appendingPathComponent("prob_calibration_replay_report.json", isDirectory: false)
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
                return name.hasPrefix("fxai_prob_calibration_")
                    && name.hasSuffix(".tsv")
                    && !name.contains("_history_")
            }

        let symbols = stateFiles.compactMap { url in
            let symbol = url.deletingPathExtension().lastPathComponent
                .replacingOccurrences(of: "fxai_prob_calibration_", with: "")
                .uppercased()
            return parseSymbol(url: url, replay: replayBySymbol[symbol] ?? [:])
        }
        .sorted { lhs, rhs in
            if lhs.abstain == rhs.abstain {
                if lhs.edgeAfterCostsPoints == rhs.edgeAfterCostsPoints {
                    return lhs.symbol < rhs.symbol
                }
                return lhs.edgeAfterCostsPoints < rhs.edgeAfterCostsPoints
            }
            return lhs.abstain && !rhs.abstain
        }

        guard !symbols.isEmpty else {
            return nil
        }

        let generatedAt = symbols.compactMap(\.generatedAt).max() ?? modificationDate(for: replayURL) ?? Date()
        return ProbCalibrationSnapshot(
            generatedAt: generatedAt,
            replayHoursBack: replayDocument?["hours_back"] as? Int ?? 72,
            symbols: symbols
        )
    }

    private func parseSymbol(url: URL, replay: [String: Any]) -> ProbCalibrationSymbolSnapshot? {
        let values = parseTSVMap(url) ?? [:]
        let symbol = values["symbol"] ?? url.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "fxai_prob_calibration_", with: "")
        guard !symbol.isEmpty else { return nil }

        let latest = replay["latest"] as? [String: Any] ?? [:]
        let latestState = latest["state"] as? [String: Any] ?? [:]
        let reasons = splitSemicolonSeparated(values["reasons_csv"]).ifEmpty(latestState["reason_codes"] as? [String] ?? [])

        return ProbCalibrationSymbolSnapshot(
            symbol: symbol,
            generatedAt: parseEpochDate(values["generated_at"]) ?? parseDate(latest["generated_at"]) ?? modificationDate(for: url),
            method: values["method"] ?? (latestState["method"] as? String) ?? "UNKNOWN",
            sessionLabel: values["session_label"] ?? (latestState["session_label"] as? String) ?? "UNKNOWN",
            regimeLabel: values["regime_label"] ?? (latestState["regime_label"] as? String) ?? "UNKNOWN",
            tierKind: values["selected_tier_kind"] ?? (latestState["selected_tier_kind"] as? String) ?? "GLOBAL",
            tierKey: values["selected_tier_key"] ?? (latestState["selected_tier_key"] as? String) ?? "GLOBAL|*|*|*",
            support: parseInt(values["selected_support"]) ?? (latestState["selected_support"] as? Int) ?? 0,
            quality: parseDouble(values["selected_quality"]) ?? (latestState["selected_quality"] as? Double) ?? 0,
            rawAction: values["raw_action"] ?? (latestState["raw_action"] as? String) ?? "SKIP",
            rawScore: parseDouble(values["raw_score"]) ?? (latestState["raw_score"] as? Double) ?? 0,
            rawBuyProb: parseDouble(values["raw_buy_prob"]) ?? (latestState["raw_buy_prob"] as? Double) ?? 0,
            rawSellProb: parseDouble(values["raw_sell_prob"]) ?? (latestState["raw_sell_prob"] as? Double) ?? 0,
            rawSkipProb: parseDouble(values["raw_skip_prob"]) ?? (latestState["raw_skip_prob"] as? Double) ?? 1,
            calibratedBuyProb: parseDouble(values["calibrated_buy_prob"]) ?? (latestState["calibrated_buy_prob"] as? Double) ?? 0,
            calibratedSellProb: parseDouble(values["calibrated_sell_prob"]) ?? (latestState["calibrated_sell_prob"] as? Double) ?? 0,
            calibratedSkipProb: parseDouble(values["calibrated_skip_prob"]) ?? (latestState["calibrated_skip_prob"] as? Double) ?? 1,
            calibratedConfidence: parseDouble(values["calibrated_confidence"]) ?? (latestState["calibrated_confidence"] as? Double) ?? 0,
            expectedMoveMeanPoints: parseDouble(values["expected_move_mean_points"]) ?? (latestState["expected_move_mean_points"] as? Double) ?? 0,
            expectedMoveQ25Points: parseDouble(values["expected_move_q25_points"]) ?? (latestState["expected_move_q25_points"] as? Double) ?? 0,
            expectedMoveQ50Points: parseDouble(values["expected_move_q50_points"]) ?? (latestState["expected_move_q50_points"] as? Double) ?? 0,
            expectedMoveQ75Points: parseDouble(values["expected_move_q75_points"]) ?? (latestState["expected_move_q75_points"] as? Double) ?? 0,
            spreadCostPoints: parseDouble(values["spread_cost_points"]) ?? (latestState["spread_cost_points"] as? Double) ?? 0,
            slippageCostPoints: parseDouble(values["slippage_cost_points"]) ?? (latestState["slippage_cost_points"] as? Double) ?? 0,
            uncertaintyScore: parseDouble(values["uncertainty_score"]) ?? (latestState["uncertainty_score"] as? Double) ?? 0,
            uncertaintyPenaltyPoints: parseDouble(values["uncertainty_penalty_points"]) ?? (latestState["uncertainty_penalty_points"] as? Double) ?? 0,
            riskPenaltyPoints: parseDouble(values["risk_penalty_points"]) ?? (latestState["risk_penalty_points"] as? Double) ?? 0,
            expectedGrossEdgePoints: parseDouble(values["expected_gross_edge_points"]) ?? (latestState["expected_gross_edge_points"] as? Double) ?? 0,
            edgeAfterCostsPoints: parseDouble(values["edge_after_costs_points"]) ?? (latestState["edge_after_costs_points"] as? Double) ?? 0,
            finalAction: values["final_action"] ?? (latestState["final_action"] as? String) ?? "SKIP",
            abstain: parseBool(values["abstain"]) ?? (latestState["abstain"] as? Bool) ?? true,
            fallbackUsed: parseBool(values["fallback_used"]) ?? (latestState["fallback_used"] as? Bool) ?? false,
            calibrationStale: parseBool(values["calibration_stale"]) ?? (latestState["calibration_stale"] as? Bool) ?? true,
            inputStale: parseBool(values["input_stale"]) ?? (latestState["input_stale"] as? Bool) ?? true,
            supportUsable: parseBool(values["support_usable"]) ?? (latestState["support_usable"] as? Bool) ?? false,
            reasons: reasons,
            replayActionCounts: keyValueRecords(from: replay["action_counts"] as? [String: Any] ?? [:]),
            replayTierCounts: keyValueRecords(from: replay["tier_counts"] as? [String: Any] ?? [:]),
            replayTopReasons: namedCounts(replay["top_reasons"] as? [[String: Any]] ?? [], nameKey: "reason"),
            recentTransitions: parseTransitions(replay["recent_transitions"] as? [[String: Any]] ?? []),
            observationCount: replay["observations"] as? Int ?? 0,
            abstainCount: replay["abstain_count"] as? Int ?? 0,
            fallbackCount: replay["fallback_count"] as? Int ?? 0,
            averageConfidence: replay["average_confidence"] as? Double ?? 0,
            averageEdgeAfterCostsPoints: replay["average_edge_after_costs_points"] as? Double ?? 0,
            averageUncertaintyScore: replay["average_uncertainty_score"] as? Double ?? 0,
            minEdgeAfterCostsPoints: replay["min_edge_after_costs_points"] as? Double ?? 0,
            maxEdgeAfterCostsPoints: replay["max_edge_after_costs_points"] as? Double ?? 0
        )
    }

    private func parseTransitions(_ raw: [[String: Any]]) -> [ProbCalibrationTransition] {
        raw.map { item in
            ProbCalibrationTransition(
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
