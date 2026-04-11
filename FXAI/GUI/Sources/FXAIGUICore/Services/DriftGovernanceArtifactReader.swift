import Foundation

public struct DriftGovernanceArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> DriftGovernanceSnapshot? {
        let reportURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DriftGovernance/Reports", isDirectory: true)
            .appendingPathComponent("drift_governance_report.json", isDirectory: false)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/DriftGovernance", isDirectory: true)
            .appendingPathComponent("drift_governance_status.json", isDirectory: false)

        guard let report = parseJSON(reportURL) else {
            return nil
        }
        let status = parseJSON(statusURL) ?? [:]
        let symbols = parseSymbols(report["symbols"] as? [[String: Any]] ?? [])
        guard !symbols.isEmpty else {
            return nil
        }

        let generatedAt = parseDate(report["generated_at"]) ?? modificationDate(for: reportURL) ?? Date()
        let statusRecords = keyValueRecords(
            from: status.filter { key, _ in key != "artifacts" && key != "latest_actions" }
        )
        let artifactPaths = keyValueRecords(
            from: (status["artifacts"] as? [String: Any]) ?? (report["artifacts"] as? [String: Any]) ?? [:]
        )

        return DriftGovernanceSnapshot(
            generatedAt: generatedAt,
            profileName: report["profile_name"] as? String ?? "",
            policyVersion: parseInt(report["policy_version"]) ?? 1,
            actionMode: report["action_mode"] as? String ?? "AUTO_APPLY_PROTECTIVE",
            symbolCount: parseInt(report["symbol_count"]) ?? symbols.count,
            pluginCount: parseInt(report["plugin_count"]) ?? symbols.reduce(0) { $0 + $1.pluginCount },
            latestActionCount: parseInt(report["latest_action_count"]) ?? 0,
            healthCounts: keyValueRecords(from: report["health_counts"] as? [String: Any] ?? [:]),
            governanceCounts: keyValueRecords(from: report["governance_counts"] as? [String: Any] ?? [:]),
            actionCounts: keyValueRecords(from: report["action_counts"] as? [String: Any] ?? [:]),
            statusRecords: statusRecords,
            artifactPaths: artifactPaths,
            symbols: symbols
        )
    }

    private func parseSymbols(_ raw: [[String: Any]]) -> [DriftGovernanceSymbolSnapshot] {
        raw.compactMap { item in
            guard let symbol = item["symbol"] as? String, !symbol.isEmpty else { return nil }
            return DriftGovernanceSymbolSnapshot(
                symbol: symbol,
                pluginCount: parseInt(item["plugin_count"]) ?? 0,
                healthCounts: keyValueRecords(from: item["health_counts"] as? [String: Any] ?? [:]),
                governanceCounts: keyValueRecords(from: item["governance_counts"] as? [String: Any] ?? [:]),
                actionCounts: keyValueRecords(from: item["action_counts"] as? [String: Any] ?? [:]),
                latestContext: contextSummary(from: item["latest_context"] as? [String: Any] ?? [:]),
                plugins: parsePlugins(item["plugins"] as? [[String: Any]] ?? []),
                recentActions: parseActions(item["recent_actions"] as? [[String: Any]] ?? [])
            )
        }
        .sorted { $0.symbol < $1.symbol }
    }

    private func parsePlugins(_ raw: [[String: Any]]) -> [DriftGovernancePluginSnapshot] {
        raw.compactMap { item in
            guard let pluginName = item["plugin_name"] as? String, !pluginName.isEmpty else { return nil }
            return DriftGovernancePluginSnapshot(
                pluginName: pluginName,
                familyID: parseInt(item["family_id"]) ?? 11,
                familyName: item["family_name"] as? String ?? "",
                baseRegistryStatus: item["base_registry_status"] as? String ?? "",
                healthState: item["health_state"] as? String ?? "HEALTHY",
                governanceState: item["governance_state"] as? String ?? "HEALTHY",
                recommendedGovernanceState: item["recommended_governance_state"] as? String ?? "HEALTHY",
                actionRecommendation: item["action_recommendation"] as? String ?? "NONE",
                actionApplied: parseBool(item["action_applied"]),
                weightMultiplier: parseDouble(item["weight_multiplier"]) ?? 1.0,
                restrictLive: parseBool(item["restrict_live"]),
                shadowOnly: parseBool(item["shadow_only"]),
                disabled: parseBool(item["disabled"]),
                aggregateRiskScore: parseDouble(item["aggregate_risk_score"]) ?? 0,
                driftScores: keyValueRecords(from: item["drift_scores"] as? [String: Any] ?? [:]),
                support: keyValueRecords(from: item["support"] as? [String: Any] ?? [:]),
                reasonCodes: item["reason_codes"] as? [String] ?? [],
                qualityFlags: keyValueRecords(from: item["quality_flags"] as? [String: Any] ?? [:]),
                contextSummary: contextSummary(from: item["context"] as? [String: Any] ?? [:]),
                challengerEvaluation: parseChallenger(item["challenger_evaluation"] as? [String: Any] ?? [:])
            )
        }
        .sorted { lhs, rhs in
            if lhs.aggregateRiskScore == rhs.aggregateRiskScore {
                return lhs.pluginName < rhs.pluginName
            }
            return lhs.aggregateRiskScore > rhs.aggregateRiskScore
        }
    }

    private func parseActions(_ raw: [[String: Any]]) -> [DriftGovernanceActionRecord] {
        raw.compactMap { item in
            guard let pluginName = item["plugin_name"] as? String, !pluginName.isEmpty else { return nil }
            return DriftGovernanceActionRecord(
                pluginName: pluginName,
                previousState: item["previous_state"] as? String ?? "",
                newState: item["new_state"] as? String ?? "",
                actionKind: item["action_kind"] as? String ?? "NONE",
                actionApplied: parseBool(item["action_applied"]),
                createdAt: parseDate(item["created_at"])
            )
        }
        .sorted { ($0.createdAt ?? .distantPast) > ($1.createdAt ?? .distantPast) }
    }

    private func parseChallenger(_ raw: [String: Any]) -> DriftGovernanceChallengerEvaluation? {
        guard !raw.isEmpty else { return nil }
        return DriftGovernanceChallengerEvaluation(
            eligibilityState: raw["eligibility_state"] as? String ?? "INSUFFICIENT",
            qualifies: parseBool(raw["qualifies"]),
            supportCount: parseInt(raw["support_count"]) ?? 0,
            shadowSupport: parseInt(raw["shadow_support"]) ?? 0,
            walkforwardScore: parseDouble(raw["walkforward_score"]) ?? 0,
            recentScore: parseDouble(raw["recent_score"]) ?? 0,
            adversarialScore: parseDouble(raw["adversarial_score"]) ?? 0,
            macroEventScore: parseDouble(raw["macro_event_score"]) ?? 0,
            calibrationError: parseDouble(raw["calibration_error"]) ?? 0,
            issueCount: parseDouble(raw["issue_count"]) ?? 0,
            liveShadowScore: parseDouble(raw["live_shadow_score"]) ?? 0,
            liveReliability: parseDouble(raw["live_reliability"]) ?? 0,
            portfolioScore: parseDouble(raw["portfolio_score"]) ?? 0,
            promotionMargin: parseDouble(raw["promotion_margin"]) ?? 0
        )
    }

    private func contextSummary(from raw: [String: Any]) -> [KeyValueRecord] {
        guard !raw.isEmpty else { return [] }
        var records: [KeyValueRecord] = []

        if let prob = raw["prob_calibration"] as? [String: Any] {
            if let uncertainty = parseDouble(prob["average_uncertainty_score"]) {
                records.append(KeyValueRecord(key: "prob.uncertainty", value: decimalString(uncertainty)))
            }
            if let quality = parseDouble(prob["average_quality"]) {
                records.append(KeyValueRecord(key: "prob.quality", value: decimalString(quality)))
            }
        }
        if let execution = raw["execution_quality"] as? [String: Any] {
            if let quality = parseDouble(execution["min_execution_quality_score"]) {
                records.append(KeyValueRecord(key: "execution.min_quality", value: decimalString(quality)))
            }
            if let slippage = parseDouble(execution["max_slippage_risk"]) {
                records.append(KeyValueRecord(key: "execution.max_slippage", value: decimalString(slippage)))
            }
            if let latest = execution["latest"] as? [String: Any],
               let state = latest["execution_state"] as? String, !state.isEmpty {
                records.append(KeyValueRecord(key: "execution.state", value: state))
            }
        }
        if let adaptive = raw["adaptive_router"] as? [String: Any],
           let latest = adaptive["latest"] as? [String: Any] {
            if let router = latest["router"] as? [String: Any],
               let posture = router["trade_posture"] as? String, !posture.isEmpty {
                records.append(KeyValueRecord(key: "router.posture", value: posture))
            }
            if let regime = latest["regime"] as? [String: Any],
               let label = regime["top_label"] as? String, !label.isEmpty {
                records.append(KeyValueRecord(key: "router.regime", value: label))
            }
        }
        if let crossAsset = raw["cross_asset"] as? [String: Any],
           let latest = crossAsset["latest"] as? [String: Any] {
            if let state = latest["macro_state"] as? String, !state.isEmpty {
                records.append(KeyValueRecord(key: "cross_asset.macro", value: state))
            }
            if let risk = parseDouble(latest["pair_cross_asset_risk_score"]) {
                records.append(KeyValueRecord(key: "cross_asset.risk", value: decimalString(risk)))
            }
        }
        if let micro = raw["microstructure"] as? [String: Any] {
            if let hostile = parseDouble(micro["max_hostile_execution_score"]) {
                records.append(KeyValueRecord(key: "micro.max_hostile", value: decimalString(hostile)))
            }
            if let liquidity = parseDouble(micro["max_liquidity_stress_score"]) {
                records.append(KeyValueRecord(key: "micro.max_liquidity", value: decimalString(liquidity)))
            }
        }
        if let ensemble = raw["dynamic_ensemble"] as? [String: Any] {
            if let quality = parseDouble(ensemble["average_quality"]) {
                records.append(KeyValueRecord(key: "ensemble.quality", value: decimalString(quality)))
            }
            if let abstain = parseDouble(ensemble["max_abstain_bias"]) {
                records.append(KeyValueRecord(key: "ensemble.max_abstain", value: decimalString(abstain)))
            }
        }
        if let newspulse = raw["newspulse"] as? [String: Any] {
            if let gate = newspulse["trade_gate"] as? String, !gate.isEmpty {
                records.append(KeyValueRecord(key: "newspulse.trade_gate", value: gate))
            }
            if let risk = parseDouble(newspulse["news_risk_score"]) {
                records.append(KeyValueRecord(key: "newspulse.risk", value: decimalString(risk)))
            }
        }

        if records.isEmpty {
            return flatten(raw).prefix(10).map { $0 }
        }
        return records.sorted { $0.key < $1.key }
    }

    private func flatten(_ raw: [String: Any], prefix: String = "", depth: Int = 0) -> [KeyValueRecord] {
        raw.keys.sorted().flatMap { key in
            let value = raw[key]
            let next = prefix.isEmpty ? key : "\(prefix).\(key)"
            if depth < 1, let nested = value as? [String: Any] {
                return flatten(nested, prefix: next, depth: depth + 1)
            }
            return [KeyValueRecord(key: next, value: displayValue(value))]
        }
    }

    private func keyValueRecords(from raw: [String: Any]) -> [KeyValueRecord] {
        raw.keys.sorted().map { key in
            KeyValueRecord(key: key, value: displayValue(raw[key]))
        }
    }

    private func displayValue(_ raw: Any?) -> String {
        switch raw {
        case let value as String:
            return value
        case let value as Double:
            return decimalString(value)
        case let value as Int:
            return String(value)
        case let value as Bool:
            return value ? "true" : "false"
        case let value as [String]:
            return value.prefix(4).joined(separator: ", ")
        case let value as [Any]:
            return "\(value.count) items"
        case let value as [String: Any]:
            return "\(value.count) fields"
        default:
            return raw.map { String(describing: $0) } ?? ""
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
        if let value = raw as? Int {
            return Date(timeIntervalSince1970: TimeInterval(value))
        }
        if let value = raw as? Double {
            return Date(timeIntervalSince1970: value)
        }
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
        if let value = raw as? Double { return Int(value) }
        if let text = raw as? String { return Int(text) }
        return nil
    }

    private func parseBool(_ raw: Any?) -> Bool {
        if let value = raw as? Bool { return value }
        if let value = raw as? Int { return value != 0 }
        if let value = raw as? Double { return value != 0 }
        if let text = raw as? String {
            return ["1", "true", "yes"].contains(text.lowercased())
        }
        return false
    }

    private func decimalString(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private func makeDateFormatter(fractional: Bool) -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = fractional
            ? [.withInternetDateTime, .withFractionalSeconds]
            : [.withInternetDateTime]
        return formatter
    }
}
