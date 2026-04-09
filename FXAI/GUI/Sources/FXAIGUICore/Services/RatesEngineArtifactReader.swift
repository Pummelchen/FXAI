import Foundation

public struct RatesEngineArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> RatesEngineSnapshot? {
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/RatesEngine", isDirectory: true)
            .appendingPathComponent("rates_engine_status.json", isDirectory: false)
        guard let document = parseJSON(statusURL) else {
            return nil
        }

        let generatedAt = parseDate(document["generated_at"]) ?? modificationDate(for: statusURL) ?? Date()
        let sourceStatuses = parseSourceStatuses(document["source_status"] as? [String: Any] ?? [:])
        let currencies = parseCurrencies(document["currencies"] as? [String: Any] ?? [:])
        let pairs = parsePairs(document["pairs"] as? [String: Any] ?? [:])
        let recentPolicyEvents = parseEvents(document["recent_policy_events"] as? [[String: Any]] ?? [])
        let healthSummary = keyValueRecords(from: document["health"] as? [String: Any] ?? [:])
        let artifactPaths = keyValueRecords(from: document["artifacts"] as? [String: Any] ?? [:])

        guard !currencies.isEmpty || !pairs.isEmpty else {
            return nil
        }
        return RatesEngineSnapshot(
            generatedAt: generatedAt,
            sourceStatuses: sourceStatuses,
            currencies: currencies,
            pairs: pairs,
            recentPolicyEvents: recentPolicyEvents,
            healthSummary: healthSummary,
            artifactPaths: artifactPaths
        )
    }

    private func parseSourceStatuses(_ raw: [String: Any]) -> [RatesEngineSourceStatus] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return RatesEngineSourceStatus(
                id: key,
                ok: parseBool(spec["ok"]) ?? false,
                stale: parseBool(spec["stale"]) ?? false,
                enabled: parseBool(spec["enabled"]) ?? true,
                required: parseBool(spec["required"]) ?? false,
                lastUpdateAt: parseDate(spec["last_update_at"]),
                mode: spec["mode"] as? String,
                coverageRatio: parseDouble(spec["coverage_ratio"]),
                updatedCurrencies: parseInt(spec["updated_currencies"])
            )
        }
        .sorted { $0.id < $1.id }
    }

    private func parseCurrencies(_ raw: [String: Any]) -> [RatesEngineCurrencyState] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return RatesEngineCurrencyState(
                currency: key,
                frontEndLevel: parseDouble(spec["front_end_level"]),
                frontEndBasis: spec["front_end_basis"] as? String ?? "unknown",
                frontEndChange1d: parseDouble(spec["front_end_change_1d"]),
                frontEndChange5d: parseDouble(spec["front_end_change_5d"]),
                expectedPathLevel: parseDouble(spec["expected_path_level"]),
                expectedPathBasis: spec["expected_path_basis"] as? String ?? "unknown",
                expectedPathChange1d: parseDouble(spec["expected_path_change_1d"]),
                expectedPathChange5d: parseDouble(spec["expected_path_change_5d"]),
                curveSlope2s10s: parseDouble(spec["curve_slope_2s10s"]),
                curveBasis: spec["curve_basis"] as? String ?? "unknown",
                curveShapeRegime: spec["curve_shape_regime"] as? String ?? "UNKNOWN",
                policyRepricingScore: parseDouble(spec["policy_repricing_score"]) ?? 0,
                policySurpriseScore: parseDouble(spec["policy_surprise_score"]) ?? 0,
                policyUncertaintyScore: parseDouble(spec["policy_uncertainty_score"]) ?? 0,
                policyDirectionScore: parseDouble(spec["policy_direction_score"]) ?? 0,
                policyRelevanceScore: parseDouble(spec["policy_relevance_score"]) ?? 0,
                preCBEventWindow: parseBool(spec["pre_cb_event_window"]) ?? false,
                postCBEventWindow: parseBool(spec["post_cb_event_window"]) ?? false,
                preMacroPolicyWindow: parseBool(spec["pre_macro_policy_window"]) ?? false,
                postMacroPolicyWindow: parseBool(spec["post_macro_policy_window"]) ?? false,
                meetingPathRepriceNow: parseBool(spec["meeting_path_reprice_now"]) ?? false,
                macroToRatesTransmissionScore: parseDouble(spec["macro_to_rates_transmission_score"]) ?? 0,
                stale: parseBool(spec["stale"]) ?? true,
                reasons: spec["reasons"] as? [String] ?? []
            )
        }
        .sorted { lhs, rhs in
            if lhs.policyRepricingScore == rhs.policyRepricingScore {
                return lhs.currency < rhs.currency
            }
            return lhs.policyRepricingScore > rhs.policyRepricingScore
        }
    }

    private func parsePairs(_ raw: [String: Any]) -> [RatesEnginePairState] {
        raw.compactMap { key, value in
            guard let spec = value as? [String: Any] else { return nil }
            return RatesEnginePairState(
                pair: key,
                baseCurrency: spec["base_currency"] as? String ?? "",
                quoteCurrency: spec["quote_currency"] as? String ?? "",
                frontEndDiff: parseDouble(spec["front_end_diff"]),
                expectedPathDiff: parseDouble(spec["expected_path_diff"]),
                curveDivergenceScore: parseDouble(spec["curve_divergence_score"]) ?? 0,
                policyDivergenceScore: parseDouble(spec["policy_divergence_score"]) ?? 0,
                ratesRegime: spec["rates_regime"] as? String ?? "UNKNOWN",
                ratesRiskScore: parseDouble(spec["rates_risk_score"]) ?? 0,
                tradeGate: spec["trade_gate"] as? String ?? "UNKNOWN",
                policyAlignment: spec["policy_alignment"] as? String ?? "balanced",
                meetingPathRepriceNow: parseBool(spec["meeting_path_reprice_now"]) ?? false,
                macroToRatesTransmissionScore: parseDouble(spec["macro_to_rates_transmission_score"]) ?? 0,
                stale: parseBool(spec["stale"]) ?? true,
                brokerSymbols: spec["broker_symbols"] as? [String] ?? [],
                reasons: spec["reasons"] as? [String] ?? []
            )
        }
        .sorted { lhs, rhs in
            if lhs.tradeGate == rhs.tradeGate {
                if lhs.ratesRiskScore == rhs.ratesRiskScore {
                    return lhs.pair < rhs.pair
                }
                return lhs.ratesRiskScore > rhs.ratesRiskScore
            }
            return posturePriority(lhs.tradeGate) > posturePriority(rhs.tradeGate)
        }
    }

    private func parseEvents(_ raw: [[String: Any]]) -> [RatesEnginePolicyEvent] {
        raw.map { item in
            RatesEnginePolicyEvent(
                id: item["id"] as? String ?? UUID().uuidString,
                currency: item["currency"] as? String ?? "UNK",
                source: item["source"] as? String ?? "",
                domain: item["domain"] as? String ?? "",
                publishedAt: parseDate(item["published_at"]),
                title: item["title"] as? String ?? "",
                url: URL(string: item["url"] as? String ?? ""),
                policyRelevanceScore: parseDouble(item["policy_relevance_score"]) ?? 0,
                direction: parseDouble(item["direction"]) ?? 0,
                centralBankEvent: parseBool(item["central_bank_event"]) ?? false,
                macroPolicyEvent: parseBool(item["macro_policy_event"]) ?? false
            )
        }
        .sorted { ($0.publishedAt ?? .distantPast) > ($1.publishedAt ?? .distantPast) }
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
