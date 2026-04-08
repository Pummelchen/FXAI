import Foundation

public struct NewsPulseArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> NewsPulseSnapshot? {
        let statusURL = projectRoot.appendingPathComponent("Tools/OfflineLab/NewsPulse/newspulse_status.json", isDirectory: false)
        guard
            let data = try? Data(contentsOf: statusURL),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        let generatedAt = parseDate(raw["generated_at"])
        let queryCount = raw["query_count"] as? Int ?? 0
        let sourceStatuses = parseSourceStatuses(raw["source_status"] as? [String: Any] ?? [:])
        let currencies = parseCurrencies(raw["currencies"] as? [String: Any] ?? [:])
        let pairs = parsePairs(raw["pairs"] as? [String: Any] ?? [:])
        let recentItems = parseRecentItems(raw["recent_items"] as? [[String: Any]] ?? [])
        let artifactPaths = parseArtifactPaths(raw["artifacts"] as? [String: Any] ?? [:])

        return NewsPulseSnapshot(
            generatedAt: generatedAt,
            queryCount: queryCount,
            sourceStatuses: sourceStatuses,
            currencies: currencies,
            pairs: pairs,
            recentItems: recentItems,
            artifactPaths: artifactPaths
        )
    }

    private func parseSourceStatuses(_ raw: [String: Any]) -> [NewsPulseSourceStatus] {
        raw.map { key, value in
            let payload = value as? [String: Any] ?? [:]
            return NewsPulseSourceStatus(
                id: key,
                ok: payload["ok"] as? Bool ?? false,
                stale: payload["stale"] as? Bool ?? true,
                lastUpdateAt: parseDate(payload["last_update_at"]),
                lastPollAt: parseDate(payload["last_poll_at"]),
                lastSuccessAt: parseDate(payload["last_success_at"]),
                cursor: payload["cursor"] as? String,
                lastError: payload["last_error"] as? String
            )
        }
        .sorted { $0.id < $1.id }
    }

    private func parseCurrencies(_ raw: [String: Any]) -> [NewsPulseCurrencyState] {
        raw.compactMap { currency, value in
            let payload = value as? [String: Any] ?? [:]
            return NewsPulseCurrencyState(
                currency: currency,
                breakingCount15m: payload["breaking_count_15m"] as? Int ?? 0,
                intensity15m: payload["intensity_15m"] as? Double ?? 0,
                toneMean15m: payload["tone_mean_15m"] as? Double ?? 0,
                toneAbsMean15m: payload["tone_abs_mean_15m"] as? Double ?? 0,
                burstScore15m: payload["burst_score_15m"] as? Double ?? 0,
                nextHighImpactETAMin: payload["next_high_impact_eta_min"] as? Int,
                timeSinceLastHighImpactMin: payload["time_since_last_high_impact_min"] as? Int,
                inPreEventWindow: payload["in_pre_event_window"] as? Bool ?? false,
                inPostEventWindow: payload["in_post_event_window"] as? Bool ?? false,
                lastSurpriseProxy: payload["last_surprise_proxy"] as? Double,
                stale: payload["stale"] as? Bool ?? true,
                riskScore: payload["risk_score"] as? Double ?? 0,
                reasons: payload["reasons"] as? [String] ?? []
            )
        }
        .sorted {
            if $0.riskScore == $1.riskScore {
                return $0.currency < $1.currency
            }
            return $0.riskScore > $1.riskScore
        }
    }

    private func parsePairs(_ raw: [String: Any]) -> [NewsPulsePairState] {
        raw.compactMap { pair, value in
            let payload = value as? [String: Any] ?? [:]
            return NewsPulsePairState(
                pair: pair,
                eventETAMin: payload["event_eta_min"] as? Int,
                newsRiskScore: payload["news_risk_score"] as? Double ?? 0,
                tradeGate: payload["trade_gate"] as? String ?? "UNKNOWN",
                newsPressure: payload["news_pressure"] as? Double ?? 0,
                stale: payload["stale"] as? Bool ?? true,
                reasons: payload["reasons"] as? [String] ?? []
            )
        }
        .sorted {
            if $0.newsRiskScore == $1.newsRiskScore {
                return $0.pair < $1.pair
            }
            return $0.newsRiskScore > $1.newsRiskScore
        }
    }

    private func parseRecentItems(_ raw: [[String: Any]]) -> [NewsPulseRecentItem] {
        raw.compactMap { item in
            NewsPulseRecentItem(
                id: item["id"] as? String ?? UUID().uuidString,
                source: item["source"] as? String ?? "unknown",
                publishedAt: parseDate(item["published_at"]),
                seenAt: parseDate(item["seen_at"]),
                currencyTags: item["currency_tags"] as? [String] ?? [],
                topicTags: item["topic_tags"] as? [String] ?? [],
                domain: item["domain"] as? String ?? "",
                title: item["title"] as? String ?? "Untitled",
                url: (item["url"] as? String).flatMap(URL.init(string:)),
                importance: item["importance"].map(String.init(describing:)),
                tone: item["tone"] as? Double ?? 0
            )
        }
    }

    private func parseArtifactPaths(_ raw: [String: Any]) -> [KeyValueRecord] {
        raw
            .map { key, value in KeyValueRecord(key: key, value: String(describing: value)) }
            .sorted { $0.key < $1.key }
    }

    private func parseDate(_ raw: Any?) -> Date? {
        guard let text = raw as? String, !text.isEmpty else { return nil }
        return ISO8601DateFormatter().date(from: text)
    }
}
