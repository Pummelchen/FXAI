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
        let officialQueryCount = raw["official_query_count"] as? Int ?? 0
        let sourceStatuses = parseSourceStatuses(raw["source_status"] as? [String: Any] ?? [:])
        let currencies = parseCurrencies(raw["currencies"] as? [String: Any] ?? [:])
        let pairs = parsePairs(raw["pairs"] as? [String: Any] ?? [:])
        let recentItems = parseRecentItems(raw["recent_items"] as? [[String: Any]] ?? [])
        let stories = parseStories(raw["stories"] as? [[String: Any]] ?? [])
        let pairTimelines = parsePairTimelines(raw["pair_timelines"] as? [String: Any] ?? [:])
        let sourceHealthTimeline = parseSourceHealthTimeline(raw["source_health_timeline"] as? [[String: Any]] ?? [])
        let daemon = parseDaemon(raw["daemon"] as? [String: Any] ?? [:])
        let policySummary = parsePolicySummary(raw["policy_summary"] as? [String: Any] ?? [:])
        let healthSummary = parseHealthSummary(raw["health"] as? [String: Any] ?? [:])
        let artifactPaths = parseArtifactPaths(raw["artifacts"] as? [String: Any] ?? [:])

        return NewsPulseSnapshot(
            generatedAt: generatedAt,
            queryCount: queryCount,
            officialQueryCount: officialQueryCount,
            sourceStatuses: sourceStatuses,
            currencies: currencies,
            pairs: pairs,
            recentItems: recentItems,
            stories: stories,
            pairTimelines: pairTimelines,
            sourceHealthTimeline: sourceHealthTimeline,
            daemon: daemon,
            policySummary: policySummary,
            healthSummary: healthSummary,
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
                enabled: payload["enabled"] as? Bool ?? true,
                required: payload["required"] as? Bool ?? false,
                lastUpdateAt: parseDate(payload["last_update_at"]),
                lastPollAt: parseDate(payload["last_poll_at"]),
                lastSuccessAt: parseDate(payload["last_success_at"]),
                backoffUntil: parseDate(payload["backoff_until"]),
                cursor: payload["cursor"] as? String,
                lastError: payload["last_error"] as? String,
                budgetExhausted: payload["budget_exhausted"] as? Bool ?? false,
                throttled: payload["throttled"] as? Bool ?? false
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
                storyCount15m: payload["story_count_15m"] as? Int ?? 0,
                storySeverity15m: payload["story_severity_15m"] as? Double ?? 0,
                officialCount24h: payload["official_count_24h"] as? Int ?? 0,
                dominantStoryIDs: payload["dominant_story_ids"] as? [String] ?? [],
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
                baseCurrency: payload["base_currency"] as? String ?? String(pair.prefix(3)),
                quoteCurrency: payload["quote_currency"] as? String ?? String(pair.suffix(3)),
                eventETAMin: payload["event_eta_min"] as? Int,
                newsRiskScore: payload["news_risk_score"] as? Double ?? 0,
                tradeGate: payload["trade_gate"] as? String ?? "UNKNOWN",
                newsPressure: payload["news_pressure"] as? Double ?? 0,
                stale: payload["stale"] as? Bool ?? true,
                reasons: payload["reasons"] as? [String] ?? [],
                storyIDs: payload["story_ids"] as? [String] ?? [],
                watchlistTags: payload["watchlist_tags"] as? [String] ?? [],
                brokerSymbols: payload["broker_symbols"] as? [String] ?? [],
                sessionProfile: payload["session_profile"] as? String ?? "default",
                calibrationProfile: payload["calibration_profile"] as? String ?? "default",
                cautionLotScale: payload["caution_lot_scale"] as? Double,
                cautionEnterProbBuffer: payload["caution_enter_prob_buffer"] as? Double,
                gateChangedAt: parseDate(payload["gate_changed_at"])
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
                tone: item["tone"] as? Double ?? 0,
                storyID: item["story_id"] as? String
            )
        }
    }

    private func parseStories(_ raw: [[String: Any]]) -> [NewsPulseStory] {
        raw.compactMap { story in
            NewsPulseStory(
                id: story["id"] as? String ?? UUID().uuidString,
                latestTitle: story["latest_title"] as? String ?? "Untitled",
                firstPublishedAt: parseDate(story["first_published_at"]),
                lastPublishedAt: parseDate(story["last_published_at"]),
                currencyTags: story["currency_tags"] as? [String] ?? [],
                topicTags: story["topic_tags"] as? [String] ?? [],
                domains: story["domains"] as? [String] ?? [],
                representativeURL: (story["representative_url"] as? String).flatMap(URL.init(string:)),
                itemCount: story["item_count"] as? Int ?? 0,
                sourceCount: story["source_count"] as? Int ?? 0,
                officialHits: story["official_hits"] as? Int ?? 0,
                severityScore: story["severity_score"] as? Double ?? 0,
                active: story["active"] as? Bool ?? false,
                itemIDs: story["item_ids"] as? [String] ?? []
            )
        }
    }

    private func parsePairTimelines(_ raw: [String: Any]) -> [String: [NewsPulsePairTimelinePoint]] {
        var result: [String: [NewsPulsePairTimelinePoint]] = [:]
        for (pair, value) in raw {
            let items = value as? [[String: Any]] ?? []
            result[pair] = items.map { point in
                NewsPulsePairTimelinePoint(
                    observedAt: parseDate(point["observed_at"]),
                    tradeGate: point["trade_gate"] as? String ?? "UNKNOWN",
                    newsRiskScore: point["news_risk_score"] as? Double ?? 0,
                    newsPressure: point["news_pressure"] as? Double ?? 0,
                    stale: point["stale"] as? Bool ?? true,
                    eventETAMin: point["event_eta_min"] as? Int,
                    sessionProfile: point["session_profile"] as? String ?? "default",
                    calibrationProfile: point["calibration_profile"] as? String ?? "default",
                    watchlistTags: point["watchlist_tags"] as? [String] ?? [],
                    storyIDs: point["story_ids"] as? [String] ?? [],
                    reasons: point["reasons"] as? [String] ?? []
                )
            }
        }
        return result
    }

    private func parseSourceHealthTimeline(_ raw: [[String: Any]]) -> [NewsPulseSourceHealthPoint] {
        raw.map { point in
            NewsPulseSourceHealthPoint(
                observedAt: parseDate(point["observed_at"]),
                calendarOK: point["calendar_ok"] as? Bool ?? false,
                calendarStale: point["calendar_stale"] as? Bool ?? true,
                gdeltOK: point["gdelt_ok"] as? Bool ?? false,
                gdeltStale: point["gdelt_stale"] as? Bool ?? true,
                officialOK: point["official_ok"] as? Bool ?? true,
                officialStale: point["official_stale"] as? Bool ?? false
            )
        }
    }

    private func parseDaemon(_ raw: [String: Any]) -> NewsPulseDaemonStatus? {
        guard !raw.isEmpty else { return nil }
        return NewsPulseDaemonStatus(
            mode: raw["mode"] as? String ?? "standalone",
            heartbeatAt: parseDate(raw["heartbeat_at"]),
            lastCycleStartedAt: parseDate(raw["last_cycle_started_at"]),
            lastCycleFinishedAt: parseDate(raw["last_cycle_finished_at"]),
            lastCycleDurationSec: raw["last_cycle_duration_sec"] as? Double ?? 0,
            intervalSeconds: raw["interval_seconds"] as? Int ?? 0,
            cyclesCompleted: raw["cycles_completed"] as? Int ?? 0,
            consecutiveFailures: raw["consecutive_failures"] as? Int ?? 0,
            degraded: raw["degraded"] as? Bool ?? false,
            degradedReasons: raw["degraded_reasons"] as? [String] ?? [],
            lastError: raw["last_error"] as? String
        )
    }

    private func parsePolicySummary(_ raw: [String: Any]) -> NewsPulsePolicySummary? {
        guard !raw.isEmpty else { return nil }
        let watchlists = (raw["watchlists"] as? [String: Any] ?? [:])
            .map { key, value in
                KeyValueListRecord(key: key, values: value as? [String] ?? [])
            }
            .sorted { $0.key < $1.key }
        return NewsPulsePolicySummary(
            activePairs: raw["active_pairs"] as? [String] ?? [],
            watchlists: watchlists,
            brokerSymbolMapCount: raw["broker_symbol_map_count"] as? Int ?? 0
        )
    }

    private func parseHealthSummary(_ raw: [String: Any]) -> NewsPulseHealthSummary? {
        guard !raw.isEmpty else { return nil }
        return NewsPulseHealthSummary(
            requiredSourcesStale: raw["required_sources_stale"] as? Bool ?? false,
            gdeltBackoffUntil: parseDate(raw["gdelt_backoff_until"]),
            historyRecordsLocal: raw["history_records_local"] as? Int ?? 0,
            storyCount: raw["story_count"] as? Int ?? 0
        )
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
