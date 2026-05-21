import Foundation

public struct LabelEngineArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> LabelEngineSnapshot? {
        let reportURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/LabelEngine/Reports", isDirectory: true)
            .appendingPathComponent("label_engine_report.json", isDirectory: false)
        let statusURL = projectRoot
            .appendingPathComponent("Tools/OfflineLab/LabelEngine", isDirectory: true)
            .appendingPathComponent("label_engine_status.json", isDirectory: false)

        guard let report = parseJSON(reportURL) else {
            return nil
        }
        let status = parseJSON(statusURL) ?? [:]
        let builds = parseBuilds(report["builds"] as? [[String: Any]] ?? [])
        guard !builds.isEmpty else {
            return nil
        }
        let generatedAt = parseDate(report["generated_at"]) ?? modificationDate(for: reportURL) ?? Date()
        let artifactCount = parseInt(report["artifact_count"]) ?? builds.count
        let latestDatasetKey = report["latest_dataset_key"] as? String ?? builds.first?.datasetKey ?? ""
        let statusRecords = keyValueRecords(
            from: status.filter { key, _ in key != "artifacts" }
        )
        let artifactPaths = keyValueRecords(from: status["artifacts"] as? [String: Any] ?? [:])

        return LabelEngineSnapshot(
            generatedAt: generatedAt,
            artifactCount: artifactCount,
            latestDatasetKey: latestDatasetKey,
            builds: builds,
            statusRecords: statusRecords,
            artifactPaths: artifactPaths
        )
    }

    private func parseBuilds(_ raw: [[String: Any]]) -> [LabelEngineBuildSnapshot] {
        raw.compactMap { item in
            guard let datasetKey = item["dataset_key"] as? String, !datasetKey.isEmpty else { return nil }
            let summaryMetrics = keyValueRecords(from: item["summary_metrics"] as? [String: Any] ?? [:])
            let metaSummary = keyValueRecords(from: item["meta_summary"] as? [String: Any] ?? [:])
            let qualityFlags = keyValueRecords(from: item["quality_flags"] as? [String: Any] ?? [:])
            let artifactPaths = keyValueRecords(from: item["artifact_paths"] as? [String: Any] ?? [:])
            let topReasons = parseTopReasons(item["top_reason_codes"] as? [[String: Any]] ?? [])
            let horizons = parseHorizons(item["horizon_summaries"] as? [[String: Any]] ?? [])
            return LabelEngineBuildSnapshot(
                datasetKey: datasetKey,
                profileName: item["profile_name"] as? String ?? "",
                symbol: item["symbol"] as? String ?? "",
                timeframe: item["timeframe"] as? String ?? "M1",
                barCount: parseInt(item["bar_count"]) ?? 0,
                pointSize: parseDouble(item["point_size"]) ?? 0,
                executionProfile: item["execution_profile"] as? String ?? "default",
                labelVersion: parseInt(item["label_version"]) ?? 1,
                generatedAt: parseDate(item["generated_at"]),
                summaryMetrics: summaryMetrics,
                metaSummary: metaSummary,
                qualityFlags: qualityFlags,
                artifactPaths: artifactPaths,
                topReasons: topReasons,
                horizons: horizons
            )
        }
        .sorted { lhs, rhs in
            if lhs.generatedAt == rhs.generatedAt {
                return lhs.datasetKey < rhs.datasetKey
            }
            return (lhs.generatedAt ?? .distantPast) > (rhs.generatedAt ?? .distantPast)
        }
    }

    private func parseHorizons(_ raw: [[String: Any]]) -> [LabelEngineHorizonSnapshot] {
        raw.compactMap { item in
            guard let horizonID = item["horizon_id"] as? String, !horizonID.isEmpty else { return nil }
            return LabelEngineHorizonSnapshot(
                horizonID: horizonID,
                bars: parseInt(item["bars"]) ?? 0,
                sampleCount: parseInt(item["sample_count"]) ?? 0,
                longTradeabilityRate: parseDouble(item["long_tradeability_rate"]) ?? 0,
                shortTradeabilityRate: parseDouble(item["short_tradeability_rate"]) ?? 0,
                candidateCount: parseInt(item["candidate_count"]) ?? 0,
                candidateAcceptanceRate: parseDouble(item["candidate_acceptance_rate"]) ?? 0,
                meanCostAdjustedReturnPoints: parseDouble(item["mean_cost_adjusted_return_points"]) ?? 0,
                medianTimeToFavorableHitSec: parseDouble(item["median_time_to_favorable_hit_sec"])
            )
        }
        .sorted { $0.bars < $1.bars }
    }

    private func parseTopReasons(_ raw: [[String: Any]]) -> [LabelEngineReasonCount] {
        raw.compactMap { item in
            guard let reason = item["reason"] as? String, !reason.isEmpty else { return nil }
            return LabelEngineReasonCount(reason: reason, count: parseInt(item["count"]) ?? 0)
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

    private func makeDateFormatter(fractional: Bool) -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = fractional
            ? [.withInternetDateTime, .withFractionalSeconds]
            : [.withInternetDateTime]
        return formatter
    }
}
