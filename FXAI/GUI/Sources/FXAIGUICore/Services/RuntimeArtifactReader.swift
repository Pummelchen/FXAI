import Foundation

public struct RuntimeArtifactReader {
    public init() {}

    public func read(projectRoot: URL) -> RuntimeOperationsSnapshot {
        let researchRoot = projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        let dashboardURL = latestFile(named: "operator_dashboard.json", under: researchRoot)
        let championsURL = latestFile(named: "champions.json", under: researchRoot)

        let dashboardDocument = dashboardURL.flatMap(parseDashboard)
        let champions = championsURL.flatMap(parseChampions) ?? []
        let deployments = dashboardDocument.map {
            parseDeployments(from: $0, profileDirectory: dashboardURL?.deletingLastPathComponent())
        } ?? []

        return RuntimeOperationsSnapshot(
            generatedAt: Date(),
            profileName: dashboardURL?.deletingLastPathComponent().lastPathComponent,
            deployments: deployments,
            champions: champions
        )
    }

    private func parseDeployments(from dashboard: [String: Any], profileDirectory: URL?) -> [RuntimeDeploymentDetail] {
        let deploymentEntries = dashboard["deployments"] as? [[String: Any]] ?? []

        return deploymentEntries.compactMap { entry in
            let liveState = entry["live_state"] as? [String: Any] ?? [:]
            let deploymentTSV = liveState["deployment_tsv"] as? [String: String] ?? [:]
            let routerTSV = liveState["router_tsv"] as? [String: String] ?? [:]
            let serviceTSV = liveState["service_tsv"] as? [String: String] ?? [:]
            let commandTSV = liveState["command_tsv"] as? [String: String] ?? [:]
            let worldTSV = liveState["world_tsv"] as? [String: String] ?? [:]

            let symbol = deploymentTSV["symbol"] ?? commandTSV["symbol"] ?? serviceTSV["symbol"] ?? "UNKNOWN"
            let profileName = deploymentTSV["profile_name"] ?? commandTSV["profile_name"] ?? "unknown"
            let promotionTier = deploymentTSV["promotion_tier"] ?? "unknown"
            let runtimeMode = deploymentTSV["runtime_mode"] ?? "unknown"

            let artifactPath = (entry["artifact_path"] as? String).map { URL(fileURLWithPath: $0) }
            let promotionsRoot = artifactPath?.deletingLastPathComponent()

            let studentRouterPath = promotionsRoot?.appendingPathComponent("fxai_student_router_\(symbol).tsv")
            let supervisorServicePath = promotionsRoot?.appendingPathComponent("fxai_supervisor_service_\(symbol).tsv")
            let supervisorCommandPath = promotionsRoot?.appendingPathComponent("fxai_supervisor_command_\(symbol).tsv")
            let worldPlanPath = promotionsRoot?.appendingPathComponent("fxai_world_plan_\(symbol).tsv")
            let attributionPath = promotionsRoot?.appendingPathComponent("fxai_attribution_\(symbol).tsv")

            let studentRouterJSON = profileDirectory?.appendingPathComponent("student_router_\(symbol).json")

            let pluginName = extractPluginName(from: entry)
            let createdAt = parseUnix(entry["created_at"])
            let reviewedAt = extractReviewedAt(from: entry)
            let health = parseArtifactHealth(entry["artifact_health"] as? [String: Any] ?? [:])

            let deploymentSection = RuntimeArtifactSection(
                title: "Deployment Profile",
                sourcePath: artifactPath,
                values: deploymentTSV.sortedKeyValues()
            )
            let routerSection = RuntimeArtifactSection(
                title: "Student Router",
                sourcePath: studentRouterPath,
                values: (parseTSV(studentRouterPath) ?? routerTSV).sortedKeyValues()
            )
            let serviceSection = RuntimeArtifactSection(
                title: "Supervisor Service",
                sourcePath: supervisorServicePath,
                values: (parseTSV(supervisorServicePath) ?? serviceTSV).sortedKeyValues()
            )
            let commandSection = RuntimeArtifactSection(
                title: "Supervisor Command",
                sourcePath: supervisorCommandPath,
                values: (parseTSV(supervisorCommandPath) ?? commandTSV).sortedKeyValues()
            )
            let worldSection = RuntimeArtifactSection(
                title: "World Plan",
                sourcePath: worldPlanPath,
                values: (parseTSV(worldPlanPath) ?? worldTSV).sortedKeyValues()
            )
            let attributionSection = RuntimeArtifactSection(
                title: "Attribution",
                sourcePath: attributionPath,
                values: parseTSV(attributionPath)?.sortedKeyValues() ?? []
            )

            let featureHighlights = parseKeyValuesFromJSONFile(studentRouterJSON, topLevelKey: "shadow_summary")
            let studentRouterWeights = parseKeyValuesFromJSONFile(studentRouterJSON, topLevelKey: "plugin_weights")
            let familyWeights = parseKeyValuesFromJSONFile(studentRouterJSON, topLevelKey: "family_weights")
            let prunedPlugins = parseStringArrayFromJSONFile(studentRouterJSON, topLevelKey: "pruned_plugins")

            return RuntimeDeploymentDetail(
                id: symbol,
                symbol: symbol,
                profileName: profileName,
                pluginName: pluginName,
                promotionTier: promotionTier,
                runtimeMode: runtimeMode,
                createdAt: createdAt,
                reviewedAt: reviewedAt,
                artifactHealth: health,
                deploymentPath: artifactPath,
                studentRouterPath: studentRouterPath,
                supervisorServicePath: supervisorServicePath,
                supervisorCommandPath: supervisorCommandPath,
                worldPlanPath: worldPlanPath,
                deploymentSections: [deploymentSection],
                routerSections: [routerSection],
                supervisorSections: [serviceSection],
                commandSections: [commandSection],
                worldSections: [worldSection],
                attributionSections: attributionSection.values.isEmpty ? [] : [attributionSection],
                featureHighlights: featureHighlights,
                studentRouterWeights: studentRouterWeights,
                familyWeights: familyWeights,
                prunedPlugins: prunedPlugins
            )
        }
        .sorted { $0.symbol < $1.symbol }
    }

    private func parseDashboard(_ url: URL) -> [String: Any]? {
        parseJSON(url)
    }

    private func parseChampions(_ url: URL) -> [PromotionChampionRecord] {
        guard
            let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            return []
        }

        return raw.compactMap { item in
            let symbol = item["symbol"] as? String ?? "UNKNOWN"
            let pluginName = item["plugin_name"] as? String ?? "unknown"
            let status = item["status"] as? String ?? "unknown"
            let promotionTier = item["promotion_tier"] as? String ?? "unknown"
            let championScore = item["champion_score"] as? Double ?? 0
            let challengerScore = item["challenger_score"] as? Double ?? 0
            let portfolioScore = item["portfolio_score"] as? Double ?? 0
            let reviewedAt = parseUnix(item["reviewed_at"])
            let setPath = (item["champion_set_path"] as? String).map { URL(fileURLWithPath: $0) }
            let profileName = item["profile_name"] as? String

            return PromotionChampionRecord(
                symbol: symbol,
                pluginName: pluginName,
                status: status,
                promotionTier: promotionTier,
                championScore: championScore,
                challengerScore: challengerScore,
                portfolioScore: portfolioScore,
                reviewedAt: reviewedAt,
                setPath: setPath,
                profileName: profileName
            )
        }
        .sorted { lhs, rhs in
            if lhs.reviewedAt == rhs.reviewedAt {
                return lhs.symbol < rhs.symbol
            }
            return (lhs.reviewedAt ?? .distantPast) > (rhs.reviewedAt ?? .distantPast)
        }
    }

    private func parseArtifactHealth(_ raw: [String: Any]) -> RuntimeArtifactHealth {
        RuntimeArtifactHealth(
            artifactExists: raw["artifact_exists"] as? Bool ?? false,
            staleArtifact: raw["stale_artifact"] as? Bool ?? true,
            missingDeployment: raw["missing_deployment"] as? Bool ?? true,
            missingRouter: raw["missing_router"] as? Bool ?? true,
            missingSupervisorService: raw["missing_supervisor_service"] as? Bool ?? true,
            missingSupervisorCommand: raw["missing_supervisor_command"] as? Bool ?? true,
            missingWorldPlan: raw["missing_world_plan"] as? Bool ?? true,
            artifactAgeSeconds: raw["artifact_age_sec"] as? Int ?? 0,
            performanceFailures: raw["performance_failures"] as? [String] ?? [],
            artifactSizeFailures: raw["artifact_size_failures"] as? [String] ?? []
        )
    }

    private func parseTSV(_ url: URL?) -> [String: String]? {
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

    private func parseKeyValuesFromJSONFile(_ url: URL?, topLevelKey: String) -> [KeyValueRecord] {
        guard
            let url,
            let raw = parseJSON(url),
            let nested = raw[topLevelKey] as? [String: Any]
        else {
            return []
        }

        return nested
            .map { key, value in KeyValueRecord(key: key, value: stringify(value)) }
            .sorted { $0.key < $1.key }
    }

    private func parseStringArrayFromJSONFile(_ url: URL?, topLevelKey: String) -> [String] {
        guard
            let url,
            let raw = parseJSON(url),
            let nested = raw[topLevelKey] as? [String]
        else {
            return []
        }
        return nested
    }

    private func parseJSON(_ url: URL) -> [String: Any]? {
        guard
            let data = try? Data(contentsOf: url),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return raw
    }

    private func extractPluginName(from entry: [String: Any]) -> String {
        let payload = entry["payload"] as? [String: Any]
        let champions = payload?["champions"] as? [[String: Any]]
        return (champions?.first?["plugin_name"] as? String) ?? "unknown"
    }

    private func extractReviewedAt(from entry: [String: Any]) -> Date? {
        let payload = entry["payload"] as? [String: Any]
        let champions = payload?["champions"] as? [[String: Any]]
        if let reviewed = (champions?.first?["reviewed_at"] as? NSNumber)?.doubleValue {
            return Date(timeIntervalSince1970: reviewed)
        }
        return nil
    }

    private func parseUnix(_ raw: Any?) -> Date? {
        if let value = raw as? NSNumber {
            return Date(timeIntervalSince1970: value.doubleValue)
        }
        if let value = raw as? Double {
            return Date(timeIntervalSince1970: value)
        }
        if let value = raw as? Int {
            return Date(timeIntervalSince1970: Double(value))
        }
        return nil
    }

    private func latestFile(named fileName: String, under root: URL) -> URL? {
        recursiveFiles(at: root)
            .filter { $0.lastPathComponent == fileName }
            .max { lhs, rhs in
                modificationDate(for: lhs) < modificationDate(for: rhs)
            }
    }

    private func recursiveFiles(at root: URL) -> [URL] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path) else { return [] }
        guard let enumerator = fm.enumerator(
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

private extension Dictionary where Key == String, Value == String {
    func sortedKeyValues() -> [KeyValueRecord] {
        map { KeyValueRecord(key: $0.key, value: $0.value) }
            .sorted { $0.key < $1.key }
    }
}
