import Foundation

public struct AdvancedVisualizationBuilder {
    public init() {}

    public func build(
        projectRoot: URL,
        runtimeSnapshot: RuntimeOperationsSnapshot?,
        researchSnapshot: ResearchOSControlSnapshot?
    ) -> AdvancedVisualizationSnapshot {
        let profileName = researchSnapshot?.profileName ?? runtimeSnapshot?.profileName
        let profileRoot = resolveProfileRoot(projectRoot: projectRoot, profileName: profileName)

        let deployments = runtimeSnapshot?.deployments ?? []
        let symbolDetails = deployments.compactMap { deployment in
            buildSymbolDetail(
                deployment: deployment,
                researchSymbol: researchSnapshot?.symbols.first(where: { $0.symbol == deployment.symbol }),
                profileRoot: profileRoot
            )
        }

        let familyStressHeatmap = buildFamilyStressHeatmap(symbolDetails: symbolDetails)
        let globalTimeline = buildGlobalTimeline(
            runtimeSnapshot: runtimeSnapshot,
            researchSnapshot: researchSnapshot,
            symbolDetails: symbolDetails,
            profileRoot: profileRoot
        )

        return AdvancedVisualizationSnapshot(
            generatedAt: Date(),
            profileName: profileName,
            familyStressHeatmap: familyStressHeatmap,
            symbolDetails: symbolDetails.sorted { $0.symbol < $1.symbol },
            globalTimeline: globalTimeline
        )
    }

    private func buildSymbolDetail(
        deployment: RuntimeDeploymentDetail,
        researchSymbol: ResearchOSSymbolControl?,
        profileRoot: URL?
    ) -> SymbolVisualizationDetail {
        let symbol = deployment.symbol
        let attribution = parseJSON(named: "attribution_\(symbol).json", under: profileRoot) ?? [:]
        let studentRouter = parseJSON(named: "student_router_\(symbol).json", under: profileRoot) ?? [:]
        let worldSimulator = parseJSON(named: "world_simulator_\(symbol).json", under: profileRoot) ?? [:]
        let supervisorService = parseJSON(named: "supervisor_service_\(symbol).json", under: profileRoot) ?? [:]
        let lineage = parseJSON(named: "lineage_\(symbol).json", under: profileRoot) ?? [:]

        let familyWeights = parseNamedWeights(attribution["family_weights"] as? [String: Any] ?? [:])
        let featureWeights = parseNamedWeights(attribution["feature_group_weights"] as? [String: Any] ?? [:])
        let pluginWeights = parseNamedWeights(studentRouter["plugin_weights"] as? [String: Any] ?? [:])
        let worldSessionScales = buildWorldSessionScales(worldSimulator)
        let worldStressMetrics = buildWorldStressMetrics(worldSimulator, supervisorService: supervisorService)
        let artifactDiffHeatmap = buildArtifactDiffHeatmap(for: deployment)
        let weakScenarios = parseStringArray(worldSimulator["weak_scenarios"])
        let timeline = buildSymbolTimeline(
            symbol: symbol,
            deployment: deployment,
            lineage: lineage,
            researchSymbol: researchSymbol,
            attribution: attribution
        )

        return SymbolVisualizationDetail(
            symbol: symbol,
            worldSessionScales: worldSessionScales,
            worldStressMetrics: worldStressMetrics,
            familyWeights: familyWeights,
            featureWeights: featureWeights,
            pluginWeights: pluginWeights,
            artifactDiffHeatmap: artifactDiffHeatmap,
            timeline: timeline.sorted { $0.date < $1.date },
            weakScenarios: weakScenarios
        )
    }

    private func buildFamilyStressHeatmap(symbolDetails: [SymbolVisualizationDetail]) -> VisualizationHeatmap? {
        var familyMaps: [(String, [String: Double])] = []

        for detail in symbolDetails {
            var weights: [String: Double] = [:]
            for point in detail.familyWeights {
                weights[point.label] = point.value
            }
            if weights.isEmpty {
                continue
            }
            familyMaps.append((detail.symbol, weights))
        }

        if familyMaps.isEmpty {
            return nil
        }

        let columns = Array(
            Set(familyMaps.flatMap { $0.1.keys })
        ).sorted()

        let rows = familyMaps.map(\.0)
        let values = familyMaps.map { _, map in
            columns.map { map[$0] ?? .nan }
        }

        return VisualizationHeatmap(
            title: "Family Stress Surface",
            subtitle: "Metal-backed matrix of promoted family weights by symbol. This becomes the dense routing surface when the plugin zoo expands.",
            rowLabels: rows,
            columnLabels: columns,
            values: values
        )
    }

    private func buildArtifactDiffHeatmap(for deployment: RuntimeDeploymentDetail) -> VisualizationHeatmap? {
        let sections: [(String, [KeyValueRecord])] = [
            ("Deployment", deployment.deploymentSections.flatMap(\.values)),
            ("Router", deployment.routerSections.flatMap(\.values)),
            ("Supervisor", deployment.supervisorSections.flatMap(\.values)),
            ("Command", deployment.commandSections.flatMap(\.values)),
            ("World", deployment.worldSections.flatMap(\.values)),
            ("Attribution", deployment.attributionSections.flatMap(\.values))
        ]

        let preferredKeys = [
            "policy_trade_floor",
            "policy_no_trade_cap",
            "student_signal_gain",
            "teacher_signal_gain",
            "supervisor_blend",
            "budget_multiplier",
            "entry_floor",
            "reduce_bias",
            "exit_bias",
            "sigma_scale",
            "spread_scale",
            "family_weight_recurrent",
            "feature_weight_price",
            "feature_weight_volatility"
        ]

        let numericMaps: [(String, [String: Double])] = sections.map { label, records in
            let pairs: [(String, Double)] = records.compactMap { record in
                guard let value = record.numericValue else { return nil }
                return (record.key, value)
            }
            return (label, Dictionary(uniqueKeysWithValues: pairs))
        }

        let availableKeys = Set(numericMaps.flatMap { $0.1.keys })
        var columns = preferredKeys.filter { availableKeys.contains($0) }
        if columns.count < 8 {
            let ranked: [String] = Dictionary(grouping: numericMaps.flatMap { $0.1.keys }, by: { $0 })
                .map { (key: $0.key, count: $0.value.count) }
                .sorted { lhs, rhs in
                    if lhs.count == rhs.count { return lhs.key < rhs.key }
                    return lhs.count > rhs.count
                }
                .map(\.key)
            for key in ranked where !columns.contains(key) {
                columns.append(key)
                if columns.count >= 10 { break }
            }
        }

        guard !columns.isEmpty else { return nil }

        let rowLabels = numericMaps.map { $0.0 }
        let values = numericMaps.map { _, map in
            columns.map { map[$0] ?? .nan }
        }

        return VisualizationHeatmap(
            title: "Artifact Diff Surface",
            subtitle: "Dense numeric comparison across deployment, router, supervisor, command, world, and attribution artifacts for one symbol.",
            rowLabels: rowLabels,
            columnLabels: columns,
            values: values
        )
    }

    private func buildWorldSessionScales(_ world: [String: Any]) -> [VisualizationSeriesPoint] {
        [
            VisualizationSeriesPoint(
                label: "Asia",
                value: doubleValue(world["asia_sigma_scale"]),
                secondaryValue: doubleValue(world["asia_spread_scale"])
            ),
            VisualizationSeriesPoint(
                label: "London",
                value: doubleValue(world["london_sigma_scale"]),
                secondaryValue: doubleValue(world["london_spread_scale"])
            ),
            VisualizationSeriesPoint(
                label: "New York",
                value: doubleValue(world["newyork_sigma_scale"]),
                secondaryValue: doubleValue(world["newyork_spread_scale"])
            )
        ]
    }

    private func buildWorldStressMetrics(
        _ world: [String: Any],
        supervisorService: [String: Any]
    ) -> [VisualizationSeriesPoint] {
        [
            VisualizationSeriesPoint(label: "Sigma", value: doubleValue(world["sigma_scale"])),
            VisualizationSeriesPoint(label: "Spread", value: doubleValue(world["spread_scale"])),
            VisualizationSeriesPoint(label: "Shock Decay", value: doubleValue(world["shock_decay"])),
            VisualizationSeriesPoint(label: "Liquidity", value: doubleValue(world["liquidity_stress"])),
            VisualizationSeriesPoint(label: "Transition Entropy", value: doubleValue(world["transition_entropy"])),
            VisualizationSeriesPoint(label: "Recovery Bias", value: doubleValue(world["recovery_bias"])),
            VisualizationSeriesPoint(label: "Supervisor Score", value: doubleValue(supervisorService["supervisor_score"])),
            VisualizationSeriesPoint(label: "Pressure Velocity", value: doubleValue(supervisorService["pressure_velocity"]))
        ]
    }

    private func buildSymbolTimeline(
        symbol: String,
        deployment: RuntimeDeploymentDetail,
        lineage: [String: Any],
        researchSymbol: ResearchOSSymbolControl?,
        attribution: [String: Any]
    ) -> [VisualizationTimelineEvent] {
        var events: [VisualizationTimelineEvent] = []

        if let createdAt = deployment.createdAt {
            events.append(
                VisualizationTimelineEvent(
                    category: "deployment",
                    title: "Deployment emitted",
                    detail: "\(symbol) deployment profile became available",
                    date: createdAt
                )
            )
        }

        if let reviewedAt = deployment.reviewedAt {
            events.append(
                VisualizationTimelineEvent(
                    category: "promotion",
                    title: "Champion reviewed",
                    detail: deployment.pluginName,
                    date: reviewedAt
                )
            )
        }

        if let deployments = lineage["deployments"] as? [[String: Any]],
           let first = deployments.first,
           let champions = first["champions"] as? [[String: Any]] {
            for champion in champions {
                if let promotedAt = parseUnix(champion["promoted_at"]) {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "promotion",
                            title: "Champion promoted",
                            detail: stringValue(champion["plugin_name"]) ?? deployment.pluginName,
                            date: promotedAt,
                            score: doubleOptional(champion["champion_score"])
                        )
                    )
                }
                if let reviewedAt = parseUnix(champion["reviewed_at"]) {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "promotion",
                            title: "Promotion reviewed",
                            detail: stringValue(champion["status"]) ?? "champion",
                            date: reviewedAt,
                            score: doubleOptional(champion["portfolio_score"])
                        )
                    )
                }
            }
        }

        if let deploymentDate = deployment.createdAt {
            let featureWeights = attribution["feature_group_weights"] as? [String: Any] ?? [:]
            for (key, value) in featureWeights {
                events.append(
                    VisualizationTimelineEvent(
                        category: "attribution",
                        title: "Feature weight set",
                        detail: "\(key)=\(String(format: "%.3f", doubleValue(value)))",
                        date: deploymentDate,
                        score: doubleOptional(value)
                    )
                )
            }
        }

        if let researchSymbol {
            for neighbor in researchSymbol.analogNeighbors.prefix(3) {
                if let date = researchSymbol.deploymentCreatedAt {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "analog",
                            title: "Analog neighbor attached",
                            detail: neighbor.pluginName,
                            date: date,
                            score: neighbor.similarity
                        )
                    )
                }
            }
        }

        return events
    }

    private func buildGlobalTimeline(
        runtimeSnapshot: RuntimeOperationsSnapshot?,
        researchSnapshot: ResearchOSControlSnapshot?,
        symbolDetails: [SymbolVisualizationDetail],
        profileRoot: URL?
    ) -> [VisualizationTimelineEvent] {
        var events: [VisualizationTimelineEvent] = []

        if let runtimeSnapshot {
            for champion in runtimeSnapshot.champions {
                if let reviewedAt = champion.reviewedAt {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "promotion",
                            title: "\(champion.symbol) champion",
                            detail: champion.pluginName,
                            date: reviewedAt,
                            score: champion.championScore
                        )
                    )
                }
            }
        }

        if let researchSnapshot {
            for branch in researchSnapshot.branches {
                if let createdAt = branch.createdAt {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "branch",
                            title: branch.name,
                            detail: branch.branchKind,
                            date: createdAt
                        )
                    )
                }
            }
            for audit in researchSnapshot.auditEvents {
                if let observedAt = audit.observedAt ?? audit.occurredAt {
                    events.append(
                        VisualizationTimelineEvent(
                            category: "audit",
                            title: audit.eventType,
                            detail: audit.targetName,
                            date: observedAt
                        )
                    )
                }
            }
        }

        for detail in symbolDetails {
            events.append(contentsOf: detail.timeline.prefix(8))
        }

        return events.sorted { lhs, rhs in
            if lhs.date == rhs.date {
                return lhs.title < rhs.title
            }
            return lhs.date < rhs.date
        }
    }

    private func resolveProfileRoot(projectRoot: URL, profileName: String?) -> URL? {
        let researchRoot = projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        if let profileName, !profileName.isEmpty {
            let candidate = researchRoot.appendingPathComponent(profileName, isDirectory: true)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }

        let enumerator = FileManager.default.enumerator(
            at: researchRoot,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )

        var latest: (URL, Date)?
        while let file = enumerator?.nextObject() as? URL {
            guard file.lastPathComponent == "operator_dashboard.json" else { continue }
            let modifiedAt = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
            let dir = file.deletingLastPathComponent()
            if latest == nil || modifiedAt > latest?.1 ?? .distantPast {
                latest = (dir, modifiedAt)
            }
        }
        return latest?.0
    }

    private func parseJSON(named fileName: String, under root: URL?) -> [String: Any]? {
        guard let root else { return nil }
        let path = root.appendingPathComponent(fileName)
        guard
            let data = try? Data(contentsOf: path),
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return raw
    }

    private func parseNamedWeights(_ raw: [String: Any]) -> [VisualizationSeriesPoint] {
        raw.map { key, value in
            VisualizationSeriesPoint(label: key, value: doubleValue(value))
        }
        .sorted { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.label < rhs.label
            }
            return lhs.value > rhs.value
        }
    }

    private func parseStringArray(_ raw: Any?) -> [String] {
        guard let values = raw as? [String] else { return [] }
        return values
    }

    private func parseUnix(_ raw: Any?) -> Date? {
        if let double = raw as? Double { return Date(timeIntervalSince1970: double) }
        if let int = raw as? Int { return Date(timeIntervalSince1970: Double(int)) }
        if let number = raw as? NSNumber { return Date(timeIntervalSince1970: number.doubleValue) }
        if let string = raw as? String, let double = Double(string) {
            return Date(timeIntervalSince1970: double)
        }
        return nil
    }

    private func stringValue(_ raw: Any?) -> String? {
        if let value = raw as? String { return value }
        if let value = raw as? NSNumber { return value.stringValue }
        return nil
    }

    private func doubleValue(_ raw: Any?) -> Double {
        if let value = raw as? Double { return value }
        if let value = raw as? NSNumber { return value.doubleValue }
        if let value = raw as? String, let parsed = Double(value) { return parsed }
        return 0
    }

    private func doubleOptional(_ raw: Any?) -> Double? {
        if let value = raw as? Double { return value }
        if let value = raw as? NSNumber { return value.doubleValue }
        if let value = raw as? String, let parsed = Double(value) { return parsed }
        return nil
    }
}
