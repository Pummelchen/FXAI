import Foundation

public enum ProjectScannerError: Error, LocalizedError {
    case invalidProjectRoot(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidProjectRoot(let url):
            return "The selected folder is not an FXAI project root: \(url.path)"
        }
    }
}

public struct ProjectScanner {
    public init() {}

    public func scan(projectRoot: URL) throws -> FXAIProjectSnapshot {
        guard ProjectPathResolver.isProjectRoot(projectRoot) else {
            throw ProjectScannerError.invalidProjectRoot(projectRoot)
        }

        let plugins = scanPlugins(projectRoot: projectRoot)
        let buildTargets = scanBuildTargets(projectRoot: projectRoot)
        let artifacts = scanArtifacts(projectRoot: projectRoot)
        let runtimeProfiles = scanRuntimeProfiles(projectRoot: projectRoot)
        let operatorSummary = scanOperatorSummary(projectRoot: projectRoot)
        let tursoSummary = scanTursoSummary(projectRoot: projectRoot)

        let pluginFamilies = Dictionary(grouping: plugins, by: \.family)
            .map { family, familyPlugins in
                PluginFamilySummary(id: family, family: family, pluginCount: familyPlugins.count)
            }
            .sorted { lhs, rhs in
                if lhs.pluginCount == rhs.pluginCount {
                    return lhs.family < rhs.family
                }
                return lhs.pluginCount > rhs.pluginCount
            }

        let reportCategories = Dictionary(grouping: artifacts, by: \.category)
            .map { category, categoryArtifacts in
                ReportCategorySummary(
                    id: category,
                    category: category,
                    fileCount: categoryArtifacts.count,
                    latestModifiedAt: categoryArtifacts.compactMap(\.modifiedAt).max()
                )
            }
            .sorted { $0.category < $1.category }

        let recentArtifacts = artifacts
            .sorted {
                let lhsDate = $0.modifiedAt ?? .distantPast
                let rhsDate = $1.modifiedAt ?? .distantPast
                if lhsDate == rhsDate {
                    return $0.name < $1.name
                }
                return lhsDate > rhsDate
            }
            .prefix(16)
            .map { $0 }

        return FXAIProjectSnapshot(
            projectRoot: projectRoot,
            generatedAt: Date(),
            buildTargets: buildTargets,
            pluginFamilies: pluginFamilies,
            plugins: plugins.sorted(by: pluginSorter),
            reportCategories: reportCategories,
            recentArtifacts: recentArtifacts,
            runtimeProfiles: runtimeProfiles.sorted { $0.symbol < $1.symbol },
            operatorSummary: operatorSummary,
            tursoSummary: tursoSummary
        )
    }

    private func scanPlugins(projectRoot: URL) -> [PluginDescriptor] {
        let pluginsRoot = projectRoot.appendingPathComponent("Plugins", isDirectory: true)
        let fm = FileManager.default

        guard let familyURLs = try? fm.contentsOfDirectory(
            at: pluginsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var results: [PluginDescriptor] = []

        for familyURL in familyURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard isDirectory(familyURL) else { continue }
            let family = familyURL.lastPathComponent

            guard let children = try? fm.contentsOfDirectory(
                at: familyURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }

            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if isDirectory(child) {
                    results.append(
                        PluginDescriptor(
                            name: child.lastPathComponent,
                            family: family,
                            sourcePath: child,
                            sourceKind: .folder
                        )
                    )
                    continue
                }

                let ext = child.pathExtension.lowercased()
                guard ext == "mqh" || ext == "mq5" else { continue }
                results.append(
                    PluginDescriptor(
                        name: child.deletingPathExtension().lastPathComponent,
                        family: family,
                        sourcePath: child,
                        sourceKind: .file
                    )
                )
            }
        }

        return Array(Set(results))
    }

    private func scanBuildTargets(projectRoot: URL) -> [BuildTargetStatus] {
        [
            buildTarget(named: "FXAI EA", relativePath: "FXAI.ex5", projectRoot: projectRoot),
            buildTarget(named: "Audit Runner", relativePath: "Tests/FXAI_AuditRunner.ex5", projectRoot: projectRoot),
            buildTarget(named: "Offline Export Runner", relativePath: "Tests/FXAI_OfflineExportRunner.ex5", projectRoot: projectRoot)
        ]
    }

    private func buildTarget(named: String, relativePath: String, projectRoot: URL) -> BuildTargetStatus {
        let url = projectRoot.appendingPathComponent(relativePath)
        let metadata = try? url.resourceValues(forKeys: [.contentModificationDateKey, .isRegularFileKey])

        return BuildTargetStatus(
            name: named,
            relativePath: relativePath,
            exists: metadata?.isRegularFile == true,
            modifiedAt: metadata?.contentModificationDate
        )
    }

    private func scanArtifacts(projectRoot: URL) -> [ReportArtifact] {
        let roots: [(String, URL)] = [
            ("Baselines", projectRoot.appendingPathComponent("Tools/Baselines", isDirectory: true)),
            ("ResearchOS", projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)),
            ("Profiles", projectRoot.appendingPathComponent("Tools/OfflineLab/Profiles", isDirectory: true)),
            ("Distillation", projectRoot.appendingPathComponent("Tools/OfflineLab/Distillation", isDirectory: true))
        ]

        var artifacts: [ReportArtifact] = []
        let allowedExtensions = Set(["json", "md", "tsv", "html", "set", "toml"])

        for (category, root) in roots {
            for fileURL in recursiveFiles(at: root) where allowedExtensions.contains(fileURL.pathExtension.lowercased()) {
                let metadata = try? fileURL.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey, .isRegularFileKey])
                guard metadata?.isRegularFile == true else { continue }

                artifacts.append(
                    ReportArtifact(
                        category: category,
                        name: fileURL.lastPathComponent,
                        path: fileURL,
                        modifiedAt: metadata?.contentModificationDate,
                        sizeBytes: Int64(metadata?.fileSize ?? 0)
                    )
                )
            }
        }

        return artifacts
    }

    private func scanRuntimeProfiles(projectRoot: URL) -> [RuntimeProfileSummary] {
        let researchRoot = projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        let fm = FileManager.default

        let candidateFiles = recursiveFiles(at: researchRoot).filter {
            $0.lastPathComponent.hasPrefix("live_deploy_") && $0.pathExtension.lowercased() == "json"
        }

        var profiles: [RuntimeProfileSummary] = []

        for fileURL in candidateFiles {
            guard
                let data = try? Data(contentsOf: fileURL),
                let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                continue
            }

            let symbol = (payload["symbol"] as? String) ?? symbolFrom(fileURL.lastPathComponent)
            let champions = payload["champions"] as? [[String: Any]]
            let firstChampion = champions?.first
            let pluginName = (firstChampion?["plugin_name"] as? String) ?? "unknown"
            let profileName = (payload["profile_name"] as? String) ?? fileURL.deletingLastPathComponent().lastPathComponent
            let promotionTier = (payload["promotion_tier"] as? String) ?? "unknown"
            let runtimeMode = (payload["runtime_mode"] as? String) ?? "unknown"

            profiles.append(
                RuntimeProfileSummary(
                    id: fileURL.path,
                    symbol: symbol,
                    pluginName: pluginName,
                    profileName: profileName,
                    promotionTier: promotionTier,
                    runtimeMode: runtimeMode,
                    sourcePath: fileURL
                )
            )
        }

        if profiles.isEmpty, fm.fileExists(atPath: researchRoot.path) {
            return []
        }

        return profiles
    }

    private func scanOperatorSummary(projectRoot: URL) -> OperatorSummary {
        let dashboardFiles = recursiveFiles(
            at: projectRoot.appendingPathComponent("Tools/OfflineLab/ResearchOS", isDirectory: true)
        ).filter {
            $0.lastPathComponent == "operator_dashboard.json"
        }

        let latestFile = dashboardFiles.max { lhs, rhs in
            modificationDate(for: lhs) < modificationDate(for: rhs)
        }

        guard
            let dashboardURL = latestFile,
            let data = try? Data(contentsOf: dashboardURL),
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return OperatorSummary(
                profileName: nil,
                championCount: 0,
                deploymentCount: 0,
                latestReviewedAt: nil
            )
        }

        let champions = payload["champions"] as? [[String: Any]] ?? []
        let deployments = payload["deployments"] as? [[String: Any]] ?? []
        let reviewedAt = champions
            .compactMap { ($0["reviewed_at"] as? NSNumber)?.doubleValue }
            .max()
            .map { Date(timeIntervalSince1970: $0) }

        return OperatorSummary(
            profileName: dashboardURL.deletingLastPathComponent().lastPathComponent,
            championCount: champions.count,
            deploymentCount: deployments.count,
            latestReviewedAt: reviewedAt
        )
    }

    private func scanTursoSummary(projectRoot: URL) -> TursoSummary {
        let environment = ProcessInfo.processInfo.environment
        let dbURL = projectRoot.appendingPathComponent("Tools/OfflineLab/fxai_offline_lab.turso.db")
        let fm = FileManager.default

        return TursoSummary(
            localDatabasePresent: fm.fileExists(atPath: dbURL.path),
            localDatabasePath: fm.fileExists(atPath: dbURL.path) ? dbURL : nil,
            embeddedReplicaConfigured: environment["TURSO_DATABASE_URL"]?.isEmpty == false
                && environment["TURSO_AUTH_TOKEN"]?.isEmpty == false,
            encryptionConfigured: environment["TURSO_ENCRYPTION_KEY"]?.isEmpty == false
        )
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

        var files: [URL] = []

        for case let fileURL as URL in enumerator {
            guard isRegularFile(fileURL) else { continue }
            files.append(fileURL)
        }

        return files
    }

    private func modificationDate(for url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }

    private func symbolFrom(_ filename: String) -> String {
        filename
            .replacingOccurrences(of: "live_deploy_", with: "")
            .replacingOccurrences(of: ".json", with: "")
    }

    private func pluginSorter(lhs: PluginDescriptor, rhs: PluginDescriptor) -> Bool {
        if lhs.family == rhs.family {
            return lhs.name < rhs.name
        }
        return lhs.family < rhs.family
    }
}
