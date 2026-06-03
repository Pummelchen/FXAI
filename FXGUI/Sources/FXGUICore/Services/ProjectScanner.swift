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
        let pluginsRoot = projectRoot.appendingPathComponent("FXPlugins", isDirectory: true)
        var resultsByID: [String: PluginDescriptor] = [:]

        for sourceFile in pluginSourceFiles(at: pluginsRoot) {
            for descriptor in swiftPluginDescriptors(sourceFile: sourceFile, sourceRoot: pluginsRoot) {
                if let existing = resultsByID[descriptor.id] {
                    resultsByID[descriptor.id] = preferredPluginDescriptor(existing: existing, candidate: descriptor)
                } else {
                    resultsByID[descriptor.id] = descriptor
                }
            }
        }

        return Array(resultsByID.values)
    }

    private func pluginSourceFiles(at pluginsRoot: URL) -> [URL] {
        recursiveFiles(at: pluginsRoot)
            .filter { $0.pathExtension.lowercased() == "swift" }
            .filter { !isSupportOnlyPluginPath($0, pluginsRoot: pluginsRoot) }
    }

    private func isSupportOnlyPluginPath(_ sourceFile: URL, pluginsRoot: URL) -> Bool {
        let components = pluginRelativeComponents(for: sourceFile, sourceRoot: pluginsRoot)
        guard let first = components.first else {
            return true
        }

        if first == "Package.swift" || first == ".build" || first == "API" {
            return true
        }

        return false
    }

    private func preferredPluginDescriptor(existing: PluginDescriptor, candidate: PluginDescriptor) -> PluginDescriptor {
        if existing.sourceKind == candidate.sourceKind {
            return existing.sourcePath.path <= candidate.sourcePath.path ? existing : candidate
        }
        return candidate.sourceKind == .file ? candidate : existing
    }

    private func swiftPluginDescriptors(sourceFile: URL, sourceRoot: URL) -> [PluginDescriptor] {
        guard let source = try? String(contentsOf: sourceFile, encoding: .utf8) else {
            return []
        }

        var descriptors: [PluginDescriptor] = []
        let directNames = regexCaptures(pattern: #"aiName:\s*"([^"]+)""#, text: source)
        let sourceFamily = pluginFolderName(for: sourceFile, sourceRoot: sourceRoot)
        let manifestFamily = regexCaptures(pattern: #"family:\s*\.([A-Za-z0-9_]+)"#, text: source).first
        descriptors.append(contentsOf: directNames.map { name in
            let family = pluginFamily(pluginName: name)
                ?? manifestPluginFamily(manifestFamily)
                ?? sourceFamily
                ?? "Plugin"
            return PluginDescriptor(name: name, family: family, sourcePath: sourceFile, sourceKind: .file)
        })

        let descriptorPattern = #"(?m)^\s*(?:FXAIPluginImplementationDescriptor\.)?(linear|tree|sequence|distribution|statistical|factor|trend|mixture|memory|world|reinforcement)\(\.[^,]+,\s*"([^"]+)""#
        for match in regexCapturePairs(pattern: descriptorPattern, text: source) {
            let family = pluginFamily(pluginName: match.second)
                ?? descriptorPluginFamily(match.first)
            descriptors.append(
                PluginDescriptor(
                    name: match.second,
                    family: family,
                    sourcePath: sourceFile,
                    sourceKind: .file
                )
            )
        }

        return descriptors
    }

    private func pluginFolderName(for sourceFile: URL, sourceRoot: URL) -> String? {
        let folderName = pluginRelativeComponents(for: sourceFile, sourceRoot: sourceRoot).first
        return folderName == "API" ? nil : folderName
    }

    private func pluginRelativeComponents(for sourceFile: URL, sourceRoot: URL) -> [String] {
        let sourceRootPath = sourceRoot.standardizedFileURL.path
        let filePath = sourceFile.standardizedFileURL.path
        guard filePath.hasPrefix(sourceRootPath) else {
            return []
        }
        let relativePath = String(filePath.dropFirst(sourceRootPath.count))
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return relativePath.split(separator: "/").map(String.init)
    }

    private func descriptorPluginFamily(_ helperName: String) -> String {
        switch helperName {
        case "linear": return "Linear"
        case "tree": return "Tree"
        case "sequence": return "Sequence"
        case "distribution": return "Distribution"
        case "statistical": return "Stat"
        case "factor": return "Factor"
        case "trend": return "Trend"
        case "mixture": return "Mixture"
        case "memory": return "Memory"
        case "world": return "World"
        case "reinforcement": return "RL"
        default: return "Plugin"
        }
    }

    private func manifestPluginFamily(_ familyName: String?) -> String? {
        switch familyName {
        case "linear": return "Linear"
        case "tree": return "Tree"
        case "recurrent", "convolutional", "transformer", "stateSpace": return "Sequence"
        case "distributional": return "Distribution"
        case "mixture": return "Mixture"
        case "retrieval": return "Memory"
        case "worldModel": return "World"
        case "ruleBased": return "Rule"
        case "other": return "Stat"
        default: return nil
        }
    }

    private func pluginFamily(pluginName: String) -> String? {
        if pluginName.hasPrefix("ai_") { return "Sequence" }
        if pluginName.hasPrefix("dist_") { return "Distribution" }
        if pluginName.hasPrefix("factor_") { return "Factor" }
        if pluginName.hasPrefix("fxbacktest_") { return "Demo" }
        if pluginName.hasPrefix("lin_") { return "Linear" }
        if pluginName.hasPrefix("mem_") { return "Memory" }
        if pluginName.hasPrefix("mix_") { return "Mixture" }
        if pluginName.hasPrefix("rl_") { return "RL" }
        if pluginName.hasPrefix("rule_") { return "Rule" }
        if pluginName.hasPrefix("stat_") { return "Stat" }
        if pluginName.hasPrefix("tree_") { return "Tree" }
        if pluginName.hasPrefix("trend_") { return "Trend" }
        if pluginName.hasPrefix("wm_") { return "World" }
        return nil
    }

    private func regexCaptures(pattern: String, text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        let nsText = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: nsText.length)).compactMap { match in
            guard match.numberOfRanges > 1, match.range(at: 1).location != NSNotFound else {
                return nil
            }
            return nsText.substring(with: match.range(at: 1))
        }
    }

    private func regexCapturePairs(pattern: String, text: String) -> [(first: String, second: String)] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        let nsText = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: nsText.length)).compactMap { match in
            guard
                match.numberOfRanges > 2,
                match.range(at: 1).location != NSNotFound,
                match.range(at: 2).location != NSNotFound
            else {
                return nil
            }
            return (
                first: nsText.substring(with: match.range(at: 1)),
                second: nsText.substring(with: match.range(at: 2))
            )
        }
    }

    private func scanBuildTargets(projectRoot: URL) -> [BuildTargetStatus] {
        [
            buildTarget(named: "FXDataEngine Swift Package", relativePath: "FXDataEngine/Package.swift", projectRoot: projectRoot),
            buildTarget(named: "FXPlugins Swift Package", relativePath: "FXPlugins/Package.swift", projectRoot: projectRoot),
            buildTarget(named: "FXBacktest Swift Package", relativePath: "FXBacktest/Package.swift", projectRoot: projectRoot),
            buildTarget(named: "FXDatabase Swift Package", relativePath: "FXDatabase/Package.swift", projectRoot: projectRoot)
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
            ("Baselines", toolsURL(projectRoot: projectRoot, relativePath: "Baselines")),
            ("NewsPulse", toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/NewsPulse")),
            ("ResearchOS", toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/ResearchOS")),
            ("Profiles", toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/Profiles")),
            ("Distillation", toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/Distillation"))
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
        let researchRoot = toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/ResearchOS")
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
            at: toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/ResearchOS")
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
        let configuration = FXAIProjectConfigurationResolver.load(projectRoot: projectRoot)
        let environment = configuration.environment
        let dbURL: URL
        if let envDefaultDB = environment["FXAI_DEFAULT_DB"], !envDefaultDB.isEmpty {
            dbURL = FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: envDefaultDB,
                baseDirectory: projectRoot,
                environment: environment
            ) ?? URL(fileURLWithPath: envDefaultDB, isDirectory: false)
        } else if let configuredDefaultDB = FXAIProjectConfigurationResolver.configuredValue(configuration: configuration, key: "default_db") {
            dbURL = Self.configuredPathURL(
                rawValue: configuredDefaultDB,
                projectRoot: projectRoot,
                configuration: configuration
            ) ?? URL(fileURLWithPath: configuredDefaultDB, isDirectory: false)
        } else {
            dbURL = toolsURL(projectRoot: projectRoot, relativePath: "OfflineLab/fxai_offline_lab.turso.db")
        }
        let fm = FileManager.default

        return TursoSummary(
            localDatabasePresent: fm.fileExists(atPath: dbURL.path),
            localDatabasePath: fm.fileExists(atPath: dbURL.path) ? dbURL : nil,
            embeddedReplicaConfigured: environment["TURSO_DATABASE_URL"]?.isEmpty == false
                && environment["TURSO_AUTH_TOKEN"]?.isEmpty == false,
            encryptionConfigured: environment["TURSO_ENCRYPTION_KEY"]?.isEmpty == false
        )
    }

    private func toolsURL(projectRoot: URL, relativePath: String) -> URL {
        let candidates = [
            projectRoot
                .appendingPathComponent("FXDataEngine", isDirectory: true)
                .appendingPathComponent("Tools", isDirectory: true)
                .appendingPathComponent(relativePath, isDirectory: true),
            projectRoot
                .appendingPathComponent("Tools", isDirectory: true)
                .appendingPathComponent(relativePath, isDirectory: true),
            projectRoot
                .appendingPathComponent("FXAI", isDirectory: true)
                .appendingPathComponent("Tools", isDirectory: true)
                .appendingPathComponent(relativePath, isDirectory: true)
        ]
        if let existing = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return existing
        }
        return candidates[0]
    }

    private static func configuredPathURL(
        rawValue: String,
        projectRoot: URL,
        configuration: FXAIProjectConfigurationSnapshot
    ) -> URL? {
        var firstResolved: URL?
        for baseDirectory in [projectRoot, configuration.configDirectory] {
            if let resolved = FXAIProjectConfigurationResolver.resolvedPathURL(
                rawValue: rawValue,
                baseDirectory: baseDirectory,
                environment: configuration.environment
            ) {
                if firstResolved == nil {
                    firstResolved = resolved
                }
                if FileManager.default.fileExists(atPath: resolved.path) {
                    return resolved
                }
            }
        }
        return firstResolved
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
