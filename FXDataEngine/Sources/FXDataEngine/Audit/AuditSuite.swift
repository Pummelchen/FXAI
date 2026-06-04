import Foundation

public struct AuditSuiteConfiguration: Codable, Hashable, Sendable {
    public var bars: Int
    public var scenarioList: String
    public var seed: UInt64
    public var runner: AuditRunnerConfiguration
    public var stopOnFailure: Bool

    public init(
        bars: Int = 20_000,
        scenarioList: String = AuditSuiteTools.defaultScenarioList,
        seed: UInt64 = 42,
        runner: AuditRunnerConfiguration = AuditRunnerConfiguration(horizonMinutes: 5, priceCostPoints: 2.0),
        stopOnFailure: Bool = false
    ) {
        self.bars = min(max(2_048, bars), 10_000_000)
        self.scenarioList = scenarioList
        self.seed = seed
        self.runner = runner
        self.stopOnFailure = stopOnFailure
    }
}

public struct AuditPluginFactory: Sendable {
    public let manifest: PluginManifestV4
    private let runScenarioBody: @Sendable (
        _ generated: AuditGeneratedScenarioSeries,
        _ spec: AuditScenarioSpec,
        _ configuration: AuditRunnerConfiguration,
        _ hyperParameters: HyperParameters
    ) throws -> AuditScenarioMetrics
    private let generateAdversarialBody: @Sendable (
        _ marketSeries: M1OHLCVSeries,
        _ configuration: AuditAdversarialConfiguration,
        _ hyperParameters: HyperParameters
    ) throws -> AuditGeneratedScenarioSeries?

    public init<Plugin: FXAIPluginV4>(
        manifest: PluginManifestV4,
        makePlugin: @escaping @Sendable () -> Plugin
    ) {
        self.manifest = manifest
        self.runScenarioBody = { generated, spec, configuration, hyperParameters in
            var plugin = makePlugin()
            return try AuditRunnerTools.runScenario(
                plugin: &plugin,
                generated: generated,
                spec: spec,
                configuration: configuration,
                hyperParameters: hyperParameters
            )
        }
        self.generateAdversarialBody = { marketSeries, configuration, hyperParameters in
            var plugin = makePlugin()
            return try AuditAdversarialTools.generateScenario(
                plugin: &plugin,
                marketSeries: marketSeries,
                configuration: configuration,
                hyperParameters: hyperParameters
            )
        }
    }

    public init<Plugin: FXAIPluginV4>(
        makePlugin: @escaping @Sendable () -> Plugin
    ) {
        let plugin = makePlugin()
        self.init(manifest: plugin.manifest, makePlugin: makePlugin)
    }

    public func runScenario(
        generated: AuditGeneratedScenarioSeries,
        spec: AuditScenarioSpec,
        configuration: AuditRunnerConfiguration,
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> AuditScenarioMetrics {
        try runScenarioBody(generated, spec, configuration, hyperParameters)
    }

    public func generateAdversarialScenario(
        marketSeries: M1OHLCVSeries,
        configuration: AuditAdversarialConfiguration,
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> AuditGeneratedScenarioSeries? {
        try generateAdversarialBody(marketSeries, configuration, hyperParameters)
    }

    public func matches(token rawToken: String) -> Bool {
        let token = AuditSuiteTools.normalizedToken(rawToken)
        guard !token.isEmpty else { return false }
        if token == "all" { return true }
        if let numericID = Int(token), numericID == manifest.aiID { return true }
        return AuditSuiteTools.normalizedToken(manifest.aiName) == token
    }
}

public struct AuditSuiteRunResult: Codable, Hashable, Sendable {
    public var metrics: [AuditScenarioMetrics]
    public var attemptedRuns: Int
    public var failedRuns: Int
    public var skippedRuns: Int
    public var reportDocument: String

    public init(
        metrics: [AuditScenarioMetrics] = [],
        attemptedRuns: Int = 0,
        failedRuns: Int = 0,
        skippedRuns: Int = 0,
        reportDocument: String = AuditReportTools.document([])
    ) {
        self.metrics = metrics
        self.attemptedRuns = attemptedRuns
        self.failedRuns = failedRuns
        self.skippedRuns = skippedRuns
        self.reportDocument = reportDocument
    }
}

public enum AuditSuiteTools {
    public static let defaultScenarioList =
        "{random_walk, drift_up, drift_down, mean_revert, vol_cluster, monotonic_up, monotonic_down, regime_shift, market_recent, market_trend, market_chop, market_session_edges, market_liquidity_shock, market_walkforward, market_macro_event, market_adversarial}"
    public static let fallbackScenarioIDs = [0, 1, 2, 4]

    public static func normalizedToken(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    public static func scenarioID(fromName rawName: String) -> Int? {
        switch normalizedToken(rawName) {
        case "random_walk":
            return 0
        case "drift_up":
            return 1
        case "drift_down":
            return 2
        case "mean_revert":
            return 3
        case "vol_cluster":
            return 4
        case "monotonic_up":
            return 5
        case "monotonic_down":
            return 6
        case "regime_shift":
            return 7
        case "market_recent":
            return 8
        case "market_trend":
            return 9
        case "market_chop":
            return 10
        case "market_session_edges":
            return 11
        case "market_liquidity_shock", "market_spread_shock":
            return 12
        case "market_walkforward":
            return 13
        case "market_macro_event":
            return 14
        case "market_adversarial":
            return 15
        default:
            return nil
        }
    }

    public static func parseScenarioList(_ raw: String) -> [Int] {
        let cleaned = raw
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: ";", with: ",")
            .replacingOccurrences(of: "|", with: ",")
        var output: [Int] = []
        for part in cleaned.split(separator: ",", omittingEmptySubsequences: false) {
            guard let id = scenarioID(fromName: String(part)), !output.contains(id) else { continue }
            output.append(id)
        }
        return output.isEmpty ? fallbackScenarioIDs : output
    }

    public static func parsePluginTokens(_ raw: String) -> [String] {
        raw
            .replacingOccurrences(of: "{", with: "")
            .replacingOccurrences(of: "}", with: "")
            .replacingOccurrences(of: ";", with: ",")
            .replacingOccurrences(of: "|", with: ",")
            .split(separator: ",", omittingEmptySubsequences: false)
            .map { normalizedToken(String($0)) }
            .filter { !$0.isEmpty }
    }

    public static func resolvePlugins(
        _ plugins: [AuditPluginFactory],
        allPlugins: Bool,
        pluginList: String,
        fallbackAIID: Int? = nil
    ) -> [AuditPluginFactory] {
        if allPlugins { return plugins }
        let tokens = parsePluginTokens(pluginList)
        let selected = plugins.filter { plugin in
            tokens.contains { plugin.matches(token: $0) }
        }
        if !selected.isEmpty { return selected }
        guard let fallbackAIID,
              let fallback = plugins.first(where: { $0.manifest.aiID == fallbackAIID }) else {
            return []
        }
        return [fallback]
    }

    public static func runSuite(
        plugins: [AuditPluginFactory],
        configuration: AuditSuiteConfiguration,
        marketSeries: M1OHLCVSeries? = nil,
        worldPlanTSV: String? = nil,
        hyperParameters: HyperParameters = HyperParameters()
    ) throws -> AuditSuiteRunResult {
        let scenarioIDs = parseScenarioList(configuration.scenarioList)
        var result = AuditSuiteRunResult()
        for plugin in plugins {
            for scenarioID in scenarioIDs {
                result.attemptedRuns += 1
                let spec = AuditScenarioTools.scenarioSpec(
                    scenarioID: scenarioID,
                    worldPlanTSV: worldPlanTSV
                )
                let generated = try generatedScenario(
                    plugin: plugin,
                    spec: spec,
                    suiteConfiguration: configuration,
                    seed: scenarioSeed(base: configuration.seed, aiID: plugin.manifest.aiID),
                    marketSeries: marketSeries,
                    hyperParameters: hyperParameters
                )
                guard let generated else {
                    result.skippedRuns += 1
                    if configuration.stopOnFailure {
                        result.failedRuns += 1
                        result.reportDocument = AuditReportTools.document(result.metrics)
                        return result
                    }
                    continue
                }

                do {
                    let metrics = try plugin.runScenario(
                        generated: generated,
                        spec: spec,
                        configuration: configuration.runner,
                        hyperParameters: hyperParameters
                    )
                    result.metrics.append(metrics)
                    if !metrics.issueFlags.isEmpty {
                        result.failedRuns += 1
                    }
                } catch {
                    result.failedRuns += 1
                    if configuration.stopOnFailure {
                        result.reportDocument = AuditReportTools.document(result.metrics)
                        return result
                    }
                }
            }
        }
        result.reportDocument = AuditReportTools.document(result.metrics)
        return result
    }

    public static func writeReport(
        metrics: [AuditScenarioMetrics],
        to reportURL: URL,
        resetOutput: Bool,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(
            at: reportURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try writeWalkForwardDiagnostics(
            metrics: metrics,
            reportURL: reportURL,
            resetOutput: resetOutput,
            fileManager: fileManager
        )
        let existingSize = try fileManager.fileSize(at: reportURL)
        if resetOutput || existingSize == nil || existingSize == 0 {
            try AuditReportTools.document(metrics).write(to: reportURL, atomically: true, encoding: .utf8)
            return
        }
        let rows = metrics.map(AuditReportTools.rowLine(for:)).joined(separator: "\r\n")
        guard !rows.isEmpty else { return }
        let suffix = rows + "\r\n"
        let handle = try FileHandle(forWritingTo: reportURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(suffix.utf8))
    }

    public static func walkForwardDiagnosticsURL(for reportURL: URL) -> URL {
        reportURL.deletingPathExtension().appendingPathExtension("walkforward.json")
    }

    private static func writeWalkForwardDiagnostics(
        metrics: [AuditScenarioMetrics],
        reportURL: URL,
        resetOutput: Bool,
        fileManager: FileManager
    ) throws {
        let diagnostics = AuditReportTools.walkForwardDiagnostics(metrics)
        let diagnosticsURL = walkForwardDiagnosticsURL(for: reportURL)
        guard !diagnostics.plugins.isEmpty else {
            if resetOutput, fileManager.fileExists(atPath: diagnosticsURL.path) {
                try fileManager.removeItem(at: diagnosticsURL)
            }
            return
        }
        let json = try AuditReportTools.walkForwardDiagnosticsJSON(metrics)
        try json.write(to: diagnosticsURL, atomically: true, encoding: .utf8)
    }

    private static func scenarioSeed(base: UInt64, aiID: Int) -> UInt64 {
        let clampedAIID = UInt64(max(aiID, 0))
        return base &+ ((clampedAIID &+ 1) &* 1_315_423_911)
    }

    private static func generatedScenario(
        plugin: AuditPluginFactory,
        spec: AuditScenarioSpec,
        suiteConfiguration: AuditSuiteConfiguration,
        seed: UInt64,
        marketSeries: M1OHLCVSeries?,
        hyperParameters: HyperParameters
    ) throws -> AuditGeneratedScenarioSeries? {
        let point = suiteConfiguration.runner.pointValue
        if spec.id == 15 {
            guard let marketSeries else { return nil }
            return try plugin.generateAdversarialScenario(
                marketSeries: marketSeries,
                configuration: AuditAdversarialConfiguration(
                    bars: suiteConfiguration.bars,
                    horizonMinutes: suiteConfiguration.runner.horizonMinutes,
                    pointValue: point,
                    priceCostPoints: suiteConfiguration.runner.priceCostPoints,
                    evThresholdPoints: suiteConfiguration.runner.evThresholdPoints,
                    normalizationMethod: suiteConfiguration.runner.normalizationMethod,
                    maxEvaluationSamples: suiteConfiguration.runner.maxSamples,
                    minEvaluationSamples: slimMinEvaluationSamples(for: suiteConfiguration)
                ),
                hyperParameters: hyperParameters
            )
        }
        if spec.id < 8 {
            return AuditScenarioTools.generateSyntheticScenarioSeries(
                spec: spec,
                bars: suiteConfiguration.bars,
                seed: seed,
                point: point
            )
        }
        guard let marketSeries else { return nil }
        return AuditScenarioTools.generateMarketScenarioSeries(
            spec: spec,
            marketSeries: marketSeries,
            bars: suiteConfiguration.bars,
            point: point
        )
    }

    private static func slimMinEvaluationSamples(for configuration: AuditSuiteConfiguration) -> Int? {
        guard configuration.runner.maxSamples != Int.max else { return nil }
        return max(1, min(configuration.runner.maxSamples, max(96, configuration.bars / 6)))
    }
}

private extension FileManager {
    func fileSize(at url: URL) throws -> Int64? {
        guard fileExists(atPath: url.path) else { return nil }
        let attributes = try attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value
    }
}
