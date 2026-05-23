import Foundation
import FXBacktestAPI

#if canImport(Metal)
import Metal
#endif

public struct FXDatabaseConnectivityAgent: Sendable {
    public typealias StatusLoader = @Sendable (FXDatabaseConnectionSettings) async throws -> FXBacktestAPIStatusResponse

    public static let descriptor = FXBacktestAgentDescriptor(
        id: .fxDatabaseConnectivity,
        displayName: "FXDatabase Connectivity",
        responsibility: "Verify FXDatabase API v1 availability before FXBacktest pulls Forex history."
    )

    private let statusLoader: StatusLoader

    public init(statusLoader: @escaping StatusLoader = Self.defaultStatusLoader) {
        self.statusLoader = statusLoader
    }

    public func check(connection: FXDatabaseConnectionSettings) async throws -> FXBacktestAgentOutcome {
        let started = Date()
        do {
            let response = try await statusLoader(connection)
            guard response.apiVersion == FXBacktestAPIV1.version else {
                return Self.descriptor.outcome(
                    status: .failed,
                    message: "FXDatabase API version mismatch: got \(response.apiVersion), expected \(FXBacktestAPIV1.version).",
                    details: ["service=\(response.service)", "status=\(response.status)"],
                    startedAtUtc: started
                )
            }
            guard response.status.lowercased() == "ok" else {
                return Self.descriptor.outcome(
                    status: .failed,
                    message: "FXDatabase reported non-ok status '\(response.status)'.",
                    details: ["service=\(response.service)", "api_version=\(response.apiVersion)"],
                    startedAtUtc: started
                )
            }
            return Self.descriptor.outcome(
                status: .ok,
                message: "FXDatabase API v1 is reachable at \(connection.apiBaseURL.absoluteString).",
                details: ["service=\(response.service)", "api_version=\(response.apiVersion)"],
                startedAtUtc: started
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return Self.descriptor.outcome(
                status: .failed,
                message: "FXDatabase API v1 connectivity failed: \(error).",
                details: ["url=\(connection.apiBaseURL.absoluteString)"],
                startedAtUtc: started
            )
        }
    }

    public static func defaultStatusLoader(connection: FXDatabaseConnectionSettings) async throws -> FXBacktestAPIStatusResponse {
        try await FXBacktestAPIClient(
            baseURL: connection.apiBaseURL,
            requestTimeoutSeconds: connection.requestTimeoutSeconds
        )
        .status()
    }
}

public struct MarketReadinessAgent: Sendable {
    public static let descriptor = FXBacktestAgentDescriptor(
        id: .marketReadiness,
        displayName: "Market Readiness",
        responsibility: "Validate loaded M1 OHLC universe integrity before any optimization run."
    )

    public init() {}

    public func evaluate(universe: OhlcMarketUniverse) -> FXBacktestAgentOutcome {
        let started = Date()
        do {
            let series = try validate(universe: universe)
            let brokerIds = Set(series.map(\.metadata.brokerSourceId))
            let source = brokerIds == ["demo"] ? "demo" : brokerIds.sorted().joined(separator: ",")
            return Self.descriptor.outcome(
                status: .ok,
                message: "Market universe ready: \(universe.symbols.joined(separator: ",")), \(universe.count.formatted()) aligned M1 bars.",
                details: ["primary=\(universe.primarySymbol)", "broker_source=\(source)"],
                startedAtUtc: started
            )
        } catch {
            return Self.descriptor.outcome(
                status: .failed,
                message: String(describing: error),
                startedAtUtc: started
            )
        }
    }

    @discardableResult
    private func validate(universe: OhlcMarketUniverse) throws -> [OhlcDataSeries] {
        guard !universe.symbols.isEmpty, !universe.seriesBySymbol.isEmpty else {
            throw FXBacktestError.invalidMarketData("Market universe is empty.")
        }
        guard universe.seriesBySymbol[universe.primarySymbol] != nil else {
            throw FXBacktestError.invalidMarketData("Primary symbol \(universe.primarySymbol) is missing.")
        }

        var series: [OhlcDataSeries] = []
        series.reserveCapacity(universe.symbols.count)
        var brokerIds = Set<String>()
        var hasDemo = false
        var hasFXDatabase = false

        for symbol in universe.symbols {
            guard let item = universe[symbol] else {
                throw FXBacktestError.invalidMarketData("Missing market series for \(symbol).")
            }
            guard !item.isEmpty else {
                throw FXBacktestError.invalidMarketData("\(symbol) has no M1 bars.")
            }
            guard item.metadata.logicalSymbol.uppercased() == symbol else {
                throw FXBacktestError.invalidMarketData("\(symbol) metadata logical symbol is \(item.metadata.logicalSymbol).")
            }
            guard item.metadata.timeframe.uppercased() == "M1" else {
                throw FXBacktestError.invalidMarketData("\(symbol) timeframe must be M1, got \(item.metadata.timeframe).")
            }
            guard !item.metadata.brokerSourceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FXBacktestError.invalidMarketData("\(symbol) broker source id is empty.")
            }
            if item.metadata.brokerSourceId == "demo" {
                hasDemo = true
            } else {
                hasFXDatabase = true
                let mt5Symbol = item.metadata.mt5Symbol?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !mt5Symbol.isEmpty else {
                    throw FXBacktestError.invalidMarketData("\(symbol) is missing FXDatabase MT5 symbol metadata.")
                }
            }
            if let firstUtc = item.metadata.firstUtc, firstUtc != item.utcTimestamps.first {
                throw FXBacktestError.invalidMarketData("\(symbol) metadata firstUtc does not match the first bar.")
            }
            if let lastUtc = item.metadata.lastUtc, lastUtc != item.utcTimestamps.last {
                throw FXBacktestError.invalidMarketData("\(symbol) metadata lastUtc does not match the last bar.")
            }
            brokerIds.insert(item.metadata.brokerSourceId)
            series.append(item)
        }

        guard !(hasDemo && hasFXDatabase) else {
            throw FXBacktestError.invalidMarketData("Cannot mix demo and FXDatabase market data in one backtest run.")
        }
        if hasFXDatabase, brokerIds.count != 1 {
            throw FXBacktestError.invalidMarketData("All FXDatabase symbols in one run must use the same broker source.")
        }

        _ = try OhlcMarketUniverse(primarySymbol: universe.primarySymbol, series: series, requireAlignedTimestamps: true)
        return series
    }
}

public struct OptimizationRunCoordinatorAgent: Sendable {
    public static let descriptor = FXBacktestAgentDescriptor(
        id: .optimizationRunCoordinator,
        displayName: "Optimization Run Coordinator",
        responsibility: "Validate run configuration and produce immutable whole-pass optimizer settings."
    )

    public init() {}

    public func prepare(
        plugin: AnyFXBacktestPlugin,
        marketUniverse: OhlcMarketUniverse,
        sweep: ParameterSweep,
        target: BacktestExecutionTarget,
        maxWorkers: Int,
        chunkSize: Int,
        initialDeposit: Double,
        contractSize: Double,
        lotSize: Double
    ) throws -> (settings: BacktestRunSettings, outcome: FXBacktestAgentOutcome) {
        let started = Date()
        let pluginOutcome = PluginValidationAgent().validate(plugin: plugin)
        guard !pluginOutcome.isBlockingFailure else {
            throw FXBacktestError.invalidParameter(pluginOutcome.message)
        }
        try validateTarget(plugin: plugin, target: target)
        guard sweep.combinationCount > 0 else {
            throw FXBacktestError.invalidSweep("Optimization sweep has no passes.")
        }
        guard maxWorkers > 0 else {
            throw FXBacktestError.invalidParameter("Workers must be > 0.")
        }
        guard chunkSize > 0 else {
            throw FXBacktestError.invalidParameter("Chunk size must be > 0.")
        }
        guard initialDeposit.isFinite, initialDeposit > 0 else {
            throw FXBacktestError.invalidParameter("Initial deposit must be a finite value > 0.")
        }
        guard contractSize.isFinite, contractSize > 0 else {
            throw FXBacktestError.invalidParameter("Contract size must be a finite value > 0.")
        }
        guard lotSize.isFinite, lotSize > 0 else {
            throw FXBacktestError.invalidParameter("Lot size must be a finite value > 0.")
        }

        let settings = BacktestRunSettings(
            target: target,
            maxWorkers: maxWorkers,
            chunkSize: chunkSize,
            initialDeposit: initialDeposit,
            contractSize: contractSize,
            lotSize: lotSize
        )
        return (
            settings,
            Self.descriptor.outcome(
                status: .ok,
                message: "Run plan ready: \(sweep.combinationCount.formatted()) passes on \(target.displayName).",
                details: [
                    "workers=\(maxWorkers)",
                    "chunk_size=\(chunkSize)",
                    "symbols=\(marketUniverse.symbols.joined(separator: ","))"
                ],
                startedAtUtc: started
            )
        )
    }

    private func validateTarget(plugin: AnyFXBacktestPlugin, target: BacktestExecutionTarget) throws {
        switch target {
        case .cpu:
            guard plugin.descriptor.supportsCPU else {
                throw FXBacktestError.invalidParameter("\(plugin.descriptor.displayName) does not declare CPU support.")
            }
        case .metal:
            guard plugin.descriptor.supportsMetal, plugin.metalKernel != nil else {
                throw FXBacktestError.metalKernelMissing(plugin: plugin.descriptor.displayName)
            }
        case .both:
            guard plugin.descriptor.supportsCPU else {
                throw FXBacktestError.invalidParameter("\(plugin.descriptor.displayName) does not declare CPU support.")
            }
            guard plugin.descriptor.supportsMetal, plugin.metalKernel != nil else {
                throw FXBacktestError.metalKernelMissing(plugin: plugin.descriptor.displayName)
            }
        }
    }
}

public actor ResultPersistenceAgent {
    public static let descriptor = FXBacktestAgentDescriptor(
        id: .resultPersistence,
        displayName: "Result Persistence",
        responsibility: "Own FXDatabase result-run lifecycle, buffered pass writes, completion updates, snapshots, and purge operations."
    )

    private let store: any BacktestResultStore
    private let runID: String
    private let batchSize: Int
    private let startedAtUtc: Date
    private var buffer: [BacktestPassResult] = []
    private var writtenResults = 0
    private var lastOutcome: FXBacktestAgentOutcome

    private init(store: any BacktestResultStore, run: BacktestStoredRun, batchSize: Int) {
        self.store = store
        self.runID = run.runID
        self.batchSize = max(1, batchSize)
        self.startedAtUtc = Date()
        self.lastOutcome = Self.descriptor.outcome(
            status: .ok,
            message: "FXDatabase result persistence started for run \(run.runID).",
            details: ["plugin=\(run.pluginIdentifier)", "engine=\(run.engine.rawValue)"],
            startedAtUtc: startedAtUtc
        )
    }

    public static func start(
        store: any BacktestResultStore,
        run: BacktestStoredRun,
        batchSize: Int = 500
    ) async throws -> ResultPersistenceAgent {
        try await store.startRun(run)
        return ResultPersistenceAgent(store: store, run: run, batchSize: batchSize)
    }

    public func append(_ result: BacktestPassResult) async throws {
        buffer.append(result)
        if buffer.count >= batchSize {
            try await flush()
        }
    }

    public func finish(progress: BacktestProgress, status: String) async throws -> FXBacktestAgentOutcome {
        try await flush()
        try await store.completeRun(runID: runID, progress: progress, status: status)
        lastOutcome = Self.descriptor.outcome(
            status: .ok,
            message: "FXDatabase result persistence finalized run \(runID) with status \(status).",
            details: ["written_results=\(writtenResults)", "completed_passes=\(progress.completedPasses)"],
            startedAtUtc: startedAtUtc
        )
        return lastOutcome
    }

    public func latestOutcome() -> FXBacktestAgentOutcome {
        lastOutcome
    }

    private func flush() async throws {
        guard !buffer.isEmpty else { return }
        let batch = buffer
        try await store.appendResults(batch, runID: runID)
        writtenResults += batch.count
        buffer.removeAll(keepingCapacity: true)
    }

    public static func saveSnapshot(
        store: any BacktestResultStore,
        run: BacktestStoredRun,
        results: [BacktestPassResult],
        progress: BacktestProgress,
        status: String
    ) async throws -> FXBacktestAgentOutcome {
        let started = Date()
        try await store.startRun(run)
        try await store.appendResults(results, runID: run.runID)
        try await store.completeRun(runID: run.runID, progress: progress, status: status)
        return descriptor.outcome(
            status: .ok,
            message: "Saved \(results.count.formatted()) held pass results to FXDatabase run \(run.runID).",
            details: ["status=\(status)", "plugin=\(run.pluginIdentifier)"],
            startedAtUtc: started
        )
    }

    public static func purgeAll(store: any BacktestResultStore) async throws -> (report: BacktestResultPurgeReport, outcome: FXBacktestAgentOutcome) {
        let started = Date()
        try await store.ensureSchema()
        let report = try await store.purgeAll()
        return (
            report,
            descriptor.outcome(
                status: .ok,
                message: "Cleaned all FXDatabase backtest result data.",
                details: ["scope=\(report.scope)", "sql_statements=\(report.sqlStatements)"],
                startedAtUtc: started
            )
        )
    }

    public static func purge(
        store: any BacktestResultStore,
        olderThanDays days: Int
    ) async throws -> (report: BacktestResultPurgeReport, outcome: FXBacktestAgentOutcome) {
        let started = Date()
        try await store.ensureSchema()
        let report = try await store.purge(olderThanDays: days)
        return (
            report,
            descriptor.outcome(
                status: .ok,
                message: "Cleaned FXDatabase backtest result data older than \(days) days.",
                details: ["scope=\(report.scope)", "sql_statements=\(report.sqlStatements)"],
                startedAtUtc: started
            )
        )
    }
}

public struct PluginValidationAgent: Sendable {
    public static let descriptor = FXBacktestAgentDescriptor(
        id: .pluginValidation,
        displayName: "Plugin Validation",
        responsibility: "Validate FXBacktest Plugin API v1 descriptors, parameters, and backend declarations before execution."
    )

    public init() {}

    public func validate(plugin: AnyFXBacktestPlugin) -> FXBacktestAgentOutcome {
        let started = Date()
        do {
            try validateOrThrow(plugin: plugin)
            return Self.descriptor.outcome(
                status: .ok,
                message: "Plugin API v1 validated for \(plugin.descriptor.displayName).",
                details: ["plugin_id=\(plugin.id)", "parameters=\(plugin.parameterDefinitions.count)"],
                startedAtUtc: started
            )
        } catch {
            return Self.descriptor.outcome(
                status: .failed,
                message: String(describing: error),
                details: ["plugin_id=\(plugin.id)"],
                startedAtUtc: started
            )
        }
    }

    private func validateOrThrow(plugin: AnyFXBacktestPlugin) throws {
        let descriptor = plugin.descriptor
        guard descriptor.apiVersion == .v1 else {
            throw FXBacktestError.invalidParameter("Unsupported plugin API \(descriptor.apiVersion.rawValue).")
        }
        for (label, value) in [
            ("plugin id", descriptor.id),
            ("display name", descriptor.displayName),
            ("version", descriptor.version),
            ("summary", descriptor.summary),
            ("author", descriptor.author)
        ] {
            guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FXBacktestError.invalidParameter("\(label) must not be empty.")
            }
        }
        guard descriptor.supportsCPU || descriptor.supportsMetal else {
            throw FXBacktestError.invalidParameter("\(descriptor.displayName) must support at least one execution backend.")
        }
        guard !plugin.parameterDefinitions.isEmpty else {
            throw FXBacktestError.invalidParameter("\(descriptor.displayName) must define at least one optimizable input.")
        }
        _ = try ParameterSweep.singlePass(definitions: plugin.parameterDefinitions)
        try PluginAccelerationPipeline().validate(plugin.accelerationDescriptor)
        guard plugin.accelerationDescriptor.pluginIdentifier == descriptor.id else {
            throw FXBacktestError.invalidParameter("Acceleration descriptor plugin id does not match descriptor id.")
        }
        if descriptor.supportsMetal {
            guard let kernel = plugin.metalKernel else {
                throw FXBacktestError.metalKernelMissing(plugin: descriptor.displayName)
            }
            guard !kernel.source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FXBacktestError.invalidParameter("\(descriptor.displayName) Metal source must not be empty.")
            }
            guard !kernel.entryPoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FXBacktestError.invalidParameter("\(descriptor.displayName) Metal entry point must not be empty.")
            }
        } else if plugin.metalKernel != nil {
            throw FXBacktestError.invalidParameter("\(descriptor.displayName) provides a Metal kernel but does not declare Metal support.")
        }
    }
}

public struct ResourceHealthAgent: Sendable {
    public static let descriptor = FXBacktestAgentDescriptor(
        id: .resourceHealth,
        displayName: "Resource Health",
        responsibility: "Check local CPU, Metal, memory, disk, and thermal readiness before high-volume backtests."
    )

    public init() {}

    public func evaluate(
        target: BacktestExecutionTarget,
        maxWorkers: Int,
        chunkSize: Int,
        workingDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> FXBacktestAgentOutcome {
        let started = Date()
        var warnings: [String] = []
        var details: [String] = []
        let process = ProcessInfo.processInfo
        let activeCores = max(1, process.activeProcessorCount)
        details.append("active_cpu_cores=\(activeCores)")
        details.append("physical_memory_bytes=\(process.physicalMemory)")

        guard maxWorkers > 0 else {
            return Self.descriptor.outcome(status: .failed, message: "Workers must be > 0.", startedAtUtc: started)
        }
        guard chunkSize > 0 else {
            return Self.descriptor.outcome(status: .failed, message: "Chunk size must be > 0.", startedAtUtc: started)
        }
        if maxWorkers > activeCores {
            warnings.append("workers \(maxWorkers) exceeds active CPU cores \(activeCores)")
        }
        if target.requiresMetalKernel {
            if let metalName = Self.metalDeviceName() {
                details.append("metal_device=\(metalName)")
            } else {
                return Self.descriptor.outcome(
                    status: .failed,
                    message: "Metal execution was requested, but no Metal device is available.",
                    details: details,
                    startedAtUtc: started
                )
            }
        }
        if let availableDiskBytes = Self.availableDiskBytes(for: workingDirectory) {
            details.append("available_disk_bytes=\(availableDiskBytes)")
            if availableDiskBytes < 5_000_000_000 {
                warnings.append("available disk capacity is below 5 GB")
            }
        }
        switch process.thermalState {
        case .nominal:
            details.append("thermal_state=nominal")
        case .fair:
            details.append("thermal_state=fair")
        case .serious:
            warnings.append("thermal state is serious")
        case .critical:
            warnings.append("thermal state is critical")
        @unknown default:
            warnings.append("thermal state is unknown")
        }

        let status: FXBacktestAgentStatus = warnings.isEmpty ? .ok : .warning
        let message = warnings.isEmpty
            ? "Resource health ready for \(target.displayName) execution."
            : "Resource health has warnings: \(warnings.joined(separator: "; "))."
        return Self.descriptor.outcome(
            status: status,
            message: message,
            details: details,
            startedAtUtc: started
        )
    }

    private static func metalDeviceName() -> String? {
        #if canImport(Metal)
        MTLCreateSystemDefaultDevice()?.name
        #else
        nil
        #endif
    }

    private static func availableDiskBytes(for url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))
            .flatMap(\.volumeAvailableCapacityForImportantUsage)
    }
}

private extension FXBacktestAgentDescriptor {
    func outcome(
        status: FXBacktestAgentStatus,
        message: String,
        details: [String] = [],
        startedAtUtc: Date
    ) -> FXBacktestAgentOutcome {
        FXBacktestAgentOutcome(
            descriptor: self,
            status: status,
            message: message,
            details: details,
            startedAtUtc: startedAtUtc,
            finishedAtUtc: Date()
        )
    }
}
