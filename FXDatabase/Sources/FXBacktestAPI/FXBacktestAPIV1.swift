import Foundation

public enum FXBacktestAPIV1 {
    public static let version = "fxdatabase.fxbacktest.v1"
    public static let statusPath = "/v1/status"
    public static let m1HistoryPath = "/v1/history/m1"
    public static let resultSchemaPath = "/v1/backtest/results/schema"
    public static let resultRunStartPath = "/v1/backtest/results/runs/start"
    public static let resultRunCompletePath = "/v1/backtest/results/runs/complete"
    public static let resultPassAppendPath = "/v1/backtest/results/passes/append"
    public static let resultPurgePath = "/v1/backtest/results/purge"
    public static let resultRunGetPath = "/v1/backtest/results/runs/get"
    public static let resultPassesGetPath = "/v1/backtest/results/passes/get"
    public static let maximumRowsLimit = 5_000_000
    public static let maximumResultBatchSize = 10_000
    public static let maximumResultReadLimit = 10_000
}

public struct FXBacktestAPIStatusResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let service: String
    public let status: String

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case service
        case status
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, service: String = "FXDatabase", status: String = "ok") {
        self.apiVersion = apiVersion
        self.service = service
        self.status = status
    }
}

public struct FXBacktestM1HistoryRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let brokerSourceId: String
    public let sourceOrigin: String
    public let logicalSymbol: String
    public let utcStartInclusive: Int64
    public let utcEndExclusive: Int64
    public let expectedMT5Symbol: String?
    public let expectedDigits: Int?
    public let maximumRows: Int?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case brokerSourceId = "broker_source_id"
        case sourceOrigin = "source_origin"
        case logicalSymbol = "logical_symbol"
        case utcStartInclusive = "utc_start_inclusive"
        case utcEndExclusive = "utc_end_exclusive"
        case expectedMT5Symbol = "expected_mt5_symbol"
        case expectedDigits = "expected_digits"
        case maximumRows = "maximum_rows"
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        brokerSourceId: String,
        sourceOrigin: String = "MT5",
        logicalSymbol: String,
        utcStartInclusive: Int64,
        utcEndExclusive: Int64,
        expectedMT5Symbol: String? = nil,
        expectedDigits: Int? = nil,
        maximumRows: Int? = nil
    ) {
        self.apiVersion = apiVersion
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.utcStartInclusive = utcStartInclusive
        self.utcEndExclusive = utcEndExclusive
        self.expectedMT5Symbol = expectedMT5Symbol
        self.expectedDigits = expectedDigits
        self.maximumRows = maximumRows
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.apiVersion = try container.decode(String.self, forKey: .apiVersion)
        self.brokerSourceId = try container.decode(String.self, forKey: .brokerSourceId)
        self.sourceOrigin = try container.decodeIfPresent(String.self, forKey: .sourceOrigin) ?? "MT5"
        self.logicalSymbol = try container.decode(String.self, forKey: .logicalSymbol)
        self.utcStartInclusive = try container.decode(Int64.self, forKey: .utcStartInclusive)
        self.utcEndExclusive = try container.decode(Int64.self, forKey: .utcEndExclusive)
        self.expectedMT5Symbol = try container.decodeIfPresent(String.self, forKey: .expectedMT5Symbol)
        self.expectedDigits = try container.decodeIfPresent(Int.self, forKey: .expectedDigits)
        self.maximumRows = try container.decodeIfPresent(Int.self, forKey: .maximumRows)
    }

    public func validate() throws {
        guard apiVersion == FXBacktestAPIV1.version else {
            throw FXBacktestAPIValidationError.unsupportedVersion(apiVersion)
        }
        guard !brokerSourceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("broker_source_id must not be empty")
        }
        guard Self.isValidSourceOrigin(sourceOrigin) else {
            throw FXBacktestAPIValidationError.invalidField("source_origin must be uppercase letters, numbers, underscore, or hyphen")
        }
        guard !logicalSymbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("logical_symbol must not be empty")
        }
        guard utcStartInclusive < utcEndExclusive else {
            throw FXBacktestAPIValidationError.invalidField("utc_start_inclusive must be before utc_end_exclusive")
        }
        guard utcStartInclusive % 60 == 0, utcEndExclusive % 60 == 0 else {
            throw FXBacktestAPIValidationError.invalidField("UTC range boundaries must be minute-aligned")
        }
        if let expectedMT5Symbol {
            guard !expectedMT5Symbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw FXBacktestAPIValidationError.invalidField("expected_mt5_symbol must not be empty when supplied")
            }
        }
        if let expectedDigits {
            guard (0...10).contains(expectedDigits) else {
                throw FXBacktestAPIValidationError.invalidField("expected_digits must be between 0 and 10")
            }
        }
        if let maximumRows {
            guard maximumRows > 0 else {
                throw FXBacktestAPIValidationError.invalidField("maximum_rows must be positive when supplied")
            }
            guard maximumRows <= FXBacktestAPIV1.maximumRowsLimit else {
                throw FXBacktestAPIValidationError.invalidField("maximum_rows must not exceed \(FXBacktestAPIV1.maximumRowsLimit)")
            }
        }
    }

    fileprivate static func isValidSourceOrigin(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed == value, trimmed == trimmed.uppercased() else { return false }
        return trimmed.allSatisfy { character in
            character.isUppercase || character.isNumber || character == "_" || character == "-"
        }
    }
}

public struct FXBacktestM1HistoryMetadata: Codable, Equatable, Sendable {
    public let brokerSourceId: String
    public let sourceOrigin: String
    public let logicalSymbol: String
    public let mt5Symbol: String
    public let timeframe: String
    public let digits: Int
    public let requestedUtcStart: Int64
    public let requestedUtcEndExclusive: Int64
    public let firstUtc: Int64?
    public let lastUtc: Int64?
    public let rowCount: Int

    enum CodingKeys: String, CodingKey {
        case brokerSourceId = "broker_source_id"
        case sourceOrigin = "source_origin"
        case logicalSymbol = "logical_symbol"
        case mt5Symbol = "mt5_symbol"
        case timeframe
        case digits
        case requestedUtcStart = "requested_utc_start"
        case requestedUtcEndExclusive = "requested_utc_end_exclusive"
        case firstUtc = "first_utc"
        case lastUtc = "last_utc"
        case rowCount = "row_count"
    }

    public init(
        brokerSourceId: String,
        sourceOrigin: String = "MT5",
        logicalSymbol: String,
        mt5Symbol: String,
        timeframe: String = "M1",
        digits: Int,
        requestedUtcStart: Int64,
        requestedUtcEndExclusive: Int64,
        firstUtc: Int64?,
        lastUtc: Int64?,
        rowCount: Int
    ) {
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.logicalSymbol = logicalSymbol
        self.mt5Symbol = mt5Symbol
        self.timeframe = timeframe
        self.digits = digits
        self.requestedUtcStart = requestedUtcStart
        self.requestedUtcEndExclusive = requestedUtcEndExclusive
        self.firstUtc = firstUtc
        self.lastUtc = lastUtc
        self.rowCount = rowCount
    }
}

public struct FXBacktestM1HistoryResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let metadata: FXBacktestM1HistoryMetadata
    public let utcTimestamps: [Int64]
    public let open: [Int64]
    public let high: [Int64]
    public let low: [Int64]
    public let close: [Int64]
    public let volume: [UInt64]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case metadata
        case utcTimestamps = "utc_timestamps"
        case open
        case high
        case low
        case close
        case volume
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        metadata: FXBacktestM1HistoryMetadata,
        utcTimestamps: [Int64],
        open: [Int64],
        high: [Int64],
        low: [Int64],
        close: [Int64],
        volume: [UInt64]
    ) {
        self.apiVersion = apiVersion
        self.metadata = metadata
        self.utcTimestamps = utcTimestamps
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }

    public func validate() throws {
        guard apiVersion == FXBacktestAPIV1.version else {
            throw FXBacktestAPIValidationError.unsupportedVersion(apiVersion)
        }
        let count = utcTimestamps.count
        guard !metadata.brokerSourceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("metadata.broker_source_id must not be empty")
        }
        guard FXBacktestM1HistoryRequest.isValidSourceOrigin(metadata.sourceOrigin) else {
            throw FXBacktestAPIValidationError.invalidField("metadata.source_origin must be a supported data source origin")
        }
        guard !metadata.logicalSymbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("metadata.logical_symbol must not be empty")
        }
        guard !metadata.mt5Symbol.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("metadata.mt5_symbol must not be empty")
        }
        guard metadata.timeframe == "M1" else {
            throw FXBacktestAPIValidationError.invalidField("metadata.timeframe must be M1")
        }
        guard (0...10).contains(metadata.digits) else {
            throw FXBacktestAPIValidationError.invalidField("metadata.digits must be between 0 and 10")
        }
        guard metadata.requestedUtcStart < metadata.requestedUtcEndExclusive else {
            throw FXBacktestAPIValidationError.invalidField("metadata requested UTC range is invalid")
        }
        guard metadata.requestedUtcStart % 60 == 0, metadata.requestedUtcEndExclusive % 60 == 0 else {
            throw FXBacktestAPIValidationError.invalidField("metadata requested UTC range must be minute-aligned")
        }
        guard metadata.rowCount == count else {
            throw FXBacktestAPIValidationError.invalidField("metadata.row_count does not match utc_timestamps count")
        }
        guard open.count == count, high.count == count, low.count == count, close.count == count, volume.count == count else {
            throw FXBacktestAPIValidationError.invalidField("OHLCV column counts do not match")
        }
        if count == 0 {
            guard metadata.firstUtc == nil, metadata.lastUtc == nil else {
                throw FXBacktestAPIValidationError.invalidField("metadata first_utc/last_utc must be null when row_count is zero")
            }
            return
        }
        guard metadata.firstUtc == utcTimestamps.first, metadata.lastUtc == utcTimestamps.last else {
            throw FXBacktestAPIValidationError.invalidField("metadata first_utc/last_utc do not match timestamp columns")
        }
        for index in 0..<count {
            if index > 0, utcTimestamps[index] <= utcTimestamps[index - 1] {
                throw FXBacktestAPIValidationError.invalidField("utc_timestamps must be strictly increasing")
            }
            guard utcTimestamps[index] % 60 == 0 else {
                throw FXBacktestAPIValidationError.invalidField("utc_timestamps must be minute-aligned")
            }
            guard utcTimestamps[index] >= metadata.requestedUtcStart,
                  utcTimestamps[index] < metadata.requestedUtcEndExclusive else {
                throw FXBacktestAPIValidationError.invalidField("utc_timestamps must stay inside the requested UTC range")
            }
            guard open[index] > 0, high[index] > 0, low[index] > 0, close[index] > 0 else {
                throw FXBacktestAPIValidationError.invalidField("OHLC values must be positive")
            }
            guard high[index] >= open[index],
                  high[index] >= close[index],
                  high[index] >= low[index],
                  low[index] <= open[index],
                  low[index] <= close[index] else {
                throw FXBacktestAPIValidationError.invalidField("OHLC invariant failed at index \(index)")
            }
        }
    }
}

public struct FXBacktestAPIErrorResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let error: FXBacktestAPIErrorBody

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case error
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, code: String, message: String) {
        self.apiVersion = apiVersion
        self.error = FXBacktestAPIErrorBody(code: code, message: message)
    }
}

public struct FXBacktestAPIErrorBody: Codable, Equatable, Sendable {
    public let code: String
    public let message: String
}

public enum FXBacktestAPIValidationError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedVersion(String)
    case invalidField(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported FXBacktest API version '\(version)'; expected '\(FXBacktestAPIV1.version)'."
        case .invalidField(let reason):
            return "Invalid FXBacktest API field: \(reason)."
        }
    }
}

public struct FXBacktestResultSchemaRequest: Codable, Equatable, Sendable {
    public let apiVersion: String

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
    }

    public init(apiVersion: String = FXBacktestAPIV1.version) {
        self.apiVersion = apiVersion
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
    }
}

public struct FXBacktestResultRunStartRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String
    public let pluginId: String
    public let engine: String
    public let brokerSourceId: String
    public let primarySymbol: String
    public let symbols: [String]
    public let settingsJSON: String
    public let parameterSpaceJSON: String
    public let totalPasses: UInt64
    public let note: String?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
        case pluginId = "plugin_id"
        case engine
        case brokerSourceId = "broker_source_id"
        case primarySymbol = "primary_symbol"
        case symbols
        case settingsJSON = "settings_json"
        case parameterSpaceJSON = "parameter_space_json"
        case totalPasses = "total_passes"
        case note
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        runId: String,
        pluginId: String,
        engine: String,
        brokerSourceId: String,
        primarySymbol: String,
        symbols: [String],
        settingsJSON: String,
        parameterSpaceJSON: String,
        totalPasses: UInt64,
        note: String? = nil
    ) {
        self.apiVersion = apiVersion
        self.runId = runId
        self.pluginId = pluginId
        self.engine = engine
        self.brokerSourceId = brokerSourceId
        self.primarySymbol = primarySymbol
        self.symbols = symbols
        self.settingsJSON = settingsJSON
        self.parameterSpaceJSON = parameterSpaceJSON
        self.totalPasses = totalPasses
        self.note = note
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        try FXBacktestAPIV1.requireNonEmpty(pluginId, "plugin_id")
        try FXBacktestAPIV1.requireNonEmpty(engine, "engine")
        try FXBacktestAPIV1.requireNonEmpty(brokerSourceId, "broker_source_id")
        try FXBacktestAPIV1.requireNonEmpty(primarySymbol, "primary_symbol")
        guard !symbols.isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("symbols must not be empty")
        }
        for symbol in symbols {
            try FXBacktestAPIV1.requireNonEmpty(symbol, "symbols")
        }
        try FXBacktestAPIV1.validateJSONObjectString(settingsJSON, field: "settings_json")
        try FXBacktestAPIV1.validateJSONObjectString(parameterSpaceJSON, field: "parameter_space_json")
        guard totalPasses > 0 else {
            throw FXBacktestAPIValidationError.invalidField("total_passes must be positive")
        }
    }
}

public struct FXBacktestResultPassDTO: Codable, Equatable, Sendable {
    public let passIndex: UInt64
    public let pluginId: String
    public let engine: String
    public let netProfit: Double
    public let grossProfit: Double
    public let grossLoss: Double
    public let maxDrawdown: Double
    public let totalTrades: UInt32
    public let winningTrades: UInt32
    public let losingTrades: UInt32
    public let winRate: Double
    public let profitFactor: Double
    public let barsProcessed: UInt32
    public let flags: UInt32
    public let errorMessage: String?
    public let parametersJSON: String

    enum CodingKeys: String, CodingKey {
        case passIndex = "pass_index"
        case pluginId = "plugin_id"
        case engine
        case netProfit = "net_profit"
        case grossProfit = "gross_profit"
        case grossLoss = "gross_loss"
        case maxDrawdown = "max_drawdown"
        case totalTrades = "total_trades"
        case winningTrades = "winning_trades"
        case losingTrades = "losing_trades"
        case winRate = "win_rate"
        case profitFactor = "profit_factor"
        case barsProcessed = "bars_processed"
        case flags
        case errorMessage = "error_message"
        case parametersJSON = "parameters_json"
    }

    public init(
        passIndex: UInt64,
        pluginId: String,
        engine: String,
        netProfit: Double,
        grossProfit: Double,
        grossLoss: Double,
        maxDrawdown: Double,
        totalTrades: UInt32,
        winningTrades: UInt32,
        losingTrades: UInt32,
        winRate: Double,
        profitFactor: Double,
        barsProcessed: UInt32,
        flags: UInt32 = 0,
        errorMessage: String? = nil,
        parametersJSON: String
    ) {
        self.passIndex = passIndex
        self.pluginId = pluginId
        self.engine = engine
        self.netProfit = netProfit
        self.grossProfit = grossProfit
        self.grossLoss = grossLoss
        self.maxDrawdown = maxDrawdown
        self.totalTrades = totalTrades
        self.winningTrades = winningTrades
        self.losingTrades = losingTrades
        self.winRate = winRate
        self.profitFactor = profitFactor
        self.barsProcessed = barsProcessed
        self.flags = flags
        self.errorMessage = errorMessage
        self.parametersJSON = parametersJSON
    }

    public func validate() throws {
        try FXBacktestAPIV1.requireNonEmpty(pluginId, "plugin_id")
        try FXBacktestAPIV1.requireNonEmpty(engine, "engine")
        for (field, value) in [
            ("net_profit", netProfit),
            ("gross_profit", grossProfit),
            ("gross_loss", grossLoss),
            ("max_drawdown", maxDrawdown),
            ("win_rate", winRate),
            ("profit_factor", profitFactor)
        ] where !value.isFinite {
            throw FXBacktestAPIValidationError.invalidField("\(field) must be finite")
        }
        try FXBacktestAPIV1.validateJSONArrayString(parametersJSON, field: "parameters_json")
    }
}

public struct FXBacktestResultPassAppendRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String
    public let results: [FXBacktestResultPassDTO]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
        case results
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, runId: String, results: [FXBacktestResultPassDTO]) {
        self.apiVersion = apiVersion
        self.runId = runId
        self.results = results
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        guard !results.isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("results must not be empty")
        }
        guard results.count <= FXBacktestAPIV1.maximumResultBatchSize else {
            throw FXBacktestAPIValidationError.invalidField("results must not exceed \(FXBacktestAPIV1.maximumResultBatchSize) rows")
        }
        try results.forEach { try $0.validate() }
    }
}

public struct FXBacktestResultRunCompleteRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String
    public let completedPasses: UInt64
    public let elapsedSeconds: Double
    public let status: String

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
        case completedPasses = "completed_passes"
        case elapsedSeconds = "elapsed_seconds"
        case status
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        runId: String,
        completedPasses: UInt64,
        elapsedSeconds: Double,
        status: String
    ) {
        self.apiVersion = apiVersion
        self.runId = runId
        self.completedPasses = completedPasses
        self.elapsedSeconds = elapsedSeconds
        self.status = status
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        try FXBacktestAPIV1.requireNonEmpty(status, "status")
        guard elapsedSeconds.isFinite, elapsedSeconds >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("elapsed_seconds must be finite and >= 0")
        }
    }
}

public struct FXBacktestResultPurgeRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let all: Bool
    public let olderThanDays: Int?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case all
        case olderThanDays = "older_than_days"
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, all: Bool = false, olderThanDays: Int? = nil) {
        self.apiVersion = apiVersion
        self.all = all
        self.olderThanDays = olderThanDays
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        if all {
            guard olderThanDays == nil else {
                throw FXBacktestAPIValidationError.invalidField("older_than_days must be omitted when all is true")
            }
            return
        }
        guard let olderThanDays, olderThanDays > 0 else {
            throw FXBacktestAPIValidationError.invalidField("older_than_days must be positive unless all is true")
        }
    }
}

public struct FXBacktestResultRunGetRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, runId: String) {
        self.apiVersion = apiVersion
        self.runId = runId
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
    }
}

public struct FXBacktestResultPassesGetRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String
    public let offset: Int
    public let limit: Int

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
        case offset
        case limit
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, runId: String, offset: Int = 0, limit: Int = 1_000) {
        self.apiVersion = apiVersion
        self.runId = runId
        self.offset = offset
        self.limit = limit
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        guard offset >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("offset must be >= 0")
        }
        guard limit > 0, limit <= FXBacktestAPIV1.maximumResultReadLimit else {
            throw FXBacktestAPIValidationError.invalidField("limit must be in 1...\(FXBacktestAPIV1.maximumResultReadLimit)")
        }
    }
}

public struct FXBacktestResultMutationResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let status: String
    public let runId: String?
    public let affectedRows: Int?
    public let sqlStatements: Int

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case status
        case runId = "run_id"
        case affectedRows = "affected_rows"
        case sqlStatements = "sql_statements"
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        status: String = "ok",
        runId: String? = nil,
        affectedRows: Int? = nil,
        sqlStatements: Int
    ) {
        self.apiVersion = apiVersion
        self.status = status
        self.runId = runId
        self.affectedRows = affectedRows
        self.sqlStatements = sqlStatements
    }
}

public struct FXBacktestResultPurgeReport: Codable, Equatable, Sendable {
    public let scope: String
    public let sqlStatements: Int

    enum CodingKeys: String, CodingKey {
        case scope
        case sqlStatements = "sql_statements"
    }

    public init(scope: String, sqlStatements: Int) {
        self.scope = scope
        self.sqlStatements = sqlStatements
    }
}

public struct FXBacktestResultPurgeResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let report: FXBacktestResultPurgeReport

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case report
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, report: FXBacktestResultPurgeReport) {
        self.apiVersion = apiVersion
        self.report = report
    }
}

public struct FXBacktestResultRunRecord: Codable, Equatable, Sendable {
    public let runId: String
    public let createdAtUnixMilliseconds: Int64
    public let completedAtUnixMilliseconds: Int64?
    public let pluginId: String
    public let engine: String
    public let brokerSourceId: String
    public let primarySymbol: String
    public let symbols: [String]
    public let apiVersion: String
    public let settingsJSON: String
    public let parameterSpaceJSON: String
    public let status: String
    public let completedPasses: UInt64
    public let totalPasses: UInt64
    public let note: String

    enum CodingKeys: String, CodingKey {
        case runId = "run_id"
        case createdAtUnixMilliseconds = "created_at_unix_ms"
        case completedAtUnixMilliseconds = "completed_at_unix_ms"
        case pluginId = "plugin_id"
        case engine
        case brokerSourceId = "broker_source_id"
        case primarySymbol = "primary_symbol"
        case symbols
        case apiVersion = "api_version"
        case settingsJSON = "settings_json"
        case parameterSpaceJSON = "parameter_space_json"
        case status
        case completedPasses = "completed_passes"
        case totalPasses = "total_passes"
        case note
    }

    public init(
        runId: String,
        createdAtUnixMilliseconds: Int64,
        completedAtUnixMilliseconds: Int64?,
        pluginId: String,
        engine: String,
        brokerSourceId: String,
        primarySymbol: String,
        symbols: [String],
        apiVersion: String,
        settingsJSON: String,
        parameterSpaceJSON: String,
        status: String,
        completedPasses: UInt64,
        totalPasses: UInt64,
        note: String
    ) {
        self.runId = runId
        self.createdAtUnixMilliseconds = createdAtUnixMilliseconds
        self.completedAtUnixMilliseconds = completedAtUnixMilliseconds
        self.pluginId = pluginId
        self.engine = engine
        self.brokerSourceId = brokerSourceId
        self.primarySymbol = primarySymbol
        self.symbols = symbols
        self.apiVersion = apiVersion
        self.settingsJSON = settingsJSON
        self.parameterSpaceJSON = parameterSpaceJSON
        self.status = status
        self.completedPasses = completedPasses
        self.totalPasses = totalPasses
        self.note = note
    }
}

public struct FXBacktestResultRunGetResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let run: FXBacktestResultRunRecord?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case run
    }

    public init(apiVersion: String = FXBacktestAPIV1.version, run: FXBacktestResultRunRecord?) {
        self.apiVersion = apiVersion
        self.run = run
    }
}

public struct FXBacktestResultPassesGetResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let runId: String
    public let offset: Int
    public let limit: Int
    public let results: [FXBacktestResultPassDTO]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case runId = "run_id"
        case offset
        case limit
        case results
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.version,
        runId: String,
        offset: Int,
        limit: Int,
        results: [FXBacktestResultPassDTO]
    ) {
        self.apiVersion = apiVersion
        self.runId = runId
        self.offset = offset
        self.limit = limit
        self.results = results
    }
}

extension FXBacktestAPIV1 {
    static func validateVersion(_ apiVersion: String) throws {
        guard apiVersion == version else {
            throw FXBacktestAPIValidationError.unsupportedVersion(apiVersion)
        }
    }

    static func requireNonEmpty(_ value: String, _ field: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("\(field) must not be empty")
        }
    }

    static func validateJSONObjectString(_ value: String, field: String) throws {
        guard let data = value.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              object is [String: Any] else {
            throw FXBacktestAPIValidationError.invalidField("\(field) must be a JSON object string")
        }
    }

    static func validateJSONArrayString(_ value: String, field: String) throws {
        guard let data = value.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              object is [Any] else {
            throw FXBacktestAPIValidationError.invalidField("\(field) must be a JSON array string")
        }
    }
}
