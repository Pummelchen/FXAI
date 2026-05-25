import Foundation

public enum FXBacktestAPIV1 {
    public static let version = "fxdatabase.fxbacktest.v1"
    public static let latestVersion = version
    public static let statusPath = "/v1/status"
    public static let m1HistoryPath = "/v1/history/m1"
    public static let resultSchemaPath = "/v1/backtest/results/schema"
    public static let resultRunStartPath = "/v1/backtest/results/runs/start"
    public static let resultRunCompletePath = "/v1/backtest/results/runs/complete"
    public static let resultPassAppendPath = "/v1/backtest/results/passes/append"
    public static let resultPurgePath = "/v1/backtest/results/purge"
    public static let resultRunGetPath = "/v1/backtest/results/runs/get"
    public static let resultPassesGetPath = "/v1/backtest/results/passes/get"
    public static let configurationSchemaPath = "/v1/backtest/configuration/schema"
    public static let configurationRegisterPath = "/v1/backtest/configuration/register"
    public static let configurationGetPath = "/v1/backtest/configuration/get"
    public static let lineageCreatePath = "/v1/backtest/lineage/create"
    public static let lineageGetPath = "/v1/backtest/lineage/get"
    public static let certificationEvidencePath = "/v1/certification/evidence"
    public static let maximumRowsLimit = 5_000_000
    public static let maximumResultBatchSize = 10_000
    public static let maximumResultReadLimit = 10_000
    public static let maximumConfigurationParameterCount = 20_000
    public static let maximumCertificationComponentCount = 100_000
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, service: String = "FXDatabase", status: String = "ok") {
        self.apiVersion = apiVersion
        self.service = service
        self.status = status
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(service, "service")
        try FXBacktestAPIV1.requireNonEmpty(status, "status")
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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
        guard apiVersion == FXBacktestAPIV1.latestVersion else {
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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
        guard apiVersion == FXBacktestAPIV1.latestVersion else {
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, code: String, message: String) {
        self.apiVersion = apiVersion
        self.error = FXBacktestAPIErrorBody(code: code, message: message)
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(error.code, "error.code")
        try FXBacktestAPIV1.requireNonEmpty(error.message, "error.message")
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
            return "Unsupported FXBacktest API version '\(version)'; expected '\(FXBacktestAPIV1.latestVersion)'."
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion) {
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, runId: String, results: [FXBacktestResultPassDTO]) {
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, all: Bool = false, olderThanDays: Int? = nil) {
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, runId: String) {
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

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, runId: String, offset: Int = 0, limit: Int = 1_000) {
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

public enum FXBacktestConfigurationValueKind: String, Codable, CaseIterable, Sendable {
    case integer
    case decimal
    case boolean
}

public struct FXBacktestConfigurationParameterDTO: Codable, Equatable, Sendable {
    public let key: String
    public let displayName: String
    public let valueKind: FXBacktestConfigurationValueKind
    public let defaultValue: Double
    public let minimum: Double
    public let step: Double
    public let maximum: Double
    public let unit: String
    public let description: String

    enum CodingKeys: String, CodingKey {
        case key
        case displayName = "display_name"
        case valueKind = "value_kind"
        case defaultValue = "default_value"
        case minimum
        case step
        case maximum
        case unit
        case description
    }

    public init(
        key: String,
        displayName: String,
        valueKind: FXBacktestConfigurationValueKind,
        defaultValue: Double,
        minimum: Double,
        step: Double,
        maximum: Double,
        unit: String = "",
        description: String = ""
    ) {
        self.key = key
        self.displayName = displayName
        self.valueKind = valueKind
        self.defaultValue = defaultValue
        self.minimum = minimum
        self.step = step
        self.maximum = maximum
        self.unit = unit
        self.description = description
    }

    public func validate() throws {
        guard FXBacktestAPIV1.isValidParameterKey(key) else {
            throw FXBacktestAPIValidationError.invalidField("configuration parameter key must be non-empty and contain only letters, numbers, or underscore")
        }
        try FXBacktestAPIV1.requireNonEmpty(displayName, "configuration display_name")
        for (field, value) in [
            ("default_value", defaultValue),
            ("minimum", minimum),
            ("step", step),
            ("maximum", maximum)
        ] where !value.isFinite {
            throw FXBacktestAPIValidationError.invalidField("\(key).\(field) must be finite")
        }
        guard minimum <= defaultValue, defaultValue <= maximum else {
            throw FXBacktestAPIValidationError.invalidField("\(key).default_value must be inside minimum...maximum")
        }
        guard minimum <= maximum else {
            throw FXBacktestAPIValidationError.invalidField("\(key).minimum must be <= maximum")
        }
        guard step > 0 else {
            throw FXBacktestAPIValidationError.invalidField("\(key).step must be > 0")
        }
        switch valueKind {
        case .integer:
            for (field, value) in [
                ("default_value", defaultValue),
                ("minimum", minimum),
                ("step", step),
                ("maximum", maximum)
            ] where value.rounded() != value {
                throw FXBacktestAPIValidationError.invalidField("\(key).\(field) must be an integer")
            }
        case .boolean:
            guard [0, 1].contains(defaultValue), [0, 1].contains(minimum), [0, 1].contains(maximum), step == 1 else {
                throw FXBacktestAPIValidationError.invalidField("\(key) boolean parameters must use 0/1 values and step 1")
            }
        case .decimal:
            break
        }
    }
}

public struct FXBacktestPluginConfigurationDTO: Codable, Equatable, Sendable {
    public let pluginId: String
    public let acceleratorId: String
    public let parameters: [FXBacktestConfigurationParameterDTO]

    enum CodingKeys: String, CodingKey {
        case pluginId = "plugin_id"
        case acceleratorId = "accelerator_id"
        case parameters
    }

    public init(pluginId: String, acceleratorId: String, parameters: [FXBacktestConfigurationParameterDTO]) {
        self.pluginId = pluginId
        self.acceleratorId = acceleratorId
        self.parameters = parameters
    }

    public func validate() throws {
        try FXBacktestAPIV1.requireNonEmpty(pluginId, "plugin_id")
        try FXBacktestAPIV1.requireNonEmpty(acceleratorId, "accelerator_id")
        guard !parameters.isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("\(pluginId)/\(acceleratorId) parameters must not be empty")
        }
        let keys = parameters.map(\.key)
        guard Set(keys).count == keys.count else {
            throw FXBacktestAPIValidationError.invalidField("\(pluginId)/\(acceleratorId) contains duplicate parameter keys")
        }
        try parameters.forEach { try $0.validate() }
    }
}

public struct FXBacktestConfigurationSchemaRequest: Codable, Equatable, Sendable {
    public let apiVersion: String

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
    }

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion) {
        self.apiVersion = apiVersion
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
    }
}

public struct FXBacktestConfigurationRegistrationRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let sharedParameters: [FXBacktestConfigurationParameterDTO]
    public let pluginConfigurations: [FXBacktestPluginConfigurationDTO]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case sharedParameters = "shared_parameters"
        case pluginConfigurations = "plugin_configurations"
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.latestVersion,
        sharedParameters: [FXBacktestConfigurationParameterDTO],
        pluginConfigurations: [FXBacktestPluginConfigurationDTO]
    ) {
        self.apiVersion = apiVersion
        self.sharedParameters = sharedParameters
        self.pluginConfigurations = pluginConfigurations
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        guard !sharedParameters.isEmpty || !pluginConfigurations.isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("configuration registration must include shared or plugin parameters")
        }
        let sharedKeys = sharedParameters.map(\.key)
        guard Set(sharedKeys).count == sharedKeys.count else {
            throw FXBacktestAPIValidationError.invalidField("shared configuration contains duplicate parameter keys")
        }
        try sharedParameters.forEach { try $0.validate() }
        try pluginConfigurations.forEach { try $0.validate() }
        let pluginScopes = pluginConfigurations.map { "\($0.pluginId)\u{1F}\($0.acceleratorId)" }
        guard Set(pluginScopes).count == pluginScopes.count else {
            throw FXBacktestAPIValidationError.invalidField("plugin configuration contains duplicate plugin/accelerator scopes")
        }
        let parameterCount = sharedParameters.count + pluginConfigurations.reduce(0) { $0 + $1.parameters.count }
        guard parameterCount <= FXBacktestAPIV1.maximumConfigurationParameterCount else {
            throw FXBacktestAPIValidationError.invalidField("configuration parameters must not exceed \(FXBacktestAPIV1.maximumConfigurationParameterCount)")
        }
    }
}

public struct FXBacktestConfigurationGetRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let pluginIds: [String]?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case pluginIds = "plugin_ids"
    }

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, pluginIds: [String]? = nil) {
        self.apiVersion = apiVersion
        self.pluginIds = pluginIds
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        if let pluginIds {
            guard !pluginIds.isEmpty else {
                throw FXBacktestAPIValidationError.invalidField("plugin_ids must be omitted or non-empty")
            }
            for pluginId in pluginIds {
                try FXBacktestAPIV1.requireNonEmpty(pluginId, "plugin_ids")
            }
        }
    }
}

public struct FXBacktestConfigurationSnapshotResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let sharedParameters: [FXBacktestConfigurationParameterDTO]
    public let pluginConfigurations: [FXBacktestPluginConfigurationDTO]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case sharedParameters = "shared_parameters"
        case pluginConfigurations = "plugin_configurations"
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.latestVersion,
        sharedParameters: [FXBacktestConfigurationParameterDTO],
        pluginConfigurations: [FXBacktestPluginConfigurationDTO]
    ) {
        self.apiVersion = apiVersion
        self.sharedParameters = sharedParameters
        self.pluginConfigurations = pluginConfigurations
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        let sharedKeys = sharedParameters.map(\.key)
        guard Set(sharedKeys).count == sharedKeys.count else {
            throw FXBacktestAPIValidationError.invalidField("shared configuration contains duplicate parameter keys")
        }
        try sharedParameters.forEach { try $0.validate() }
        try pluginConfigurations.forEach { try $0.validate() }
        let pluginScopes = pluginConfigurations.map { "\($0.pluginId)\u{1F}\($0.acceleratorId)" }
        guard Set(pluginScopes).count == pluginScopes.count else {
            throw FXBacktestAPIValidationError.invalidField("plugin configuration contains duplicate plugin/accelerator scopes")
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(status, "status")
        if let runId {
            try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        }
        if let affectedRows, affectedRows < 0 {
            throw FXBacktestAPIValidationError.invalidField("affected_rows must be >= 0 when supplied")
        }
        guard sqlStatements >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("sql_statements must be >= 0")
        }
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

    public func validate() throws {
        try FXBacktestAPIV1.requireNonEmpty(scope, "report.scope")
        guard sqlStatements >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("report.sql_statements must be >= 0")
        }
    }
}

public struct FXBacktestResultPurgeResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let report: FXBacktestResultPurgeReport

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case report
    }

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, report: FXBacktestResultPurgeReport) {
        self.apiVersion = apiVersion
        self.report = report
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try report.validate()
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

    public func validate() throws {
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        guard createdAtUnixMilliseconds > 0 else {
            throw FXBacktestAPIValidationError.invalidField("created_at_unix_ms must be positive")
        }
        if let completedAtUnixMilliseconds, completedAtUnixMilliseconds < createdAtUnixMilliseconds {
            throw FXBacktestAPIValidationError.invalidField("completed_at_unix_ms must be >= created_at_unix_ms")
        }
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
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.validateJSONObjectString(settingsJSON, field: "settings_json")
        try FXBacktestAPIV1.validateJSONObjectString(parameterSpaceJSON, field: "parameter_space_json")
        try FXBacktestAPIV1.requireNonEmpty(status, "status")
        guard totalPasses > 0 else {
            throw FXBacktestAPIValidationError.invalidField("total_passes must be positive")
        }
        guard completedPasses <= totalPasses else {
            throw FXBacktestAPIValidationError.invalidField("completed_passes must not exceed total_passes")
        }
    }
}

public struct FXBacktestResultRunGetResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let run: FXBacktestResultRunRecord?

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case run
    }

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, run: FXBacktestResultRunRecord?) {
        self.apiVersion = apiVersion
        self.run = run
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try run?.validate()
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
        apiVersion: String = FXBacktestAPIV1.latestVersion,
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

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(runId, "run_id")
        guard offset >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("offset must be >= 0")
        }
        guard limit > 0, limit <= FXBacktestAPIV1.maximumResultReadLimit else {
            throw FXBacktestAPIValidationError.invalidField("limit must be in 1...\(FXBacktestAPIV1.maximumResultReadLimit)")
        }
        guard results.count <= limit else {
            throw FXBacktestAPIValidationError.invalidField("results count must not exceed limit")
        }
        try results.forEach { try $0.validate() }
    }
}

public struct FXAILineageManifestDTO: Codable, Equatable, Sendable {
    public let lineageId: String
    public let lineageHash: String
    public let datasetId: String
    public let sourceProviderId: String
    public let sourceConnectorAPIVersion: String
    public let brokerSourceId: String
    public let sourceOrigin: String
    public let symbol: String
    public let timeframe: String
    public let utcStartInclusive: Int64
    public let utcEndExclusive: Int64
    public let sourceDataSnapshotHash: String
    public let fxDatabaseValidationStatus: String
    public let sineTestSyncStatus: String
    public let fxDataEngineAPIVersion: String
    public let featureGraphHash: String
    public let normalizationStateHash: String
    public let labelPolicyHash: String
    public let leakageAuditHash: String
    public let pluginId: String
    public let pluginAPIVersion: String
    public let pluginCodeHash: String
    public let acceleratorBackend: String
    public let acceleratorCodeHash: String
    public let pluginParameterSetHash: String
    public let sharedConfigurationHash: String
    public let fxBacktestRuntimeKernelVersion: String
    public let swiftVersion: String
    public let xcodeVersion: String
    public let macOSVersion: String
    public let hardwareClass: String
    public let metalDeviceId: String?
    public let pythonPackageManifestJSON: String?
    public let commandOrGUIActionId: String
    public let operatorId: String?

    enum CodingKeys: String, CodingKey {
        case lineageId = "lineage_id"
        case lineageHash = "lineage_hash"
        case datasetId = "dataset_id"
        case sourceProviderId = "source_provider_id"
        case sourceConnectorAPIVersion = "source_connector_api_version"
        case brokerSourceId = "broker_source_id"
        case sourceOrigin = "source_origin"
        case symbol
        case timeframe
        case utcStartInclusive = "utc_start_inclusive"
        case utcEndExclusive = "utc_end_exclusive"
        case sourceDataSnapshotHash = "source_data_snapshot_hash"
        case fxDatabaseValidationStatus = "fxdatabase_validation_status"
        case sineTestSyncStatus = "sinetest_sync_status"
        case fxDataEngineAPIVersion = "fxdataengine_api_version"
        case featureGraphHash = "feature_graph_hash"
        case normalizationStateHash = "normalization_state_hash"
        case labelPolicyHash = "label_policy_hash"
        case leakageAuditHash = "leakage_audit_hash"
        case pluginId = "plugin_id"
        case pluginAPIVersion = "plugin_api_version"
        case pluginCodeHash = "plugin_code_hash"
        case acceleratorBackend = "accelerator_backend"
        case acceleratorCodeHash = "accelerator_code_hash"
        case pluginParameterSetHash = "plugin_parameter_set_hash"
        case sharedConfigurationHash = "shared_configuration_hash"
        case fxBacktestRuntimeKernelVersion = "fxbacktest_runtime_kernel_version"
        case swiftVersion = "swift_version"
        case xcodeVersion = "xcode_version"
        case macOSVersion = "macos_version"
        case hardwareClass = "hardware_class"
        case metalDeviceId = "metal_device_id"
        case pythonPackageManifestJSON = "python_package_manifest_json"
        case commandOrGUIActionId = "command_or_gui_action_id"
        case operatorId = "operator_id"
    }

    public init(
        lineageId: String,
        lineageHash: String,
        datasetId: String,
        sourceProviderId: String,
        sourceConnectorAPIVersion: String,
        brokerSourceId: String,
        sourceOrigin: String,
        symbol: String,
        timeframe: String,
        utcStartInclusive: Int64,
        utcEndExclusive: Int64,
        sourceDataSnapshotHash: String,
        fxDatabaseValidationStatus: String,
        sineTestSyncStatus: String,
        fxDataEngineAPIVersion: String,
        featureGraphHash: String,
        normalizationStateHash: String,
        labelPolicyHash: String,
        leakageAuditHash: String,
        pluginId: String,
        pluginAPIVersion: String,
        pluginCodeHash: String,
        acceleratorBackend: String,
        acceleratorCodeHash: String,
        pluginParameterSetHash: String,
        sharedConfigurationHash: String,
        fxBacktestRuntimeKernelVersion: String,
        swiftVersion: String,
        xcodeVersion: String,
        macOSVersion: String,
        hardwareClass: String,
        metalDeviceId: String? = nil,
        pythonPackageManifestJSON: String? = nil,
        commandOrGUIActionId: String,
        operatorId: String? = nil
    ) {
        self.lineageId = lineageId
        self.lineageHash = lineageHash
        self.datasetId = datasetId
        self.sourceProviderId = sourceProviderId
        self.sourceConnectorAPIVersion = sourceConnectorAPIVersion
        self.brokerSourceId = brokerSourceId
        self.sourceOrigin = sourceOrigin
        self.symbol = symbol
        self.timeframe = timeframe
        self.utcStartInclusive = utcStartInclusive
        self.utcEndExclusive = utcEndExclusive
        self.sourceDataSnapshotHash = sourceDataSnapshotHash
        self.fxDatabaseValidationStatus = fxDatabaseValidationStatus
        self.sineTestSyncStatus = sineTestSyncStatus
        self.fxDataEngineAPIVersion = fxDataEngineAPIVersion
        self.featureGraphHash = featureGraphHash
        self.normalizationStateHash = normalizationStateHash
        self.labelPolicyHash = labelPolicyHash
        self.leakageAuditHash = leakageAuditHash
        self.pluginId = pluginId
        self.pluginAPIVersion = pluginAPIVersion
        self.pluginCodeHash = pluginCodeHash
        self.acceleratorBackend = acceleratorBackend
        self.acceleratorCodeHash = acceleratorCodeHash
        self.pluginParameterSetHash = pluginParameterSetHash
        self.sharedConfigurationHash = sharedConfigurationHash
        self.fxBacktestRuntimeKernelVersion = fxBacktestRuntimeKernelVersion
        self.swiftVersion = swiftVersion
        self.xcodeVersion = xcodeVersion
        self.macOSVersion = macOSVersion
        self.hardwareClass = hardwareClass
        self.metalDeviceId = metalDeviceId
        self.pythonPackageManifestJSON = pythonPackageManifestJSON
        self.commandOrGUIActionId = commandOrGUIActionId
        self.operatorId = operatorId
    }

    public func validate() throws {
        for (field, value) in [
            ("lineage_id", lineageId),
            ("lineage_hash", lineageHash),
            ("dataset_id", datasetId),
            ("source_provider_id", sourceProviderId),
            ("source_connector_api_version", sourceConnectorAPIVersion),
            ("broker_source_id", brokerSourceId),
            ("source_origin", sourceOrigin),
            ("symbol", symbol),
            ("timeframe", timeframe),
            ("source_data_snapshot_hash", sourceDataSnapshotHash),
            ("fxdatabase_validation_status", fxDatabaseValidationStatus),
            ("sinetest_sync_status", sineTestSyncStatus),
            ("fxdataengine_api_version", fxDataEngineAPIVersion),
            ("feature_graph_hash", featureGraphHash),
            ("normalization_state_hash", normalizationStateHash),
            ("label_policy_hash", labelPolicyHash),
            ("leakage_audit_hash", leakageAuditHash),
            ("plugin_id", pluginId),
            ("plugin_api_version", pluginAPIVersion),
            ("plugin_code_hash", pluginCodeHash),
            ("accelerator_backend", acceleratorBackend),
            ("accelerator_code_hash", acceleratorCodeHash),
            ("plugin_parameter_set_hash", pluginParameterSetHash),
            ("shared_configuration_hash", sharedConfigurationHash),
            ("fxbacktest_runtime_kernel_version", fxBacktestRuntimeKernelVersion),
            ("swift_version", swiftVersion),
            ("xcode_version", xcodeVersion),
            ("macos_version", macOSVersion),
            ("hardware_class", hardwareClass),
            ("command_or_gui_action_id", commandOrGUIActionId)
        ] {
            try FXBacktestAPIV1.requireNonEmpty(value, field)
        }
        guard utcStartInclusive < utcEndExclusive else {
            throw FXBacktestAPIValidationError.invalidField("lineage utc_start_inclusive must be before utc_end_exclusive")
        }
        if let pythonPackageManifestJSON {
            try FXBacktestAPIV1.validateJSONObjectString(pythonPackageManifestJSON, field: "python_package_manifest_json")
        }
    }
}

public struct FXAILineageCreateRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let manifest: FXAILineageManifestDTO

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case manifest
    }

    public init(apiVersion: String = FXBacktestAPIV1.latestVersion, manifest: FXAILineageManifestDTO) {
        self.apiVersion = apiVersion
        self.manifest = manifest
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try manifest.validate()
    }
}

public struct FXAILineageCreateResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let lineageId: String
    public let lineageHash: String
    public let accepted: Bool

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case lineageId = "lineage_id"
        case lineageHash = "lineage_hash"
        case accepted
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.latestVersion,
        lineageId: String,
        lineageHash: String,
        accepted: Bool
    ) {
        self.apiVersion = apiVersion
        self.lineageId = lineageId
        self.lineageHash = lineageHash
        self.accepted = accepted
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(lineageId, "lineage_id")
        try FXBacktestAPIV1.requireNonEmpty(lineageHash, "lineage_hash")
    }
}

public struct FXAICertificationComponentResultDTO: Codable, Equatable, Sendable {
    public let componentId: String
    public let componentType: String
    public let status: String
    public let durationSeconds: Double
    public let evidenceHash: String
    public let detail: String

    enum CodingKeys: String, CodingKey {
        case componentId = "component_id"
        case componentType = "component_type"
        case status
        case durationSeconds = "duration_seconds"
        case evidenceHash = "evidence_hash"
        case detail
    }

    public init(
        componentId: String,
        componentType: String,
        status: String,
        durationSeconds: Double,
        evidenceHash: String,
        detail: String = ""
    ) {
        self.componentId = componentId
        self.componentType = componentType
        self.status = status
        self.durationSeconds = durationSeconds
        self.evidenceHash = evidenceHash
        self.detail = detail
    }

    public func validate() throws {
        try FXBacktestAPIV1.requireNonEmpty(componentId, "component_id")
        try FXBacktestAPIV1.requireNonEmpty(componentType, "component_type")
        try FXBacktestAPIV1.requireNonEmpty(status, "status")
        try FXBacktestAPIV1.requireNonEmpty(evidenceHash, "evidence_hash")
        guard durationSeconds.isFinite, durationSeconds >= 0 else {
            throw FXBacktestAPIValidationError.invalidField("duration_seconds must be finite and >= 0")
        }
    }
}

public struct FXAICertificationEvidenceRequest: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let certificationRunId: String
    public let gitCommit: String
    public let workingTreeClean: Bool
    public let hostHardwareClass: String
    public let macOSVersion: String
    public let xcodeVersion: String
    public let swiftVersion: String
    public let metalDeviceName: String
    public let pythonVersion: String
    public let pyTorchStatus: String
    public let tensorflowStatus: String
    public let startedAtUTC: Int64
    public let completedAtUTC: Int64
    public let overallStatus: String
    public let evidenceHash: String
    public let componentResults: [FXAICertificationComponentResultDTO]

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case certificationRunId = "certification_run_id"
        case gitCommit = "git_commit"
        case workingTreeClean = "working_tree_clean"
        case hostHardwareClass = "host_hardware_class"
        case macOSVersion = "macos_version"
        case xcodeVersion = "xcode_version"
        case swiftVersion = "swift_version"
        case metalDeviceName = "metal_device_name"
        case pythonVersion = "python_version"
        case pyTorchStatus = "pytorch_status"
        case tensorflowStatus = "tensorflow_status"
        case startedAtUTC = "started_at_utc"
        case completedAtUTC = "completed_at_utc"
        case overallStatus = "overall_status"
        case evidenceHash = "evidence_hash"
        case componentResults = "component_results"
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.latestVersion,
        certificationRunId: String,
        gitCommit: String,
        workingTreeClean: Bool,
        hostHardwareClass: String,
        macOSVersion: String,
        xcodeVersion: String,
        swiftVersion: String,
        metalDeviceName: String,
        pythonVersion: String,
        pyTorchStatus: String,
        tensorflowStatus: String,
        startedAtUTC: Int64,
        completedAtUTC: Int64,
        overallStatus: String,
        evidenceHash: String,
        componentResults: [FXAICertificationComponentResultDTO]
    ) {
        self.apiVersion = apiVersion
        self.certificationRunId = certificationRunId
        self.gitCommit = gitCommit
        self.workingTreeClean = workingTreeClean
        self.hostHardwareClass = hostHardwareClass
        self.macOSVersion = macOSVersion
        self.xcodeVersion = xcodeVersion
        self.swiftVersion = swiftVersion
        self.metalDeviceName = metalDeviceName
        self.pythonVersion = pythonVersion
        self.pyTorchStatus = pyTorchStatus
        self.tensorflowStatus = tensorflowStatus
        self.startedAtUTC = startedAtUTC
        self.completedAtUTC = completedAtUTC
        self.overallStatus = overallStatus
        self.evidenceHash = evidenceHash
        self.componentResults = componentResults
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        for (field, value) in [
            ("certification_run_id", certificationRunId),
            ("git_commit", gitCommit),
            ("host_hardware_class", hostHardwareClass),
            ("macos_version", macOSVersion),
            ("xcode_version", xcodeVersion),
            ("swift_version", swiftVersion),
            ("metal_device_name", metalDeviceName),
            ("python_version", pythonVersion),
            ("pytorch_status", pyTorchStatus),
            ("tensorflow_status", tensorflowStatus),
            ("overall_status", overallStatus),
            ("evidence_hash", evidenceHash)
        ] {
            try FXBacktestAPIV1.requireNonEmpty(value, field)
        }
        guard startedAtUTC > 0, completedAtUTC >= startedAtUTC else {
            throw FXBacktestAPIValidationError.invalidField("certification timestamps are invalid")
        }
        guard !componentResults.isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("component_results must not be empty")
        }
        guard componentResults.count <= FXBacktestAPIV1.maximumCertificationComponentCount else {
            throw FXBacktestAPIValidationError.invalidField("component_results exceeds \(FXBacktestAPIV1.maximumCertificationComponentCount)")
        }
        try componentResults.forEach { try $0.validate() }
    }
}

public struct FXAICertificationEvidenceResponse: Codable, Equatable, Sendable {
    public let apiVersion: String
    public let certificationRunId: String
    public let evidenceHash: String
    public let accepted: Bool

    enum CodingKeys: String, CodingKey {
        case apiVersion = "api_version"
        case certificationRunId = "certification_run_id"
        case evidenceHash = "evidence_hash"
        case accepted
    }

    public init(
        apiVersion: String = FXBacktestAPIV1.latestVersion,
        certificationRunId: String,
        evidenceHash: String,
        accepted: Bool
    ) {
        self.apiVersion = apiVersion
        self.certificationRunId = certificationRunId
        self.evidenceHash = evidenceHash
        self.accepted = accepted
    }

    public func validate() throws {
        try FXBacktestAPIV1.validateVersion(apiVersion)
        try FXBacktestAPIV1.requireNonEmpty(certificationRunId, "certification_run_id")
        try FXBacktestAPIV1.requireNonEmpty(evidenceHash, "evidence_hash")
    }
}

extension FXBacktestAPIV1 {
    static func validateVersion(_ apiVersion: String) throws {
        guard apiVersion == latestVersion else {
            throw FXBacktestAPIValidationError.unsupportedVersion(apiVersion)
        }
    }

    static func requireNonEmpty(_ value: String, _ field: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXBacktestAPIValidationError.invalidField("\(field) must not be empty")
        }
    }

    static func isValidParameterKey(_ value: String) -> Bool {
        !value.isEmpty && value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
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
