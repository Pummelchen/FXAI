import Foundation

public enum FXExecutionContractsV1 {
    public static let latestVersion = "fxexecution.contracts.v1"
}

public enum FXExecutionEnvironment: String, Codable, CaseIterable, Sendable {
    case demo
    case live
}

public enum FXBrokerAdapterKind: String, Codable, CaseIterable, Sendable {
    case mt5
    case ibkr
    case tradingView
    case brokerREST
    case brokerFIX
}

public struct FXExecutionAccountScope: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let environment: FXExecutionEnvironment
    public let brokerAdapter: FXBrokerAdapterKind
    public let accountId: String
    public let accountLabel: String
    public let baseCurrency: String
    public let allowedSymbols: [String]

    public init(
        apiVersion: String = FXExecutionContractsV1.latestVersion,
        environment: FXExecutionEnvironment,
        brokerAdapter: FXBrokerAdapterKind,
        accountId: String,
        accountLabel: String,
        baseCurrency: String,
        allowedSymbols: [String]
    ) {
        self.apiVersion = apiVersion
        self.environment = environment
        self.brokerAdapter = brokerAdapter
        self.accountId = accountId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.accountLabel = accountLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        self.baseCurrency = baseCurrency.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        self.allowedSymbols = allowedSymbols.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }.filter { !$0.isEmpty }.sorted()
    }

    public func validate() throws {
        try Self.validateLatest(apiVersion)
        try Self.require(accountId, "account_id")
        try Self.require(accountLabel, "account_label")
        try Self.require(baseCurrency, "base_currency")
        guard !allowedSymbols.isEmpty else {
            throw FXExecutionContractError.invalidField("allowed_symbols must not be empty")
        }
    }
}

public struct FXExecutionRiskLimits: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let maxDailyLossUSD: Double
    public let maxOpenTrades: Int
    public let maxPositionLots: Double
    public let maxOrderRatePerMinute: Int
    public let maxSymbolExposureLots: Double
    public let requireBacktestLineage: Bool
    public let requireDemoPromotionForLive: Bool

    public init(
        apiVersion: String = FXExecutionContractsV1.latestVersion,
        maxDailyLossUSD: Double,
        maxOpenTrades: Int,
        maxPositionLots: Double,
        maxOrderRatePerMinute: Int,
        maxSymbolExposureLots: Double,
        requireBacktestLineage: Bool = true,
        requireDemoPromotionForLive: Bool = true
    ) {
        self.apiVersion = apiVersion
        self.maxDailyLossUSD = maxDailyLossUSD
        self.maxOpenTrades = maxOpenTrades
        self.maxPositionLots = maxPositionLots
        self.maxOrderRatePerMinute = maxOrderRatePerMinute
        self.maxSymbolExposureLots = maxSymbolExposureLots
        self.requireBacktestLineage = requireBacktestLineage
        self.requireDemoPromotionForLive = requireDemoPromotionForLive
    }

    public func validate(for environment: FXExecutionEnvironment) throws {
        try FXExecutionAccountScope.validateLatest(apiVersion)
        guard maxDailyLossUSD.isFinite, maxDailyLossUSD > 0 else {
            throw FXExecutionContractError.invalidField("max_daily_loss_usd must be finite and > 0")
        }
        guard maxOpenTrades > 0 else {
            throw FXExecutionContractError.invalidField("max_open_trades must be > 0")
        }
        guard maxPositionLots.isFinite, maxPositionLots > 0 else {
            throw FXExecutionContractError.invalidField("max_position_lots must be finite and > 0")
        }
        guard maxOrderRatePerMinute > 0 else {
            throw FXExecutionContractError.invalidField("max_order_rate_per_minute must be > 0")
        }
        guard maxSymbolExposureLots.isFinite, maxSymbolExposureLots > 0 else {
            throw FXExecutionContractError.invalidField("max_symbol_exposure_lots must be finite and > 0")
        }
        if environment == .live {
            guard requireBacktestLineage, requireDemoPromotionForLive else {
                throw FXExecutionContractError.invalidField("live execution requires lineage and demo promotion gates")
            }
        }
    }
}

public struct FXExecutionKillSwitchState: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let globalEnabled: Bool
    public let accountEnabled: Bool
    public let symbolEnabled: Bool
    public let reason: String
    public let updatedAtUTC: Int64

    public init(
        apiVersion: String = FXExecutionContractsV1.latestVersion,
        globalEnabled: Bool,
        accountEnabled: Bool,
        symbolEnabled: Bool,
        reason: String,
        updatedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.globalEnabled = globalEnabled
        self.accountEnabled = accountEnabled
        self.symbolEnabled = symbolEnabled
        self.reason = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        self.updatedAtUTC = updatedAtUTC
    }

    public var allowsOrders: Bool {
        globalEnabled && accountEnabled && symbolEnabled
    }

    public func validate() throws {
        try FXExecutionAccountScope.validateLatest(apiVersion)
        guard updatedAtUTC > 0 else {
            throw FXExecutionContractError.invalidField("kill switch updated_at_utc must be positive")
        }
        if !allowsOrders {
            try FXExecutionAccountScope.require(reason, "kill_switch_reason")
        }
    }
}

public struct FXExecutionOrderIntent: Codable, Hashable, Sendable {
    public let apiVersion: String
    public let lineageId: String
    public let promotionId: String?
    public let symbol: String
    public let side: String
    public let lots: Double
    public let maxSlippagePoints: Double
    public let requestedAtUTC: Int64

    public init(
        apiVersion: String = FXExecutionContractsV1.latestVersion,
        lineageId: String,
        promotionId: String?,
        symbol: String,
        side: String,
        lots: Double,
        maxSlippagePoints: Double,
        requestedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) {
        self.apiVersion = apiVersion
        self.lineageId = lineageId.trimmingCharacters(in: .whitespacesAndNewlines)
        self.promotionId = promotionId?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.symbol = symbol.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        self.side = side.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        self.lots = lots
        self.maxSlippagePoints = maxSlippagePoints
        self.requestedAtUTC = requestedAtUTC
    }
}

public enum FXExecutionSafetyGate {
    public static func validateOrderIntent(
        _ intent: FXExecutionOrderIntent,
        account: FXExecutionAccountScope,
        limits: FXExecutionRiskLimits,
        killSwitch: FXExecutionKillSwitchState
    ) throws {
        try FXExecutionAccountScope.validateLatest(intent.apiVersion)
        try account.validate()
        try limits.validate(for: account.environment)
        try killSwitch.validate()
        guard killSwitch.allowsOrders else {
            throw FXExecutionContractError.killSwitchActive(killSwitch.reason)
        }
        try FXExecutionAccountScope.require(intent.lineageId, "lineage_id")
        if account.environment == .live {
            guard let promotionId = intent.promotionId, !promotionId.isEmpty else {
                throw FXExecutionContractError.invalidField("live order intent requires demo promotion id")
            }
        }
        guard account.allowedSymbols.contains(intent.symbol) else {
            throw FXExecutionContractError.invalidField("\(intent.symbol) is outside account allowed_symbols")
        }
        guard ["buy", "sell"].contains(intent.side) else {
            throw FXExecutionContractError.invalidField("side must be buy or sell")
        }
        guard intent.lots.isFinite, intent.lots > 0, intent.lots <= limits.maxPositionLots else {
            throw FXExecutionContractError.invalidField("lots must be finite, > 0, and <= max_position_lots")
        }
        guard intent.maxSlippagePoints.isFinite, intent.maxSlippagePoints >= 0 else {
            throw FXExecutionContractError.invalidField("max_slippage_points must be finite and >= 0")
        }
        guard intent.requestedAtUTC > 0 else {
            throw FXExecutionContractError.invalidField("requested_at_utc must be positive")
        }
    }
}

public enum FXExecutionContractError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedVersion(String)
    case invalidField(String)
    case killSwitchActive(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "Unsupported FXExecution contract \(version); expected \(FXExecutionContractsV1.latestVersion)."
        case .invalidField(let field):
            return "Invalid FXExecution contract field: \(field)"
        case .killSwitchActive(let reason):
            return "Execution kill switch is active: \(reason)"
        }
    }
}

private extension FXExecutionAccountScope {
    static func validateLatest(_ apiVersion: String) throws {
        guard apiVersion == FXExecutionContractsV1.latestVersion else {
            throw FXExecutionContractError.unsupportedVersion(apiVersion)
        }
    }

    static func require(_ value: String, _ field: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXExecutionContractError.invalidField("\(field) must not be empty")
        }
    }
}
