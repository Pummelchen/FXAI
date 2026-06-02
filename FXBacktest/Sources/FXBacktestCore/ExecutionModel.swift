import Foundation

public enum FXBacktestOrderSide: Int, Codable, Sendable {
    case buy = 1
    case sell = -1
}

public struct FXBacktestPositionV2: Codable, Hashable, Identifiable, Sendable {
    public let id: Int64
    public let symbol: String
    public let side: FXBacktestOrderSide
    public let lots: Double
    public let entryPrice: Int64
    public let openedAtUtc: Int64?
    public let digits: Int

    public init(
        id: Int64,
        symbol: String,
        side: FXBacktestOrderSide,
        lots: Double,
        entryPrice: Int64,
        openedAtUtc: Int64?,
        digits: Int
    ) {
        self.id = id
        self.symbol = symbol.uppercased()
        self.side = side
        self.lots = lots
        self.entryPrice = entryPrice
        self.openedAtUtc = openedAtUtc
        self.digits = digits
    }
}

public struct FXBacktestTradeLedgerEntry: Codable, Hashable, Identifiable, Sendable {
    public let id: Int64
    public let positionId: Int64
    public let symbol: String
    public let side: FXBacktestOrderSide
    public let lots: Double
    public let entryPrice: Int64
    public let exitPrice: Int64
    public let openedAtUtc: Int64?
    public let closedAtUtc: Int64?
    public let grossProfit: Double
    public let netProfit: Double
    public let balanceAfter: Double

    public init(
        id: Int64,
        positionId: Int64,
        symbol: String,
        side: FXBacktestOrderSide,
        lots: Double,
        entryPrice: Int64,
        exitPrice: Int64,
        openedAtUtc: Int64?,
        closedAtUtc: Int64?,
        grossProfit: Double,
        netProfit: Double,
        balanceAfter: Double
    ) {
        self.id = id
        self.positionId = positionId
        self.symbol = symbol.uppercased()
        self.side = side
        self.lots = lots
        self.entryPrice = entryPrice
        self.exitPrice = exitPrice
        self.openedAtUtc = openedAtUtc
        self.closedAtUtc = closedAtUtc
        self.grossProfit = grossProfit
        self.netProfit = netProfit
        self.balanceAfter = balanceAfter
    }
}

public struct BacktestBrokerV2: Sendable {
    public private(set) var balance: Double
    public private(set) var equity: Double
    public private(set) var equityPeak: Double
    public private(set) var maxDrawdown: Double
    public private(set) var grossProfit: Double
    public private(set) var grossLoss: Double
    public private(set) var totalTrades: Int
    public private(set) var winningTrades: Int
    public private(set) var losingTrades: Int
    public private(set) var positions: [FXBacktestPositionV2]
    public private(set) var ledger: [FXBacktestTradeLedgerEntry]

    private let context: BacktestContext
    private var currentClosePrices: [String: Int64]
    private var nextPositionId: Int64
    private var nextLedgerId: Int64

    public init(context: BacktestContext) {
        self.context = context
        self.balance = context.settings.initialDeposit
        self.equity = context.settings.initialDeposit
        self.equityPeak = context.settings.initialDeposit
        self.maxDrawdown = 0
        self.grossProfit = 0
        self.grossLoss = 0
        self.totalTrades = 0
        self.winningTrades = 0
        self.losingTrades = 0
        self.positions = []
        self.ledger = []
        self.currentClosePrices = [:]
        self.nextPositionId = 1
        self.nextLedgerId = 1
    }

    public mutating func openMarket(
        symbol: String,
        side: FXBacktestOrderSide,
        midPrice: Int64,
        lots: Double? = nil,
        openedAtUtc: Int64? = nil,
        digits: Int? = nil
    ) throws -> Int64 {
        let normalizedSymbol = symbol.uppercased()
        let resolvedLots = lots ?? context.settings.lotSize
        let resolvedDigits = try Self.validateDigits(digits ?? context.digits, symbol: normalizedSymbol)
        try Self.validate(price: midPrice, symbol: normalizedSymbol)
        guard resolvedLots.isFinite, resolvedLots > 0 else {
            throw FXBacktestError.invalidParameter("\(normalizedSymbol): lot size must be a finite value > 0.")
        }

        let position = FXBacktestPositionV2(
            id: nextPositionId,
            symbol: normalizedSymbol,
            side: side,
            lots: resolvedLots,
            entryPrice: midPrice,
            openedAtUtc: openedAtUtc,
            digits: resolvedDigits
        )
        nextPositionId += 1
        positions.append(position)
        currentClosePrices[normalizedSymbol] = midPrice
        recomputeEquity()
        return position.id
    }

    @discardableResult
    public mutating func closePosition(
        id: Int64,
        midPrice: Int64,
        closedAtUtc: Int64? = nil,
        digits: Int? = nil
    ) throws -> Bool {
        guard let index = positions.firstIndex(where: { $0.id == id }) else { return false }
        let position = positions[index]
        if let digits {
            let closeDigits = try Self.validateDigits(digits, symbol: position.symbol)
            guard closeDigits == position.digits else {
                throw FXBacktestError.invalidParameter("\(position.symbol): close digits \(closeDigits) do not match position digits \(position.digits).")
            }
        }
        try Self.validate(price: midPrice, symbol: position.symbol)
        positions.remove(at: index)

        let pnl = Self.profit(
            position: position,
            exitPrice: midPrice,
            contractSize: context.settings.contractSize
        )
        balance += pnl
        grossProfit += max(0, pnl)
        grossLoss += min(0, pnl)
        totalTrades += 1
        if pnl > 0 {
            winningTrades += 1
        } else if pnl < 0 {
            losingTrades += 1
        }
        ledger.append(FXBacktestTradeLedgerEntry(
            id: nextLedgerId,
            positionId: position.id,
            symbol: position.symbol,
            side: position.side,
            lots: position.lots,
            entryPrice: position.entryPrice,
            exitPrice: midPrice,
            openedAtUtc: position.openedAtUtc,
            closedAtUtc: closedAtUtc,
            grossProfit: pnl,
            netProfit: pnl,
            balanceAfter: balance
        ))
        nextLedgerId += 1
        currentClosePrices[position.symbol] = midPrice
        recomputeEquity()
        return true
    }

    public mutating func closeAll(
        midPrices: [String: Int64],
        closedAtUtc: Int64? = nil,
        digitsBySymbol: [String: Int] = [:]
    ) throws {
        let openIds = positions.map(\.id)
        for id in openIds {
            guard let position = positions.first(where: { $0.id == id }) else { continue }
            let closePrice = midPrices[position.symbol] ?? currentClosePrices[position.symbol] ?? position.entryPrice
            try closePosition(
                id: id,
                midPrice: closePrice,
                closedAtUtc: closedAtUtc,
                digits: digitsBySymbol[position.symbol]
            )
        }
    }

    public mutating func markToMarket(symbol: String, midPrice: Int64) {
        guard midPrice > 0 else { return }
        currentClosePrices[symbol.uppercased()] = midPrice
        recomputeEquity()
    }

    public var netProfit: Double {
        balance - context.settings.initialDeposit
    }

    public var equityNetProfit: Double {
        equity - context.settings.initialDeposit
    }

    public var winRate: Double {
        totalTrades == 0 ? 0 : Double(winningTrades) / Double(totalTrades)
    }

    public var profitFactor: Double {
        grossLoss == 0 ? (grossProfit > 0 ? Double.infinity : 0) : grossProfit / abs(grossLoss)
    }

    private mutating func recomputeEquity() {
        var floating = 0.0
        for position in positions {
            let closePrice = currentClosePrices[position.symbol] ?? position.entryPrice
            floating += Self.profit(
                position: position,
                exitPrice: closePrice,
                contractSize: context.settings.contractSize
            )
        }
        equity = balance + floating
        equityPeak = max(equityPeak, equity)
        maxDrawdown = max(maxDrawdown, equityPeak - equity)
    }

    private static func profit(
        position: FXBacktestPositionV2,
        exitPrice: Int64,
        contractSize: Double
    ) -> Double {
        let direction = Double(position.side.rawValue)
        let priceScale = pow(10.0, Double(position.digits))
        let priceDelta = Double(exitPrice - position.entryPrice) / priceScale
        return direction * priceDelta * contractSize * position.lots
    }

    private static func validate(price: Int64, symbol: String) throws {
        guard price > 0 else {
            throw FXBacktestError.invalidParameter("\(symbol): OHLC close price must be > 0.")
        }
    }

    private static func validateDigits(_ digits: Int, symbol: String) throws -> Int {
        guard (0...10).contains(digits) else {
            throw FXBacktestError.invalidParameter("\(symbol): digits must be in 0...10.")
        }
        return digits
    }
}
