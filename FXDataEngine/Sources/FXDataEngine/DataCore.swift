import Foundation

public struct DataCoreRequest: Codable, Hashable, Sendable {
    public var liveMode: Bool
    public var symbol: String
    public var signalTimestampUTC: Int64?
    public var neededBars: Int
    public var alignUpToIndex: Int?
    public var contextSymbols: [String]

    public init(
        liveMode: Bool = false,
        symbol: String,
        signalTimestampUTC: Int64? = nil,
        neededBars: Int,
        alignUpToIndex: Int? = nil,
        contextSymbols: [String] = []
    ) {
        self.liveMode = liveMode
        self.symbol = symbol.uppercased()
        self.signalTimestampUTC = signalTimestampUTC
        self.neededBars = neededBars
        self.alignUpToIndex = alignUpToIndex
        self.contextSymbols = Array(contextSymbols.prefix(FXDataEngineConstants.maxContextSymbols)).map { $0.uppercased() }
    }
}

public struct DataCoreBundle: Sendable {
    public let ready: Bool
    public let request: DataCoreRequest
    public let universe: MarketUniverse
    public let sampleIndex: Int

    public var primary: M1OHLCVSeries { universe.primary }
    public var sampleTimeUTC: Int64 { primary.utcTimestamps[sampleIndex] }

    public init(ready: Bool, request: DataCoreRequest, universe: MarketUniverse, sampleIndex: Int) {
        self.ready = ready
        self.request = request
        self.universe = universe
        self.sampleIndex = sampleIndex
    }
}

public struct DataCore: Sendable {
    public init() {}

    public func buildBundle(request: DataCoreRequest, universe: MarketUniverse) throws -> DataCoreBundle {
        guard request.neededBars > 0 else {
            throw FXDataEngineError.invalidRequest("neededBars must be positive")
        }
        guard universe.primarySymbol == request.symbol.uppercased() else {
            throw FXDataEngineError.invalidRequest("request symbol \(request.symbol) does not match primary universe symbol \(universe.primarySymbol)")
        }
        guard universe.primary.count >= request.neededBars else {
            throw FXDataEngineError.insufficientData("primary series has \(universe.primary.count) rows, need \(request.neededBars)")
        }

        let sampleIndex: Int
        if let alignUpToIndex = request.alignUpToIndex {
            sampleIndex = min(max(alignUpToIndex, 0), universe.primary.count - 1)
        } else if let timestamp = request.signalTimestampUTC {
            guard let found = universe.primary.utcTimestamps.lastIndex(where: { $0 <= timestamp }) else {
                throw FXDataEngineError.insufficientData("no primary M1 bar at or before requested signal timestamp")
            }
            sampleIndex = found
        } else {
            sampleIndex = universe.primary.count - 1
        }

        guard sampleIndex + 1 >= request.neededBars else {
            throw FXDataEngineError.insufficientData("sample index \(sampleIndex) does not leave enough lookback for \(request.neededBars) bars")
        }

        for contextSymbol in request.contextSymbols where universe[contextSymbol] == nil {
            throw FXDataEngineError.missingSeries(symbol: contextSymbol, timeframe: .m1)
        }

        return DataCoreBundle(ready: true, request: request, universe: universe, sampleIndex: sampleIndex)
    }
}
