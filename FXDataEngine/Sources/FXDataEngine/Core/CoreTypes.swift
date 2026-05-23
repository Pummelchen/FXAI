import Foundation

public enum LabelClass: Int, Codable, Sendable, CaseIterable {
    case sell = 0
    case buy = 1
    case skip = 2
}

public enum AIModelID: Int, Codable, Sendable, CaseIterable {
    case autoformer = 0
    case catboost
    case chronos
    case enhash
    case ftrlLogit
    case geodesicAttention
    case lightgbm
    case lstm
    case lstmg
    case mlpTiny
    case paLinear
    case patchTST
    case quantile
    case s4
    case sgdLogit
    case stmn
    case tcn
    case tft
    case timesfm
    case tst
    case xgbFast
    case xgboost
    case cfxWorld
    case loffm
    case trr
    case graphWM
    case moeConformal
    case retrDiff
    case m1Sync
    case buyOnly
    case sellOnly
    case randomNoSkip
    case qcew
    case fewc
    case gha
    case tesseract
    case statMSGARCH
    case statARIMAXGARCH
    case treeRF
    case statCointVECM
    case statOUSpread
    case rlPPO
    case statMicroflowProxy
    case statHMMRegime
    case linElasticLogit
    case linProfitLogit
    case cnnLSTM
    case attnCNNBiLSTM
    case statEMDHHT
    case statVMD
    case statTVPKalman
    case factorPCAPanel
    case factorPPPValue
    case factorCarry
    case factorCMVPanel
    case trendTSMOMVol
    case trendXSMOMRank
    case trendVolBreakout
    case statXRateConsistency
    case gru
    case bilstm
    case lstmTCN
    case mythosRDT
    case demoMovingAverageCross
    case demoFXStupid
    case demoFX7

    public var usesDeepNormalizationCandidates: Bool {
        switch self {
        case .lstm, .lstmg, .tcn, .tft, .tst,
             .autoformer, .patchTST, .stmn, .s4,
             .chronos, .timesfm, .geodesicAttention,
             .qcew, .fewc, .gha, .tesseract:
            true
        default:
            false
        }
    }
}

public enum AIFamily: Int, Codable, Sendable, CaseIterable {
    case linear = 0
    case tree
    case recurrent
    case convolutional
    case transformer
    case stateSpace
    case distributional
    case mixture
    case retrieval
    case worldModel
    case ruleBased
    case other
}

public enum ReferenceTier: Int, Codable, Sendable, CaseIterable {
    case fullNative = 0
    case compressedNative
    case surrogate
    case ruleBaseline
}

public enum SequenceStyle: Int, Codable, Sendable, CaseIterable {
    case generic = 0
    case recurrent
    case convolutional
    case transformer
    case stateSpace
    case world
}

public struct PluginCapability: OptionSet, Codable, Sendable, Hashable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public static let onlineLearning = PluginCapability(rawValue: 1)
    public static let replay = PluginCapability(rawValue: 2)
    public static let stateful = PluginCapability(rawValue: 4)
    public static let windowContext = PluginCapability(rawValue: 8)
    public static let multiHorizon = PluginCapability(rawValue: 16)
    public static let nativeDistribution = PluginCapability(rawValue: 32)
    public static let selfTest = PluginCapability(rawValue: 64)
}

public enum FeatureGroup: Int, Codable, Sendable, CaseIterable {
    case price = 0
    case multiTimeframe
    case volatility
    case time
    case context
    case volume
    case microstructure
    case filters

    public var name: String {
        switch self {
        case .price: "price"
        case .multiTimeframe: "multi_timeframe"
        case .volatility: "volatility"
        case .time: "time_calendar"
        case .context: "context"
        case .volume: "volume"
        case .microstructure: "microstructure"
        case .filters: "filters"
        }
    }
}

public struct FeatureGroupMask: OptionSet, Codable, Sendable, Hashable {
    public let rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }

    public init(group: FeatureGroup) {
        self.rawValue = 1 << UInt64(group.rawValue)
    }

    public static let price = FeatureGroupMask(group: .price)
    public static let multiTimeframe = FeatureGroupMask(group: .multiTimeframe)
    public static let volatility = FeatureGroupMask(group: .volatility)
    public static let time = FeatureGroupMask(group: .time)
    public static let context = FeatureGroupMask(group: .context)
    public static let volume = FeatureGroupMask(group: .volume)
    public static let microstructure = FeatureGroupMask(group: .microstructure)
    public static let filters = FeatureGroupMask(group: .filters)
    public static let all: FeatureGroupMask = [.price, .multiTimeframe, .volatility, .time, .context, .volume, .microstructure, .filters]
}

public enum FeatureSchema: Int, Codable, Sendable, CaseIterable {
    case full = 1
    case sparseStat = 2
    case sequence = 3
    case rule = 4
    case tree = 5
    case contextual = 6
}

public enum FeatureNormalizationMethod: Int, Codable, Sendable, CaseIterable {
    case existing = 0
    case minMaxBuffer5
    case changePercent
    case binary01
    case logReturn
    case relativeChangePercent
    case candleGeometry
    case volatilityStdReturns
    case atrNatrUnit
    case zScore
    case robustMedianIQR
    case quantileToNormal
    case powerYeoJohnson
    case revin
    case dain
    case minMaxBuffer2
    case minMaxBuffer3

    public var needsPrevious: Bool {
        switch self {
        case .changePercent, .binary01, .logReturn, .relativeChangePercent, .candleGeometry:
            true
        default:
            false
        }
    }
}

public enum MTFStateMetric: Int, Codable, Sendable, CaseIterable {
    case bodyBias = 0
    case closeLocation
    case rangePressure
    case volumePressure
}

public enum FeatureProvenance: String, Codable, Sendable, CaseIterable {
    case priceBar = "price_bar"
    case multiTimeframe = "multi_timeframe"
    case contextSymbol = "context_symbol"
    case timeCalendar = "time_calendar"
    case symbolContract = "symbol_contract"
    case eventMacro = "event_macro"
    case derivedFilter = "derived_filter"
}

public enum FXDataEngineError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidRequest(String)
    case insufficientData(String)
    case missingSeries(symbol: String, timeframe: MarketTimeframe)
    case validation(String)
    case externalBackend(String)

    public var description: String {
        switch self {
        case .invalidRequest(let reason): "invalid request: \(reason)"
        case .insufficientData(let reason): "insufficient data: \(reason)"
        case .missingSeries(let symbol, let timeframe): "missing series: \(symbol) \(timeframe.rawValue)"
        case .validation(let reason): "validation failed: \(reason)"
        case .externalBackend(let reason): "external backend failed: \(reason)"
        }
    }
}

public enum MarketTimeframe: String, Codable, Sendable, CaseIterable, Comparable {
    case m1 = "M1"
    case m5 = "M5"
    case m15 = "M15"
    case m30 = "M30"
    case h1 = "H1"

    public var minutes: Int {
        switch self {
        case .m1: 1
        case .m5: 5
        case .m15: 15
        case .m30: 30
        case .h1: 60
        }
    }

    public var seconds: TimeInterval {
        TimeInterval(minutes * 60)
    }

    public static func < (lhs: MarketTimeframe, rhs: MarketTimeframe) -> Bool {
        lhs.minutes < rhs.minutes
    }
}
