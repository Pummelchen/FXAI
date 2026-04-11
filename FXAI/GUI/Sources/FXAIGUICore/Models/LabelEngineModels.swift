import Foundation

public struct LabelEngineReasonCount: Identifiable, Hashable, Sendable {
    public let id: String
    public let reason: String
    public let count: Int

    public init(reason: String, count: Int) {
        id = reason
        self.reason = reason
        self.count = count
    }
}

public struct LabelEngineHorizonSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let horizonID: String
    public let bars: Int
    public let sampleCount: Int
    public let longTradeabilityRate: Double
    public let shortTradeabilityRate: Double
    public let candidateCount: Int
    public let candidateAcceptanceRate: Double
    public let meanCostAdjustedReturnPoints: Double
    public let medianTimeToFavorableHitSec: Double?

    public init(
        horizonID: String,
        bars: Int,
        sampleCount: Int,
        longTradeabilityRate: Double,
        shortTradeabilityRate: Double,
        candidateCount: Int,
        candidateAcceptanceRate: Double,
        meanCostAdjustedReturnPoints: Double,
        medianTimeToFavorableHitSec: Double?
    ) {
        id = horizonID
        self.horizonID = horizonID
        self.bars = bars
        self.sampleCount = sampleCount
        self.longTradeabilityRate = longTradeabilityRate
        self.shortTradeabilityRate = shortTradeabilityRate
        self.candidateCount = candidateCount
        self.candidateAcceptanceRate = candidateAcceptanceRate
        self.meanCostAdjustedReturnPoints = meanCostAdjustedReturnPoints
        self.medianTimeToFavorableHitSec = medianTimeToFavorableHitSec
    }
}

public struct LabelEngineBuildSnapshot: Identifiable, Hashable, Sendable {
    public let id: String
    public let datasetKey: String
    public let profileName: String
    public let symbol: String
    public let timeframe: String
    public let barCount: Int
    public let pointSize: Double
    public let executionProfile: String
    public let labelVersion: Int
    public let generatedAt: Date?
    public let summaryMetrics: [KeyValueRecord]
    public let metaSummary: [KeyValueRecord]
    public let qualityFlags: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]
    public let topReasons: [LabelEngineReasonCount]
    public let horizons: [LabelEngineHorizonSnapshot]

    public init(
        datasetKey: String,
        profileName: String,
        symbol: String,
        timeframe: String,
        barCount: Int,
        pointSize: Double,
        executionProfile: String,
        labelVersion: Int,
        generatedAt: Date?,
        summaryMetrics: [KeyValueRecord],
        metaSummary: [KeyValueRecord],
        qualityFlags: [KeyValueRecord],
        artifactPaths: [KeyValueRecord],
        topReasons: [LabelEngineReasonCount],
        horizons: [LabelEngineHorizonSnapshot]
    ) {
        id = "\(profileName)::\(datasetKey)"
        self.datasetKey = datasetKey
        self.profileName = profileName
        self.symbol = symbol
        self.timeframe = timeframe
        self.barCount = barCount
        self.pointSize = pointSize
        self.executionProfile = executionProfile
        self.labelVersion = labelVersion
        self.generatedAt = generatedAt
        self.summaryMetrics = summaryMetrics
        self.metaSummary = metaSummary
        self.qualityFlags = qualityFlags
        self.artifactPaths = artifactPaths
        self.topReasons = topReasons
        self.horizons = horizons
    }
}

public struct LabelEngineSnapshot: Sendable {
    public let generatedAt: Date
    public let artifactCount: Int
    public let latestDatasetKey: String
    public let builds: [LabelEngineBuildSnapshot]
    public let statusRecords: [KeyValueRecord]
    public let artifactPaths: [KeyValueRecord]

    public init(
        generatedAt: Date,
        artifactCount: Int,
        latestDatasetKey: String,
        builds: [LabelEngineBuildSnapshot],
        statusRecords: [KeyValueRecord],
        artifactPaths: [KeyValueRecord]
    ) {
        self.generatedAt = generatedAt
        self.artifactCount = artifactCount
        self.latestDatasetKey = latestDatasetKey
        self.builds = builds
        self.statusRecords = statusRecords
        self.artifactPaths = artifactPaths
    }
}
