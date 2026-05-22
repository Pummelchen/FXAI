import Foundation

public enum RuntimeArtifactSectionOwner: String, Codable, Hashable, Sendable {
    case fxDataEngine
    case fxPlugins
    case fxBacktest
}

public enum RuntimeArtifactPayloadSectionKind: String, Codable, CaseIterable, Sendable {
    case normalizationWindows
    case normalizationWindowConfig
    case normalizationHistory
    case normalizationFit
    case replayReservoir
    case conformalCalibration
    case featureDrift
    case sharedTransferTensor
    case foundationTensor
    case studentTensor
    case analogMemory
    case brokerExecution

    public var owner: RuntimeArtifactSectionOwner {
        switch self {
        case .sharedTransferTensor, .foundationTensor, .studentTensor:
            .fxPlugins
        case .brokerExecution:
            .fxBacktest
        default:
            .fxDataEngine
        }
    }
}

public struct RuntimeArtifactPayloadSection: Codable, Hashable, Sendable {
    public var kind: RuntimeArtifactPayloadSectionKind
    public var owner: RuntimeArtifactSectionOwner
    public var offset: Int
    public var byteCount: Int

    public init(
        kind: RuntimeArtifactPayloadSectionKind,
        owner: RuntimeArtifactSectionOwner? = nil,
        offset: Int,
        byteCount: Int
    ) {
        self.kind = kind
        self.owner = owner ?? kind.owner
        self.offset = max(0, offset)
        self.byteCount = max(0, byteCount)
    }

    public var range: Range<Int> {
        offset..<(offset + byteCount)
    }
}

public struct RuntimeArtifactPayloadLayout: Codable, Hashable, Sendable {
    public var header: RuntimeArtifactHeader
    public var sections: [RuntimeArtifactPayloadSection]

    public init(header: RuntimeArtifactHeader, sections: [RuntimeArtifactPayloadSection]) {
        self.header = header
        self.sections = sections
    }

    public var payloadByteCount: Int {
        sections.last.map { $0.offset + $0.byteCount } ?? 0
    }

    public func section(_ kind: RuntimeArtifactPayloadSectionKind) -> RuntimeArtifactPayloadSection? {
        sections.first { $0.kind == kind }
    }

    public func slice(_ payload: Data, kind: RuntimeArtifactPayloadSectionKind) throws -> Data {
        guard let section = section(kind) else {
            throw FXDataEngineError.validation("runtime artifact section is not present: \(kind.rawValue)")
        }
        guard payload.count >= section.offset + section.byteCount else {
            throw FXDataEngineError.validation(
                "runtime artifact payload is \(payload.count) bytes, need \(section.offset + section.byteCount)"
            )
        }
        return payload.subdata(in: section.range)
    }
}

public enum RuntimeArtifactPayloadMaterializer {
    public static func layout(header: RuntimeArtifactHeader = RuntimeArtifactHeader()) -> RuntimeArtifactPayloadLayout {
        var offset = 0
        var sections: [RuntimeArtifactPayloadSection] = []

        func append(_ kind: RuntimeArtifactPayloadSectionKind, _ byteCount: Int) {
            sections.append(RuntimeArtifactPayloadSection(kind: kind, offset: offset, byteCount: byteCount))
            offset += byteCount
        }

        append(.normalizationWindows, normalizationWindowsByteCount(header: header))
        append(.normalizationWindowConfig, normalizationWindowConfigByteCount(header: header))
        append(.normalizationHistory, normalizationHistoryByteCount(header: header))
        append(.normalizationFit, normalizationFitByteCount(header: header))
        append(.replayReservoir, replayReservoirByteCount(header: header))
        append(.conformalCalibration, conformalCalibrationByteCount(header: header))
        append(.featureDrift, featureDriftByteCount)
        append(.sharedTransferTensor, sharedTransferTensorByteCount)
        append(.foundationTensor, foundationTensorByteCount)
        append(.studentTensor, studentTensorByteCount)
        append(.analogMemory, analogMemoryByteCount)
        append(.brokerExecution, brokerExecutionByteCount)

        return RuntimeArtifactPayloadLayout(header: header, sections: sections)
    }

    public static let intByteCount = MemoryLayout<Int32>.size
    public static let longByteCount = MemoryLayout<Int64>.size
    public static let doubleByteCount = MemoryLayout<Double>.size
    public static let normalizationQuantileKnots = 17

    public static func preparedSampleByteCount(featureWeights: Int = FXDataEngineConstants.aiWeights) -> Int {
        (6 * intByteCount) + (23 * doubleByteCount) + longByteCount + (featureWeights * doubleByteCount)
    }

    public static func normalizationWindowsByteCount(header: RuntimeArtifactHeader) -> Int {
        (2 + header.featureCount) * intByteCount
    }

    public static func normalizationWindowConfigByteCount(header: RuntimeArtifactHeader) -> Int {
        (3 + header.featureCount) * intByteCount
    }

    public static func normalizationHistoryByteCount(header: RuntimeArtifactHeader) -> Int {
        intByteCount +
            header.normalizationMethodCount *
            (longByteCount + intByteCount +
                header.featureCount *
                (2 * intByteCount + header.normalizationRollWindowMax * doubleByteCount))
    }

    public static func normalizationFitByteCount(header: RuntimeArtifactHeader) -> Int {
        intByteCount +
            header.maxHorizons *
            header.normalizationMethodCount *
            (2 * intByteCount +
                header.featureCount *
                ((9 + normalizationQuantileKnots) * doubleByteCount))
    }

    public static func replayReservoirByteCount(header: RuntimeArtifactHeader) -> Int {
        (2 * intByteCount) +
            header.maxHorizons * longByteCount +
            header.regimeCount * header.maxHorizons * intByteCount +
            header.replayCapacity *
            (2 * intByteCount + doubleByteCount + preparedSampleByteCount(featureWeights: header.featureCount + 1))
    }

    public static func conformalCalibrationByteCount(header: RuntimeArtifactHeader) -> Int {
        let scoreCellBytes = 2 * intByteCount + header.conformalDepth * 3 * doubleByteCount
        let scoreBytes = header.aiCount * header.regimeCount * header.maxHorizons * scoreCellBytes
        let pendingBytesPerAI = 2 * intByteCount +
            header.reliabilityPendingCapacity *
            (3 * intByteCount + 7 * doubleByteCount)
        return scoreBytes + header.aiCount * pendingBytesPerAI
    }

    public static let featureDriftByteCount: Int =
        intByteCount + longByteCount + FeatureGroup.allCases.count * (2 * intByteCount + 5 * doubleByteCount)

    public static let sharedTransferTensorByteCount: Int = {
        let latent = FXDataEngineConstants.sharedTransferLatent
        let perLatent = 2 * doubleByteCount +
            3 * doubleByteCount +
            FXDataEngineConstants.sharedTransferFeatures * doubleByteCount +
            FXDataEngineConstants.sharedTransferSequenceTokens * doubleByteCount +
            FXDataEngineConstants.sharedTransferBarFeatures * doubleByteCount +
            FXDataEngineConstants.sharedTransferBarFeatures * doubleByteCount
        return 2 * intByteCount +
            latent * perLatent +
            FXDataEngineConstants.sharedTransferDomainBuckets * latent * doubleByteCount +
            FXDataEngineConstants.pluginHorizonBuckets * latent * doubleByteCount +
            FXDataEngineConstants.pluginSessionBuckets * latent * doubleByteCount +
            latent * (2 * doubleByteCount + FXDataEngineConstants.sharedTransferStateFeatures * doubleByteCount)
    }()

    public static let foundationTensorByteCount: Int =
        2 * intByteCount +
        FXDataEngineConstants.sharedTransferLatent * 4 * doubleByteCount +
        4 * doubleByteCount

    public static let studentTensorByteCount: Int =
        2 * intByteCount +
        3 * (doubleByteCount + FXDataEngineConstants.sharedTransferLatent * doubleByteCount) +
        3 * doubleByteCount +
        FXDataEngineConstants.sharedTransferLatent * 3 * doubleByteCount

    public static let analogMemoryByteCount: Int =
        3 * intByteCount +
        FXDataEngineConstants.analogMemoryCapacity *
        (longByteCount +
            3 * intByteCount +
            7 * doubleByteCount +
            FXDataEngineConstants.analogMemoryFeatures * doubleByteCount)

    public static let brokerExecutionLibraryCells: Int =
        FXDataEngineConstants.brokerExecutionSymbolBuckets *
        FXDataEngineConstants.pluginSessionBuckets *
        FXDataEngineConstants.pluginHorizonBuckets *
        FXDataEngineConstants.brokerExecutionSideCount *
        FXDataEngineConstants.brokerExecutionOrderTypeCount

    public static let brokerExecutionByteCount: Int =
        intByteCount +
        FXDataEngineConstants.pluginSessionBuckets * FXDataEngineConstants.pluginHorizonBuckets * 5 * doubleByteCount +
        brokerExecutionLibraryCells * 6 * doubleByteCount +
        brokerExecutionLibraryCells * FXDataEngineConstants.brokerExecutionEventKindCount * doubleByteCount +
        2 * intByteCount +
        FXDataEngineConstants.brokerExecutionTraceCapacity *
        (longByteCount + 6 * intByteCount + 5 * doubleByteCount)
}

public struct RuntimeArtifactBinaryWriter {
    public private(set) var data: Data

    public init(data: Data = Data()) {
        self.data = data
    }

    public mutating func appendInt32(_ value: Int) throws {
        guard value >= Int(Int32.min), value <= Int(Int32.max) else {
            throw FXDataEngineError.validation("runtime artifact int32 value exceeds range")
        }
        var raw = Int32(value).littleEndian
        withUnsafeBytes(of: &raw) {
            data.append(contentsOf: $0)
        }
    }

    public mutating func appendInt64(_ value: Int64) {
        var raw = value.littleEndian
        withUnsafeBytes(of: &raw) {
            data.append(contentsOf: $0)
        }
    }

    public mutating func appendDouble(_ value: Double) {
        var raw = fxSafeFinite(value).bitPattern.littleEndian
        withUnsafeBytes(of: &raw) {
            data.append(contentsOf: $0)
        }
    }

    public mutating func appendZeroes(byteCount: Int) {
        guard byteCount > 0 else { return }
        data.append(Data(count: byteCount))
    }
}

public struct RuntimeArtifactBinaryReader {
    public let data: Data
    public private(set) var offset: Int

    public init(data: Data, offset: Int = 0) {
        self.data = data
        self.offset = max(0, offset)
    }

    public var remainingByteCount: Int {
        max(0, data.count - offset)
    }

    public mutating func readInt32() throws -> Int {
        try require(RuntimeArtifactPayloadMaterializer.intByteCount)
        let bytes = [UInt8](data[offset..<(offset + RuntimeArtifactPayloadMaterializer.intByteCount)])
        offset += RuntimeArtifactPayloadMaterializer.intByteCount
        var raw = UInt32(bytes[0])
        raw |= UInt32(bytes[1]) << 8
        raw |= UInt32(bytes[2]) << 16
        raw |= UInt32(bytes[3]) << 24
        return Int(Int32(bitPattern: raw))
    }

    public mutating func readInt64() throws -> Int64 {
        try require(RuntimeArtifactPayloadMaterializer.longByteCount)
        let bytes = [UInt8](data[offset..<(offset + RuntimeArtifactPayloadMaterializer.longByteCount)])
        offset += RuntimeArtifactPayloadMaterializer.longByteCount
        var raw = UInt64(0)
        for index in 0..<RuntimeArtifactPayloadMaterializer.longByteCount {
            raw |= UInt64(bytes[index]) << UInt64(index * 8)
        }
        return Int64(bitPattern: raw)
    }

    public mutating func readDouble() throws -> Double {
        try require(RuntimeArtifactPayloadMaterializer.doubleByteCount)
        let bytes = [UInt8](data[offset..<(offset + RuntimeArtifactPayloadMaterializer.doubleByteCount)])
        offset += RuntimeArtifactPayloadMaterializer.doubleByteCount
        var raw = UInt64(0)
        for index in 0..<RuntimeArtifactPayloadMaterializer.doubleByteCount {
            raw |= UInt64(bytes[index]) << UInt64(index * 8)
        }
        return fxSafeFinite(Double(bitPattern: raw))
    }

    public mutating func skip(byteCount: Int) throws {
        guard byteCount >= 0 else {
            throw FXDataEngineError.validation("runtime artifact cannot skip a negative byte count")
        }
        try require(byteCount)
        offset += byteCount
    }

    private func require(_ byteCount: Int) throws {
        guard offset + byteCount <= data.count else {
            throw FXDataEngineError.validation("runtime artifact binary read exceeds section bounds")
        }
    }
}

public struct RuntimeArtifactPreparedSample: Codable, Hashable, Sendable {
    public var valid: Bool
    public var labelClass: LabelClass
    public var regimeID: Int
    public var horizonMinutes: Int
    public var horizonSlot: Int
    public var movePoints: Double
    public var minMovePoints: Double
    public var costPoints: Double
    public var sampleWeight: Double
    public var qualityScore: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var spreadStress: Double
    public var traceSpreadMeanRatio: Double
    public var traceSpreadPeakRatio: Double
    public var traceRangeMeanRatio: Double
    public var traceBodyEfficiency: Double
    public var traceGapRatio: Double
    public var traceReversalRatio: Double
    public var traceSessionTransition: Double
    public var traceRollover: Double
    public var timeToHitFraction: Double
    public var pathFlags: SamplePathFlags
    public var maskedStepTarget: Double
    public var nextVolumeTarget: Double
    public var regimeShiftTarget: Double
    public var contextLeadTarget: Double
    public var pointValue: Double
    public var domainHash: Double
    public var sampleTimeUTC: Int64
    public var x: [Double]

    public init(
        valid: Bool = false,
        labelClass: LabelClass = .skip,
        regimeID: Int = 0,
        horizonMinutes: Int = 1,
        horizonSlot: Int = 0,
        movePoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        costPoints: Double = 0.0,
        sampleWeight: Double = 1.0,
        qualityScore: Double = 1.0,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        spreadStress: Double = 0.0,
        traceSpreadMeanRatio: Double = 0.0,
        traceSpreadPeakRatio: Double = 0.0,
        traceRangeMeanRatio: Double = 1.0,
        traceBodyEfficiency: Double = 0.5,
        traceGapRatio: Double = 0.0,
        traceReversalRatio: Double = 0.0,
        traceSessionTransition: Double = 0.0,
        traceRollover: Double = 0.0,
        timeToHitFraction: Double = 1.0,
        pathFlags: SamplePathFlags = [],
        maskedStepTarget: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeShiftTarget: Double = 0.0,
        contextLeadTarget: Double = 0.5,
        pointValue: Double = 1.0,
        domainHash: Double = 0.0,
        sampleTimeUTC: Int64 = 0,
        x: [Double] = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
    ) {
        self.valid = valid
        self.labelClass = labelClass
        self.regimeID = Int(fxClamp(Double(regimeID), 0.0, Double(FXDataEngineConstants.pluginRegimeBuckets - 1)))
        self.horizonMinutes = TrainingSampleTools.clampHorizon(horizonMinutes)
        self.horizonSlot = Int(fxClamp(Double(horizonSlot), 0.0, Double(RuntimeArtifactConstants.maxHorizons - 1)))
        self.movePoints = fxSafeFinite(movePoints)
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.costPoints = max(0.0, fxSafeFinite(costPoints))
        self.sampleWeight = fxClamp(sampleWeight, 0.0, 10.0)
        self.qualityScore = fxClamp(qualityScore, 0.0, 4.0)
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.spreadStress = max(0.0, fxSafeFinite(spreadStress))
        self.traceSpreadMeanRatio = max(0.0, fxSafeFinite(traceSpreadMeanRatio))
        self.traceSpreadPeakRatio = max(0.0, fxSafeFinite(traceSpreadPeakRatio))
        self.traceRangeMeanRatio = max(0.0, fxSafeFinite(traceRangeMeanRatio))
        self.traceBodyEfficiency = fxClamp(traceBodyEfficiency, 0.0, 1.0)
        self.traceGapRatio = fxClamp(traceGapRatio, 0.0, 1.0)
        self.traceReversalRatio = fxClamp(traceReversalRatio, 0.0, 1.0)
        self.traceSessionTransition = fxClamp(traceSessionTransition, 0.0, 1.0)
        self.traceRollover = fxClamp(traceRollover, 0.0, 1.0)
        self.timeToHitFraction = fxClamp(timeToHitFraction, 0.0, 1.0)
        self.pathFlags = pathFlags
        self.maskedStepTarget = fxSafeFinite(maskedStepTarget)
        self.nextVolumeTarget = max(0.0, fxSafeFinite(nextVolumeTarget))
        self.regimeShiftTarget = fxClamp(regimeShiftTarget, 0.0, 1.0)
        self.contextLeadTarget = fxClamp(contextLeadTarget, 0.0, 1.0)
        self.pointValue = pointValue > 0.0 ? pointValue : 1.0
        self.domainHash = fxClamp(domainHash, 0.0, 1.0)
        self.sampleTimeUTC = max(0, sampleTimeUTC)
        self.x = Self.artifactVector(x)
    }

    public init(sample: PreparedTrainingSample) {
        self.init(
            valid: sample.valid,
            labelClass: sample.labelClass,
            regimeID: sample.regimeID,
            horizonMinutes: sample.horizonMinutes,
            horizonSlot: sample.horizonSlot,
            movePoints: sample.movePoints,
            minMovePoints: sample.minMovePoints,
            costPoints: sample.costPoints,
            sampleWeight: sample.sampleWeight,
            qualityScore: sample.qualityScore,
            mfePoints: sample.mfePoints,
            maePoints: sample.maePoints,
            spreadStress: 0.0,
            traceSpreadMeanRatio: 2.0 * sample.fillRisk,
            traceSpreadPeakRatio: 0.0,
            traceRangeMeanRatio: 1.0,
            traceBodyEfficiency: 0.5,
            traceGapRatio: sample.pathRisk,
            traceReversalRatio: sample.pathRisk,
            timeToHitFraction: sample.timeToHitFraction,
            pathFlags: sample.pathFlags,
            maskedStepTarget: sample.maskedStepTarget,
            nextVolumeTarget: sample.nextVolumeTarget,
            regimeShiftTarget: sample.regimeShiftTarget,
            contextLeadTarget: sample.contextLeadTarget,
            pointValue: sample.pointValue,
            domainHash: sample.domainHash,
            sampleTimeUTC: sample.sampleTimeUTC,
            x: sample.x
        )
    }

    public static func artifactVector(_ vector: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<min(vector.count, FXDataEngineConstants.aiWeights) {
            output[index] = fxSafeFinite(vector[index])
        }
        return output
    }

    public var preparedTrainingSample: PreparedTrainingSample {
        PreparedTrainingSample(
            valid: valid,
            labelClass: labelClass,
            regimeID: regimeID,
            horizonMinutes: horizonMinutes,
            horizonSlot: horizonSlot,
            movePoints: movePoints,
            minMovePoints: minMovePoints,
            costPoints: costPoints,
            sampleWeight: sampleWeight,
            qualityScore: qualityScore,
            mfePoints: mfePoints,
            maePoints: maePoints,
            timeToHitFraction: timeToHitFraction,
            pathFlags: pathFlags,
            pathRisk: fxClamp(0.5 * traceReversalRatio + 0.5 * traceGapRatio, 0.0, 1.0),
            fillRisk: fxClamp(traceSpreadMeanRatio / 2.0, 0.0, 1.0),
            maskedStepTarget: maskedStepTarget,
            nextVolumeTarget: nextVolumeTarget,
            regimeShiftTarget: regimeShiftTarget,
            contextLeadTarget: contextLeadTarget,
            pointValue: pointValue,
            domainHash: domainHash,
            sampleTimeUTC: sampleTimeUTC,
            x: x
        )
    }
}

public enum RuntimeArtifactPreparedSampleCodec {
    public static let byteCount = RuntimeArtifactPayloadMaterializer.preparedSampleByteCount()

    public static func encode(_ sample: RuntimeArtifactPreparedSample) throws -> Data {
        var writer = RuntimeArtifactBinaryWriter()
        try encode(sample, into: &writer)
        return writer.data
    }

    public static func encode(_ sample: RuntimeArtifactPreparedSample, into writer: inout RuntimeArtifactBinaryWriter) throws {
        try writer.appendInt32(sample.valid ? 1 : 0)
        try writer.appendInt32(sample.labelClass.rawValue)
        try writer.appendInt32(sample.regimeID)
        try writer.appendInt32(sample.horizonMinutes)
        try writer.appendInt32(sample.horizonSlot)
        writer.appendDouble(sample.movePoints)
        writer.appendDouble(sample.minMovePoints)
        writer.appendDouble(sample.costPoints)
        writer.appendDouble(sample.sampleWeight)
        writer.appendDouble(sample.qualityScore)
        writer.appendDouble(sample.mfePoints)
        writer.appendDouble(sample.maePoints)
        writer.appendDouble(sample.spreadStress)
        writer.appendDouble(sample.traceSpreadMeanRatio)
        writer.appendDouble(sample.traceSpreadPeakRatio)
        writer.appendDouble(sample.traceRangeMeanRatio)
        writer.appendDouble(sample.traceBodyEfficiency)
        writer.appendDouble(sample.traceGapRatio)
        writer.appendDouble(sample.traceReversalRatio)
        writer.appendDouble(sample.traceSessionTransition)
        writer.appendDouble(sample.traceRollover)
        writer.appendDouble(sample.timeToHitFraction)
        try writer.appendInt32(sample.pathFlags.rawValue)
        writer.appendDouble(sample.maskedStepTarget)
        writer.appendDouble(sample.nextVolumeTarget)
        writer.appendDouble(sample.regimeShiftTarget)
        writer.appendDouble(sample.contextLeadTarget)
        writer.appendDouble(sample.pointValue)
        writer.appendDouble(sample.domainHash)
        writer.appendInt64(sample.sampleTimeUTC)
        for value in RuntimeArtifactPreparedSample.artifactVector(sample.x) {
            writer.appendDouble(value)
        }
    }

    public static func decode(from data: Data) throws -> RuntimeArtifactPreparedSample {
        var reader = RuntimeArtifactBinaryReader(data: data)
        return try decode(from: &reader)
    }

    public static func decode(from reader: inout RuntimeArtifactBinaryReader) throws -> RuntimeArtifactPreparedSample {
        let valid = try reader.readInt32() != 0
        let label = LabelClass(rawValue: try reader.readInt32()) ?? .skip
        let regime = try reader.readInt32()
        let horizon = try reader.readInt32()
        let hslot = try reader.readInt32()
        let move = try reader.readDouble()
        let minMove = try reader.readDouble()
        let cost = try reader.readDouble()
        let weight = try reader.readDouble()
        let quality = try reader.readDouble()
        let mfe = try reader.readDouble()
        let mae = try reader.readDouble()
        let spreadStress = try reader.readDouble()
        let traceSpreadMean = try reader.readDouble()
        let traceSpreadPeak = try reader.readDouble()
        let traceRangeMean = try reader.readDouble()
        let traceBody = try reader.readDouble()
        let traceGap = try reader.readDouble()
        let traceReversal = try reader.readDouble()
        let traceSession = try reader.readDouble()
        let traceRollover = try reader.readDouble()
        let hitFraction = try reader.readDouble()
        let flags = SamplePathFlags(rawValue: try reader.readInt32())
        let maskedStep = try reader.readDouble()
        let nextVolume = try reader.readDouble()
        let regimeShift = try reader.readDouble()
        let contextLead = try reader.readDouble()
        let pointValue = try reader.readDouble()
        let domainHash = try reader.readDouble()
        let sampleTime = try reader.readInt64()
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            x[index] = try reader.readDouble()
        }
        return RuntimeArtifactPreparedSample(
            valid: valid,
            labelClass: label,
            regimeID: regime,
            horizonMinutes: horizon,
            horizonSlot: hslot,
            movePoints: move,
            minMovePoints: minMove,
            costPoints: cost,
            sampleWeight: weight,
            qualityScore: quality,
            mfePoints: mfe,
            maePoints: mae,
            spreadStress: spreadStress,
            traceSpreadMeanRatio: traceSpreadMean,
            traceSpreadPeakRatio: traceSpreadPeak,
            traceRangeMeanRatio: traceRangeMean,
            traceBodyEfficiency: traceBody,
            traceGapRatio: traceGap,
            traceReversalRatio: traceReversal,
            traceSessionTransition: traceSession,
            traceRollover: traceRollover,
            timeToHitFraction: hitFraction,
            pathFlags: flags,
            maskedStepTarget: maskedStep,
            nextVolumeTarget: nextVolume,
            regimeShiftTarget: regimeShift,
            contextLeadTarget: contextLead,
            pointValue: pointValue,
            domainHash: domainHash,
            sampleTimeUTC: sampleTime,
            x: x
        )
    }
}

public struct RuntimeFeatureDriftGroupState: Codable, Hashable, Sendable {
    public var baselineObservations: Int
    public var liveObservations: Int
    public var baselineMean: Double
    public var baselineAbs: Double
    public var liveMean: Double
    public var liveAbs: Double
    public var driftEMA: Double

    public init(
        baselineObservations: Int = 0,
        liveObservations: Int = 0,
        baselineMean: Double = 0.0,
        baselineAbs: Double = 0.0,
        liveMean: Double = 0.0,
        liveAbs: Double = 0.0,
        driftEMA: Double = 0.0
    ) {
        self.baselineObservations = max(0, baselineObservations)
        self.liveObservations = max(0, liveObservations)
        self.baselineMean = fxSafeFinite(baselineMean)
        self.baselineAbs = max(0.0, fxSafeFinite(baselineAbs))
        self.liveMean = fxSafeFinite(liveMean)
        self.liveAbs = max(0.0, fxSafeFinite(liveAbs))
        self.driftEMA = fxClamp(driftEMA, 0.0, 4.0)
    }
}

public struct RuntimeFeatureDriftState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var lastTimeUTC: Int64
    public var groups: [RuntimeFeatureDriftGroupState]

    public init(
        ready: Bool = false,
        lastTimeUTC: Int64 = 0,
        groups: [RuntimeFeatureDriftGroupState] = []
    ) {
        self.ready = ready
        self.lastTimeUTC = max(0, lastTimeUTC)
        var resolved = Array(groups.prefix(FeatureGroup.allCases.count))
        if resolved.count < FeatureGroup.allCases.count {
            resolved.append(contentsOf: Array(
                repeating: RuntimeFeatureDriftGroupState(),
                count: FeatureGroup.allCases.count - resolved.count
            ))
        }
        self.groups = resolved
    }
}

public enum RuntimeFeatureDriftCodec {
    public static let byteCount = RuntimeArtifactPayloadMaterializer.featureDriftByteCount

    public static func encode(_ state: RuntimeFeatureDriftState) throws -> Data {
        var writer = RuntimeArtifactBinaryWriter()
        try writer.appendInt32(state.ready ? 1 : 0)
        writer.appendInt64(state.lastTimeUTC)
        let normalized = RuntimeFeatureDriftState(
            ready: state.ready,
            lastTimeUTC: state.lastTimeUTC,
            groups: state.groups
        )
        for group in normalized.groups {
            try writer.appendInt32(group.baselineObservations)
            try writer.appendInt32(group.liveObservations)
            writer.appendDouble(group.baselineMean)
            writer.appendDouble(group.baselineAbs)
            writer.appendDouble(group.liveMean)
            writer.appendDouble(group.liveAbs)
            writer.appendDouble(group.driftEMA)
        }
        return writer.data
    }

    public static func decode(from data: Data) throws -> RuntimeFeatureDriftState {
        var reader = RuntimeArtifactBinaryReader(data: data)
        let ready = try reader.readInt32() != 0
        let lastTime = try reader.readInt64()
        var groups: [RuntimeFeatureDriftGroupState] = []
        groups.reserveCapacity(FeatureGroup.allCases.count)
        for _ in FeatureGroup.allCases {
            groups.append(RuntimeFeatureDriftGroupState(
                baselineObservations: try reader.readInt32(),
                liveObservations: try reader.readInt32(),
                baselineMean: try reader.readDouble(),
                baselineAbs: try reader.readDouble(),
                liveMean: try reader.readDouble(),
                liveAbs: try reader.readDouble(),
                driftEMA: try reader.readDouble()
            ))
        }
        return RuntimeFeatureDriftState(ready: ready, lastTimeUTC: lastTime, groups: groups)
    }
}

public enum RuntimeArtifactAnalogMemoryCodec {
    public static let byteCount = RuntimeArtifactPayloadMaterializer.analogMemoryByteCount

    public static func encode(_ store: AnalogMemoryStore) throws -> Data {
        var writer = RuntimeArtifactBinaryWriter()
        try writer.appendInt32(store.ready ? 1 : 0)
        try writer.appendInt32(store.head)
        try writer.appendInt32(store.size)
        let entries = Array(store.entries.prefix(FXDataEngineConstants.analogMemoryCapacity))
        for index in 0..<FXDataEngineConstants.analogMemoryCapacity {
            let entry = index < entries.count ? entries[index] : AnalogMemoryEntry()
            writer.appendInt64(entry.sampleTimeUTC)
            try writer.appendInt32(entry.regimeID)
            try writer.appendInt32(entry.sessionBucket)
            try writer.appendInt32(entry.horizonBucket)
            writer.appendDouble(entry.domainHash)
            writer.appendDouble(entry.direction)
            writer.appendDouble(entry.edgeNorm)
            writer.appendDouble(entry.quality)
            writer.appendDouble(entry.pathRisk)
            writer.appendDouble(entry.fillRisk)
            writer.appendDouble(entry.weight)
            for value in AnalogMemoryTools.normalizedVector(entry.vector) {
                writer.appendDouble(value)
            }
        }
        return writer.data
    }

    public static func decode(from data: Data) throws -> AnalogMemoryStore {
        var reader = RuntimeArtifactBinaryReader(data: data)
        _ = try reader.readInt32()
        let head = try reader.readInt32()
        let size = try reader.readInt32()
        var entries: [AnalogMemoryEntry] = []
        entries.reserveCapacity(FXDataEngineConstants.analogMemoryCapacity)
        for _ in 0..<FXDataEngineConstants.analogMemoryCapacity {
            let time = try reader.readInt64()
            let regime = try reader.readInt32()
            let session = try reader.readInt32()
            let horizon = try reader.readInt32()
            let domain = try reader.readDouble()
            let direction = try reader.readDouble()
            let edge = try reader.readDouble()
            let quality = try reader.readDouble()
            let path = try reader.readDouble()
            let fill = try reader.readDouble()
            let weight = try reader.readDouble()
            var vector = Array(repeating: 0.0, count: FXDataEngineConstants.analogMemoryFeatures)
            for index in 0..<FXDataEngineConstants.analogMemoryFeatures {
                vector[index] = try reader.readDouble()
            }
            entries.append(AnalogMemoryEntry(
                sampleTimeUTC: time,
                regimeID: regime,
                sessionBucket: session,
                horizonBucket: horizon,
                domainHash: domain,
                vector: vector,
                direction: direction,
                edgeNorm: edge,
                quality: quality,
                pathRisk: path,
                fillRisk: fill,
                weight: weight
            ))
        }
        return AnalogMemoryStore(
            capacity: FXDataEngineConstants.analogMemoryCapacity,
            head: head,
            size: size,
            entries: entries
        )
    }
}
