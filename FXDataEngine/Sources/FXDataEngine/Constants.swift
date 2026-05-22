import Foundation

public enum FXDataEngineConstants {
    public static let contextTopSymbols = 3
    public static let mtfStateFeaturesPerTimeframe = 4
    public static let mainMTFTimeframeCount = 4
    public static let contextMTFTimeframeCount = 5
    public static let mainMTFFeatureOffset = 84
    public static let contextMTFFeatureOffset = mainMTFFeatureOffset + mainMTFTimeframeCount * mtfStateFeaturesPerTimeframe
    public static let contextSlotMTFFeatures = contextMTFTimeframeCount * mtfStateFeaturesPerTimeframe
    public static let macroEventFeatureOffset = contextMTFFeatureOffset + contextTopSymbols * contextSlotMTFFeatures
    public static let macroEventFeatures = 20
    public static let aiFeatures = macroEventFeatureOffset + macroEventFeatures
    public static let aiWeights = aiFeatures + 1
    public static let contextBaseSymbolFeatures = 4
    public static let contextSharedAdapterFeatures = 4
    public static let contextSharedOffset = contextTopSymbols * contextBaseSymbolFeatures
    public static let contextMTFOffset = contextSharedOffset + contextSharedAdapterFeatures
    public static let contextExtraFeatures = contextMTFOffset + contextTopSymbols * contextSlotMTFFeatures
    public static let contextDynamicPool = 12
    public static let aiCount = 65
    public static let analogMemoryCapacity = 384
    public static let analogMemoryFeatures = 12
    public static let analogMemoryMinMatches = 3
    public static let pluginHorizonBuckets = 8
    public static let pluginReplayCapacity = 96
    public static let pluginReplaySteps = 2
    public static let sharedTransferHorizonBuckets = pluginHorizonBuckets
    public static let sharedTransferFeatures = 28
    public static let sharedTransferLatent = 12
    public static let sharedTransferSequenceTokens = 16
    public static let sharedTransferBarFeatures = 12
    public static let sharedTransferStateFeatures = sharedTransferBarFeatures
    public static let sharedTransferDomainBuckets = 8
    public static let executionTraceBars = 12
    public static let brokerExecutionTraceCapacity = 192
    public static let brokerExecutionSymbolBuckets = 12
    public static let brokerExecutionSideCount = 3
    public static let brokerExecutionOrderTypeCount = 5
    public static let brokerExecutionEventKindCount = 4
    public static let aiMLPHidden = 12
    public static let apiVersionV4 = 4
    public static let maxSequenceBars = 96
    public static let maxContextSymbols = 48
    public static let pluginRegimeBuckets = 12
    public static let pluginSessionBuckets = 6
    public static let normMethodCount = 17
    public static let normalizationCandidateMax = 12
    public static let horizonPolicyFeatures = 48
    public static let horizonPolicyHidden = 16
    public static let stackFeatures = 84
    public static let stackHidden = 28
    public static let tradeGateHidden = 16
    public static let policyFeatures = 32
    public static let policyHidden = 16
    public static let unitRangeFloor = 0.0001
    public static let unitRangeCeil = 0.9999
    public static let signedUnitRangeFloor = -0.9999
    public static let signedUnitRangeCeil = 0.9999
}

@inlinable
public func fxClamp(_ value: Double, _ lower: Double, _ upper: Double) -> Double {
    guard value.isFinite else { return 0.0 }
    return min(max(value, lower), upper)
}

@inlinable
public func fxClampUnit(_ value: Double) -> Double {
    fxClamp(value, FXDataEngineConstants.unitRangeFloor, FXDataEngineConstants.unitRangeCeil)
}

@inlinable
public func fxClampSignedUnit(_ value: Double) -> Double {
    fxClamp(value, FXDataEngineConstants.signedUnitRangeFloor, FXDataEngineConstants.signedUnitRangeCeil)
}

@inlinable
public func fxSafeFinite(_ value: Double, fallback: Double = 0.0) -> Double {
    value.isFinite ? value : fallback
}

@inlinable
public func fxSign(_ value: Double) -> Double {
    if value > 0.0 { return 1.0 }
    if value < 0.0 { return -1.0 }
    return 0.0
}
