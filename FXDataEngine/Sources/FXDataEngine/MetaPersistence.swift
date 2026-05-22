import Foundation

public enum MetaArtifactConstants {
    public static let directory = "FXAI/Meta"
    public static let version = 8
    public static let headerIntCount = 11
}

public enum MetaArtifactPaths {
    public static func metaArtifactFile(symbol: String) -> String {
        "\(MetaArtifactConstants.directory)/fxai_meta_\(RuntimeArtifactPaths.safeSymbol(symbol)).bin"
    }
}

public struct MetaArtifactHeader: Codable, Hashable, Sendable {
    public var version: Int
    public var regimeCount: Int
    public var stackFeatures: Int
    public var stackHidden: Int
    public var tradeGateFeatures: Int
    public var tradeGateHidden: Int
    public var horizonPolicyFeatures: Int
    public var horizonPolicyHidden: Int
    public var policyFeatures: Int
    public var policyHidden: Int
    public var aiCount: Int

    public init(
        version: Int = MetaArtifactConstants.version,
        regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets,
        stackFeatures: Int = FXDataEngineConstants.stackFeatures,
        stackHidden: Int = FXDataEngineConstants.stackHidden,
        tradeGateFeatures: Int = FXDataEngineConstants.stackFeatures,
        tradeGateHidden: Int = FXDataEngineConstants.tradeGateHidden,
        horizonPolicyFeatures: Int = FXDataEngineConstants.horizonPolicyFeatures,
        horizonPolicyHidden: Int = FXDataEngineConstants.horizonPolicyHidden,
        policyFeatures: Int = FXDataEngineConstants.policyFeatures,
        policyHidden: Int = FXDataEngineConstants.policyHidden,
        aiCount: Int = FXDataEngineConstants.aiCount
    ) {
        self.version = version
        self.regimeCount = regimeCount
        self.stackFeatures = stackFeatures
        self.stackHidden = stackHidden
        self.tradeGateFeatures = tradeGateFeatures
        self.tradeGateHidden = tradeGateHidden
        self.horizonPolicyFeatures = horizonPolicyFeatures
        self.horizonPolicyHidden = horizonPolicyHidden
        self.policyFeatures = policyFeatures
        self.policyHidden = policyHidden
        self.aiCount = aiCount
    }

    public static var expected: MetaArtifactHeader {
        MetaArtifactHeader()
    }

    public var fieldsInLegacyOrder: [Int] {
        [
            version,
            regimeCount,
            stackFeatures,
            stackHidden,
            tradeGateFeatures,
            tradeGateHidden,
            horizonPolicyFeatures,
            horizonPolicyHidden,
            policyFeatures,
            policyHidden,
            aiCount
        ]
    }

    public func mismatchReasons(expected: MetaArtifactHeader = .expected) -> [String] {
        var reasons: [String] = []
        let current = fieldsInLegacyOrder
        let expectedFields = expected.fieldsInLegacyOrder
        let names = [
            "version",
            "regime_count",
            "stack_features",
            "stack_hidden",
            "trade_gate_features",
            "trade_gate_hidden",
            "horizon_policy_features",
            "horizon_policy_hidden",
            "policy_features",
            "policy_hidden",
            "ai_count"
        ]
        for index in 0..<min(current.count, expectedFields.count) where current[index] != expectedFields[index] {
            reasons.append(names[index])
        }
        return reasons
    }

    public func isCompatible(expected: MetaArtifactHeader = .expected) -> Bool {
        mismatchReasons(expected: expected).isEmpty
    }
}

public enum MetaArtifactCodec {
    public static func encodeHeader(_ header: MetaArtifactHeader = .expected) throws -> Data {
        var writer = RuntimeArtifactBinaryWriter()
        for field in header.fieldsInLegacyOrder {
            try writer.appendInt32(field)
        }
        return writer.data
    }

    public static func decodeHeader(_ data: Data) throws -> MetaArtifactHeader {
        var reader = RuntimeArtifactBinaryReader(data: data)
        return try decodeHeader(reader: &reader)
    }

    public static func decodeHeader(reader: inout RuntimeArtifactBinaryReader) throws -> MetaArtifactHeader {
        MetaArtifactHeader(
            version: try reader.readInt32(),
            regimeCount: try reader.readInt32(),
            stackFeatures: try reader.readInt32(),
            stackHidden: try reader.readInt32(),
            tradeGateFeatures: try reader.readInt32(),
            tradeGateHidden: try reader.readInt32(),
            horizonPolicyFeatures: try reader.readInt32(),
            horizonPolicyHidden: try reader.readInt32(),
            policyFeatures: try reader.readInt32(),
            policyHidden: try reader.readInt32(),
            aiCount: try reader.readInt32()
        )
    }
}

public enum MetaArtifactSavePolicy {
    public static let defaultMinSaveIntervalSeconds: Int64 = 900

    public static func shouldSave(
        dirty: Bool,
        lastSaveTimeUTC: Int64,
        barTimeUTC: Int64,
        nowUTC: Int64,
        minIntervalSeconds: Int64 = defaultMinSaveIntervalSeconds
    ) -> Bool {
        guard dirty else { return false }
        let effectiveNow = barTimeUTC > 0 ? barTimeUTC : nowUTC
        guard effectiveNow > 0 else { return true }
        guard lastSaveTimeUTC > 0 else { return true }
        return (effectiveNow - lastSaveTimeUTC) >= minIntervalSeconds
    }
}
