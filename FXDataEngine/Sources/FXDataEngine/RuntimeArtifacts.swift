import Foundation

public enum RuntimeArtifactConstants {
    public static let directory = "FXAI/Runtime"
    public static let version = 15
    public static let normalizationRollWindowMax = 512
    public static let replayCapacity = 384
    public static let maxHorizons = 8
    public static let conformalDepth = 96
    public static let reliabilityPendingCapacity = 2_048
}

public enum RuntimeArtifactPaths {
    public static func safeSymbol(_ symbol: String, defaultValue: String = "default") -> String {
        var clean = symbol.isEmpty ? defaultValue : symbol
        for character in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"] {
            clean = clean.replacingOccurrences(of: character, with: "_")
        }
        return clean
    }

    public static func runtimeArtifactFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_runtime_\(safeSymbol(symbol)).bin"
    }

    public static func persistenceManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_persistence_\(safeSymbol(symbol)).tsv"
    }

    public static func featureManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_features_\(safeSymbol(symbol)).tsv"
    }

    public static func macroManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_macro_\(safeSymbol(symbol)).tsv"
    }

    public static func performanceManifestFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_perf_\(safeSymbol(symbol)).tsv"
    }

    public static func shadowLedgerFile(symbol: String) -> String {
        "\(RuntimeArtifactConstants.directory)/fxai_shadow_\(safeSymbol(symbol)).tsv"
    }
}

public struct RuntimeArtifactHeader: Codable, Hashable, Sendable {
    public var version: Int
    public var featureCount: Int
    public var normalizationMethodCount: Int
    public var normalizationRollWindowMax: Int
    public var replayCapacity: Int
    public var aiCount: Int
    public var regimeCount: Int
    public var maxHorizons: Int
    public var conformalDepth: Int
    public var reliabilityPendingCapacity: Int

    public init(
        version: Int = RuntimeArtifactConstants.version,
        featureCount: Int = FXDataEngineConstants.aiFeatures,
        normalizationMethodCount: Int = FXDataEngineConstants.normMethodCount,
        normalizationRollWindowMax: Int = RuntimeArtifactConstants.normalizationRollWindowMax,
        replayCapacity: Int = RuntimeArtifactConstants.replayCapacity,
        aiCount: Int = FXDataEngineConstants.aiCount,
        regimeCount: Int = FXDataEngineConstants.pluginRegimeBuckets,
        maxHorizons: Int = RuntimeArtifactConstants.maxHorizons,
        conformalDepth: Int = RuntimeArtifactConstants.conformalDepth,
        reliabilityPendingCapacity: Int = RuntimeArtifactConstants.reliabilityPendingCapacity
    ) {
        self.version = version
        self.featureCount = featureCount
        self.normalizationMethodCount = normalizationMethodCount
        self.normalizationRollWindowMax = normalizationRollWindowMax
        self.replayCapacity = replayCapacity
        self.aiCount = aiCount
        self.regimeCount = regimeCount
        self.maxHorizons = maxHorizons
        self.conformalDepth = conformalDepth
        self.reliabilityPendingCapacity = reliabilityPendingCapacity
    }

    public var isCompatibleWithCurrentContract: Bool {
        self == RuntimeArtifactHeader()
    }
}

public struct FeatureClipBounds: Codable, Hashable, Sendable {
    public let lower: Double
    public let upper: Double

    public init(lower: Double, upper: Double) {
        self.lower = lower
        self.upper = upper
    }
}

public extension FeatureRegistry {
    func clipBounds(for featureIndex: Int) -> FeatureClipBounds {
        var lower = -8.0
        var upper = 8.0

        if featureIndex == 5 {
            lower = 0.0
            upper = 10.0
        } else if featureIndex == 6 {
            lower = 0.0
            upper = 12.0
        } else if featureIndex == 12 {
            lower = -1.0
            upper = 1.0
        } else if (15...17).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if featureIndex == 40 {
            lower = -1.2
            upper = 1.2
        } else if (41...43).contains(featureIndex) {
            lower = 0.0
            upper = 40.0
        } else if (44...45).contains(featureIndex) {
            lower = 0.0
            upper = 40.0
        } else if featureIndex == 47 {
            lower = -12.0
            upper = 12.0
        } else if (62...67).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if (68...69).contains(featureIndex) {
            lower = -8.0
            upper = 8.0
        } else if featureIndex == 70 {
            lower = 0.0
            upper = 8.0
        } else if featureIndex == 71 {
            lower = -6.0
            upper = 6.0
        } else if (72...75).contains(featureIndex) {
            lower = -1.2
            upper = 1.2
        } else if featureIndex == 76 || featureIndex == 77 {
            lower = -4.5
            upper = 4.5
        } else if featureIndex == 78 || featureIndex == 79 {
            lower = -6.0
            upper = 6.0
        } else if featureIndex == 80 {
            lower = 0.0
            upper = 6.0
        } else if featureIndex == 81 {
            lower = -8.0
            upper = 8.0
        } else if featureIndex == 82 {
            lower = 0.0
            upper = 4.5
        } else if featureIndex == 83 {
            lower = -1.1
            upper = 1.1
        } else if featureIndex >= FXDataEngineConstants.mainMTFFeatureOffset,
                  featureIndex < FXDataEngineConstants.macroEventFeatureOffset {
            let relative: Int
            if featureIndex >= FXDataEngineConstants.contextMTFFeatureOffset {
                relative = (featureIndex - FXDataEngineConstants.contextMTFFeatureOffset) %
                    FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            } else {
                relative = (featureIndex - FXDataEngineConstants.mainMTFFeatureOffset) %
                    FXDataEngineConstants.mtfStateFeaturesPerTimeframe
            }
            if relative <= 1 {
                lower = -1.2
                upper = 1.2
            } else if relative == 2 {
                lower = -6.0
                upper = 6.0
            } else {
                lower = -6.0
                upper = 8.0
            }
        } else if featureIndex >= FXDataEngineConstants.macroEventFeatureOffset,
                  featureIndex < FXDataEngineConstants.aiFeatures {
            let relative = featureIndex - FXDataEngineConstants.macroEventFeatureOffset
            if relative <= 2 {
                lower = 0.0
                upper = 1.2
            } else if relative == 3 {
                lower = -6.0
                upper = 6.0
            } else if relative == 4 {
                lower = 0.0
                upper = 6.0
            } else {
                lower = -1.2
                upper = 1.2
            }
        } else if featureIndex >= 50, featureIndex < FXDataEngineConstants.aiFeatures {
            let relative = (featureIndex - 50) % 4
            if relative == 3 {
                lower = -1.1
                upper = 1.1
            }
        }

        return FeatureClipBounds(lower: lower, upper: upper)
    }

    func leakageGuarded(for featureIndex: Int) -> Bool {
        let provenance = provenance(for: featureIndex)
        return FeatureProvenance.allCases.contains(provenance)
    }
}

public struct PersistenceCoverageManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "ai_id",
        "ai_name",
        "reference_tier",
        "coverage_tag",
        "checkpoint_depth",
        "persistent",
        "state_version",
        "capability_mask",
        "stateful_checkpoint",
        "native_snapshot",
        "deterministic_replay",
        "native_required",
        "promotion_ready",
        "state_file_size",
        "state_file",
        "coverage_note"
    ]

    public var aiID: Int
    public var aiName: String
    public var referenceTier: ReferenceTier
    public var coverageTag: String
    public var checkpointDepth: String
    public var persistent: Bool
    public var stateVersion: Int
    public var capabilityMask: PluginCapability
    public var nativeSnapshot: Bool
    public var deterministicReplay: Bool
    public var stateFileSize: Int64
    public var stateFile: String

    public init(
        aiID: Int,
        aiName: String,
        referenceTier: ReferenceTier,
        coverageTag: String,
        checkpointDepth: String,
        persistent: Bool,
        stateVersion: Int,
        capabilityMask: PluginCapability,
        nativeSnapshot: Bool,
        deterministicReplay: Bool,
        stateFileSize: Int64,
        stateFile: String
    ) {
        self.aiID = aiID
        self.aiName = aiName
        self.referenceTier = referenceTier
        self.coverageTag = coverageTag
        self.checkpointDepth = checkpointDepth
        self.persistent = persistent
        self.stateVersion = stateVersion
        self.capabilityMask = capabilityMask
        self.nativeSnapshot = nativeSnapshot
        self.deterministicReplay = deterministicReplay
        self.stateFileSize = stateFileSize
        self.stateFile = stateFile
    }

    public var statefulCheckpoint: Bool {
        capabilityMask.contains(.onlineLearning) ||
            capabilityMask.contains(.replay) ||
            capabilityMask.contains(.stateful)
    }

    public var nativeRequired: Bool {
        statefulCheckpoint
    }

    public var promotionReady: Bool {
        !nativeRequired || (coverageTag == "native_model" && nativeSnapshot)
    }

    public var coverageNote: String {
        if nativeRequired && !promotionReady {
            "stateful model blocked from live promotion until native parameter snapshot coverage is implemented"
        } else if coverageTag == "native_model" {
            "native checkpoint verified"
        } else if coverageTag == "native_replay" {
            "deterministic replay checkpoint available for audit and research recovery only"
        } else {
            "checkpoint not required"
        }
    }
}

public struct FeatureRegistryManifestRow: Codable, Hashable, Sendable {
    public static let header = [
        "feature_idx",
        "feature_name",
        "feature_group",
        "provenance",
        "leakage_guarded",
        "clip_lo",
        "clip_hi"
    ]

    public var featureIndex: Int
    public var featureName: String
    public var featureGroup: String
    public var provenance: String
    public var leakageGuarded: Bool
    public var clipLower: Double
    public var clipUpper: Double

    public init(featureIndex: Int, registry: FeatureRegistry = FeatureRegistry()) {
        let bounds = registry.clipBounds(for: featureIndex)
        self.featureIndex = featureIndex
        self.featureName = registry.name(for: featureIndex)
        self.featureGroup = registry.group(for: featureIndex).name
        self.provenance = registry.provenance(for: featureIndex).rawValue
        self.leakageGuarded = registry.leakageGuarded(for: featureIndex)
        self.clipLower = bounds.lower
        self.clipUpper = bounds.upper
    }
}

public enum RuntimeStage: Int, Codable, Sendable, CaseIterable {
    case total = 0
    case featurePipeline
    case transfer
    case router
    case policy
    case shadow
    case controlPlane

    public var name: String {
        switch self {
        case .total: "total"
        case .featurePipeline: "feature_pipeline"
        case .transfer: "transfer"
        case .router: "router"
        case .policy: "policy"
        case .shadow: "shadow"
        case .controlPlane: "control_plane"
        }
    }
}
