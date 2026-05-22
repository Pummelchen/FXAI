import Foundation

public enum PluginPersistenceConstants {
    public static let directory = "FXAI/Runtime/Plugins"
    public static let artifactVersion = 12
}

public struct PluginPersistenceDescriptor: Codable, Hashable, Sendable {
    public var supportsPersistentState: Bool
    public var stateVersion: Int
    public var supportsDeterministicReplayCheckpoint: Bool
    public var supportsNativeParameterSnapshot: Bool

    public init(
        supportsPersistentState: Bool = true,
        stateVersion: Int = PluginPersistenceConstants.artifactVersion,
        supportsDeterministicReplayCheckpoint: Bool = true,
        supportsNativeParameterSnapshot: Bool = false
    ) {
        self.supportsPersistentState = supportsPersistentState
        self.stateVersion = max(0, stateVersion)
        self.supportsDeterministicReplayCheckpoint = supportsDeterministicReplayCheckpoint
        self.supportsNativeParameterSnapshot = supportsNativeParameterSnapshot
    }
}

public enum PluginPersistenceTools {
    public static func safeToken(_ raw: String, defaultValue: String = "default") -> String {
        RuntimeArtifactPaths.safeSymbol(raw, defaultValue: defaultValue)
    }

    public static func stateFile(symbol: String, aiName: String) -> String {
        "\(PluginPersistenceConstants.directory)/fxai_plugin_\(safeToken(symbol))_\(safeToken(aiName)).bin"
    }

    public static func isStatefulCheckpoint(manifest: PluginManifestV4) -> Bool {
        manifest.capabilityMask.contains(.onlineLearning) ||
            manifest.capabilityMask.contains(.replay) ||
            manifest.capabilityMask.contains(.stateful)
    }

    public static func defaultReferenceTier(aiID: Int) -> ReferenceTier {
        guard let model = AIModelID(rawValue: aiID) else { return .fullNative }
        switch model {
        case .chronos, .timesfm, .cfxWorld, .graphWM:
            return .surrogate
        case .s4, .stmn, .trr, .retrDiff, .loffm, .moeConformal,
             .tst, .autoformer, .patchTST, .geodesicAttention,
             .qcew, .fewc, .gha, .tesseract, .catboost, .lightgbm,
             .xgbFast, .xgboost, .quantile, .enhash, .mlpTiny,
             .lstm, .lstmg, .tcn, .tft, .statMSGARCH,
             .statARIMAXGARCH, .treeRF, .statCointVECM,
             .statOUSpread, .rlPPO, .statMicroflowProxy,
             .statHMMRegime, .cnnLSTM, .attnCNNBiLSTM,
             .statEMDHHT, .statVMD, .statTVPKalman,
             .factorPCAPanel, .factorPPPValue, .factorCarry,
             .factorCMVPanel, .trendTSMOMVol, .trendXSMOMRank,
             .trendVolBreakout, .statXRateConsistency, .gru,
             .bilstm, .lstmTCN, .mythosRDT:
            return .compressedNative
        case .ftrlLogit, .paLinear, .sgdLogit, .linElasticLogit, .linProfitLogit:
            return .fullNative
        case .m1Sync, .buyOnly, .sellOnly, .randomNoSkip,
             .demoMovingAverageCross, .demoFXStupid:
            return .ruleBaseline
        }
    }

    public static func depthTag(
        manifest: PluginManifestV4,
        descriptor: PluginPersistenceDescriptor = PluginPersistenceDescriptor()
    ) -> String {
        guard isStatefulCheckpoint(manifest: manifest) else {
            return "stateless"
        }
        if descriptor.supportsNativeParameterSnapshot {
            return "native_parameters"
        }
        if descriptor.supportsDeterministicReplayCheckpoint {
            return "deterministic_replay"
        }
        return "base_only"
    }

    public static func coverageTag(
        manifest: PluginManifestV4,
        descriptor: PluginPersistenceDescriptor = PluginPersistenceDescriptor()
    ) -> String {
        guard isStatefulCheckpoint(manifest: manifest) else {
            return defaultReferenceTier(aiID: manifest.aiID).runtimeArtifactName
        }
        return descriptor.supportsNativeParameterSnapshot ? "native_model" : "native_replay"
    }

    public static func coverageManifestRow(
        manifest: PluginManifestV4,
        symbol: String,
        descriptor: PluginPersistenceDescriptor = PluginPersistenceDescriptor(),
        stateFileSize: Int64 = -1
    ) throws -> PersistenceCoverageManifestRow {
        try manifest.validate()
        return PersistenceCoverageManifestRow(
            aiID: manifest.aiID,
            aiName: manifest.aiName,
            referenceTier: manifest.referenceTier,
            coverageTag: coverageTag(manifest: manifest, descriptor: descriptor),
            checkpointDepth: depthTag(manifest: manifest, descriptor: descriptor),
            persistent: descriptor.supportsPersistentState,
            stateVersion: descriptor.stateVersion,
            capabilityMask: manifest.capabilityMask,
            nativeSnapshot: descriptor.supportsNativeParameterSnapshot,
            deterministicReplay: descriptor.supportsDeterministicReplayCheckpoint,
            stateFileSize: stateFileSize,
            stateFile: stateFile(symbol: symbol, aiName: manifest.aiName)
        )
    }
}
