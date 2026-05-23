import FXDataEngine
import Foundation

public enum FactorPCAPanelAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "factor_pca_panel",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy PCA panel factor with online mean/loading/variance updates over sparse OHLCV/context state. Accelerate/Swift SIMD suit the small dense-vector projections; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
