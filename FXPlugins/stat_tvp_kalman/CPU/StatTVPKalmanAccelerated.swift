import FXDataEngine
import Foundation

public enum StatTVPKalmanAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_tvp_kalman",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy time-varying-parameter Kalman branch with recursive diagonal covariance updates over sparse OHLCV/context state. Accelerate/Swift SIMD fit the tiny vector recursions; Metal, PyTorch, TensorFlow, and NLP variants are not required for this online model."
    )
}
