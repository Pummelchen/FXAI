import FXDataEngine
import Foundation

public enum StatCointVECMAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_coint_vecm",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework cointegration/VECM branch. It keeps the deterministic OU residual state, online mean-reversion speed and variance updates, volume-aware confidence and volatility handling, move EMA, and native distribution outputs. Metal, PyTorch, TensorFlow, and NLP are not required for this small online statistical model."
    )
}
