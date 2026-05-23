import FXDataEngine
import Foundation

public enum StatMSGARCHAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_msgarch",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework MSGARCH branch with three-regime HMM filtering, GARCH volatility recursion, online regime updates, move EMA, and native distribution outputs. Accelerate/Swift SIMD are suitable for the tiny vector operations; Metal, PyTorch, TensorFlow, and NLP are not appropriate for this small statistical state-space model."
    )
}
