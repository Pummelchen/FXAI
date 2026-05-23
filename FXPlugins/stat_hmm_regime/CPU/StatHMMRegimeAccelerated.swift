import FXDataEngine
import Foundation

public enum StatHMMRegimeAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_hmm_regime",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework HMM regime branch with online forward probabilities, transition updates, regime mean/variance adaptation, volume-aware volatility/confidence, move EMA, and native distribution outputs. Metal, PyTorch, TensorFlow, and NLP are not required for this compact online HMM."
    )
}
