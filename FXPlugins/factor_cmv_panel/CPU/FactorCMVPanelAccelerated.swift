import FXDataEngine
import Foundation

public enum FactorCMVPanelAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "factor_cmv_panel",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy CMV panel factor branch with seeded online linear base policy, panel context pressure, trend slope, and value gap inputs. Accelerate/Swift SIMD are suitable for the small vector math; Metal, PyTorch, TensorFlow, and NLP variants are not required."
    )
}
