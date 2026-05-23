import FXDataEngine
import Foundation

public enum StatMicroflowProxyAccelerated {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "stat_microflow_proxy",
        primaryBackends: [.swiftScalar, .accelerate],
        candidateBackends: [.swiftSIMD],
        usesVolumeWhenAvailable: true,
        notes: "Plugin-local Swift CPU port of the legacy framework OHLCV microflow proxy branch. It scores contextual flow, rolling-window slope/range, liquidity pressure from volume, online move EMA, and native path-quality heads. Metal, PyTorch, TensorFlow, and NLP are not required until the plugin consumes batched order-flow tensors or text/event payloads."
    )
}
