import FXDataEngine
import Foundation

public struct AIBiLSTMCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 20,
            architectureID: 59,
            modelID: AIModelID.bilstm.rawValue,
            architectureMode: "bidirectional",
            family: .recurrent
        ))
    }

    public mutating func reset() {
        self = AIBiLSTMCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
