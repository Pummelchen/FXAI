import FXDataEngine
import Foundation

public struct AIGRUCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 16,
            architectureID: 58,
            modelID: AIModelID.gru.rawValue,
            architectureMode: "gru",
            family: .recurrent
        ))
    }

    public mutating func reset() {
        self = AIGRUCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
