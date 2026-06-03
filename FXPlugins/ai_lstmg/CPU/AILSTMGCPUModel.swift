import FXDataEngine
import Foundation

public struct AILSTMGCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 18,
            architectureID: AIModelID.lstmg.rawValue,
            architectureMode: "gatedRecurrent",
            family: .recurrent
        ))
    }

    public mutating func reset() {
        self = AILSTMGCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
