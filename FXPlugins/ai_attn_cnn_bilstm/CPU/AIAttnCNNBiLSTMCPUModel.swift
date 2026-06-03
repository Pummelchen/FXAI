import FXDataEngine
import Foundation

public struct AIAttnCNNBiLSTMCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 22,
            architectureID: AIModelID.attnCNNBiLSTM.rawValue,
            architectureMode: "attentionCNNBiLSTM",
            family: .convolutional
        ))
    }

    public mutating func reset() {
        self = AIAttnCNNBiLSTMCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
