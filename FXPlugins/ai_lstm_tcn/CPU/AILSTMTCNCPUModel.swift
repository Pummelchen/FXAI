import FXDataEngine
import Foundation

public struct AILSTMTCNCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 20,
            architectureID: 60,
            modelID: AIModelID.lstmTCN.rawValue,
            architectureMode: "lstmTCN",
            family: .convolutional
        ))
    }

    public mutating func reset() {
        self = AILSTMTCNCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
