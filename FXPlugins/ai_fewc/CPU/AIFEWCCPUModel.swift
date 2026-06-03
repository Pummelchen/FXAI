import FXDataEngine
import Foundation

public struct AIFEWCCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 16,
            architectureID: AIModelID.fewc.rawValue,
            architectureMode: "fewc",
            family: .stateSpace
        ))
    }

    public mutating func reset() {
        self = AIFEWCCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
