import FXDataEngine
import Foundation

public struct AIStmnCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 18,
            architectureID: AIModelID.stmn.rawValue,
            architectureMode: "stmn",
            family: .stateSpace
        ))
    }

    public mutating func reset() {
        self = AIStmnCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
