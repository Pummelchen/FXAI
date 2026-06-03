import FXDataEngine
import Foundation

public struct AIPatchtstCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 20,
            architectureID: AIModelID.patchTST.rawValue,
            architectureMode: "patchTransformer",
            family: .transformer
        ))
    }

    public mutating func reset() {
        self = AIPatchtstCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
