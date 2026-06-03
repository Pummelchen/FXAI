import FXDataEngine
import Foundation

public struct AIMythosRDTCPUModel: Sendable {
    private var core: FXAISequenceArchitectureCPUModel

    public init() {
        self.core = FXAISequenceArchitectureCPUModel(configuration: FXAISequenceArchitectureCPUConfiguration(
            hiddenCount: 24,
            architectureID: 61,
            modelID: AIModelID.mythosRDT.rawValue,
            architectureMode: "mythosRDT",
            family: .transformer
        ))
    }

    public mutating func reset() {
        self = AIMythosRDTCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        core.train(request, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) -> PredictionV4 {
        core.predict(request, hyperParameters: hyperParameters)
    }
}
