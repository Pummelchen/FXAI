import FXDataEngine
import Foundation

public struct AILSTMTCNPlugin: FXAIPlannedPlugin {
    private static let descriptor = FXAIPluginImplementationDescriptor.sequence(.lstmTCN, "ai_lstm_tcn", .convolutional, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine])
    private var runtime: FXAIReferencePluginRuntime

    public var manifest: PluginManifestV4 { Self.descriptor.manifest }
    public var accelerationPlan: FXPluginAccelerationPlan { Self.descriptor.accelerationPlan }

    public init() {
        self.runtime = FXAIReferencePluginRuntime(descriptor: Self.descriptor)
    }

    public mutating func reset() {
        runtime = FXAIReferencePluginRuntime(descriptor: Self.descriptor)
    }

    public func selfTest() -> Bool {
        (try? manifest.validate()) != nil && !Self.descriptor.primaryBackends.isEmpty
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        runtime.train(request, descriptor: Self.descriptor, hyperParameters: hyperParameters)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        try request.validate()
        try PluginContractTools.validateCompatibility(manifest: manifest, context: request.context)
        return runtime.predict(request, descriptor: Self.descriptor, hyperParameters: hyperParameters)
    }
}
