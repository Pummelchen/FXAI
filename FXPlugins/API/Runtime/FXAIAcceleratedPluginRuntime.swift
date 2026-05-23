import FXDataEngine
import Foundation

public struct FXAIPluginRuntimeConfiguration: Hashable, Sendable {
    public var mode: FXPluginRuntimeMode
    public var fallbackPolicy: FXPluginRuntimeFallbackPolicy
    public var environment: FXPluginRuntimeEnvironment
    public var pythonExecutable: String
    public var pythonEnvironment: [String: String]

    public init(
        mode: FXPluginRuntimeMode = .automatic,
        fallbackPolicy: FXPluginRuntimeFallbackPolicy = .fallBackToCPU,
        environment: FXPluginRuntimeEnvironment = .local,
        pythonExecutable: String = ProcessInfo.processInfo.environment["FXAI_PYTHON"] ?? "python3",
        pythonEnvironment: [String: String] = [:]
    ) {
        self.mode = mode
        self.fallbackPolicy = fallbackPolicy
        self.environment = environment
        self.pythonExecutable = pythonExecutable
        var resolvedEnvironment = pythonEnvironment
        resolvedEnvironment["FXAI_PLUGIN_ROOT"] = resolvedEnvironment["FXAI_PLUGIN_ROOT"] ?? FXAIPluginBackendDiscovery.pluginRootURL.path
        self.pythonEnvironment = resolvedEnvironment
    }
}

public struct FXAIAcceleratedPluginRuntime: FXAIPlannedPlugin {
    private var basePlugin: any FXAIPlannedPlugin
    public var configuration: FXAIPluginRuntimeConfiguration

    public init(
        plugin: any FXAIPlannedPlugin,
        configuration: FXAIPluginRuntimeConfiguration = FXAIPluginRuntimeConfiguration()
    ) {
        self.basePlugin = plugin
        self.configuration = configuration
    }

    public var manifest: PluginManifestV4 {
        basePlugin.manifest
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        basePlugin.accelerationPlan
    }

    public mutating func reset() {
        basePlugin.reset()
    }

    public func selfTest() -> Bool {
        basePlugin.selfTest()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        let resolution = try resolveRuntimeBackend(
            mode: configuration.mode,
            fallbackPolicy: configuration.fallbackPolicy,
            environment: configuration.environment
        )
        switch resolution.selectedBackend {
        case .pyTorchMPS, .tensorFlowMetal:
            do {
                try trainExternal(request, backend: resolution.selectedBackend)
            } catch {
                try trainCPUFallback(request, hyperParameters: hyperParameters, error: error)
            }
        case .foundationNLP:
            return
        case .metal:
            _ = try FXAIPluginMetalBackendDiscovery.executeRuntimeProbe(pluginName: manifest.aiName)
            try basePlugin.train(request, hyperParameters: hyperParameters)
        case .coreMLNeuralEngine:
            throw FXDataEngineError.externalBackend("\(resolution.selectedBackend.rawValue) training is not declared for \(manifest.aiName)")
        case .swiftScalar, .swiftSIMD, .accelerate:
            try basePlugin.train(request, hyperParameters: hyperParameters)
        }
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        let resolution = try resolveRuntimeBackend(
            mode: configuration.mode,
            fallbackPolicy: configuration.fallbackPolicy,
            environment: configuration.environment
        )
        switch resolution.selectedBackend {
        case .pyTorchMPS, .tensorFlowMetal, .foundationNLP:
            do {
                return try predictExternal(request, backend: resolution.selectedBackend)
            } catch {
                return try predictCPUFallback(request, hyperParameters: hyperParameters, error: error)
            }
        case .metal:
            _ = try FXAIPluginMetalBackendDiscovery.executeRuntimeProbe(pluginName: manifest.aiName)
            return try basePlugin.predict(request, hyperParameters: hyperParameters)
        case .coreMLNeuralEngine:
            throw FXDataEngineError.externalBackend("\(resolution.selectedBackend.rawValue) inference is not declared for \(manifest.aiName)")
        case .swiftScalar, .swiftSIMD, .accelerate:
            return try basePlugin.predict(request, hyperParameters: hyperParameters)
        }
    }

    private func predictExternal(
        _ request: PredictRequestV4,
        backend: FXPluginAccelerationBackend
    ) throws -> PredictionV4 {
        let bridge = try pythonBridge(backend: backend)
        let payload = MLBackendFactory.inferencePayload(descriptor: bridge.descriptor, request: request)
        return try bridge.predictSynchronously(payload)
    }

    private func trainExternal(
        _ request: TrainRequestV4,
        backend: FXPluginAccelerationBackend
    ) throws {
        let bridge = try pythonBridge(backend: backend)
        guard bridge.descriptor.supportsTraining else { return }
        let payload = MLBackendFactory.trainingPayload(descriptor: bridge.descriptor, request: request)
        try bridge.trainSynchronously(payload)
    }

    private func pythonBridge(backend: FXPluginAccelerationBackend) throws -> PythonMLBackendBridge {
        guard let descriptor = FXAIPluginBackendDiscovery.externalPythonDescriptor(
            pluginName: manifest.aiName,
            backend: backend,
            executable: configuration.pythonExecutable
        ) else {
            throw FXDataEngineError.externalBackend("\(manifest.aiName) has no external Python descriptor for \(backend.rawValue)")
        }
        let framework = try descriptor.externalFramework()
        return PythonMLBackendBridge(
            framework: framework,
            executable: configuration.pythonExecutable,
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: manifest.aiName,
            environment: configuration.pythonEnvironment
        )
    }

    private mutating func trainCPUFallback(
        _ request: TrainRequestV4,
        hyperParameters: HyperParameters,
        error: Error
    ) throws {
        guard configuration.fallbackPolicy == .fallBackToCPU else {
            throw error
        }
        try basePlugin.train(request, hyperParameters: hyperParameters)
    }

    private func predictCPUFallback(
        _ request: PredictRequestV4,
        hyperParameters: HyperParameters,
        error: Error
    ) throws -> PredictionV4 {
        guard configuration.fallbackPolicy == .fallBackToCPU else {
            throw error
        }
        return try basePlugin.predict(request, hyperParameters: hyperParameters)
    }
}

private extension MLBackendDescriptor {
    func externalFramework() throws -> MLFramework {
        switch mode {
        case .externalPython(let framework, _, _):
            return framework
        case .inProcess:
            throw FXDataEngineError.externalBackend("descriptor is not an external Python backend")
        }
    }
}
