import XCTest
import FXDataEngine
@testable import FXAIPlugins

final class PluginRuntimeIntegrationTests: XCTestCase {
    private static let appleM2 = AppleSiliconHardware(architecture: "arm64", cpuBrand: "Apple M2 Pro")

    func testEveryPlannedPluginResolvesCPUOnlyAndAutomaticFallback() throws {
        let plugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }
        XCTAssertEqual(plugins.count, FXAIPluginRegistry.availablePlugins().count)

        let noAcceleratorEnvironment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
            pythonExecutable: nil,
            pyTorchMPSAvailable: false,
            tensorFlowMetalAvailable: false,
            foundationNLPAvailable: false,
            coreMLNeuralEngineAvailable: false
        )

        for plugin in plugins {
            let cpu = try plugin.resolveRuntimeBackend(mode: .cpuOnly, environment: noAcceleratorEnvironment)
            XCTAssertTrue(cpu.selectedBackend.isCPUOnly, plugin.manifest.aiName)
            XCTAssertEqual(cpu.pluginName, plugin.manifest.aiName)

            let automatic = try plugin.resolveRuntimeBackend(environment: noAcceleratorEnvironment)
            XCTAssertTrue(automatic.selectedBackend.isCPUOnly, plugin.manifest.aiName)
            XCTAssertEqual(automatic.pluginName, plugin.manifest.aiName)
        }
    }

    func testFoundationNLPBackendsAreDeclaredAndSelectableForNLPPlugins() throws {
        let plans = Dictionary(uniqueKeysWithValues: FXAIPluginRegistry.accelerationPlans().map { ($0.pluginName, $0) })
        for pluginName in ["ai_chronos", "ai_mythos_rdt", "ai_timesfm"] {
            let plan = try XCTUnwrap(plans[pluginName])
            XCTAssertTrue(plan.declares(.foundationNLP), pluginName)

            let environment = FXPluginRuntimeEnvironment(
                metalDevice: MetalAccelerationDevice(
                    available: true,
                    deviceName: "Test GPU",
                    supportsUnifiedMemory: true,
                    hardware: Self.appleM2
                ),
                pythonExecutable: nil,
                foundationNLPAvailable: true
            )
            let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)
            XCTAssertEqual(resolution.selectedBackend, .foundationNLP, pluginName)
            XCTAssertEqual(resolution.mlFramework, .foundationNLP)
        }
    }

    func testDeclaredExternalPythonBackendsHaveDiscoverablePluginLocalFiles() throws {
        XCTAssertTrue(FileManager.default.fileExists(atPath: FXAIPluginBackendDiscovery.moduleBackendURL.path))

        for plan in FXAIPluginRegistry.accelerationPlans() {
            for backend in plan.declaredBackends where backend.requiresExternalPython || backend == .foundationNLP {
                let backendURL = try XCTUnwrap(
                    FXAIPluginBackendDiscovery.pluginBackendURL(pluginName: plan.pluginName, backend: backend),
                    "\(plan.pluginName) \(backend.rawValue)"
                )
                XCTAssertTrue(
                    FileManager.default.fileExists(atPath: backendURL.path),
                    "\(plan.pluginName) declares \(backend.rawValue) but \(backendURL.path) is missing"
                )

                let descriptor = try XCTUnwrap(
                    FXAIPluginBackendDiscovery.externalPythonDescriptor(pluginName: plan.pluginName, backend: backend),
                    "\(plan.pluginName) \(backend.rawValue)"
                )
                XCTAssertEqual(descriptor.modelIdentifier, plan.pluginName)
                XCTAssertTrue(descriptor.supportsInference)
            }
        }
    }

    func testFoundationNLPModuleBackendPredictsThroughSwiftBridge() throws {
        let pythonExecutable = try BackendPythonTestSupport.requireAnyPython()
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = PythonMLBackendBridge(
            framework: .foundationNLP,
            executable: pythonExecutable,
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: "ai_chronos",
            environment: [
                "FXAI_PLUGIN_ROOT": FXAIPluginBackendDiscovery.pluginRootURL.path,
                "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path,
                "FXAI_FORCE_PYTORCH_CPU": "1"
            ]
        )
        let payload = MLInferencePayload(
            modelIdentifier: "ai_chronos",
            framework: .foundationNLP,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: 1,
            priceCostPoints: 0.5,
            minMovePoints: 1.0,
            x: Self.sampleFeatures(volume: 0.8),
            xWindow: []
        )

        let prediction = try bridge.predictSynchronously(payload)

        try prediction.validate()
        XCTAssertGreaterThanOrEqual(prediction.classProbabilities[2], 0.0)
    }

    func testPyTorchModuleBackendPredictsAndTrainsWhenTorchIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: pythonExecutable,
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: "ai_lstm",
            environment: [
                "FXAI_PLUGIN_ROOT": FXAIPluginBackendDiscovery.pluginRootURL.path,
                "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path,
                "FXAI_FORCE_PYTORCH_CPU": "1"
            ]
        )
        let payload = MLInferencePayload(
            modelIdentifier: "ai_lstm",
            framework: .pyTorch,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: 1,
            priceCostPoints: 0.5,
            minMovePoints: 1.0,
            x: Self.sampleFeatures(volume: 0.8),
            xWindow: []
        )

        let prediction = try bridge.predictSynchronously(payload)
        try prediction.validate()

        let request = TrainRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                priceCostPoints: 0.5,
                minMovePoints: 1.0,
                dataHasVolume: true
            ),
            labelClass: .buy,
            movePoints: 2.5,
            sampleWeight: 1.0,
            x: Self.sampleFeatures(volume: 0.8)
        )
        try bridge.trainSynchronously(MLBackendFactory.trainingPayload(descriptor: bridge.descriptor, request: request))
    }

    func testPyTorchModuleBackendUsesSequenceWindowWhenTorchIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = Self.pythonBridge(
            framework: .pyTorch,
            executable: pythonExecutable,
            modelIdentifier: "ai_lstm",
            stateDirectory: temporaryDirectory
        )
        let upWindowPrediction = try bridge.predictSynchronously(Self.sequencePayload(windowDirection: 1.0))
        let downWindowPrediction = try bridge.predictSynchronously(Self.sequencePayload(windowDirection: -1.0))

        try upWindowPrediction.validate()
        try downWindowPrediction.validate()
        let probabilityDelta = zip(upWindowPrediction.classProbabilities, downWindowPrediction.classProbabilities)
            .map { abs($0 - $1) }
            .reduce(0.0, +)
        let moveDelta = abs(upWindowPrediction.moveMeanPoints - downWindowPrediction.moveMeanPoints)
        XCTAssertGreaterThan(probabilityDelta + moveDelta, 1.0e-7)
    }

    func testPyTorchColdPredictionsAreDeterministicWhenTorchIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = Self.pythonBridge(
            framework: .pyTorch,
            executable: pythonExecutable,
            modelIdentifier: "ai_lstm",
            stateDirectory: temporaryDirectory
        )
        let payload = Self.sequencePayload(windowDirection: 1.0)
        let first = try bridge.predictSynchronously(payload)
        let second = try bridge.predictSynchronously(payload)

        for (lhs, rhs) in zip(first.classProbabilities, second.classProbabilities) {
            XCTAssertEqual(lhs, rhs, accuracy: 1.0e-12)
        }
        XCTAssertEqual(first.moveMeanPoints, second.moveMeanPoints, accuracy: 1.0e-12)
        XCTAssertEqual(first.moveQ25Points, second.moveQ25Points, accuracy: 1.0e-12)
        XCTAssertEqual(first.moveQ50Points, second.moveQ50Points, accuracy: 1.0e-12)
        XCTAssertEqual(first.moveQ75Points, second.moveQ75Points, accuracy: 1.0e-12)
    }

    func testAcceleratedRuntimeWrapsNLPPluginThroughExternalBackend() throws {
        let pythonExecutable = try BackendPythonTestSupport.requireAnyPython()
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_chronos")
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .foundationNLP,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    metalDevice: MetalAccelerationDevice(
                        available: true,
                        deviceName: "Test GPU",
                        supportsUnifiedMemory: true,
                        hardware: Self.appleM2
                    ),
                    foundationNLPAvailable: true
                ),
                pythonExecutable: pythonExecutable,
                pythonEnvironment: [
                    "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path
                ]
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())

        try prediction.validate()
        XCTAssertEqual(runtime.manifest.aiName, "ai_chronos")
    }

    func testFoundationNLPRuntimeTrainingDelegatesToBasePlugin() throws {
        let plugin = FoundationTrainingSpyPlugin()
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .foundationNLP,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    metalDevice: MetalAccelerationDevice(
                        available: true,
                        deviceName: "Test GPU",
                        supportsUnifiedMemory: true,
                        hardware: Self.appleM2
                    ),
                    foundationNLPAvailable: true
                )
            )
        )

        try runtime.train(Self.trainRequest(), hyperParameters: HyperParameters())

        XCTAssertEqual(plugin.trainCount, 1)
    }

    func testAcceleratedRuntimeWrapsPyTorchPluginWhenTorchIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_lstm")
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    metalDevice: MetalAccelerationDevice(
                        available: true,
                        deviceName: "Test GPU",
                        supportsUnifiedMemory: true,
                        hardware: Self.appleM2
                    ),
                    pythonExecutable: pythonExecutable,
                    pyTorchMPSAvailable: true
                ),
                pythonExecutable: pythonExecutable,
                pythonEnvironment: [
                    "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path,
                    "FXAI_FORCE_PYTORCH_CPU": "1"
                ]
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())
        try prediction.validate()
        try runtime.train(Self.trainRequest(), hyperParameters: HyperParameters())
    }

    func testAcceleratedRuntimeWrapsTensorFlowPluginWhenTensorFlowIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requireTensorFlowMetalPython()
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_lstm")
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .tensorFlowMetal,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    metalDevice: MetalAccelerationDevice(
                        available: true,
                        deviceName: "Test GPU",
                        supportsUnifiedMemory: true,
                        hardware: Self.appleM2
                    ),
                    pythonExecutable: pythonExecutable,
                    tensorFlowMetalAvailable: true
                ),
                pythonExecutable: pythonExecutable,
                pythonEnvironment: [
                    "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path,
                    "FXAI_ALLOW_CPU_TENSOR_FALLBACK": "1"
                ]
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())
        try prediction.validate()
        try runtime.train(Self.trainRequest(), hyperParameters: HyperParameters())
    }

    func testRuntimeRecordsResolverCPUFallbackDiagnostics() throws {
        let diagnostics = FXAIPluginRuntimeFallbackDiagnostics(capacity: 4)
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: RuntimeFallbackDiagnosticsSpyPlugin(),
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .fallBackToCPU,
                environment: Self.noAcceleratorEnvironment()
            ),
            fallbackDiagnostics: diagnostics
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())

        try prediction.validate()
        let event = try XCTUnwrap(diagnostics.latest)
        XCTAssertEqual(diagnostics.count, 1)
        XCTAssertEqual(event.sequence, 1)
        XCTAssertEqual(event.pluginName, "runtime_fallback_diagnostics_spy")
        XCTAssertEqual(event.operation, .predict)
        XCTAssertEqual(event.outcome, .resolvedToCPUFallback)
        XCTAssertEqual(event.requestedMode, .pyTorchMPS)
        XCTAssertEqual(event.selectedBackend, .swiftScalar)
        XCTAssertEqual(event.fallbackBackend, .swiftScalar)
        XCTAssertEqual(event.fallbackPolicy, .fallBackToCPU)
        XCTAssertTrue(event.reason.contains("pyTorchMPS is not available"))
        XCTAssertNil(event.underlyingError)
        XCTAssertEqual(event.horizonMinutes, 15)
        XCTAssertEqual(event.sequenceBars, 1)
        XCTAssertTrue(event.dataHasVolume)
    }

    func testRuntimeRecordsExternalPredictAndTrainFallbackDiagnostics() throws {
        let diagnostics = FXAIPluginRuntimeFallbackDiagnostics(capacity: 4)
        let plugin = RuntimeFallbackDiagnosticsSpyPlugin()
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .fallBackToCPU,
                environment: Self.pyTorchMPSEnvironment(),
                pythonExecutable: "python3"
            ),
            fallbackDiagnostics: diagnostics
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())
        try prediction.validate()
        try runtime.train(Self.trainRequest(), hyperParameters: HyperParameters())

        let events = diagnostics.snapshot()
        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events.map(\.sequence), [1, 2])
        XCTAssertEqual(events.map(\.operation), [.predict, .train])
        XCTAssertEqual(events.map(\.outcome), [.externalBackendFailureFallback, .externalBackendFailureFallback])
        XCTAssertTrue(events.allSatisfy { $0.selectedBackend == .pyTorchMPS })
        XCTAssertTrue(events.allSatisfy { $0.fallbackBackend == .swiftScalar })
        XCTAssertTrue(events.allSatisfy { $0.fallbackPolicy == .fallBackToCPU })
        XCTAssertTrue(events.allSatisfy { !($0.underlyingError ?? "").isEmpty })
        XCTAssertEqual(plugin.trainCount, 1)
    }

    func testStrictExternalFailureRecordsBlockedFallbackDiagnostic() throws {
        let diagnostics = FXAIPluginRuntimeFallbackDiagnostics(capacity: 4)
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: RuntimeFallbackDiagnosticsSpyPlugin(),
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .strict,
                environment: Self.pyTorchMPSEnvironment(),
                pythonExecutable: "python3"
            ),
            fallbackDiagnostics: diagnostics
        )

        XCTAssertThrowsError(try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters()))

        let event = try XCTUnwrap(diagnostics.latest)
        XCTAssertEqual(diagnostics.count, 1)
        XCTAssertEqual(event.operation, .predict)
        XCTAssertEqual(event.outcome, .strictPolicyBlockedFallback)
        XCTAssertEqual(event.selectedBackend, .pyTorchMPS)
        XCTAssertEqual(event.fallbackBackend, .swiftScalar)
        XCTAssertEqual(event.fallbackPolicy, .strict)
        XCTAssertTrue(event.reason.contains("strict fallback policy blocked CPU fallback"))
        XCTAssertFalse((event.underlyingError ?? "").isEmpty)
    }

    func testExternalFailureRecordsFallbackUnavailableWhenNoCPUBackendIsDeclared() throws {
        let diagnostics = FXAIPluginRuntimeFallbackDiagnostics(capacity: 4)
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: RuntimeFallbackUnavailableSpyPlugin(),
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .fallBackToCPU,
                environment: Self.pyTorchMPSEnvironment(),
                pythonExecutable: "python3"
            ),
            fallbackDiagnostics: diagnostics
        )

        XCTAssertThrowsError(try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters()))

        let event = try XCTUnwrap(diagnostics.latest)
        XCTAssertEqual(diagnostics.count, 1)
        XCTAssertEqual(event.operation, .predict)
        XCTAssertEqual(event.outcome, .fallbackUnavailable)
        XCTAssertEqual(event.selectedBackend, .pyTorchMPS)
        XCTAssertNil(event.fallbackBackend)
        XCTAssertEqual(event.fallbackPolicy, .fallBackToCPU)
        XCTAssertTrue(event.reason.contains("no CPU fallback backend is declared"))
        XCTAssertFalse((event.underlyingError ?? "").isEmpty)
    }

    func testFallbackDiagnosticsResetWithRuntimeReset() throws {
        let diagnostics = FXAIPluginRuntimeFallbackDiagnostics(capacity: 4)
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: RuntimeFallbackDiagnosticsSpyPlugin(),
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .fallBackToCPU,
                environment: Self.noAcceleratorEnvironment()
            ),
            fallbackDiagnostics: diagnostics
        )

        _ = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())
        XCTAssertEqual(diagnostics.count, 1)

        runtime.reset()

        XCTAssertEqual(diagnostics.count, 0)
        XCTAssertNil(diagnostics.latest)
    }

    private static func sampleFeatures(volume: Double) -> [Double] {
        var values = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        values[0] = 0.14
        values[3] = 0.10
        values[6] = volume
        values[7] = 0.18
        values[8] = -0.04
        values[12] = 0.22
        return values
    }

    private static func sequencePayload(windowDirection: Double) -> MLInferencePayload {
        var current = sampleFeatures(volume: 0.8)
        current[1] = 0.0
        current[2] = 0.0
        let window = (0..<16).map { index -> [Double] in
            var row = sampleFeatures(volume: 0.8)
            let step = Double(index + 1) / 16.0
            row[1] = windowDirection * step * 2.0
            row[2] = windowDirection * step
            row[7] = windowDirection * 0.25
            row[12] = windowDirection * 0.40
            return row
        }
        return MLInferencePayload(
            modelIdentifier: "ai_lstm",
            framework: .pyTorch,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: window.count + 1,
            priceCostPoints: 0.5,
            minMovePoints: 1.0,
            x: current,
            xWindow: window
        )
    }

    private static func pythonBridge(
        framework: MLFramework,
        executable: String,
        modelIdentifier: String,
        stateDirectory: URL
    ) -> PythonMLBackendBridge {
        PythonMLBackendBridge(
            framework: framework,
            executable: executable,
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: modelIdentifier,
            environment: [
                "FXAI_PLUGIN_ROOT": FXAIPluginBackendDiscovery.pluginRootURL.path,
                "FXAI_PLUGIN_STATE_DIR": stateDirectory.path,
                "FXAI_FORCE_PYTORCH_CPU": "1"
            ]
        )
    }

    private static func predictRequest() -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                priceCostPoints: 0.5,
                minMovePoints: 1.0,
                dataHasVolume: true
            ),
            x: sampleFeatures(volume: 0.8)
        )
    }

    private static func trainRequest() -> TrainRequestV4 {
        TrainRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                priceCostPoints: 0.5,
                minMovePoints: 1.0,
                dataHasVolume: true
            ),
            labelClass: .buy,
            movePoints: 2.5,
            sampleWeight: 1.0,
            x: sampleFeatures(volume: 0.8)
        )
    }

    private static func plannedPlugin(named name: String) throws -> any FXAIPlannedPlugin {
        try XCTUnwrap(
            FXAIPluginRegistry.availablePlugins().first { $0.manifest.aiName == name } as? any FXAIPlannedPlugin
        )
    }

    private static func noAcceleratorEnvironment() -> FXPluginRuntimeEnvironment {
        FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
            pythonExecutable: nil,
            pyTorchMPSAvailable: false,
            tensorFlowMetalAvailable: false,
            foundationNLPAvailable: false,
            coreMLNeuralEngineAvailable: false
        )
    }

    private static func pyTorchMPSEnvironment() -> FXPluginRuntimeEnvironment {
        FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(
                available: true,
                deviceName: "Test GPU",
                supportsUnifiedMemory: true,
                hardware: Self.appleM2
            ),
            pythonExecutable: "python3",
            pyTorchMPSAvailable: true
        )
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fxai-plugin-runtime-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private final class FoundationTrainingSpyPlugin: FXAIPlannedPlugin, @unchecked Sendable {
    let manifest = PluginManifestV4(
        aiID: 1,
        aiName: "foundation_training_spy",
        family: .linear,
        capabilityMask: [.selfTest]
    )
    let accelerationPlan = FXPluginAccelerationPlan(
        pluginName: "foundation_training_spy",
        primaryBackends: [.foundationNLP],
        candidateBackends: [.swiftScalar],
        notes: "Test spy for FoundationNLP training delegation."
    )
    private(set) var trainCount = 0

    func reset() {}

    func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        trainCount += 1
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        PredictionV4.skip
    }
}

private final class RuntimeFallbackDiagnosticsSpyPlugin: FXAIPlannedPlugin, @unchecked Sendable {
    let manifest = PluginManifestV4(
        aiID: 1,
        aiName: "runtime_fallback_diagnostics_spy",
        family: .other,
        capabilityMask: [.selfTest]
    )
    let accelerationPlan = FXPluginAccelerationPlan(
        pluginName: "runtime_fallback_diagnostics_spy",
        primaryBackends: [.pyTorchMPS],
        candidateBackends: [.swiftScalar],
        notes: "Test spy for runtime fallback diagnostics."
    )
    private(set) var trainCount = 0
    private(set) var resetCount = 0

    func reset() {
        resetCount += 1
    }

    func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        trainCount += 1
    }

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        PredictionV4.skip
    }
}

private final class RuntimeFallbackUnavailableSpyPlugin: FXAIPlannedPlugin, @unchecked Sendable {
    let manifest = PluginManifestV4(
        aiID: 1,
        aiName: "runtime_fallback_unavailable_spy",
        family: .other,
        capabilityMask: [.selfTest]
    )
    let accelerationPlan = FXPluginAccelerationPlan(
        pluginName: "runtime_fallback_unavailable_spy",
        primaryBackends: [.pyTorchMPS],
        notes: "Test spy for missing CPU fallback diagnostics."
    )

    func reset() {}

    func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {}

    func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        PredictionV4.skip
    }
}
