import XCTest
import FXDataEngine
@testable import FXAIPlugins

final class PluginRuntimeIntegrationTests: XCTestCase {
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
                metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
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
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = PythonMLBackendBridge(
            framework: .foundationNLP,
            executable: "python3",
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
        guard Self.pythonCanImport("torch") else {
            throw XCTSkip("PyTorch is not installed for this runner")
        }
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let bridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "python3",
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

    func testAcceleratedRuntimeWrapsNLPPluginThroughExternalBackend() throws {
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_chronos")
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .foundationNLP,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(foundationNLPAvailable: true),
                pythonEnvironment: [
                    "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path
                ]
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())

        try prediction.validate()
        XCTAssertEqual(runtime.manifest.aiName, "ai_chronos")
    }

    func testAcceleratedRuntimeWrapsPyTorchPluginWhenTorchIsInstalled() throws {
        guard Self.pythonCanImport("torch") else {
            throw XCTSkip("PyTorch is not installed for this runner")
        }
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_lstm")
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .pyTorchMPS,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    pythonExecutable: "python3",
                    pyTorchMPSAvailable: true
                ),
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
        guard Self.pythonCanImport("tensorflow") else {
            throw XCTSkip("TensorFlow is not installed for this runner")
        }
        let temporaryDirectory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let plugin = try Self.plannedPlugin(named: "ai_lstm")
        var runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .tensorFlowMetal,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(
                    pythonExecutable: "python3",
                    tensorFlowMetalAvailable: true
                ),
                pythonEnvironment: [
                    "FXAI_PLUGIN_STATE_DIR": temporaryDirectory.path
                ]
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())
        try prediction.validate()
        try runtime.train(Self.trainRequest(), hyperParameters: HyperParameters())
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

    private static func makeTemporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fxai-plugin-runtime-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func pythonCanImport(_ module: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", "-c", "import \(module)"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
}
