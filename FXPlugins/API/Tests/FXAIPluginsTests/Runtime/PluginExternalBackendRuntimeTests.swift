import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class PluginExternalBackendRuntimeTests: XCTestCase {
    func testEveryDeclaredPyTorchBackendPredictsTrainsPersistsAndReloadsWhenTorchIsInstalled() throws {
        guard Self.pythonCanImport("torch") else {
            throw XCTSkip("PyTorch is not installed for this runner")
        }
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.pyTorchMPS) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "torch")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(pluginName: plan.pluginName, framework: .pyTorch, stateDirectory: temporaryDirectory)
                let payload = Self.inferencePayload(pluginName: plan.pluginName, framework: .pyTorch, includeText: false)

                let prediction = try bridge.predictSynchronously(payload)
                try prediction.validate()

                try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))
                XCTAssertTrue(
                    Self.stateDirectoryHasArtifact(temporaryDirectory),
                    "\(plan.pluginName) PyTorch train did not create a persistence artifact"
                )

                let reloadedBridge = Self.bridge(pluginName: plan.pluginName, framework: .pyTorch, stateDirectory: temporaryDirectory)
                let reloadedPrediction = try reloadedBridge.predictSynchronously(payload)
                try reloadedPrediction.validate()
            } catch {
                XCTFail("\(plan.pluginName) PyTorch backend failed: \(error)")
            }
        }
    }

    func testEveryDeclaredTensorFlowBackendPredictsTrainsPersistsAndReloadsWhenTensorFlowIsInstalled() throws {
        guard Self.pythonCanImport("tensorflow") else {
            throw XCTSkip("TensorFlow is not installed for this runner")
        }
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.tensorFlowMetal) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "tensorflow")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(pluginName: plan.pluginName, framework: .tensorFlow, stateDirectory: temporaryDirectory)
                let payload = Self.inferencePayload(pluginName: plan.pluginName, framework: .tensorFlow, includeText: false)

                let prediction = try bridge.predictSynchronously(payload)
                try prediction.validate()

                try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))
                XCTAssertTrue(
                    Self.stateDirectoryHasArtifact(temporaryDirectory),
                    "\(plan.pluginName) TensorFlow train did not create a persistence artifact"
                )

                let reloadedBridge = Self.bridge(pluginName: plan.pluginName, framework: .tensorFlow, stateDirectory: temporaryDirectory)
                let reloadedPrediction = try reloadedBridge.predictSynchronously(payload)
                try reloadedPrediction.validate()
            } catch {
                XCTFail("\(plan.pluginName) TensorFlow backend failed: \(error)")
            }
        }
    }

    func testEveryDeclaredNLPBackendConsumesTextEventsAndHasNoTextFallback() throws {
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.foundationNLP) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "nlp")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(pluginName: plan.pluginName, framework: .foundationNLP, stateDirectory: temporaryDirectory)
                let noTextPayload = Self.inferencePayload(pluginName: plan.pluginName, framework: .foundationNLP, includeText: false)
                let textPayload = Self.inferencePayload(pluginName: plan.pluginName, framework: .foundationNLP, includeText: true)

                XCTAssertTrue(noTextPayload.eventTexts.isEmpty)
                XCTAssertFalse(textPayload.eventTexts.isEmpty)
                XCTAssertEqual(textPayload.tokenizerContract.version, PluginTokenizerContractV4.defaultVersion)

                let noTextPrediction = try bridge.predictSynchronously(noTextPayload)
                let textPrediction = try bridge.predictSynchronously(textPayload)
                try noTextPrediction.validate()
                try textPrediction.validate()
                XCTAssertNotEqual(
                    noTextPrediction.classProbabilities,
                    textPrediction.classProbabilities,
                    "\(plan.pluginName) NLP backend ignored text event context"
                )
            } catch {
                XCTFail("\(plan.pluginName) NLP backend failed: \(error)")
            }
        }
    }

    private static func bridge(
        pluginName: String,
        framework: MLFramework,
        stateDirectory: URL
    ) -> PythonMLBackendBridge {
        PythonMLBackendBridge(
            framework: framework,
            executable: "python3",
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: pluginName,
            environment: Self.acceleratorEnvironment(
                framework: framework,
                stateDirectory: stateDirectory
            )
        )
    }

    private static func acceleratorEnvironment(
        framework: MLFramework,
        stateDirectory: URL
    ) -> [String: String] {
        var environment = [
            "FXAI_PLUGIN_ROOT": FXAIPluginBackendDiscovery.pluginRootURL.path,
            "FXAI_PLUGIN_STATE_DIR": stateDirectory.path
        ]
        switch framework {
        case .pyTorch:
            environment["FXAI_REQUIRE_PYTORCH_MPS"] = "1"
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        case .tensorFlow:
            environment["FXAI_REQUIRE_TENSORFLOW_METAL"] = "1"
        case .nativeSwift, .metal, .foundationNLP:
            break
        }
        return environment
    }

    private static func inferencePayload(
        pluginName: String,
        framework: MLFramework,
        includeText: Bool
    ) -> MLInferencePayload {
        MLInferencePayload(
            modelIdentifier: pluginName,
            framework: framework,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: 1,
            priceCostPoints: 0.5,
            minMovePoints: 1.0,
            x: sampleFeatures(volume: 0.8),
            xWindow: [],
            textEvents: includeText ? [
                PluginTextEventV4(
                    eventTimeUTC: 1_800_000_000,
                    source: "calendar",
                    headline: "USD growth beat supports risk-on flows",
                    body: "Central bank language remains hawkish before inflation data.",
                    importance: 0.85,
                    symbols: ["EURUSD", "USDJPY"]
                )
            ] : []
        )
    }

    private static func trainingPayload(payload: MLInferencePayload, label: LabelClass) -> MLTrainingPayload {
        let request = TrainRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: payload.horizonMinutes,
                sequenceBars: payload.sequenceBars,
                priceCostPoints: payload.priceCostPoints,
                minMovePoints: payload.minMovePoints,
                dataHasVolume: payload.dataHasVolume,
                tokenizerContract: payload.tokenizerContract,
                textEvents: payload.textEvents
            ),
            labelClass: label,
            movePoints: 2.5,
            sampleWeight: 1.0,
            nextVolumeTarget: 1.0,
            x: payload.x,
            xWindow: payload.xWindow
        )
        return MLTrainingPayload(inference: payload, request: request)
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

    private static func makeTemporaryDirectory(_ pluginName: String, suffix: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fxai-\(pluginName)-\(suffix)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func stateDirectoryHasArtifact(_ url: URL) -> Bool {
        guard let contents = try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else {
            return false
        }
        return contents.contains { !$0.hasDirectoryPath }
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
