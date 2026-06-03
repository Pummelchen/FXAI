import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class PluginExternalBackendRuntimeTests: XCTestCase {
    func testEveryDeclaredPyTorchBackendPredictsTrainsPersistsAndReloadsWhenTorchIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.pyTorchMPS) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "torch")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(
                    pluginName: plan.pluginName,
                    framework: .pyTorch,
                    executable: pythonExecutable,
                    stateDirectory: temporaryDirectory
                )
                let payload = Self.inferencePayload(pluginName: plan.pluginName, framework: .pyTorch, includeText: false)

                let prediction = try bridge.predictSynchronously(payload)
                try prediction.validate()

                try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))
                XCTAssertTrue(
                    Self.stateDirectoryHasArtifact(temporaryDirectory),
                    "\(plan.pluginName) PyTorch train did not create a persistence artifact"
                )

                let reloadedBridge = Self.bridge(
                    pluginName: plan.pluginName,
                    framework: .pyTorch,
                    executable: pythonExecutable,
                    stateDirectory: temporaryDirectory
                )
                let reloadedPrediction = try reloadedBridge.predictSynchronously(payload)
                try reloadedPrediction.validate()
            } catch {
                XCTFail("\(plan.pluginName) PyTorch backend failed: \(error)")
            }
        }
    }

    func testEveryDeclaredTensorFlowBackendPredictsTrainsPersistsAndReloadsWhenTensorFlowIsInstalled() throws {
        let pythonExecutable = try BackendPythonTestSupport.requireTensorFlowMetalPython()
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.tensorFlowMetal) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "tensorflow")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(
                    pluginName: plan.pluginName,
                    framework: .tensorFlow,
                    executable: pythonExecutable,
                    stateDirectory: temporaryDirectory
                )
                let payload = Self.inferencePayload(pluginName: plan.pluginName, framework: .tensorFlow, includeText: false)

                let prediction = try bridge.predictSynchronously(payload)
                try prediction.validate()

                try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))
                XCTAssertTrue(
                    Self.stateDirectoryHasArtifact(temporaryDirectory),
                    "\(plan.pluginName) TensorFlow train did not create a persistence artifact"
                )

                let reloadedBridge = Self.bridge(
                    pluginName: plan.pluginName,
                    framework: .tensorFlow,
                    executable: pythonExecutable,
                    stateDirectory: temporaryDirectory
                )
                let reloadedPrediction = try reloadedBridge.predictSynchronously(payload)
                try reloadedPrediction.validate()
            } catch {
                XCTFail("\(plan.pluginName) TensorFlow backend failed: \(error)")
            }
        }
    }

    func testPyTorchCheckpointManifestPinsDeterministicMetadataAndReloadsDeterministically() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let pluginName = "ai_lstm"
        let temporaryDirectory = try Self.makeTemporaryDirectory(pluginName, suffix: "torch-manifest")
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let bridge = Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: temporaryDirectory
        )
        let payload = Self.inferencePayload(pluginName: pluginName, framework: .pyTorch, includeText: false)

        try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))

        let manifest = try Self.singleCheckpointManifest(in: temporaryDirectory)
        let stateFileName = try XCTUnwrap(manifest.payload["stateFileName"] as? String)
        let stateURL = temporaryDirectory.appendingPathComponent(stateFileName)
        let stateSize = try XCTUnwrap(try stateURL.resourceValues(forKeys: [.fileSizeKey]).fileSize)
        XCTAssertEqual(manifest.payload["schemaVersion"] as? String, "fxai_backend_checkpoint_v1")
        XCTAssertEqual(manifest.payload["pluginName"] as? String, pluginName)
        XCTAssertEqual(manifest.payload["framework"] as? String, "pyTorch")
        XCTAssertEqual(manifest.payload["modelIdentifier"] as? String, pluginName)
        XCTAssertEqual(manifest.payload["stateFormat"] as? String, "torch_pickle_state_v1")
        XCTAssertEqual(Self.intValue(manifest.payload["stateBytes"]), stateSize)
        XCTAssertEqual((manifest.payload["stateSha256"] as? String)?.count, 64)
        XCTAssertEqual((manifest.payload["backendSha256"] as? String)?.count, 64)
        XCTAssertEqual(Self.boolValue(manifest.payload["deterministic"]), true)
        XCTAssertGreaterThan(Self.intValue(manifest.payload["stableSeed"]) ?? 0, 0)

        let firstReload = try Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: temporaryDirectory
        ).predictSynchronously(payload)
        let secondReload = try Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: temporaryDirectory
        ).predictSynchronously(payload)

        try firstReload.validate()
        try secondReload.validate()
        Self.assertPredictionsEqual(firstReload, secondReload, accuracy: 1.0e-10)
    }

    func testPyTorchCheckpointManifestRejectsCorruptStateAndKeepsPredictionValid() throws {
        let pythonExecutable = try BackendPythonTestSupport.requirePythonImporting("torch")
        let pluginName = "ai_lstm"
        let temporaryDirectory = try Self.makeTemporaryDirectory(pluginName, suffix: "torch-corrupt")
        let coldDirectory = try Self.makeTemporaryDirectory(pluginName, suffix: "torch-cold")
        defer {
            try? FileManager.default.removeItem(at: temporaryDirectory)
            try? FileManager.default.removeItem(at: coldDirectory)
        }
        let payload = Self.inferencePayload(pluginName: pluginName, framework: .pyTorch, includeText: false)
        let bridge = Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: temporaryDirectory
        )
        try bridge.trainSynchronously(Self.trainingPayload(payload: payload, label: .buy))

        let manifest = try Self.singleCheckpointManifest(in: temporaryDirectory)
        let stateFileName = try XCTUnwrap(manifest.payload["stateFileName"] as? String)
        let stateURL = temporaryDirectory.appendingPathComponent(stateFileName)
        try Data("not a valid checkpoint".utf8).write(to: stateURL, options: .atomic)

        let corruptPrediction = try Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: temporaryDirectory
        ).predictSynchronously(payload)
        let coldPrediction = try Self.bridge(
            pluginName: pluginName,
            framework: .pyTorch,
            executable: pythonExecutable,
            stateDirectory: coldDirectory
        ).predictSynchronously(payload)

        try corruptPrediction.validate()
        try coldPrediction.validate()
        Self.assertPredictionsEqual(corruptPrediction, coldPrediction, accuracy: 1.0e-10)
    }

    func testEveryDeclaredNLPBackendConsumesTextEventsAndHasNoTextFallback() throws {
        let pythonExecutable = try BackendPythonTestSupport.requireAnyPython()
        let plans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.foundationNLP) }
        XCTAssertFalse(plans.isEmpty)

        for plan in plans {
            do {
                let temporaryDirectory = try Self.makeTemporaryDirectory(plan.pluginName, suffix: "nlp")
                defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
                let bridge = Self.bridge(
                    pluginName: plan.pluginName,
                    framework: .foundationNLP,
                    executable: pythonExecutable,
                    stateDirectory: temporaryDirectory
                )
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
        executable: String,
        stateDirectory: URL
    ) -> PythonMLBackendBridge {
        PythonMLBackendBridge(
            framework: framework,
            executable: executable,
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

    private static func singleCheckpointManifest(in url: URL) throws -> (url: URL, payload: [String: Any]) {
        let contents = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        let manifests = contents.filter { $0.lastPathComponent.hasSuffix(".manifest.json") }
        XCTAssertEqual(manifests.count, 1)
        let manifestURL = try XCTUnwrap(manifests.first)
        let data = try Data(contentsOf: manifestURL)
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (manifestURL, payload)
    }

    private static func intValue(_ value: Any?) -> Int? {
        if let number = value as? NSNumber {
            return number.intValue
        }
        return value as? Int
    }

    private static func boolValue(_ value: Any?) -> Bool? {
        if let bool = value as? Bool {
            return bool
        }
        if let number = value as? NSNumber {
            return number.boolValue
        }
        return nil
    }

    private static func assertPredictionsEqual(_ lhs: PredictionV4, _ rhs: PredictionV4, accuracy: Double) {
        for (left, right) in zip(lhs.classProbabilities, rhs.classProbabilities) {
            XCTAssertEqual(left, right, accuracy: accuracy)
        }
        XCTAssertEqual(lhs.moveMeanPoints, rhs.moveMeanPoints, accuracy: accuracy)
        XCTAssertEqual(lhs.moveQ25Points, rhs.moveQ25Points, accuracy: accuracy)
        XCTAssertEqual(lhs.moveQ50Points, rhs.moveQ50Points, accuracy: accuracy)
        XCTAssertEqual(lhs.moveQ75Points, rhs.moveQ75Points, accuracy: accuracy)
        XCTAssertEqual(lhs.confidence, rhs.confidence, accuracy: accuracy)
        XCTAssertEqual(lhs.reliability, rhs.reliability, accuracy: accuracy)
    }
}
