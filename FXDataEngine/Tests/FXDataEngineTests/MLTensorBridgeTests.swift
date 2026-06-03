import XCTest
@testable import FXDataEngine

final class MLTensorBridgeTests: XCTestCase {
    func testTensorContextDescriptorMatchesLegacyStyleRules() {
        let generic = MLTensorBridgeTools.contextDescriptor(style: .generic, maxSteps: 2, horizonMinutes: 1)
        XCTAssertEqual(generic.modelDim, 16)
        XCTAssertEqual(generic.hiddenDim, 12)
        XCTAssertEqual(generic.headCount, 2)
        XCTAssertEqual(generic.headDim, 8)
        XCTAssertEqual(generic.sequenceCapacity, 4)
        XCTAssertEqual(generic.stride, 1)
        XCTAssertEqual(generic.patchSize, 1)
        XCTAssertEqual(generic.dilation, 1)
        XCTAssertEqual(generic.positionStepPenalty, 0.06, accuracy: 0.0)

        let transformer = MLTensorBridgeTools.contextDescriptor(style: .transformer, maxSteps: 30, horizonMinutes: 60)
        XCTAssertEqual(transformer.modelDim, 24)
        XCTAssertEqual(transformer.headCount, 4)
        XCTAssertEqual(transformer.headDim, 6)
        XCTAssertEqual(transformer.sequenceCapacity, 30)
        XCTAssertEqual(transformer.stride, 2)
        XCTAssertEqual(transformer.patchSize, 2)
        XCTAssertEqual(transformer.dilation, 1)

        let world = MLTensorBridgeTools.contextDescriptor(style: .world, maxSteps: 40, horizonMinutes: 30)
        XCTAssertEqual(world.modelDim, 22)
        XCTAssertEqual(world.headCount, 4)
        XCTAssertEqual(world.headDim, 5)
        XCTAssertEqual(world.sequenceCapacity, 40)
        XCTAssertEqual(world.stride, 2)
        XCTAssertEqual(world.patchSize, 1)
        XCTAssertEqual(world.positionStepPenalty, 0.04, accuracy: 0.0)
    }

    func testSequenceRuntimeDescriptorAndInputClipping() {
        let dims = MLTensorBridgeTools.contextDescriptor(style: .convolutional, maxSteps: 8, horizonMinutes: 1)
        let runtime = MLTensorBridgeTools.sequenceRuntimeDescriptor(dims: dims, normalize: false, includeCurrent: false)
        XCTAssertEqual(runtime.maxSteps, 8)
        XCTAssertEqual(runtime.stride, 1)
        XCTAssertEqual(runtime.patchSize, 2)
        XCTAssertFalse(runtime.normalize)
        XCTAssertFalse(runtime.includeCurrent)
        XCTAssertEqual(runtime.positionStepPenalty, 0.06, accuracy: 0.0)

        let clipped = MLTensorBridgeTools.clippedCurrentInput([42.0, 20.0, -20.0, .nan, 0.5])
        XCTAssertEqual(clipped.count, FXDataEngineConstants.aiWeights)
        XCTAssertEqual(clipped[0], 1.0, accuracy: 0.0)
        XCTAssertEqual(clipped[1], 8.0, accuracy: 0.0)
        XCTAssertEqual(clipped[2], -8.0, accuracy: 0.0)
        XCTAssertEqual(clipped[3], 0.0, accuracy: 0.0)
        XCTAssertEqual(clipped[4], 0.5, accuracy: 0.0)
    }

    func testPythonMLBackendBridgeRunsProcessPredictAndTrain() async throws {
        let temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fx-python-bridge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let backendURL = temporaryDirectory.appendingPathComponent("backend.py")
        let backendScript = """
import json
import sys

command = json.loads(sys.stdin.read() or "{}")
assert command["apiVersion"] == 4

if command.get("operation") == "train":
    training = command.get("training") or {}
    inference = training.get("inference") or {}
    assert inference["apiVersion"] == 4
    assert inference["framework"] == "pyTorch"
    assert inference["dataHasVolume"] is True
    assert abs(inference["priceCostPoints"] - 0.7) < 0.0001
    assert abs(inference["minMovePoints"] - 1.5) < 0.0001
    assert inference["eventTexts"][0].startswith("USD growth")
    assert inference["tokenizerContract"]["version"] == "fxai-tokenizer-v1"
    print(json.dumps({"apiVersion": 4, "ok": True, "prediction": None, "error": None}))
else:
    inference = command.get("inference") or {}
    assert inference["apiVersion"] == 4
    assert inference["modelIdentifier"] == "bridge_test"
    assert inference["framework"] == "pyTorch"
    assert inference["horizonMinutes"] == 15
    assert inference["sequenceBars"] == 1
    assert inference["dataHasVolume"] is True
    assert abs(inference["priceCostPoints"] - 0.7) < 0.0001
    assert abs(inference["minMovePoints"] - 1.5) < 0.0001
    assert inference["eventTexts"][0].startswith("USD growth")
    assert inference["textEvents"][0]["source"] == "calendar"
    print(json.dumps({
        "apiVersion": 4,
        "ok": True,
        "prediction": {
            "apiVersion": 4,
            "classProbabilities": [0.1, 0.8, 0.1],
            "moveMeanPoints": 2.0,
            "moveQ25Points": 1.5,
            "moveQ50Points": 2.0,
            "moveQ75Points": 2.5,
            "mfeMeanPoints": 2.2,
            "maeMeanPoints": 0.4,
            "hitTimeFraction": 0.5,
            "pathRisk": 0.1,
            "fillRisk": 0.0,
            "confidence": 0.8,
            "reliability": 0.7
        },
        "error": None
    }))
"""
        try backendScript.write(to: backendURL, atomically: true, encoding: .utf8)

        let bridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "python3",
            module: backendURL.path,
            modelIdentifier: "bridge_test"
        )
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        x[6] = 0.75
        let payload = MLInferencePayload(
            modelIdentifier: "bridge_test",
            framework: .pyTorch,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: 1,
            priceCostPoints: 0.7,
            minMovePoints: 1.5,
            x: x,
            xWindow: [],
            textEvents: [
                PluginTextEventV4(
                    eventTimeUTC: 1_800_000_000,
                    source: "calendar",
                    headline: "USD growth beat supports risk-on flows",
                    importance: 0.8,
                    symbols: ["EURUSD"]
                )
            ]
        )

        let prediction = try await bridge.predict(payload)
        XCTAssertEqual(prediction.classProbabilities[1], 0.8, accuracy: 0.0001)
        XCTAssertEqual(prediction.moveMeanPoints, 2.0, accuracy: 0.0001)

        let trainRequest = TrainRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                priceCostPoints: 0.7,
                minMovePoints: 1.5,
                dataHasVolume: true
            ),
            labelClass: .buy,
            movePoints: 2.0,
            sampleWeight: 1.0,
            x: x
        )
        try await bridge.train(MLTrainingPayload(inference: payload, request: trainRequest))

        let syncPrediction = try bridge.predictSynchronously(payload)
        XCTAssertEqual(syncPrediction.classProbabilities[1], 0.8, accuracy: 0.0001)
        try bridge.trainSynchronously(MLTrainingPayload(inference: payload, request: trainRequest))
    }

    func testPythonMLBackendBridgeReportsConfigurationErrorsWithoutLaunchingProcess() {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        let unsupportedFrameworkPayload = MLInferencePayload(
            modelIdentifier: "bad_bridge",
            framework: .metal,
            dataHasVolume: false,
            x: x,
            xWindow: []
        )
        let unsupportedFrameworkBridge = PythonMLBackendBridge(
            framework: .metal,
            executable: "python3",
            module: "/tmp/fxai-backend-that-must-not-run.py",
            modelIdentifier: "bad_bridge"
        )

        XCTAssertThrowsError(try unsupportedFrameworkBridge.predictSynchronously(unsupportedFrameworkPayload)) { error in
            XCTAssertEqual(
                String(describing: error),
                "external backend failed: Python bridge does not support metal; use pyTorch, tensorFlow, foundationNLP, or onnxRuntime"
            )
        }

        let emptyExecutableBridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "  ",
            module: "/tmp/fxai-backend-that-must-not-run.py",
            modelIdentifier: "bad_bridge"
        )
        let payload = MLInferencePayload(
            modelIdentifier: "bad_bridge",
            framework: .pyTorch,
            dataHasVolume: false,
            x: x,
            xWindow: []
        )

        XCTAssertThrowsError(try emptyExecutableBridge.predictSynchronously(payload)) { error in
            XCTAssertEqual(
                String(describing: error),
                "external backend failed: Python bridge executable must not be empty"
            )
        }

        let emptyModuleBridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "python3",
            module: "\n\t",
            modelIdentifier: "bad_bridge"
        )

        XCTAssertThrowsError(try emptyModuleBridge.predictSynchronously(payload)) { error in
            XCTAssertEqual(
                String(describing: error),
                "external backend failed: Python bridge module must not be empty"
            )
        }

        let badEnvironmentBridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "python3",
            module: "json",
            modelIdentifier: "bad_bridge",
            environment: ["BAD=KEY": "value"]
        )

        XCTAssertThrowsError(try badEnvironmentBridge.predictSynchronously(payload)) { error in
            XCTAssertEqual(
                String(describing: error),
                "external backend failed: Python bridge environment key is invalid: BAD=KEY"
            )
        }
    }

    func testPythonMLBackendBridgeAcceptsONNXRuntimeFramework() throws {
        let temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fx-python-bridge-onnx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let backendURL = temporaryDirectory.appendingPathComponent("backend.py")
        let backendScript = """
import json
import os
import sys

command = json.loads(sys.stdin.read() or "{}")
inference = command.get("inference") or {}
assert inference["framework"] == "onnxRuntime"
assert os.environ["FXAI_ONNX_MODEL_PATH"].endswith("model.onnx")
print(json.dumps({
    "apiVersion": 4,
    "ok": True,
    "prediction": {
        "apiVersion": 4,
        "classProbabilities": [0.2, 0.7, 0.1],
        "moveMeanPoints": 3.0,
        "moveQ25Points": 2.0,
        "moveQ50Points": 3.0,
        "moveQ75Points": 4.0,
        "mfeMeanPoints": 3.2,
        "maeMeanPoints": 0.8,
        "hitTimeFraction": 0.4,
        "pathRisk": 0.1,
        "fillRisk": 0.0,
        "confidence": 0.7,
        "reliability": 0.6
    },
    "error": None
}))
"""
        try backendScript.write(to: backendURL, atomically: true, encoding: .utf8)

        let bridge = PythonMLBackendBridge(
            framework: .onnxRuntime,
            executable: "python3",
            module: backendURL.path,
            modelIdentifier: "onnx_bridge",
            environment: ["FXAI_ONNX_MODEL_PATH": temporaryDirectory.appendingPathComponent("model.onnx").path]
        )
        let payload = Self.inferencePayload(modelIdentifier: "onnx_bridge", framework: .onnxRuntime)

        let prediction = try bridge.predictSynchronously(payload)
        XCTAssertEqual(prediction.classProbabilities[1], 0.7, accuracy: 0.0001)
        XCTAssertEqual(prediction.moveMeanPoints, 3.0, accuracy: 0.0001)
    }

    func testRemoteRPCMLBackendBridgePredictsWithInjectedTransport() throws {
        let transport = RecordingRemoteRPCTransport(
            response: RemoteRPCMLBackendResponse(
                ok: true,
                prediction: Self.samplePrediction(),
                modelIdentifier: "remote_bridge",
                modelVersion: "test-v1",
                latencyMilliseconds: 12.0
            )
        )
        let bridge = RemoteRPCMLBackendBridge(
            modelIdentifier: "remote_bridge",
            configuration: RemoteRPCMLBackendConfiguration(
                endpoint: "https://inference.example.test/fxai/predict",
                authToken: "secret",
                timeoutSeconds: 2.0
            ),
            transport: transport
        )
        let payload = MLBackendFactory.inferencePayload(
            descriptor: bridge.descriptor,
            request: Self.predictRequest()
        )

        let prediction = try bridge.predictSynchronously(payload)

        XCTAssertEqual(payload.framework, .remoteRPC)
        XCTAssertEqual(prediction.classProbabilities[1], 0.7, accuracy: 0.0001)
        XCTAssertEqual(transport.capturedRequest()?.apiVersion, FXDataEngineConstants.latestPluginAPIVersion)
        XCTAssertEqual(transport.capturedRequest()?.inference.modelIdentifier, "remote_bridge")
        XCTAssertEqual(transport.capturedConfiguration()?.authToken, "secret")
    }

    func testRemoteRPCMLBackendBridgeRejectsInvalidEndpointBeforeTransport() {
        let transport = RecordingRemoteRPCTransport(
            response: RemoteRPCMLBackendResponse(ok: true, prediction: Self.samplePrediction())
        )
        let bridge = RemoteRPCMLBackendBridge(
            modelIdentifier: "bad_remote",
            configuration: RemoteRPCMLBackendConfiguration(endpoint: "file:///tmp/fxai.sock"),
            transport: transport
        )
        let payload = Self.inferencePayload(modelIdentifier: "bad_remote", framework: .remoteRPC)

        XCTAssertThrowsError(try bridge.predictSynchronously(payload)) { error in
            XCTAssertEqual(String(describing: error), "external backend failed: Remote RPC endpoint must use http or https")
        }
        XCTAssertNil(transport.capturedRequest())
    }

    func testRemoteRPCMLBackendBridgeRejectsUnsupportedTrainingAndBadResponseVersion() {
        let staleTransport = RecordingRemoteRPCTransport(
            response: RemoteRPCMLBackendResponse(apiVersion: 3, ok: true, prediction: Self.samplePrediction())
        )
        let bridge = RemoteRPCMLBackendBridge(
            modelIdentifier: "remote_bridge",
            configuration: RemoteRPCMLBackendConfiguration(endpoint: "https://inference.example.test/fxai/predict"),
            transport: staleTransport
        )
        let payload = Self.inferencePayload(modelIdentifier: "remote_bridge", framework: .remoteRPC)

        XCTAssertThrowsError(try bridge.predictSynchronously(payload)) { error in
            XCTAssertEqual(
                String(describing: error),
                "external backend failed: Remote RPC backend API version 3 is not supported; expected 4"
            )
        }
        let wrongModelPayload = Self.inferencePayload(modelIdentifier: "other_remote_bridge", framework: .remoteRPC)
        XCTAssertThrowsError(try bridge.predictSynchronously(wrongModelPayload)) { error in
            XCTAssertEqual(String(describing: error), "validation failed: remoteRPCPayload.modelIdentifier")
        }

        let trainRequest = TrainRequestV4(
            valid: true,
            context: PluginContextV4(dataHasVolume: true),
            labelClass: .buy,
            movePoints: 1.5,
            sampleWeight: 1.0,
            x: payload.x
        )
        XCTAssertThrowsError(try bridge.trainSynchronously(MLTrainingPayload(inference: payload, request: trainRequest))) { error in
            XCTAssertEqual(String(describing: error), "external backend failed: Remote RPC training is not supported")
        }
    }

    func testPythonMLBackendBridgeTerminatesHungProcess() throws {
        let temporaryDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fx-python-bridge-timeout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }

        let backendURL = temporaryDirectory.appendingPathComponent("backend.py")
        let backendScript = """
import time

time.sleep(5)
"""
        try backendScript.write(to: backendURL, atomically: true, encoding: .utf8)

        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        let payload = MLInferencePayload(
            modelIdentifier: "timeout_bridge",
            framework: .pyTorch,
            dataHasVolume: false,
            x: x,
            xWindow: []
        )
        let bridge = PythonMLBackendBridge(
            framework: .pyTorch,
            executable: "python3",
            module: backendURL.path,
            modelIdentifier: "timeout_bridge",
            timeoutSeconds: 0.20
        )

        let start = Date()
        XCTAssertThrowsError(try bridge.predictSynchronously(payload)) { error in
            XCTAssertTrue(String(describing: error).contains("Python backend timed out"))
        }
        XCTAssertLessThan(Date().timeIntervalSince(start), 3.0)
    }

    func testMLPayloadRequiresLatestAPIVersion() throws {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        let payload = MLInferencePayload(
            apiVersion: FXDataEngineConstants.latestPluginAPIVersion - 1,
            modelIdentifier: "stale_payload",
            framework: .pyTorch,
            dataHasVolume: false,
            x: x,
            xWindow: []
        )

        XCTAssertThrowsError(try payload.validateLatestAPI()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: mlPayload.apiVersion")
        }
    }

    func testMLPayloadEnforcesSequenceWindowContract() throws {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        let missingWindow = MLInferencePayload(
            modelIdentifier: "missing_window",
            framework: .pyTorch,
            dataHasVolume: false,
            sequenceBars: 2,
            x: x,
            xWindow: []
        )
        XCTAssertThrowsError(try missingWindow.validateLatestAPI()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: mlPayload.xWindowPayload")
        }

        let oversizedWindow = MLInferencePayload(
            modelIdentifier: "oversized_window",
            framework: .pyTorch,
            dataHasVolume: false,
            sequenceBars: 2,
            x: x,
            xWindow: [x, x]
        )
        XCTAssertThrowsError(try oversizedWindow.validateLatestAPI()) { error in
            XCTAssertEqual(String(describing: error), "validation failed: mlPayload.xWindowSequence")
        }
    }

    func testMLPayloadDecodingRequiresExplicitAPIVersion() {
        let json = """
        {
          "modelIdentifier": "missing-api-version",
          "framework": "pyTorch",
          "dataHasVolume": false,
          "x": \(Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)),
          "xWindow": []
        }
        """.data(using: .utf8)!

        XCTAssertThrowsError(try JSONDecoder().decode(MLInferencePayload.self, from: json))
    }

    private static func predictRequest() -> PredictRequestV4 {
        PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                sequenceBars: 2,
                priceCostPoints: 0.7,
                minMovePoints: 1.5,
                dataHasVolume: true
            ),
            windowSize: 2,
            x: featureVector(),
            xWindow: [featureVector()]
        )
    }

    private static func inferencePayload(modelIdentifier: String, framework: MLFramework) -> MLInferencePayload {
        MLInferencePayload(
            modelIdentifier: modelIdentifier,
            framework: framework,
            dataHasVolume: true,
            horizonMinutes: 15,
            sequenceBars: 2,
            priceCostPoints: 0.7,
            minMovePoints: 1.5,
            x: featureVector(),
            xWindow: [featureVector()]
        )
    }

    private static func featureVector() -> [Double] {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[0] = 1.0
        x[6] = 0.5
        return x
    }

    private static func samplePrediction() -> PredictionV4 {
        PredictionV4(
            classProbabilities: [0.2, 0.7, 0.1],
            moveMeanPoints: 2.0,
            moveQ25Points: 1.0,
            moveQ50Points: 2.0,
            moveQ75Points: 3.0,
            mfeMeanPoints: 2.2,
            maeMeanPoints: 0.6,
            hitTimeFraction: 0.5,
            pathRisk: 0.1,
            fillRisk: 0.0,
            confidence: 0.7,
            reliability: 0.6
        )
    }
}

private final class RecordingRemoteRPCTransport: @unchecked Sendable, RemoteRPCMLBackendTransport {
    private let lock = NSLock()
    private let response: RemoteRPCMLBackendResponse
    private var request: RemoteRPCMLBackendRequest?
    private var configuration: RemoteRPCMLBackendConfiguration?

    init(response: RemoteRPCMLBackendResponse) {
        self.response = response
    }

    func send(
        _ request: RemoteRPCMLBackendRequest,
        configuration: RemoteRPCMLBackendConfiguration
    ) throws -> RemoteRPCMLBackendResponse {
        lock.lock()
        self.request = request
        self.configuration = configuration
        lock.unlock()
        return response
    }

    func capturedRequest() -> RemoteRPCMLBackendRequest? {
        lock.lock()
        defer { lock.unlock() }
        return request
    }

    func capturedConfiguration() -> RemoteRPCMLBackendConfiguration? {
        lock.lock()
        defer { lock.unlock() }
        return configuration
    }
}
