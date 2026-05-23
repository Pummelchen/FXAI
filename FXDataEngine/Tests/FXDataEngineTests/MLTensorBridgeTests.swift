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

if command.get("operation") == "train":
    training = command.get("training") or {}
    inference = training.get("inference") or {}
    assert inference["framework"] == "pyTorch"
    assert inference["dataHasVolume"] is True
    assert abs(inference["priceCostPoints"] - 0.7) < 0.0001
    assert abs(inference["minMovePoints"] - 1.5) < 0.0001
    print(json.dumps({"ok": True, "prediction": None, "error": None}))
else:
    inference = command.get("inference") or {}
    assert inference["modelIdentifier"] == "bridge_test"
    assert inference["framework"] == "pyTorch"
    assert inference["horizonMinutes"] == 15
    assert inference["sequenceBars"] == 1
    assert inference["dataHasVolume"] is True
    assert abs(inference["priceCostPoints"] - 0.7) < 0.0001
    assert abs(inference["minMovePoints"] - 1.5) < 0.0001
    print(json.dumps({
        "ok": True,
        "prediction": {
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
            xWindow: []
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
}
