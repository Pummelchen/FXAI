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
}
