import XCTest
@testable import FXDataEngine

final class PluginContractTests: XCTestCase {
    func testManifestValidationRequiresCoherentCapabilities() throws {
        let valid = PluginManifestV4(
            aiID: 2,
            aiName: "SequenceModel",
            family: .transformer,
            capabilityMask: [.selfTest, .onlineLearning, .replay, .windowContext],
            minSequenceBars: 4,
            maxSequenceBars: 32
        )
        XCTAssertNoThrow(try valid.validate())
        XCTAssertEqual(valid.resolvedSequenceBars(horizonMinutes: 3), 24)

        let invalid = PluginManifestV4(
            aiID: 3,
            aiName: "BadReplay",
            family: .linear,
            capabilityMask: [.selfTest, .replay]
        )
        XCTAssertThrowsError(try invalid.validate())
    }

    func testPredictRequestValidationChecksWindowContract() throws {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(sequenceBars: 2, dataHasVolume: true)
        let valid = PredictRequestV4(valid: true, context: context, windowSize: 1, x: x, xWindow: [x])

        XCTAssertNoThrow(try valid.validate())

        let invalid = PredictRequestV4(valid: true, context: context, windowSize: 0, x: x)
        XCTAssertThrowsError(try invalid.validate())
    }

    func testMLPayloadCarriesVolumeAvailability() {
        let x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        let context = PluginContextV4(dataHasVolume: true)
        let request = PredictRequestV4(valid: true, context: context, x: x)
        let descriptor = MLBackendDescriptor(mode: .inProcess(.metal), modelIdentifier: "model")
        let payload = MLBackendFactory.inferencePayload(descriptor: descriptor, request: request)

        XCTAssertEqual(payload.framework, .metal)
        XCTAssertTrue(payload.dataHasVolume)
        XCTAssertTrue(descriptor.usesVolumeFeatures)
    }
}
