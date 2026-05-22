import XCTest
@testable import FXDataEngine

final class PluginTransferSupportTests: XCTestCase {
    func testTransferSlotSignalsAndPriorBlendMatchLegacyFormulas() {
        let x = transferInput()

        XCTAssertEqual(PluginTransferSupportTools.inputFeature(x, featureIndex: 50), 0.8, accuracy: 0.0)
        XCTAssertEqual(PluginTransferSupportTools.transferSlotSignal(x: x, slot: 0), 0.4, accuracy: 1e-12)
        XCTAssertEqual(PluginTransferSupportTools.transferSlotSignal(x: x, slot: 1), -0.28, accuracy: 1e-12)
        XCTAssertEqual(PluginTransferSupportTools.transferSlotSignal(x: x, slot: 3), 0.0, accuracy: 0.0)

        var adapter = Array(repeating: 0.0, count: FXDataEngineConstants.sharedTransferFeatures)
        XCTAssertFalse(PluginTransferSupportTools.hasSharedAdapterSignal(adapter))
        adapter[4] = 1.0
        XCTAssertTrue(PluginTransferSupportTools.hasSharedAdapterSignal(adapter))
        XCTAssertEqual(PluginTransferSupportTools.sharedAdapterSignalStrength(adapter), 0.21, accuracy: 1e-12)

        let slots = [
            PluginTransferSlotState(observations: 12.0, alignment: 0.5, lead: 0.6, moveScale: 1.4),
            PluginTransferSlotState(observations: 24.0, alignment: -0.25, lead: 0.8, moveScale: 0.9),
            PluginTransferSlotState()
        ]
        let priors = PluginTransferSupportTools.blendTransferSlotPriors(x: x, slots: slots)

        XCTAssertEqual(priors.domainWeight, 0.412224, accuracy: 1e-12)
        XCTAssertEqual(priors.classProbabilities[LabelClass.sell.rawValue], 0.356432044424026, accuracy: 1e-12)
        XCTAssertEqual(priors.classProbabilities[LabelClass.buy.rawValue], 0.256137640353892, accuracy: 1e-12)
        XCTAssertEqual(priors.classProbabilities[LabelClass.skip.rawValue], 0.387430315222082, accuracy: 1e-12)
        XCTAssertEqual(priors.moveScaleMultiplier, 1.067772861356932, accuracy: 1e-12)
        XCTAssertEqual(priors.reliabilityBoost, 0.20, accuracy: 1e-12)
    }

    func testCrossSymbolTransferSlotUpdatesMatchLegacyEMAAndCaps() {
        let x = transferInput()
        var slots = [PluginTransferSlotState]()

        PluginTransferSupportTools.updateCrossSymbolTransferSlots(
            x: x,
            slots: &slots,
            movePoints: 2.0,
            sampleWeight: 2.0,
            minMovePoints: 1.0
        )

        XCTAssertEqual(slots.count, FXDataEngineConstants.contextTopSymbols)
        XCTAssertEqual(slots[0].observations, 1.0944, accuracy: 1e-12)
        XCTAssertEqual(slots[0].alignment, 1.0, accuracy: 0.0)
        XCTAssertEqual(slots[0].lead, 1.0, accuracy: 0.0)
        XCTAssertEqual(slots[0].moveScale, 2.5, accuracy: 0.0)
        XCTAssertEqual(slots[1].observations, 1.3072, accuracy: 1e-12)
        XCTAssertEqual(slots[1].alignment, -1.0, accuracy: 0.0)
        XCTAssertEqual(slots[1].lead, 0.0, accuracy: 0.0)
        XCTAssertEqual(slots[1].moveScale, 2.5, accuracy: 0.0)
        XCTAssertEqual(slots[2].observations, 0.0, accuracy: 0.0)

        PluginTransferSupportTools.updateCrossSymbolTransferSlots(
            x: x,
            slots: &slots,
            movePoints: -1.0,
            sampleWeight: 1.0,
            minMovePoints: 1.0
        )

        XCTAssertEqual(slots[0].observations, 1.6416, accuracy: 1e-12)
        XCTAssertEqual(slots[0].alignment, 0.945869200811318, accuracy: 1e-12)
        XCTAssertEqual(slots[0].lead, 0.972934600405659, accuracy: 1e-12)
        XCTAssertEqual(slots[0].moveScale, 2.5, accuracy: 0.0)
    }

    private func transferInput() -> [Double] {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        setFeature(&x, 50, 0.8)
        setFeature(&x, 51, 0.4)
        setFeature(&x, 52, -0.2)
        setFeature(&x, 53, 0.6)
        setFeature(&x, 54, -0.5)
        setFeature(&x, 55, -0.3)
        setFeature(&x, 56, 0.1)
        setFeature(&x, 57, -0.8)
        setFeature(&x, 65, 0.4)
        return x
    }

    private func setFeature(_ x: inout [Double], _ featureIndex: Int, _ value: Double) {
        x[featureIndex + 1] = value
    }
}
