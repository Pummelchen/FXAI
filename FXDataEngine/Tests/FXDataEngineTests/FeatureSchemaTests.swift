import XCTest
@testable import FXDataEngine

final class FeatureSchemaTests: XCTestCase {
    func testMigratedDimensionsMatchReferenceContract() {
        XCTAssertEqual(FXDataEngineConstants.aiFeatures, 180)
        XCTAssertEqual(FXDataEngineConstants.aiWeights, 181)
        XCTAssertEqual(FXDataEngineConstants.maxSequenceBars, 96)
        XCTAssertEqual(FXDataEngineConstants.mainMTFFeatureOffset, 84)
        XCTAssertEqual(FXDataEngineConstants.contextMTFFeatureOffset, 100)
        XCTAssertEqual(FXDataEngineConstants.macroEventFeatureOffset, 160)
    }

    func testSpreadCostSlotsAreVolumeAwareInSwiftContract() {
        let registry = FeatureRegistry()
        XCTAssertEqual(registry.group(for: 6), .volume)
        XCTAssertEqual(registry.group(for: 74), .volume)
        XCTAssertEqual(registry.group(for: 80), .volume)
        XCTAssertEqual(registry.name(for: 6), "volume_norm")
        XCTAssertFalse(registry.name(for: 68).contains("spread"))
        XCTAssertFalse(registry.name(for: 80).contains("spread"))
    }

    func testDefaultSchemasAndGroupsFollowMQLFamilyPolicyWithVolume() {
        let policy = FeatureSchemaPolicy()
        XCTAssertEqual(policy.defaultSchema(for: .linear), .sparseStat)
        XCTAssertEqual(policy.defaultSchema(for: .ruleBased), .rule)
        XCTAssertEqual(policy.defaultSchema(for: .transformer), .sequence)
        XCTAssertTrue(policy.defaultGroups(for: .linear).contains(.volume))
        XCTAssertTrue(policy.defaultGroups(for: .ruleBased).contains(.volume))
    }

    func testFeatureSchemaMaskingPreservesMTFDynamicFeatures() {
        let policy = FeatureSchemaPolicy()
        XCTAssertFalse(policy.isFeatureEnabled(featureIndex: 50, schema: .sparseStat, groups: .all))
        XCTAssertTrue(policy.isFeatureEnabled(featureIndex: 6, schema: .rule, groups: [.price, .time, .volume]))
        XCTAssertFalse(policy.isFeatureEnabled(featureIndex: 10, schema: .rule, groups: .all))
        XCTAssertTrue(policy.isFeatureEnabled(
            featureIndex: FXDataEngineConstants.mainMTFFeatureOffset,
            schema: .rule,
            groups: []
        ))
    }
}
