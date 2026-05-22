import XCTest
@testable import FXDataEngine

final class MetaPersistenceTests: XCTestCase {
    func testMetaArtifactPathSanitizesLegacyUnsafeSymbolCharacters() {
        XCTAssertEqual(
            MetaArtifactPaths.metaArtifactFile(symbol: #"EUR/USD:live*?<>|"#),
            "FXAI/Meta/fxai_meta_EUR_USD_live_____.bin"
        )
        XCTAssertEqual(
            MetaArtifactPaths.metaArtifactFile(symbol: ""),
            "FXAI/Meta/fxai_meta_default.bin"
        )
    }

    func testMetaArtifactHeaderEncodesLegacyDimensionOrder() throws {
        let header = MetaArtifactHeader.expected
        let data = try MetaArtifactCodec.encodeHeader(header)
        XCTAssertEqual(data.count, MetaArtifactConstants.headerIntCount * RuntimeArtifactPayloadMaterializer.intByteCount)

        let decoded = try MetaArtifactCodec.decodeHeader(data)
        XCTAssertEqual(decoded, header)
        XCTAssertTrue(decoded.isCompatible())
        XCTAssertEqual(decoded.fieldsInLegacyOrder, [
            MetaArtifactConstants.version,
            FXDataEngineConstants.pluginRegimeBuckets,
            FXDataEngineConstants.stackFeatures,
            FXDataEngineConstants.stackHidden,
            FXDataEngineConstants.stackFeatures,
            FXDataEngineConstants.tradeGateHidden,
            FXDataEngineConstants.horizonPolicyFeatures,
            FXDataEngineConstants.horizonPolicyHidden,
            FXDataEngineConstants.policyFeatures,
            FXDataEngineConstants.policyHidden,
            FXDataEngineConstants.aiCount
        ])
    }

    func testMetaArtifactHeaderRejectsVersionAndDimensionMismatches() {
        let wrongVersion = MetaArtifactHeader(version: MetaArtifactConstants.version - 1)
        XCTAssertFalse(wrongVersion.isCompatible())
        XCTAssertEqual(wrongVersion.mismatchReasons(), ["version"])

        let wrongDimensions = MetaArtifactHeader(
            regimeCount: 99,
            stackFeatures: 12,
            aiCount: 1
        )
        XCTAssertEqual(
            wrongDimensions.mismatchReasons(),
            ["regime_count", "stack_features", "ai_count"]
        )
    }

    func testMetaArtifactSavePolicyMatchesLegacyDirtyThrottle() {
        XCTAssertFalse(MetaArtifactSavePolicy.shouldSave(
            dirty: false,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 2_000,
            nowUTC: 2_000
        ))
        XCTAssertFalse(MetaArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_899,
            nowUTC: 2_000
        ))
        XCTAssertTrue(MetaArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 1_000,
            barTimeUTC: 1_900,
            nowUTC: 2_000
        ))
        XCTAssertTrue(MetaArtifactSavePolicy.shouldSave(
            dirty: true,
            lastSaveTimeUTC: 0,
            barTimeUTC: 0,
            nowUTC: 0
        ))
    }
}
