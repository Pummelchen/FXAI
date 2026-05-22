import XCTest
@testable import FXDataEngine

final class AuditScenariosTests: XCTestCase {
    func testScenarioSpecDefaultsMatchLegacyAuditCasesWithoutPublicSpreadFields() {
        let randomWalk = AuditScenarioTools.scenarioSpec(scenarioID: 0)
        XCTAssertEqual(randomWalk.name, "random_walk")
        XCTAssertEqual(randomWalk.driftPerBar, 0.0, accuracy: 0.0)
        XCTAssertEqual(randomWalk.sigmaPerBar, 0.00018, accuracy: 0.0)
        XCTAssertEqual(randomWalk.fillRiskPoints, 1.2, accuracy: 0.0)
        XCTAssertEqual(randomWalk.worldTrendPersistence, 0.5, accuracy: 0.0)

        let monotonicUp = AuditScenarioTools.scenarioSpec(scenarioID: 5)
        XCTAssertEqual(monotonicUp.name, "monotonic_up")
        XCTAssertEqual(monotonicUp.driftPerBar, 0.00022, accuracy: 0.0)
        XCTAssertEqual(monotonicUp.sigmaPerBar, 0.00003, accuracy: 0.0)
        XCTAssertEqual(monotonicUp.fillRiskPoints, 0.8, accuracy: 0.0)

        let macro = AuditScenarioTools.scenarioSpec(scenarioID: 14)
        XCTAssertEqual(macro.name, "market_macro_event")
        XCTAssertEqual(macro.fillRiskPoints, 1.7, accuracy: 0.0)
        XCTAssertEqual(macro.macroFocus, 1.0, accuracy: 0.0)
    }

    func testWorldPlanParsesLegacyKeysIntoFillRiskAndClampsValues() {
        let plan = [
            "sigma_scale\t9",
            "drift_bias\t0.009",
            "spread_scale\t9",
            "gap_prob\t0.99",
            "gap_scale\t99",
            "flip_prob\t0.99",
            "context_corr_bias\t-9",
            "liquidity_stress\t9",
            "session_edge_focus\t9",
            "trend_persistence\t-1",
            "shock_memory\t9",
            "recovery_bias\t-9",
            "spread_shock_prob\t9",
            "spread_shock_scale\t99",
            "regime_transition_burst\t9",
            "transition_entropy\t9",
            "mean_revert_bias\t9",
            "vol_cluster_bias\t9",
            "shock_decay\t9",
            "asia_sigma_scale\t0.1",
            "london_sigma_scale\t9",
            "newyork_sigma_scale\t2.2",
            "asia_spread_scale\t0.1",
            "london_spread_scale\t9",
            "newyork_spread_scale\t2.5",
            "macro_focus\t9"
        ].joined(separator: "\n")

        let spec = AuditScenarioTools.scenarioSpec(scenarioID: 14, worldPlanTSV: plan)
        XCTAssertEqual(spec.worldSigmaScale, 3.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldDriftBias, 0.00054, accuracy: 1e-12)
        XCTAssertEqual(spec.worldFillRiskScale, 4.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldGapProbability, 0.30, accuracy: 0.0)
        XCTAssertEqual(spec.worldGapScale, 8.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldFlipProbability, 0.50, accuracy: 0.0)
        XCTAssertEqual(spec.worldContextCorrelationBias, -1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldLiquidityStress, 3.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldSessionEdgeFocus, 1.5, accuracy: 0.0)
        XCTAssertEqual(spec.worldTrendPersistence, 0.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldShockMemory, 1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldRecoveryBias, -1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldLiquidityShockProbability, 0.50, accuracy: 0.0)
        XCTAssertEqual(spec.worldLiquidityShockScale, 8.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldRegimeTransitionBurst, 1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldTransitionEntropy, 1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldMeanRevertBias, 1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldVolatilityClusterBias, 1.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldShockDecay, 1.5, accuracy: 0.0)
        XCTAssertEqual(spec.worldAsiaSigmaScale, 0.50, accuracy: 0.0)
        XCTAssertEqual(spec.worldLondonSigmaScale, 3.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldNewYorkSigmaScale, 2.2, accuracy: 0.0)
        XCTAssertEqual(spec.worldAsiaFillRiskScale, 0.50, accuracy: 0.0)
        XCTAssertEqual(spec.worldLondonFillRiskScale, 4.0, accuracy: 0.0)
        XCTAssertEqual(spec.worldNewYorkFillRiskScale, 2.5, accuracy: 0.0)
        XCTAssertEqual(spec.macroFocus, 1.5, accuracy: 0.0)
    }

    func testWorldPlanPathSessionHelpersAndSyntheticBarNormalization() {
        XCTAssertEqual(
            AuditScenarioTools.worldPlanFile(symbol: "EUR/USD?"),
            "FXAI/Offline/Promotions/fxai_world_plan_EUR_USD_.tsv"
        )

        let janFirst2024UTC = Int64(1_704_067_200)
        XCTAssertEqual(AuditScenarioTools.hourOf(timestampUTC: janFirst2024UTC + 8 * 3_600), 8)
        XCTAssertEqual(
            AuditScenarioTools.sessionEdgeStrength(timestampUTC: janFirst2024UTC + 8 * 3_600),
            1.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            AuditScenarioTools.sessionEdgeStrength(timestampUTC: janFirst2024UTC + 10 * 3_600),
            0.5,
            accuracy: 0.0
        )
        XCTAssertEqual(
            AuditScenarioTools.worldHashUnit(timestampUTC: janFirst2024UTC, salt: 5),
            0.164332,
            accuracy: 1e-12
        )
        XCTAssertEqual(AuditScenarioTools.worldSign(timestampUTC: janFirst2024UTC, salt: 5), -1.0, accuracy: 0.0)

        let spec = AuditScenarioSpec(
            worldAsiaSigmaScale: 0.1,
            worldLondonSigmaScale: 2.5,
            worldNewYorkSigmaScale: 9.0,
            worldAsiaFillRiskScale: 0.1,
            worldLondonFillRiskScale: 2.8,
            worldNewYorkFillRiskScale: 9.0
        )
        XCTAssertEqual(AuditScenarioTools.sessionSigmaScale(spec: spec, hour: 2), 0.5, accuracy: 0.0)
        XCTAssertEqual(AuditScenarioTools.sessionSigmaScale(spec: spec, hour: 8), 2.5, accuracy: 0.0)
        XCTAssertEqual(AuditScenarioTools.sessionSigmaScale(spec: spec, hour: 16), 3.0, accuracy: 0.0)
        XCTAssertEqual(AuditScenarioTools.sessionFillRiskScale(spec: spec, hour: 2), 0.5, accuracy: 0.0)
        XCTAssertEqual(AuditScenarioTools.sessionFillRiskScale(spec: spec, hour: 8), 2.8, accuracy: 0.0)
        XCTAssertEqual(AuditScenarioTools.sessionFillRiskScale(spec: spec, hour: 16), 4.0, accuracy: 0.0)

        let normalized = AuditScenarioTools.normalizeSyntheticBar(
            AuditScenarioDoubleBar(
                timestampUTC: janFirst2024UTC,
                open: 0.0,
                high: 0.5,
                low: 2.0,
                close: 1.1,
                volume: 125.0,
                fillRiskPoints: -2.0
            ),
            point: 0.0001
        )
        XCTAssertGreaterThan(normalized.open, 0.0)
        XCTAssertGreaterThanOrEqual(normalized.high, max(normalized.open, normalized.close))
        XCTAssertLessThanOrEqual(normalized.low, min(normalized.open, normalized.close))
        XCTAssertEqual(normalized.high - normalized.low, 0.0002, accuracy: 1e-12)
        XCTAssertEqual(normalized.volume, 125.0, accuracy: 0.0)
        XCTAssertEqual(normalized.fillRiskPoints, 0.0, accuracy: 0.0)
    }

    func testApplyWorldPlanToAsSeriesBarsTransformsOHLCVFillRiskAndVolume() {
        let baseTime: Int64 = 1_704_096_000
        let bars = [
            AuditScenarioDoubleBar(
                timestampUTC: baseTime + 2 * 60,
                open: 1.1020,
                high: 1.1035,
                low: 1.1010,
                close: 1.1030,
                volume: 100.0,
                fillRiskPoints: 1.0
            ),
            AuditScenarioDoubleBar(
                timestampUTC: baseTime + 60,
                open: 1.1010,
                high: 1.1025,
                low: 1.1000,
                close: 1.1020,
                volume: 100.0,
                fillRiskPoints: 1.0
            ),
            AuditScenarioDoubleBar(
                timestampUTC: baseTime,
                open: 1.1000,
                high: 1.1015,
                low: 1.0990,
                close: 1.1010,
                volume: 100.0,
                fillRiskPoints: 1.0
            )
        ]
        let spec = AuditScenarioSpec(
            sigmaPerBar: 0.00018,
            fillRiskPoints: 1.0,
            worldSigmaScale: 1.8,
            worldDriftBias: 0.00005,
            worldFillRiskScale: 2.0,
            worldGapProbability: 0.0,
            worldGapScale: 0.0,
            worldFlipProbability: 0.0,
            worldLiquidityStress: 2.0,
            worldSessionEdgeFocus: 1.0,
            worldTrendPersistence: 0.8,
            worldShockMemory: 0.4,
            worldRecoveryBias: 0.2,
            worldLiquidityShockProbability: 1.0,
            worldLiquidityShockScale: 2.0,
            worldRegimeTransitionBurst: 0.2,
            worldTransitionEntropy: 0.1,
            worldMeanRevertBias: 0.0,
            worldVolatilityClusterBias: 0.3,
            worldShockDecay: 0.5,
            worldAsiaSigmaScale: 1.0,
            worldLondonSigmaScale: 1.5,
            worldNewYorkSigmaScale: 1.0,
            worldAsiaFillRiskScale: 1.0,
            worldLondonFillRiskScale: 1.5,
            worldNewYorkFillRiskScale: 1.0
        )

        let transformed = AuditScenarioTools.applyWorldPlanToAsSeriesBars(bars, spec: spec, point: 0.0001)

        XCTAssertEqual(transformed.count, bars.count)
        XCTAssertEqual(transformed.map(\.timestampUTC), bars.map(\.timestampUTC))
        XCTAssertNotEqual(transformed[0].close, bars[0].close)
        XCTAssertGreaterThan(transformed[0].high, max(transformed[0].open, transformed[0].close))
        XCTAssertLessThan(transformed[0].low, min(transformed[0].open, transformed[0].close))
        XCTAssertGreaterThan(transformed[0].fillRiskPoints, bars[0].fillRiskPoints)
        XCTAssertLessThan(transformed[0].volume, bars[0].volume)
        XCTAssertEqual(
            AuditScenarioTools.applyWorldPlanToAsSeriesBars(bars, spec: spec, point: 0.0),
            bars
        )
    }

    func testGenerateSyntheticScenarioSeriesBuildsAsSeriesMTFContextAndVolume() throws {
        let spec = AuditScenarioTools.scenarioSpec(scenarioID: 5)
        let generated = try XCTUnwrap(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: spec,
            bars: 512,
            seed: 42,
            point: 0.0001
        ))
        let repeated = try XCTUnwrap(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: spec,
            bars: 512,
            seed: 42,
            point: 0.0001
        ))

        XCTAssertEqual(generated.primary.count, 512)
        XCTAssertTrue(generated.primary.isConsistent)
        XCTAssertGreaterThan(generated.primary.timeUTC[0], generated.primary.timeUTC[511])
        XCTAssertGreaterThan(generated.primary.close[0], generated.primary.close[511])
        XCTAssertEqual(generated.primary.close[0], repeated.primary.close[0], accuracy: 0.0)
        XCTAssertTrue(generated.primary.volume.allSatisfy { $0 > 0.0 })
        XCTAssertTrue(generated.primary.fillRiskPoints.allSatisfy { $0 > 0.0 })

        XCTAssertEqual(generated.m5.close.count, 102)
        XCTAssertEqual(generated.m15.close.count, 34)
        XCTAssertEqual(generated.m30.close.count, 17)
        XCTAssertEqual(generated.h1.close.count, 8)
        XCTAssertEqual(generated.m5.alignedIndexMap.count, 512)
        XCTAssertGreaterThan(generated.m5.alignedIndexMap.filter { $0 >= 0 }.count, 0)

        XCTAssertEqual(generated.contexts.count, 3)
        XCTAssertTrue(generated.contexts.allSatisfy(\.isConsistent))
        XCTAssertEqual(generated.contextFeatures.count, 512)
        XCTAssertEqual(
            generated.contextFeatures.extra.count,
            512 * FXDataEngineConstants.contextExtraFeatures
        )
    }

    func testGenerateMarketScenarioSeriesUsesRecentFXDatabaseM1OHLCVWindow() throws {
        let series = try makeAuditMarketSeries(count: 1_024)
        let generated = try XCTUnwrap(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
            marketSeries: series,
            bars: 512,
            point: 0.0001,
            applyWorldPlan: false
        ))

        XCTAssertEqual(generated.primary.count, 512)
        XCTAssertTrue(generated.primary.isConsistent)
        XCTAssertEqual(generated.primary.timeUTC[0], series.utcTimestamps[1_023])
        XCTAssertEqual(generated.primary.timeUTC[511], series.utcTimestamps[512])
        XCTAssertTrue(generated.primary.volume.allSatisfy { $0 > 0.0 })
        XCTAssertTrue(generated.primary.fillRiskPoints.allSatisfy { $0 > 0.0 })
        XCTAssertEqual(generated.m5.close.count, 102)
        XCTAssertEqual(generated.h1.close.count, 8)
        XCTAssertEqual(generated.contexts.count, 3)
        XCTAssertTrue(generated.contexts.allSatisfy(\.isConsistent))
        XCTAssertEqual(generated.contextFeatures.count, 512)
    }

    func testGenerateMarketScenarioSeriesScoresTrendWindowWithLegacyWindowRules() throws {
        let series = try makeAuditMarketSeries(count: 1_024)
        let generated = try XCTUnwrap(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 9),
            marketSeries: series,
            bars: 512,
            point: 0.0001,
            applyWorldPlan: false
        ))

        XCTAssertEqual(generated.primary.timeUTC[0], series.utcTimestamps[511])
        XCTAssertEqual(generated.primary.timeUTC[511], series.utcTimestamps[0])
        XCTAssertGreaterThan(generated.primary.close[0], generated.primary.close[511])
    }

    func testGenerateSyntheticScenarioSeriesRejectsUnsupportedInputs() {
        XCTAssertNil(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 0),
            bars: 511,
            seed: 1,
            point: 0.0001
        ))
        XCTAssertNil(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 0),
            bars: 512,
            seed: 1,
            point: 0.0
        ))
        XCTAssertNil(AuditScenarioTools.generateSyntheticScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
            bars: 512,
            seed: 1,
            point: 0.0001
        ))
    }

    func testGenerateMarketScenarioSeriesRejectsUnsupportedInputs() throws {
        let series = try makeAuditMarketSeries(count: 1_024)
        XCTAssertNil(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 7),
            marketSeries: series,
            bars: 512,
            point: 0.0001
        ))
        XCTAssertNil(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
            marketSeries: series,
            bars: 511,
            point: 0.0001
        ))
        XCTAssertNil(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
            marketSeries: try makeAuditMarketSeries(count: 575),
            bars: 512,
            point: 0.0001
        ))
        XCTAssertNil(AuditScenarioTools.generateMarketScenarioSeries(
            spec: AuditScenarioTools.scenarioSpec(scenarioID: 8),
            marketSeries: series,
            bars: 512,
            point: 0.0
        ))
    }

    private func makeAuditMarketSeries(count: Int) throws -> M1OHLCVSeries {
        let safeCount = max(1, count)
        let start = Int64(1_704_067_200)
        var utc = ContiguousArray<Int64>()
        var open = ContiguousArray<Int64>()
        var high = ContiguousArray<Int64>()
        var low = ContiguousArray<Int64>()
        var close = ContiguousArray<Int64>()
        var volume = ContiguousArray<UInt64>()
        utc.reserveCapacity(safeCount)
        open.reserveCapacity(safeCount)
        high.reserveCapacity(safeCount)
        low.reserveCapacity(safeCount)
        close.reserveCapacity(safeCount)
        volume.reserveCapacity(safeCount)

        var price = Int64(110_000)
        for index in 0..<safeCount {
            let move: Int64
            if index < 512 {
                move = 20
            } else if index == 512 {
                move = -8_000
            } else {
                move = index % 2 == 0 ? 4 : -3
            }
            let barOpen = price
            let barClose = max(90_000, price + move)
            utc.append(start + Int64(index * 60))
            open.append(barOpen)
            high.append(max(barOpen, barClose) + Int64(8 + index % 5))
            low.append(min(barOpen, barClose) - Int64(8 + index % 3))
            close.append(barClose)
            volume.append(UInt64(index == 700 ? 10_000 : 100 + index % 17))
            price = barClose
        }

        return try M1OHLCVSeries(
            metadata: FXMarketMetadata(
                brokerSourceId: "demo",
                sourceOrigin: "DUKASCOPY",
                logicalSymbol: "EURUSD",
                providerSymbol: "EUR/USD",
                digits: 5,
                firstUTC: utc.first,
                lastUTC: utc.last
            ),
            utcTimestamps: utc,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        )
    }
}
