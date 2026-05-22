import XCTest
@testable import FXDataEngine

final class MetaCalibrationTests: XCTestCase {
    func testPortfolioObjectiveAndFactorMatchLegacyFormula() {
        let diagnostics = MetaCalibrationPortfolioDiagnostics(
            meanEdgePoints: 2.5,
            stability: 0.8,
            correlationPenalty: 0.2,
            diversification: 0.7,
            symbolCount: 4
        )

        XCTAssertTrue(diagnostics.ready)
        XCTAssertEqual(diagnostics.objective, 0.193, accuracy: 1e-12)
        XCTAssertEqual(
            MetaCalibrationTools.portfolioEdgeNorm(diagnostics, minMovePoints: 1.25),
            0.5,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            MetaCalibrationTools.portfolioFactor(diagnostics, minMovePoints: 1.25),
            1.2483625,
            accuracy: 1e-12
        )

        let weak = MetaCalibrationPortfolioDiagnostics(
            meanEdgePoints: -1.0,
            stability: 0.2,
            correlationPenalty: 0.8,
            diversification: 0.1,
            symbolCount: 1
        )
        XCTAssertEqual(weak.objective, -0.511, accuracy: 1e-12)
        XCTAssertEqual(MetaCalibrationTools.portfolioFactor(weak, minMovePoints: 0.5), 0.6204625, accuracy: 1e-12)
        XCTAssertEqual(
            MetaCalibrationTools.portfolioFactor(MetaCalibrationPortfolioDiagnostics(), minMovePoints: 2.0),
            1.0,
            accuracy: 0.0
        )
    }

    func testRouteActionUtilityAndStateUpdateMatchLegacyFormula() {
        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteActionUtility(action: .buy, labelClass: .buy, realizedRatio: 1.5),
            1.0,
            accuracy: 0.0
        )
        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteActionUtility(action: .buy, labelClass: .sell, realizedRatio: 1.2),
            -0.9,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteActionUtility(action: .sell, labelClass: .skip, realizedRatio: 0.5),
            -0.295,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteActionUtility(action: .skip, labelClass: .skip, realizedRatio: 0.5),
            0.305,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteActionUtility(action: .skip, labelClass: .buy, realizedRatio: 3.0),
            -0.9,
            accuracy: 1e-12
        )

        let first = MetaCalibrationTools.updatedPluginRouteCell(
            MetaCalibrationPluginRouteCell(),
            labelClass: .buy,
            signal: 1,
            realizedNetPoints: 3.0,
            minMovePoints: 2.0,
            predictedEdgePoints: 0.5,
            sampleWeight: 1.25
        )
        XCTAssertEqual(first.value, 1.0, accuracy: 0.0)
        XCTAssertEqual(first.counterfactual, 0.75, accuracy: 1e-12)
        XCTAssertEqual(first.regret, 0.0, accuracy: 0.0)
        XCTAssertTrue(first.ready)
        XCTAssertEqual(first.observations, 1)

        let second = MetaCalibrationTools.updatedPluginRouteCell(
            first,
            labelClass: .sell,
            signal: 1,
            realizedNetPoints: -2.0,
            minMovePoints: 2.0,
            predictedEdgePoints: 0.25,
            sampleWeight: 0.5
        )
        XCTAssertEqual(second.value, 0.8419041881823376, accuracy: 1e-12)
        XCTAssertEqual(second.counterfactual, 0.6415287068917706, accuracy: 1e-12)
        XCTAssertEqual(second.regret, 0.03864564288876191, accuracy: 1e-12)
        XCTAssertEqual(second.observations, 2)
    }

    func testRouteFactorPruningAndMetaScoreMatchLegacyPreparedState() {
        let diagnostics = MetaCalibrationPortfolioDiagnostics(
            meanEdgePoints: 2.5,
            stability: 0.8,
            correlationPenalty: 0.2,
            diversification: 0.7,
            symbolCount: 4
        )
        var routeCell = MetaCalibrationTools.updatedPluginRouteCell(
            MetaCalibrationPluginRouteCell(),
            labelClass: .buy,
            signal: 1,
            realizedNetPoints: 3.0,
            minMovePoints: 2.0,
            predictedEdgePoints: 0.5,
            sampleWeight: 1.25
        )
        routeCell = MetaCalibrationTools.updatedPluginRouteCell(
            routeCell,
            labelClass: .sell,
            signal: 1,
            realizedNetPoints: -2.0,
            minMovePoints: 2.0,
            predictedEdgePoints: 0.25,
            sampleWeight: 0.5
        )

        XCTAssertEqual(
            MetaCalibrationTools.pluginRouteFactor(
                routeCell,
                contextTrust: 0.5,
                portfolioObjective: 0.25
            ),
            1.0185755043840794,
            accuracy: 1e-12
        )
        XCTAssertEqual(MetaCalibrationTools.pluginRouteFactor(MetaCalibrationPluginRouteCell(), contextTrust: 1.0, portfolioObjective: 1.0), 1.0)

        XCTAssertTrue(MetaCalibrationTools.isModelPruned(reliability: 0.29))
        XCTAssertTrue(MetaCalibrationTools.isModelPruned(reliability: 1.0, regimeObservations: 24, regimeEdgePoints: -0.36))
        XCTAssertTrue(MetaCalibrationTools.isModelPruned(reliability: 1.0, globalEdgeReady: true, globalEdgePoints: -0.46))
        XCTAssertFalse(MetaCalibrationTools.isModelPruned(reliability: 0.30, regimeObservations: 23, regimeEdgePoints: -0.50))

        XCTAssertEqual(
            MetaCalibrationTools.modelMetaScore(
                reliability: 1.2,
                metaWeight: 1.1,
                regimeEdgePoints: 0.75,
                contextEdgePoints: 0.5,
                contextRegret: 0.1,
                contextObservations: 32,
                portfolioDiagnostics: diagnostics,
                routeCell: routeCell,
                minMovePoints: 1.25
            ),
            3.0071278141859996,
            accuracy: 1e-12
        )
    }
}
