import XCTest
@testable import FXDataEngine

final class RiskPolicyTests: XCTestCase {
    func testRiskPolicyKillSwitchReasonPriorityMatchesLegacyRules() {
        let config = RiskPolicyConfig(killTradeGate: 0.24, killPathRisk: 0.92, killFillRisk: 0.90)

        XCTAssertEqual(
            RiskPolicyTools.regimeKillSwitch(config: config, signal: RiskPolicySignalState(tradeGate: 0.24)).reason,
            "kill_trade_gate"
        )
        XCTAssertEqual(
            RiskPolicyTools.regimeKillSwitch(
                config: config,
                signal: RiskPolicySignalState(tradeGate: 0.50, policyAction: .exit, policyEnterProb: 0.20, policyNoTradeProb: 0.0)
            ).reason,
            "kill_policy_exit"
        )
        XCTAssertEqual(
            RiskPolicyTools.regimeKillSwitch(
                config: config,
                signal: RiskPolicySignalState(tradeGate: 0.50, pathRisk: 0.93, policyNoTradeProb: 0.0)
            ).reason,
            "kill_path_risk"
        )
        XCTAssertTrue(
            RiskPolicyTools.regimeKillSwitch(
                config: config,
                signal: RiskPolicySignalState(tradeGate: 0.50, policyEnterProb: 0.50, policyNoTradeProb: 0.0)
            ).allowed
        )
    }

    func testRiskAdmissionBlocksPreparedStateFailuresInLegacyOrder() {
        let config = RiskPolicyConfig(minConfidence: 0.52, minMacroStateQuality: 0.24)
        let passingSignal = RiskPolicySignalState(
            confidence: 0.70,
            reliability: 0.70,
            tradeGate: 0.70,
            pathRisk: 0.20,
            fillRisk: 0.20,
            hierarchyScore: 0.70,
            hierarchyConsistency: 0.70,
            hierarchyTradability: 0.70,
            hierarchyExecution: 0.70,
            macroStateQuality: 0.70,
            policyEnterProb: 0.30,
            policyNoTradeProb: 0.20
        )

        XCTAssertEqual(RiskPolicyTools.admissionDecision(config: config, signal: passingSignal, direction: -1).reason, "invalid_direction")
        XCTAssertEqual(
            RiskPolicyTools.admissionDecision(
                config: config,
                signal: passingSignal,
                direction: 1,
                systemHealth: SystemHealthState(ready: true, healthScore: 0.30, posture: .degraded)
            ).reason,
            "risk_system_health_degraded"
        )
        XCTAssertEqual(
            RiskPolicyTools.admissionDecision(
                config: config,
                signal: passingSignal,
                direction: 1,
                factorContext: PairFactorContext(ready: true, blendedScore: 0.50, biasDirection: 0)
            ).reason,
            "risk_factor_context_opposed"
        )
        XCTAssertEqual(
            RiskPolicyTools.admissionDecision(
                config: config,
                signal: RiskPolicySignalState(confidence: 0.20),
                direction: 1
            ).reason,
            "risk_confidence_floor"
        )
        XCTAssertTrue(RiskPolicyTools.admissionDecision(config: config, signal: passingSignal, direction: 1).allowed)
    }

    func testRiskAdmissionUsesDeploymentMacroAndNoTradeFloors() {
        var deployment = LiveDeploymentProfile()
        deployment.macroQualityFloor = 0.40
        deployment.policyNoTradeCap = 0.60
        let signal = RiskPolicySignalState(
            confidence: 0.80,
            reliability: 0.80,
            tradeGate: 0.80,
            pathRisk: 0.20,
            fillRisk: 0.20,
            hierarchyScore: 0.80,
            hierarchyConsistency: 0.80,
            hierarchyTradability: 0.80,
            hierarchyExecution: 0.80,
            macroStateQuality: 0.30,
            policyEnterProb: 0.30,
            policyNoTradeProb: 0.20
        )

        XCTAssertEqual(
            RiskPolicyTools.admissionDecision(
                config: RiskPolicyConfig(),
                signal: signal,
                direction: 1,
                deployment: deployment,
                macroEventLeakageSafe: true
            ).reason,
            "risk_macro_state_floor"
        )

        var noTradeSignal = signal
        noTradeSignal.macroStateQuality = 0.80
        noTradeSignal.policyNoTradeProb = 0.70
        XCTAssertEqual(
            RiskPolicyTools.admissionDecision(config: RiskPolicyConfig(), signal: noTradeSignal, direction: 1, deployment: deployment).reason,
            "risk_policy_no_trade"
        )
    }

    func testRiskConvictionMatchesLegacyBlendWithoutBrokerInputs() {
        let signal = RiskPolicySignalState(
            confidence: 0.50,
            reliability: 0.50,
            tradeGate: 0.50,
            pathRisk: 0.50,
            fillRisk: 0.50,
            minMovePoints: 1.0,
            tradeEdgePoints: 1.0,
            hierarchyScore: 0.50,
            hierarchyConsistency: 0.50,
            contextStrength: 1.0,
            macroStateQuality: 0.50,
            policySizeMultiplier: 1.0,
            policyPortfolioFit: 0.50,
            policyCapitalEfficiency: 0.50
        )

        XCTAssertEqual(RiskPolicyTools.conviction(signal: signal, direction: 1), 0.6845, accuracy: 1e-12)

        let withFactor = RiskPolicyTools.conviction(
            signal: signal,
            direction: 1,
            systemHealth: SystemHealthState(ready: true, healthScore: 1.0, posture: .healthy),
            factorContext: PairFactorContext(ready: true, blendedScore: 0.40, biasDirection: 1)
        )
        XCTAssertEqual(withFactor, 0.6845 * 0.99 * 1.10, accuracy: 1e-12)
    }

    func testRiskBudgetAndSizingResultMatchLegacyLotMathBeforeBrokerNormalization() {
        var signal = RiskPolicySignalState(tradeEdgePoints: 20.0, expectedMovePoints: 50.0)
        let budget = RiskBudgetInput(equity: 10_000.0, moneyPerPointPerLot: 10.0, riskPoints: 100.0)

        let convictionConfig = RiskPolicyConfig(
            baseLot: 0.10,
            positionSizing: .conviction,
            riskPerTradePct: 1.0,
            evThresholdPoints: 10.0
        )
        let conviction = RiskPolicyTools.sizingResult(config: convictionConfig, signal: signal, conviction: 0.50, budget: budget)
        XCTAssertEqual(conviction.riskBudgetLot, 0.10, accuracy: 1e-12)
        XCTAssertEqual(conviction.requestedLot, 0.05, accuracy: 1e-12)
        XCTAssertEqual(conviction.reason, "ok")

        let volTargetConfig = RiskPolicyConfig(
            baseLot: 0.10,
            positionSizing: .volatilityTarget,
            riskPerTradePct: 1.0,
            evThresholdPoints: 10.0
        )
        let volTarget = RiskPolicyTools.sizingResult(config: volTargetConfig, signal: signal, conviction: 0.80, budget: budget)
        XCTAssertEqual(volTarget.requestedLot, 0.08, accuracy: 1e-12)

        signal.tradeEdgePoints = 4.0
        let blocked = RiskPolicyTools.sizingResult(config: convictionConfig, signal: signal, conviction: 0.50, budget: budget)
        XCTAssertEqual(blocked.requestedLot, 0.0)
        XCTAssertEqual(blocked.reason, "risk_edge_floor")

        signal.tradeEdgePoints = 20.0
        let pressureBlocked = RiskPolicyTools.sizingResult(
            config: RiskPolicyConfig(baseLot: 0.10, maxPortfolioPressure: 0.78),
            signal: signal,
            conviction: 0.50,
            budget: budget,
            portfolioPressure: 0.80
        )
        XCTAssertEqual(pressureBlocked.requestedLot, 0.0)
        XCTAssertEqual(pressureBlocked.reason, "risk_portfolio_pressure")
    }

    func testEstimatedRiskPointsMatchLegacyFallbackAndConfiguredTarget() {
        let signal = RiskPolicySignalState(
            pathRisk: 0.50,
            fillRisk: 0.25,
            minMovePoints: 2.0,
            expectedMovePoints: 10.0
        )

        XCTAssertEqual(
            RiskPolicyTools.estimatedRiskPoints(config: RiskPolicyConfig(riskTargetMovePoints: 0.0), signal: signal),
            10.5,
            accuracy: 1e-12
        )
        XCTAssertEqual(
            RiskPolicyTools.estimatedRiskPoints(config: RiskPolicyConfig(riskTargetMovePoints: 12.0), signal: signal),
            12.0,
            accuracy: 1e-12
        )
    }

    func testPortfolioPressureUsesPreparedExposureAndControlPlaneInputs() {
        var aggregate = ControlPlaneAggregate()
        aggregate.score = 0.40
        aggregate.macroOverlap = 0.30

        var service = SupervisorServiceState()
        service.macroPressure = 0.20

        let result = RiskPolicyTools.portfolioPressure(
            exposure: RiskPortfolioExposureState(
                grossExposureLots: 0.15,
                correlatedExposureLots: 0.10,
                directionalClusterLots: 0.09,
                maxPortfolioExposureLots: 0.30,
                maxCorrelatedExposureLots: 0.20,
                maxDirectionalClusterLots: 0.18
            ),
            signal: RiskPolicySignalState(hierarchyScore: 0.80, macroStateQuality: 0.70),
            aggregate: aggregate,
            serviceState: service,
            direction: 1,
            macroEventLeakageSafe: true,
            supervisorScore: 0.20,
            serviceScore: 0.10
        )

        XCTAssertEqual(result.pressure, 0.54, accuracy: 1e-12)
        XCTAssertEqual(result.controlPlaneScore, 0.28975, accuracy: 1e-12)
    }
}
