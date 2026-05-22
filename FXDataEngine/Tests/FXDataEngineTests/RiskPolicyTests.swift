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

    func testSupervisorOverlayBlocksInLegacyOrder() {
        let signal = RiskPolicySignalState(policyEnterProb: 0.50)
        var aggregate = ControlPlaneAggregate()
        aggregate.maxCapitalRiskPct = 1.30
        var supervisor = PortfolioSupervisorProfile()
        supervisor.capitalRiskCapPct = 1.20

        XCTAssertEqual(
            RiskPolicyTools.supervisorOverlayDecision(
                signal: signal,
                direction: 1,
                overlay: RiskControlPlaneOverlayState(aggregate: aggregate, supervisor: supervisor, supervisorScore: 0.0)
            ).reason,
            "risk_supervisor_capital"
        )

        aggregate.maxCapitalRiskPct = 1.00
        supervisor.hardBlockScore = 1.08
        XCTAssertEqual(
            RiskPolicyTools.supervisorOverlayDecision(
                signal: signal,
                direction: 1,
                overlay: RiskControlPlaneOverlayState(aggregate: aggregate, supervisor: supervisor, supervisorScore: 1.20)
            ).reason,
            "risk_supervisor_block"
        )

        var service = SupervisorServiceState(symbol: "EURUSD")
        service.ready = true
        service.blockScore = 1.10
        XCTAssertEqual(
            RiskPolicyTools.supervisorOverlayDecision(
                signal: signal,
                direction: 1,
                overlay: RiskControlPlaneOverlayState(
                    aggregate: aggregate,
                    supervisor: supervisor,
                    serviceState: service,
                    supervisorScore: 0.20,
                    serviceScore: 1.20
                )
            ).reason,
            "risk_supervisor_service_block"
        )

        service.entryFloor = 0.60
        XCTAssertEqual(
            RiskPolicyTools.supervisorOverlayDecision(
                signal: signal,
                direction: 1,
                overlay: RiskControlPlaneOverlayState(
                    aggregate: aggregate,
                    supervisor: supervisor,
                    serviceState: service,
                    supervisorScore: 0.20,
                    serviceScore: 0.20
                )
            ).reason,
            "risk_supervisor_service_entry_floor"
        )

        service.entryFloor = 0.40
        var command = SupervisorCommandState(symbol: "EURUSD")
        command.ready = true
        command.longBlock = true
        XCTAssertEqual(
            RiskPolicyTools.supervisorOverlayDecision(
                signal: RiskPolicySignalState(policyEnterProb: 0.70),
                direction: 1,
                overlay: RiskControlPlaneOverlayState(
                    aggregate: aggregate,
                    supervisor: supervisor,
                    serviceState: service,
                    commandState: command,
                    supervisorScore: 0.20,
                    serviceScore: 0.20
                )
            ).reason,
            "risk_supervisor_command_block"
        )
    }

    func testSupervisorOverlayMultiplierMatchesLegacyScaling() {
        var service = SupervisorServiceState(symbol: "EURUSD")
        service.ready = true
        service.longEntryBudgetMultiplier = 0.80

        var command = SupervisorCommandState(symbol: "EURUSD")
        command.ready = true
        command.longEntryBudgetMultiplier = 0.90

        let multiplier = RiskPolicyTools.supervisorOverlayLotMultiplier(
            signal: RiskPolicySignalState(
                policyEnterProb: 0.60,
                policySizeMultiplier: 1.20,
                policyCapitalEfficiency: 0.50
            ),
            direction: 1,
            portfolioPressure: 0.50,
            overlay: RiskControlPlaneOverlayState(
                serviceState: service,
                commandState: command,
                supervisorScore: 0.30,
                serviceScore: 0.10,
                controlPlaneScore: 0.40,
                controlPlaneBuyScore: 0.70
            )
        )

        XCTAssertEqual(multiplier, 0.4197639465676799, accuracy: 1e-15)
    }

    func testApplyExposureCapsMatchesLegacyOrderingAndCapsLot() {
        var supervisor = PortfolioSupervisorProfile()
        supervisor.grossBudgetBias = 1.0
        supervisor.correlatedBudgetBias = 1.0
        supervisor.directionalBudgetBias = 1.0

        let blocked = RiskPolicyTools.applyExposureCaps(
            requestedLot: 0.20,
            hardCapLot: 1.00,
            exposure: RiskPortfolioExposureState(
                grossExposureLots: 0.50,
                maxPortfolioExposureLots: 0.50
            ),
            supervisor: supervisor
        )
        XCTAssertEqual(blocked.requestedLot, 0.0)
        XCTAssertEqual(blocked.reason, "risk_portfolio_cap")

        let capped = RiskPolicyTools.applyExposureCaps(
            requestedLot: 0.50,
            hardCapLot: 1.00,
            riskBudgetLot: 0.90,
            exposure: RiskPortfolioExposureState(
                grossExposureLots: 0.40,
                correlatedExposureLots: 0.20,
                directionalClusterLots: 0.10,
                maxPortfolioExposureLots: 1.00,
                maxCorrelatedExposureLots: 0.50,
                maxDirectionalClusterLots: 0.30
            ),
            supervisor: supervisor
        )
        XCTAssertEqual(capped.requestedLot, 0.20, accuracy: 1e-12)
        XCTAssertEqual(capped.hardCapLot, 0.20, accuracy: 1e-12)
        XCTAssertEqual(capped.riskBudgetLot, 0.90, accuracy: 1e-12)
        XCTAssertEqual(capped.reason, "ok")

        let invalid = RiskPolicyTools.applyExposureCaps(
            requestedLot: .nan,
            hardCapLot: 1.00,
            exposure: RiskPortfolioExposureState(),
            supervisor: supervisor
        )
        XCTAssertEqual(invalid.requestedLot, 0.0)
        XCTAssertEqual(invalid.reason, "risk_lot_invalid")
    }
}
