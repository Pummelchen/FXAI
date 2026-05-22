import XCTest
@testable import FXDataEngine

final class RuntimePolicyStageTests: XCTestCase {
    func testEnsemblePolicyDecisionUsesLegacyGatePriorityAndChosenEdge() {
        var deployment = LiveDeploymentProfile(symbol: "EURUSD")
        deployment.policyTradeFloor = 0.50
        deployment.policyNoTradeCap = 0.70
        deployment.macroQualityFloor = 0.24
        deployment = deployment.normalized()

        let decision = RuntimePolicyStageTools.resolveEnsembleDecision(
            RuntimeEnsemblePolicyInputs(
                buyPercent: 62.0,
                sellPercent: 20.0,
                skipPercent: 18.0,
                ensembleBuyProbability: 0.62,
                ensembleSkipProbability: 0.18,
                averageBuyEVPoints: 2.0,
                averageSellEVPoints: 0.4,
                stackBuyEVPoints: 2.4,
                stackSellEVPoints: 0.3,
                buyPolicyScore: 0.8,
                sellPolicyScore: 0.5,
                policyGate: 0.70,
                tradeGateThreshold: 0.52,
                agreePercent: 55.0,
                evThresholdPoints: 1.0,
                hierarchyConsistency: 0.80,
                hierarchyExecutionViability: 0.70,
                macroLeakageSafe: true,
                macroQuality: 0.50,
                policyDecision: MetaPolicyDecision(
                    tradeProbability: 0.8,
                    noTradeProbability: 0.20,
                    enterProbability: 0.55,
                    action: .enter
                ),
                deployment: deployment
            )
        )

        XCTAssertEqual(decision.decision, 1)
        XCTAssertEqual(decision.reason, "policy_buy")
        XCTAssertEqual(decision.policyGateFloor, 0.55, accuracy: 1e-12)
        XCTAssertEqual(decision.chosenEdgePoints, 2.4, accuracy: 1e-12)
    }

    func testEnsemblePolicyDecisionKeepsConservativeFallbackAndBlocks() {
        let deployment = LiveDeploymentProfile(symbol: "EURUSD").normalized()
        let base = RuntimeEnsemblePolicyInputs(
            buyPercent: 60.0,
            sellPercent: 35.0,
            skipPercent: 5.0,
            ensembleBuyProbability: 0.55,
            ensembleSkipProbability: 0.20,
            averageBuyEVPoints: 1.4,
            averageSellEVPoints: 0.8,
            stackBuyEVPoints: 0.4,
            stackSellEVPoints: 0.7,
            buyPolicyScore: 0.30,
            sellPolicyScore: 0.60,
            policyGate: 0.70,
            tradeGateThreshold: 0.44,
            agreePercent: 55.0,
            evThresholdPoints: 1.0,
            hierarchyConsistency: 0.80,
            hierarchyExecutionViability: 0.70,
            macroLeakageSafe: false,
            macroQuality: 0.50,
            policyDecision: MetaPolicyDecision(
                tradeProbability: 0.7,
                noTradeProbability: 0.20,
                enterProbability: 0.45,
                action: .enter
            ),
            deployment: deployment
        )

        let fallback = RuntimePolicyStageTools.resolveEnsembleDecision(base)
        XCTAssertEqual(fallback.decision, 1)
        XCTAssertEqual(fallback.reason, "fallback_buy")
        XCTAssertEqual(fallback.chosenEdgePoints, 0.4, accuracy: 1e-12)

        var blockedByHierarchy = base
        blockedByHierarchy.hierarchyConsistency = 0.20
        XCTAssertEqual(RuntimePolicyStageTools.resolveEnsembleDecision(blockedByHierarchy).reason, "hierarchy_block")

        var blockedByGate = base
        blockedByGate.policyGate = 0.30
        XCTAssertEqual(RuntimePolicyStageTools.resolveEnsembleDecision(blockedByGate).reason, "policy_gate_floor")

        var blockedBySkip = base
        blockedBySkip.ensembleSkipProbability = 0.60
        XCTAssertEqual(RuntimePolicyStageTools.resolveEnsembleDecision(blockedBySkip).reason, "ensemble_skip_block")
    }

    func testDynamicEnsemblePostureMutatesPolicyStateAndConflictingScoreBlocks() {
        let policy = MetaPolicyDecision(
            tradeProbability: 0.8,
            noTradeProbability: 0.20,
            enterProbability: 0.60,
            sizeMultiplier: 1.0,
            action: .enter
        )
        let initial = RuntimePolicyMutableState(decision: 1, tradeGate: 0.80, policyDecision: policy)

        let caution = RuntimePolicyStageTools.applyDynamicEnsemblePosture(
            initial,
            dynamicState: DynamicEnsembleRuntimeState(
                ready: true,
                tradePosture: "CAUTION",
                abstainBias: 0.10,
                finalScore: 0.0
            ),
            cautionEnterProbabilityBuffer: 0.07
        )
        XCTAssertEqual(caution.decision, 1)
        XCTAssertEqual(caution.policyDecision.sizeMultiplier, 0.86, accuracy: 1e-12)
        XCTAssertEqual(caution.policyDecision.enterProbability, 0.53, accuracy: 1e-12)
        XCTAssertEqual(caution.policyDecision.noTradeProbability, 0.30, accuracy: 1e-12)
        XCTAssertEqual(caution.tradeGate, 0.744, accuracy: 1e-12)

        let conflict = RuntimePolicyStageTools.applyDynamicEnsemblePosture(
            initial,
            dynamicState: DynamicEnsembleRuntimeState(
                ready: true,
                tradePosture: "NORMAL",
                finalScore: -0.09
            )
        )
        XCTAssertEqual(conflict.decision, -1)

        let notApplied = RuntimePolicyStageTools.applyDynamicEnsemblePosture(
            initial,
            dynamicState: DynamicEnsembleRuntimeState(
                ready: true,
                tradePosture: "BLOCK",
                finalScore: -1.0
            ),
            dynamicEnsembleApplied: false
        )
        XCTAssertEqual(notApplied, initial)
    }

    func testAdaptiveRouterPostureMutatesPolicyStateAndBlocks() {
        let policy = MetaPolicyDecision(
            tradeProbability: 0.7,
            noTradeProbability: 0.20,
            enterProbability: 0.45,
            sizeMultiplier: 1.0,
            action: .enter
        )
        let initial = RuntimePolicyMutableState(decision: 1, tradeGate: 0.80, policyDecision: policy)

        let abstain = RuntimePolicyStageTools.applyAdaptiveRouterPosture(
            initial,
            posture: "ABSTAIN_BIAS",
            abstainBias: 0.12
        )
        XCTAssertEqual(abstain.decision, -1)
        XCTAssertEqual(abstain.policyDecision.sizeMultiplier, 0.68, accuracy: 1e-12)
        XCTAssertEqual(abstain.policyDecision.enterProbability, 0.31, accuracy: 1e-12)
        XCTAssertEqual(abstain.policyDecision.noTradeProbability, 0.38, accuracy: 1e-12)
        XCTAssertEqual(abstain.tradeGate, 0.656, accuracy: 1e-12)

        let block = RuntimePolicyStageTools.applyAdaptiveRouterPosture(
            initial,
            posture: "BLOCK",
            abstainBias: 0.0
        )
        XCTAssertEqual(block.decision, -1)
        XCTAssertEqual(block.policyDecision.sizeMultiplier, 0.25, accuracy: 1e-12)
        XCTAssertEqual(block.policyDecision.enterProbability, 0.0, accuracy: 0.0)
        XCTAssertEqual(block.policyDecision.noTradeProbability, 0.96, accuracy: 1e-12)
        XCTAssertEqual(block.tradeGate, 0.32, accuracy: 1e-12)
    }
}
