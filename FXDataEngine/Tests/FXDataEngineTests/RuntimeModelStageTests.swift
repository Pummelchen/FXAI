import XCTest
@testable import FXDataEngine

final class RuntimeModelStageTests: XCTestCase {
    func testInitialActiveModelSelectionPicksTopTenAndExplorationArm() {
        var deploy = LiveDeploymentProfile(symbol: "EURUSD")
        deploy.maxRuntimeModels = 12
        deploy = deploy.normalized()

        let candidates = (0..<12).map { id in
            RuntimeModelCandidate(
                aiID: id,
                pluginName: "P\(id)",
                metaScore: Double(id + 1),
                voteWeight: Double(id),
                regimeObservations: id == 1 ? 0 : 100
            )
        }

        let result = RuntimeModelStageTools.selectActiveModels(
            candidates: candidates,
            ensembleMode: true,
            aiType: 0,
            signalBarUTC: 1_704_067_200,
            regimeID: 3,
            explorePercent: 100.0,
            deployProfile: deploy
        )

        XCTAssertEqual(result.preCapAIIDs, [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        XCTAssertEqual(result.activeAIIDs, result.preCapAIIDs)
        XCTAssertFalse(result.fallbackUsed)
        XCTAssertTrue(result.explorationAdded)
        XCTAssertEqual(result.runtimeModelCap, 12)
    }

    func testInitialActiveModelSelectionFallsBackToVoteWeightWhenAllPruned() {
        var deploy = LiveDeploymentProfile(symbol: "EURUSD")
        deploy.maxRuntimeModels = 4
        deploy = deploy.normalized()

        let candidates = [
            RuntimeModelCandidate(aiID: 0, pluginName: "A", pruned: true, metaScore: 10.0, voteWeight: 0.40),
            RuntimeModelCandidate(aiID: 1, pluginName: "B", pruned: true, metaScore: 8.0, voteWeight: 0.90),
            RuntimeModelCandidate(aiID: 2, pluginName: "C", available: false, pruned: true, metaScore: 20.0, voteWeight: 2.0)
        ]

        let result = RuntimeModelStageTools.selectActiveModels(
            candidates: candidates,
            ensembleMode: true,
            aiType: 0,
            signalBarUTC: 1,
            regimeID: 0,
            explorePercent: 0.0,
            deployProfile: deploy
        )

        XCTAssertEqual(result.activeAIIDs, [1])
        XCTAssertTrue(result.fallbackUsed)
        XCTAssertFalse(result.explorationAdded)
    }

    func testSingleModelSelectionRequiresAvailableRequestedPlugin() {
        let deploy = LiveDeploymentProfile(symbol: "EURUSD").normalized()
        let candidates = [
            RuntimeModelCandidate(aiID: 4, pluginName: "Requested", available: false, metaScore: 1.0),
            RuntimeModelCandidate(aiID: 5, pluginName: "Other", available: true, metaScore: 2.0)
        ]

        let unavailable = RuntimeModelStageTools.selectActiveModels(
            candidates: candidates,
            ensembleMode: false,
            aiType: 4,
            signalBarUTC: 1,
            regimeID: 0,
            explorePercent: 100.0,
            deployProfile: deploy
        )
        XCTAssertTrue(unavailable.activeAIIDs.isEmpty)

        let available = RuntimeModelStageTools.selectActiveModels(
            candidates: candidates,
            ensembleMode: false,
            aiType: 5,
            signalBarUTC: 1,
            regimeID: 0,
            explorePercent: 100.0,
            deployProfile: deploy
        )
        XCTAssertEqual(available.activeAIIDs, [5])
    }

    func testStudentAndAdaptiveRoutingFiltersAndCutsToMaxActiveModels() {
        let candidates = [
            RuntimeModelCandidate(aiID: 0, pluginName: "Alpha", family: .linear, metaScore: 1.0),
            RuntimeModelCandidate(aiID: 1, pluginName: "Beta", family: .tree, metaScore: 1.5),
            RuntimeModelCandidate(aiID: 2, pluginName: "Gamma", family: .transformer, metaScore: 0.4),
            RuntimeModelCandidate(aiID: 3, pluginName: "Delta", family: .ruleBased, metaScore: 0.7)
        ]

        var student = StudentRouterProfile(symbol: "EURUSD")
        student.maxActiveModels = 2
        student.minMetaWeight = 0.10
        student.pluginWeightsCSV = "Alpha=1.0,Beta=1.0,Gamma=1.0,Delta=1.0"
        student = student.normalized()

        var adaptive = AdaptiveRouterProfile(symbol: "EURUSD")
        adaptive.enabled = true
        adaptive.fallbackToStudentRouterOnly = false
        adaptive.pluginGlobalWeightsCSV = "Alpha=1.20,Beta=0.20,Gamma=1.50,Delta=1.0"
        adaptive = adaptive.normalized()

        let route = RuntimeModelStageTools.routeActiveModels(
            activeAIIDs: [0, 1, 2, 3],
            candidates: candidates,
            ensembleMode: true,
            studentRouter: student,
            adaptiveProfile: adaptive,
            adaptiveRegimeState: .reset,
            adaptiveRouterEnabled: true
        )

        XCTAssertTrue(route.adaptiveRouterActive)
        XCTAssertEqual(route.selectedAIIDs, [0, 3])
        XCTAssertEqual(route.routeCutoff, 0.602, accuracy: 1e-12)

        let beta = route.records.first { $0.aiID == 1 }
        XCTAssertEqual(beta?.adaptiveStatus, .suppressed)
        XCTAssertEqual(beta?.reason, "adaptive_router_suppressed")

        let gamma = route.records.first { $0.aiID == 2 }
        XCTAssertNotNil(gamma)
        XCTAssertEqual(gamma?.routedMetaWeight ?? .nan, 0.516, accuracy: 1e-12)
        XCTAssertEqual(gamma?.reason, "below_route_cutoff")
    }
}
