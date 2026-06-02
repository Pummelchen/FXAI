import FXExecutionContracts
import FXLiveAgentCore
import Foundation
import Testing

@Suite("FXLiveAgent contracts")
struct LiveAgentContractsTests {
    @Test func liveWorkloadAcceptsPromotedIntent() throws {
        let request = Self.request()
        let plan = try FXLiveAgentRuntime.plan(request)

        #expect(plan.apiVersion == FXLiveAgentProtocolV1.latestVersion)
        #expect(plan.acceptedIntentCount == 1)
        #expect(plan.promotionId == "promotion-1")
        #expect(plan.requiresHumanRelease)
    }

    @Test func liveWorkloadRejectsDemoAccountScope() throws {
        let demoAccount = FXExecutionAccountScope(
            environment: .demo,
            brokerAdapter: .mt5,
            accountId: "demo-1",
            accountLabel: "Demo",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        )
        let request = Self.request(account: demoAccount)

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func liveWorkloadRejectsPromotionMismatch() throws {
        let request = Self.request(
            orderIntents: [
                FXExecutionOrderIntent(
                    lineageId: "lineage-1",
                    promotionId: "wrong-promotion",
                    symbol: "EURUSD",
                    side: "buy",
                    lots: 0.01,
                    maxSlippagePoints: 1
                )
            ]
        )

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func liveWorkloadRejectsKillSwitch() throws {
        let request = Self.request(
            killSwitch: FXExecutionKillSwitchState(
                globalEnabled: false,
                accountEnabled: true,
                symbolEnabled: true,
                reason: "operator stop"
            )
        )

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func liveWorkloadRejectsOldAPIVersion() throws {
        let request = Self.request(apiVersion: "fxlive.agent.v0")

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func liveWorkloadRejectsStalePromotionEvidence() throws {
        let issuedAtUTC = Int64(Date().timeIntervalSince1970)
        let staleEvidence = FXLivePromotionEvidence(
            promotionId: "promotion-1",
            sourceBacktestRunId: "backtest-run-1",
            demoRunId: "demo-run-1",
            lineageId: "lineage-1",
            certificationRunId: "certification-1",
            approver: "operator",
            approvedAtUTC: issuedAtUTC - FXLiveAgentProtocolV1.defaultPromotionMaxAgeSeconds - 1
        )
        let request = Self.request(promotionEvidence: staleEvidence, issuedAtUTC: issuedAtUTC)

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func liveWorkloadRejectsFuturePromotionEvidence() throws {
        let issuedAtUTC = Int64(Date().timeIntervalSince1970)
        let futureEvidence = FXLivePromotionEvidence(
            promotionId: "promotion-1",
            sourceBacktestRunId: "backtest-run-1",
            demoRunId: "demo-run-1",
            lineageId: "lineage-1",
            certificationRunId: "certification-1",
            approver: "operator",
            approvedAtUTC: issuedAtUTC + 1
        )
        let request = Self.request(promotionEvidence: futureEvidence, issuedAtUTC: issuedAtUTC)

        #expect(throws: FXLiveAgentError.self) {
            _ = try FXLiveAgentRuntime.plan(request)
        }
    }

    @Test func livePackageDoesNotUseDirectDatabaseAccess() throws {
        let forbidden = [
            ["Click", "House"].joined(),
            ["click", "house://"].joined(),
            ":812" + "3",
            "812" + "3/"
        ]
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        for file in try Self.swiftFiles(under: packageRoot) {
            let text = try String(contentsOf: file, encoding: .utf8)
            for token in forbidden {
                #expect(!text.contains(token))
            }
        }
    }

    private static func request(
        apiVersion: String = FXLiveAgentProtocolV1.latestVersion,
        account: FXExecutionAccountScope = FXExecutionAccountScope(
            environment: .live,
            brokerAdapter: .mt5,
            accountId: "live-1",
            accountLabel: "Live",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        ),
        orderIntents: [FXExecutionOrderIntent] = [
            FXExecutionOrderIntent(
                lineageId: "lineage-1",
                promotionId: "promotion-1",
                symbol: "EURUSD",
                side: "buy",
                lots: 0.01,
                maxSlippagePoints: 1
            )
        ],
        killSwitch: FXExecutionKillSwitchState = FXExecutionKillSwitchState(
            globalEnabled: true,
            accountEnabled: true,
            symbolEnabled: true,
            reason: ""
        ),
        promotionEvidence: FXLivePromotionEvidence? = nil,
        issuedAtUTC: Int64 = Int64(Date().timeIntervalSince1970)
    ) -> FXLiveAgentWorkloadRequest {
        FXLiveAgentWorkloadRequest(
            apiVersion: apiVersion,
            requestId: "live-request-1",
            pluginId: "rule_buyonly",
            acceleratorId: "swiftScalar",
            parameterSetId: "parameters-1",
            account: account,
            riskLimits: FXExecutionRiskLimits(
                maxDailyLossUSD: 100,
                maxOpenTrades: 1,
                maxPositionLots: 0.01,
                maxOrderRatePerMinute: 1,
                maxSymbolExposureLots: 0.01
            ),
            killSwitch: killSwitch,
            promotionEvidence: promotionEvidence ?? FXLivePromotionEvidence(
                promotionId: "promotion-1",
                sourceBacktestRunId: "backtest-run-1",
                demoRunId: "demo-run-1",
                lineageId: "lineage-1",
                certificationRunId: "certification-1",
                approver: "operator",
                approvedAtUTC: issuedAtUTC
            ),
            orderIntents: orderIntents,
            issuedAtUTC: issuedAtUTC
        )
    }

    private static func swiftFiles(under root: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }
        var files: [URL] = []
        while let item = enumerator.nextObject() as? URL {
            guard item.pathExtension == "swift" else { continue }
            guard (try? item.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            files.append(item)
        }
        return files
    }
}
