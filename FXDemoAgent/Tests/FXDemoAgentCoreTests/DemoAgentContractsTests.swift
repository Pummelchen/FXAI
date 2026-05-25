import FXDemoAgentCore
import FXExecutionContracts
import Foundation
import Testing

@Suite("FXDemoAgent contracts")
struct DemoAgentContractsTests {
    @Test func demoWorkloadAcceptsSafeDryRunIntent() throws {
        let request = Self.request()
        let plan = try FXDemoAgentRuntime.plan(request)

        #expect(plan.apiVersion == FXDemoAgentProtocolV1.latestVersion)
        #expect(plan.acceptedIntentCount == 1)
        #expect(plan.symbols == ["EURUSD"])
        #expect(plan.dryRunOnly)
    }

    @Test func demoWorkloadRejectsLiveAccountScope() throws {
        let liveAccount = FXExecutionAccountScope(
            environment: .live,
            brokerAdapter: .mt5,
            accountId: "live-1",
            accountLabel: "Live",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        )
        let request = Self.request(account: liveAccount)

        #expect(throws: FXDemoAgentError.self) {
            _ = try FXDemoAgentRuntime.plan(request)
        }
    }

    @Test func demoWorkloadRejectsKillSwitch() throws {
        let request = Self.request(
            killSwitch: FXExecutionKillSwitchState(
                globalEnabled: false,
                accountEnabled: true,
                symbolEnabled: true,
                reason: "operator stop"
            )
        )

        #expect(throws: FXDemoAgentError.self) {
            _ = try FXDemoAgentRuntime.plan(request)
        }
    }

    @Test func demoWorkloadRejectsOldAPIVersion() throws {
        let request = Self.request(apiVersion: "fxdemo.agent.v0")

        #expect(throws: FXDemoAgentError.self) {
            _ = try FXDemoAgentRuntime.plan(request)
        }
    }

    @Test func demoPackageDoesNotUseDirectDatabaseAccess() throws {
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
        apiVersion: String = FXDemoAgentProtocolV1.latestVersion,
        account: FXExecutionAccountScope = FXExecutionAccountScope(
            environment: .demo,
            brokerAdapter: .mt5,
            accountId: "demo-1",
            accountLabel: "Demo",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        ),
        killSwitch: FXExecutionKillSwitchState = FXExecutionKillSwitchState(
            globalEnabled: true,
            accountEnabled: true,
            symbolEnabled: true,
            reason: ""
        )
    ) -> FXDemoAgentWorkloadRequest {
        FXDemoAgentWorkloadRequest(
            apiVersion: apiVersion,
            requestId: "demo-request-1",
            sourceBacktestRunId: "backtest-run-1",
            lineageId: "lineage-1",
            pluginId: "rule_buyonly",
            acceleratorId: "swiftScalar",
            parameterSetId: "parameters-1",
            account: account,
            riskLimits: FXExecutionRiskLimits(
                maxDailyLossUSD: 100,
                maxOpenTrades: 1,
                maxPositionLots: 0.01,
                maxOrderRatePerMinute: 1,
                maxSymbolExposureLots: 0.01,
                requireDemoPromotionForLive: false
            ),
            killSwitch: killSwitch,
            orderIntents: [
                FXExecutionOrderIntent(
                    lineageId: "lineage-1",
                    promotionId: nil,
                    symbol: "EURUSD",
                    side: "buy",
                    lots: 0.01,
                    maxSlippagePoints: 1
                )
            ]
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
