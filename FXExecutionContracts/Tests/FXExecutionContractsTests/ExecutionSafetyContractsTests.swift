import FXExecutionContracts
import Testing

@Suite("FXExecution safety contracts")
struct ExecutionSafetyContractsTests {
    @Test func liveOrdersRequirePromotionAndLineage() throws {
        let account = FXExecutionAccountScope(
            environment: .live,
            brokerAdapter: .mt5,
            accountId: "live-1",
            accountLabel: "Live",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        )
        let limits = FXExecutionRiskLimits(
            maxDailyLossUSD: 100,
            maxOpenTrades: 1,
            maxPositionLots: 0.01,
            maxOrderRatePerMinute: 1,
            maxSymbolExposureLots: 0.01
        )
        let killSwitch = FXExecutionKillSwitchState(globalEnabled: true, accountEnabled: true, symbolEnabled: true, reason: "")
        let intent = FXExecutionOrderIntent(
            lineageId: "lineage",
            promotionId: nil,
            symbol: "EURUSD",
            side: "buy",
            lots: 0.01,
            maxSlippagePoints: 1
        )

        #expect(throws: FXExecutionContractError.self) {
            try FXExecutionSafetyGate.validateOrderIntent(intent, account: account, limits: limits, killSwitch: killSwitch)
        }
    }

    @Test func killSwitchBlocksOrders() throws {
        let account = FXExecutionAccountScope(
            environment: .demo,
            brokerAdapter: .mt5,
            accountId: "demo-1",
            accountLabel: "Demo",
            baseCurrency: "USD",
            allowedSymbols: ["EURUSD"]
        )
        let limits = FXExecutionRiskLimits(
            maxDailyLossUSD: 100,
            maxOpenTrades: 1,
            maxPositionLots: 0.01,
            maxOrderRatePerMinute: 1,
            maxSymbolExposureLots: 0.01,
            requireDemoPromotionForLive: false
        )
        let killSwitch = FXExecutionKillSwitchState(globalEnabled: false, accountEnabled: true, symbolEnabled: true, reason: "operator stop")
        let intent = FXExecutionOrderIntent(
            lineageId: "lineage",
            promotionId: nil,
            symbol: "EURUSD",
            side: "buy",
            lots: 0.01,
            maxSlippagePoints: 1
        )

        #expect(throws: FXExecutionContractError.self) {
            try FXExecutionSafetyGate.validateOrderIntent(intent, account: account, limits: limits, killSwitch: killSwitch)
        }
    }
}
