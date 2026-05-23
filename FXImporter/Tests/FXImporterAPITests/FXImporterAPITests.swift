import FXImporterAPI
import XCTest

final class FXImporterAPITests: XCTestCase {
    func testM1BarKeepsVolumeForProvidersThatSupplyIt() {
        let bar = FXImporterM1Bar(
            sourceSymbol: "EURUSD",
            sourceTimestamp: 1_700_000_000,
            utcTimestamp: 1_699_992_800,
            open: "1.10000",
            high: "1.10010",
            low: "1.09990",
            close: "1.10005",
            volume: 42
        )

        XCTAssertEqual(bar.volume, 42)
    }

    func testConnectorDescriptorNamesProviderKindAndCapabilities() {
        let descriptor = FXImporterConnectorDescriptor(
            id: "mt5-local",
            displayName: "MetaTrader 5 Local Bridge",
            kind: .metaTrader5,
            version: "1.0",
            capabilities: FXImporterCapabilities(
                supportsSymbolDiscovery: false,
                supportsHistoricalM1OHLC: true,
                supportsLiveM1OHLC: true,
                providesBrokerServerTime: true,
                providesVolume: false
            )
        )

        XCTAssertEqual(descriptor.kind, .metaTrader5)
        XCTAssertFalse(descriptor.capabilities.supportsSymbolDiscovery)
        XCTAssertTrue(descriptor.capabilities.supportsHistoricalM1OHLC)
        XCTAssertFalse(descriptor.capabilities.providesVolume)
    }
}
