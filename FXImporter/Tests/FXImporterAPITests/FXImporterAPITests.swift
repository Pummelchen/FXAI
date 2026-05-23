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
        XCTAssertEqual(descriptor.apiVersion, FXImporterAPIV1.latestVersion)
        XCTAssertNoThrow(try descriptor.validateLatestAPI())
        XCTAssertFalse(descriptor.capabilities.supportsSymbolDiscovery)
        XCTAssertTrue(descriptor.capabilities.supportsHistoricalM1OHLC)
        XCTAssertFalse(descriptor.capabilities.supportsHistoricalD1OHLC)
        XCTAssertFalse(descriptor.capabilities.providesVolume)
    }

    func testConnectorDescriptorRejectsStaleAPIVersion() {
        let descriptor = FXImporterConnectorDescriptor(
            apiVersion: "fximporter.connector.v0",
            id: "stale",
            displayName: "Stale Connector",
            kind: .custom,
            version: "1.0",
            capabilities: FXImporterCapabilities(
                supportsSymbolDiscovery: true,
                supportsHistoricalM1OHLC: false,
                supportsLiveM1OHLC: false,
                providesBrokerServerTime: false,
                providesVolume: false
            )
        )

        XCTAssertThrowsError(try descriptor.validateLatestAPI()) { error in
            XCTAssertTrue(String(describing: error).contains(FXImporterAPIV1.latestVersion))
        }
    }

    func testD1BarKeepsAdjustedCloseAndVolumeForDailyHistoryProviders() {
        let bar = FXImporterD1Bar(
            sourceSymbol: "AAPL",
            sourceTimestamp: 1_704_067_200,
            utcTimestamp: 1_704_067_200,
            open: "180.10",
            high: "182.50",
            low: "179.95",
            close: "181.20",
            adjustedClose: "180.75",
            volume: 55_000_000
        )

        XCTAssertEqual(bar.adjustedClose, "180.75")
        XCTAssertEqual(bar.volume, 55_000_000)
    }
}
