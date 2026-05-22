import XCTest
@testable import FXDataEngine

final class ExposureTests: XCTestCase {
    func testParseSymbolLegsUsesHintsThenLetterFallback() {
        XCTAssertEqual(
            ExposureTools.parseSymbolLegs("broker-symbol", baseHint: "eur", quoteHint: "usd"),
            ExposureSymbolLegs(base: "EUR", quote: "USD")
        )
        XCTAssertEqual(
            ExposureTools.parseSymbolLegs("EUR/USD.r"),
            ExposureSymbolLegs(base: "EUR", quote: "USD")
        )
        XCTAssertFalse(ExposureTools.parseSymbolLegs("bad").isValidPair)
    }

    func testCorrelationExposureWeightsMatchLegacyCurrencyRules() {
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "EURUSD"), 1.0)
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "USDEUR"), 1.0)
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "EURJPY"), 0.85)
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "GBPUSD"), 0.85)
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "GBPEUR"), 0.70)
        XCTAssertEqual(ExposureTools.correlationExposureWeight(anchorSymbol: "EURUSD", otherSymbol: "AUDNZD"), 0.0)
    }

    func testRuntimeCurrencyHelpersMatchLegacyPrefixBehavior() {
        XCTAssertEqual(ExposureTools.runtimeBaseCurrency("EURUSD.r"), "EUR")
        XCTAssertEqual(ExposureTools.runtimeQuoteCurrency("EURUSD.r"), "USD")
        XCTAssertEqual(ExposureTools.runtimeBaseCurrency("EUR/USD"), "EUR")
        XCTAssertEqual(ExposureTools.runtimeQuoteCurrency("EUR/USD"), "/US")
        XCTAssertEqual(ExposureTools.runtimeBaseCurrency("bad"), "")
    }

    func testDirectionalExposureSignMatchesBuySellBaseQuoteRules() {
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: 1, currency: "EUR"), 1)
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: 1, currency: "USD"), -1)
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: 0, currency: "EUR"), -1)
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: 0, currency: "USD"), 1)
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: -1, currency: "EUR"), 0)
        XCTAssertEqual(ExposureTools.directionalExposureSign(symbol: "EURUSD", direction: 1, currency: "JPY"), 0)
    }

    func testDirectionalClusterAlignmentMatchesLegacyRules() {
        XCTAssertEqual(
            ExposureTools.directionalClusterAlignment(anchorSymbol: "EURUSD", anchorDirection: 1, otherSymbol: "EURUSD", otherDirection: 1),
            1.0
        )
        XCTAssertEqual(
            ExposureTools.directionalClusterAlignment(anchorSymbol: "EURUSD", anchorDirection: 1, otherSymbol: "USDEUR", otherDirection: 0),
            1.0
        )
        XCTAssertEqual(
            ExposureTools.directionalClusterAlignment(anchorSymbol: "EURUSD", anchorDirection: 1, otherSymbol: "EURJPY", otherDirection: 1),
            0.5
        )
        XCTAssertEqual(
            ExposureTools.directionalClusterAlignment(anchorSymbol: "EURUSD", anchorDirection: 1, otherSymbol: "EURJPY", otherDirection: 0),
            0.0
        )
    }
}
