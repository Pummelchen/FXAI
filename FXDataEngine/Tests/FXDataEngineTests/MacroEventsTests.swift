import XCTest
@testable import FXDataEngine

final class MacroEventsTests: XCTestCase {
    func testMacroEventToolsMirrorLegacyClassificationAndCurrencyRules() {
        XCTAssertEqual(MacroEventTools.parseEventClass("FOMC rate decision"), MacroEventClass.rates.rawValue)
        XCTAssertEqual(MacroEventTools.parseEventClass("CPI"), MacroEventClass.inflation.rawValue)
        XCTAssertEqual(MacroEventTools.parseEventClass("nonfarm payroll"), MacroEventClass.labor.rawValue)
        XCTAssertEqual(MacroEventTools.parseEventClass("services PMI"), MacroEventClass.growth.rawValue)
        XCTAssertEqual(MacroEventTools.parseEventClass("current account balance"), MacroEventClass.trade.rawValue)
        XCTAssertEqual(MacroEventTools.countryToCurrency("United States"), "USD")
        XCTAssertEqual(MacroEventTools.currencyRelevance(currency: "USD", symbol: "EURUSD"), 1.0)
        XCTAssertEqual(MacroEventTools.currencyOrientation(currency: "USD", symbol: "EURUSD"), -1.0)
        XCTAssertEqual(MacroEventTools.currencyRelevance(currency: "CNY", symbol: "AUDJPY"), 0.55)
        XCTAssertEqual(MacroEventTools.sourceTrust("official statistics bureau"), 1.0)
        XCTAssertEqual(MacroEventTools.sourceTrust("Reuters calendar"), 0.92)
    }

    func testMacroTSVParserBuildsStatsAndLeakagePolicy() throws {
        let clean = Self.cleanMacroTSV
        let cleanDataset = MacroEventDataset.parseTSV(clean)

        XCTAssertEqual(cleanDataset.records.count, 2)
        XCTAssertEqual(cleanDataset.stats.recordCount, 2)
        XCTAssertEqual(cleanDataset.stats.parseErrors, 0)
        XCTAssertEqual(cleanDataset.stats.distinctEventIDs, 2)
        XCTAssertEqual(cleanDataset.stats.distinctCurrencies, 2)
        XCTAssertEqual(cleanDataset.stats.familyRatesCount, 1)
        XCTAssertEqual(cleanDataset.stats.familyInflationCount, 1)
        XCTAssertGreaterThan(cleanDataset.stats.leakageGuardScore, 0.78)
        XCTAssertTrue(cleanDataset.leakageSafe)

        let dirtyDataset = MacroEventDataset.parseTSV(clean + "EURUSD\tbad-time\t1\t1\t0.5\t0\t0\t0\trates\n")
        XCTAssertEqual(dirtyDataset.stats.recordCount, 2)
        XCTAssertEqual(dirtyDataset.stats.parseErrors, 1)
        XCTAssertFalse(dirtyDataset.leakageSafe)
    }

    func testMacroEventFeaturesUsePreAndPostWindows() throws {
        let dataset = MacroEventDataset.parseTSV(Self.cleanMacroTSV)
        let eventTime = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))

        let pre = dataset.features(symbol: "EURUSD", sampleTimeUTC: eventTime - 30 * 60)
        XCTAssertGreaterThan(pre.preEmbargo, 0.60)
        XCTAssertLessThanOrEqual(pre.postEmbargo, FXDataEngineConstants.unitRangeFloor)
        XCTAssertEqual(pre.ratesActivity, FXDataEngineConstants.unitRangeFloor, accuracy: 1e-12)
        XCTAssertEqual(pre.eventClassBias, 0.5, accuracy: 1e-9)

        let post = dataset.features(symbol: "EURUSD", sampleTimeUTC: eventTime + 10 * 60)
        XCTAssertGreaterThan(post.postEmbargo, 0.80)
        XCTAssertGreaterThan(post.eventImportance, 0.80)
        XCTAssertGreaterThan(post.ratesActivity, 0.80)
        XCTAssertGreaterThan(post.surpriseSigned, 1.0)
        XCTAssertGreaterThan(post.surpriseAbs, 1.0)
        XCTAssertGreaterThan(post.surpriseZScore, 1.4)
        XCTAssertEqual(post.currencyRelevance, FXDataEngineConstants.unitRangeCeil)
        XCTAssertEqual(post.provenanceTrust, FXDataEngineConstants.unitRangeCeil)
    }

    func testMacroStateAndFeatureVectorMapToExistingFeatureSchemaSlots() throws {
        let dataset = MacroEventDataset.parseTSV(Self.cleanMacroTSV)
        let eventTime = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))
        let sampleTime = eventTime + 10 * 60

        let state = dataset.state(symbol: "EURUSD", sampleTimeUTC: sampleTime)
        XCTAssertLessThan(state.policyPressure, 0.0)
        XCTAssertLessThan(state.policyDivergence, 0.0)
        XCTAssertGreaterThan(state.eventDecay, 0.80)
        XCTAssertGreaterThan(state.stateQuality, 0.70)

        let vector = dataset.macroFeatureVector(symbol: "EURUSD", sampleTimeUTC: sampleTime)
        XCTAssertEqual(vector.count, FXDataEngineConstants.macroEventFeatures)
        XCTAssertEqual(vector[10], dataset.features(symbol: "EURUSD", sampleTimeUTC: sampleTime).ratesActivity)
        XCTAssertEqual(vector[14], state.policyDivergence)
        XCTAssertEqual(vector[15], state.policyPressure)
        XCTAssertEqual(vector[19], state.stateQuality)

        var fullVector = Array(repeating: 0.0, count: FXDataEngineConstants.aiFeatures)
        dataset.fillMacroFeatures(into: &fullVector, symbol: "EURUSD", sampleTimeUTC: sampleTime)
        XCTAssertEqual(fullVector[FXDataEngineConstants.macroEventFeatureOffset + 15], state.policyPressure)
    }

    func testMacroWindowScoreSamplesEventActivity() throws {
        let dataset = MacroEventDataset.parseTSV(Self.cleanMacroTSV)
        let eventTime = try XCTUnwrap(MacroEventTools.parseEventTimeUTC("2024.01.01 12:00:00"))
        let times = (0..<180).map { eventTime - 60 * 60 + Int64($0 * 60) }

        let score = dataset.windowScore(symbol: "EURUSD", sampleTimesUTC: times)

        XCTAssertGreaterThan(score, 0.05)
        XCTAssertLessThanOrEqual(score, 1.0)
    }

    private static let cleanMacroTSV = """
symbol\tevent_time\tpre_window_min\tpost_window_min\timportance\tsurprise\tactual_delta\tforecast_delta\tclass\tevent_id\tcountry\tcurrency\tsource\trevision_delta\tprior_delta\tsurprise_z
EURUSD\t2024.01.01 12:00:00\t60\t120\t0.90\t1.20\t0.50\t0.10\trates\tUSD-FOMC-1\tUS\tUSD\tofficial statistics bureau\t0.20\t-0.10\t1.00
EURUSD\t2024-01-01T12:30:00Z\t45\t90\t0.75\t-0.60\t-0.20\t0.00\tCPI\tEUR-CPI-1\tEU\tEUR\tReuters\t0.10\t0.00\t0.00

"""
}
