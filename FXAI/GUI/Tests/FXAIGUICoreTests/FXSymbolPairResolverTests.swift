import Testing
@testable import FXAIGUICore

struct FXSymbolPairResolverTests {
    @Test
    func resolvesBrokerSuffixPair() {
        #expect(FXSymbolPairResolver.pairID(from: "EURUSD.a") == "EURUSD")
        #expect(FXSymbolPairResolver.pairID(from: "mUSDJPY") == "USDJPY")
    }

    @Test
    func rejectsUnsupportedSyntheticSymbols() {
        #expect(FXSymbolPairResolver.pairID(from: "XAUUSD") == nil)
        #expect(FXSymbolPairResolver.pairID(from: "US30.cash") == nil)
    }

    @Test
    func honorsPreferredPairUniverseWhenAmbiguous() {
        let preferred = ["EURUSD", "GBPUSD"]
        #expect(FXSymbolPairResolver.pairID(from: "broker_EURUSDm", preferredPairs: preferred) == "EURUSD")
    }
}
