import FXDataEngine
import Foundation

enum TrendPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.trend(.trendTSMOMVol, "trend_tsmom_vol"),
        FXAIPluginDefinitionFactory.trend(.trendXSMOMRank, "trend_xsmom_rank"),
        FXAIPluginDefinitionFactory.trend(.trendVolBreakout, "trend_vol_breakout")
    ]
}
