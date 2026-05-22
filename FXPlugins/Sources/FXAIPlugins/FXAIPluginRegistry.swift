import FXDataEngine
import Foundation

public enum FXAIPluginRegistry {
    public static func availablePlugins() -> [any FXAIPluginV4] {
        [
            RuleBuyOnlyPlugin(),
            RuleSellOnlyPlugin(),
            RuleRandomPlugin(),
            RuleM1SyncPlugin(),
            MovingAverageCrossFXDataEnginePlugin(),
            FXStupidFXDataEnginePlugin()
        ]
    }

    public static func accelerationPlans() -> [FXPluginAccelerationPlan] {
        [
            RuleBuyOnlyPlugin().accelerationPlan,
            RuleSellOnlyPlugin().accelerationPlan,
            RuleRandomPlugin().accelerationPlan,
            RuleM1SyncPlugin().accelerationPlan,
            MovingAverageCrossFXDataEnginePlugin().accelerationPlan,
            FXStupidFXDataEnginePlugin().accelerationPlan
        ]
    }
}
