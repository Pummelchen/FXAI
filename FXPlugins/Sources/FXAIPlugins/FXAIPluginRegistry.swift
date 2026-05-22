import FXDataEngine
import Foundation

public enum FXAIPluginRegistry {
    public static func availablePlugins() -> [any FXAIPluginV4] {
        let handwritten: [any FXAIPluginV4] = [
            RuleBuyOnlyPlugin(),
            RuleSellOnlyPlugin(),
            RuleRandomPlugin(),
            RuleM1SyncPlugin(),
            MovingAverageCrossFXDataEnginePlugin(),
            FXStupidFXDataEnginePlugin()
        ]
        let generated: [any FXAIPluginV4] = FXAIGeneratedPluginAdapter.generatedPlugins()
        return (generated + handwritten).sorted { $0.manifest.aiID < $1.manifest.aiID }
    }

    public static func accelerationPlans() -> [FXPluginAccelerationPlan] {
        let handwritten = [
            RuleBuyOnlyPlugin().accelerationPlan,
            RuleSellOnlyPlugin().accelerationPlan,
            RuleRandomPlugin().accelerationPlan,
            RuleM1SyncPlugin().accelerationPlan,
            MovingAverageCrossFXDataEnginePlugin().accelerationPlan,
            FXStupidFXDataEnginePlugin().accelerationPlan
        ]
        return (FXAIGeneratedPluginAdapter.generatedAccelerationPlans() + handwritten)
            .sorted { $0.pluginName < $1.pluginName }
    }
}
