import FXDataEngine
import Foundation

enum RLPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.reinforcement(.rlPPO, "rl_ppo")
    ]
}
