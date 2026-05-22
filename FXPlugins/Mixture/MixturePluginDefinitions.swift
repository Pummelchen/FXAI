import FXDataEngine
import Foundation

enum MixturePluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.mixture(.loffm, "mix_loffm"),
        FXAIPluginDefinitionFactory.mixture(.moeConformal, "mix_moe_conformal")
    ]
}
