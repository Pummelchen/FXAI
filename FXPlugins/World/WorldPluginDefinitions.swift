import FXDataEngine
import Foundation

enum WorldPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.world(.cfxWorld, "wm_cfx"),
        FXAIPluginDefinitionFactory.world(.graphWM, "wm_graph")
    ]
}
