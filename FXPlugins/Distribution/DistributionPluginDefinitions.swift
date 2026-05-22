import FXDataEngine
import Foundation

enum DistributionPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.distribution(.quantile, "dist_quantile")
    ]
}
