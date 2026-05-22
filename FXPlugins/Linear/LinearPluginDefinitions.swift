import FXDataEngine
import Foundation

enum LinearPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.linear(.enhash, "lin_enhash"),
        FXAIPluginDefinitionFactory.linear(.ftrlLogit, "lin_ftrl"),
        FXAIPluginDefinitionFactory.linear(.paLinear, "lin_pa"),
        FXAIPluginDefinitionFactory.linear(.sgdLogit, "lin_sgd"),
        FXAIPluginDefinitionFactory.linear(.linElasticLogit, "lin_elastic_logit"),
        FXAIPluginDefinitionFactory.linear(.linProfitLogit, "lin_profit_logit")
    ]
}
