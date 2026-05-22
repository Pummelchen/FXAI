import FXDataEngine
import Foundation

enum FactorPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.factor(.factorPCAPanel, "factor_pca_panel"),
        FXAIPluginDefinitionFactory.factor(.factorPPPValue, "factor_ppp_value"),
        FXAIPluginDefinitionFactory.factor(.factorCarry, "factor_carry"),
        FXAIPluginDefinitionFactory.factor(.factorCMVPanel, "factor_cmv_panel")
    ]
}
