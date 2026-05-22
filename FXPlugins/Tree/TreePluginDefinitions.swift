import FXDataEngine
import Foundation

enum TreePluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.tree(.catboost, "tree_catboost"),
        FXAIPluginDefinitionFactory.tree(.lightgbm, "tree_lgbm"),
        FXAIPluginDefinitionFactory.tree(.xgbFast, "tree_xgb_fast"),
        FXAIPluginDefinitionFactory.tree(.xgboost, "tree_xgb"),
        FXAIPluginDefinitionFactory.tree(.treeRF, "tree_rf")
    ]
}
