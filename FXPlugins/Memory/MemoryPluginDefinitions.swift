import FXDataEngine
import Foundation

enum MemoryPluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.memory(.retrDiff, "mem_retrdiff")
    ]
}
