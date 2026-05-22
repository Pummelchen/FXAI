import FXDataEngine
import Foundation

enum SequencePluginDefinitions {
    static let all: [FXAIGeneratedPluginDefinition] = [
        FXAIPluginDefinitionFactory.sequence(.autoformer, "ai_autoformer", .transformer, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.chronos, "ai_chronos", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.geodesicAttention, "ai_geodesic", .transformer, [.pyTorchMPS], [.metal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.lstm, "ai_lstm", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.lstmg, "ai_lstmg", .recurrent, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.mlpTiny, "ai_mlp", .convolutional, [.accelerate], [.metal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.patchTST, "ai_patchtst", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.s4, "ai_s4", .stateSpace, [.pyTorchMPS], [.metal]),
        FXAIPluginDefinitionFactory.sequence(.stmn, "ai_stmn", .stateSpace, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.tcn, "ai_tcn", .convolutional, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.tft, "ai_tft", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.timesfm, "ai_timesfm", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.tst, "ai_tst", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.trr, "ai_trr", .recurrent, [.pyTorchMPS], [.accelerate]),
        FXAIPluginDefinitionFactory.distribution(.qcew, "ai_qcew"),
        FXAIPluginDefinitionFactory.sequence(.fewc, "ai_fewc", .transformer, [.pyTorchMPS], [.accelerate]),
        FXAIPluginDefinitionFactory.sequence(.gha, "ai_gha", .transformer, [.accelerate], [.pyTorchMPS]),
        FXAIPluginDefinitionFactory.sequence(.tesseract, "ai_tesseract", .transformer, [.metal], [.pyTorchMPS]),
        FXAIPluginDefinitionFactory.sequence(.cnnLSTM, "ai_cnn_lstm", .convolutional, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.attnCNNBiLSTM, "ai_attn_cnn_bilstm", .convolutional, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.gru, "ai_gru", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.bilstm, "ai_bilstm", .recurrent, [.tensorFlowMetal], [.pyTorchMPS, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.lstmTCN, "ai_lstm_tcn", .convolutional, [.pyTorchMPS], [.tensorFlowMetal, .coreMLNeuralEngine]),
        FXAIPluginDefinitionFactory.sequence(.mythosRDT, "ai_mythos_rdt", .transformer, [.pyTorchMPS], [.coreMLNeuralEngine])
    ]
}
