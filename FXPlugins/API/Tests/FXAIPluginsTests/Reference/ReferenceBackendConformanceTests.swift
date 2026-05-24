import XCTest
@testable import FXAIPlugins

final class ReferenceBackendConformanceTests: XCTestCase {
    func testPyTorchBackendsContainReferenceLayerFamilies() throws {
        let expectations: [(plugin: String, required: [String])] = [
            ("ai_mlp", ["ARCHITECTURE_MODE = \"MLP\"", "class AIMLPReferenceModel(nn.Module)", "nn.LayerNorm", "torch.optim.AdamW"]),
            ("ai_lstm", ["ARCHITECTURE_MODE = \"LSTM\"", "class AILSTMReferenceModel(nn.Module)", "nn.LSTM(", "last_cell_state"]),
            ("ai_lstmg", ["ARCHITECTURE_MODE = \"LSTMG\"", "nn.LSTM(", "self.gate", "last_gate"]),
            ("ai_gru", ["ARCHITECTURE_MODE = \"GRU\"", "nn.GRU(", "last_hidden_state"]),
            ("ai_bilstm", ["ARCHITECTURE_MODE = \"BILSTM\"", "bidirectional=True", "self.bilstm"]),
            ("ai_lstm_tcn", ["ARCHITECTURE_MODE = \"LSTM_TCN\"", "nn.LSTM(", "CausalTCNBlock"]),
            ("ai_cnn_lstm", ["ARCHITECTURE_MODE = \"CNN_LSTM\"", "nn.Conv1d", "nn.LSTM("]),
            ("ai_attn_cnn_bilstm", ["ARCHITECTURE_MODE = \"ATTN_CNN_BILSTM\"", "nn.MultiheadAttention", "self.bilstm"]),
            ("ai_tcn", ["ARCHITECTURE_MODE = \"TCN\"", "CausalTCNBlock", "dilation"]),
            ("ai_tst", ["ARCHITECTURE_MODE = \"TST\"", "nn.TransformerEncoderLayer", "nn.TransformerEncoder"]),
            ("ai_tft", ["ARCHITECTURE_MODE = \"TFT\"", "VariableSelectionNetwork", "GatedResidualNetwork", "interpretable_attention"]),
            ("ai_autoformer", ["ARCHITECTURE_MODE = \"AUTOFORMER\"", "SeriesDecomposition", "AutoformerEncoderBlock", "decomposition_consistency_loss"]),
            ("ai_patchtst", ["ARCHITECTURE_MODE = \"PATCHTST\"", "patch_embedding", "unfold"]),
            ("ai_s4", ["ARCHITECTURE_MODE = \"S4\"", "S4DLayer", "F.conv1d"]),
            ("ai_stmn", ["ARCHITECTURE_MODE = \"STMN\"", "memory_slots", "write_cell"]),
            ("ai_chronos", ["ARCHITECTURE_MODE = \"CHRONOS\"", "ChronosTokenizer", "causal_transformer", "last_token_reconstruction"]),
            ("ai_timesfm", ["ARCHITECTURE_MODE = \"TIMESFM\"", "TimesFMPatchExtractor", "horizon_quantiles", "checkpoint_metadata"]),
            ("ai_fewc", ["ARCHITECTURE_MODE = \"FEWC\"", "fisher_diagonal", "ewc_penalty"]),
            ("ai_geodesic", ["ARCHITECTURE_MODE = \"GEODESIC\"", "last_geodesic_attention", "self.landmarks"]),
            ("ai_gha", ["ARCHITECTURE_MODE = \"GHA\"", "_orthonormal_components", "last_reconstruction_error"]),
            ("ai_qcew", ["ARCHITECTURE_MODE = \"QCEW\"", "_pinball_loss", "quantile_head"]),
            ("ai_tesseract", ["ARCHITECTURE_MODE = \"TESSERACT\"", "torch.einsum", "feature_factor"]),
            ("ai_trr", ["ARCHITECTURE_MODE = \"TRR\"", "transition_logits", "last_regime_probabilities"]),
            ("ai_mythos_rdt", ["ARCHITECTURE_MODE = \"MYTHOS_RDT\"", "DecisionTrajectoryTools", "action_embedding", "last_pseudo_actions"]),
            ("wm_cfx", ["ARCHITECTURE_MODE = \"WM_CFX\"", "currency_exposure", "factor_gru", "cross_rate_decoder", "last_cross_rate_consistency"]),
            ("wm_graph", ["ARCHITECTURE_MODE = \"WM_GRAPH\"", "FXGraphTopology", "base_adjacency", "cycle_consistency_loss"]),
            ("rl_ppo", ["ARCHITECTURE_MODE = \"proximalPolicyOptimization\"", "ActorCriticPPO", "OfflineFXRolloutEnvironment", "append_offline_rollout", "ppo_clipped_loss", "state.rollout.clear()"])
        ]

        for expectation in expectations {
            let text = try readPluginFile(plugin: expectation.plugin, backend: "PyTorch", filename: "\(expectation.plugin)_torch.py")
            for required in expectation.required {
                XCTAssertTrue(text.contains(required), "\(expectation.plugin) PyTorch backend missing \(required)")
            }
        }
    }

    func testMixturePyTorchBackendsUseModernTrainableModules() throws {
        let expectations: [(plugin: String, required: [String])] = [
            ("mix_loffm", ["LoffmMixtureModule(nn.Module)", "torch.optim.AdamW", "load_balance_loss", "feature_factors"]),
            ("mix_moe_conformal", ["MoeConformalModule(nn.Module)", "torch.optim.AdamW", "split_conformal_cutoff", "load_balance_loss"])
        ]

        for expectation in expectations {
            let text = try readPluginFile(plugin: expectation.plugin, backend: "PyTorch", filename: "\(expectation.plugin)_torch.py")
            for required in expectation.required {
                XCTAssertTrue(text.contains(required), "\(expectation.plugin) PyTorch backend missing \(required)")
            }
        }
    }

    func testTensorFlowBackendsContainKerasReferenceLayers() throws {
        let expectations: [(plugin: String, required: [String])] = [
            ("ai_mlp", ["ARCHITECTURE_MODE = \"MLP\"", "tf.keras.Model", "tf.keras.layers.Dense"]),
            ("ai_lstm", ["ARCHITECTURE_MODE = \"LSTM\"", "tf.keras.Model", "tf.keras.layers.LSTM"]),
            ("ai_lstmg", ["ARCHITECTURE_MODE = \"LSTMG\"", "tf.keras.layers.LSTM", "self.gate"]),
            ("ai_gru", ["ARCHITECTURE_MODE = \"GRU\"", "tf.keras.layers.GRU"]),
            ("ai_bilstm", ["ARCHITECTURE_MODE = \"BILSTM\"", "tf.keras.layers.Bidirectional", "tf.keras.layers.LSTM"]),
            ("ai_lstm_tcn", ["ARCHITECTURE_MODE = \"LSTM_TCN\"", "tf.keras.layers.LSTM", "CausalTCNBlock"]),
            ("ai_cnn_lstm", ["ARCHITECTURE_MODE = \"CNN_LSTM\"", "tf.keras.layers.Conv1D", "tf.keras.layers.LSTM"]),
            ("ai_attn_cnn_bilstm", ["ARCHITECTURE_MODE = \"ATTN_CNN_BILSTM\"", "tf.keras.layers.MultiHeadAttention", "tf.keras.layers.Bidirectional"]),
            ("ai_tcn", ["ARCHITECTURE_MODE = \"TCN\"", "CausalTCNBlock", "tf.keras.layers.Conv1D"])
        ]

        for expectation in expectations {
            let text = try readPluginFile(plugin: expectation.plugin, backend: "TensorFlow", filename: "\(expectation.plugin)_tensorflow.py")
            for required in expectation.required {
                XCTAssertTrue(text.contains(required), "\(expectation.plugin) TensorFlow backend missing \(required)")
            }
        }
    }

    func testFoundationNLPBackendsContainEventFeatureExtraction() throws {
        for plugin in ["ai_chronos", "ai_timesfm", "ai_mythos_rdt"] {
            let text = try readPluginFile(plugin: plugin, backend: "NLP", filename: "\(plugin)_nlp.py")
            XCTAssertTrue(text.contains("NLPFeatures"), "\(plugin) NLP backend missing typed feature output")
            XCTAssertTrue(text.contains("_ngrams"), "\(plugin) NLP backend missing n-gram extraction")
            XCTAssertTrue(text.contains("currency_focus"), "\(plugin) NLP backend missing FX currency focus feature")
            XCTAssertTrue(text.contains("novelty"), "\(plugin) NLP backend missing novelty feature")
        }
    }

    func testSequenceReferencePlansDoNotDeclareProjectionOnlyMetalBackends() throws {
        let plans = Dictionary(uniqueKeysWithValues: FXAIPluginRegistry.accelerationPlans().map { ($0.pluginName, $0) })
        let sequencePlugins = [
            "ai_lstm",
            "ai_lstmg",
            "ai_cnn_lstm",
            "ai_attn_cnn_bilstm",
            "ai_lstm_tcn",
            "ai_tft",
            "ai_tst",
            "ai_patchtst",
            "ai_autoformer"
        ]

        for pluginName in sequencePlugins {
            let plan = try XCTUnwrap(plans[pluginName], pluginName)
            XCTAssertFalse(plan.declaredBackends.contains(.metal), "\(pluginName) must not expose projection-only Metal as runtime backend")
        }
    }

    func testReferencePlanCoversEveryPluginAndRevisionPass() throws {
        let planURL = packageRoot()
            .appendingPathComponent("API")
            .appendingPathComponent("Docs")
            .appendingPathComponent("PLUGIN_99_REFERENCE_IMPLEMENTATION_PLAN.md")
        let text = try String(contentsOf: planURL, encoding: .utf8)
        let rows = text.components(separatedBy: .newlines).filter { $0.hasPrefix("| `") }

        XCTAssertEqual(rows.count, 66)
        XCTAssertTrue(text.contains("Revision Pass 1"))
        XCTAssertTrue(text.contains("Revision Pass 2"))
        XCTAssertTrue(text.contains("AI/RL/world plugins"))
    }

    private func readPluginFile(plugin: String, backend: String, filename: String) throws -> String {
        let url = packageRoot()
            .appendingPathComponent(plugin)
            .appendingPathComponent(backend)
            .appendingPathComponent(filename)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func packageRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
    }
}
