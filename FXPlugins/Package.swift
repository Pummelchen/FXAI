// swift-tools-version: 6.3
import PackageDescription

let fxaiPluginExcludedPaths = [
    ".build",
    "Package.swift",
    "README.md",
    "API/Backends",
    "API/Docs",
    "API/Tests",
    "ai_attn_cnn_bilstm/PyTorch",
    "ai_attn_cnn_bilstm/TensorFlow",
    "ai_autoformer/PyTorch",
    "ai_bilstm/PyTorch",
    "ai_bilstm/TensorFlow",
    "ai_chronos/NLP",
    "ai_chronos/PyTorch",
    "ai_cnn_lstm/PyTorch",
    "ai_cnn_lstm/TensorFlow",
    "ai_fewc/PyTorch",
    "ai_geodesic/PyTorch",
    "ai_gha/PyTorch",
    "ai_gru/PyTorch",
    "ai_gru/TensorFlow",
    "ai_lstm/PyTorch",
    "ai_lstm/TensorFlow",
    "ai_lstm_tcn/PyTorch",
    "ai_lstm_tcn/TensorFlow",
    "ai_lstmg/PyTorch",
    "ai_lstmg/TensorFlow",
    "ai_mlp/PyTorch",
    "ai_mlp/TensorFlow",
    "ai_mythos_rdt/NLP",
    "ai_mythos_rdt/PyTorch",
    "ai_patchtst/PyTorch",
    "ai_qcew/PyTorch",
    "ai_s4/PyTorch",
    "ai_stmn/PyTorch",
    "ai_tcn/PyTorch",
    "ai_tcn/TensorFlow",
    "ai_tesseract/PyTorch",
    "ai_tft/PyTorch",
    "ai_timesfm/NLP",
    "ai_timesfm/PyTorch",
    "ai_trr/PyTorch",
    "ai_tst/PyTorch",
    "demo_plugin_template/NLP",
    "demo_plugin_template/ONNX",
    "demo_plugin_template/PyTorch",
    "demo_plugin_template/README.md",
    "demo_plugin_template/RemoteRPC",
    "demo_plugin_template/TensorFlow",
    "fx7/Backtest",
    "mix_loffm/PyTorch",
    "mix_moe_conformal/PyTorch",
    "rl_ppo/PyTorch",
    "wm_cfx/PyTorch",
    "wm_graph/PyTorch"
]

let package = Package(
    name: "FXPlugins",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXAIPlugins", targets: ["FXAIPlugins"])
    ],
    dependencies: [
        .package(path: "../FXDataEngine")
    ],
    targets: [
        .target(
            name: "FXAIPlugins",
            dependencies: [
                .product(name: "FXDataEngine", package: "FXDataEngine")
            ],
            path: ".",
            exclude: fxaiPluginExcludedPaths
        ),
        .testTarget(
            name: "FXAIPluginsTests",
            dependencies: [
                "FXAIPlugins"
            ],
            path: "API/Tests/FXAIPluginsTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
