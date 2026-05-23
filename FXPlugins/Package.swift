// swift-tools-version: 6.3
import PackageDescription

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
            exclude: [
                "API/Backends/Python",
                "API/Docs",
                "API/Tests",
                "ai_mlp/TensorFlow",
                "ai_mlp/PyTorch",
                "mix_moe_conformal/PyTorch",
                "mix_loffm/PyTorch",
                "ai_lstm/PyTorch",
                "ai_lstm/TensorFlow",
                "ai_lstmg/PyTorch",
                "ai_lstmg/TensorFlow",
                "ai_gru/PyTorch",
                "ai_gru/TensorFlow",
                "ai_bilstm/PyTorch",
                "ai_bilstm/TensorFlow",
                "ai_lstm_tcn/PyTorch",
                "ai_lstm_tcn/TensorFlow",
                "ai_cnn_lstm/PyTorch",
                "ai_cnn_lstm/TensorFlow",
                "Package.swift",
                "README.md"
            ]
        ),
        .testTarget(
            name: "FXAIPluginsTests",
            dependencies: ["FXAIPlugins"],
            path: "API/Tests/FXAIPluginsTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
