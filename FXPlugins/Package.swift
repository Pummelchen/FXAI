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
        .package(path: "../FXDataEngine"),
        .package(path: "../FXBacktest")
    ],
    targets: [
        .target(
            name: "FXAIPlugins",
            dependencies: [
                .product(name: "FXDataEngine", package: "FXDataEngine"),
                .product(name: "FXBacktestCore", package: "FXBacktest"),
                .product(name: "FXBacktestPlugins", package: "FXBacktest")
            ],
            path: "Sources/FXAIPlugins"
        ),
        .testTarget(
            name: "FXAIPluginsTests",
            dependencies: ["FXAIPlugins"],
            path: "Tests/FXAIPluginsTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
