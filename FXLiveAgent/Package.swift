// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXLiveAgent",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXLiveAgentCore", targets: ["FXLiveAgentCore"]),
        .executable(name: "FXLiveAgent", targets: ["FXLiveAgent"])
    ],
    dependencies: [
        .package(path: "../FXExecutionContracts")
    ],
    targets: [
        .target(
            name: "FXLiveAgentCore",
            dependencies: [
                .product(name: "FXExecutionContracts", package: "FXExecutionContracts")
            ]
        ),
        .executableTarget(
            name: "FXLiveAgent",
            dependencies: ["FXLiveAgentCore"]
        ),
        .testTarget(
            name: "FXLiveAgentCoreTests",
            dependencies: ["FXLiveAgentCore"]
        )
    ],
    swiftLanguageModes: [.v6]
)
