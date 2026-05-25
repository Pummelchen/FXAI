// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXDemoAgent",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXDemoAgentCore", targets: ["FXDemoAgentCore"]),
        .executable(name: "FXDemoAgent", targets: ["FXDemoAgent"])
    ],
    dependencies: [
        .package(path: "../FXExecutionContracts")
    ],
    targets: [
        .target(
            name: "FXDemoAgentCore",
            dependencies: [
                .product(name: "FXExecutionContracts", package: "FXExecutionContracts")
            ]
        ),
        .executableTarget(
            name: "FXDemoAgent",
            dependencies: ["FXDemoAgentCore"]
        ),
        .testTarget(
            name: "FXDemoAgentCoreTests",
            dependencies: ["FXDemoAgentCore"]
        )
    ],
    swiftLanguageModes: [.v6]
)
