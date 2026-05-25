// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXBacktestAgent",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXBacktestAgentCore", targets: ["FXBacktestAgentCore"]),
        .executable(name: "FXBacktestAgent", targets: ["FXBacktestAgent"])
    ],
    targets: [
        .target(name: "FXBacktestAgentCore"),
        .executableTarget(name: "FXBacktestAgent", dependencies: ["FXBacktestAgentCore"]),
        .testTarget(name: "FXBacktestAgentCoreTests", dependencies: ["FXBacktestAgentCore"])
    ],
    swiftLanguageModes: [.v6]
)
