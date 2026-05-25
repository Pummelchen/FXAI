// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXExecutionContracts",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXExecutionContracts", targets: ["FXExecutionContracts"])
    ],
    targets: [
        .target(name: "FXExecutionContracts"),
        .testTarget(name: "FXExecutionContractsTests", dependencies: ["FXExecutionContracts"])
    ],
    swiftLanguageModes: [.v6]
)
