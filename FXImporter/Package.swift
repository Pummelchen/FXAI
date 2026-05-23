// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXImporter",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXImporter", targets: ["FXImporter"]),
        .library(name: "FXImporterAPI", targets: ["FXImporterAPI"]),
        .library(name: "MT5Bridge", targets: ["MT5Bridge"])
    ],
    targets: [
        .target(name: "FXImporterAPI"),
        .target(name: "MT5Bridge"),
        .target(name: "FXImporter", dependencies: ["FXImporterAPI", "MT5Bridge"]),
        .testTarget(name: "FXImporterAPITests", dependencies: ["FXImporter"]),
        .testTarget(name: "MT5BridgeTests", dependencies: ["MT5Bridge"])
    ],
    swiftLanguageModes: [.v6]
)
