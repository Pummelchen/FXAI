// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXImporter",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXImporter", targets: ["FXImporter"]),
        .library(name: "FXImporterAPI", targets: ["FXImporterAPI"])
    ],
    dependencies: [
        .package(path: "../FXMT5Bridge")
    ],
    targets: [
        .target(name: "FXImporterAPI"),
        .target(name: "FXImporter", dependencies: ["FXImporterAPI", .product(name: "MT5Bridge", package: "FXMT5Bridge")]),
        .testTarget(name: "FXImporterAPITests", dependencies: ["FXImporter"]),
        .testTarget(name: "MT5BridgeTests", dependencies: [.product(name: "MT5Bridge", package: "FXMT5Bridge")])
    ],
    swiftLanguageModes: [.v6]
)
