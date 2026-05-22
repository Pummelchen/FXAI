// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXDataEngine",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXDataEngine", targets: ["FXDataEngine"]),
        .executable(name: "FXDataEngineCLI", targets: ["FXDataEngineCLI"])
    ],
    dependencies: [
        .package(path: "../FXDatabase")
    ],
    targets: [
        .target(
            name: "FXDataEngine",
            dependencies: [
                .product(name: "FXDatabaseFXBacktestAPI", package: "FXDatabase")
            ]
        ),
        .executableTarget(
            name: "FXDataEngineCLI",
            dependencies: ["FXDataEngine"]
        ),
        .testTarget(
            name: "FXDataEngineTests",
            dependencies: ["FXDataEngine"]
        )
    ],
    swiftLanguageModes: [.v6]
)
