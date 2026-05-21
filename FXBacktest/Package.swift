// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FXBacktest",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "FXBacktest", targets: ["FXBacktest"]),
        .library(name: "FXBacktestCore", targets: ["FXBacktestCore"]),
        .library(name: "FXBacktestPlugins", targets: ["FXBacktestPlugins"])
    ],
    dependencies: [
        .package(path: "../../FXExport/FXDatabase")
    ],
    targets: [
        .target(
            name: "FXBacktestCore",
            dependencies: [
                .product(name: "FXDatabaseFXBacktestAPI", package: "FXDatabase")
            ]
        ),
        .target(
            name: "FXBacktestPlugins",
            dependencies: ["FXBacktestCore"],
            resources: [
                .copy("FXStupid/FXStupid.config.json")
            ]
        ),
        .executableTarget(
            name: "FXBacktest",
            dependencies: [
                "FXBacktestCore",
                "FXBacktestPlugins"
            ]
        ),
        .testTarget(
            name: "FXBacktestCoreTests",
            dependencies: [
                "FXBacktestCore",
                "FXBacktestPlugins",
                .product(name: "FXDatabaseFXBacktestAPI", package: "FXDatabase")
            ]
        )
    ],
    swiftLanguageModes: [.v6]
)
