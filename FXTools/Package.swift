// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXTools",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .executable(name: "FXAICertify", targets: ["FXAICertify"])
    ],
    dependencies: [
        .package(path: "../FXDatabase")
    ],
    targets: [
        .executableTarget(
            name: "FXAICertify",
            dependencies: [
                .product(name: "FXDatabaseFXBacktestAPI", package: "FXDatabase")
            ]
        ),
        .testTarget(
            name: "FXToolsTests",
            dependencies: []
        )
    ],
    swiftLanguageModes: [.v6]
)
