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
        .package(path: "../FXDataEngine")
    ],
    targets: [
        .target(
            name: "FXAIPlugins",
            dependencies: [
                .product(name: "FXDataEngine", package: "FXDataEngine")
            ],
            path: ".",
            exclude: [
                "API/Backends/Python",
                "API/Docs",
                "API/Tests",
                "Package.swift",
                "README.md"
            ]
        ),
        .testTarget(
            name: "FXAIPluginsTests",
            dependencies: ["FXAIPlugins"],
            path: "API/Tests/FXAIPluginsTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
