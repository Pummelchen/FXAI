// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "FXAIGUI",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "FXAIGUICore",
            targets: ["FXAIGUICore"]
        ),
        .executable(
            name: "FXAIGUI",
            targets: ["FXAIGUIApp"]
        )
    ],
    targets: [
        .target(
            name: "FXAIGUICore",
            path: "Sources/FXAIGUICore",
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "FXAIGUIApp",
            dependencies: ["FXAIGUICore"],
            path: "Sources/FXAIGUIApp"
        ),
        .testTarget(
            name: "FXAIGUICoreTests",
            dependencies: ["FXAIGUICore"],
            path: "Tests/FXAIGUICoreTests"
        )
    ]
)
