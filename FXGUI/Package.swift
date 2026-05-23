// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "FXGUI",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(
            name: "FXGUICore",
            targets: ["FXGUICore"]
        ),
        .executable(
            name: "FXGUI",
            targets: ["FXGUIApp"]
        )
    ],
    targets: [
        .target(
            name: "FXGUICore",
            path: "Sources/FXGUICore",
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "FXGUIApp",
            dependencies: ["FXGUICore"],
            path: "Sources/FXGUIApp"
        ),
        .testTarget(
            name: "FXGUICoreTests",
            dependencies: ["FXGUICore"],
            path: "Tests/FXGUICoreTests"
        ),
        .testTarget(
            name: "FXGUIAppTests",
            dependencies: ["FXGUICore", "FXGUIApp"],
            path: "Tests/FXGUIAppTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
