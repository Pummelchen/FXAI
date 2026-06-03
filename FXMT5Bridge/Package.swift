// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXMT5Bridge",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "MT5Bridge", targets: ["MT5Bridge"])
    ],
    targets: [
        .target(name: "MT5Bridge")
    ],
    swiftLanguageModes: [.v6]
)
