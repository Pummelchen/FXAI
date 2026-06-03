// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "FXDatabase",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "FXDatabaseHistoryCore", targets: ["FXDatabaseHistoryCore"]),
        .library(name: "FXDatabaseFXBacktestAPI", targets: ["FXBacktestAPI"]),
        .executable(name: "FXDatabase", targets: ["FXDatabaseCLI"])
    ],
    dependencies: [
        .package(path: "../FXMT5Bridge")
    ],
    targets: [
        .target(name: "Domain"),
        .target(name: "AppCore", dependencies: ["Domain"]),
        .target(name: "Config", dependencies: ["Domain", "AppCore"]),
        .target(name: "TimeMapping", dependencies: ["Domain"]),
        .target(name: "Validation", dependencies: ["Domain", "TimeMapping"]),
        .target(name: "ClickHouse", dependencies: ["Domain", "AppCore", "Config"]),
        .target(name: "Ingestion", dependencies: ["Domain", "AppCore", "Config", .product(name: "MT5Bridge", package: "FXMT5Bridge"), "ClickHouse", "TimeMapping", "Validation"]),
        .target(name: "Verification", dependencies: ["Domain", "AppCore", "Config", .product(name: "MT5Bridge", package: "FXMT5Bridge"), "ClickHouse", "TimeMapping", "Validation", "Ingestion"]),
        .target(name: "FXDatabaseHistoryCore", dependencies: ["Domain", "ClickHouse"]),
        .target(name: "FXDatabaseHistoryMetal", dependencies: ["Domain", "FXDatabaseHistoryCore"]),
        .target(name: "Operations", dependencies: ["Domain", "AppCore", "Config", .product(name: "MT5Bridge", package: "FXMT5Bridge"), "ClickHouse", "TimeMapping", "Validation", "Ingestion", "Verification", "FXDatabaseHistoryCore"]),
        .target(name: "FXBacktestAPI"),
        .target(name: "FXBacktestAPIServer", dependencies: ["FXBacktestAPI", "AppCore", "FXDatabaseHistoryCore", "ClickHouse", "Config", "Domain", "Operations"]),
        .executableTarget(
            name: "FXDatabaseCLI",
            dependencies: ["AppCore", "Config", .product(name: "MT5Bridge", package: "FXMT5Bridge"), "ClickHouse", "Ingestion", "Verification", "FXDatabaseHistoryCore", "FXDatabaseHistoryMetal", "TimeMapping", "Operations", "FXBacktestAPIServer"]
        ),
        .testTarget(name: "DomainTests", dependencies: ["Domain"]),
        .testTarget(name: "ValidationTests", dependencies: ["Domain", "Validation", "TimeMapping", "Config"]),
        .testTarget(name: "TimeMappingTests", dependencies: ["Domain", "TimeMapping"]),
        .testTarget(name: "ClickHouseTests", dependencies: ["ClickHouse", "Domain"]),
        .testTarget(name: "IngestionTests", dependencies: ["Domain", "Ingestion", "ClickHouse", .product(name: "MT5Bridge", package: "FXMT5Bridge"), "TimeMapping"]),
        .testTarget(name: "VerificationTests", dependencies: ["Domain", "Verification", "TimeMapping"]),
        .testTarget(name: "OperationsTests", dependencies: ["Domain", "Config", "ClickHouse", "TimeMapping", "Operations"]),
        .testTarget(name: "BacktestTests", dependencies: ["Domain", "ClickHouse", "FXDatabaseHistoryCore"]),
        .testTarget(name: "FXBacktestAPITests", dependencies: ["FXBacktestAPI", "FXBacktestAPIServer", "ClickHouse", "Config", "Domain"])
    ],
    swiftLanguageModes: [.v6]
)
