// swift-tools-version: 6.3
import Foundation
import PackageDescription

let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let fxaiPluginImplementationSourceFolders = Set(["CPU", "Metal"])
let fxaiPinnedSharedRuntimeSources = [
    "ai_autoformer/CPU/FXAISequenceArchitectureCPUModel.swift"
]

func fxaiPluginSwiftSources() -> [String] {
    guard let enumerator = FileManager.default.enumerator(
        at: packageDirectory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants]
    ) else {
        return []
    }

    var sources: [String] = []
    let allowedAPIFolders = Set(["Registry", "Runtime"])
    let packagePathPrefix = packageDirectory.path + "/"

    while let url = enumerator.nextObject() as? URL {
        guard url.pathExtension == "swift" else { continue }
        guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }

        let path = url.path
        guard FileManager.default.fileExists(atPath: path) else { continue }
        guard path.hasPrefix(packagePathPrefix) else { continue }
        let relativePath = String(path.dropFirst(packagePathPrefix.count))
        guard relativePath != "Package.swift" else { continue }

        let parts = relativePath.split(separator: "/").map(String.init)
        guard let top = parts.first else { continue }

        if top == "API" {
            guard parts.count >= 3, allowedAPIFolders.contains(parts[1]) else { continue }
            sources.append(relativePath)
            continue
        }

        if parts.count == 2 {
            sources.append(relativePath)
            continue
        }

        if parts.count >= 3, fxaiPluginImplementationSourceFolders.contains(parts[1]) {
            sources.append(relativePath)
        }
    }

    return sources.sorted()
}

func fxaiPluginExcludedPaths() -> [String] {
    var excluded = Set([
        "Package.swift",
        "README.md",
        "API/Backends",
        "API/Docs",
        "API/Tests",
        "demo_plugin_template/README.md"
    ])

    guard let rootEntries = try? FileManager.default.contentsOfDirectory(
        at: packageDirectory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ) else {
        return excluded.sorted()
    }

    for pluginURL in rootEntries where pluginURL.lastPathComponent != "API" {
        guard (try? pluginURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: pluginURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            continue
        }
        for child in children {
            guard (try? child.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { continue }
            guard !fxaiPluginImplementationSourceFolders.contains(child.lastPathComponent) else { continue }
            excluded.insert("\(pluginURL.lastPathComponent)/\(child.lastPathComponent)")
        }
    }

    return excluded.sorted()
}

let fxaiPluginSources = Array(Set(fxaiPluginSwiftSources()).union(fxaiPinnedSharedRuntimeSources)).sorted()
precondition(!fxaiPluginSources.isEmpty, "FXPlugins manifest did not discover Swift plugin sources")

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
            exclude: fxaiPluginExcludedPaths(),
            sources: fxaiPluginSources
        ),
        .testTarget(
            name: "FXAIPluginsTests",
            dependencies: [
                "FXAIPlugins"
            ],
            path: "API/Tests/FXAIPluginsTests"
        )
    ],
    swiftLanguageModes: [.v6]
)
