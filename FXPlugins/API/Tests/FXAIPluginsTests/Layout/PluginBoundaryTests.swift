import XCTest
@testable import FXAIPlugins

final class PluginBoundaryTests: XCTestCase {
    func testFXPluginsPackageDoesNotDeclareDatabaseOrBacktestDependencies() throws {
        let manifest = try String(contentsOf: Self.packageRoot.appendingPathComponent("Package.swift"), encoding: .utf8)

        XCTAssertTrue(manifest.contains(".package(path: \"../FXDataEngine\")"))
        XCTAssertFalse(manifest.contains(".package(path: \"../FXDatabase\")"))
        XCTAssertFalse(manifest.contains("FXDatabaseFXBacktestAPI"))
        XCTAssertFalse(manifest.contains("FXDatabaseBacktestCore"))
    }

    func testProductionSourcesDoNotImportDatabaseOrBacktestModules() throws {
        let forbiddenImports = [
            moduleImport("FXBacktestAPI"),
            moduleImport("FXDatabase"),
            moduleImport("BacktestCore"),
            moduleImport("ClickHouse"),
            moduleImport("Domain")
        ]
        let sourceFiles = try productionSwiftSourceFiles()
        XCTAssertFalse(sourceFiles.isEmpty)

        for sourceFile in sourceFiles {
            let text = try String(contentsOf: sourceFile, encoding: .utf8)
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                XCTAssertFalse(
                    forbiddenImports.contains(trimmed),
                    "\(sourceFile.path) contains forbidden module boundary import: \(trimmed)"
                )
            }
        }
    }

    func testDemoPluginTemplateUsesNeutralConfigurationShape() throws {
        let template = DemoPluginTemplate(aiID: 0)
        let rows = template.configurationRows()

        XCTAssertEqual(rows.count, template.accelerationPlan.declaredBackends.count)
        XCTAssertGreaterThanOrEqual(rows.count, 5)
        XCTAssertFalse(FXAIPluginRegistry.availablePlugins().contains { $0.manifest.aiName == "demo_plugin_template" })
        for row in rows {
            try row.validate()
            XCTAssertEqual(row.parameters.map(\.key), ["lookback_bars", "confidence_floor", "use_volume_when_available"])
        }
    }

    private static var packageRoot: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
    }

    private func productionSwiftSourceFiles() throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: Self.packageRoot,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        var files: [URL] = []
        let rootPath = Self.packageRoot.path + "/"
        while let file = enumerator.nextObject() as? URL {
            guard file.pathExtension == "swift" else { continue }
            guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            let relativePath = String(file.path.dropFirst(rootPath.count))
            guard relativePath != "Package.swift" else { continue }
            guard !relativePath.hasPrefix("API/Tests/") else { continue }
            files.append(file)
        }
        return files.sorted { $0.path < $1.path }
    }

    private func moduleImport(_ moduleName: String) -> String {
        ["import", moduleName].joined(separator: " ")
    }
}
