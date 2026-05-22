import FXAIPlugins
import FXDataEngine
import XCTest

final class PluginZooLayoutTests: XCTestCase {
    func testFXPluginsRootIsFlatPluginZoo() throws {
        let root = Self.packageRoot()
        let expectedPluginDirectories = Set(FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName))
        let rootDirectories = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
            return values?.isDirectory == true
        }
        let actualPluginDirectories = Set(rootDirectories.map(\.lastPathComponent)).subtracting(["API"])

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("API").path),
            "missing FXPlugins/API"
        )
        XCTAssertEqual(expectedPluginDirectories.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(actualPluginDirectories, expectedPluginDirectories)

        for directory in expectedPluginDirectories {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(directory).path),
                "missing FXPlugins/\(directory)"
            )
        }

        for retiredStagingDirectory in [
            "Backends",
            "Common",
            "Demo",
            "Distribution",
            "Factor",
            "Linear",
            "Memory",
            "Mixture",
            "Python",
            "RL",
            "Rule",
            "Sequence",
            "Sources",
            "Stat",
            "Tests",
            "Tree",
            "Trend",
            "World"
        ] {
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(retiredStagingDirectory).path),
                "FXPlugins/\(retiredStagingDirectory) should not exist in the flat plugin-owned zoo layout"
            )
        }

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("PLUGIN_CONVERSION_PLAN.md").path),
            "conversion plan belongs under API/Docs, not the plugin-zoo root"
        )
    }

    private static func packageRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
    }
}
