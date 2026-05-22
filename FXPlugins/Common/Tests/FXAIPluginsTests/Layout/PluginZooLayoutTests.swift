import XCTest

final class PluginZooLayoutTests: XCTestCase {
    func testFXPluginsRootIsFamilyFirstPluginZoo() throws {
        let root = Self.packageRoot()
        let requiredZooDirectories = [
            "Backends",
            "Common",
            "Demo",
            "Distribution",
            "Factor",
            "Linear",
            "Memory",
            "Mixture",
            "RL",
            "Rule",
            "Sequence",
            "Stat",
            "Tree",
            "Trend",
            "World"
        ]

        for directory in requiredZooDirectories {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(directory).path),
                "missing FXPlugins/\(directory)"
            )
        }

        for retiredStagingDirectory in ["Sources", "Tests", "Python"] {
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(retiredStagingDirectory).path),
                "FXPlugins/\(retiredStagingDirectory) should not exist in the family-first zoo layout"
            )
        }

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("PLUGIN_CONVERSION_PLAN.md").path),
            "conversion plan belongs under Common/Docs, not the plugin-zoo root"
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
