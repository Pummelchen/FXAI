import XCTest

final class RepositoryGovernanceDocsTests: XCTestCase {
    func testRootGovernanceContractIsDiscoverableAndPinsReleaseEvidence() throws {
        let root = Self.repositoryRoot()
        let governanceURL = root.appendingPathComponent("GOVERNANCE.md")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: governanceURL.path),
            "repository root must contain GOVERNANCE.md"
        )

        let governance = try String(contentsOf: governanceURL, encoding: .utf8)
        for requiredText in [
            "# FXAI Governance",
            "## Authorities",
            "## Change Classes",
            "## Promotion Lifecycle",
            "## Release Gates",
            "## Documentation Governance",
            "## Incident Workflow",
            "./fxai certify --all",
            "swift test --package-path FXPlugins",
            "FXDataEngine/Tools/fxai_testlab.py verify-all",
            "tensorflow==2.18.1",
            "tensorflow-metal==1.2.0",
            "FXExecutionContracts"
        ] {
            XCTAssertTrue(
                governance.contains(requiredText),
                "GOVERNANCE.md must document \(requiredText)"
            )
        }

        let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(readme.contains("[FXAI Governance](GOVERNANCE.md)"))
        XCTAssertTrue(readme.contains("https://github.com/Pummelchen/FXAI/wiki/Governance"))
        XCTAssertTrue(readme.contains("- [Governance](GOVERNANCE.md)"))
        XCTAssertTrue(readme.contains("./fxai certify --all"))
    }

    func testSubsystemDocsLinkToRootGovernanceContract() throws {
        let root = Self.repositoryRoot()
        for (path, expectedLink) in [
            ("FXPlugins/README.md", "../GOVERNANCE.md"),
            ("FXDemoAgent/README.md", "../GOVERNANCE.md"),
            ("FXLiveAgent/README.md", "../GOVERNANCE.md"),
            ("FXDataEngine/Tools/OfflineLab/README.md", "../../../GOVERNANCE.md"),
            ("FXDataEngine/Tools/OfflineLab/DriftGovernance/README.md", "../../../../GOVERNANCE.md")
        ] {
            let source = try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
            XCTAssertTrue(
                source.contains(expectedLink),
                "\(path) must link to the root governance contract"
            )
        }
    }

    private static func repositoryRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            url.deleteLastPathComponent()
        }
        return url
    }
}
