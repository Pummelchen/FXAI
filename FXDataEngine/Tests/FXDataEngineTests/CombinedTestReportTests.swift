import Foundation
import XCTest
@testable import FXDataEngine

final class CombinedTestReportTests: XCTestCase {
    func testCombinedReportMatchesLegacyRunnerJsonShape() {
        var tensor = TestSuiteTools.reset("tensorcore")
        tensor.addCase(name: "tensorcore_skipped", passed: true)
        var plugin = TestSuiteTools.reset("plugin_contracts")
        plugin.addCase(name: "registry_lifecycle", passed: true)

        XCTAssertEqual(
            CombinedTestReportTools.jsonDocument(
                seed: 42,
                generatedAtUTC: 1_704_153_600,
                suites: [tensor, plugin]
            ),
            #"{"seed":42,"generated_at":1704153600,"ok":true,"suites":[{"suite_name":"tensorcore","total":1,"failed":0,"passed":true,"cases":[{"name":"tensorcore_skipped","passed":true,"reason":""}]},{"suite_name":"plugin_contracts","total":1,"failed":0,"passed":true,"cases":[{"name":"registry_lifecycle","passed":true,"reason":""}]}]}"#
        )

        plugin.addCase(name: "predict_request_contract", passed: false, reason: "probability_sum")
        XCTAssertTrue(CombinedTestReportTools.jsonDocument(seed: 7, generatedAtUTC: -1, suites: [plugin]).contains(#""ok":false"#))
        XCTAssertTrue(CombinedTestReportTools.jsonDocument(seed: 7, generatedAtUTC: -1, suites: [plugin]).contains(#""generated_at":0"#))
        XCTAssertTrue(CombinedTestReportTools.jsonDocument(seed: 7, generatedAtUTC: 1, suites: []).contains(#""ok":false"#))
    }

    func testSkippedSuiteAndWriteCombinedReport() throws {
        let suite = CombinedTestReportTools.skippedSuite(suiteName: "plugin_contracts", caseName: "plugin_contracts_skipped")
        XCTAssertTrue(suite.passed)
        XCTAssertEqual(suite.legacyReason, "")

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CombinedTestReportTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("FXAI/Reports/tensorcore_contract_report.json", isDirectory: false)
        try CombinedTestReportTools.write(seed: 11, generatedAtUTC: 22, suites: [suite], to: url)

        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(
            try String(contentsOf: url, encoding: .utf8),
            CombinedTestReportTools.jsonDocument(seed: 11, generatedAtUTC: 22, suites: [suite])
        )
    }
}
