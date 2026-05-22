import XCTest
@testable import FXDataEngine

final class TestSuiteResultTests: XCTestCase {
    func testSuiteResetAddCaseAndPassedRules() {
        var suite = TestSuiteTools.reset("plugin_contracts")

        XCTAssertEqual(suite.suiteName, "plugin_contracts")
        XCTAssertEqual(suite.total, 0)
        XCTAssertEqual(suite.failed, 0)
        XCTAssertFalse(suite.passed)

        suite.addCase(name: "registry_lifecycle", passed: true)
        suite.addCase(name: "predict_request_contract", passed: false, reason: "probability_sum")

        XCTAssertEqual(suite.total, 2)
        XCTAssertEqual(suite.failed, 1)
        XCTAssertFalse(suite.passed)

        suite.reset(suiteName: "runtime")
        XCTAssertEqual(suite.suiteName, "runtime")
        XCTAssertTrue(suite.cases.isEmpty)
        XCTAssertFalse(suite.passed)
    }

    func testLegacyReasonUsesFirstFailedCase() {
        var suite = TestSuiteTools.reset("plugin_contracts")
        suite.addCase(name: "registry_lifecycle", passed: true)
        suite.addCase(name: "manifest_and_selftest", passed: false, reason: "plugin_null_2")
        suite.addCase(name: "predict_request_contract", passed: false, reason: "class_probs")

        XCTAssertEqual(suite.legacyReason, "manifest_and_selftest:plugin_null_2")

        var noReason = TestSuiteTools.reset("plugin_contracts")
        noReason.addCase(name: "persistent_state_roundtrip", passed: false)
        XCTAssertEqual(noReason.legacyReason, "persistent_state_roundtrip")

        var allPassed = TestSuiteTools.reset("plugin_contracts")
        allPassed.addCase(name: "registry_lifecycle", passed: true)
        XCTAssertEqual(allPassed.legacyReason, "")
        XCTAssertTrue(allPassed.passed)
    }

    func testJsonEscapeMatchesLegacyEscapes() {
        XCTAssertEqual(
            TestSuiteTools.jsonEscape("a\\b\"c\rd\ne\tf"),
            #"a\\b\"c\rd\ne\tf"#
        )
    }

    func testJsonDocumentUsesLegacyFieldOrder() {
        var suite = TestSuiteTools.reset("suite\"name")
        suite.addCase(name: "ok", passed: true)
        suite.addCase(name: "bad", passed: false, reason: "line\nbreak")

        XCTAssertEqual(
            suite.jsonDocument(),
            #"{"suite_name":"suite\"name","total":2,"failed":1,"passed":false,"cases":[{"name":"ok","passed":true,"reason":""},{"name":"bad","passed":false,"reason":"line\nbreak"}]}"#
        )
    }
}
