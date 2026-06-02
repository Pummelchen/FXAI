import Foundation
import XCTest

final class FXToolsSmokeTests: XCTestCase {
    func testFXAICertifyCommandContractIsPresent() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = packageRoot
            .appendingPathComponent("Sources")
            .appendingPathComponent("FXAICertify")
            .appendingPathComponent("main.swift")

        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        XCTAssertTrue(source.contains("CertificationRunner"))
        XCTAssertTrue(source.contains("usage: fxai certify [--all|--build-only]"))
        XCTAssertTrue(source.contains("FXAICertificationEvidenceRequest"))
    }
}
