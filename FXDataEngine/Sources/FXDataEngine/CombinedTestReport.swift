import Foundation

public enum CombinedTestReportTools {
    public static func skippedSuite(suiteName: String, caseName: String) -> TestSuiteResult {
        var suite = TestSuiteTools.reset(suiteName)
        suite.addCase(name: caseName, passed: true)
        return suite
    }

    public static func jsonDocument(
        seed: Int,
        generatedAtUTC: Int64,
        suites: [TestSuiteResult]
    ) -> String {
        let ok = !suites.isEmpty && suites.allSatisfy(\.passed)
        var json = "{"
        json += "\"seed\":\(seed),"
        json += "\"generated_at\":\(max(0, generatedAtUTC)),"
        json += "\"ok\":\(ok ? "true" : "false"),"
        json += "\"suites\":["
        for (index, suite) in suites.enumerated() {
            if index > 0 {
                json += ","
            }
            json += suite.jsonDocument()
        }
        json += "]"
        json += "}"
        return json
    }

    public static func write(
        seed: Int,
        generatedAtUTC: Int64,
        suites: [TestSuiteResult],
        to fileURL: URL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try jsonDocument(seed: seed, generatedAtUTC: generatedAtUTC, suites: suites)
            .write(to: fileURL, atomically: true, encoding: .utf8)
    }
}
