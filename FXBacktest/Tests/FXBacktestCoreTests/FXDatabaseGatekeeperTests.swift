import XCTest

final class FXDatabaseGatekeeperTests: XCTestCase {
    func testBacktestAndRuntimeAgentsDoNotUseDirectClickHouseAccess() throws {
        let repositoryRoot = try Self.repositoryRoot()
        let scannedRoots = ["FXBacktest/Sources", "FXDemoAgent", "FXLiveAgent"]
        let forbiddenPatterns = [
            "import ClickHouse",
            "ClickHouseHTTPClient",
            "ClickHouseQuery",
            "clickhouse://",
            ":8123",
            "8123/",
            "INSERT INTO ",
            "SELECT "
        ]

        var violations: [String] = []
        for relativeRoot in scannedRoots {
            let root = repositoryRoot.appendingPathComponent(relativeRoot)
            guard FileManager.default.fileExists(atPath: root.path) else { continue }
            for fileURL in try Self.swiftAndMarkdownFiles(under: root) {
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                for pattern in forbiddenPatterns where text.contains(pattern) {
                    let relativePath = fileURL.path.replacingOccurrences(of: repositoryRoot.path + "/", with: "")
                    violations.append("\(relativePath): \(pattern)")
                }
            }
        }

        let message = "FXBacktest, FXDemoAgent, and FXLiveAgent must access ClickHouse only through FXDatabase API. Violations: \(violations.joined(separator: "; "))"
        XCTAssertTrue(violations.isEmpty, message)
    }

    private static func repositoryRoot() throws -> URL {
        var url = URL(fileURLWithPath: #filePath)
        while url.lastPathComponent != "FXAI" && url.path != "/" {
            url.deleteLastPathComponent()
        }
        guard url.lastPathComponent == "FXAI" else {
            throw NSError(domain: "FXDatabaseGatekeeperTests", code: 1)
        }
        return url
    }

    private static func swiftAndMarkdownFiles(under root: URL) throws -> [URL] {
        let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        var urls: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            guard ["swift", "md"].contains(url.pathExtension.lowercased()) else { continue }
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                urls.append(url)
            }
        }
        return urls
    }
}
