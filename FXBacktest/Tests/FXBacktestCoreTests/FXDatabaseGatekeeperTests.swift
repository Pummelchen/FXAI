import XCTest

final class FXDatabaseGatekeeperTests: XCTestCase {
    func testNonDatabaseProjectsDoNotUseDirectClickHouseAccess() throws {
        let repositoryRoot = try Self.repositoryRoot()
        let scannedRoots = [
            "FXBacktest/Sources",
            "FXDataEngine/Sources",
            "FXPlugins",
            "FXImporter/Sources",
            "FXGUI/Sources",
            "FXBacktestAgent",
            "FXDemoAgent",
            "FXLiveAgent"
        ]
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
            for fileURL in try Self.auditedFiles(under: root) {
                guard !Self.isAllowedCertificationScanner(fileURL) else { continue }
                let text = try String(contentsOf: fileURL, encoding: .utf8)
                for pattern in forbiddenPatterns where text.contains(pattern) {
                    let relativePath = fileURL.path.replacingOccurrences(of: repositoryRoot.path + "/", with: "")
                    violations.append("\(relativePath): \(pattern)")
                }
            }
        }

        let message = "Non-FXDatabase projects must access ClickHouse only through versioned FXDatabase APIs. Violations: \(violations.joined(separator: "; "))"
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

    private static func auditedFiles(under root: URL) throws -> [URL] {
        let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        let auditedExtensions: Set<String> = [
            "swift",
            "metal",
            "py",
            "sh",
            "sql",
            "json",
            "toml",
            "md"
        ]
        var urls: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            guard auditedExtensions.contains(url.pathExtension.lowercased()) else { continue }
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                urls.append(url)
            }
        }
        return urls
    }

    private static func isAllowedCertificationScanner(_ url: URL) -> Bool {
        url.lastPathComponent == "FXAIPluginCertificationRegistry.swift"
    }
}
