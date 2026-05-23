import FXAIPlugins
import Foundation
import XCTest

final class ReferenceScorecardTests: XCTestCase {
    func testReferenceScorecardCoversRegistryAndAllPostUpgradeScoresAreAtLeast99() throws {
        let text = try String(contentsOf: scorecardURL(), encoding: .utf8)
        let rows = try parseRows(text)
        let registryNames = Set(FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName))

        XCTAssertEqual(Set(rows.map(\.plugin)), registryNames)
        XCTAssertEqual(rows.count, registryNames.count)
        XCTAssertGreaterThanOrEqual(rows.map(\.post).min() ?? 0.0, 99.0)
    }

    func testReferenceScorecardIdentifiesTheFiveUpgradedLowestPlugins() throws {
        let text = try String(contentsOf: scorecardURL(), encoding: .utf8)
        let rows = try parseRows(text)
        let lowestFive = rows.sorted { lhs, rhs in
            if lhs.pre == rhs.pre {
                return lhs.plugin < rhs.plugin
            }
            return lhs.pre < rhs.pre
        }
        .prefix(5)
        .map(\.plugin)

        XCTAssertEqual(lowestFive, ["rl_ppo", "ai_chronos", "ai_timesfm", "ai_mythos_rdt", "wm_graph"])
        for row in rows where lowestFive.contains(row.plugin) {
            XCTAssertGreaterThanOrEqual(row.post, 99.0, row.plugin)
            XCTAssertGreaterThan(row.post, row.pre, row.plugin)
        }
    }

    private struct ScoreRow {
        let plugin: String
        let pre: Double
        let post: Double
    }

    private func parseRows(_ text: String) throws -> [ScoreRow] {
        try text.components(separatedBy: .newlines)
            .filter { $0.hasPrefix("| `") }
            .map { line in
                let columns = line.split(separator: "|", omittingEmptySubsequences: false)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                guard columns.count >= 5 else {
                    throw ScorecardParseError.invalidRow(line)
                }
                let plugin = columns[1].trimmingCharacters(in: CharacterSet(charactersIn: "`"))
                guard let pre = Double(columns[2]), let post = Double(columns[3]) else {
                    throw ScorecardParseError.invalidRow(line)
                }
                return ScoreRow(plugin: plugin, pre: pre, post: post)
            }
    }

    private enum ScorecardParseError: Error {
        case invalidRow(String)
    }

    private func scorecardURL() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
            .appendingPathComponent("API")
            .appendingPathComponent("Docs")
            .appendingPathComponent("PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md")
    }
}
