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

    func testReferenceScorecardIdentifiesCurrentPassUpgrades() throws {
        let text = try String(contentsOf: scorecardURL(), encoding: .utf8)
        let rows = try parseRows(text)
        let upgraded = ["wm_cfx", "mix_loffm", "mix_moe_conformal", "ai_autoformer", "factor_pca_panel"]

        for plugin in upgraded {
            XCTAssertTrue(text.contains("`\(plugin)`"), plugin)
        }
        for row in rows where upgraded.contains(row.plugin) {
            XCTAssertGreaterThanOrEqual(row.post, 99.0, row.plugin)
            XCTAssertGreaterThan(row.post, row.pre, row.plugin)
        }
    }

    func testBackendScoreMatrixCoversEveryRegistryPlugin() throws {
        let text = try String(contentsOf: scorecardURL(), encoding: .utf8)
        let registryNames = Set(FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName))
        let backendRows = parseBackendRows(text)

        XCTAssertEqual(Set(backendRows.keys), registryNames)
        for (plugin, scores) in backendRows {
            XCTAssertGreaterThanOrEqual(scores.cpu, 99.0, plugin)
            for score in [scores.metal, scores.pyTorch, scores.tensorFlow, scores.nlp].compactMap({ $0 }) {
                XCTAssertGreaterThanOrEqual(score, 99.0, plugin)
            }
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

    private struct BackendScores {
        let cpu: Double
        let metal: Double?
        let pyTorch: Double?
        let tensorFlow: Double?
        let nlp: Double?
    }

    private func parseBackendRows(_ text: String) -> [String: BackendScores] {
        guard let start = text.range(of: "```tsv"),
              let end = text[start.upperBound...].range(of: "```")
        else { return [:] }
        let lines = text[start.upperBound..<end.lowerBound]
            .split(separator: "\n")
            .map(String.init)
            .dropFirst()
        return Dictionary(uniqueKeysWithValues: lines.compactMap { line in
            let columns = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard columns.count == 6, let cpu = Double(columns[1]) else { return nil }
            return (
                columns[0],
                BackendScores(
                    cpu: cpu,
                    metal: Self.score(columns[2]),
                    pyTorch: Self.score(columns[3]),
                    tensorFlow: Self.score(columns[4]),
                    nlp: Self.score(columns[5])
                )
            )
        })
    }

    private static func score(_ value: String) -> Double? {
        value == "N/A" ? nil : Double(value)
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
