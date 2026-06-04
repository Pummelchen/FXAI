import Foundation
import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class PluginBoundaryTests: XCTestCase {
    func testFXPluginsPackageDoesNotDeclareDatabaseOrBacktestDependencies() throws {
        let manifest = try String(contentsOf: Self.packageRoot.appendingPathComponent("Package.swift"), encoding: .utf8)

        XCTAssertTrue(manifest.contains(".package(path: \"../FXDataEngine\")"))
        XCTAssertFalse(manifest.contains(".package(path: \"../FXDatabase\")"))
        XCTAssertFalse(manifest.contains("FXDatabaseFXBacktestAPI"))
        XCTAssertFalse(manifest.contains("FXDatabaseBacktestCore"))
    }

    func testProductionSourcesDoNotImportDatabaseOrBacktestModules() throws {
        let forbiddenImports = [
            moduleImport("FXBacktestAPI"),
            moduleImport("FXDatabase"),
            moduleImport("BacktestCore"),
            moduleImport("ClickHouse"),
            moduleImport("Domain")
        ]
        let sourceFiles = try productionSwiftSourceFiles()
        XCTAssertFalse(sourceFiles.isEmpty)

        for sourceFile in sourceFiles {
            let text = try String(contentsOf: sourceFile, encoding: .utf8)
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                XCTAssertFalse(
                    forbiddenImports.contains(trimmed),
                    "\(sourceFile.path) contains forbidden module boundary import: \(trimmed)"
                )
            }
        }
    }

    func testDemoPluginTemplateUsesNeutralConfigurationShape() throws {
        let template = DemoPluginTemplate(aiID: 0)
        let rows = template.configurationRows()
        let expectedBackends: [FXPluginAccelerationBackend] = [
            .swiftScalar,
            .metal,
            .pyTorchMPS,
            .tensorFlowMetal,
            .foundationNLP,
            .onnxRuntime,
            .remoteRPC
        ]

        XCTAssertEqual(rows.count, template.accelerationPlan.declaredBackends.count)
        XCTAssertEqual(template.accelerationPlan.declaredBackends, expectedBackends)
        XCTAssertEqual(rows.count, expectedBackends.count)
        XCTAssertFalse(FXAIPluginRegistry.availablePlugins().contains { $0.manifest.aiName == "demo_plugin_template" })
        for row in rows {
            try row.validate()
            XCTAssertEqual(row.parameters.map(\.key), ["lookback_bars", "confidence_floor", "use_volume_when_available"])
        }
    }

    func testDemoPluginTemplateCoversEveryRuntimeSurface() throws {
        let root = Self.packageRoot.appendingPathComponent("demo_plugin_template")
        let requiredPaths = [
            "CPU/DemoPluginTemplate.swift",
            "Metal/DemoPluginTemplateMetal.swift",
            "PyTorch/demo_plugin_template_torch.py",
            "TensorFlow/demo_plugin_template_tensorflow.py",
            "NLP/demo_plugin_template_nlp.py",
            "ONNX/demo_plugin_template.manifest.json",
            "ONNX/README.md",
            "RemoteRPC/README.md",
            "RemoteRPC/remote_rpc_response.example.json",
            "README.md"
        ]

        for path in requiredPaths {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(path).path),
                "demo plugin template is missing \(path)"
            )
        }
    }

    func testDemoPluginTemplateDocumentsRuntimeBridgeEntryPoints() throws {
        let root = Self.packageRoot.appendingPathComponent("demo_plugin_template")
        let bridgeExpectations = [
            ("PyTorch/demo_plugin_template_torch.py", ["def predict_batch(", "def train_step(", "financial_targets", "financial_loss_config", "class DemoPluginTemplateTorch"]),
            ("TensorFlow/demo_plugin_template_tensorflow.py", ["def predict_batch(", "def train_step(", "financial_targets", "financial_loss_config", "class DemoPluginTemplateTensorFlow"]),
            ("NLP/demo_plugin_template_nlp.py", ["def merge_into_numeric_features(", "def predict_batch(", "def train_step("]),
            ("RemoteRPC/remote_rpc_response.example.json", ["\"apiVersion\": 4", "\"prediction\"", "\"classProbabilities\""]),
            ("ONNX/demo_plugin_template.manifest.json", ["\"pluginName\"", "\"modelSha256\"", "\"inputName\""])
        ]

        for (path, requiredSnippets) in bridgeExpectations {
            let url = root.appendingPathComponent(path)
            let text = try String(contentsOf: url, encoding: .utf8)
            for snippet in requiredSnippets {
                XCTAssertTrue(text.contains(snippet), "\(path) missing \(snippet)")
            }
            if path.hasSuffix(".json") {
                XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(contentsOf: url)))
            }
        }
    }

    func testPluginPythonBackendDiscoveryDefaultsToPython312() throws {
        let descriptor = try XCTUnwrap(FXAIPluginBackendDiscovery.externalPythonDescriptor(
            pluginName: "demo_plugin_template",
            backend: .pyTorchMPS
        ))

        guard case .externalPython(_, let executable, _) = descriptor.mode else {
            XCTFail("Expected external Python descriptor")
            return
        }
        XCTAssertNotEqual(executable, "python3")
        XCTAssertTrue(
            executable.contains("python3.12") || executable.contains("python@3.12"),
            "Plugin Python backend discovery must resolve Python 3.12, got \(executable)"
        )
    }

    private static var packageRoot: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
    }

    private func productionSwiftSourceFiles() throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: Self.packageRoot,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        var files: [URL] = []
        let rootPath = Self.packageRoot.path + "/"
        while let file = enumerator.nextObject() as? URL {
            guard file.pathExtension == "swift" else { continue }
            guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            let relativePath = String(file.path.dropFirst(rootPath.count))
            guard relativePath != "Package.swift" else { continue }
            guard !relativePath.hasPrefix("API/Tests/") else { continue }
            files.append(file)
        }
        return files.sorted { $0.path < $1.path }
    }

    private func moduleImport(_ moduleName: String) -> String {
        ["import", moduleName].joined(separator: " ")
    }
}
