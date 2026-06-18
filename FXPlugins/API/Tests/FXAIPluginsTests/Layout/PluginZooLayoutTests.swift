import FXAIPlugins
import FXDataEngine
import XCTest

final class PluginZooLayoutTests: XCTestCase {
    func testFXPluginsRootIsFlatPluginZoo() throws {
        let root = Self.packageRoot()
        let expectedPluginDirectories = Set(FXAIPluginRegistry.availablePlugins().map(\.manifest.aiName))
        let allowedNonRuntimeTemplateDirectories: Set<String> = ["demo_plugin_template"]
        let rootDirectories = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
            return values?.isDirectory == true
        }
        let actualPluginDirectories = Set(rootDirectories.map(\.lastPathComponent))
            .subtracting(["API"])
            .subtracting(allowedNonRuntimeTemplateDirectories)

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("API").path),
            "missing FXPlugins/API"
        )
        XCTAssertEqual(expectedPluginDirectories.count, FXDataEngineConstants.aiCount)
        XCTAssertEqual(actualPluginDirectories, expectedPluginDirectories)

        for directory in expectedPluginDirectories {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(directory).path),
                "missing FXPlugins/\(directory)"
            )
        }

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("demo_plugin_template").path),
            "missing compile-checked demo plugin template"
        )
        XCTAssertFalse(
            expectedPluginDirectories.contains("demo_plugin_template"),
            "demo_plugin_template must stay out of the runnable production plugin registry"
        )

        for retiredStagingDirectory in [
            "Backends",
            "Common",
            "Demo",
            "Distribution",
            "Factor",
            "Linear",
            "Memory",
            "Mixture",
            "Python",
            "RL",
            "Rule",
            "Sequence",
            "Sources",
            "Stat",
            "Tests",
            "Tree",
            "Trend",
            "World"
        ] {
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(retiredStagingDirectory).path),
                "FXPlugins/\(retiredStagingDirectory) should not exist in the flat plugin-owned zoo layout"
            )
        }

        let apiDocs = root.appendingPathComponent("API").appendingPathComponent("Docs")
        let planDocs = ((try? FileManager.default.contentsOfDirectory(
            at: apiDocs,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )) ?? []).filter { url in
            url.pathExtension.lowercased() == "md" &&
                url.lastPathComponent.localizedCaseInsensitiveContains("plan")
        }
        XCTAssertTrue(
            planDocs.isEmpty,
            "FXPlugins/API/Docs should not contain standalone plan docs: \(planDocs.map(\.lastPathComponent).joined(separator: ", "))"
        )
    }

    func testSequenceArchitectureCPUModelsUseSharedRuntimeImplementation() throws {
        let root = Self.packageRoot()
        let wrapperPaths = [
            "ai_attn_cnn_bilstm/CPU/AIAttnCNNBiLSTMCPUModel.swift",
            "ai_autoformer/CPU/AIAutoformerCPUModel.swift",
            "ai_bilstm/CPU/AIBiLSTMCPUModel.swift",
            "ai_chronos/CPU/AIChronosCPUModel.swift",
            "ai_cnn_lstm/CPU/AICNNLSTMCPUModel.swift",
            "ai_fewc/CPU/AIFEWCCPUModel.swift",
            "ai_geodesic/CPU/AIGeodesicCPUModel.swift",
            "ai_gha/CPU/AIGHACPUModel.swift",
            "ai_gru/CPU/AIGRUCPUModel.swift",
            "ai_lstm/CPU/AILSTMCPUModel.swift",
            "ai_lstm_tcn/CPU/AILSTMTCNCPUModel.swift",
            "ai_lstmg/CPU/AILSTMGCPUModel.swift",
            "ai_mythos_rdt/CPU/AIMythosRDTCPUModel.swift",
            "ai_nbeats/CPU/AINbeatsCPUModel.swift",
            "ai_nhits/CPU/AINhitsCPUModel.swift",
            "ai_patchtst/CPU/AIPatchtstCPUModel.swift",
            "ai_qcew/CPU/AIQCEWCPUModel.swift",
            "ai_s4/CPU/AIS4CPUModel.swift",
            "ai_stmn/CPU/AIStmnCPUModel.swift",
            "ai_tcn/CPU/AITCNCPUModel.swift",
            "ai_tesseract/CPU/AITesseractCPUModel.swift",
            "ai_tft/CPU/AITFTCPUModel.swift",
            "ai_timesfm/CPU/AITimesfmCPUModel.swift",
            "ai_trr/CPU/AITRRCPUModel.swift",
            "ai_tst/CPU/AITSTCPUModel.swift"
        ]

        let sharedSourceURL = root
            .appendingPathComponent("ai_autoformer")
            .appendingPathComponent("CPU")
            .appendingPathComponent("FXAISequenceArchitectureCPUModel.swift")
        let sharedSource = try String(contentsOf: sharedSourceURL, encoding: .utf8)
        XCTAssertTrue(
            sharedSource.contains("public struct FXAISequenceArchitectureCPUModel"),
            "shared sequence architecture CPU runtime is missing"
        )
        XCTAssertTrue(
            sharedSource.contains("private func architectureSignals"),
            "shared sequence architecture CPU runtime must own the architecture signal body"
        )

        for wrapperPath in wrapperPaths {
            let sourceURL = root.appendingPathComponent(wrapperPath)
            let source = try String(contentsOf: sourceURL, encoding: .utf8)
            XCTAssertTrue(
                source.contains("private var core: FXAISequenceArchitectureCPUModel"),
                "\(wrapperPath) must delegate to the shared sequence architecture CPU runtime"
            )
            XCTAssertTrue(
                source.contains("FXAISequenceArchitectureCPUConfiguration("),
                "\(wrapperPath) must keep only its architecture metadata configuration"
            )
            XCTAssertFalse(
                source.contains("private func architectureSignals"),
                "\(wrapperPath) reintroduced duplicated architecture signal code"
            )
            XCTAssertFalse(
                source.contains("private func buildFeatures"),
                "\(wrapperPath) reintroduced duplicated feature-building code"
            )
            XCTAssertFalse(
                source.contains("private func hiddenActivations"),
                "\(wrapperPath) reintroduced duplicated hidden encoder code"
            )
        }

        for (wrapperPath, architectureID, modelID) in [
            ("ai_gru/CPU/AIGRUCPUModel.swift", 58, "AIModelID.gru.rawValue"),
            ("ai_bilstm/CPU/AIBiLSTMCPUModel.swift", 59, "AIModelID.bilstm.rawValue"),
            ("ai_lstm_tcn/CPU/AILSTMTCNCPUModel.swift", 60, "AIModelID.lstmTCN.rawValue"),
            ("ai_mythos_rdt/CPU/AIMythosRDTCPUModel.swift", 61, "AIModelID.mythosRDT.rawValue")
        ] {
            let sourceURL = root.appendingPathComponent(wrapperPath)
            let source = try String(contentsOf: sourceURL, encoding: .utf8)
            XCTAssertTrue(
                source.contains("architectureID: \(architectureID),"),
                "\(wrapperPath) must preserve its legacy architecture flavor ID for deterministic CPU seeding"
            )
            XCTAssertTrue(
                source.contains("modelID: \(modelID),"),
                "\(wrapperPath) must preserve its manifest model ID separately from the architecture flavor ID"
            )
        }
    }

    private static func packageRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
    }
}
