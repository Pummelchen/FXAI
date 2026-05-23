import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class PluginCertificationGateTests: XCTestCase {
    func testCertificationAuditCoversEveryPluginAndDeclaredBackend() throws {
        try FXAIPluginCertificationRegistry.validateAuditCoverage()

        let reports = FXAIPluginCertificationRegistry.certificationReports()
        let plugins = FXAIPluginRegistry.availablePlugins().compactMap { $0 as? any FXAIPlannedPlugin }

        XCTAssertEqual(reports.count, plugins.count)
        XCTAssertEqual(reports.count, FXDataEngineConstants.aiCount)
        for plugin in plugins {
            let report = try XCTUnwrap(reports.first { $0.pluginName == plugin.manifest.aiName })
            XCTAssertEqual(report.backendStatuses.map(\.backend), plugin.accelerationPlan.declaredBackends)
            for status in report.backendStatuses {
                XCTAssertFalse(status.requiredGates.isEmpty, "\(plugin.manifest.aiName) \(status.backend.rawValue)")
                XCTAssertTrue(
                    Set(status.satisfiedGates).isSubset(of: Set(status.requiredGates)),
                    "\(plugin.manifest.aiName) \(status.backend.rawValue)"
                )
            }
        }
    }

    func testHundredPercentCertificationPassesWithCurrentEvidenceSet() throws {
        XCTAssertNoThrow(try FXAIPluginCertificationRegistry.requireAllPlugins100PercentCertified())

        let reports = FXAIPluginCertificationRegistry.certificationReports()
        XCTAssertEqual(reports.count, FXDataEngineConstants.aiCount)
        XCTAssertTrue(reports.allSatisfy(\.is100PercentCertified))
        XCTAssertTrue(reports.allSatisfy { $0.blockingGates.isEmpty })
    }

    func testDeclaredAcceleratorsKeepTheirBackendSpecificBlockingGates() throws {
        let statuses = FXAIPluginCertificationRegistry.certificationReports()
            .flatMap { report in
                report.backendStatuses.map { (report.pluginName, $0) }
            }

        let metalStatuses = statuses.filter { $0.1.backend == .metal }
        XCTAssertFalse(metalStatuses.isEmpty)
        XCTAssertTrue(
            metalStatuses.allSatisfy { $0.1.satisfiedGates.contains(.metalLiveBufferParity) },
            "Metal declarations must prove live buffer execution/parity evidence."
        )
        XCTAssertTrue(
            metalStatuses.allSatisfy { $0.1.satisfiedGates.contains(.metalSourceCompilation) },
            "Metal declarations must prove per-plugin kernel compilation evidence."
        )

        let pyTorchStatuses = statuses.filter { $0.1.backend == .pyTorchMPS }
        XCTAssertFalse(pyTorchStatuses.isEmpty)
        XCTAssertTrue(
            pyTorchStatuses.allSatisfy { $0.1.satisfiedGates.contains(.pyTorchLiveTrainPredictPersistence) },
            "PyTorch declarations need per-plugin train/predict/persistence evidence."
        )

        let tensorFlowStatuses = statuses.filter { $0.1.backend == .tensorFlowMetal }
        XCTAssertFalse(tensorFlowStatuses.isEmpty)
        XCTAssertTrue(
            tensorFlowStatuses.allSatisfy { $0.1.satisfiedGates.contains(.tensorFlowLiveTrainPredictPersistence) },
            "TensorFlow declarations need per-plugin train/predict/persistence evidence."
        )

        let nlpStatuses = statuses.filter { $0.1.backend == .foundationNLP }
        XCTAssertFalse(nlpStatuses.isEmpty)
        XCTAssertTrue(
            nlpStatuses.allSatisfy { $0.1.satisfiedGates.contains(.nlpLiveContextPayload) },
            "NLP declarations need live text/context payload evidence."
        )

        let coreMLStatuses = statuses.filter { $0.1.backend == .coreMLNeuralEngine }
        XCTAssertTrue(coreMLStatuses.isEmpty, "CoreML/Neural Engine must not be declared until export and parity evidence exists.")
    }

    func testCurrentCertificationEvidenceMatchesVerifiedRuntimeFoundation() throws {
        let reports = FXAIPluginCertificationRegistry.certificationReports()

        for report in reports {
            for status in report.backendStatuses {
                XCTAssertTrue(status.satisfiedGates.contains(.registryCoverage), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.swiftCPURuntimeSmoke), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.ohlcvVolumeContract), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.historicalOrReferenceParity), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.backendSelectionPolicy), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.sineTestRuntimeSmoke), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.fxDatabaseAPIOnlyDataPath), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.fullVerificationRun), report.pluginName)
                XCTAssertTrue(status.blockingGates.isEmpty, report.pluginName)
            }
        }

        let externalStatuses = reports.flatMap(\.backendStatuses).filter {
            $0.backend.requiresExternalPython || $0.backend == .foundationNLP
        }
        XCTAssertFalse(externalStatuses.isEmpty)
        XCTAssertTrue(externalStatuses.allSatisfy { $0.satisfiedGates.contains(.externalBackendDiscovery) })
    }
}
