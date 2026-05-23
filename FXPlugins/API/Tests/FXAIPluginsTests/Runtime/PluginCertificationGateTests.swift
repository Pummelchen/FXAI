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

    func testHundredPercentCertificationFailsClosedUntilAllEvidenceExists() throws {
        XCTAssertThrowsError(try FXAIPluginCertificationRegistry.requireAllPlugins100PercentCertified()) { error in
            guard case FXPluginCertificationError.incompleteCertification(let reports) = error else {
                return XCTFail("Unexpected certification error: \(error)")
            }

            XCTAssertEqual(reports.count, FXDataEngineConstants.aiCount)
            XCTAssertTrue(reports.allSatisfy { !$0.is100PercentCertified })
            XCTAssertTrue(
                reports.allSatisfy { $0.blockingGates.contains(.historicalOrReferenceParity) },
                "Every plugin needs MQL5/golden or standard-reference parity evidence before 100% certification."
            )
        }
    }

    func testDeclaredAcceleratorsKeepTheirBackendSpecificBlockingGates() throws {
        let statuses = FXAIPluginCertificationRegistry.certificationReports()
            .flatMap { report in
                report.backendStatuses.map { (report.pluginName, $0) }
            }

        let metalStatuses = statuses.filter { $0.1.backend == .metal }
        XCTAssertFalse(metalStatuses.isEmpty)
        XCTAssertTrue(
            metalStatuses.allSatisfy { $0.1.blockingGates.contains(.metalLiveBufferParity) },
            "Metal declarations must remain uncertified until live buffer execution and CPU parity are proven per plugin."
        )
        XCTAssertTrue(
            metalStatuses.allSatisfy { $0.1.blockingGates.contains(.metalSourceCompilation) },
            "Metal declarations must also prove per-plugin kernel compilation before 100% certification."
        )

        let pyTorchStatuses = statuses.filter { $0.1.backend == .pyTorchMPS }
        XCTAssertFalse(pyTorchStatuses.isEmpty)
        XCTAssertTrue(
            pyTorchStatuses.allSatisfy { $0.1.blockingGates.contains(.pyTorchLiveTrainPredictPersistence) },
            "PyTorch declarations need per-plugin train/predict/persistence evidence before 100% certification."
        )

        let tensorFlowStatuses = statuses.filter { $0.1.backend == .tensorFlowMetal }
        XCTAssertFalse(tensorFlowStatuses.isEmpty)
        XCTAssertTrue(
            tensorFlowStatuses.allSatisfy { $0.1.blockingGates.contains(.tensorFlowLiveTrainPredictPersistence) },
            "TensorFlow declarations need per-plugin train/predict/persistence evidence before 100% certification."
        )

        let nlpStatuses = statuses.filter { $0.1.backend == .foundationNLP }
        XCTAssertFalse(nlpStatuses.isEmpty)
        XCTAssertTrue(
            nlpStatuses.allSatisfy { $0.1.blockingGates.contains(.nlpLiveContextPayload) },
            "NLP declarations need live text/context payload evidence before 100% certification."
        )

        let coreMLStatuses = statuses.filter { $0.1.backend == .coreMLNeuralEngine }
        XCTAssertFalse(coreMLStatuses.isEmpty)
        XCTAssertTrue(
            coreMLStatuses.allSatisfy { $0.1.blockingGates.contains(.coreMLNeuralEngineLiveParity) },
            "CoreML/Neural Engine declarations need live parity evidence before 100% certification."
        )
    }

    func testCurrentCertificationEvidenceMatchesVerifiedRuntimeFoundation() throws {
        let reports = FXAIPluginCertificationRegistry.certificationReports()

        for report in reports {
            for status in report.backendStatuses {
                XCTAssertTrue(status.satisfiedGates.contains(.registryCoverage), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.swiftCPURuntimeSmoke), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.ohlcvVolumeContract), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.backendSelectionPolicy), report.pluginName)
                XCTAssertTrue(status.satisfiedGates.contains(.sineTestRuntimeSmoke), report.pluginName)
            }
        }

        let externalStatuses = reports.flatMap(\.backendStatuses).filter {
            $0.backend.requiresExternalPython || $0.backend == .foundationNLP
        }
        XCTAssertFalse(externalStatuses.isEmpty)
        XCTAssertTrue(externalStatuses.allSatisfy { $0.satisfiedGates.contains(.externalBackendDiscovery) })
    }
}
