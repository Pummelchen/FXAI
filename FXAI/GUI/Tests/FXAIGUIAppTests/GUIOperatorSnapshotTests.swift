import AppKit
@testable import FXAIGUIApp
import FXAIGUICore
import Foundation
import Testing

@MainActor
struct GUIOperatorSnapshotTests {
    private let destinations: [SidebarDestination] = [
        .liveOverview,
        .demoOverview,
        .researchWorkspace,
        .platformControl,
        .crossAsset,
        .driftGovernance,
        .probCalibration,
        .runtimeMonitor,
        .reports,
        .settings
    ]

    @Test
    func operatorShellRendersAcrossResolutionMatrix() throws {
        _ = NSApplication.shared
        let outputDirectory = snapshotOutputDirectory()

        for scenario in GUIValidationScenario.screenshotMatrix {
            for destination in destinations {
                let stats = try GUISnapshotRenderer.captureRootView(
                    selection: destination,
                    scenario: scenario,
                    outputDirectory: outputDirectory
                )

                #expect(stats.opaquePixelRatio > 0.50, "\(destination.title) at \(scenario.title) rendered with too much transparency.")
                #expect(stats.luminanceVariance > 0.0025, "\(destination.title) at \(scenario.title) looks visually flat or blank.")
                #expect(stats.bucketCount > 7, "\(destination.title) at \(scenario.title) does not show enough visual differentiation.")
            }
        }
    }

    private func snapshotOutputDirectory() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FXAI_GUI_WRITE_SNAPSHOTS"] == "1" else {
            return nil
        }
        if let explicit = environment["FXAI_GUI_SNAPSHOT_DIR"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit, isDirectory: true)
        }
        return URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("fxai-gui-snapshots", isDirectory: true)
    }
}
