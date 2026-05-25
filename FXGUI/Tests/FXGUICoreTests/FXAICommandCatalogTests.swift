import FXGUICore
import Foundation
import Testing

@Suite("FXAI command catalog")
struct FXAICommandCatalogTests {
    @Test func exposesCertificationAndSafetyCommands() throws {
        let snapshot = FXAICommandCatalog.snapshot(projectRoot: URL(fileURLWithPath: "/tmp/FXAI"))

        #expect(snapshot.command(id: "fxai.certify.all") != nil)
        #expect(snapshot.command(id: "fxdatabase.status")?.executionPath == .versionedAPI)
        #expect(snapshot.command(id: "fxexecution.kill_switch.status")?.riskLevel == .liveExecution)
    }
}
