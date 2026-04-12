import FXAIGUICore
@testable import FXAIGUIApp
import Testing

@MainActor
struct RoleWorkspaceNavigationTests {
    @Test
    func activateRoleWorkspaceRoutesToDocumentedDefaultDestinations() {
        let model = FXAIGUIModel.validationFixture()

        model.activateRoleWorkspace(.liveTrader)
        #expect(model.selectedRole == .liveTrader)
        #expect(model.selection == .liveOverview)

        model.activateRoleWorkspace(.demoTrader)
        #expect(model.selectedRole == .demoTrader)
        #expect(model.selection == .demoOverview)

        model.activateRoleWorkspace(.researcher)
        #expect(model.selectedRole == .researcher)
        #expect(model.selection == .researchWorkspace)

        model.activateRoleWorkspace(.architect)
        #expect(model.selectedRole == .architect)
        #expect(model.selection == .platformControl)
    }

    @Test
    func roleWorkspacePanelsCanMoveResizeAndReset() throws {
        let model = FXAIGUIModel.validationFixture()
        let baseline = model.roleWorkspaceLayout(for: .liveTrader)
        let panel = try #require(baseline.panels.first(where: { $0.kind == .quickScreens }))

        model.moveRoleWorkspacePanelOnGrid(role: .liveTrader, panelID: panel.id, columnDelta: 1, rowDelta: 1)
        let moved = try #require(model.roleWorkspaceLayout(for: .liveTrader).panels.first(where: { $0.id == panel.id }))
        #expect(moved.columnUnits == panel.columnUnits + 1)
        #expect(moved.rowUnits == panel.rowUnits + 1)

        model.resizeRoleWorkspacePanel(role: .liveTrader, panelID: panel.id, widthDelta: 1, heightDelta: 1)
        let resized = try #require(model.roleWorkspaceLayout(for: .liveTrader).panels.first(where: { $0.id == panel.id }))
        #expect(resized.widthUnits >= panel.widthUnits)
        #expect(resized.heightUnits >= panel.heightUnits)

        model.resetRoleWorkspaceLayout(for: .liveTrader)
        let reset = model.roleWorkspaceLayout(for: .liveTrader)
        #expect(reset.panels.map(\.kind) == baseline.panels.map(\.kind))
        #expect(reset.panels.map(\.widthUnits) == baseline.panels.map(\.widthUnits))
        #expect(reset.panels.map(\.heightUnits) == baseline.panels.map(\.heightUnits))
        #expect(reset.panels.map(\.columnUnits) == baseline.panels.map(\.columnUnits))
        #expect(reset.panels.map(\.rowUnits) == baseline.panels.map(\.rowUnits))
    }
}
