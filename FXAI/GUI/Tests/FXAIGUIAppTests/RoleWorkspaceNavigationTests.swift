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
}
