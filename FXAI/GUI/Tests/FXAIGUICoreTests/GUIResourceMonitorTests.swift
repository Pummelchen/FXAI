import Testing
@testable import FXAIGUICore

@MainActor
struct GUIResourceMonitorTests {
    @Test
    func memoryPressureAndAppActivityRecomputeProfile() {
        let monitor = GUIResourceMonitor(initialProfile: .default)

        monitor.handleMemoryPressure(.warning)
        #expect(monitor.profile.memoryPressure == .warning)
        #expect(monitor.profile.pressureLevel == .constrained)

        monitor.handleApplicationActivityChange(isActive: false)
        #expect(monitor.profile.isAppActive == false)
        #expect(monitor.profile.pressureLevel >= .constrained)
    }
}
