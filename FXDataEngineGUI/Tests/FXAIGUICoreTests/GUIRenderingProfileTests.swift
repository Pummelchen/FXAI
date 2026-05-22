import FXAIGUICore
import Foundation
import Testing

struct GUIRenderingProfileTests {
    @Test
    func normalProfileKeepsPremiumEffectsEnabled() {
        let profile = GUIRenderingProfile.resolve(
            appIsActive: true,
            thermalState: .nominal,
            memoryPressure: .normal,
            isLowPowerModeEnabled: false
        )

        #expect(profile.pressureLevel == .normal)
        #expect(profile.allowsGlassBackdrop)
        #expect(profile.allowsMetalEffects)
        #expect(profile.allowsHeatmapMetal)
        #expect(profile.softReconnectInterval == 10)
        #expect(profile.maximumHeatmapCellCountForMetal == 1_024)
    }

    @Test
    func constrainedProfileBacksOffHeavyEffectsBeforeDisablingCoreUi() {
        let profile = GUIRenderingProfile.resolve(
            appIsActive: true,
            thermalState: .serious,
            memoryPressure: .warning,
            isLowPowerModeEnabled: false
        )

        #expect(profile.pressureLevel == .constrained)
        #expect(profile.allowsGlassBackdrop)
        #expect(!profile.allowsMetalEffects)
        #expect(!profile.allowsHeatmapMetal)
        #expect(profile.glassOpacityScale < 0.9)
        #expect(profile.backgroundBlurScale < 0.7)
        #expect(profile.softReconnectInterval >= 18)
    }

    @Test
    func criticalOrInactiveProfilePrefersProtectionOverDecoration() {
        let profile = GUIRenderingProfile.resolve(
            appIsActive: false,
            thermalState: .critical,
            memoryPressure: .critical,
            isLowPowerModeEnabled: true
        )

        #expect(profile.pressureLevel == .critical)
        #expect(!profile.allowsGlassBackdrop)
        #expect(!profile.allowsMetalEffects)
        #expect(!profile.allowsHeatmapMetal)
        #expect(profile.decorativeOpacityScale < 0.5)
        #expect(profile.softReconnectInterval >= 30)
        #expect(profile.maximumHeatmapCellCountForMetal == 0)
    }
}
