import Foundation
import SwiftUI

public enum GUIMemoryPressureLevel: Int, Sendable, Comparable {
    case normal
    case warning
    case critical

    public static func < (lhs: GUIMemoryPressureLevel, rhs: GUIMemoryPressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public enum GUIResourcePressureLevel: Int, Sendable, Comparable {
    case normal
    case elevated
    case constrained
    case critical

    public static func < (lhs: GUIResourcePressureLevel, rhs: GUIResourcePressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public struct GUIRenderingProfile: Sendable, Equatable {
    public let pressureLevel: GUIResourcePressureLevel
    public let memoryPressure: GUIMemoryPressureLevel
    public let isAppActive: Bool
    public let allowsGlassBackdrop: Bool
    public let allowsMetalEffects: Bool
    public let allowsHeatmapMetal: Bool
    public let glassOpacityScale: Double
    public let shadowOpacityScale: Double
    public let backgroundBlurScale: Double
    public let decorativeOpacityScale: Double
    public let softReconnectInterval: TimeInterval
    public let maximumHeatmapCellCountForMetal: Int

    public init(
        pressureLevel: GUIResourcePressureLevel,
        memoryPressure: GUIMemoryPressureLevel,
        isAppActive: Bool,
        allowsGlassBackdrop: Bool,
        allowsMetalEffects: Bool,
        allowsHeatmapMetal: Bool,
        glassOpacityScale: Double,
        shadowOpacityScale: Double,
        backgroundBlurScale: Double,
        decorativeOpacityScale: Double,
        softReconnectInterval: TimeInterval,
        maximumHeatmapCellCountForMetal: Int
    ) {
        self.pressureLevel = pressureLevel
        self.memoryPressure = memoryPressure
        self.isAppActive = isAppActive
        self.allowsGlassBackdrop = allowsGlassBackdrop
        self.allowsMetalEffects = allowsMetalEffects
        self.allowsHeatmapMetal = allowsHeatmapMetal
        self.glassOpacityScale = glassOpacityScale
        self.shadowOpacityScale = shadowOpacityScale
        self.backgroundBlurScale = backgroundBlurScale
        self.decorativeOpacityScale = decorativeOpacityScale
        self.softReconnectInterval = softReconnectInterval
        self.maximumHeatmapCellCountForMetal = maximumHeatmapCellCountForMetal
    }

    public var usesReducedEffects: Bool {
        pressureLevel >= .constrained || !isAppActive
    }

    public var allowsPeriodicReconnectChecks: Bool {
        pressureLevel < .critical || isAppActive
    }

    public static let `default` = GUIRenderingProfile.resolve(
        appIsActive: true,
        thermalState: .nominal,
        memoryPressure: .normal,
        isLowPowerModeEnabled: false
    )

    public static func resolve(
        appIsActive: Bool,
        thermalState: ProcessInfo.ThermalState,
        memoryPressure: GUIMemoryPressureLevel,
        isLowPowerModeEnabled: Bool
    ) -> GUIRenderingProfile {
        let pressureLevel = resolvedPressureLevel(
            appIsActive: appIsActive,
            thermalState: thermalState,
            memoryPressure: memoryPressure,
            isLowPowerModeEnabled: isLowPowerModeEnabled
        )

        switch pressureLevel {
        case .normal:
            return GUIRenderingProfile(
                pressureLevel: pressureLevel,
                memoryPressure: memoryPressure,
                isAppActive: appIsActive,
                allowsGlassBackdrop: true,
                allowsMetalEffects: true,
                allowsHeatmapMetal: true,
                glassOpacityScale: 1.0,
                shadowOpacityScale: 1.0,
                backgroundBlurScale: 1.0,
                decorativeOpacityScale: 1.0,
                softReconnectInterval: 10,
                maximumHeatmapCellCountForMetal: 1_024
            )
        case .elevated:
            return GUIRenderingProfile(
                pressureLevel: pressureLevel,
                memoryPressure: memoryPressure,
                isAppActive: appIsActive,
                allowsGlassBackdrop: true,
                allowsMetalEffects: !isLowPowerModeEnabled && appIsActive,
                allowsHeatmapMetal: appIsActive,
                glassOpacityScale: 0.92,
                shadowOpacityScale: 0.88,
                backgroundBlurScale: 0.82,
                decorativeOpacityScale: 0.86,
                softReconnectInterval: appIsActive ? 12 : 20,
                maximumHeatmapCellCountForMetal: 640
            )
        case .constrained:
            return GUIRenderingProfile(
                pressureLevel: pressureLevel,
                memoryPressure: memoryPressure,
                isAppActive: appIsActive,
                allowsGlassBackdrop: appIsActive,
                allowsMetalEffects: false,
                allowsHeatmapMetal: false,
                glassOpacityScale: 0.78,
                shadowOpacityScale: 0.72,
                backgroundBlurScale: 0.56,
                decorativeOpacityScale: 0.62,
                softReconnectInterval: appIsActive ? 18 : 30,
                maximumHeatmapCellCountForMetal: 0
            )
        case .critical:
            return GUIRenderingProfile(
                pressureLevel: pressureLevel,
                memoryPressure: memoryPressure,
                isAppActive: appIsActive,
                allowsGlassBackdrop: false,
                allowsMetalEffects: false,
                allowsHeatmapMetal: false,
                glassOpacityScale: 0.64,
                shadowOpacityScale: 0.54,
                backgroundBlurScale: 0.38,
                decorativeOpacityScale: 0.38,
                softReconnectInterval: appIsActive ? 30 : 45,
                maximumHeatmapCellCountForMetal: 0
            )
        }
    }

    private static func resolvedPressureLevel(
        appIsActive: Bool,
        thermalState: ProcessInfo.ThermalState,
        memoryPressure: GUIMemoryPressureLevel,
        isLowPowerModeEnabled: Bool
    ) -> GUIResourcePressureLevel {
        var pressureLevel: GUIResourcePressureLevel = .normal

        if !appIsActive {
            pressureLevel = max(pressureLevel, .elevated)
        }

        if isLowPowerModeEnabled {
            pressureLevel = max(pressureLevel, .elevated)
        }

        switch thermalState {
        case .nominal:
            break
        case .fair:
            pressureLevel = max(pressureLevel, .elevated)
        case .serious:
            pressureLevel = max(pressureLevel, .constrained)
        case .critical:
            pressureLevel = .critical
        @unknown default:
            pressureLevel = max(pressureLevel, .constrained)
        }

        switch memoryPressure {
        case .normal:
            break
        case .warning:
            pressureLevel = max(pressureLevel, .constrained)
        case .critical:
            pressureLevel = .critical
        }

        return pressureLevel
    }
}

private struct GUIRenderingProfileKey: EnvironmentKey {
    static let defaultValue = GUIRenderingProfile.default
}

public extension EnvironmentValues {
    var guiRenderingProfile: GUIRenderingProfile {
        get { self[GUIRenderingProfileKey.self] }
        set { self[GUIRenderingProfileKey.self] = newValue }
    }
}
