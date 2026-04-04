import CoreGraphics

public enum DashboardAdaptiveRules {
    public static let defaultContentPriorities: [DashboardZone: DashboardContentPriority] = [
        .header: .critical,
        .sidebar: .high,
        .kpis: .critical,
        .invoices: .high,
        .chart: .high,
        .footer: .medium,
        .amountOwedOverlay: .high,
        .ambientDecorations: .decorative
    ]

    public static func classify(containerSize: CGSize) -> DashboardLayoutClass {
        if containerSize.width < 1400 || containerSize.height < 920 {
            return .compactDesktop
        }
        if containerSize.width < 1900 {
            return .standardDesktop
        }
        if containerSize.width < 3200 {
            return .wideDesktop
        }
        return .ultraWideDesktop
    }

    public static func decorativeVisibility(
        layoutClass: DashboardLayoutClass,
        scale: CGFloat,
        theme: any AppTheme
    ) -> DashboardDecorativeVisibility {
        switch layoutClass {
        case .compactDesktop:
            return DashboardDecorativeVisibility(
                glowIntensity: max(0.58, scale * 0.78),
                ambientOpacity: 0.28,
                hideSecondaryDecorations: true,
                metalOverlayEnabled: true
            )
        case .standardDesktop:
            return DashboardDecorativeVisibility(
                glowIntensity: max(0.82, scale * 0.9),
                ambientOpacity: 0.44,
                hideSecondaryDecorations: false,
                metalOverlayEnabled: true
            )
        case .wideDesktop:
            return DashboardDecorativeVisibility(
                glowIntensity: min(1.08, scale),
                ambientOpacity: 0.56,
                hideSecondaryDecorations: false,
                metalOverlayEnabled: true
            )
        case .ultraWideDesktop:
            return DashboardDecorativeVisibility(
                glowIntensity: min(1.22, scale * 0.92),
                ambientOpacity: 0.66,
                hideSecondaryDecorations: false,
                metalOverlayEnabled: true
            )
        }
    }

    public static func typographyScale(for baseScale: CGFloat, policy: DashboardScalePolicy) -> CGFloat {
        min(policy.maximumTypographyScale, max(policy.minimumTypographyScale, baseScale))
    }

    public static func spacingScale(for baseScale: CGFloat, policy: DashboardScalePolicy) -> CGFloat {
        min(policy.maximumSpacingScale, max(policy.minimumSpacingScale, baseScale))
    }
}
