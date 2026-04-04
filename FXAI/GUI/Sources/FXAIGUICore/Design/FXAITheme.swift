import SwiftUI

@MainActor
public enum FXAITheme {
    private static var current: any AppTheme { ThemeManager.shared.currentTheme }

    public static var background: Color { current.colors.outerBackground }
    public static var backgroundSecondary: Color { current.colors.mainPanel.opacity(0.92) }
    public static var panel: Color { current.colors.mainPanel.opacity(0.92) }
    public static var panelStrong: Color { current.colors.mainPanel }
    public static var stroke: Color { current.colors.cardStroke }
    public static var textPrimary: Color { current.colors.textPrimary }
    public static var textSecondary: Color { current.colors.textSecondary }
    public static var textMuted: Color { current.colors.textMuted }
    public static var accent: Color { current.colors.successGreen }
    public static var accentSoft: Color { current.colors.warningGreen }
    public static var warning: Color { current.colors.warningGreen }
    public static var success: Color { current.colors.successGreen }
    public static var canvasGradient: LinearGradient { current.gradients.canvas }
    public static var heroGradient: LinearGradient { current.gradients.standardCard }
}
