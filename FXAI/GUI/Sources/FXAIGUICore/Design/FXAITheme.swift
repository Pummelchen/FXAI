import SwiftUI

public enum FXAITheme {
    public static let background = Color(red: 0.04, green: 0.055, blue: 0.08)
    public static let backgroundSecondary = Color(red: 0.07, green: 0.09, blue: 0.13)
    public static let panel = Color(red: 0.10, green: 0.13, blue: 0.18).opacity(0.88)
    public static let panelStrong = Color(red: 0.12, green: 0.16, blue: 0.22).opacity(0.96)
    public static let stroke = Color.white.opacity(0.09)
    public static let textPrimary = Color.white.opacity(0.94)
    public static let textSecondary = Color.white.opacity(0.68)
    public static let textMuted = Color.white.opacity(0.46)
    public static let accent = Color(red: 0.16, green: 0.84, blue: 0.79)
    public static let accentSoft = Color(red: 0.25, green: 0.60, blue: 0.97)
    public static let warning = Color(red: 1.00, green: 0.73, blue: 0.25)
    public static let success = Color(red: 0.36, green: 0.91, blue: 0.63)

    public static let canvasGradient = LinearGradient(
        colors: [
            Color(red: 0.03, green: 0.05, blue: 0.08),
            Color(red: 0.04, green: 0.08, blue: 0.11),
            Color(red: 0.02, green: 0.04, blue: 0.07)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    public static let heroGradient = LinearGradient(
        colors: [
            Color(red: 0.12, green: 0.18, blue: 0.28),
            Color(red: 0.06, green: 0.24, blue: 0.24),
            Color(red: 0.03, green: 0.08, blue: 0.11)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}
