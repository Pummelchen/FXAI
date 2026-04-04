import FXAIGUICore
import SwiftUI

struct AmountOwedGlassCardView: View {
    let frame: CGRect
    let theme: any AppTheme
    let scale: CGFloat
    let intensity: CGFloat
    let reducedEffects: Bool

    var body: some View {
        let style = theme.components.amountOwed
        let cornerRadius = theme.cornerRadii.glassCard * scale

        ZStack(alignment: .topLeading) {
            GlowRenderer(glow: style.glow, scale: scale, intensity: intensity)
                .offset(x: frame.width * 0.2, y: frame.height * 0.14)
                .opacity(reducedEffects ? 0.42 : 1)

            ShadowStackRenderer(size: frame.size, cornerRadius: cornerRadius, shadow: style.shadow, scale: scale)

            GlassRenderer(size: frame.size, cornerRadius: cornerRadius, theme: theme, scale: scale)

            VStack(alignment: .leading, spacing: 14 * scale) {
                Image(systemName: "chart.bar.doc.horizontal")
                    .font(.system(size: 26 * scale, weight: .semibold))
                    .foregroundStyle(theme.colors.textPrimary.opacity(0.9))

                Text("Amount Owed")
                    .font(theme.typography.cardTitle.font(scaledBy: scale))
                    .foregroundStyle(theme.colors.textSecondary.opacity(0.9))

                Text("₹ 8,76,489")
                    .font(.system(size: 26 * scale, weight: .semibold, design: .rounded))
                    .tracking(-0.38 * scale)
                    .foregroundStyle(theme.colors.textPrimary.opacity(0.92))
            }
            .padding(.horizontal, (theme.spacing.cardPadding + 2) * scale)
            .padding(.vertical, theme.spacing.stackGap * scale)
        }
        .frame(width: frame.width, height: frame.height)
        .position(x: frame.midX, y: frame.midY)
    }
}
