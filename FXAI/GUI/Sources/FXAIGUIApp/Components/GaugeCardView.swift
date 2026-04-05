import FXAIGUICore
import SwiftUI

struct GaugeCardView: View {
    let frame: CGRect
    let stageSize: CGSize
    let theme: any AppTheme
    let scale: CGFloat

    var body: some View {
        let style = theme.components.gaugeCard
        let cornerRadius = theme.cornerRadii.standardCard * scale

        ZStack(alignment: .topLeading) {
            ShadowStackRenderer(
                size: frame.size,
                cornerRadius: cornerRadius,
                shadow: style.shadow,
                scale: scale,
                context: ShadowProjectionContext(frame: frame, stageSize: stageSize, lightSource: theme.shadows.lightSource)
            )

            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(style.backgroundGradient)
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(theme.colors.cardStroke, lineWidth: 0.75 * scale)
                )

            VStack(alignment: .leading, spacing: 24 * scale) {
                HStack {
                    Spacer()
                    Image(systemName: "ellipsis")
                        .font(.system(size: 15 * scale, weight: .semibold))
                        .foregroundStyle(theme.colors.textSecondary.opacity(0.84))
                }

                GaugeRenderer(progress: 0.45, theme: theme, scale: scale)
                    .frame(height: 168 * scale)
                    .padding(.top, 12 * scale)

                Spacer()
            }
            .padding(.horizontal, theme.spacing.cardPadding * scale - 2 * scale)
            .padding(.top, theme.spacing.stackGap * scale)
        }
        .frame(width: frame.width, height: frame.height)
        .position(x: frame.midX, y: frame.midY)
    }
}
