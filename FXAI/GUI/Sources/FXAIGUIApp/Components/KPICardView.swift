import FXAIGUICore
import SwiftUI

struct KPICardView: View {
    let kind: SVGKPIKind
    let frame: CGRect
    let content: DashboardKPIContent
    let theme: any AppTheme
    let scale: CGFloat
    let showShadowDebug: Bool
    let decorativeIntensity: CGFloat

    var body: some View {
        let style = theme.components.kpiCard
        let shadow = content.highlightStyle == .pending ? style.pendingShadow : style.shadow
        let cornerRadius = theme.cornerRadii.standardCard * scale

        ZStack(alignment: .topLeading) {
            if content.highlightStyle == .pending {
                GlowRenderer(glow: theme.glows.pendingCard, scale: scale, intensity: decorativeIntensity)
                    .offset(x: frame.width * 0.18, y: frame.height * 0.18)
            }

            ShadowStackRenderer(size: frame.size, cornerRadius: cornerRadius, shadow: shadow, scale: scale)

            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(content.highlightStyle == .pending ? AnyShapeStyle(style.pendingGradient) : AnyShapeStyle(style.gradient))
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(theme.colors.cardStroke, lineWidth: showShadowDebug ? 1.0 * scale : 0.75 * scale)
                )

            VStack(alignment: .leading, spacing: 18 * scale) {
                HStack(alignment: .top) {
                    Image(systemName: content.iconSystemName)
                        .font(.system(size: 20 * scale, weight: .semibold))
                        .foregroundStyle(theme.colors.textPrimary.opacity(0.88))

                    Spacer(minLength: 12 * scale)

                    TrendRingRenderer(
                        text: content.delta,
                        progress: content.ringProgress,
                        scale: scale,
                        theme: theme,
                        tint: content.highlightStyle == .pending ? theme.colors.warningGreen : theme.colors.textSecondary
                    )
                    .frame(width: style.ringSize * scale, height: style.ringSize * scale)
                }

                Spacer(minLength: 0)

                VStack(alignment: .leading, spacing: 8 * scale) {
                    Text(content.title)
                        .font(theme.typography.cardTitle.font(scaledBy: scale))
                        .tracking(theme.typography.cardTitle.tracking * scale)
                        .foregroundStyle(theme.colors.textSecondary.opacity(theme.typography.cardTitle.opacity))

                    HStack(alignment: .firstTextBaseline, spacing: 2 * scale) {
                        Text(content.majorValue)
                            .font(theme.typography.kpiValueMajor.font(scaledBy: scale))
                            .tracking(theme.typography.kpiValueMajor.tracking * scale)
                            .foregroundStyle(theme.colors.textPrimary)

                        if !content.minorValue.isEmpty {
                            Text(content.minorValue)
                                .font(theme.typography.kpiValueMinor.font(scaledBy: scale))
                                .tracking(theme.typography.kpiValueMinor.tracking * scale)
                                .foregroundStyle(theme.colors.textSecondary)
                        }
                    }

                    Text(content.subtitle)
                        .font(theme.typography.caption.font(scaledBy: scale))
                        .tracking(theme.typography.caption.tracking * scale)
                        .foregroundStyle(theme.colors.textMuted.opacity(theme.typography.caption.opacity))
                        .lineSpacing(theme.typography.caption.lineSpacing * scale)
                }
            }
            .padding(.horizontal, theme.spacing.cardPadding * scale)
            .padding(.vertical, theme.spacing.compactCardPadding * scale)
        }
        .frame(width: frame.width, height: frame.height)
        .position(x: frame.midX, y: frame.midY)
    }
}
