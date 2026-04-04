import FXAIGUICore
import SwiftUI

struct InvoiceMetricCardView: View {
    let kind: SVGInvoiceCardKind
    let frame: CGRect
    let content: InvoiceMetricContent
    let theme: any AppTheme
    let scale: CGFloat

    var body: some View {
        let cornerRadius = theme.cornerRadii.smallCard * scale
        let ringTint = content.ringTintHex.map { Color(hex: $0) } ?? theme.colors.textSecondary

        ZStack(alignment: .topLeading) {
            ShadowStackRenderer(size: frame.size, cornerRadius: cornerRadius, shadow: theme.shadows.smallCard, scale: scale)

            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(theme.gradients.standardCard)
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(theme.colors.cardStroke, lineWidth: 0.72 * scale)
                )

            VStack(alignment: .leading, spacing: 14 * scale) {
                HStack(alignment: .top) {
                    Image(systemName: kind == .paidInvoices ? "creditcard" : "creditcard.fill")
                        .font(.system(size: 16 * scale, weight: .semibold))
                        .foregroundStyle(theme.colors.textPrimary.opacity(0.86))

                    Spacer(minLength: 8 * scale)

                    if let ringText = content.ringText, let progress = content.ringProgress {
                        TrendRingRenderer(
                            text: ringText,
                            progress: progress,
                            scale: scale * 0.92,
                            theme: theme,
                            tint: ringTint
                        )
                        .frame(width: 48 * scale, height: 48 * scale)
                    }
                }

                Spacer(minLength: 0)

                VStack(alignment: .leading, spacing: 6 * scale) {
                    Text(content.title)
                        .font(theme.typography.cardTitle.font(scaledBy: scale))
                        .foregroundStyle(theme.colors.textSecondary.opacity(0.9))

                    Text(content.value)
                        .font(theme.typography.bodyValue.font(scaledBy: scale * 1.16))
                        .tracking(theme.typography.bodyValue.tracking * scale)
                        .foregroundStyle(theme.colors.textPrimary)

                    Text(content.subtitle)
                        .font(theme.typography.caption.font(scaledBy: scale))
                        .foregroundStyle(theme.colors.textMuted)
                }
            }
            .padding(.horizontal, 18 * scale)
            .padding(.vertical, 16 * scale)
        }
        .frame(width: frame.width, height: frame.height)
        .position(x: frame.midX, y: frame.midY)
    }
}
