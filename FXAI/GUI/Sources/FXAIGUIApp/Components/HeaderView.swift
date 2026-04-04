import FXAIGUICore
import SwiftUI

struct HeaderView: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme

    var body: some View {
        ZStack(alignment: .topLeading) {
            HStack(spacing: 8 * frameModel.scale) {
                Text("Dashboard")
                    .font(theme.typography.headerTitle.font(scaledBy: frameModel.scale))
                    .tracking(theme.typography.headerTitle.tracking * frameModel.scale)
                    .foregroundStyle(theme.colors.textPrimary)

                Circle()
                    .trim(from: 0.06, to: 0.92)
                    .stroke(theme.colors.textSecondary.opacity(0.9), style: StrokeStyle(lineWidth: 1.6 * frameModel.scale, lineCap: .round, dash: [1.6 * frameModel.scale, 3.2 * frameModel.scale]))
                    .frame(width: 18 * frameModel.scale, height: 18 * frameModel.scale)
                    .offset(y: 2 * frameModel.scale)
            }
            .position(
                x: frameModel.headerTitleOrigin.x + 90 * frameModel.scale,
                y: frameModel.headerTitleOrigin.y + 6 * frameModel.scale
            )

            HStack(spacing: 4 * frameModel.scale) {
                Text("Financial")
                    .font(theme.typography.headerSubtitle.font(scaledBy: frameModel.scale))
                    .tracking(theme.typography.headerSubtitle.tracking * frameModel.scale)
                    .foregroundStyle(theme.colors.textSecondary.opacity(theme.typography.headerSubtitle.opacity))

                Image(systemName: "chevron.down")
                    .font(.system(size: 10 * frameModel.scale, weight: .semibold, design: .rounded))
                    .foregroundStyle(theme.colors.textMuted)
                    .offset(y: 1 * frameModel.scale)
            }
            .position(
                x: frameModel.headerSubtitleOrigin.x + 38 * frameModel.scale,
                y: frameModel.headerSubtitleOrigin.y + 3 * frameModel.scale
            )

            Text("Bills")
                .font(theme.typography.sectionTitle.font(scaledBy: frameModel.scale))
                .tracking(theme.typography.sectionTitle.tracking * frameModel.scale)
                .foregroundStyle(theme.colors.textPrimary)
                .position(
                    x: frameModel.billsOrigin.x + 34 * frameModel.scale,
                    y: frameModel.billsOrigin.y - 8 * frameModel.scale
                )

            Text("Invoices")
                .font(theme.typography.sectionTitle.font(scaledBy: frameModel.scale))
                .tracking(theme.typography.sectionTitle.tracking * frameModel.scale)
                .foregroundStyle(theme.colors.textPrimary)
                .position(
                    x: frameModel.invoicesOrigin.x + 54 * frameModel.scale,
                    y: frameModel.invoicesOrigin.y - 8 * frameModel.scale
                )
        }
    }
}
