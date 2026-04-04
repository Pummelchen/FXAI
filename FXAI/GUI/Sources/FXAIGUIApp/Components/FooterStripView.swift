import FXAIGUICore
import SwiftUI

struct FooterStripView: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text(SVGMetrics.footerDate)
                .font(theme.typography.footer.font(scaledBy: frameModel.scale * 1.32))
                .foregroundStyle(theme.colors.textSecondary.opacity(0.68))
                .position(frameModel.footerDateCenter)

            Text(SVGMetrics.footerTime)
                .font(theme.typography.footer.font(scaledBy: frameModel.scale * 1.18))
                .foregroundStyle(theme.colors.textSecondary.opacity(0.58))
                .position(frameModel.footerTimeCenter)

            Text(SVGMetrics.footerDay)
                .font(theme.typography.footer.font(scaledBy: frameModel.scale * 1.18))
                .foregroundStyle(theme.colors.textSecondary.opacity(0.58))
                .position(frameModel.footerDayCenter)
        }
    }
}
