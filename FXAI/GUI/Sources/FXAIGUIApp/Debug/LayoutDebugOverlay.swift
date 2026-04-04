import FXAIGUICore
import SwiftUI

struct LayoutDebugOverlay: View {
    let frameModel: DashboardFrameModel
    let layoutOutput: DashboardLayoutOutput
    let theme: any AppTheme

    var body: some View {
        ZStack(alignment: .topLeading) {
            ForEach(SVGSemanticMapping.baselineMappings, id: \.zone) { mapping in
                let frame = SVGMetrics.scaledRect(mapping.frame, scale: frameModel.scale)
                let isHidden = layoutOutput.hiddenZones.contains(mapping.zone)

                RoundedRectangle(cornerRadius: 8 * frameModel.scale, style: .continuous)
                    .stroke(
                        isHidden ? theme.colors.textMuted.opacity(0.45) : theme.colors.debugGuide.opacity(0.72),
                        style: StrokeStyle(
                            lineWidth: 1 * frameModel.scale,
                            dash: [8 * frameModel.scale, 5 * frameModel.scale]
                        )
                    )
                    .frame(width: frame.width, height: frame.height)
                    .offset(x: frame.minX, y: frame.minY)

                Text(mapping.zone.rawValue)
                    .font(.system(size: 10 * frameModel.scale, weight: .semibold, design: .rounded))
                    .foregroundStyle(isHidden ? theme.colors.textMuted : theme.colors.debugGuide)
                    .padding(.horizontal, 6 * frameModel.scale)
                    .padding(.vertical, 3 * frameModel.scale)
                    .background(
                        Capsule(style: .continuous)
                            .fill(theme.colors.mainPanel.opacity(0.74))
                    )
                    .offset(x: frame.minX + 6 * frameModel.scale, y: frame.minY - 18 * frameModel.scale)
            }

            VStack(alignment: .leading, spacing: 4 * frameModel.scale) {
                Text("Layout \(frameModel.layoutClass.displayName)")
                Text("KPI \(layoutOutput.kpiArrangement.rawValue)")
                Text("Chart \(layoutOutput.chartPlacement.rawValue)")
            }
            .font(.system(size: 11 * frameModel.scale, weight: .semibold, design: .rounded))
            .foregroundStyle(theme.colors.debugOutline)
            .padding(10 * frameModel.scale)
            .background(
                RoundedRectangle(cornerRadius: 10 * frameModel.scale, style: .continuous)
                    .fill(theme.colors.mainPanel.opacity(0.74))
            )
            .offset(x: frameModel.mainPanelFrame.minX + 18 * frameModel.scale, y: frameModel.mainPanelFrame.minY + 18 * frameModel.scale)
        }
        .allowsHitTesting(false)
    }
}
