import FXAIGUICore
import SwiftUI

struct LayoutGuideOverlay: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme

    var body: some View {
        ZStack(alignment: .topLeading) {
            outline(frameModel.mainPanelFrame, color: theme.colors.debugOutline, label: "Main Panel")
            outline(frameModel.footerFrame, color: theme.colors.debugGuide, label: "Footer")
            outline(frameModel.gaugeFrame, color: theme.colors.debugGuide, label: "Gauge")
            outline(frameModel.amountOwedFrame, color: theme.colors.debugOutline, label: "Amount Owed")
            outline(frameModel.chartPlotFrame, color: theme.colors.debugGuide, label: "Chart")

            ForEach(Array(frameModel.topCardFrames.keys), id: \.self) { key in
                if let frame = frameModel.topCardFrames[key] {
                    outline(frame, color: theme.colors.debugGuide.opacity(0.72), label: key.rawValue)
                }
            }
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private func outline(_ frame: CGRect, color: Color, label: String) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 6 * frameModel.scale, style: .continuous)
                .stroke(color.opacity(0.72), style: StrokeStyle(lineWidth: 1 * frameModel.scale, dash: [6 * frameModel.scale, 5 * frameModel.scale]))
                .frame(width: frame.width, height: frame.height)
                .offset(x: frame.minX, y: frame.minY)

            Text(label)
                .font(.system(size: 10 * frameModel.scale, weight: .semibold, design: .rounded))
                .padding(.horizontal, 6 * frameModel.scale)
                .padding(.vertical, 3 * frameModel.scale)
                .background(Capsule().fill(color.opacity(0.16)))
                .foregroundStyle(color)
                .offset(x: frame.minX + 8 * frameModel.scale, y: frame.minY - 18 * frameModel.scale)
        }
    }
}
