import FXAIGUICore
import SwiftUI

struct SidebarView: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme

    private let icons: [String] = [
        "speedometer",
        "person.crop.square",
        "paperplane.fill",
        "list.bullet.clipboard",
        "gearshape.fill"
    ]

    var body: some View {
        let baseX = 30 * frameModel.scale
        let startY = 318 * frameModel.scale
        let step = 125 * frameModel.scale

        ZStack(alignment: .topLeading) {
            ForEach(Array(icons.enumerated()), id: \.offset) { item in
                let index = item.offset
                let icon = item.element
                Image(systemName: icon)
                    .font(.system(size: 22 * frameModel.scale, weight: .semibold))
                    .symbolRenderingMode(.monochrome)
                    .foregroundStyle(theme.colors.textSecondary.opacity(index == 2 ? 0.92 : 0.68))
                    .frame(width: 38 * frameModel.scale, height: 38 * frameModel.scale)
                    .position(x: baseX, y: startY + CGFloat(index) * step)
            }
        }
    }
}
