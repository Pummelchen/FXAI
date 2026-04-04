import FXAIGUICore
import SwiftUI

struct BarChartView: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme

    var body: some View {
        ChartRenderer(frameModel: frameModel, theme: theme)
    }
}
