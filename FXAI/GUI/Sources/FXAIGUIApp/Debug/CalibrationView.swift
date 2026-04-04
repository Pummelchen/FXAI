import FXAIGUICore
import SwiftUI

struct CalibrationView: View {
    let frameModel: DashboardFrameModel
    let theme: any AppTheme
    let debugState: DashboardDebugState

    var body: some View {
        ZStack(alignment: .topLeading) {
            if debugState.overlayEnabled {
                SVGReferenceOverlay(showScreenshotReference: debugState.showScreenshotReference)
                    .opacity(debugState.overlayOpacity)
                    .clipShape(
                        RoundedRectangle(cornerRadius: theme.cornerRadii.panel * frameModel.scale, style: .continuous)
                    )
                    .frame(width: frameModel.mainPanelFrame.width, height: frameModel.mainPanelFrame.height)
                    .offset(x: frameModel.mainPanelFrame.minX, y: frameModel.mainPanelFrame.minY)
            }

            if debugState.showLayoutGuides || debugState.showFrameOutlines {
                LayoutGuideOverlay(frameModel: frameModel, theme: theme)
            }
        }
    }
}
