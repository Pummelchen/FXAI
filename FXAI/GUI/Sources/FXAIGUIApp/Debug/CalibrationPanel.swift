import FXAIGUICore
import SwiftUI

struct CalibrationPanel: View {
    @Binding var debugState: DashboardDebugState

    let layoutOutput: DashboardLayoutOutput
    let theme: any AppTheme
    let connectionStatus: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            DebugOverlayControls(
                debugState: $debugState,
                frameModel: layoutOutput.frameModel,
                theme: theme,
                connectionStatus: connectionStatus
            )

            ThemeInspectorView(layoutOutput: layoutOutput)
        }
        .frame(width: 280)
    }
}
