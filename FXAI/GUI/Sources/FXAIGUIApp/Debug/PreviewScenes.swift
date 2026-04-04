import FXAIGUICore
import SwiftUI

private struct PreviewDashboardHarness: View {
    @StateObject private var model = FXAIGUIModel()
    @StateObject private var themeEnvironment = ThemeEnvironment.preview()

    var body: some View {
        DashboardRootView()
            .environmentObject(model)
            .environmentObject(themeEnvironment)
    }
}

#Preview("MacBook 14") {
    PreviewDashboardHarness()
        .frame(width: 1512, height: 982)
}

#Preview("Standard Desktop") {
    PreviewDashboardHarness()
        .frame(width: 1728, height: 1117)
}

#Preview("Wide Desktop") {
    PreviewDashboardHarness()
        .frame(width: 2200, height: 1280)
}

#Preview("4K / Ultra Wide") {
    PreviewDashboardHarness()
        .frame(width: 3440, height: 1440)
}

#Preview("8K-Like") {
    PreviewDashboardHarness()
        .frame(width: 5120, height: 2160)
}
