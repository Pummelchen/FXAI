import FXGUICore
import SwiftUI

@MainActor
final class FinanceAppState: ObservableObject {
    @Published var themeEnvironment: ThemeEnvironment
    @Published var resourceMonitor: GUIResourceMonitor
    @Published var model: FXGUIModel

    init(
        themeEnvironment: ThemeEnvironment = ThemeBootstrap.makeThemeEnvironment(),
        resourceMonitor: GUIResourceMonitor = .shared,
        model: FXGUIModel? = nil
    ) {
        self.themeEnvironment = themeEnvironment
        self.resourceMonitor = resourceMonitor
        self.model = model ?? FXGUIModel(resourceMonitor: resourceMonitor)
    }
}
