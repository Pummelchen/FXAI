import FXAIGUICore
import SwiftUI

@MainActor
final class FinanceAppState: ObservableObject {
    @Published var themeEnvironment: ThemeEnvironment
    @Published var resourceMonitor: GUIResourceMonitor
    @Published var model: FXAIGUIModel

    init(
        themeEnvironment: ThemeEnvironment = ThemeBootstrap.makeThemeEnvironment(),
        resourceMonitor: GUIResourceMonitor = .shared,
        model: FXAIGUIModel? = nil
    ) {
        self.themeEnvironment = themeEnvironment
        self.resourceMonitor = resourceMonitor
        self.model = model ?? FXAIGUIModel(resourceMonitor: resourceMonitor)
    }
}
