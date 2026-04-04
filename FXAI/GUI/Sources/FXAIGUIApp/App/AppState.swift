import FXAIGUICore
import SwiftUI

@MainActor
final class FinanceAppState: ObservableObject {
    @Published var themeEnvironment: ThemeEnvironment
    @Published var model: FXAIGUIModel

    init(
        themeEnvironment: ThemeEnvironment = ThemeBootstrap.makeThemeEnvironment(),
        model: FXAIGUIModel = FXAIGUIModel()
    ) {
        self.themeEnvironment = themeEnvironment
        self.model = model
    }
}
