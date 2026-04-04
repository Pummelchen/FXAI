import SwiftUI

@main
struct FXAIGUIApp: App {
    @StateObject private var model = FXAIGUIModel()

    var body: some Scene {
        WindowGroup("FXAI GUI") {
            FXAIRootView()
                .environmentObject(model)
                .background(FXAIBackgroundView())
                .overlay(WindowConfigurator().allowsHitTesting(false))
        }
        .defaultSize(width: 1520, height: 980)
        .windowResizability(.contentSize)
    }
}
