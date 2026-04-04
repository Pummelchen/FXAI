import SwiftUI

@main
struct FXAIGUIApp: App {
    @StateObject private var model = FXAIGUIModel()

    var body: some Scene {
        WindowGroup("FXAI GUI") {
            DashboardRootView()
                .environmentObject(model)
                .background(Color.clear)
                .overlay(WindowConfigurator().allowsHitTesting(false))
        }
        .defaultSize(width: 1728, height: 1117)
        .windowResizability(.automatic)
        .commands {
            CommandMenu("FXAI") {
                Button("Refresh State") {
                    Task { await model.refresh() }
                }
                .keyboardShortcut("r", modifiers: [.command])

                Button("Save Current View") {
                    model.saveCurrentView()
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])

                Divider()

                Button("Connect Project") {
                    model.chooseProjectRoot()
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])

                Button("Reconnect Project") {
                    model.reconnectProject()
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])

                Button("Disconnect Project") {
                    model.disconnectProject()
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            }
        }
    }
}
