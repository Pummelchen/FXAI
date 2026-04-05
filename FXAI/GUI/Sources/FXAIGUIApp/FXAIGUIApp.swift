import SwiftUI

@main
struct FXAIGUIApp: App {
    @StateObject private var appState = FinanceAppState()

    var body: some Scene {
        WindowGroup("FXAI GUI") {
            FXAIRootView()
                .environmentObject(appState.model)
                .environmentObject(appState.themeEnvironment)
                .background(Color.clear)
                .overlay(WindowConfigurator().allowsHitTesting(false))
        }
        .defaultSize(width: 1728, height: 1117)
        .windowResizability(.automatic)
        .commands {
            CommandMenu("FXAI") {
                Button("Refresh State") {
                    Task { await appState.model.refresh() }
                }
                .keyboardShortcut("r", modifiers: [.command])

                Button("Save Current View") {
                    appState.model.saveCurrentView()
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])

                Divider()

                Button("Connect Project") {
                    appState.model.chooseProjectRoot()
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])

                Button("Reconnect Project") {
                    appState.model.reconnectProject()
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])

                Button("Disconnect Project") {
                    appState.model.disconnectProject()
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            }

            CommandMenu("Theme") {
                ForEach(appState.themeEnvironment.allThemes, id: \.themeID) { theme in
                    Button(theme.displayName) {
                        appState.themeEnvironment.activateTheme(theme.themeID)
                    }
                    .keyboardShortcut(theme.themeID == .financialDashboardV1 ? "1" : "0", modifiers: [.command, .option])
                }
            }
        }
    }
}
