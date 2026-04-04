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

                Button("Open Onboarding") {
                    model.navigate(to: .onboarding)
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])

                Button("Open Incident Center") {
                    model.navigate(to: .incidents)
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])

                Button("Open Command Center") {
                    model.navigate(to: .commands)
                }
                .keyboardShortcut("t", modifiers: [.command, .shift])
            }

            CommandMenu("Navigate") {
                Button("Overview") {
                    model.navigate(to: .overview)
                }
                .keyboardShortcut("1", modifiers: [.command])

                Button("Role Workspaces") {
                    model.navigate(to: .roles)
                }
                .keyboardShortcut("2", modifiers: [.command])

                Button("Runtime Monitor") {
                    model.navigate(to: .runtimeMonitor)
                }
                .keyboardShortcut("3", modifiers: [.command])

                Button("Research OS Control") {
                    model.navigate(to: .researchControl)
                }
                .keyboardShortcut("4", modifiers: [.command])

                Button("Advanced Visuals") {
                    model.navigate(to: .advancedVisuals)
                }
                .keyboardShortcut("5", modifiers: [.command])

                Button("Command Center") {
                    model.navigate(to: .commands)
                }
                .keyboardShortcut("6", modifiers: [.command])
            }
        }
    }
}
