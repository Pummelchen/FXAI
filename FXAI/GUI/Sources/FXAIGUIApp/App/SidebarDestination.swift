import Foundation

enum SidebarDestination: String, CaseIterable, Identifiable {
    case overview
    case roles
    case plugins
    case reports
    case commands
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .roles: "Role Workspaces"
        case .plugins: "Plugin Zoo"
        case .reports: "Reports Explorer"
        case .commands: "Command Center"
        case .settings: "Settings"
        }
    }

    var symbolName: String {
        switch self {
        case .overview: "square.grid.2x2.fill"
        case .roles: "person.3.fill"
        case .plugins: "shippingbox.fill"
        case .reports: "doc.text.image.fill"
        case .commands: "terminal.fill"
        case .settings: "slider.horizontal.3"
        }
    }
}
