import Foundation

enum SidebarDestination: String, CaseIterable, Identifiable {
    case overview
    case roles
    case auditLab
    case backtestBuilder
    case offlineLab
    case plugins
    case reports
    case commands
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .roles: "Role Workspaces"
        case .auditLab: "Audit Lab Builder"
        case .backtestBuilder: "Backtest Builder"
        case .offlineLab: "Offline Lab Builder"
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
        case .auditLab: "checklist.checked"
        case .backtestBuilder: "gauge.with.needle.fill"
        case .offlineLab: "gearshape.2.fill"
        case .plugins: "shippingbox.fill"
        case .reports: "doc.text.image.fill"
        case .commands: "terminal.fill"
        case .settings: "slider.horizontal.3"
        }
    }
}
