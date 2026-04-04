import Foundation

public enum DashboardLayoutClass: String, CaseIterable {
    case compactDesktop
    case standardDesktop
    case wideDesktop
    case ultraWideDesktop

    public var displayName: String {
        switch self {
        case .compactDesktop: "Compact Desktop"
        case .standardDesktop: "Standard Desktop"
        case .wideDesktop: "Wide Desktop"
        case .ultraWideDesktop: "Ultra-Wide / 4K+"
        }
    }
}
