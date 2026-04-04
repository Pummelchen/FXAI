import Foundation

public enum DashboardZone: String, CaseIterable, Sendable {
    case header
    case sidebar
    case kpis
    case invoices
    case chart
    case footer
    case amountOwedOverlay
    case ambientDecorations
}
