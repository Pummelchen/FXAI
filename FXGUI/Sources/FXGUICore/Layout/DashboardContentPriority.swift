import Foundation

public enum DashboardContentPriority: Int, Comparable, Sendable {
    case decorative = 0
    case medium = 1
    case high = 2
    case critical = 3

    public static func < (lhs: DashboardContentPriority, rhs: DashboardContentPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
