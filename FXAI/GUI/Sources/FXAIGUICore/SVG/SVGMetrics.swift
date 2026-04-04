import CoreGraphics
import SwiftUI

public enum SVGKPIKind: String, CaseIterable, Hashable, Sendable {
    case readyToAssign
    case pendingSignOffs
    case declined
    case occured
}

public enum SVGInvoiceCardKind: String, CaseIterable, Hashable, Sendable {
    case paidInvoices
    case liveFundUpdate
}

public enum SVGChartMonth: String, CaseIterable, Hashable, Sendable {
    case feb = "Feb"
    case mar = "Mar"
    case april = "April"
    case may = "May"
    case june = "June"
    case july = "July"
    case aug = "Aug"
}

public struct DashboardKPIContent: Hashable, Sendable {
    public let title: String
    public let majorValue: String
    public let minorValue: String
    public let subtitle: String
    public let delta: String
    public let iconSystemName: String
    public let ringProgress: CGFloat
    public let highlightStyle: KPIHighlightStyle

    public enum KPIHighlightStyle: Hashable, Sendable {
        case neutral
        case pending
    }
}

public struct InvoiceMetricContent: Hashable, Sendable {
    public let title: String
    public let value: String
    public let subtitle: String
    public let ringText: String?
    public let ringProgress: CGFloat?
    public let ringTintHex: UInt32?
}

public enum SVGMetrics {
    public static let canvasSize = CGSize(width: 1728, height: 1117)
    public static let mainPanelFrame = CGRect(x: 80, y: 0, width: 1648, height: 1117)
    public static let footerFrame = CGRect(x: 477, y: 975, width: 1251, height: 85)
    public static let headerDivider = (x: CGFloat(80), y: CGFloat(141), width: CGFloat(1648), height: CGFloat(2))

    public static let headerTitleOrigin = CGPoint(x: 126, y: 49)
    public static let headerSubtitleOrigin = CGPoint(x: 128, y: 94)
    public static let billsOrigin = CGPoint(x: 126, y: 197)
    public static let invoicesOrigin = CGPoint(x: 126, y: 512)

    public static let sidebarPathPoints: [CGPoint] = [
        CGPoint(x: 0.0004, y: 267.189),
        CGPoint(x: 0.0004, y: 846.972),
        CGPoint(x: 8.2136, y: 869.002),
        CGPoint(x: 79.4042, y: 895.689),
        CGPoint(x: 98.5, y: 863.114),
        CGPoint(x: 98.5, y: 252.514),
        CGPoint(x: 79.8931, y: 220.38),
        CGPoint(x: 8.6127, y: 244.677)
    ]

    public static let topCardFrames: [SVGKPIKind: CGRect] = [
        .readyToAssign: CGRect(x: 186, y: 229, width: 300, height: 212),
        .pendingSignOffs: CGRect(x: 572, y: 229, width: 300, height: 212),
        .declined: CGRect(x: 922, y: 229, width: 300, height: 212),
        .occured: CGRect(x: 1290, y: 229, width: 300, height: 212)
    ]

    public static let gaugeFrame = CGRect(x: 186, y: 604, width: 300, height: 457)
    public static let invoiceMetricFrames: [SVGInvoiceCardKind: CGRect] = [
        .paidInvoices: CGRect(x: 572, y: 548, width: 253, height: 156),
        .liveFundUpdate: CGRect(x: 572, y: 763, width: 253, height: 156)
    ]
    public static let amountOwedFrame = CGRect(x: 129, y: 828.745, width: 333, height: 189.776)
    public static let chartPlotFrame = CGRect(x: 958, y: 524, width: 571, height: 330)
    public static let tooltipFrame = CGRect(x: 1424, y: 534, width: 92.3265, height: 24.2653)

    public static let chartBars: [SVGChartMonth: CGRect] = [
        .feb: CGRect(x: 1011, y: 674, width: 44, height: 149.6),
        .mar: CGRect(x: 1083.8, y: 634.8, width: 44, height: 188.8),
        .april: CGRect(x: 1156.6, y: 654.8, width: 44, height: 168.8),
        .may: CGRect(x: 1229.4, y: 586, width: 44, height: 237.6),
        .june: CGRect(x: 1302.2, y: 609.2, width: 44, height: 214.4),
        .july: CGRect(x: 1375, y: 609.2, width: 44, height: 214.4),
        .aug: CGRect(x: 1448, y: 586, width: 44, height: 238)
    ]

    public static let footerDate = "26-Dec-2021"
    public static let footerTime = "19:22"
    public static let footerDay = "Sunday"
    public static let tooltipText = "Aug 2021"

    public static let kpiContent: [SVGKPIKind: DashboardKPIContent] = [
        .readyToAssign: DashboardKPIContent(
            title: "Ready to assign",
            majorValue: "200",
            minorValue: "-42",
            subtitle: "Bills in this week",
            delta: "+42%",
            iconSystemName: "calendar.badge.clock",
            ringProgress: 0.42,
            highlightStyle: .neutral
        ),
        .pendingSignOffs: DashboardKPIContent(
            title: "Pending SIgn Offs",
            majorValue: "63",
            minorValue: "-22",
            subtitle: "Signed Off in this week",
            delta: "+22%",
            iconSystemName: "doc.text",
            ringProgress: 0.22,
            highlightStyle: .pending
        ),
        .declined: DashboardKPIContent(
            title: "Declined",
            majorValue: "5",
            minorValue: "",
            subtitle: "Declined this week 2",
            delta: "-5%",
            iconSystemName: "plus.forwardslash.minus",
            ringProgress: 0.05,
            highlightStyle: .neutral
        ),
        .occured: DashboardKPIContent(
            title: "Occured",
            majorValue: "7",
            minorValue: "",
            subtitle: "Occured this week 4",
            delta: "+5%",
            iconSystemName: "calendar.badge.plus",
            ringProgress: 0.05,
            highlightStyle: .neutral
        )
    ]

    public static let invoiceCards: [SVGInvoiceCardKind: InvoiceMetricContent] = [
        .paidInvoices: InvoiceMetricContent(
            title: "Paid Invoices",
            value: "₹ 78,921",
            subtitle: "Current Financial Year",
            ringText: "+5%",
            ringProgress: 0.05,
            ringTintHex: 0xFFFFFF
        ),
        .liveFundUpdate: InvoiceMetricContent(
            title: "Live Fund Update",
            value: "₹ 16,85,651",
            subtitle: "Current Financial Year",
            ringText: "82%",
            ringProgress: 0.82,
            ringTintHex: 0x5C9800
        )
    ]

    public static func scaledRect(_ rect: CGRect, scale: CGFloat, origin: CGPoint = .zero) -> CGRect {
        CGRect(
            x: origin.x + rect.origin.x * scale,
            y: origin.y + rect.origin.y * scale,
            width: rect.size.width * scale,
            height: rect.size.height * scale
        )
    }
}
