import CoreGraphics

public struct SVGSemanticZoneMapping: Sendable {
    public let zone: DashboardZone
    public let frame: CGRect
    public let description: String
}

public enum SVGSemanticMapping {
    public static let baselineMappings: [SVGSemanticZoneMapping] = [
        .init(zone: .header, frame: CGRect(x: 80, y: 0, width: 1648, height: 180), description: "Header title, subtitle, and divider."),
        .init(zone: .sidebar, frame: CGRect(x: 0, y: 220.38, width: 98.5, height: 675.309), description: "Left navigation slab with muted finance icons."),
        .init(zone: .kpis, frame: CGRect(x: 186, y: 229, width: 1404, height: 212), description: "Top KPI row that reflows under constrained widths."),
        .init(zone: .invoices, frame: CGRect(x: 129, y: 548, width: 696, height: 513), description: "Gauge, invoice metric cards, and floating amount owed overlay."),
        .init(zone: .chart, frame: CGRect(x: 958, y: 524, width: 571, height: 330), description: "Bar chart block with tooltip and month labels."),
        .init(zone: .footer, frame: SVGMetrics.footerFrame, description: "Footer strip with date, time, and day rhythm."),
        .init(zone: .amountOwedOverlay, frame: SVGMetrics.amountOwedFrame, description: "Deliberate overlapping glass card."),
        .init(zone: .ambientDecorations, frame: CGRect(x: 400, y: 180, width: 1100, height: 860), description: "Glow extents and pooled shadows.")
    ]
}
