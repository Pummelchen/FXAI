import CoreGraphics
import SwiftUI

public struct DashboardStageStyle {
    public let panelCornerRadius: CGFloat
    public let canvasGradient: LinearGradient
    public let outerGlowOpacity: Double
}

public struct SidebarComponentStyle {
    public let iconOpacity: Double
    public let activeIconOpacity: Double
    public let iconSize: CGFloat
}

public struct HeaderComponentStyle {
    public let titleFont: ThemeFontToken
    public let subtitleFont: ThemeFontToken
    public let titleSpacing: CGFloat
}

public struct DividerComponentStyle {
    public let color: Color
    public let lineWidth: CGFloat
}

public struct KPICardComponentStyle {
    public let gradient: LinearGradient
    public let pendingGradient: LinearGradient
    public let shadow: ShadowStack
    public let pendingShadow: ShadowStack
    public let ringSize: CGFloat
}

public struct GaugeCardComponentStyle {
    public let backgroundGradient: LinearGradient
    public let shadow: ShadowStack
}

public struct InvoiceMetricCardComponentStyle {
    public let backgroundGradient: LinearGradient
    public let shadow: ShadowStack
}

public struct AmountOwedOverlayStyle {
    public let shadow: ShadowStack
    public let glow: GlowStack
    public let glassOpacity: Double
}

public struct TrendRingComponentStyle {
    public let lineWidth: CGFloat
    public let endpointSize: CGFloat
}

public struct TooltipComponentStyle {
    public let background: LinearGradient
    public let pointerWidth: CGFloat
    public let pointerHeight: CGFloat
}

public struct BarChartComponentStyle {
    public let monthOpacity: Double
    public let primaryShadow: ShadowStack
    public let defaultShadow: ShadowStack
}

public struct FooterStripComponentStyle {
    public let gradient: LinearGradient
    public let shadow: ShadowStack
}

public struct IconBadgeComponentStyle {
    public let foreground: Color
    public let opacity: Double
}

public struct ThemeComponentStyles {
    public let stage: DashboardStageStyle
    public let sidebar: SidebarComponentStyle
    public let header: HeaderComponentStyle
    public let divider: DividerComponentStyle
    public let kpiCard: KPICardComponentStyle
    public let gaugeCard: GaugeCardComponentStyle
    public let invoiceMetricCard: InvoiceMetricCardComponentStyle
    public let amountOwed: AmountOwedOverlayStyle
    public let trendRing: TrendRingComponentStyle
    public let tooltip: TooltipComponentStyle
    public let barChart: BarChartComponentStyle
    public let footerStrip: FooterStripComponentStyle
    public let iconBadge: IconBadgeComponentStyle
}
