import Foundation
import SwiftUI

public struct ThemeChartStyle {
    public let monthLabelOpacity: Double
    public let barShadowBoostMonth: SVGChartMonth
    public let tooltipPointerHeight: CGFloat
    public let tooltipPointerWidth: CGFloat
    public let baselineOpacity: Double

    public init(
        monthLabelOpacity: Double,
        barShadowBoostMonth: SVGChartMonth,
        tooltipPointerHeight: CGFloat,
        tooltipPointerWidth: CGFloat,
        baselineOpacity: Double
    ) {
        self.monthLabelOpacity = monthLabelOpacity
        self.barShadowBoostMonth = barShadowBoostMonth
        self.tooltipPointerHeight = tooltipPointerHeight
        self.tooltipPointerWidth = tooltipPointerWidth
        self.baselineOpacity = baselineOpacity
    }
}
