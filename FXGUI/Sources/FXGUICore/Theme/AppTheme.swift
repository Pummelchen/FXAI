import Foundation
import SwiftUI

public protocol AppTheme {
    var themeID: ThemeID { get }
    var id: String { get }
    var displayName: String { get }
    var colors: ThemeColors { get }
    var gradients: ThemeGradients { get }
    var shadows: ThemeShadows { get }
    var glows: ThemeGlows { get }
    var cornerRadii: ThemeCornerRadii { get }
    var typography: ThemeTypography { get }
    var spacing: ThemeSpacing { get }
    var layoutMetrics: ThemeLayoutMetrics { get }
    var materials: ThemeMaterials { get }
    var chartStyle: ThemeChartStyle { get }
    var components: ThemeComponentStyles { get }
    var renderingPolicy: ThemeRenderingPolicy { get }
}

public extension AppTheme {
    var id: String { themeID.rawValue }
}
