import SwiftUI

public struct SidebarSlabShape: Shape {
    public init() {}

    public func path(in rect: CGRect) -> Path {
        let referenceRect = CGRect(x: 0, y: 220.38, width: 98.5, height: 675.309)
        let scaleX = rect.width / referenceRect.width
        let scaleY = rect.height / referenceRect.height
        let points = SVGMetrics.sidebarPathPoints.map { point in
            CGPoint(
                x: (point.x - referenceRect.minX) * scaleX,
                y: (point.y - referenceRect.minY) * scaleY
            )
        }

        var path = Path()
        guard let first = points.first else { return path }
        path.move(to: first)
        for point in points.dropFirst() {
            path.addLine(to: point)
        }
        path.closeSubpath()
        return path
    }
}

public struct DashboardRenderer: View {
    public let frameModel: DashboardFrameModel
    public let theme: any AppTheme
    public let reducedEffects: Bool

    public init(frameModel: DashboardFrameModel, theme: any AppTheme, reducedEffects: Bool) {
        self.frameModel = frameModel
        self.theme = theme
        self.reducedEffects = reducedEffects
    }

    public var body: some View {
        let stageStyle = theme.components.stage
        let footerStyle = theme.components.footerStrip
        let dividerStyle = theme.components.divider
        let panel = frameModel.mainPanelFrame
        let scale = frameModel.scale

        ZStack(alignment: .topLeading) {
            GlowRenderer(
                glow: theme.glows.ambientStage,
                scale: scale,
                intensity: frameModel.decorativeVisibility.ambientOpacity
            )
            .opacity(reducedEffects ? 0.34 : 1)
            .offset(x: panel.minX + panel.width * 0.36, y: panel.height * 0.38)

            RoundedRectangle(cornerRadius: stageStyle.panelCornerRadius * scale, style: .continuous)
                .fill(theme.colors.mainPanel)
                .frame(width: panel.width, height: panel.height)
                .offset(x: panel.minX, y: panel.minY)

            SidebarSlabShape()
                .fill(theme.colors.sidebar)
                .frame(width: 98.5 * scale, height: 675.309 * scale)
                .offset(x: 0, y: 220.38 * scale)

            Rectangle()
                .fill(footerStyle.gradient)
                .frame(width: frameModel.footerFrame.width, height: frameModel.footerFrame.height)
                .offset(x: frameModel.footerFrame.minX, y: frameModel.footerFrame.minY)

            Rectangle()
                .fill(dividerStyle.color.opacity(theme.materials.dividerOpacity))
                .frame(width: frameModel.headerDividerFrame.width, height: dividerStyle.lineWidth * scale)
                .offset(x: frameModel.headerDividerFrame.minX, y: frameModel.headerDividerFrame.minY)
        }
        .frame(width: frameModel.stageFrame.width, height: frameModel.stageFrame.height)
        .allowsHitTesting(false)
    }
}
