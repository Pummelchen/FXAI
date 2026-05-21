import SwiftUI

public struct ChartBarDescriptor: Hashable, Identifiable {
    public let id = UUID()
    public let month: SVGChartMonth
    public let frame: CGRect
    public let color: Color

    public init(month: SVGChartMonth, frame: CGRect, color: Color) {
        self.month = month
        self.frame = frame
        self.color = color
    }
}

public struct ChartRenderer: View {
    public let frameModel: DashboardFrameModel
    public let theme: any AppTheme

    public init(frameModel: DashboardFrameModel, theme: any AppTheme) {
        self.frameModel = frameModel
        self.theme = theme
    }

    public var body: some View {
        let style = theme.components.barChart

        ZStack(alignment: .topLeading) {
            ForEach(descriptors) { descriptor in
                ZStack {
                    if descriptor.month == theme.chartStyle.barShadowBoostMonth {
                        ShadowStackRenderer(
                            size: descriptor.frame.size,
                            cornerRadius: descriptor.month == .aug ? 11.125 * frameModel.scale : theme.cornerRadii.bar * frameModel.scale,
                            shadow: style.primaryShadow,
                            scale: frameModel.scale,
                            context: ShadowProjectionContext(frame: descriptor.frame, stageSize: frameModel.stageFrame.size, lightSource: theme.shadows.lightSource)
                        )
                    } else {
                        ShadowStackRenderer(
                            size: descriptor.frame.size,
                            cornerRadius: theme.cornerRadii.bar * frameModel.scale,
                            shadow: style.defaultShadow,
                            scale: frameModel.scale,
                            context: ShadowProjectionContext(frame: descriptor.frame, stageSize: frameModel.stageFrame.size, lightSource: theme.shadows.lightSource)
                        )
                    }

                    RoundedRectangle(cornerRadius: descriptor.month == .aug ? 11.125 * frameModel.scale : theme.cornerRadii.bar * frameModel.scale, style: .continuous)
                        .fill(descriptor.color)
                        .frame(width: descriptor.frame.width, height: descriptor.frame.height)
                }
                .position(x: descriptor.frame.midX, y: descriptor.frame.midY)

                Text(descriptor.month.rawValue)
                    .font(theme.typography.caption.font(scaledBy: frameModel.scale))
                    .foregroundStyle(theme.colors.textSecondary.opacity(style.monthOpacity))
                    .position(
                        x: descriptor.frame.midX,
                        y: frameModel.chartPlotFrame.maxY + theme.layoutMetrics.chartLabelOffset * frameModel.scale
                    )
            }

            tooltip
        }
    }

    private var descriptors: [ChartBarDescriptor] {
        [
            .init(month: .feb, frame: frameModel.chartBars[.feb] ?? .zero, color: theme.colors.chartOlive),
            .init(month: .mar, frame: frameModel.chartBars[.mar] ?? .zero, color: theme.colors.chartGreen),
            .init(month: .april, frame: frameModel.chartBars[.april] ?? .zero, color: theme.colors.chartForest),
            .init(month: .may, frame: frameModel.chartBars[.may] ?? .zero, color: Color(hex: 0xB5B41E)),
            .init(month: .june, frame: frameModel.chartBars[.june] ?? .zero, color: theme.colors.chartLime),
            .init(month: .july, frame: frameModel.chartBars[.july] ?? .zero, color: Color(hex: 0xB9B921)),
            .init(month: .aug, frame: frameModel.chartBars[.aug] ?? .zero, color: theme.colors.chartLime)
        ]
    }

    private var tooltip: some View {
        let tooltipFrame = frameModel.tooltipFrame
        let tooltipStyle = theme.components.tooltip
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: theme.cornerRadii.tooltip * frameModel.scale, style: .continuous)
                .fill(tooltipStyle.background)
                .frame(width: tooltipFrame.width, height: tooltipFrame.height)

            Text(SVGMetrics.tooltipText)
                .font(theme.typography.tooltip.font(scaledBy: frameModel.scale))
                .foregroundStyle(theme.colors.tooltipText)
                .position(x: tooltipFrame.midX, y: tooltipFrame.midY - 0.5 * frameModel.scale)

            TriangleShape()
                .fill(theme.colors.tooltipBackground)
                .frame(
                    width: tooltipStyle.pointerWidth * frameModel.scale,
                    height: tooltipStyle.pointerHeight * frameModel.scale
                )
                .position(
                    x: tooltipFrame.midX,
                    y: tooltipFrame.maxY + (tooltipStyle.pointerHeight * frameModel.scale / 2)
                )
        }
    }
}

private struct TriangleShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.closeSubpath()
        return path
    }
}
