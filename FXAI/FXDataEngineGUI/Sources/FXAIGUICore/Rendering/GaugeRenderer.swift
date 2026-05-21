import SwiftUI

private struct GaugeArcShape: Shape {
    let startAngle: Angle
    let endAngle: Angle

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addArc(
            center: CGPoint(x: rect.midX, y: rect.maxY),
            radius: min(rect.width / 2, rect.height),
            startAngle: startAngle,
            endAngle: endAngle,
            clockwise: false
        )
        return path
    }
}

public struct GaugeRenderer: View {
    public let progress: CGFloat
    public let theme: any AppTheme
    public let scale: CGFloat

    public init(progress: CGFloat, theme: any AppTheme, scale: CGFloat) {
        self.progress = progress
        self.theme = theme
        self.scale = scale
    }

    public var body: some View {
        GeometryReader { geometry in
            let lineWidth = 28 * scale
            let rect = CGRect(x: lineWidth * 0.5, y: lineWidth * 0.5, width: geometry.size.width - lineWidth, height: geometry.size.height - lineWidth * 0.9)
            let split = min(max(progress, 0.18), 0.82)
            let splitAngle = Angle(degrees: 180 - Double(split) * 180)

            ZStack {
                GaugeArcShape(startAngle: .degrees(180), endAngle: .degrees(0))
                    .stroke(theme.colors.cardStroke.opacity(0.7), style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))

                GaugeArcShape(startAngle: .degrees(180), endAngle: splitAngle)
                    .stroke(theme.gradients.gaugeLeft, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                    .shadow(color: theme.colors.successGreen.opacity(0.28), radius: 12 * scale, x: 0, y: 8 * scale)

                GaugeArcShape(startAngle: splitAngle, endAngle: .degrees(0))
                    .stroke(theme.gradients.gaugeRight, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                    .shadow(color: theme.colors.warningGreen.opacity(0.24), radius: 14 * scale, x: 0, y: 8 * scale)

                Circle()
                    .fill(Color.white.opacity(0.94))
                    .frame(width: 2.5 * lineWidth / 2.4, height: 2.5 * lineWidth / 2.4)
                    .offset(x: -geometry.size.width * 0.11, y: -geometry.size.height * 0.06)
                    .rotationEffect(.degrees(-14))

                Text("45%")
                    .font(theme.typography.bodyValue.font(scaledBy: scale))
                    .foregroundStyle(theme.colors.textPrimary)
                    .offset(y: geometry.size.height * 0.20)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .drawingGroup()
            .contentShape(Rectangle())
            .padding(.top, rect.minY)
        }
    }
}
