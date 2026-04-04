import SwiftUI

private struct RingArcShape: Shape {
    let progress: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let radius = min(rect.width, rect.height) / 2
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let start = Angle(degrees: -90)
        let end = Angle(degrees: -90 + Double(progress) * 360)
        path.addArc(center: center, radius: radius, startAngle: start, endAngle: end, clockwise: false)
        return path
    }
}

public struct TrendRingRenderer: View {
    public let text: String
    public let progress: CGFloat
    public let scale: CGFloat
    public let theme: any AppTheme
    public let tint: Color

    public init(text: String, progress: CGFloat, scale: CGFloat, theme: any AppTheme, tint: Color) {
        self.text = text
        self.progress = progress
        self.scale = scale
        self.theme = theme
        self.tint = tint
    }

    public var body: some View {
        GeometryReader { geometry in
            let lineWidth = 5.4 * scale
            ZStack {
                Circle()
                    .stroke(theme.colors.textMuted.opacity(0.35), lineWidth: lineWidth)

                RingArcShape(progress: progress)
                    .stroke(tint, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))

                Circle()
                    .fill(Color.white.opacity(0.9))
                    .frame(width: 6 * scale, height: 6 * scale)
                    .offset(x: geometry.size.width * 0.32, y: geometry.size.height * 0.22)

                Text(text)
                    .font(theme.typography.ringValue.font(scaledBy: scale))
                    .foregroundStyle(theme.colors.textPrimary.opacity(0.92))
            }
        }
    }
}
