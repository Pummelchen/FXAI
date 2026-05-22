import FXAIGUICore
import SwiftUI

struct MetricCard: View {
    let title: String
    let value: String
    let footnote: String
    let symbolName: String
    let tint: Color

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(title)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(FXAITheme.textSecondary)
                        Text(value)
                            .font(.system(size: 30, weight: .semibold, design: .rounded))
                            .foregroundStyle(FXAITheme.textPrimary)
                    }

                    Spacer(minLength: 12)

                    Image(systemName: symbolName)
                        .font(.title2.weight(.semibold))
                        .foregroundStyle(tint)
                }

                Text(footnote)
                    .font(.footnote)
                    .foregroundStyle(FXAITheme.textMuted)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
