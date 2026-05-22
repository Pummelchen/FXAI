import FXAIGUICore
import SwiftUI

struct StatusBadge: View {
    let title: String
    let value: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title.uppercased())
                .font(.caption2.weight(.bold))
                .foregroundStyle(FXAITheme.textMuted)
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(
                    FXAIGlassCapsuleBackground(tint: tint)
                )
        }
    }
}
