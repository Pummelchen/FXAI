import FXAIGUICore
import SwiftUI

struct EmptyStateView: View {
    let title: String
    let message: String
    let symbolName: String

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(spacing: 14) {
                Image(systemName: symbolName)
                    .font(.system(size: 38, weight: .medium))
                    .foregroundStyle(FXAITheme.accent)

                Text(title)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(FXAITheme.textPrimary)

                Text(message)
                    .font(.body)
                    .foregroundStyle(FXAITheme.textSecondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, minHeight: 220)
        }
    }
}
