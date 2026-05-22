import FXAIGUICore
import SwiftUI

struct SectionHeader: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 24, weight: .semibold, design: .rounded))
                .foregroundStyle(FXAITheme.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)

            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(FXAITheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
