import FXAIGUICore
import SwiftUI

struct FXAIBackgroundView: View {
    var body: some View {
        ZStack {
            FXAITheme.canvasGradient
                .ignoresSafeArea()

            Circle()
                .fill(FXAITheme.accent.opacity(0.18))
                .frame(width: 440, height: 440)
                .blur(radius: 90)
                .offset(x: -380, y: -280)

            Circle()
                .fill(FXAITheme.accentSoft.opacity(0.16))
                .frame(width: 520, height: 520)
                .blur(radius: 120)
                .offset(x: 420, y: -240)

            Rectangle()
                .fill(.black.opacity(0.12))
                .ignoresSafeArea()
        }
    }
}
