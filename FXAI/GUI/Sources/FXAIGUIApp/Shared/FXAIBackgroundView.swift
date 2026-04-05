import FXAIGUICore
import SwiftUI

struct FXAIBackgroundView: View {
    @Environment(\.guiRenderingProfile) private var renderingProfile

    var body: some View {
        ZStack {
            FXAITheme.canvasGradient
                .ignoresSafeArea()

            if renderingProfile.decorativeOpacityScale > 0.1 {
                Circle()
                    .fill(FXAITheme.accent.opacity(0.18 * renderingProfile.decorativeOpacityScale))
                    .frame(width: 440, height: 440)
                    .blur(radius: 90 * CGFloat(renderingProfile.backgroundBlurScale))
                    .offset(x: -380, y: -280)

                Circle()
                    .fill(FXAITheme.accentSoft.opacity(0.16 * renderingProfile.decorativeOpacityScale))
                    .frame(width: 520, height: 520)
                    .blur(radius: 120 * CGFloat(renderingProfile.backgroundBlurScale))
                    .offset(x: 420, y: -240)
            }

            Rectangle()
                .fill(.black.opacity(0.12))
                .ignoresSafeArea()
        }
    }
}
