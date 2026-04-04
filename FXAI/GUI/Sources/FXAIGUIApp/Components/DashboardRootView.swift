import FXAIGUICore
import SwiftUI

struct DashboardRootView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @ObservedObject private var themeManager = ThemeManager.shared
    @State private var debugState = DashboardDebugState()

    var body: some View {
        GeometryReader { geometry in
            let theme = themeManager.currentTheme
            let frameModel = DashboardLayoutEngine.makeFrameModel(containerSize: geometry.size, theme: theme)

            ZStack(alignment: .topLeading) {
                theme.gradients.canvas
                    .ignoresSafeArea()

                RadialGradient(
                    colors: [
                        theme.colors.outerVignette.opacity(0.4),
                        theme.colors.outerBackground.opacity(0)
                    ],
                    center: .center,
                    startRadius: 60,
                    endRadius: max(geometry.size.width, geometry.size.height)
                )
                .ignoresSafeArea()

                dashboardStage(frameModel: frameModel, theme: theme)
                    .frame(width: frameModel.stageFrame.width, height: frameModel.stageFrame.height)
                    .position(x: frameModel.stageFrame.midX, y: frameModel.stageFrame.midY)

                DebugOverlayControls(
                    debugState: $debugState,
                    frameModel: frameModel,
                    theme: theme,
                    connectionStatus: model.connectionStatusLabel
                )
                .padding(.top, max(22, geometry.safeAreaInsets.top + 12))
                .padding(.trailing, 22)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            }
        }
    }

    @ViewBuilder
    private func dashboardStage(frameModel: DashboardFrameModel, theme: any AppTheme) -> some View {
        ZStack(alignment: .topLeading) {
            DashboardRenderer(frameModel: frameModel, theme: theme, reducedEffects: debugState.reducedEffects)

            if frameModel.decorativeVisibility.metalOverlayEnabled {
                MetalEffectRenderer(
                    intensity: debugState.reducedEffects
                        ? theme.materials.reducedEffectsMetalIntensity
                        : theme.materials.metalIntensity
                )
                .frame(width: frameModel.mainPanelFrame.width, height: frameModel.mainPanelFrame.height)
                .clipShape(
                    RoundedRectangle(cornerRadius: theme.cornerRadii.panel * frameModel.scale, style: .continuous)
                )
                .offset(x: frameModel.mainPanelFrame.minX, y: frameModel.mainPanelFrame.minY)
                .blendMode(.screen)
                .opacity(debugState.showShadowGlowDebug ? 0.88 : 0.56)
            }

            HeaderView(frameModel: frameModel, theme: theme)
            SidebarView(frameModel: frameModel, theme: theme)

            ForEach(SVGKPIKind.allCases, id: \.self) { kind in
                if let frame = frameModel.topCardFrames[kind], let content = SVGMetrics.kpiContent[kind] {
                    KPICardView(
                        kind: kind,
                        frame: frame,
                        content: content,
                        theme: theme,
                        scale: frameModel.scale,
                        showShadowDebug: debugState.showShadowGlowDebug,
                        decorativeIntensity: frameModel.decorativeVisibility.glowIntensity
                    )
                }
            }

            GaugeCardView(frame: frameModel.gaugeFrame, theme: theme, scale: frameModel.scale)

            ForEach(SVGInvoiceCardKind.allCases, id: \.self) { kind in
                if let frame = frameModel.invoiceMetricFrames[kind], let content = SVGMetrics.invoiceCards[kind] {
                    InvoiceMetricCardView(kind: kind, frame: frame, content: content, theme: theme, scale: frameModel.scale)
                }
            }

            AmountOwedGlassCardView(
                frame: frameModel.amountOwedFrame,
                theme: theme,
                scale: frameModel.scale,
                intensity: frameModel.decorativeVisibility.glowIntensity,
                reducedEffects: debugState.reducedEffects
            )

            BarChartView(frameModel: frameModel, theme: theme)
            FooterStripView(frameModel: frameModel, theme: theme)

            CalibrationView(frameModel: frameModel, theme: theme, debugState: debugState)
        }
    }
}
