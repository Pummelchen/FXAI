import AppKit
import FXAIGUICore
import SwiftUI

struct DashboardRootView: View {
    @EnvironmentObject private var model: FXAIGUIModel
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment
    @State private var debugState = DashboardDebugState()

    var body: some View {
        GeometryReader { geometry in
            let theme = themeEnvironment.currentTheme
            let layoutOutput = DashboardLayoutEngine.makeLayout(
                input: DashboardLayoutInput(
                    windowSize: geometry.size,
                    effectiveContentSize: geometry.size,
                    backingScaleFactor: NSScreen.main?.backingScaleFactor ?? 2,
                    theme: theme,
                    contentPriorities: DashboardAdaptiveRules.defaultContentPriorities,
                    overlapPolicy: .preserveFloatingCard,
                    scalePolicy: .themeDefault(for: theme),
                    reducedEffects: debugState.reducedEffects
                )
            )
            let frameModel = layoutOutput.frameModel

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

                dashboardStage(layoutOutput: layoutOutput, theme: theme)
                    .frame(width: frameModel.stageFrame.width, height: frameModel.stageFrame.height)
                    .position(x: frameModel.stageFrame.midX, y: frameModel.stageFrame.midY)

                CalibrationPanel(
                    debugState: $debugState,
                    layoutOutput: layoutOutput,
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
    private func dashboardStage(layoutOutput: DashboardLayoutOutput, theme: any AppTheme) -> some View {
        let frameModel = layoutOutput.frameModel

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
                        stageSize: frameModel.stageFrame.size,
                        content: content,
                        theme: theme,
                        scale: frameModel.scale,
                        showShadowDebug: debugState.showShadowGlowDebug,
                        decorativeIntensity: frameModel.decorativeVisibility.glowIntensity
                    )
                }
            }

            GaugeCardView(frame: frameModel.gaugeFrame, stageSize: frameModel.stageFrame.size, theme: theme, scale: frameModel.scale)

            ForEach(SVGInvoiceCardKind.allCases, id: \.self) { kind in
                if let frame = frameModel.invoiceMetricFrames[kind], let content = SVGMetrics.invoiceCards[kind] {
                    InvoiceMetricCardView(
                        kind: kind,
                        frame: frame,
                        stageSize: frameModel.stageFrame.size,
                        content: content,
                        theme: theme,
                        scale: frameModel.scale
                    )
                }
            }

            AmountOwedGlassCardView(
                frame: frameModel.amountOwedFrame,
                stageSize: frameModel.stageFrame.size,
                theme: theme,
                scale: frameModel.scale,
                intensity: frameModel.decorativeVisibility.glowIntensity,
                reducedEffects: debugState.reducedEffects
            )

            BarChartView(frameModel: frameModel, theme: theme)
            FooterStripView(frameModel: frameModel, theme: theme)

            CalibrationView(frameModel: frameModel, layoutOutput: layoutOutput, theme: theme, debugState: debugState)
        }
    }
}
