import SwiftUI

public struct FinancialDashboardThemeV1: AppTheme {
    public let themeID: ThemeID = .financialDashboardV1
    public let displayName = "FXAI Operator Theme"
    public let colors: ThemeColors
    public let gradients: ThemeGradients
    public let shadows: ThemeShadows
    public let glows: ThemeGlows
    public let cornerRadii: ThemeCornerRadii
    public let typography: ThemeTypography
    public let spacing: ThemeSpacing
    public let layoutMetrics: ThemeLayoutMetrics
    public let materials: ThemeMaterials
    public let chartStyle: ThemeChartStyle
    public let components: ThemeComponentStyles
    public let renderingPolicy: ThemeRenderingPolicy

    public init() {
        let colors = ThemeColors(
            outerBackground: Color(hex: 0x000000),
            outerVignette: Color(hex: 0x000000),
            mainPanel: Color(hex: 0x2A2A2A),
            sidebar: Color(hex: 0x3C3C3C),
            footer: Color(hex: 0x464646),
            divider: Color(hex: 0x646464),
            cardBase: Color(hex: 0x383838),
            cardHighlight: Color(hex: 0x2D2D2D),
            cardStroke: Color.white.opacity(0.04),
            glassFillStart: Color(hex: 0x444444),
            glassFillEnd: Color(hex: 0x0E0E0E),
            glassBorderStart: Color.white,
            glassBorderEnd: Color(hex: 0x414141),
            textPrimary: Color.white.opacity(0.96),
            textSecondary: Color.white.opacity(0.82),
            textMuted: Color.white.opacity(0.58),
            successGreen: Color(hex: 0x98CB43),
            warningGreen: Color(hex: 0xDADC37),
            chartOlive: Color(hex: 0xB8BA23),
            chartGreen: Color(hex: 0xA2C540),
            chartForest: Color(hex: 0x849F39),
            chartLime: Color(hex: 0xA2C638),
            tooltipBackground: .white,
            tooltipText: .black,
            debugOutline: Color(hex: 0x7EE2FF),
            debugGuide: Color(hex: 0xF9FF64)
        )
        self.colors = colors

        gradients = ThemeGradients(
            canvas: LinearGradient(
                colors: [colors.outerBackground, colors.outerBackground, colors.outerVignette],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            standardCard: LinearGradient(
                colors: [Color(hex: 0x383838), Color(hex: 0x2D2D2D)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            pendingCard: LinearGradient(
                stops: [
                    .init(color: Color(hex: 0x313131, opacity: 0.92), location: 0),
                    .init(color: Color(hex: 0x313131, opacity: 0.27), location: 0.222794),
                    .init(color: Color(hex: 0xA7A948, opacity: 0.78), location: 0.505649),
                    .init(color: Color(hex: 0x8B9832, opacity: 0.9), location: 1.0)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            glassCard: LinearGradient(
                colors: [colors.glassFillStart.opacity(0.74), colors.glassFillEnd.opacity(0.74)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            glassBorder: LinearGradient(
                colors: [colors.glassBorderStart.opacity(0.72), colors.glassBorderEnd.opacity(0.72)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            gaugeLeft: LinearGradient(
                colors: [Color(hex: 0x98CB43), Color(hex: 0xA7E03E)],
                startPoint: .leading,
                endPoint: .trailing
            ),
            gaugeRight: LinearGradient(
                colors: [Color(hex: 0xDADC37), Color(hex: 0xE8E92B)],
                startPoint: .leading,
                endPoint: .trailing
            ),
            footer: LinearGradient(
                colors: [colors.footer.opacity(0.95), colors.footer.opacity(0.78)],
                startPoint: .top,
                endPoint: .bottom
            ),
            chartTooltip: LinearGradient(
                colors: [Color.white, Color(hex: 0xF3F3F3)],
                startPoint: .top,
                endPoint: .bottom
            )
        )

        shadows = ThemeShadows(
            lightSource: ShadowLightSource(
                normalizedPosition: CGPoint(x: 0.92, y: 0.08),
                referenceDirection: CGVector(dx: -0.88, dy: 0.47),
                lateralResponse: 0.68,
                verticalResponse: 0.58,
                poolingResponse: 1.18
            ),
            kpiCard: ShadowStack([
                ShadowLayer(radius: 24, x: -25, y: 16, opacity: 0.28),
                ShadowLayer(radius: 34, x: -35, y: 24, opacity: 0.25),
                ShadowLayer(radius: 48, x: -26, y: 38, opacity: 0.22),
                ShadowLayer(radius: 74, x: -34, y: 63, opacity: 0.18, lightInfluence: 0.94, lateralResponse: 0.54, verticalResponse: 0.52, poolingBias: 0.28)
            ]),
            pendingCard: ShadowStack([
                ShadowLayer(radius: 28, x: -18, y: 14, opacity: 0.32),
                ShadowLayer(radius: 46, x: -28, y: 32, opacity: 0.26),
                ShadowLayer(radius: 82, x: 10, y: 54, opacity: 0.18, lightInfluence: 0.72, lateralResponse: 0.28, verticalResponse: 0.46, poolingBias: 0.24)
            ]),
            smallCard: ShadowStack([
                ShadowLayer(radius: 22, x: -20, y: 12, opacity: 0.22, poolingBias: 0.18),
                ShadowLayer(radius: 32, x: -16, y: 24, opacity: 0.18, poolingBias: 0.18),
                ShadowLayer(radius: 46, x: -8, y: 34, opacity: 0.16, lightInfluence: 0.80, lateralResponse: 0.44, verticalResponse: 0.42, poolingBias: 0.18),
                ShadowLayer(radius: 56, x: 0, y: 46, opacity: 0.12, lightInfluence: 0.76, lateralResponse: 0.0, verticalResponse: 0.44, poolingBias: 0.22)
            ]),
            gaugeCard: ShadowStack([
                ShadowLayer(radius: 28, x: -24, y: 22, opacity: 0.26, poolingBias: 0.22),
                ShadowLayer(radius: 40, x: -22, y: 36, opacity: 0.22, poolingBias: 0.24),
                ShadowLayer(radius: 62, x: -18, y: 52, opacity: 0.18, lightInfluence: 0.92, lateralResponse: 0.56, verticalResponse: 0.56, poolingBias: 0.28),
                ShadowLayer(radius: 88, x: -14, y: 78, opacity: 0.14, lightInfluence: 0.96, lateralResponse: 0.42, verticalResponse: 0.60, poolingBias: 0.32)
            ]),
            amountOwed: ShadowStack([
                ShadowLayer(radius: 32, x: -22, y: 20, opacity: 0.26, poolingBias: 0.24),
                ShadowLayer(radius: 54, x: -28, y: 36, opacity: 0.22, poolingBias: 0.26),
                ShadowLayer(radius: 88, x: -8, y: 68, opacity: 0.18, lightInfluence: 0.94, lateralResponse: 0.38, verticalResponse: 0.58, poolingBias: 0.34)
            ]),
            chartBarPrimary: ShadowStack([
                ShadowLayer(radius: 20, x: 0, y: 10, opacity: 0.34, lightInfluence: 0.44, lateralResponse: 0.0, verticalResponse: 0.82, poolingBias: 0.12),
                ShadowLayer(radius: 32, x: 0, y: 18, opacity: 0.24, lightInfluence: 0.48, lateralResponse: 0.0, verticalResponse: 0.84, poolingBias: 0.14)
            ]),
            chartBarDefault: ShadowStack([
                ShadowLayer(radius: 12, x: 0, y: 6, opacity: 0.16, lightInfluence: 0.30, lateralResponse: 0.0, verticalResponse: 0.78, poolingBias: 0.10)
            ]),
            footer: ShadowStack([
                ShadowLayer(radius: 26, x: 0, y: 6, opacity: 0.14, lightInfluence: 0.18, lateralResponse: 0.0, verticalResponse: 0.52, poolingBias: 0.08)
            ])
        )

        glows = ThemeGlows(
            pendingCard: GlowStack([
                GlowOrb(color: Color(hex: 0xCDFF64), size: CGSize(width: 360, height: 280), offset: CGSize(width: 36, height: 86), blur: 52, opacity: 0.42),
                GlowOrb(color: Color(hex: 0xF6FF64), size: CGSize(width: 420, height: 300), offset: CGSize(width: 70, height: 100), blur: 74, opacity: 0.28),
                GlowOrb(color: Color(hex: 0x6A9C00), size: CGSize(width: 260, height: 220), offset: CGSize(width: 20, height: 34), blur: 64, opacity: 0.12)
            ]),
            amountOwed: GlowStack([
                GlowOrb(color: Color(hex: 0xCDFF64), size: CGSize(width: 320, height: 260), offset: CGSize(width: 88, height: 74), blur: 56, opacity: 0.40),
                GlowOrb(color: Color(hex: 0xDADC37), size: CGSize(width: 370, height: 270), offset: CGSize(width: 120, height: 82), blur: 78, opacity: 0.26),
                GlowOrb(color: Color(hex: 0x6A9C00), size: CGSize(width: 260, height: 210), offset: CGSize(width: 70, height: 92), blur: 64, opacity: 0.20)
            ]),
            ambientStage: GlowStack([
                GlowOrb(color: Color(hex: 0x4E4E4E), size: CGSize(width: 660, height: 540), offset: CGSize(width: -240, height: -260), blur: 110, opacity: 0.08),
                GlowOrb(color: Color(hex: 0xB9B921), size: CGSize(width: 520, height: 420), offset: CGSize(width: 320, height: 160), blur: 130, opacity: 0.10)
            ])
        )

        cornerRadii = ThemeCornerRadii(
            panel: 20,
            standardCard: 19.0275,
            smallCard: 20.2326,
            glassCard: 27.055,
            bar: 7.01158,
            tooltip: 5.91837
        )

        typography = ThemeTypography(
            headerTitle: ThemeFontToken(size: 31, weight: .semibold, tracking: -0.8, opacity: 1, lineSpacing: 0),
            headerSubtitle: ThemeFontToken(size: 14.5, weight: .medium, tracking: -0.1, opacity: 0.82),
            sectionTitle: ThemeFontToken(size: 24, weight: .medium, tracking: -0.3, opacity: 0.96),
            cardTitle: ThemeFontToken(size: 15, weight: .medium, tracking: -0.15, opacity: 0.9),
            kpiValueMajor: ThemeFontToken(size: 22, weight: .semibold, tracking: -0.45, opacity: 1),
            kpiValueMinor: ThemeFontToken(size: 22, weight: .regular, tracking: -0.45, opacity: 0.82),
            bodyValue: ThemeFontToken(size: 17, weight: .semibold, tracking: -0.3, opacity: 0.92),
            caption: ThemeFontToken(size: 12.2, weight: .medium, tracking: -0.08, opacity: 0.7, lineSpacing: 1.5),
            ringValue: ThemeFontToken(size: 12, weight: .semibold, tracking: -0.15, opacity: 0.95),
            footer: ThemeFontToken(size: 13.4, weight: .regular, tracking: -0.1, opacity: 0.68),
            tooltip: ThemeFontToken(size: 9.8, weight: .semibold, tracking: -0.08, opacity: 1)
        )

        spacing = ThemeSpacing(
            shellInset: 26,
            cardPadding: 22,
            compactCardPadding: 18,
            stackGap: 24,
            compactVerticalGap: 26,
            footerHorizontalInset: 32,
            ultraWideMaxContentWidth: 5400
        )

        layoutMetrics = ThemeLayoutMetrics(
            referenceCanvasSize: SVGMetrics.canvasSize,
            minimumScale: 0.64,
            maximumScale: 3.4,
            outerPadding: 26,
            wideMaxWidth: 5400,
            compactCardGap: 24,
            standardCardPadding: 22,
            chartLabelOffset: 20,
            footerHeight: 85,
            decorativeFadeThreshold: 0.86,
            ultrawideBreathingRoom: 180
        )

        materials = ThemeMaterials(
            glassOpacity: 0.74,
            glassStrokeOpacity: 0.72,
            glassBlurRadius: 20,
            footerOpacity: 0.96,
            dividerOpacity: 0.9,
            metalIntensity: 0.44,
            reducedEffectsMetalIntensity: 0.18
        )

        chartStyle = ThemeChartStyle(
            monthLabelOpacity: 0.70,
            barShadowBoostMonth: .may,
            tooltipPointerHeight: 8,
            tooltipPointerWidth: 10,
            baselineOpacity: 0.0
        )

        components = ThemeComponentStyles(
            stage: DashboardStageStyle(
                panelCornerRadius: cornerRadii.panel,
                canvasGradient: gradients.canvas,
                outerGlowOpacity: 0.56
            ),
            sidebar: SidebarComponentStyle(
                iconOpacity: 0.62,
                activeIconOpacity: 0.96,
                iconSize: 21
            ),
            header: HeaderComponentStyle(
                titleFont: typography.headerTitle,
                subtitleFont: typography.headerSubtitle,
                titleSpacing: 8
            ),
            divider: DividerComponentStyle(
                color: colors.divider,
                lineWidth: 2
            ),
            kpiCard: KPICardComponentStyle(
                gradient: gradients.standardCard,
                pendingGradient: gradients.pendingCard,
                shadow: shadows.kpiCard,
                pendingShadow: shadows.pendingCard,
                ringSize: 54
            ),
            gaugeCard: GaugeCardComponentStyle(
                backgroundGradient: gradients.standardCard,
                shadow: shadows.gaugeCard
            ),
            invoiceMetricCard: InvoiceMetricCardComponentStyle(
                backgroundGradient: gradients.standardCard,
                shadow: shadows.smallCard
            ),
            amountOwed: AmountOwedOverlayStyle(
                shadow: shadows.amountOwed,
                glow: glows.amountOwed,
                glassOpacity: materials.glassOpacity
            ),
            trendRing: TrendRingComponentStyle(
                lineWidth: 5.4,
                endpointSize: 6
            ),
            tooltip: TooltipComponentStyle(
                background: gradients.chartTooltip,
                pointerWidth: chartStyle.tooltipPointerWidth,
                pointerHeight: chartStyle.tooltipPointerHeight
            ),
            barChart: BarChartComponentStyle(
                monthOpacity: chartStyle.monthLabelOpacity,
                primaryShadow: shadows.chartBarPrimary,
                defaultShadow: shadows.chartBarDefault
            ),
            footerStrip: FooterStripComponentStyle(
                gradient: gradients.footer,
                shadow: shadows.footer
            ),
            iconBadge: IconBadgeComponentStyle(
                foreground: colors.textPrimary,
                opacity: 0.88
            )
        )

        renderingPolicy = ThemeRenderingPolicy(
            policies: [
                .stage: ComponentRenderingPolicy(
                    component: .stage,
                    preferredTier: .swiftUI,
                    fallbackTier: .coreGraphics,
                    capabilities: [.realtimeResizing]
                ),
                .sidebar: ComponentRenderingPolicy(
                    component: .sidebar,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.vectorOverlay, .realtimeResizing]
                ),
                .header: ComponentRenderingPolicy(
                    component: .header,
                    preferredTier: .swiftUI,
                    fallbackTier: .swiftUI,
                    capabilities: [.realtimeResizing]
                ),
                .kpiCard: ComponentRenderingPolicy(
                    component: .kpiCard,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.shadowStacks, .bloomGlow, .realtimeResizing]
                ),
                .invoiceMetricCard: ComponentRenderingPolicy(
                    component: .invoiceMetricCard,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.shadowStacks, .realtimeResizing]
                ),
                .gaugeCard: ComponentRenderingPolicy(
                    component: .gaugeCard,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.gaugePrecision, .shadowStacks, .realtimeResizing]
                ),
                .amountOwedGlassCard: ComponentRenderingPolicy(
                    component: .amountOwedGlassCard,
                    preferredTier: .coreAnimation,
                    fallbackTier: .coreGraphics,
                    capabilities: [.glassComposite, .bloomGlow, .shadowStacks, .realtimeResizing]
                ),
                .trendRing: ComponentRenderingPolicy(
                    component: .trendRing,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.chartPrecision, .realtimeResizing]
                ),
                .barChart: ComponentRenderingPolicy(
                    component: .barChart,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.chartPrecision, .shadowStacks, .realtimeResizing]
                ),
                .footerStrip: ComponentRenderingPolicy(
                    component: .footerStrip,
                    preferredTier: .swiftUI,
                    fallbackTier: .coreGraphics,
                    capabilities: [.shadowStacks]
                ),
                .tooltip: ComponentRenderingPolicy(
                    component: .tooltip,
                    preferredTier: .coreGraphics,
                    fallbackTier: .swiftUI,
                    capabilities: [.chartPrecision]
                ),
                .calibrationOverlay: ComponentRenderingPolicy(
                    component: .calibrationOverlay,
                    preferredTier: .metal,
                    fallbackTier: .coreAnimation,
                    capabilities: [.vectorOverlay, .metalCompositing, .realtimeResizing]
                )
            ],
            compactGlowReductionThreshold: 0.86,
            chartMinimumReadableHeight: 240,
            chartMinimumReadableWidth: 420
        )
    }
}
