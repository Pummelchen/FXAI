import CoreGraphics

public enum DashboardLayoutEngine {
    public static func makeLayout(input: DashboardLayoutInput) -> DashboardLayoutOutput {
        let metrics = input.theme.layoutMetrics
        let layoutClass = DashboardAdaptiveRules.classify(containerSize: input.effectiveContentSize)
        let baseScale = fittedScale(
            contentSize: input.effectiveContentSize,
            scalePolicy: input.scalePolicy,
            metrics: metrics
        )
        let typographyScale = DashboardAdaptiveRules.typographyScale(for: baseScale, policy: input.scalePolicy)
        let spacingScale = DashboardAdaptiveRules.spacingScale(for: baseScale, policy: input.scalePolicy)

        let minimumChartHeight = input.theme.renderingPolicy.chartMinimumReadableHeight * spacingScale
        let minimumChartWidth = input.theme.renderingPolicy.chartMinimumReadableWidth * spacingScale
        let shouldCompactChart = layoutClass == .compactDesktop || input.effectiveContentSize.width < minimumChartWidth * 2.18
        let chartPlacement: DashboardChartPlacement = shouldCompactChart ? .belowInvoices : .anchoredRight
        let kpiArrangement: DashboardKPIArrangement = layoutClass == .compactDesktop ? .gridTwoByTwo : .singleRow
        let decorativePriority = input.contentPriorities[.ambientDecorations] ?? .decorative
        let reducedDecorativeGlow =
            input.reducedEffects ||
            baseScale < input.theme.renderingPolicy.compactGlowReductionThreshold ||
            input.effectiveContentSize.height < minimumChartHeight * 2.5

        let hiddenZones: Set<DashboardZone> =
            reducedDecorativeGlow && decorativePriority <= .decorative ? [.ambientDecorations] : []

        var frameModel = makeFrameModel(
            containerSize: input.windowSize,
            contentSize: input.effectiveContentSize,
            theme: input.theme,
            overrideLayoutClass: shouldCompactChart ? .compactDesktop : layoutClass,
            overrideScale: baseScale
        )

        if reducedDecorativeGlow {
            let visibility = frameModel.decorativeVisibility
            frameModel = DashboardFrameModel(
                containerSize: frameModel.containerSize,
                stageFrame: frameModel.stageFrame,
                layoutClass: frameModel.layoutClass,
                scale: frameModel.scale,
                mainPanelFrame: frameModel.mainPanelFrame,
                footerFrame: frameModel.footerFrame,
                headerDividerFrame: frameModel.headerDividerFrame,
                headerTitleOrigin: frameModel.headerTitleOrigin,
                headerSubtitleOrigin: frameModel.headerSubtitleOrigin,
                billsOrigin: frameModel.billsOrigin,
                invoicesOrigin: frameModel.invoicesOrigin,
                topCardFrames: frameModel.topCardFrames,
                gaugeFrame: frameModel.gaugeFrame,
                invoiceMetricFrames: frameModel.invoiceMetricFrames,
                amountOwedFrame: frameModel.amountOwedFrame,
                chartPlotFrame: frameModel.chartPlotFrame,
                chartBars: frameModel.chartBars,
                tooltipFrame: frameModel.tooltipFrame,
                footerDateCenter: frameModel.footerDateCenter,
                footerTimeCenter: frameModel.footerTimeCenter,
                footerDayCenter: frameModel.footerDayCenter,
                decorativeVisibility: DashboardDecorativeVisibility(
                    glowIntensity: max(0.24, visibility.glowIntensity * (input.reducedEffects ? 0.42 : 0.72)),
                    ambientOpacity: max(0.12, visibility.ambientOpacity * 0.6),
                    hideSecondaryDecorations: true,
                    metalOverlayEnabled: !input.reducedEffects && visibility.metalOverlayEnabled
                )
            )
        }

        return DashboardLayoutOutput(
            frameModel: frameModel,
            typographyScale: typographyScale,
            spacingScale: spacingScale,
            chartPlacement: chartPlacement,
            kpiArrangement: kpiArrangement,
            reducedDecorativeGlow: reducedDecorativeGlow,
            hiddenZones: hiddenZones
        )
    }

    public static func makeFrameModel(
        containerSize: CGSize,
        theme: any AppTheme
    ) -> DashboardFrameModel {
        makeFrameModel(
            containerSize: containerSize,
            contentSize: containerSize,
            theme: theme,
            overrideLayoutClass: nil,
            overrideScale: nil
        )
    }

    public static func makeFrameModel(
        containerSize: CGSize,
        contentSize: CGSize,
        theme: any AppTheme,
        overrideLayoutClass: DashboardLayoutClass?,
        overrideScale: CGFloat?
    ) -> DashboardFrameModel {
        let metrics = theme.layoutMetrics
        let layoutClass = overrideLayoutClass ?? DashboardAdaptiveRules.classify(containerSize: contentSize)
        let outerPadding = max(metrics.outerPadding, min(contentSize.width, contentSize.height) * 0.025)
        let aspect = metrics.referenceCanvasSize.width / metrics.referenceCanvasSize.height
        let maxStageWidth = min(contentSize.width - outerPadding * 2, metrics.wideMaxWidth > 0 ? metrics.wideMaxWidth : .greatestFiniteMagnitude)
        let fittedWidth = min(maxStageWidth, (contentSize.height - outerPadding * 2) * aspect)
        let fittedHeight = fittedWidth / aspect
        let stageOrigin = CGPoint(
            x: (containerSize.width - fittedWidth) / 2,
            y: (containerSize.height - fittedHeight) / 2
        )
        let stageFrame = CGRect(origin: stageOrigin, size: CGSize(width: fittedWidth, height: fittedHeight))
        let scale = overrideScale ?? max(metrics.minimumScale, min(metrics.maximumScale, fittedWidth / metrics.referenceCanvasSize.width))

        switch layoutClass {
        case .compactDesktop:
            return compactFrameModel(containerSize: containerSize, stageFrame: stageFrame, scale: scale, theme: theme)
        case .standardDesktop, .wideDesktop, .ultraWideDesktop:
            return referenceFrameModel(containerSize: containerSize, stageFrame: stageFrame, scale: scale, layoutClass: layoutClass, theme: theme)
        }
    }

    private static func fittedScale(
        contentSize: CGSize,
        scalePolicy: DashboardScalePolicy,
        metrics: ThemeLayoutMetrics
    ) -> CGFloat {
        let outerPadding = max(metrics.outerPadding, min(contentSize.width, contentSize.height) * 0.025)
        let aspect = metrics.referenceCanvasSize.width / metrics.referenceCanvasSize.height
        let maxStageWidth = min(contentSize.width - outerPadding * 2, metrics.wideMaxWidth > 0 ? metrics.wideMaxWidth : .greatestFiniteMagnitude)
        let fittedWidth = min(maxStageWidth, (contentSize.height - outerPadding * 2) * aspect)
        let rawScale = fittedWidth / metrics.referenceCanvasSize.width
        return min(scalePolicy.maximumScale, max(scalePolicy.minimumScale, rawScale))
    }

    private static func referenceFrameModel(
        containerSize: CGSize,
        stageFrame: CGRect,
        scale: CGFloat,
        layoutClass: DashboardLayoutClass,
        theme: any AppTheme
    ) -> DashboardFrameModel {
        let localOrigin = CGPoint.zero
        let footerFrame = SVGMetrics.scaledRect(SVGMetrics.footerFrame, scale: scale, origin: localOrigin)
        let headerDividerFrame = CGRect(
            x: 80 * scale,
            y: 141 * scale,
            width: 1648 * scale,
            height: 2 * scale
        )
        let footerDateCenter = CGPoint(x: footerFrame.midX - 58 * scale, y: footerFrame.midY + 1 * scale)
        let footerTimeCenter = CGPoint(x: footerFrame.maxX - 240 * scale, y: footerFrame.midY + 1 * scale)
        let footerDayCenter = CGPoint(x: footerFrame.maxX - 96 * scale, y: footerFrame.midY + 1 * scale)

        return DashboardFrameModel(
            containerSize: containerSize,
            stageFrame: stageFrame,
            layoutClass: layoutClass,
            scale: scale,
            mainPanelFrame: SVGMetrics.scaledRect(SVGMetrics.mainPanelFrame, scale: scale, origin: localOrigin),
            footerFrame: footerFrame,
            headerDividerFrame: headerDividerFrame,
            headerTitleOrigin: CGPoint(x: SVGMetrics.headerTitleOrigin.x * scale, y: SVGMetrics.headerTitleOrigin.y * scale),
            headerSubtitleOrigin: CGPoint(x: SVGMetrics.headerSubtitleOrigin.x * scale, y: SVGMetrics.headerSubtitleOrigin.y * scale),
            billsOrigin: CGPoint(x: SVGMetrics.billsOrigin.x * scale, y: SVGMetrics.billsOrigin.y * scale),
            invoicesOrigin: CGPoint(x: SVGMetrics.invoicesOrigin.x * scale, y: SVGMetrics.invoicesOrigin.y * scale),
            topCardFrames: SVGMetrics.topCardFrames.mapValues { SVGMetrics.scaledRect($0, scale: scale) },
            gaugeFrame: SVGMetrics.scaledRect(SVGMetrics.gaugeFrame, scale: scale),
            invoiceMetricFrames: SVGMetrics.invoiceMetricFrames.mapValues { SVGMetrics.scaledRect($0, scale: scale) },
            amountOwedFrame: SVGMetrics.scaledRect(SVGMetrics.amountOwedFrame, scale: scale),
            chartPlotFrame: SVGMetrics.scaledRect(SVGMetrics.chartPlotFrame, scale: scale),
            chartBars: SVGMetrics.chartBars.mapValues { SVGMetrics.scaledRect($0, scale: scale) },
            tooltipFrame: SVGMetrics.scaledRect(SVGMetrics.tooltipFrame, scale: scale),
            footerDateCenter: footerDateCenter,
            footerTimeCenter: footerTimeCenter,
            footerDayCenter: footerDayCenter,
            decorativeVisibility: DashboardAdaptiveRules.decorativeVisibility(layoutClass: layoutClass, scale: scale, theme: theme)
        )
    }

    private static func compactFrameModel(
        containerSize: CGSize,
        stageFrame: CGRect,
        scale: CGFloat,
        theme: any AppTheme
    ) -> DashboardFrameModel {
        let insetX = 126 * scale
        let availableWidth = stageFrame.width - insetX - 130 * scale
        let cardGap = 24 * scale
        let cardWidth = max(250 * scale, (availableWidth - cardGap) / 2)
        let cardHeight = 212 * scale
        let firstRowY = 229 * scale
        let secondRowY = firstRowY + cardHeight + cardGap

        let topCards: [SVGKPIKind: CGRect] = [
            .readyToAssign: CGRect(x: insetX, y: firstRowY, width: cardWidth, height: cardHeight),
            .pendingSignOffs: CGRect(x: insetX + cardWidth + cardGap, y: firstRowY, width: cardWidth, height: cardHeight),
            .declined: CGRect(x: insetX, y: secondRowY, width: cardWidth, height: cardHeight),
            .occured: CGRect(x: insetX + cardWidth + cardGap, y: secondRowY, width: cardWidth, height: cardHeight)
        ]

        let invoicesY = secondRowY + cardHeight + 88 * scale
        let gaugeFrame = CGRect(x: insetX, y: invoicesY + 66 * scale, width: 300 * scale, height: 457 * scale)
        let paidFrame = CGRect(x: gaugeFrame.maxX + 78 * scale, y: invoicesY + 36 * scale, width: 253 * scale, height: 156 * scale)
        let liveFrame = CGRect(x: gaugeFrame.maxX + 78 * scale, y: paidFrame.maxY + 54 * scale, width: 253 * scale, height: 156 * scale)
        let chartFrame = CGRect(
            x: insetX + 36 * scale,
            y: liveFrame.maxY + 62 * scale,
            width: availableWidth - 72 * scale,
            height: 304 * scale
        )

        let barBaseHeight = 238 * scale
        let monthGap = 28.8 * scale
        let barWidth = 44 * scale
        let chartStartX = chartFrame.minX + 72 * scale
        let chartBottom = chartFrame.maxY - 54 * scale
        let chartBars: [SVGChartMonth: CGRect] = [
            .feb: CGRect(x: chartStartX, y: chartBottom - 149.6 * scale, width: barWidth, height: 149.6 * scale),
            .mar: CGRect(x: chartStartX + (barWidth + monthGap), y: chartBottom - 188.8 * scale, width: barWidth, height: 188.8 * scale),
            .april: CGRect(x: chartStartX + 2 * (barWidth + monthGap), y: chartBottom - 168.8 * scale, width: barWidth, height: 168.8 * scale),
            .may: CGRect(x: chartStartX + 3 * (barWidth + monthGap), y: chartBottom - 237.6 * scale, width: barWidth, height: 237.6 * scale),
            .june: CGRect(x: chartStartX + 4 * (barWidth + monthGap), y: chartBottom - 214.4 * scale, width: barWidth, height: 214.4 * scale),
            .july: CGRect(x: chartStartX + 5 * (barWidth + monthGap), y: chartBottom - 214.4 * scale, width: barWidth, height: 214.4 * scale),
            .aug: CGRect(x: chartStartX + 6 * (barWidth + monthGap), y: chartBottom - barBaseHeight, width: barWidth, height: barBaseHeight)
        ]
        let augFrame = chartBars[.aug] ?? .zero
        let tooltipFrame = CGRect(x: augFrame.midX - 46 * scale, y: augFrame.minY - 52 * scale, width: 92.3265 * scale, height: 24.2653 * scale)
        let amountOwedFrame = CGRect(
            x: gaugeFrame.minX - 43 * scale,
            y: gaugeFrame.maxY - 233 * scale,
            width: 333 * scale,
            height: 189.776 * scale
        )
        let footerFrame = CGRect(x: insetX + 320 * scale, y: stageFrame.height - 142 * scale, width: stageFrame.width - insetX - 48 * scale, height: 85 * scale)

        return DashboardFrameModel(
            containerSize: containerSize,
            stageFrame: stageFrame,
            layoutClass: .compactDesktop,
            scale: scale,
            mainPanelFrame: CGRect(x: 80 * scale, y: 0, width: 1648 * scale, height: 1117 * scale),
            footerFrame: footerFrame,
            headerDividerFrame: CGRect(x: 80 * scale, y: 141 * scale, width: 1648 * scale, height: 2 * scale),
            headerTitleOrigin: CGPoint(x: SVGMetrics.headerTitleOrigin.x * scale, y: SVGMetrics.headerTitleOrigin.y * scale),
            headerSubtitleOrigin: CGPoint(x: SVGMetrics.headerSubtitleOrigin.x * scale, y: SVGMetrics.headerSubtitleOrigin.y * scale),
            billsOrigin: CGPoint(x: SVGMetrics.billsOrigin.x * scale, y: 197 * scale),
            invoicesOrigin: CGPoint(x: SVGMetrics.invoicesOrigin.x * scale, y: invoicesY),
            topCardFrames: topCards,
            gaugeFrame: gaugeFrame,
            invoiceMetricFrames: [.paidInvoices: paidFrame, .liveFundUpdate: liveFrame],
            amountOwedFrame: amountOwedFrame,
            chartPlotFrame: chartFrame,
            chartBars: chartBars,
            tooltipFrame: tooltipFrame,
            footerDateCenter: CGPoint(x: footerFrame.midX - 44 * scale, y: footerFrame.midY),
            footerTimeCenter: CGPoint(x: footerFrame.maxX - 220 * scale, y: footerFrame.midY),
            footerDayCenter: CGPoint(x: footerFrame.maxX - 86 * scale, y: footerFrame.midY),
            decorativeVisibility: DashboardAdaptiveRules.decorativeVisibility(layoutClass: .compactDesktop, scale: scale, theme: theme)
        )
    }
}
