import CoreGraphics

public enum DashboardLayoutAuditSeverity: Int, Comparable, Sendable {
    case info = 0
    case warning = 1
    case critical = 2

    public static func < (lhs: DashboardLayoutAuditSeverity, rhs: DashboardLayoutAuditSeverity) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

public struct DashboardLayoutAuditIssue: Identifiable, Hashable, Sendable {
    public let id: String
    public let severity: DashboardLayoutAuditSeverity
    public let summary: String

    public init(id: String, severity: DashboardLayoutAuditSeverity, summary: String) {
        self.id = id
        self.severity = severity
        self.summary = summary
    }
}

public struct DashboardLayoutAuditReport {
    public let scenario: GUIValidationScenario
    public let output: DashboardLayoutOutput
    public let issues: [DashboardLayoutAuditIssue]
    public let score: Int

    public var passed: Bool {
        issues.allSatisfy { $0.severity != .critical }
    }
}

public enum DashboardLayoutAuditor {
    public static func audit(
        scenario: GUIValidationScenario,
        theme: any AppTheme,
        contentPriorities: [DashboardZone: DashboardContentPriority] = DashboardAdaptiveRules.defaultContentPriorities,
        overlapPolicy: DashboardOverlapPolicy = .preserveFloatingCard,
        scalePolicy: DashboardScalePolicy? = nil,
        reducedEffects: Bool = false
    ) -> DashboardLayoutAuditReport {
        let resolvedScalePolicy = scalePolicy ?? .themeDefault(for: theme)
        let output = DashboardLayoutEngine.makeLayout(
            input: DashboardLayoutInput(
                windowSize: scenario.windowSize,
                effectiveContentSize: scenario.windowSize,
                backingScaleFactor: scenario.backingScaleFactor,
                theme: theme,
                contentPriorities: contentPriorities,
                overlapPolicy: overlapPolicy,
                scalePolicy: resolvedScalePolicy,
                reducedEffects: reducedEffects
            )
        )

        let frameModel = output.frameModel
        let stageBounds = CGRect(origin: .zero, size: frameModel.stageFrame.size)
        let minReadableChartWidth = theme.renderingPolicy.chartMinimumReadableWidth * output.spacingScale * (output.chartPlacement == .belowInvoices ? 0.82 : 0.94)
        let minReadableChartHeight = theme.renderingPolicy.chartMinimumReadableHeight * output.spacingScale * (output.chartPlacement == .belowInvoices ? 0.82 : 0.9)
        let scaledHeaderTitle = theme.typography.headerTitle.size * output.typographyScale
        let scaledSectionTitle = theme.typography.sectionTitle.size * output.typographyScale
        let scaledCardTitle = theme.typography.cardTitle.size * output.typographyScale
        let scaledCaption = theme.typography.caption.size * output.typographyScale

        var issues: [DashboardLayoutAuditIssue] = []
        func issue(_ id: String, _ severity: DashboardLayoutAuditSeverity, _ summary: String) {
            issues.append(DashboardLayoutAuditIssue(id: id, severity: severity, summary: summary))
        }

        if frameModel.layoutClass != scenario.expectedLayoutClass {
            issue("layout-class", .critical, "Expected \(scenario.expectedLayoutClass.rawValue) for \(scenario.title), got \(frameModel.layoutClass.rawValue).")
        }

        if frameModel.mainPanelFrame.width > stageBounds.width + 0.5 || frameModel.mainPanelFrame.height > stageBounds.height + 0.5 {
            issue("main-panel-overflow", .critical, "Main panel overflows the available stage bounds.")
        }

        let anchoredFrames = frameModel.topCardFrames.values + [frameModel.gaugeFrame, frameModel.amountOwedFrame, frameModel.chartPlotFrame, frameModel.footerFrame]
        if scenario.strictStageContainment {
            for (index, frame) in anchoredFrames.enumerated() where !contains(frame, in: stageBounds, tolerance: 2) {
                issue("frame-out-of-bounds-\(index)", .critical, "A major dashboard frame falls outside the stage at \(scenario.title).")
            }
        }

        let topCards = frameModel.topCardFrames
        if topCards.count == 4 {
            let sortedByX = topCards.values.sorted { lhs, rhs in lhs.minX < rhs.minX }
            let horizontalGaps = zip(sortedByX, sortedByX.dropFirst()).map { $1.minX - $0.maxX }
            if output.kpiArrangement == .singleRow, let smallestGap = horizontalGaps.min(), smallestGap < 12 * output.spacingScale {
                issue("kpi-gap-single-row", .critical, "Single-row KPI cards compress too tightly for readable scanning.")
            }

            if output.kpiArrangement == .gridTwoByTwo {
                guard
                    let ready = topCards[.readyToAssign],
                    let pending = topCards[.pendingSignOffs],
                    let declined = topCards[.declined]
                else {
                    issue("kpi-grid-missing", .critical, "Compact KPI grid could not be validated.")
                    return DashboardLayoutAuditReport(scenario: scenario, output: output, issues: issues, score: score(for: issues))
                }

                let horizontalGap = pending.minX - ready.maxX
                let verticalGap = declined.minY - ready.maxY
                if horizontalGap < 14 * output.spacingScale {
                    issue("kpi-gap-grid-horizontal", .critical, "Compact KPI columns are too close together.")
                }
                if verticalGap < 16 * output.spacingScale {
                    issue("kpi-gap-grid-vertical", .critical, "Compact KPI rows are too close together.")
                }
            }
        }

        if frameModel.chartPlotFrame.width < minReadableChartWidth {
            issue("chart-width", .critical, "Chart width drops below the minimum readable threshold for \(scenario.title).")
        }
        if frameModel.chartPlotFrame.height < minReadableChartHeight {
            issue("chart-height", .critical, "Chart height drops below the minimum readable threshold for \(scenario.title).")
        }

        if scaledHeaderTitle < 21 {
            issue("header-font", .critical, "Header title scale falls below the readable threshold.")
        }
        if scaledSectionTitle < 17 {
            issue("section-font", .critical, "Section title scale falls below the readable threshold.")
        }
        if scaledCardTitle < 11 {
            issue("card-font", .warning, "Card title scale is starting to get cramped.")
        }
        if scaledCaption < 9.2 {
            issue("caption-font", .warning, "Caption scale is below the preferred readability floor.")
        }

        if output.typographyScale < resolvedScalePolicy.minimumTypographyScale || output.typographyScale > resolvedScalePolicy.maximumTypographyScale {
            issue("typography-scale-bounds", .critical, "Typography scale escaped the configured policy clamps.")
        }
        if output.spacingScale < resolvedScalePolicy.minimumSpacingScale || output.spacingScale > resolvedScalePolicy.maximumSpacingScale {
            issue("spacing-scale-bounds", .critical, "Spacing scale escaped the configured policy clamps.")
        }

        if scenario.expectsReducedDecorativeGlow && !output.reducedDecorativeGlow {
            issue("decorative-glow-reduction", .warning, "Compact scenario did not reduce decorative glow as expected.")
        }

        if !scenario.expectsReducedDecorativeGlow && scenario.windowSize.height >= 1_000 && output.hiddenZones.contains(.ambientDecorations) && !reducedEffects {
            issue("ambient-hidden-too-early", .warning, "Ambient decoration was hidden earlier than expected for a spacious scenario.")
        }

        if frameModel.amountOwedFrame.maxY < frameModel.gaugeFrame.minY || frameModel.amountOwedFrame.minX > frameModel.gaugeFrame.maxX {
            issue("glass-card-overlap", .critical, "The Amount Owed overlay no longer preserves its intended relationship with the invoice zone.")
        }

        if scenario.strictStageContainment && frameModel.footerFrame.minY < frameModel.chartPlotFrame.maxY - 4 {
            issue("footer-collision", .critical, "Footer strip collides with chart content.")
        }

        if frameModel.layoutClass == .ultraWideDesktop, frameModel.stageFrame.width > theme.spacing.ultraWideMaxContentWidth + 1 {
            issue("ultrawide-stretch", .critical, "Ultra-wide stage exceeds the configured max content band.")
        }

        return DashboardLayoutAuditReport(
            scenario: scenario,
            output: output,
            issues: issues.sorted { lhs, rhs in
                if lhs.severity == rhs.severity {
                    return lhs.id < rhs.id
                }
                return lhs.severity > rhs.severity
            },
            score: score(for: issues)
        )
    }

    private static func score(for issues: [DashboardLayoutAuditIssue]) -> Int {
        let deductions = issues.reduce(into: 0) { partialResult, issue in
            switch issue.severity {
            case .info:
                partialResult += 1
            case .warning:
                partialResult += 6
            case .critical:
                partialResult += 18
            }
        }
        return max(0, 100 - deductions)
    }

    private static func contains(_ frame: CGRect, in bounds: CGRect, tolerance: CGFloat) -> Bool {
        frame.minX >= bounds.minX - tolerance &&
        frame.minY >= bounds.minY - tolerance &&
        frame.maxX <= bounds.maxX + tolerance &&
        frame.maxY <= bounds.maxY + tolerance
    }
}
