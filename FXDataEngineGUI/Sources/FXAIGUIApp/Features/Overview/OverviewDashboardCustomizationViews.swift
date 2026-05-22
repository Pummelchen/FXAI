import CoreTransferable
import FXAIGUICore
import SwiftUI
import UniformTypeIdentifiers

private extension UTType {
    static let fxaiDashboardSection = UTType(exportedAs: "com.fxai.gui.dashboard-section")
    static let fxaiDashboardWidget = UTType(exportedAs: "com.fxai.gui.dashboard-widget")
}

struct OverviewSectionDragPayload: Codable, Transferable, Sendable {
    let sectionID: UUID

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .fxaiDashboardSection)
    }
}

struct OverviewWidgetDragPayload: Codable, Transferable, Sendable {
    let sectionID: UUID
    let widgetID: UUID

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .fxaiDashboardWidget)
    }
}

struct OverviewDashboardSectionView: View {
    @EnvironmentObject private var model: FXAIGUIModel

    let section: OverviewDashboardSectionLayout
    let availableWidth: CGFloat
    let customizationEnabled: Bool
    let widgets: [OverviewDashboardWidgetLayout]
    let contentProvider: (OverviewDashboardWidgetLayout) -> AnyView

    var body: some View {
        let plan = OverviewDashboardGridPlanner.plan(
            availableWidth: availableWidth,
            widgets: widgets,
            baseGridUnitPoints: CGFloat(model.overviewLayout.gridUnitPoints)
        )

        VStack(alignment: .leading, spacing: 14) {
            headerCard(plan: plan)

            if widgets.isEmpty {
                emptyDropZone
            } else {
                ZStack(alignment: .topLeading) {
                    ForEach(plan.placements) { placement in
                        if let widget = widgets.first(where: { $0.id == placement.widgetID }) {
                            OverviewDashboardWidgetContainer(
                                sectionID: section.id,
                                widget: widget,
                                placement: placement,
                                gridStepPoints: plan.unitPoints + plan.gapPoints,
                                customizationEnabled: customizationEnabled,
                                content: contentProvider(widget)
                            )
                        }
                    }
                }
                .frame(maxWidth: .infinity, minHeight: max(plan.contentHeight, 1), alignment: .topLeading)
            }
        }
        .contentShape(Rectangle())
        .dropDestination(for: OverviewSectionDragPayload.self) { items, _ in
            guard customizationEnabled, let payload = items.first else { return false }
            model.reorderOverviewSection(draggedSectionID: payload.sectionID, before: section.id)
            return true
        }
    }

    @ViewBuilder
    private func headerCard(plan: OverviewDashboardGridPlan) -> some View {
        let card = FXAIVisualEffectSurface(style: .card, cornerRadius: 20, contentPadding: 14, tint: FXAITheme.accent.opacity(0.06)) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 14) {
                    headerLabels(plan: plan)
                    Spacer(minLength: 12)
                    if customizationEnabled {
                        sectionControls
                    }
                }
                VStack(alignment: .leading, spacing: 12) {
                    headerLabels(plan: plan)
                    if customizationEnabled {
                        sectionControls
                    }
                }
            }
        }
        .frame(maxWidth: .infinity)

        if customizationEnabled {
            card.draggable(OverviewSectionDragPayload(sectionID: section.id))
        } else {
            card
        }
    }

    private func headerLabels(plan: OverviewDashboardGridPlan) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: customizationEnabled ? "line.3.horizontal" : "square.grid.2x2")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(FXAITheme.accent)

                Text(section.kind.title)
                    .font(.headline)
                    .foregroundStyle(FXAITheme.textPrimary)
            }
            Text(section.kind.subtitle)
                .font(.subheadline)
                .foregroundStyle(FXAITheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Grid: \(plan.columnCount) cols • \(Int(plan.unitPoints.rounded())) pt units")
                .font(.caption)
                .foregroundStyle(FXAITheme.textMuted)
        }
    }

    private var sectionControls: some View {
        HStack(spacing: 8) {
            chromeButton(systemName: "arrow.up") {
                model.moveOverviewSection(section.id, by: -1)
            }
            chromeButton(systemName: "arrow.down") {
                model.moveOverviewSection(section.id, by: 1)
            }
        }
    }

    private var emptyDropZone: some View {
        RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(FXAITheme.panel.opacity(0.22))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(FXAITheme.stroke.opacity(0.8), style: StrokeStyle(lineWidth: 1, dash: [6, 6]))
            )
            .frame(height: 92)
            .overlay(
                Text(customizationEnabled ? "Drop widgets here" : "No visible widgets in this category right now.")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(FXAITheme.textSecondary)
            )
    }

    private func chromeButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
                .frame(width: 28, height: 28)
                .background(
                    FXAIGlassCapsuleBackground(style: .badge, tint: FXAITheme.accent.opacity(0.10))
                )
        }
        .buttonStyle(.plain)
    }
}

struct OverviewDashboardWidgetContainer: View {
    @EnvironmentObject private var model: FXAIGUIModel

    let sectionID: UUID
    let widget: OverviewDashboardWidgetLayout
    let placement: OverviewDashboardGridPlacement
    let gridStepPoints: CGFloat
    let customizationEnabled: Bool
    let content: AnyView

    var body: some View {
        let spec = OverviewDashboardLayoutState.spec(for: widget.kind)
        DashboardGridSurfaceContainer(
            title: widget.kind.title,
            placement: DashboardGridSurfacePlacement(placement),
            gridStepPoints: gridStepPoints,
            minimumWidthUnits: spec.minimumWidthUnits,
            maximumWidthUnits: spec.maximumWidthUnits,
            minimumHeightUnits: spec.minimumHeightUnits,
            maximumHeightUnits: spec.maximumHeightUnits,
            customizationEnabled: customizationEnabled,
            onMove: { columnDelta, rowDelta in
                model.moveOverviewWidgetOnGrid(
                    sectionID: sectionID,
                    widgetID: widget.id,
                    columnDelta: columnDelta,
                    rowDelta: rowDelta
                )
            },
            onResize: { widthDelta, heightDelta in
                model.resizeOverviewWidget(
                    sectionID: sectionID,
                    widgetID: widget.id,
                    widthDelta: widthDelta,
                    heightDelta: heightDelta
                )
            },
            content: content
        )
    }
}
