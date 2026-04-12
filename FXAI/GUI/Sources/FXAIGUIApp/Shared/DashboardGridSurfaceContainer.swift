import FXAIGUICore
import SwiftUI

struct DashboardGridSurfacePlacement: Hashable {
    let frame: CGRect
    let columnIndex: Int
    let rowIndex: Int
    let widthUnits: Int
    let heightUnits: Int

    init(_ placement: OverviewDashboardGridPlacement) {
        frame = placement.frame
        columnIndex = placement.columnIndex
        rowIndex = placement.rowIndex
        widthUnits = placement.widthUnits
        heightUnits = placement.heightUnits
    }

    init(_ placement: RoleWorkspaceDashboardGridPlacement) {
        frame = placement.frame
        columnIndex = placement.columnIndex
        rowIndex = placement.rowIndex
        widthUnits = placement.widthUnits
        heightUnits = placement.heightUnits
    }
}

struct DashboardGridSurfaceContainer: View {
    @State private var dragTranslation: CGSize = .zero
    @State private var resizeTranslation: CGSize = .zero

    let title: String
    let placement: DashboardGridSurfacePlacement
    let gridStepPoints: CGFloat
    let minimumWidthUnits: Int
    let maximumWidthUnits: Int
    let minimumHeightUnits: Int
    let maximumHeightUnits: Int
    let customizationEnabled: Bool
    let onMove: (Int, Int) -> Void
    let onResize: (Int, Int) -> Void
    let content: AnyView

    var body: some View {
        let displayWidth = clampedPreviewDimension(
            currentPoints: placement.frame.width,
            currentUnits: placement.widthUnits,
            minimumUnits: minimumWidthUnits,
            maximumUnits: maximumWidthUnits,
            translationPoints: resizeTranslation.width
        )
        let displayHeight = clampedPreviewDimension(
            currentPoints: placement.frame.height,
            currentUnits: placement.heightUnits,
            minimumUnits: minimumHeightUnits,
            maximumUnits: maximumHeightUnits,
            translationPoints: resizeTranslation.height
        )

        let card = ZStack(alignment: .topTrailing) {
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            if customizationEnabled {
                customizationOverlay
                    .padding(10)
            }
        }
        .frame(width: displayWidth, height: displayHeight)
        .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))

        card
            .position(
                x: placement.frame.minX + (displayWidth / 2),
                y: placement.frame.minY + (displayHeight / 2)
            )
            .offset(x: dragTranslation.width, y: dragTranslation.height)
    }

    private var customizationOverlay: some View {
        VStack(alignment: .trailing, spacing: 8) {
            HStack(spacing: 8) {
                labelChip
                Spacer(minLength: 8)
                movementPad
            }

            HStack(spacing: 8) {
                resizeGroup(
                    title: "W",
                    decrease: { onResize(-1, 0) },
                    increase: { onResize(1, 0) }
                )
                resizeGroup(
                    title: "H",
                    decrease: { onResize(0, -1) },
                    increase: { onResize(0, 1) }
                )
            }

            Spacer(minLength: 0)

            resizeHandle
        }
    }

    private var labelChip: some View {
        HStack(spacing: 8) {
            Image(systemName: "hand.draw")
            Text(title)
                .lineLimit(1)
            Text("C\(placement.columnIndex + 1) R\(placement.rowIndex + 1)")
                .foregroundStyle(FXAITheme.textMuted)
            Text("\(placement.widthUnits)×\(placement.heightUnits)")
                .foregroundStyle(FXAITheme.textMuted)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(FXAITheme.textPrimary)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(FXAIGlassCapsuleBackground(style: .badge, tint: FXAITheme.accentSoft.opacity(0.12)))
        .gesture(moveGesture)
        .help("Drag to move this panel on the 1 cm grid")
    }

    private var movementPad: some View {
        VStack(spacing: 6) {
            controlButton(systemName: "arrow.up") {
                onMove(0, -1)
            }

            HStack(spacing: 6) {
                controlButton(systemName: "arrow.left") {
                    onMove(-1, 0)
                }
                controlButton(systemName: "arrow.down") {
                    onMove(0, 1)
                }
                controlButton(systemName: "arrow.right") {
                    onMove(1, 0)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(FXAIGlassCapsuleBackground(style: .badge, tint: FXAITheme.accent.opacity(0.10)))
    }

    private func resizeGroup(title: String, decrease: @escaping () -> Void, increase: @escaping () -> Void) -> some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.caption2.weight(.bold))
                .foregroundStyle(FXAITheme.textSecondary)
            controlButton(systemName: "minus", action: decrease)
            controlButton(systemName: "plus", action: increase)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(FXAIGlassCapsuleBackground(style: .badge, tint: FXAITheme.accent.opacity(0.08)))
    }

    private var resizeHandle: some View {
        Image(systemName: "arrow.up.left.and.arrow.down.right")
            .font(.caption.weight(.semibold))
            .foregroundStyle(FXAITheme.textPrimary)
            .frame(width: 28, height: 28)
            .background(FXAIGlassCapsuleBackground(style: .badge, tint: FXAITheme.accent.opacity(0.12)))
            .gesture(resizeGesture)
            .help("Drag to resize this panel on the 1 cm grid")
    }

    private func controlButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.caption.weight(.semibold))
                .foregroundStyle(FXAITheme.textPrimary)
                .frame(width: 24, height: 24)
        }
        .buttonStyle(.plain)
    }

    private var moveGesture: some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { value in
                dragTranslation = value.translation
            }
            .onEnded { value in
                defer { dragTranslation = .zero }

                let snapStep = max(gridStepPoints, 1)
                let columnDelta = Int((value.translation.width / snapStep).rounded())
                let rowDelta = Int((value.translation.height / snapStep).rounded())
                guard columnDelta != 0 || rowDelta != 0 else { return }

                onMove(columnDelta, rowDelta)
            }
    }

    private var resizeGesture: some Gesture {
        DragGesture(minimumDistance: 3)
            .onChanged { value in
                resizeTranslation = value.translation
            }
            .onEnded { value in
                defer { resizeTranslation = .zero }

                let snapStep = max(gridStepPoints, 1)
                let widthDelta = Int((value.translation.width / snapStep).rounded())
                let heightDelta = Int((value.translation.height / snapStep).rounded())
                guard widthDelta != 0 || heightDelta != 0 else { return }

                onResize(widthDelta, heightDelta)
            }
    }

    private func clampedPreviewDimension(
        currentPoints: CGFloat,
        currentUnits: Int,
        minimumUnits: Int,
        maximumUnits: Int,
        translationPoints: CGFloat
    ) -> CGFloat {
        let minimumPoints = currentPoints + (CGFloat(minimumUnits - currentUnits) * gridStepPoints)
        let maximumPoints = currentPoints + (CGFloat(maximumUnits - currentUnits) * gridStepPoints)
        return min(max(currentPoints + translationPoints, minimumPoints), maximumPoints)
    }
}
