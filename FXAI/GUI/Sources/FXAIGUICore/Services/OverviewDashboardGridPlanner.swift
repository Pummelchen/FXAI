import CoreGraphics
import Foundation

public struct OverviewDashboardGridPlacement: Identifiable, Hashable, Sendable {
    public let id: UUID
    public let widgetID: UUID
    public let widgetKind: OverviewDashboardWidgetKind
    public let frame: CGRect
    public let widthUnits: Int
    public let heightUnits: Int

    public init(
        widgetID: UUID,
        widgetKind: OverviewDashboardWidgetKind,
        frame: CGRect,
        widthUnits: Int,
        heightUnits: Int
    ) {
        self.id = widgetID
        self.widgetID = widgetID
        self.widgetKind = widgetKind
        self.frame = frame
        self.widthUnits = widthUnits
        self.heightUnits = heightUnits
    }
}

public struct OverviewDashboardGridPlan: Hashable, Sendable {
    public let columnCount: Int
    public let unitPoints: CGFloat
    public let gapPoints: CGFloat
    public let placements: [OverviewDashboardGridPlacement]
    public let contentHeight: CGFloat

    public init(
        columnCount: Int,
        unitPoints: CGFloat,
        gapPoints: CGFloat,
        placements: [OverviewDashboardGridPlacement],
        contentHeight: CGFloat
    ) {
        self.columnCount = columnCount
        self.unitPoints = unitPoints
        self.gapPoints = gapPoints
        self.placements = placements
        self.contentHeight = contentHeight
    }
}

public enum OverviewDashboardGridPlanner {
    public static let minimumGridUnitPoints: CGFloat = 40
    public static let preferredGridUnitPoints: CGFloat = 58
    public static let maximumGridUnitPoints: CGFloat = 84
    public static let gridGapPoints: CGFloat = 12

    public static func plan(
        availableWidth: CGFloat,
        widgets: [OverviewDashboardWidgetLayout],
        baseGridUnitPoints: CGFloat
    ) -> OverviewDashboardGridPlan {
        let gap = gridGapPoints
        let unitBase = max(minimumGridUnitPoints, baseGridUnitPoints)
        let columnCount = resolvedColumnCount(for: availableWidth, preferredUnit: unitBase)
        let unitPoints = resolvedUnitPoints(for: availableWidth, columns: columnCount, gap: gap)

        var occupancy: [[Bool]] = []
        var placements: [OverviewDashboardGridPlacement] = []
        var deepestRow = 0

        for widget in widgets {
            let spec = OverviewDashboardLayoutState.spec(for: widget.kind)
            let widthUnits = min(max(widget.widthUnits, spec.minimumWidthUnits), min(spec.maximumWidthUnits, columnCount))
            let heightUnits = min(max(widget.heightUnits, spec.minimumHeightUnits), spec.maximumHeightUnits)

            let slot = firstAvailableSlot(
                occupancy: &occupancy,
                columns: columnCount,
                widthUnits: widthUnits,
                heightUnits: heightUnits
            )

            markOccupied(
                occupancy: &occupancy,
                originRow: slot.row,
                originColumn: slot.column,
                widthUnits: widthUnits,
                heightUnits: heightUnits
            )

            let x = CGFloat(slot.column) * (unitPoints + gap)
            let y = CGFloat(slot.row) * (unitPoints + gap)
            let width = (CGFloat(widthUnits) * unitPoints) + (CGFloat(max(0, widthUnits - 1)) * gap)
            let height = (CGFloat(heightUnits) * unitPoints) + (CGFloat(max(0, heightUnits - 1)) * gap)

            placements.append(
                OverviewDashboardGridPlacement(
                    widgetID: widget.id,
                    widgetKind: widget.kind,
                    frame: CGRect(x: x, y: y, width: width, height: height),
                    widthUnits: widthUnits,
                    heightUnits: heightUnits
                )
            )
            deepestRow = max(deepestRow, slot.row + heightUnits)
        }

        let contentHeight: CGFloat
        if deepestRow == 0 {
            contentHeight = 0
        } else {
            contentHeight = (CGFloat(deepestRow) * unitPoints) + (CGFloat(max(0, deepestRow - 1)) * gap)
        }

        return OverviewDashboardGridPlan(
            columnCount: columnCount,
            unitPoints: unitPoints,
            gapPoints: gap,
            placements: placements,
            contentHeight: contentHeight
        )
    }

    private static func resolvedColumnCount(for availableWidth: CGFloat, preferredUnit: CGFloat) -> Int {
        guard availableWidth > minimumGridUnitPoints else { return 1 }

        let target = max(minimumGridUnitPoints, preferredUnit)
        var columns = max(1, Int((availableWidth + gridGapPoints) / (target + gridGapPoints)))
        columns = min(16, max(1, columns))

        while columns > 1 {
            let unit = resolvedUnitPoints(for: availableWidth, columns: columns, gap: gridGapPoints)
            if unit >= minimumGridUnitPoints {
                return columns
            }
            columns -= 1
        }
        return 1
    }

    private static func resolvedUnitPoints(for availableWidth: CGFloat, columns: Int, gap: CGFloat) -> CGFloat {
        guard columns > 0 else { return minimumGridUnitPoints }
        let totalGap = CGFloat(max(0, columns - 1)) * gap
        let proposed = floor((availableWidth - totalGap) / CGFloat(columns))
        return max(minimumGridUnitPoints, min(maximumGridUnitPoints, proposed))
    }

    private static func firstAvailableSlot(
        occupancy: inout [[Bool]],
        columns: Int,
        widthUnits: Int,
        heightUnits: Int
    ) -> (row: Int, column: Int) {
        var row = 0
        while true {
            ensureRows(&occupancy, count: row + heightUnits, columns: columns)
            for column in 0...(max(0, columns - widthUnits)) {
                if isAreaFree(occupancy: occupancy, row: row, column: column, widthUnits: widthUnits, heightUnits: heightUnits) {
                    return (row, column)
                }
            }
            row += 1
        }
    }

    private static func ensureRows(_ occupancy: inout [[Bool]], count: Int, columns: Int) {
        while occupancy.count < count {
            occupancy.append(Array(repeating: false, count: columns))
        }
    }

    private static func isAreaFree(
        occupancy: [[Bool]],
        row: Int,
        column: Int,
        widthUnits: Int,
        heightUnits: Int
    ) -> Bool {
        for rowOffset in 0..<heightUnits {
            for columnOffset in 0..<widthUnits {
                if occupancy[row + rowOffset][column + columnOffset] {
                    return false
                }
            }
        }
        return true
    }

    private static func markOccupied(
        occupancy: inout [[Bool]],
        originRow: Int,
        originColumn: Int,
        widthUnits: Int,
        heightUnits: Int
    ) {
        for rowOffset in 0..<heightUnits {
            for columnOffset in 0..<widthUnits {
                occupancy[originRow + rowOffset][originColumn + columnOffset] = true
            }
        }
    }
}
