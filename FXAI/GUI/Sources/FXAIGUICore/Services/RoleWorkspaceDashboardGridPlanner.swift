import CoreGraphics
import Foundation

public struct RoleWorkspaceDashboardGridPlacement: Identifiable, Hashable, Sendable {
    public let id: UUID
    public let panelID: UUID
    public let panelKind: RoleWorkspaceDashboardPanelKind
    public let frame: CGRect
    public let columnIndex: Int
    public let rowIndex: Int
    public let widthUnits: Int
    public let heightUnits: Int

    public init(
        panelID: UUID,
        panelKind: RoleWorkspaceDashboardPanelKind,
        frame: CGRect,
        columnIndex: Int,
        rowIndex: Int,
        widthUnits: Int,
        heightUnits: Int
    ) {
        id = panelID
        self.panelID = panelID
        self.panelKind = panelKind
        self.frame = frame
        self.columnIndex = columnIndex
        self.rowIndex = rowIndex
        self.widthUnits = widthUnits
        self.heightUnits = heightUnits
    }
}

public struct RoleWorkspaceDashboardGridPlan: Hashable, Sendable {
    public let columnCount: Int
    public let unitPoints: CGFloat
    public let gapPoints: CGFloat
    public let placements: [RoleWorkspaceDashboardGridPlacement]
    public let contentHeight: CGFloat

    public init(
        columnCount: Int,
        unitPoints: CGFloat,
        gapPoints: CGFloat,
        placements: [RoleWorkspaceDashboardGridPlacement],
        contentHeight: CGFloat
    ) {
        self.columnCount = columnCount
        self.unitPoints = unitPoints
        self.gapPoints = gapPoints
        self.placements = placements
        self.contentHeight = contentHeight
    }
}

public enum RoleWorkspaceDashboardGridPlanner {
    public static let minimumGridUnitPoints: CGFloat = 72.0 / 2.54
    public static let preferredGridUnitPoints: CGFloat = 36
    public static let maximumGridUnitPoints: CGFloat = 56
    public static let gridGapPoints: CGFloat = 12

    public static func plan(
        availableWidth: CGFloat,
        panels: [RoleWorkspaceDashboardPanelLayout],
        baseGridUnitPoints: CGFloat
    ) -> RoleWorkspaceDashboardGridPlan {
        let gap = gridGapPoints
        let unitBase = max(minimumGridUnitPoints, baseGridUnitPoints)
        let columnCount = resolvedColumnCount(for: availableWidth, preferredUnit: unitBase)
        let unitPoints = resolvedUnitPoints(for: availableWidth, columns: columnCount, gap: gap)

        var occupancy: [[Bool]] = []
        var placements: [RoleWorkspaceDashboardGridPlacement] = []
        var deepestRow = 0

        for panel in panels {
            let spec = RoleWorkspaceDashboardLayoutState.spec(for: panel.kind)
            let widthUnits = min(max(panel.widthUnits, spec.minimumWidthUnits), min(spec.maximumWidthUnits, columnCount))
            let heightUnits = min(max(panel.heightUnits, spec.minimumHeightUnits), spec.maximumHeightUnits)

            let slot = preferredAvailableSlot(
                occupancy: &occupancy,
                columns: columnCount,
                panel: panel,
                widthUnits: widthUnits,
                heightUnits: heightUnits
            ) ?? firstAvailableSlot(
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
                RoleWorkspaceDashboardGridPlacement(
                    panelID: panel.id,
                    panelKind: panel.kind,
                    frame: CGRect(x: x, y: y, width: width, height: height),
                    columnIndex: slot.column,
                    rowIndex: slot.row,
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

        return RoleWorkspaceDashboardGridPlan(
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

    private static func preferredAvailableSlot(
        occupancy: inout [[Bool]],
        columns: Int,
        panel: RoleWorkspaceDashboardPanelLayout,
        widthUnits: Int,
        heightUnits: Int
    ) -> (row: Int, column: Int)? {
        let preferredRow = max(0, panel.rowUnits)
        let preferredColumn = min(max(0, panel.columnUnits), max(0, columns - widthUnits))
        ensureRows(&occupancy, count: preferredRow + heightUnits, columns: columns)
        guard isAreaFree(
            occupancy: occupancy,
            row: preferredRow,
            column: preferredColumn,
            widthUnits: widthUnits,
            heightUnits: heightUnits
        ) else {
            return nil
        }
        return (preferredRow, preferredColumn)
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
