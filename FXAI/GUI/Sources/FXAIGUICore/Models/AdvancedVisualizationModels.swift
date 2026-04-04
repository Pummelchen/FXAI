import Foundation

public struct VisualizationHeatmap: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let subtitle: String
    public let rowLabels: [String]
    public let columnLabels: [String]
    public let values: [[Double]]

    public init(
        title: String,
        subtitle: String,
        rowLabels: [String],
        columnLabels: [String],
        values: [[Double]]
    ) {
        self.id = title
        self.title = title
        self.subtitle = subtitle
        self.rowLabels = rowLabels
        self.columnLabels = columnLabels
        self.values = values
    }

    public var valueRange: ClosedRange<Double> {
        let flattened = values.flatMap { row in row.filter { !$0.isNaN } }
        guard let minimum = flattened.min(), let maximum = flattened.max() else {
            return 0...1
        }
        if minimum == maximum {
            return minimum...(minimum == 0 ? 1 : minimum)
        }
        return minimum...maximum
    }

    public func formattedValue(row: Int, column: Int) -> String {
        guard values.indices.contains(row), values[row].indices.contains(column) else {
            return "—"
        }
        let value = values[row][column]
        if value.isNaN {
            return "—"
        }
        return String(format: "%.3f", value)
    }
}

public struct VisualizationSeriesPoint: Identifiable, Hashable, Sendable {
    public let id: String
    public let label: String
    public let value: Double
    public let secondaryValue: Double?

    public init(label: String, value: Double, secondaryValue: Double? = nil) {
        self.id = label
        self.label = label
        self.value = value
        self.secondaryValue = secondaryValue
    }
}

public struct VisualizationTimelineEvent: Identifiable, Hashable, Sendable {
    public let id: String
    public let category: String
    public let title: String
    public let detail: String
    public let date: Date
    public let score: Double?

    public init(category: String, title: String, detail: String, date: Date, score: Double? = nil) {
        self.id = "\(category)::\(title)::\(date.timeIntervalSince1970)"
        self.category = category
        self.title = title
        self.detail = detail
        self.date = date
        self.score = score
    }
}

public struct SymbolVisualizationDetail: Identifiable, Hashable, Sendable {
    public let id: String
    public let symbol: String
    public let worldSessionScales: [VisualizationSeriesPoint]
    public let worldStressMetrics: [VisualizationSeriesPoint]
    public let familyWeights: [VisualizationSeriesPoint]
    public let featureWeights: [VisualizationSeriesPoint]
    public let pluginWeights: [VisualizationSeriesPoint]
    public let artifactDiffHeatmap: VisualizationHeatmap?
    public let timeline: [VisualizationTimelineEvent]
    public let weakScenarios: [String]

    public init(
        symbol: String,
        worldSessionScales: [VisualizationSeriesPoint],
        worldStressMetrics: [VisualizationSeriesPoint],
        familyWeights: [VisualizationSeriesPoint],
        featureWeights: [VisualizationSeriesPoint],
        pluginWeights: [VisualizationSeriesPoint],
        artifactDiffHeatmap: VisualizationHeatmap?,
        timeline: [VisualizationTimelineEvent],
        weakScenarios: [String]
    ) {
        self.id = symbol
        self.symbol = symbol
        self.worldSessionScales = worldSessionScales
        self.worldStressMetrics = worldStressMetrics
        self.familyWeights = familyWeights
        self.featureWeights = featureWeights
        self.pluginWeights = pluginWeights
        self.artifactDiffHeatmap = artifactDiffHeatmap
        self.timeline = timeline
        self.weakScenarios = weakScenarios
    }
}

public struct AdvancedVisualizationSnapshot: Sendable {
    public let generatedAt: Date
    public let profileName: String?
    public let familyStressHeatmap: VisualizationHeatmap?
    public let symbolDetails: [SymbolVisualizationDetail]
    public let globalTimeline: [VisualizationTimelineEvent]

    public var symbols: [String] {
        symbolDetails.map(\.symbol)
    }
}
