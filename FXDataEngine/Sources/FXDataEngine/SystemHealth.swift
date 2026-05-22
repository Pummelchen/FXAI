import Foundation

public enum SystemHealthPosture: String, Codable, Sendable, CaseIterable {
    case unknown = "UNKNOWN"
    case healthy = "HEALTHY"
    case caution = "CAUTION"
    case degraded = "DEGRADED"
}

public struct SystemHealthComponentState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var stale: Bool

    public init(ready: Bool = false, stale: Bool = true) {
        self.ready = ready
        self.stale = stale
    }
}

public struct SystemHealthExecutionQualityState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var stale: Bool
    public var dataStale: Bool

    public init(ready: Bool = false, stale: Bool = true, dataStale: Bool = false) {
        self.ready = ready
        self.stale = stale
        self.dataStale = dataStale
    }

    public var healthComponent: SystemHealthComponentState {
        SystemHealthComponentState(ready: ready, stale: stale || dataStale)
    }
}

public struct SystemHealthInput: Sendable {
    public var news: SystemHealthComponentState?
    public var rates: SystemHealthComponentState?
    public var crossAsset: SystemHealthComponentState?
    public var microstructure: SystemHealthComponentState?
    public var executionQuality: SystemHealthExecutionQualityState?
    public var calendar: CalendarCachePairState?
    public var factor: PairFactorContext?

    public init(
        news: SystemHealthComponentState? = nil,
        rates: SystemHealthComponentState? = nil,
        crossAsset: SystemHealthComponentState? = nil,
        microstructure: SystemHealthComponentState? = nil,
        executionQuality: SystemHealthExecutionQualityState? = nil,
        calendar: CalendarCachePairState? = nil,
        factor: PairFactorContext? = nil
    ) {
        self.news = news
        self.rates = rates
        self.crossAsset = crossAsset
        self.microstructure = microstructure
        self.executionQuality = executionQuality
        self.calendar = calendar
        self.factor = factor
    }
}

public struct SystemHealthState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var generatedAt: Int64
    public var healthScore: Double
    public var degradedCount: Int
    public var newsReady: Bool
    public var newsStale: Bool
    public var ratesReady: Bool
    public var ratesStale: Bool
    public var crossReady: Bool
    public var crossStale: Bool
    public var microReady: Bool
    public var microStale: Bool
    public var executionReady: Bool
    public var executionStale: Bool
    public var calendarReady: Bool
    public var calendarStale: Bool
    public var factorReady: Bool
    public var factorStale: Bool
    public var posture: SystemHealthPosture
    public var reasonsCSV: String

    public init(
        ready: Bool = false,
        generatedAt: Int64 = 0,
        healthScore: Double = 0,
        degradedCount: Int = 0,
        newsReady: Bool = false,
        newsStale: Bool = true,
        ratesReady: Bool = false,
        ratesStale: Bool = true,
        crossReady: Bool = false,
        crossStale: Bool = true,
        microReady: Bool = false,
        microStale: Bool = true,
        executionReady: Bool = false,
        executionStale: Bool = true,
        calendarReady: Bool = false,
        calendarStale: Bool = true,
        factorReady: Bool = false,
        factorStale: Bool = true,
        posture: SystemHealthPosture = .unknown,
        reasonsCSV: String = ""
    ) {
        self.ready = ready
        self.generatedAt = max(0, generatedAt)
        self.healthScore = fxClamp(healthScore, 0.0, 1.0)
        self.degradedCount = max(0, degradedCount)
        self.newsReady = newsReady
        self.newsStale = newsStale
        self.ratesReady = ratesReady
        self.ratesStale = ratesStale
        self.crossReady = crossReady
        self.crossStale = crossStale
        self.microReady = microReady
        self.microStale = microStale
        self.executionReady = executionReady
        self.executionStale = executionStale
        self.calendarReady = calendarReady
        self.calendarStale = calendarStale
        self.factorReady = factorReady
        self.factorStale = factorStale
        self.posture = posture
        self.reasonsCSV = reasonsCSV
    }

    public static var reset: SystemHealthState {
        SystemHealthState()
    }
}

public enum SystemHealthTools {
    public static func appendReason(_ csv: inout String, _ reason: String) {
        guard !reason.isEmpty, !csv.contains(reason) else { return }
        if !csv.isEmpty {
            csv += "; "
        }
        csv += reason
    }

    public static func refresh(generatedAt: Int64, input: SystemHealthInput) -> SystemHealthState {
        var output = SystemHealthState.reset
        output.generatedAt = max(0, generatedAt)

        var scoreSum = 0.0
        var weightSum = 0.0
        let news = applyComponent(
            input.news,
            staleScore: 0.45,
            staleReason: "newspulse_stale",
            unavailableReason: "newspulse_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.newsReady = news.ready
        output.newsStale = news.stale

        let rates = applyComponent(
            input.rates,
            staleScore: 0.45,
            staleReason: "rates_stale",
            unavailableReason: "rates_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.ratesReady = rates.ready
        output.ratesStale = rates.stale

        let cross = applyComponent(
            input.crossAsset,
            staleScore: 0.45,
            staleReason: "cross_asset_stale",
            unavailableReason: "cross_asset_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.crossReady = cross.ready
        output.crossStale = cross.stale

        let micro = applyComponent(
            input.microstructure,
            staleScore: 0.40,
            staleReason: "microstructure_stale",
            unavailableReason: "microstructure_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.microReady = micro.ready
        output.microStale = micro.stale

        let execution = applyComponent(
            input.executionQuality?.healthComponent,
            staleScore: 0.45,
            staleReason: "execution_quality_stale",
            unavailableReason: "execution_quality_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.executionReady = execution.ready
        output.executionStale = execution.stale

        let calendarComponent: SystemHealthComponentState?
        if let calendar = input.calendar {
            calendarComponent = SystemHealthComponentState(ready: calendar.ready, stale: calendar.stale)
        } else {
            calendarComponent = nil
        }
        let calendar = applyComponent(
            calendarComponent,
            staleScore: 0.45,
            staleReason: "calendar_cache_stale",
            unavailableReason: "calendar_cache_unavailable",
            degradedCount: &output.degradedCount,
            reasonsCSV: &output.reasonsCSV,
            scoreSum: &scoreSum,
            weightSum: &weightSum
        )
        output.calendarReady = calendar.ready
        output.calendarStale = calendar.stale

        if let factor = input.factor {
            output.factorReady = factor.ready
            output.factorStale = factor.stale
            scoreSum += factor.ready ? 1.0 : 0.0
            weightSum += 1.0
        } else {
            output.degradedCount += 1
            weightSum += 1.0
            appendReason(&output.reasonsCSV, "factor_context_unavailable")
        }

        output.healthScore = weightSum > 1e-9 ? scoreSum / weightSum : 0.0
        if output.healthScore >= 0.85, output.degradedCount <= 1 {
            output.posture = .healthy
        } else if output.healthScore >= 0.60 {
            output.posture = .caution
        } else {
            output.posture = .degraded
        }
        output.ready = true
        return output
    }

    private static func applyComponent(
        _ component: SystemHealthComponentState?,
        staleScore: Double,
        staleReason: String,
        unavailableReason: String,
        degradedCount: inout Int,
        reasonsCSV: inout String,
        scoreSum: inout Double,
        weightSum: inout Double
    ) -> SystemHealthComponentState {
        guard let component else {
            degradedCount += 1
            weightSum += 1.0
            appendReason(&reasonsCSV, unavailableReason)
            return SystemHealthComponentState()
        }

        scoreSum += component.ready ? (component.stale ? staleScore : 1.0) : 0.0
        weightSum += 1.0
        if component.stale {
            degradedCount += 1
            appendReason(&reasonsCSV, staleReason)
        }
        return component
    }
}
