import Foundation

public enum CalendarCacheConstants {
    public static let maxReasons = 6
    public static let defaultNewsPulseFreshnessMaxSeconds = 360
}

public enum CalendarTradeGate: String, Codable, Sendable, CaseIterable {
    case unknown = "UNKNOWN"
    case safe = "SAFE"
    case caution = "CAUTION"
    case block = "BLOCK"
}

public enum CalendarEventClass: Int, Codable, Sendable, CaseIterable {
    case unknown = 0
    case rates = 1
    case inflation = 2
    case labor = 3
    case growth = 4
    case speech = 5
}

public struct CalendarCacheState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var ok: Bool
    public var stale: Bool
    public var lastUpdateTradeServer: Int64
    public var collectorGeneratedAt: Int64
    public var tradeServerOffsetSeconds: Int
    public var recordCount: Int
    public var lastError: String

    public init(
        ready: Bool = false,
        ok: Bool = false,
        stale: Bool = true,
        lastUpdateTradeServer: Int64 = 0,
        collectorGeneratedAt: Int64 = 0,
        tradeServerOffsetSeconds: Int = 0,
        recordCount: Int = 0,
        lastError: String = ""
    ) {
        self.ready = ready
        self.ok = ok
        self.stale = stale
        self.lastUpdateTradeServer = max(0, lastUpdateTradeServer)
        self.collectorGeneratedAt = max(0, collectorGeneratedAt)
        self.tradeServerOffsetSeconds = tradeServerOffsetSeconds
        self.recordCount = max(0, recordCount)
        self.lastError = lastError
    }

    public static var reset: CalendarCacheState {
        CalendarCacheState()
    }
}

public struct CalendarCachePairState: Codable, Hashable, Sendable {
    public var ready: Bool
    public var stale: Bool
    public var generatedAt: Int64
    public var tradeServerOffsetSeconds: Int
    public var nextEventETAMinutes: Int
    public var tradeGate: CalendarTradeGate
    public var eventRiskScore: Double
    public var cautionLotScale: Double
    public var cautionEnterProbabilityBuffer: Double
    public var reasons: [String]

    public init(
        ready: Bool = false,
        stale: Bool = true,
        generatedAt: Int64 = 0,
        tradeServerOffsetSeconds: Int = 0,
        nextEventETAMinutes: Int = -1,
        tradeGate: CalendarTradeGate = .unknown,
        eventRiskScore: Double = 0,
        cautionLotScale: Double = 1,
        cautionEnterProbabilityBuffer: Double = 0,
        reasons: [String] = []
    ) {
        self.ready = ready
        self.stale = stale
        self.generatedAt = max(0, generatedAt)
        self.tradeServerOffsetSeconds = tradeServerOffsetSeconds
        self.nextEventETAMinutes = nextEventETAMinutes
        self.tradeGate = tradeGate
        self.eventRiskScore = fxClamp(eventRiskScore, 0.0, 1.0)
        self.cautionLotScale = fxClamp(cautionLotScale, 0.0, 1.0)
        self.cautionEnterProbabilityBuffer = fxClamp(cautionEnterProbabilityBuffer, 0.0, 1.0)
        self.reasons = Self.uniqueReasons(reasons)
    }

    public var reasonCount: Int {
        reasons.count
    }

    public var reasonsCSV: String {
        reasons.filter { !$0.isEmpty }.joined(separator: "; ")
    }

    public mutating func appendReason(_ reason: String) {
        let value = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty,
              !reasons.contains(value),
              reasons.count < CalendarCacheConstants.maxReasons else {
            return
        }
        reasons.append(value)
    }

    public static var reset: CalendarCachePairState {
        CalendarCachePairState()
    }

    private static func uniqueReasons(_ input: [String]) -> [String] {
        var output: [String] = []
        output.reserveCapacity(min(input.count, CalendarCacheConstants.maxReasons))
        for raw in input {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty,
                  !output.contains(value),
                  output.count < CalendarCacheConstants.maxReasons else {
                continue
            }
            output.append(value)
        }
        return output
    }
}

public struct CalendarCacheFeedRecord: Codable, Hashable, Sendable {
    public var eventID: UInt64
    public var eventKey: String
    public var title: String
    public var countryCode: String
    public var countryName: String
    public var currency: String
    public var eventTimeTradeServer: Int64
    public var importance: Int
    public var actual: Double?
    public var forecast: Double?
    public var previous: Double?
    public var revisedPrevious: Double?
    public var surpriseProxy: Double?
    public var collectorSeenTradeServer: Int64
    public var changeID: UInt64
    public var eventTimeUTC: Int64
    public var collectorSeenUTC: Int64
    public var tradeServerOffsetSeconds: Int

    public init(
        eventID: UInt64 = 0,
        eventKey: String = "",
        title: String = "",
        countryCode: String = "",
        countryName: String = "",
        currency: String,
        eventTimeTradeServer: Int64,
        importance: Int,
        actual: Double? = nil,
        forecast: Double? = nil,
        previous: Double? = nil,
        revisedPrevious: Double? = nil,
        surpriseProxy: Double? = nil,
        collectorSeenTradeServer: Int64 = 0,
        changeID: UInt64 = 0,
        eventTimeUTC: Int64? = nil,
        collectorSeenUTC: Int64? = nil,
        tradeServerOffsetSeconds: Int = 0
    ) {
        self.eventID = eventID
        self.eventKey = eventKey
        self.title = title
        self.countryCode = countryCode
        self.countryName = countryName
        self.currency = MacroEventTools.normalizedCurrencyToken(currency)
        self.eventTimeTradeServer = max(0, eventTimeTradeServer)
        self.importance = max(0, importance)
        self.actual = actual?.isFinite == true ? actual : nil
        self.forecast = forecast?.isFinite == true ? forecast : nil
        self.previous = previous?.isFinite == true ? previous : nil
        self.revisedPrevious = revisedPrevious?.isFinite == true ? revisedPrevious : nil
        self.surpriseProxy = surpriseProxy?.isFinite == true ? surpriseProxy : nil
        self.collectorSeenTradeServer = max(0, collectorSeenTradeServer)
        self.changeID = changeID
        self.eventTimeUTC = max(0, eventTimeUTC ?? eventTimeTradeServer)
        self.collectorSeenUTC = max(0, collectorSeenUTC ?? collectorSeenTradeServer)
        self.tradeServerOffsetSeconds = tradeServerOffsetSeconds
    }
}

public struct CalendarCacheSnapshot: Sendable {
    public var state: CalendarCacheState
    public var feedRecords: [CalendarCacheFeedRecord]

    public init(state: CalendarCacheState, feedRecords: [CalendarCacheFeedRecord] = []) {
        self.state = state
        self.feedRecords = feedRecords
    }

    public static func parse(stateTSV: String, feedTSV: String = "") -> CalendarCacheSnapshot {
        CalendarCacheSnapshot(
            state: CalendarCacheTools.parseStateTSV(stateTSV),
            feedRecords: CalendarCacheTools.parseFeedTSV(feedTSV)
        )
    }

    public static func load(stateURL: URL, feedURL: URL? = nil, encoding: String.Encoding = .utf8) throws -> CalendarCacheSnapshot {
        let stateText = try String(contentsOf: stateURL, encoding: encoding)
        let feedText = try feedURL.map { try String(contentsOf: $0, encoding: encoding) } ?? ""
        return parse(stateTSV: stateText, feedTSV: feedText)
    }

    public func pairState(
        symbol: String,
        nowTradeServer: Int64,
        newsPulseFreshnessMaxSeconds: Int = CalendarCacheConstants.defaultNewsPulseFreshnessMaxSeconds
    ) -> CalendarCachePairState {
        CalendarCacheTools.pairState(
            symbol: symbol,
            state: state,
            feedRecords: feedRecords,
            nowTradeServer: nowTradeServer,
            newsPulseFreshnessMaxSeconds: newsPulseFreshnessMaxSeconds
        )
    }
}

public enum CalendarCacheTools {
    public static func parseStateTSV(_ text: String) -> CalendarCacheState {
        var state = CalendarCacheState.reset
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = String(rawLine).trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 2 else { continue }
            let key = MacroEventTools.normalizedToken(parts[0])
            let value = parts[1]
            switch key {
            case "ok":
                state.ok = parseInt(value) != 0
            case "stale":
                state.stale = parseInt(value) != 0
            case "last_update_trade_server":
                state.lastUpdateTradeServer = parseTime(value)
            case "collector_generated_at":
                state.collectorGeneratedAt = parseTime(value)
            case "trade_server_offset_sec":
                state.tradeServerOffsetSeconds = parseInt(value)
            case "record_count":
                state.recordCount = max(0, parseInt(value))
            case "last_error":
                state.lastError = value
            default:
                continue
            }
        }
        state.ready = state.ok || state.recordCount > 0 || !state.lastError.isEmpty
        return state
    }

    public static func parseFeedTSV(_ text: String) -> [CalendarCacheFeedRecord] {
        var records: [CalendarCacheFeedRecord] = []
        var headerSeen = false
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = String(rawLine).trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
            guard !line.isEmpty else { continue }
            if !headerSeen {
                headerSeen = true
                continue
            }
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 8 else { continue }
            let hasExtendedFields = parts.count >= 20
            let collectorSeenTradeServer = parts.count > 13 ? Int64(parseInt(parts[13])) : 0
            let defaultEventTime = Int64(parseInt(parts[6]))
            records.append(
                CalendarCacheFeedRecord(
                    eventID: UInt64(max(0, parseInt(parts[0]))),
                    eventKey: field(parts, 1),
                    title: field(parts, 2),
                    countryCode: field(parts, 3),
                    countryName: field(parts, 4),
                    currency: field(parts, 5),
                    eventTimeTradeServer: defaultEventTime,
                    importance: parseInt(parts[7]),
                    actual: parseOptionalDouble(field(parts, 8)),
                    forecast: parseOptionalDouble(field(parts, 9)),
                    previous: parseOptionalDouble(field(parts, 10)),
                    revisedPrevious: parseOptionalDouble(field(parts, 11)),
                    surpriseProxy: parseOptionalDouble(field(parts, 12)),
                    collectorSeenTradeServer: hasExtendedFields ? parseTime(field(parts, 16)) : collectorSeenTradeServer,
                    changeID: UInt64(max(0, parseInt(field(parts, 14)))),
                    eventTimeUTC: hasExtendedFields ? Int64(parseInt(field(parts, 17))) : defaultEventTime,
                    collectorSeenUTC: hasExtendedFields ? Int64(parseInt(field(parts, 18))) : collectorSeenTradeServer,
                    tradeServerOffsetSeconds: hasExtendedFields ? parseInt(field(parts, 19)) : 0
                )
            )
        }
        return records
    }

    public static func eventClassFromTitle(_ rawTitle: String) -> Int {
        let title = rawTitle.lowercased()
        if title.contains("rate") || title.contains("fomc")
            || title.contains("ecb") || title.contains("boe")
            || title.contains("boj") || title.contains("rba")
            || title.contains("rbnz") || title.contains("boc")
            || title.contains("snb") {
            return CalendarEventClass.rates.rawValue
        }
        if title.contains("cpi") || title.contains("ppi")
            || title.contains("pce") || title.contains("inflation")
            || title.contains("price") {
            return CalendarEventClass.inflation.rawValue
        }
        if title.contains("payroll") || title.contains("employment")
            || title.contains("job") || title.contains("wage")
            || title.contains("unemployment") {
            return CalendarEventClass.labor.rawValue
        }
        if title.contains("gdp") || title.contains("pmi")
            || title.contains("retail") || title.contains("production")
            || title.contains("confidence") || title.contains("sentiment") {
            return CalendarEventClass.growth.rawValue
        }
        if title.contains("speech") || title.contains("testimony") {
            return CalendarEventClass.speech.rawValue
        }
        return CalendarEventClass.unknown.rawValue
    }

    public static func importanceWeight(_ importance: Int) -> Double {
        if importance >= 3 { return 1.0 }
        if importance == 2 { return 0.60 }
        if importance == 1 { return 0.30 }
        return 0.10
    }

    public static func parseTime(_ rawValue: String) -> Int64 {
        MacroEventTools.parseEventTimeUTC(rawValue) ?? 0
    }

    public static func eventAffectsSymbol(symbol rawSymbol: String, currency rawCurrency: String) -> Bool {
        let currency = MacroEventTools.normalizedCurrencyToken(rawCurrency)
        let (base, quote) = symbolLegs(rawSymbol)
        guard base.count == 3, quote.count == 3, currency.count == 3 else { return false }
        return base == currency || quote == currency
    }

    public static func symbolLegs(_ rawSymbol: String) -> (base: String, quote: String) {
        let symbol = MacroEventTools.normalizedToken(rawSymbol).uppercased(with: Locale(identifier: "en_US_POSIX"))
        var letters = ""
        letters.reserveCapacity(6)
        for scalar in symbol.unicodeScalars {
            guard scalar.value >= 65, scalar.value <= 90 else { continue }
            letters.append(Character(scalar))
            if letters.count >= 6 { break }
        }
        guard letters.count >= 6 else { return ("", "") }
        return (String(letters.prefix(3)), String(letters.dropFirst(3).prefix(3)))
    }

    public static func pairState(
        symbol: String,
        state: CalendarCacheState,
        feedRecords: [CalendarCacheFeedRecord],
        nowTradeServer: Int64,
        newsPulseFreshnessMaxSeconds: Int = CalendarCacheConstants.defaultNewsPulseFreshnessMaxSeconds
    ) -> CalendarCachePairState {
        guard state.ready else { return .reset }

        var output = CalendarCachePairState(
            ready: state.ready,
            stale: state.stale,
            generatedAt: state.lastUpdateTradeServer,
            tradeServerOffsetSeconds: state.tradeServerOffsetSeconds
        )
        let now = nowTradeServer > 0 ? nowTradeServer : state.lastUpdateTradeServer
        var eventCount = 0

        for record in feedRecords {
            guard eventAffectsSymbol(symbol: symbol, currency: record.currency) else { continue }

            let etaMinutes = Int((record.eventTimeTradeServer - now) / 60)
            if etaMinutes < output.nextEventETAMinutes || output.nextEventETAMinutes < 0 {
                output.nextEventETAMinutes = etaMinutes
            }
            guard abs(etaMinutes) <= 360 else { continue }

            let eventClass = eventClassFromTitle(record.title)
            var eventWeight = importanceWeight(record.importance)
            if eventClass == CalendarEventClass.rates.rawValue {
                eventWeight *= 1.20
            } else if eventClass == CalendarEventClass.inflation.rawValue || eventClass == CalendarEventClass.labor.rawValue {
                eventWeight *= 1.05
            } else if eventClass == CalendarEventClass.speech.rawValue {
                eventWeight *= 0.65
            }

            if etaMinutes >= -20, etaMinutes <= 30 {
                output.tradeGate = .block
                output.eventRiskScore = max(output.eventRiskScore, fxClamp(eventWeight, 0.0, 1.0))
                output.appendReason("calendar_blackout")
            } else if etaMinutes >= -90, etaMinutes <= 90 {
                if output.tradeGate != .block {
                    output.tradeGate = .caution
                }
                output.eventRiskScore = max(output.eventRiskScore, fxClamp(0.70 * eventWeight, 0.0, 1.0))
                output.appendReason("calendar_caution")
            }

            if eventClass == CalendarEventClass.rates.rawValue {
                output.appendReason("calendar_central_bank")
            } else if eventClass == CalendarEventClass.inflation.rawValue {
                output.appendReason("calendar_inflation")
            } else if eventClass == CalendarEventClass.labor.rawValue {
                output.appendReason("calendar_labor")
            }

            eventCount += 1
        }

        if output.nextEventETAMinutes < 0, output.tradeGate == .unknown {
            output.tradeGate = .safe
        }

        switch output.tradeGate {
        case .block:
            output.cautionLotScale = 0.0
            output.cautionEnterProbabilityBuffer = 0.10
        case .caution:
            output.cautionLotScale = 0.55
            output.cautionEnterProbabilityBuffer = 0.05
        case .safe, .unknown:
            output.cautionLotScale = 1.0
            output.cautionEnterProbabilityBuffer = 0.0
            output.eventRiskScore = fxClamp(output.eventRiskScore, 0.0, 0.35)
        }

        let effectiveFreshnessMax = max(newsPulseFreshnessMaxSeconds, 600)
        if state.lastUpdateTradeServer > 0,
           now > state.lastUpdateTradeServer,
           now - state.lastUpdateTradeServer > Int64(effectiveFreshnessMax) {
            output.stale = true
        }

        if eventCount <= 0, !state.ok, !state.lastError.isEmpty {
            output.appendReason("calendar_state_error")
        }
        return output
    }

    private static func parseInt(_ rawValue: String) -> Int {
        MacroEventTools.parseInt(rawValue)
    }

    private static func parseOptionalDouble(_ rawValue: String) -> Double? {
        let token = MacroEventTools.normalizedToken(rawValue)
        guard !token.isEmpty else { return nil }
        let value = MacroEventTools.parseDouble(token)
        return value.isFinite ? value : nil
    }

    private static func field(_ fields: [String], _ index: Int) -> String {
        index < fields.count ? fields[index] : ""
    }
}
