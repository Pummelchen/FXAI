import Foundation

public struct NewsPulseReplayRecord: Codable, Hashable, Sendable {
    public var pairID: String
    public var observedAtUTC: Int64
    public var eventETAMinutes: Int
    public var newsRiskScore: Double
    public var newsPressure: Double
    public var stale: Bool
    public var tradeGate: String

    public init(
        pairID: String,
        observedAtUTC: Int64,
        eventETAMinutes: Int,
        newsRiskScore: Double,
        newsPressure: Double,
        stale: Bool,
        tradeGate: String
    ) {
        self.pairID = pairID.uppercased()
        self.observedAtUTC = max(0, observedAtUTC)
        self.eventETAMinutes = eventETAMinutes
        self.newsRiskScore = newsRiskScore.isFinite ? newsRiskScore : 0.0
        self.newsPressure = newsPressure.isFinite ? newsPressure : 0.0
        self.stale = stale
        self.tradeGate = tradeGate.isEmpty ? "UNKNOWN" : tradeGate
    }
}

public enum NewsPulseReplayTools {
    public static func parseTimeline(
        tsv: String,
        symbol: String,
        symbolMap: [NewsPulseSymbolMapEntry] = []
    ) -> [NewsPulseReplayRecord] {
        let pairID = NewsPulseTools.pairID(symbol: symbol, symbolMap: symbolMap)
        guard pairID.count == 6 else { return [] }

        var records: [NewsPulseReplayRecord] = []
        for line in tsv.components(separatedBy: .newlines) where !line.isEmpty {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 8 else { continue }
            let rawPairID = String(parts[0])
            guard rawPairID != "pair_id", rawPairID.uppercased() == pairID else { continue }

            records.append(
                NewsPulseReplayRecord(
                    pairID: rawPairID,
                    observedAtUTC: Int64(parts[2]) ?? 0,
                    eventETAMinutes: parts[7].isEmpty ? -1 : (Int(parts[7]) ?? 0),
                    newsRiskScore: Double(parts[4]) ?? 0.0,
                    newsPressure: Double(parts[5]) ?? 0.0,
                    stale: (Int(parts[6]) ?? 0) != 0,
                    tradeGate: String(parts[3])
                )
            )
        }
        return records
    }

    public static func parseTimeline(
        tsv: String,
        symbol: String,
        symbolMapTSV: String?
    ) -> [NewsPulseReplayRecord] {
        parseTimeline(
            tsv: tsv,
            symbol: symbol,
            symbolMap: symbolMapTSV.map(NewsPulseTools.parseSymbolMap(tsv:)) ?? []
        )
    }

    public static func replayIndex(records: [NewsPulseReplayRecord], queryTimeUTC: Int64) -> Int? {
        guard !records.isEmpty else { return nil }
        var best: Int?
        for index in records.indices {
            if records[index].observedAtUTC <= queryTimeUTC {
                best = index
            } else {
                break
            }
        }
        return best ?? records.startIndex
    }

    public static func state(
        records: [NewsPulseReplayRecord],
        queryTimeUTC: Int64
    ) -> NewsPulseReplayRecord? {
        guard let index = replayIndex(records: records, queryTimeUTC: queryTimeUTC),
              records.indices.contains(index) else {
            return nil
        }
        return records[index]
    }

    public static func windowScore(
        records: [NewsPulseReplayRecord],
        sampleTimesUTC: [Int64],
        start: Int,
        bars: Int
    ) -> Double {
        guard bars > 0,
              start >= 0,
              start + bars - 1 < sampleTimesUTC.count else {
            return 0.0
        }

        let end = start + bars - 1
        let checkpoints = [
            start,
            start + bars / 3,
            start + (2 * bars) / 3,
            end,
        ]

        var best = 0.0
        for rawIndex in checkpoints {
            let index = min(max(rawIndex, start), end)
            guard let replayState = state(records: records, queryTimeUTC: sampleTimesUTC[index]) else {
                continue
            }

            var score = fxClamp(replayState.newsRiskScore, 0.0, 1.0)
            if replayState.stale {
                score = max(score, 0.90)
            }
            if replayState.tradeGate == "BLOCK" {
                score = max(score, 0.98)
            } else if replayState.tradeGate == "CAUTION" {
                score = max(score, 0.72)
            }
            if replayState.eventETAMinutes >= 0, replayState.eventETAMinutes <= 15 {
                score = max(score, 0.84)
            }
            best = max(best, score)
        }
        return fxClamp(best, 0.0, 1.0)
    }

    public static func windowScore(
        symbol: String,
        sampleTimesUTC: [Int64],
        start: Int,
        bars: Int,
        timelineTSV: String,
        symbolMapTSV: String? = nil
    ) -> Double {
        let records = parseTimeline(tsv: timelineTSV, symbol: symbol, symbolMapTSV: symbolMapTSV)
        return windowScore(records: records, sampleTimesUTC: sampleTimesUTC, start: start, bars: bars)
    }
}
