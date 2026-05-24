import AppCore
import BacktestCore
import ClickHouse
import Domain
import Foundation
import Ingestion

public enum SineWaveDatabaseSyncError: Error, CustomStringConvertible, Sendable {
    case invalidChunkRows(Int)
    case invalidRange(UtcSecond, UtcSecond)
    case invalidCoverageRow(String)

    public var description: String {
        switch self {
        case .invalidChunkRows(let rows):
            return "SineTest sync chunkRows must be positive, got \(rows)."
        case .invalidRange(let start, let end):
            return "SineTest sync range is invalid or not minute-aligned: \(start.rawValue)..<\(end.rawValue)."
        case .invalidCoverageRow(let row):
            return "SineTest sync received invalid verified coverage row: \(row)"
        }
    }
}

public struct SineWaveDatabaseSyncResult: Sendable, Equatable {
    public let targetStartUtc: UtcSecond
    public let targetEndExclusiveUtc: UtcSecond
    public let chunksInserted: Int
    public let rowsInserted: Int
    public let alreadyCurrent: Bool

    public init(
        targetStartUtc: UtcSecond,
        targetEndExclusiveUtc: UtcSecond,
        chunksInserted: Int,
        rowsInserted: Int,
        alreadyCurrent: Bool
    ) {
        self.targetStartUtc = targetStartUtc
        self.targetEndExclusiveUtc = targetEndExclusiveUtc
        self.chunksInserted = chunksInserted
        self.rowsInserted = rowsInserted
        self.alreadyCurrent = alreadyCurrent
    }
}

public struct SineWaveDatabaseSyncAgent: Sendable {
    public static let coverageVerificationMethod = "synthetic_sinetest_min_0_001_v1"

    private let clickHouse: ClickHouseClientProtocol
    private let database: String
    private let brokerSourceId: BrokerSourceId
    private let chunkRows: Int
    private let certifyInsertedRanges: Bool

    public init(
        clickHouse: ClickHouseClientProtocol,
        database: String,
        brokerSourceId: BrokerSourceId = SineTestSecurity.defaultBrokerSourceId,
        chunkRows: Int = 50_000,
        certifyInsertedRanges: Bool = true
    ) throws {
        guard chunkRows > 0 else {
            throw SineWaveDatabaseSyncError.invalidChunkRows(chunkRows)
        }
        self.clickHouse = clickHouse
        self.database = database
        self.brokerSourceId = brokerSourceId
        self.chunkRows = chunkRows
        self.certifyInsertedRanges = certifyInsertedRanges
    }

    public func syncThroughRuntimeNow(now: Date = Date()) async throws -> SineWaveDatabaseSyncResult {
        try await sync(utcEndExclusive: Self.runtimeEndExclusive(now: now))
    }

    public func sync(utcEndExclusive: UtcSecond) async throws -> SineWaveDatabaseSyncResult {
        try validateRange(start: SineTestSecurity.genesisUtc, end: utcEndExclusive)

        let targetStart = SineTestSecurity.genesisUtc
        let targetEnd = utcEndExclusive
        var intervals = mergeIntervals(try await loadCoverageIntervals(start: targetStart, end: targetEnd))
        if firstGap(from: targetStart.rawValue, to: targetEnd.rawValue, coveredBy: intervals) == nil {
            let hasBoundaryRows = try await canonicalBoundaryRowsArePresent(start: targetStart, end: targetEnd)
            if !hasBoundaryRows {
                intervals.removeAll()
            }
        }
        var insertedChunks = 0
        var insertedRows = 0

        while let gap = firstGap(from: targetStart.rawValue, to: targetEnd.rawValue, coveredBy: intervals) {
            try Task.checkCancellation()
            let chunkStart = UtcSecond(rawValue: gap.start)
            let maxChunkEnd = gap.start + (Int64(chunkRows) * Timeframe.m1.seconds)
            let chunkEnd = UtcSecond(rawValue: min(gap.end, maxChunkEnd))
            let rows = try await replaceChunk(start: chunkStart, end: chunkEnd)
            insertedChunks += 1
            insertedRows += rows
            intervals.append(CoverageInterval(start: chunkStart.rawValue, end: chunkEnd.rawValue))
            intervals = mergeIntervals(intervals)
        }

        return SineWaveDatabaseSyncResult(
            targetStartUtc: targetStart,
            targetEndExclusiveUtc: targetEnd,
            chunksInserted: insertedChunks,
            rowsInserted: insertedRows,
            alreadyCurrent: insertedChunks == 0
        )
    }

    public static func runtimeEndExclusive(now: Date) -> UtcSecond {
        let seconds = Int64(now.timeIntervalSince1970.rounded(.down))
        let minuteStart = floorDiv(seconds, Timeframe.m1.seconds) * Timeframe.m1.seconds
        return UtcSecond(rawValue: minuteStart + Timeframe.m1.seconds)
    }

    private func replaceChunk(start: UtcSecond, end: UtcSecond) async throws -> Int {
        let bars = try makeValidatedBars(start: start, end: end)
        guard !bars.isEmpty else { return 0 }

        let insertBuilder = ClickHouseInsertBuilder(database: database)
        let mt5Start = MT5ServerSecond(rawValue: start.rawValue)
        let mt5End = MT5ServerSecond(rawValue: end.rawValue)
        let first = bars[0]
        let batchId = first.batchId
        let sourceSHA256 = syntheticSourceSHA256(bars: bars, start: mt5Start, end: mt5End)
        let offsetSHA256 = syntheticOffsetAuthoritySHA256(start: mt5Start, end: mt5End, rowCount: bars.count)

        if try await canonicalRowsExist(start: start, end: end) {
            _ = try await clickHouse.execute(try insertBuilder.canonicalRangeDelete(
                bars,
                mt5Start: mt5Start,
                mt5EndExclusive: mt5End
            ))
        }
        _ = try await clickHouse.execute(try insertBuilder.canonicalBarsInsert(bars))
        let verification = try await CanonicalInsertVerifier(
            clickHouse: clickHouse,
            insertBuilder: insertBuilder
        ).verify(bars, mt5Start: mt5Start, mt5EndExclusive: mt5End)

        try await IngestAuditStore(clickHouse: clickHouse, database: database).recordVerifiedCoverage(VerifiedCoverageRecord(
            brokerSourceId: brokerSourceId,
            sourceOrigin: SineTestSecurity.sourceOrigin,
            logicalSymbol: SineTestSecurity.logicalSymbol,
            mt5Symbol: SineTestSecurity.providerSymbol,
            timeframe: .m1,
            mt5Start: mt5Start,
            mt5EndExclusive: mt5End,
            utcStart: start,
            utcEndExclusive: end,
            sourceBarCount: bars.count,
            canonicalRowCount: verification.rowCount,
            sourceHash: sourceSHA256.rawValue,
            hashSchemaVersion: ChunkHashSchemaVersion.current,
            mt5SourceSHA256: sourceSHA256,
            canonicalReadbackSHA256: verification.canonicalReadbackSHA256,
            offsetAuthoritySHA256: offsetSHA256,
            verificationMethod: Self.coverageVerificationMethod,
            batchId: batchId,
            verifiedAtUtc: utcNow()
        ))

        if certifyInsertedRanges {
            try await DataCertificateStore(clickHouse: clickHouse, database: database).certify(
                brokerSourceId: brokerSourceId,
                sourceOrigin: SineTestSecurity.sourceOrigin,
                logicalSymbol: SineTestSecurity.logicalSymbol,
                utcStart: start,
                utcEndExclusive: end
            )
        }

        try await ClickHouseCheckpointStore(
            client: clickHouse,
            insertBuilder: insertBuilder,
            database: database
        ).save(IngestState(
            brokerSourceId: brokerSourceId,
            sourceOrigin: SineTestSecurity.sourceOrigin,
            logicalSymbol: SineTestSecurity.logicalSymbol,
            mt5Symbol: SineTestSecurity.providerSymbol,
            oldestMT5ServerTime: MT5ServerSecond(rawValue: SineTestSecurity.genesisUtc.rawValue),
            latestIngestedClosedMT5ServerTime: MT5ServerSecond(rawValue: end.rawValue - Timeframe.m1.seconds),
            latestIngestedClosedUtcTime: UtcSecond(rawValue: end.rawValue - Timeframe.m1.seconds),
            status: .live,
            lastBatchId: batchId,
            updatedAtUtc: utcNow()
        ))

        return bars.count
    }

    private func canonicalRowsExist(start: UtcSecond, end: UtcSecond) async throws -> Bool {
        let body = try await clickHouse.execute(.select("""
        SELECT 1
        FROM \(database).ohlc_m1_canonical
        WHERE broker_source_id = '\(SQLText.literal(brokerSourceId.rawValue))'
          AND source_origin = '\(SQLText.literal(SineTestSecurity.sourceOrigin.rawValue))'
          AND logical_symbol = '\(SQLText.literal(SineTestSecurity.logicalSymbol.rawValue))'
          AND timeframe = '\(SQLText.literal(Timeframe.m1.rawValue))'
          AND ts_utc >= \(start.rawValue)
          AND ts_utc < \(end.rawValue)
        LIMIT 1
        FORMAT TabSeparated
        """))
        return !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func makeValidatedBars(start: UtcSecond, end: UtcSecond) throws -> [ValidatedBar] {
        let series = try SineWaveAgent.generateM1Ohlc(
            brokerSourceId: brokerSourceId,
            utcStartInclusive: start,
            utcEndExclusive: end,
            digits: SineTestSecurity.digits
        )
        let batchId = BatchId.deterministic(
            brokerSourceId: brokerSourceId,
            sourceOrigin: SineTestSecurity.sourceOrigin,
            logicalSymbol: SineTestSecurity.logicalSymbol,
            start: MT5ServerSecond(rawValue: start.rawValue),
            end: MT5ServerSecond(rawValue: end.rawValue)
        )
        let ingestedAt = utcNow()
        let digits = SineTestSecurity.digits
        var bars: [ValidatedBar] = []
        bars.reserveCapacity(series.count)
        for index in 0..<series.count {
            let utc = UtcSecond(rawValue: series.utcTimestamps[index])
            bars.append(ValidatedBar(
                sourceOrigin: SineTestSecurity.sourceOrigin,
                brokerSourceId: brokerSourceId,
                logicalSymbol: SineTestSecurity.logicalSymbol,
                mt5Symbol: SineTestSecurity.providerSymbol,
                timeframe: .m1,
                mt5ServerTime: MT5ServerSecond(rawValue: utc.rawValue),
                utcTime: utc,
                serverUtcOffset: OffsetSeconds(rawValue: 0),
                offsetSource: .configured,
                offsetConfidence: .verified,
                open: PriceScaled(rawValue: series.open[index], digits: digits),
                high: PriceScaled(rawValue: series.high[index], digits: digits),
                low: PriceScaled(rawValue: series.low[index], digits: digits),
                close: PriceScaled(rawValue: series.close[index], digits: digits),
                volume: M1Volume(rawValue: series.volume[index]),
                digits: digits,
                batchId: batchId,
                sourceStatus: .closedM1Bar,
                ingestedAtUtc: ingestedAt
            ))
        }
        return bars
    }

    private func loadCoverageIntervals(start: UtcSecond, end: UtcSecond) async throws -> [CoverageInterval] {
        let body = try await clickHouse.execute(.select("""
        SELECT utc_range_start, utc_range_end_exclusive
        FROM \(database).ohlc_m1_verified_coverage
        WHERE broker_source_id = '\(SQLText.literal(brokerSourceId.rawValue))'
          AND source_origin = '\(SQLText.literal(SineTestSecurity.sourceOrigin.rawValue))'
          AND logical_symbol = '\(SQLText.literal(SineTestSecurity.logicalSymbol.rawValue))'
          AND timeframe = 'M1'
          AND hash_schema_version = '\(SQLText.literal(ChunkHashSchemaVersion.current))'
          AND verification_method = '\(SQLText.literal(Self.coverageVerificationMethod))'
          AND length(mt5_source_sha256) = 64
          AND length(canonical_readback_sha256) = 64
          AND length(offset_authority_sha256) = 64
          AND utc_range_end_exclusive > \(start.rawValue)
          AND utc_range_start < \(end.rawValue)
        FORMAT TabSeparated
        """))
        return try body
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { line in
                let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
                guard fields.count == 2,
                      let rowStart = Int64(fields[0]),
                      let rowEnd = Int64(fields[1]),
                      rowStart < rowEnd else {
                    throw SineWaveDatabaseSyncError.invalidCoverageRow(String(line))
                }
                return CoverageInterval(
                    start: max(start.rawValue, rowStart),
                    end: min(end.rawValue, rowEnd)
                )
            }
    }

    private func canonicalBoundaryRowsArePresent(start: UtcSecond, end: UtcSecond) async throws -> Bool {
        let lastUtc = end.rawValue - Timeframe.m1.seconds
        let expectedRows = start.rawValue == lastUtc ? 1 : 2
        let body = try await clickHouse.execute(.select("""
        SELECT
            count(),
            uniqExact(ts_utc),
            countIf(
                mt5_symbol != '\(SQLText.literal(SineTestSecurity.providerSymbol.rawValue))'
                OR timeframe != 'M1'
                OR mt5_server_ts_raw != ts_utc
                OR server_utc_offset_seconds != 0
                OR offset_source != '\(SQLText.literal(OffsetSource.configured.rawValue))'
                OR offset_confidence != '\(SQLText.literal(OffsetConfidence.verified.rawValue))'
                OR source_status != '\(SQLText.literal(SourceStatus.closedM1Bar.rawValue))'
                OR digits != \(SineTestSecurity.digits.rawValue)
                OR open_scaled <= 0
                OR high_scaled <= 0
                OR low_scaled <= 0
                OR close_scaled <= 0
                OR volume = 0
            )
        FROM \(database).ohlc_m1_canonical
        WHERE broker_source_id = '\(SQLText.literal(brokerSourceId.rawValue))'
          AND source_origin = '\(SQLText.literal(SineTestSecurity.sourceOrigin.rawValue))'
          AND logical_symbol = '\(SQLText.literal(SineTestSecurity.logicalSymbol.rawValue))'
          AND ts_utc IN (\(start.rawValue), \(lastUtc))
        FORMAT TabSeparated
        """))
        let fields = body.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\t", omittingEmptySubsequences: false)
        guard fields.count == 3,
              let count = Int64(fields[0]),
              let uniqueUtc = Int64(fields[1]),
              let badRows = Int64(fields[2]) else {
            throw SineWaveDatabaseSyncError.invalidCoverageRow(body)
        }
        return count == Int64(expectedRows)
            && uniqueUtc == Int64(expectedRows)
            && badRows == 0
    }

    private func syntheticSourceSHA256(bars: [ValidatedBar], start: MT5ServerSecond, end: MT5ServerSecond) -> SHA256DigestHex {
        var hasher = SHA256ChunkHasher(namespace: "synthetic_sinetest_m1_source_chunk")
        hasher.appendField("verification_method", Self.coverageVerificationMethod)
        hasher.appendField("source_origin", SineTestSecurity.sourceOrigin.rawValue)
        hasher.appendField("logical_symbol", SineTestSecurity.logicalSymbol.rawValue)
        hasher.appendField("mt5_symbol", SineTestSecurity.providerSymbol.rawValue)
        hasher.appendField("timeframe", Timeframe.m1.rawValue)
        hasher.appendField("mt5_range_start", start.rawValue)
        hasher.appendField("mt5_range_end_exclusive", end.rawValue)
        hasher.appendField("minimum_normalized_value_ppm", 1_000)
        hasher.appendField("row_count", bars.count)
        for (index, bar) in bars.enumerated() {
            hasher.appendField("row_index", index)
            hasher.appendField("mt5_server_ts_raw", bar.mt5ServerTime.rawValue)
            hasher.appendField("open_scaled", bar.open.rawValue)
            hasher.appendField("high_scaled", bar.high.rawValue)
            hasher.appendField("low_scaled", bar.low.rawValue)
            hasher.appendField("close_scaled", bar.close.rawValue)
            hasher.appendField("volume", Int64(bitPattern: bar.volume.rawValue))
        }
        return hasher.finalize()
    }

    private func syntheticOffsetAuthoritySHA256(start: MT5ServerSecond, end: MT5ServerSecond, rowCount: Int) -> SHA256DigestHex {
        var hasher = SHA256ChunkHasher(namespace: "synthetic_sinetest_offset_authority")
        hasher.appendField("broker_source_id", brokerSourceId.rawValue)
        hasher.appendField("source_origin", SineTestSecurity.sourceOrigin.rawValue)
        hasher.appendField("logical_symbol", SineTestSecurity.logicalSymbol.rawValue)
        hasher.appendField("mt5_range_start", start.rawValue)
        hasher.appendField("mt5_range_end_exclusive", end.rawValue)
        hasher.appendField("server_utc_offset_seconds", 0)
        hasher.appendField("offset_source", OffsetSource.configured.rawValue)
        hasher.appendField("offset_confidence", OffsetConfidence.verified.rawValue)
        hasher.appendField("row_count", rowCount)
        return hasher.finalize()
    }

    private func validateRange(start: UtcSecond, end: UtcSecond) throws {
        guard start.rawValue < end.rawValue,
              start.isMinuteAligned,
              end.isMinuteAligned else {
            throw SineWaveDatabaseSyncError.invalidRange(start, end)
        }
    }

    private func mergeIntervals(_ intervals: [CoverageInterval]) -> [CoverageInterval] {
        let sorted = intervals
            .filter { $0.start < $0.end }
            .sorted {
                if $0.start != $1.start { return $0.start < $1.start }
                return $0.end < $1.end
            }
        var merged: [CoverageInterval] = []
        for interval in sorted {
            guard var last = merged.popLast() else {
                merged.append(interval)
                continue
            }
            if interval.start <= last.end {
                last.end = max(last.end, interval.end)
                merged.append(last)
            } else {
                merged.append(last)
                merged.append(interval)
            }
        }
        return merged
    }

    private func firstGap(from start: Int64, to end: Int64, coveredBy intervals: [CoverageInterval]) -> CoverageInterval? {
        var cursor = start
        for interval in intervals where interval.end > cursor {
            if interval.start > cursor {
                return CoverageInterval(start: cursor, end: min(interval.start, end))
            }
            cursor = max(cursor, interval.end)
            if cursor >= end {
                return nil
            }
        }
        return cursor < end ? CoverageInterval(start: cursor, end: end) : nil
    }

    private static func floorDiv(_ numerator: Int64, _ denominator: Int64) -> Int64 {
        precondition(denominator > 0)
        let quotient = numerator / denominator
        let remainder = numerator % denominator
        return remainder < 0 ? quotient - 1 : quotient
    }
}

private struct CoverageInterval: Sendable {
    var start: Int64
    var end: Int64
}

public struct SineTestDataProductionAgent: ProductionAgent {
    public let descriptor: AgentDescriptor

    public init(intervalSeconds: Int = SineTestSecurity.syncIntervalSeconds) {
        self.descriptor = AgentDescriptor(
            kind: .sineTestSynchronizer,
            intervalSeconds: intervalSeconds,
            requiresMT5Bridge: false
        )
    }

    public func run(context: AgentRuntimeContext, startedAt: Date) async throws -> AgentOutcome {
        let sync = try SineWaveDatabaseSyncAgent(
            clickHouse: context.clickHouse,
            database: context.config.clickHouse.database,
            chunkRows: context.config.app.chunkSize
        )
        let result = try await sync.syncThroughRuntimeNow()
        let factory = AgentOutcomeFactory(kind: descriptor.kind, startedAt: startedAt)
        if result.alreadyCurrent {
            return factory.ok(
                "SineTest data is current",
                details: "target_utc=\(result.targetStartUtc.rawValue)..<\(result.targetEndExclusiveUtc.rawValue)"
            )
        }
        return factory.ok(
            "SineTest data synchronized",
            details: "chunks_inserted=\(result.chunksInserted); rows_inserted=\(result.rowsInserted); target_utc=\(result.targetStartUtc.rawValue)..<\(result.targetEndExclusiveUtc.rawValue)"
        )
    }
}
