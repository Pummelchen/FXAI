import AppCore
import ClickHouse
import Config
import Domain
import Foundation
import MT5Bridge

public struct IngestionWritePipeline: Sendable {
    private let clickHouse: ClickHouseClientProtocol
    private let brokerSourceId: BrokerSourceId

    public init(clickHouse: ClickHouseClientProtocol, brokerSourceId: BrokerSourceId) {
        self.clickHouse = clickHouse
        self.brokerSourceId = brokerSourceId
    }

    public func writeValidatedBars(
        _ bars: [ValidatedBar],
        insertBuilder: ClickHouseInsertBuilder,
        auditStore: IngestAuditStore,
        mapping: SymbolMapping,
        operationType: IngestOperationType,
        batchId: BatchId,
        range: (from: MT5ServerSecond, toExclusive: MT5ServerSecond),
        sourceBarCount: Int,
        sourceHash: String,
        mt5SourceSHA256: SHA256DigestHex,
        offsetAuthoritySHA256: SHA256DigestHex,
        emptyChunkMessage: String
    ) async throws -> CanonicalInsertVerificationResult {
        guard let first = bars.first else {
            throw IngestError.invalidChunk(emptyChunkMessage)
        }
        let rawInsert = insertBuilder.rawBarsInsert(bars)
        let canonicalDelete = try insertBuilder.canonicalRangeDelete(
            bars,
            mt5Start: range.from,
            mt5EndExclusive: range.toExclusive
        )
        let canonicalInsert = try insertBuilder.canonicalBarsInsert(bars)
        try await CanonicalConflictRecorder(clickHouse: clickHouse, insertBuilder: insertBuilder)
            .recordConflictsBeforeCanonicalReplace(bars, detectedAtUtc: first.ingestedAtUtc)

        _ = try await clickHouse.execute(rawInsert)
        try await recordChunkOperation(
            auditStore: auditStore,
            mapping: mapping,
            operationType: operationType,
            batchId: batchId,
            range: range,
            status: .rawWritten,
            stage: "raw_audit_written",
            sourceBarCount: sourceBarCount,
            canonicalRowCount: nil,
            sourceHash: sourceHash,
            mt5SourceSHA256: mt5SourceSHA256,
            offsetAuthoritySHA256: offsetAuthoritySHA256
        )

        _ = try await clickHouse.execute(canonicalDelete)
        try await recordChunkOperation(
            auditStore: auditStore,
            mapping: mapping,
            operationType: operationType,
            batchId: batchId,
            range: range,
            status: .canonicalDeleted,
            stage: "canonical_range_deleted",
            sourceBarCount: sourceBarCount,
            canonicalRowCount: nil,
            sourceHash: sourceHash,
            mt5SourceSHA256: mt5SourceSHA256,
            offsetAuthoritySHA256: offsetAuthoritySHA256
        )

        _ = try await clickHouse.execute(canonicalInsert)
        try await recordChunkOperation(
            auditStore: auditStore,
            mapping: mapping,
            operationType: operationType,
            batchId: batchId,
            range: range,
            status: .canonicalWritten,
            stage: "canonical_written",
            sourceBarCount: sourceBarCount,
            canonicalRowCount: bars.count,
            sourceHash: sourceHash,
            mt5SourceSHA256: mt5SourceSHA256,
            offsetAuthoritySHA256: offsetAuthoritySHA256
        )

        let canonicalVerification = try await CanonicalInsertVerifier(clickHouse: clickHouse, insertBuilder: insertBuilder).verify(
            bars,
            mt5Start: range.from,
            mt5EndExclusive: range.toExclusive
        )
        try await recordChunkOperation(
            auditStore: auditStore,
            mapping: mapping,
            operationType: operationType,
            batchId: batchId,
            range: range,
            status: .readbackVerified,
            stage: "canonical_readback_verified",
            sourceBarCount: sourceBarCount,
            canonicalRowCount: bars.count,
            sourceHash: sourceHash,
            mt5SourceSHA256: mt5SourceSHA256,
            canonicalReadbackSHA256: canonicalVerification.canonicalReadbackSHA256,
            offsetAuthoritySHA256: offsetAuthoritySHA256
        )
        return canonicalVerification
    }

    public func recordChunkOperation(
        auditStore: IngestAuditStore,
        mapping: SymbolMapping,
        operationType: IngestOperationType,
        batchId: BatchId,
        range: (from: MT5ServerSecond, toExclusive: MT5ServerSecond),
        status: IngestOperationStatus,
        stage: String,
        sourceBarCount: Int?,
        canonicalRowCount: Int?,
        sourceHash: String?,
        mt5SourceSHA256: SHA256DigestHex? = nil,
        canonicalReadbackSHA256: SHA256DigestHex? = nil,
        offsetAuthoritySHA256: SHA256DigestHex? = nil,
        errorMessage: String? = nil
    ) async throws {
        let hasSHA256Evidence = mt5SourceSHA256 != nil || canonicalReadbackSHA256 != nil || offsetAuthoritySHA256 != nil
        try await auditStore.recordOperation(
            brokerSourceId: brokerSourceId,
            sourceOrigin: mapping.sourceOrigin,
            logicalSymbol: mapping.logicalSymbol,
            mt5Symbol: mapping.mt5Symbol,
            operationType: operationType,
            batchId: batchId,
            mt5Start: range.from,
            mt5EndExclusive: range.toExclusive,
            status: status,
            stage: stage,
            sourceBarCount: sourceBarCount,
            canonicalRowCount: canonicalRowCount,
            sourceHash: sourceHash,
            hashSchemaVersion: hasSHA256Evidence ? ChunkHashing.schemaVersion : nil,
            mt5SourceSHA256: mt5SourceSHA256,
            canonicalReadbackSHA256: canonicalReadbackSHA256,
            offsetAuthoritySHA256: offsetAuthoritySHA256,
            errorMessage: errorMessage
        )
    }

    public static func validateClosedBarsInRange(
        _ bars: [ClosedM1Bar],
        from: MT5ServerSecond,
        toExclusive: MT5ServerSecond
    ) throws {
        for bar in bars {
            guard bar.mt5ServerTime.rawValue >= from.rawValue,
                  bar.mt5ServerTime.rawValue < toExclusive.rawValue else {
                throw IngestError.invalidBridgeResponse("MT5 bar \(bar.mt5ServerTime.rawValue) is outside requested range \(from.rawValue)..<\(toExclusive.rawValue)")
            }
        }
    }

    public static func loadTerminalIdentity(
        bridge: MT5BridgeClient,
        brokerSourceId: BrokerSourceId,
        expected: ExpectedTerminalIdentity?,
        logger: Logger
    ) throws -> BrokerServerIdentity {
        let actual = try bridge.terminalInfo()
        do {
            return try TerminalIdentityPolicy().resolve(
                actual: actual,
                brokerSourceId: brokerSourceId,
                expected: expected,
                logger: logger
            )
        } catch let error as TerminalIdentityPolicyError {
            throw IngestError.terminalIdentityMismatch(error.description)
        }
    }
}
