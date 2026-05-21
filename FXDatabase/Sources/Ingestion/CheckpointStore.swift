import ClickHouse
import Domain
import Foundation

public protocol CheckpointStore: Sendable {
    func latestState(brokerSourceId: BrokerSourceId, sourceOrigin: DataSourceOrigin, logicalSymbol: LogicalSymbol) async throws -> IngestState?
    func save(_ state: IngestState) async throws
}

public enum CheckpointError: Error, CustomStringConvertible, Sendable {
    case invalidReadbackRow(String)

    public var description: String {
        switch self {
        case .invalidReadbackRow(let row):
            return "Invalid ClickHouse ingest_state row: \(row)"
        }
    }
}

public actor InMemoryCheckpointStore: CheckpointStore {
    private var states: [String: IngestState] = [:]

    public init() {}

    public func latestState(brokerSourceId: BrokerSourceId, sourceOrigin: DataSourceOrigin = .mt5, logicalSymbol: LogicalSymbol) async throws -> IngestState? {
        states[key(brokerSourceId: brokerSourceId, sourceOrigin: sourceOrigin, logicalSymbol: logicalSymbol)]
    }

    public func save(_ state: IngestState) async throws {
        states[key(brokerSourceId: state.brokerSourceId, sourceOrigin: state.sourceOrigin, logicalSymbol: state.logicalSymbol)] = state
    }

    private func key(brokerSourceId: BrokerSourceId, sourceOrigin: DataSourceOrigin, logicalSymbol: LogicalSymbol) -> String {
        brokerSourceId.rawValue + "|" + sourceOrigin.rawValue + "|" + logicalSymbol.rawValue
    }
}

public struct ClickHouseCheckpointStore: CheckpointStore {
    private let client: ClickHouseClientProtocol
    private let insertBuilder: ClickHouseInsertBuilder
    private let database: String

    public init(client: ClickHouseClientProtocol, insertBuilder: ClickHouseInsertBuilder, database: String) {
        self.client = client
        self.insertBuilder = insertBuilder
        self.database = database
    }

    public func latestState(brokerSourceId: BrokerSourceId, sourceOrigin: DataSourceOrigin = .mt5, logicalSymbol: LogicalSymbol) async throws -> IngestState? {
        let sql = """
        SELECT broker_source_id, source_origin, logical_symbol, mt5_symbol, oldest_mt5_server_ts_raw,
               latest_ingested_closed_mt5_server_ts_raw, latest_ingested_closed_ts_utc,
               status, last_batch_id, updated_at_utc
        FROM \(database).ingest_state
        WHERE broker_source_id = '\(Self.sqlLiteral(brokerSourceId.rawValue))'
          AND source_origin = '\(Self.sqlLiteral(sourceOrigin.rawValue))'
          AND logical_symbol = '\(Self.sqlLiteral(logicalSymbol.rawValue))'
        ORDER BY latest_ingested_closed_mt5_server_ts_raw DESC, updated_at_utc DESC
        LIMIT 1
        FORMAT TabSeparated
        """
        let body = try await client.execute(.select(sql))
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let fields = trimmed.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        guard fields.count == 10 else { throw CheckpointError.invalidReadbackRow(trimmed) }
        guard let sourceOrigin = DataSourceOrigin(rawValue: fields[1]),
              let oldest = Int64(fields[4]),
              let latestMT5 = Int64(fields[5]),
              let latestUTC = Int64(fields[6]),
              let status = IngestStatus(rawValue: fields[7]),
              let updatedAt = Int64(fields[9]) else {
            throw CheckpointError.invalidReadbackRow(trimmed)
        }
        return try IngestState(
            brokerSourceId: BrokerSourceId(fields[0]),
            sourceOrigin: sourceOrigin,
            logicalSymbol: LogicalSymbol(fields[2]),
            mt5Symbol: MT5Symbol(fields[3]),
            oldestMT5ServerTime: MT5ServerSecond(rawValue: oldest),
            latestIngestedClosedMT5ServerTime: MT5ServerSecond(rawValue: latestMT5),
            latestIngestedClosedUtcTime: UtcSecond(rawValue: latestUTC),
            status: status,
            lastBatchId: BatchId(rawValue: fields[8]),
            updatedAtUtc: UtcSecond(rawValue: updatedAt)
        )
    }

    public func save(_ state: IngestState) async throws {
        let query = insertBuilder.ingestStateUpsert(
            brokerSourceId: state.brokerSourceId,
            sourceOrigin: state.sourceOrigin,
            logicalSymbol: state.logicalSymbol,
            mt5Symbol: state.mt5Symbol,
            oldestMT5ServerTime: state.oldestMT5ServerTime,
            latestMT5ServerTime: state.latestIngestedClosedMT5ServerTime,
            latestUtcTime: state.latestIngestedClosedUtcTime,
            status: state.status.rawValue,
            batchId: state.lastBatchId,
            updatedAtUtc: state.updatedAtUtc
        )
        _ = try await client.execute(query)
    }

    private static func sqlLiteral(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'")
    }
}
