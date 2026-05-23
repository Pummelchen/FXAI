import Foundation

public final class MT5BridgeClient: @unchecked Sendable {
    private let connection: MT5Connection
    private let codec: FramedProtocolCodec

    public init(connection: MT5Connection, codec: FramedProtocolCodec = FramedProtocolCodec()) {
        self.connection = connection
        self.codec = codec
    }

    public static func connect(
        host: String,
        port: UInt16,
        connectTimeoutSeconds: Double,
        requestTimeoutSeconds: Double
    ) throws -> MT5BridgeClient {
        let connection = try MT5Connection.connect(
            host: host,
            port: port,
            connectTimeoutSeconds: connectTimeoutSeconds,
            requestTimeoutSeconds: requestTimeoutSeconds
        )
        return MT5BridgeClient(connection: connection)
    }

    public static func listen(
        host: String,
        port: UInt16,
        connectTimeoutSeconds: Double,
        requestTimeoutSeconds: Double
    ) throws -> MT5BridgeClient {
        let connection = try MT5Connection.listenOnce(
            host: host,
            port: port,
            connectTimeoutSeconds: connectTimeoutSeconds,
            requestTimeoutSeconds: requestTimeoutSeconds
        )
        return MT5BridgeClient(connection: connection)
    }

    public func request<RequestPayload: Encodable, ResponsePayload: Decodable & Sendable>(
        command: MT5Command,
        payload: RequestPayload,
        responseType: ResponsePayload.Type
    ) throws -> ResponsePayload {
        let requestId = UUID().uuidString
        let frame = try codec.encode(
            command: command,
            requestId: requestId,
            timestampSentUtc: ProtocolTimestampUtc(rawValue: Int64(Date().timeIntervalSince1970)),
            payload: payload
        )
        try connection.sendFrame(frame)
        let responseBody = try connection.readFrameBody()
        let message = try codec.decode(responseBody, payloadType: responseType)
        guard message.requestId == requestId else {
            throw ProtocolError.invalidField("request_id")
        }
        guard message.command == command else {
            throw ProtocolError.invalidField("command")
        }
        return message.payload
    }

    public func hello() throws -> HelloResponseDTO {
        try request(command: .hello, payload: EmptyPayload(), responseType: HelloResponseDTO.self)
    }

    public func ping() throws -> EmptyPayload {
        try request(command: .ping, payload: EmptyPayload(), responseType: EmptyPayload.self)
    }

    public func terminalInfo() throws -> TerminalInfoDTO {
        try request(command: .getTerminalInfo, payload: EmptyPayload(), responseType: TerminalInfoDTO.self)
    }

    public func prepareSymbol(_ mt5Symbol: String) throws -> SymbolInfoDTO {
        try request(command: .prepareSymbol, payload: SymbolPayload(mt5Symbol: mt5Symbol), responseType: SymbolInfoDTO.self)
    }

    public func prepareSymbol<Symbol: RawRepresentable>(_ mt5Symbol: Symbol) throws -> SymbolInfoDTO where Symbol.RawValue == String {
        try prepareSymbol(mt5Symbol.rawValue)
    }

    public func symbolInfo(_ mt5Symbol: String) throws -> SymbolInfoDTO {
        try request(command: .getSymbolInfo, payload: SymbolPayload(mt5Symbol: mt5Symbol), responseType: SymbolInfoDTO.self)
    }

    public func symbolInfo<Symbol: RawRepresentable>(_ mt5Symbol: Symbol) throws -> SymbolInfoDTO where Symbol.RawValue == String {
        try symbolInfo(mt5Symbol.rawValue)
    }

    public func historyStatus(_ mt5Symbol: String) throws -> HistoryStatusDTO {
        try request(command: .getHistoryStatus, payload: SymbolPayload(mt5Symbol: mt5Symbol), responseType: HistoryStatusDTO.self)
    }

    public func historyStatus<Symbol: RawRepresentable>(_ mt5Symbol: Symbol) throws -> HistoryStatusDTO where Symbol.RawValue == String {
        try historyStatus(mt5Symbol.rawValue)
    }

    public func ensureM1MonthHistory(
        mt5Symbol: String,
        monthStartMT5ServerTs: Int64,
        monthEndMT5ServerTsExclusive: Int64
    ) throws -> M1MonthHistoryStatusDTO {
        guard monthStartMT5ServerTs >= 0 else {
            throw ProtocolError.invalidField("month_start_mt5_server_ts")
        }
        guard monthEndMT5ServerTsExclusive > monthStartMT5ServerTs else {
            throw ProtocolError.invalidField("month_end_mt5_server_ts_exclusive")
        }
        guard Self.isMinuteAligned(monthStartMT5ServerTs), Self.isMinuteAligned(monthEndMT5ServerTsExclusive) else {
            throw ProtocolError.invalidField("month range must be minute-aligned")
        }
        return try request(
            command: .ensureM1MonthHistory,
            payload: M1MonthHistoryPayload(
                mt5Symbol: mt5Symbol,
                monthStartMT5ServerTs: monthStartMT5ServerTs,
                monthEndMT5ServerTsExclusive: monthEndMT5ServerTsExclusive
            ),
            responseType: M1MonthHistoryStatusDTO.self
        )
    }

    public func ensureM1MonthHistory<Symbol: RawRepresentable, Second: RawRepresentable>(
        mt5Symbol: Symbol,
        monthStart: Second,
        monthEndExclusive: Second
    ) throws -> M1MonthHistoryStatusDTO where Symbol.RawValue == String, Second.RawValue == Int64 {
        try ensureM1MonthHistory(
            mt5Symbol: mt5Symbol.rawValue,
            monthStartMT5ServerTs: monthStart.rawValue,
            monthEndMT5ServerTsExclusive: monthEndExclusive.rawValue
        )
    }

    public func oldestM1BarTime(_ mt5Symbol: String) throws -> SingleTimeResponseDTO {
        try request(command: .getOldestM1BarTime, payload: SymbolPayload(mt5Symbol: mt5Symbol), responseType: SingleTimeResponseDTO.self)
    }

    public func oldestM1BarTime<Symbol: RawRepresentable>(_ mt5Symbol: Symbol) throws -> SingleTimeResponseDTO where Symbol.RawValue == String {
        try oldestM1BarTime(mt5Symbol.rawValue)
    }

    public func latestClosedM1Bar(_ mt5Symbol: String) throws -> SingleTimeResponseDTO {
        try request(command: .getLatestClosedM1Bar, payload: SymbolPayload(mt5Symbol: mt5Symbol), responseType: SingleTimeResponseDTO.self)
    }

    public func latestClosedM1Bar<Symbol: RawRepresentable>(_ mt5Symbol: Symbol) throws -> SingleTimeResponseDTO where Symbol.RawValue == String {
        try latestClosedM1Bar(mt5Symbol.rawValue)
    }

    public func ratesRange(
        mt5Symbol: String,
        fromMT5ServerTs: Int64,
        toMT5ServerTsExclusive: Int64,
        maxBars: Int
    ) throws -> RatesResponseDTO {
        try request(
            command: .getRatesRange,
            payload: RatesRangePayload(
                mt5Symbol: mt5Symbol,
                fromMT5ServerTs: fromMT5ServerTs,
                toMT5ServerTsExclusive: toMT5ServerTsExclusive,
                maxBars: maxBars
            ),
            responseType: RatesResponseDTO.self
        )
    }

    public func ratesRange<Symbol: RawRepresentable, Second: RawRepresentable>(
        mt5Symbol: Symbol,
        from: Second,
        toExclusive: Second,
        maxBars: Int
    ) throws -> RatesResponseDTO where Symbol.RawValue == String, Second.RawValue == Int64 {
        try ratesRange(
            mt5Symbol: mt5Symbol.rawValue,
            fromMT5ServerTs: from.rawValue,
            toMT5ServerTsExclusive: toExclusive.rawValue,
            maxBars: maxBars
        )
    }

    public func ratesFromPosition(mt5Symbol: String, startPosition: Int, count: Int) throws -> RatesResponseDTO {
        guard startPosition >= 1 else {
            throw ProtocolError.invalidField("start_pos")
        }
        guard count > 0 else {
            throw ProtocolError.invalidField("count")
        }
        return try request(
            command: .getRatesFromPosition,
            payload: RatesFromPositionPayload(
                mt5Symbol: mt5Symbol,
                startPosition: startPosition,
                count: count
            ),
            responseType: RatesResponseDTO.self
        )
    }

    public func ratesFromPosition<Symbol: RawRepresentable>(
        mt5Symbol: Symbol,
        startPosition: Int,
        count: Int
    ) throws -> RatesResponseDTO where Symbol.RawValue == String {
        try ratesFromPosition(mt5Symbol: mt5Symbol.rawValue, startPosition: startPosition, count: count)
    }

    public func serverTimeSnapshot() throws -> ServerTimeSnapshotDTO {
        try request(command: .getServerTimeSnapshot, payload: EmptyPayload(), responseType: ServerTimeSnapshotDTO.self)
    }

    public func close() {
        connection.close()
    }

    private static func isMinuteAligned(_ timestamp: Int64) -> Bool {
        timestamp % 60 == 0
    }
}
