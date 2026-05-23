import Foundation
import FXBacktestAPI

public protocol FXBacktestHistoryProviding: Sendable {
    func loadM1History(_ request: FXBacktestM1HistoryRequest) async throws -> FXBacktestM1HistoryResponse
}

public protocol FXBacktestResultProviding: Sendable {
    func ensureResultSchema(_ request: FXBacktestResultSchemaRequest) async throws -> FXBacktestResultMutationResponse
    func startRun(_ request: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse
    func appendPassResults(_ request: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse
    func completeRun(_ request: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse
    func purgeResults(_ request: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse
    func getRun(_ request: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse
    func getPasses(_ request: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse
}

public struct FXBacktestHTTPResponse: Sendable, Equatable {
    public let statusCode: Int
    public let contentType: String
    public let body: Data

    public init(statusCode: Int, contentType: String = "application/json; charset=utf-8", body: Data) {
        self.statusCode = statusCode
        self.contentType = contentType
        self.body = body
    }
}

public struct FXBacktestAPIHTTPHandler: Sendable {
    private let historyProvider: any FXBacktestHistoryProviding
    private let resultProvider: (any FXBacktestResultProviding)?

    public init(historyProvider: any FXBacktestHistoryProviding, resultProvider: (any FXBacktestResultProviding)? = nil) {
        self.historyProvider = historyProvider
        self.resultProvider = resultProvider
    }

    public func handle(method: String, path: String, body: Data) async -> FXBacktestHTTPResponse {
        do {
            switch (method.uppercased(), path) {
            case ("GET", FXBacktestAPIV1.statusPath):
                return try json(FXBacktestAPIStatusResponse())

            case ("POST", FXBacktestAPIV1.m1HistoryPath):
                let request = try JSONDecoder().decode(FXBacktestM1HistoryRequest.self, from: body)
                try request.validate()
                let response = try await historyProvider.loadM1History(request)
                try response.validate()
                return try json(response)

            case ("POST", FXBacktestAPIV1.resultSchemaPath):
                let request = try JSONDecoder().decode(FXBacktestResultSchemaRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().ensureResultSchema(request))

            case ("POST", FXBacktestAPIV1.resultRunStartPath):
                let request = try JSONDecoder().decode(FXBacktestResultRunStartRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().startRun(request))

            case ("POST", FXBacktestAPIV1.resultPassAppendPath):
                let request = try JSONDecoder().decode(FXBacktestResultPassAppendRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().appendPassResults(request))

            case ("POST", FXBacktestAPIV1.resultRunCompletePath):
                let request = try JSONDecoder().decode(FXBacktestResultRunCompleteRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().completeRun(request))

            case ("POST", FXBacktestAPIV1.resultPurgePath):
                let request = try JSONDecoder().decode(FXBacktestResultPurgeRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().purgeResults(request))

            case ("POST", FXBacktestAPIV1.resultRunGetPath):
                let request = try JSONDecoder().decode(FXBacktestResultRunGetRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().getRun(request))

            case ("POST", FXBacktestAPIV1.resultPassesGetPath):
                let request = try JSONDecoder().decode(FXBacktestResultPassesGetRequest.self, from: body)
                try request.validate()
                return try json(try await requireResultProvider().getPasses(request))

            default:
                return try error(status: 404, code: "not_found", message: "Unknown FXBacktest API endpoint \(method) \(path)")
            }
        } catch let error as FXBacktestAPIValidationError {
            return safeError(status: 400, code: "invalid_request", message: error.description)
        } catch let error as DecodingError {
            return safeError(status: 400, code: "invalid_json", message: String(describing: error))
        } catch let error as FXBacktestAPIServiceError {
            return safeError(status: error.httpStatus, code: error.code, message: error.description)
        } catch {
            return safeError(status: 500, code: "internal_error", message: String(describing: error))
        }
    }

    private func json<T: Encodable>(_ value: T, status: Int = 200) throws -> FXBacktestHTTPResponse {
        FXBacktestHTTPResponse(statusCode: status, body: try Self.makeEncoder().encode(value))
    }

    private func requireResultProvider() throws -> any FXBacktestResultProviding {
        guard let resultProvider else {
            throw FXBacktestAPIServiceError.resultStoreUnavailable("FXDatabase was started without a backtest result provider.")
        }
        return resultProvider
    }

    private func error(status: Int, code: String, message: String) throws -> FXBacktestHTTPResponse {
        try json(FXBacktestAPIErrorResponse(code: code, message: message), status: status)
    }

    private func safeError(status: Int, code: String, message: String) -> FXBacktestHTTPResponse {
        do {
            return try error(status: status, code: code, message: message)
        } catch {
            let fallback = #"{"api_version":"\#(FXBacktestAPIV1.version)","error":{"code":"encoding_error","message":"Could not encode API error response"}}"#
            return FXBacktestHTTPResponse(statusCode: 500, body: Data(fallback.utf8))
        }
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

public enum FXBacktestAPIServiceError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidRequest(String)
    case unconfiguredSymbol(String)
    case brokerMismatch(expected: String, actual: String)
    case mt5SymbolMismatch(expected: String, actual: String)
    case digitsMismatch(expected: Int, actual: Int)
    case readinessBlocked(String)
    case historyUnavailable(String)
    case resultStoreUnavailable(String)

    public var httpStatus: Int {
        switch self {
        case .invalidRequest, .unconfiguredSymbol, .brokerMismatch, .mt5SymbolMismatch, .digitsMismatch:
            return 400
        case .readinessBlocked:
            return 409
        case .historyUnavailable:
            return 502
        case .resultStoreUnavailable:
            return 503
        }
    }

    public var code: String {
        switch self {
        case .invalidRequest:
            return "invalid_request"
        case .unconfiguredSymbol:
            return "unconfigured_symbol"
        case .brokerMismatch:
            return "broker_mismatch"
        case .mt5SymbolMismatch:
            return "mt5_symbol_mismatch"
        case .digitsMismatch:
            return "digits_mismatch"
        case .readinessBlocked:
            return "readiness_blocked"
        case .historyUnavailable:
            return "history_unavailable"
        case .resultStoreUnavailable:
            return "result_store_unavailable"
        }
    }

    public var description: String {
        switch self {
        case .invalidRequest(let reason):
            return reason
        case .unconfiguredSymbol(let symbol):
            return "\(symbol) is not configured in FXDatabase symbols.json."
        case .brokerMismatch(let expected, let actual):
            return "Requested broker_source_id \(actual) does not match FXDatabase broker_source_id \(expected)."
        case .mt5SymbolMismatch(let expected, let actual):
            return "Requested expected_mt5_symbol \(actual) does not match FXDatabase configured MT5 symbol \(expected)."
        case .digitsMismatch(let expected, let actual):
            return "Requested expected_digits \(actual) does not match FXDatabase configured digits \(expected)."
        case .readinessBlocked(let reason):
            return "FXDatabase backtest readiness gate blocked the request: \(reason)"
        case .historyUnavailable(let reason):
            return "FXDatabase could not load verified M1 history: \(reason)"
        case .resultStoreUnavailable(let reason):
            return "FXDatabase backtest result store is unavailable: \(reason)"
        }
    }
}
