import Foundation

public struct FXBacktestAPIClient: Sendable {
    private let baseURL: URL
    private let session: URLSession
    private let requestTimeoutSeconds: Double

    public init(baseURL: URL, requestTimeoutSeconds: Double = 120, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
        self.requestTimeoutSeconds = requestTimeoutSeconds
    }

    public func status() async throws -> FXBacktestAPIStatusResponse {
        var request = URLRequest(url: try endpoint(FXBacktestAPIV1.statusPath))
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let data = try await perform(request)
        let response = try JSONDecoder().decode(FXBacktestAPIStatusResponse.self, from: data)
        guard response.apiVersion == FXBacktestAPIV1.version else {
            throw FXBacktestAPIClientError.apiVersionMismatch(response.apiVersion)
        }
        return response
    }

    public func loadM1History(_ historyRequest: FXBacktestM1HistoryRequest) async throws -> FXBacktestM1HistoryResponse {
        try historyRequest.validate()
        let response: FXBacktestM1HistoryResponse = try await post(historyRequest, to: FXBacktestAPIV1.m1HistoryPath)
        try response.validate()
        return response
    }

    public func ensureBacktestResultSchema() async throws -> FXBacktestResultMutationResponse {
        let request = FXBacktestResultSchemaRequest()
        try request.validate()
        let response: FXBacktestResultMutationResponse = try await post(request, to: FXBacktestAPIV1.resultSchemaPath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func startBacktestRun(_ run: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse {
        try run.validate()
        let response: FXBacktestResultMutationResponse = try await post(run, to: FXBacktestAPIV1.resultRunStartPath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func appendBacktestResults(_ results: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse {
        try results.validate()
        let response: FXBacktestResultMutationResponse = try await post(results, to: FXBacktestAPIV1.resultPassAppendPath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func completeBacktestRun(_ completion: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse {
        try completion.validate()
        let response: FXBacktestResultMutationResponse = try await post(completion, to: FXBacktestAPIV1.resultRunCompletePath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func purgeBacktestResults(_ purge: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse {
        try purge.validate()
        let response: FXBacktestResultPurgeResponse = try await post(purge, to: FXBacktestAPIV1.resultPurgePath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func getBacktestRun(_ run: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse {
        try run.validate()
        let response: FXBacktestResultRunGetResponse = try await post(run, to: FXBacktestAPIV1.resultRunGetPath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    public func getBacktestPasses(_ passes: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse {
        try passes.validate()
        let response: FXBacktestResultPassesGetResponse = try await post(passes, to: FXBacktestAPIV1.resultPassesGetPath)
        try validateAPIVersion(response.apiVersion)
        return response
    }

    private func endpoint(_ path: String) throws -> URL {
        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            throw FXBacktestAPIClientError.invalidBaseURL(baseURL.absoluteString)
        }
        components.path = path
        components.query = nil
        guard let url = components.url else {
            throw FXBacktestAPIClientError.invalidBaseURL(baseURL.absoluteString)
        }
        return url
    }

    private func post<Request: Encodable, Response: Decodable>(_ body: Request, to path: String) async throws -> Response {
        var request = URLRequest(url: try endpoint(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try Self.makeEncoder().encode(body)
        let data = try await perform(request)
        return try JSONDecoder().decode(Response.self, from: data)
    }

    private func validateAPIVersion(_ apiVersion: String) throws {
        guard apiVersion == FXBacktestAPIV1.version else {
            throw FXBacktestAPIClientError.apiVersionMismatch(apiVersion)
        }
    }

    private func perform(_ request: URLRequest) async throws -> Data {
        var request = request
        request.timeoutInterval = requestTimeoutSeconds
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw FXBacktestAPIClientError.transport(error.localizedDescription)
        }
        guard let httpResponse = response as? HTTPURLResponse else {
            throw FXBacktestAPIClientError.invalidResponse("Response was not HTTP.")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let errorResponse: FXBacktestAPIErrorResponse
            do {
                errorResponse = try JSONDecoder().decode(FXBacktestAPIErrorResponse.self, from: data)
            } catch {
                let body = String(data: data, encoding: .utf8) ?? ""
                let detail = body.isEmpty
                    ? "Could not decode FXDatabase API error response: \(error)"
                    : "\(body) (could not decode FXDatabase API error response: \(error))"
                throw FXBacktestAPIClientError.httpStatus(httpResponse.statusCode, detail)
            }
            throw FXBacktestAPIClientError.server(code: errorResponse.error.code, message: errorResponse.error.message)
        }
        return data
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

public enum FXBacktestAPIClientError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidBaseURL(String)
    case transport(String)
    case invalidResponse(String)
    case apiVersionMismatch(String)
    case httpStatus(Int, String)
    case server(code: String, message: String)

    public var description: String {
        switch self {
        case .invalidBaseURL(let url):
            return "Invalid FXDatabase API base URL: \(url)"
        case .transport(let reason):
            return "FXDatabase API transport failed: \(reason)"
        case .invalidResponse(let reason):
            return "Invalid FXDatabase API response: \(reason)"
        case .apiVersionMismatch(let version):
            return "FXDatabase API version mismatch: got \(version), expected \(FXBacktestAPIV1.version)"
        case .httpStatus(let status, let body):
            return "FXDatabase API HTTP \(status): \(body)"
        case .server(let code, let message):
            return "FXDatabase API error \(code): \(message)"
        }
    }
}
