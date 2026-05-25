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
        try validateResponse(response)
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
        try validateResponse(response)
        return response
    }

    public func startBacktestRun(_ run: FXBacktestResultRunStartRequest) async throws -> FXBacktestResultMutationResponse {
        try run.validate()
        let response: FXBacktestResultMutationResponse = try await post(run, to: FXBacktestAPIV1.resultRunStartPath)
        try validateResponse(response)
        return response
    }

    public func appendBacktestResults(_ results: FXBacktestResultPassAppendRequest) async throws -> FXBacktestResultMutationResponse {
        try results.validate()
        let response: FXBacktestResultMutationResponse = try await post(results, to: FXBacktestAPIV1.resultPassAppendPath)
        try validateResponse(response)
        return response
    }

    public func completeBacktestRun(_ completion: FXBacktestResultRunCompleteRequest) async throws -> FXBacktestResultMutationResponse {
        try completion.validate()
        let response: FXBacktestResultMutationResponse = try await post(completion, to: FXBacktestAPIV1.resultRunCompletePath)
        try validateResponse(response)
        return response
    }

    public func purgeBacktestResults(_ purge: FXBacktestResultPurgeRequest) async throws -> FXBacktestResultPurgeResponse {
        try purge.validate()
        let response: FXBacktestResultPurgeResponse = try await post(purge, to: FXBacktestAPIV1.resultPurgePath)
        try validateResponse(response)
        return response
    }

    public func getBacktestRun(_ run: FXBacktestResultRunGetRequest) async throws -> FXBacktestResultRunGetResponse {
        try run.validate()
        let response: FXBacktestResultRunGetResponse = try await post(run, to: FXBacktestAPIV1.resultRunGetPath)
        try validateResponse(response)
        return response
    }

    public func getBacktestPasses(_ passes: FXBacktestResultPassesGetRequest) async throws -> FXBacktestResultPassesGetResponse {
        try passes.validate()
        let response: FXBacktestResultPassesGetResponse = try await post(passes, to: FXBacktestAPIV1.resultPassesGetPath)
        try validateResponse(response)
        return response
    }

    public func ensureBacktestConfigurationSchema() async throws -> FXBacktestResultMutationResponse {
        let request = FXBacktestConfigurationSchemaRequest()
        try request.validate()
        let response: FXBacktestResultMutationResponse = try await post(request, to: FXBacktestAPIV1.configurationSchemaPath)
        try validateResponse(response)
        return response
    }

    public func registerBacktestConfiguration(_ registration: FXBacktestConfigurationRegistrationRequest) async throws -> FXBacktestResultMutationResponse {
        try registration.validate()
        let response: FXBacktestResultMutationResponse = try await post(registration, to: FXBacktestAPIV1.configurationRegisterPath)
        try validateResponse(response)
        return response
    }

    public func getBacktestConfiguration(_ request: FXBacktestConfigurationGetRequest = FXBacktestConfigurationGetRequest()) async throws -> FXBacktestConfigurationSnapshotResponse {
        try request.validate()
        let response: FXBacktestConfigurationSnapshotResponse = try await post(request, to: FXBacktestAPIV1.configurationGetPath)
        try validateResponse(response)
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

    private func validateResponse(_ response: FXBacktestAPIStatusResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateResponse(_ response: FXBacktestResultMutationResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateResponse(_ response: FXBacktestResultPurgeResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateResponse(_ response: FXBacktestResultRunGetResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateResponse(_ response: FXBacktestResultPassesGetResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateResponse(_ response: FXBacktestConfigurationSnapshotResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func validateErrorResponse(_ response: FXBacktestAPIErrorResponse) throws {
        try mapValidationError { try response.validate() }
    }

    private func mapValidationError(_ body: () throws -> Void) throws {
        do {
            try body()
        } catch let error as FXBacktestAPIValidationError {
            switch error {
            case .unsupportedVersion(let version):
                throw FXBacktestAPIClientError.apiVersionMismatch(version)
            case .invalidField(let reason):
                throw FXBacktestAPIClientError.invalidResponse(reason)
            }
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
            try validateErrorResponse(errorResponse)
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
            return "FXDatabase API version mismatch: got \(version), expected \(FXBacktestAPIV1.latestVersion)"
        case .httpStatus(let status, let body):
            return "FXDatabase API HTTP \(status): \(body)"
        case .server(let code, let message):
            return "FXDatabase API error \(code): \(message)"
        }
    }
}
