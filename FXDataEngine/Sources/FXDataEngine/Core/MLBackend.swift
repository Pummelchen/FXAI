import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Darwin)
import Darwin
#endif

public enum MLFramework: String, Codable, Hashable, Sendable {
    case nativeSwift
    case metal
    case pyTorch
    case tensorFlow
    case foundationNLP
    case onnxRuntime
    case remoteRPC
}

public enum MLBackendMode: Codable, Hashable, Sendable {
    case inProcess(MLFramework)
    case externalPython(framework: MLFramework, executable: String, module: String)
    case remoteRPC(endpoint: String)
}

public struct MLBackendDescriptor: Codable, Hashable, Sendable {
    public let mode: MLBackendMode
    public let modelIdentifier: String
    public let supportsTraining: Bool
    public let supportsInference: Bool
    public let usesVolumeFeatures: Bool

    public init(
        mode: MLBackendMode,
        modelIdentifier: String,
        supportsTraining: Bool = true,
        supportsInference: Bool = true,
        usesVolumeFeatures: Bool = true
    ) {
        self.mode = mode
        self.modelIdentifier = modelIdentifier
        self.supportsTraining = supportsTraining
        self.supportsInference = supportsInference
        self.usesVolumeFeatures = usesVolumeFeatures
    }
}

public struct MLInferencePayload: Codable, Hashable, Sendable {
    public let apiVersion: Int
    public let modelIdentifier: String
    public let framework: MLFramework
    public let dataHasVolume: Bool
    public let horizonMinutes: Int
    public let sequenceBars: Int
    public let priceCostPoints: Double
    public let minMovePoints: Double
    public let x: [Double]
    public let xWindow: [[Double]]
    public let tokenizerContract: PluginTokenizerContractV4
    public let textEvents: [PluginTextEventV4]
    public let eventTexts: [String]

    public init(
        apiVersion: Int = FXDataEngineConstants.latestPluginAPIVersion,
        modelIdentifier: String,
        framework: MLFramework,
        dataHasVolume: Bool,
        horizonMinutes: Int = 1,
        sequenceBars: Int = 1,
        priceCostPoints: Double = 0.0,
        minMovePoints: Double = 0.0,
        x: [Double],
        xWindow: [[Double]],
        tokenizerContract: PluginTokenizerContractV4 = PluginTokenizerContractV4(),
        textEvents: [PluginTextEventV4] = [],
        eventTexts: [String] = []
    ) {
        self.apiVersion = apiVersion
        self.modelIdentifier = modelIdentifier
        self.framework = framework
        self.dataHasVolume = dataHasVolume
        self.horizonMinutes = max(1, horizonMinutes)
        self.sequenceBars = min(max(1, sequenceBars), FXDataEngineConstants.maxSequenceBars)
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
        self.x = x
        self.xWindow = xWindow
        self.tokenizerContract = tokenizerContract
        self.textEvents = Array(textEvents.prefix(PluginContextV4.maxTextEvents))
        let cleanedEventTexts = eventTexts.map {
            String($0.trimmingCharacters(in: .whitespacesAndNewlines).prefix(PluginTextEventV4.maxTextLength))
        }.filter { !$0.isEmpty }
        self.eventTexts = Array((cleanedEventTexts.isEmpty ? self.textEvents.map(\.mergedText) : cleanedEventTexts).prefix(PluginContextV4.maxTextEvents))
    }

    private enum CodingKeys: String, CodingKey {
        case apiVersion
        case modelIdentifier
        case framework
        case dataHasVolume
        case horizonMinutes
        case sequenceBars
        case priceCostPoints
        case minMovePoints
        case x
        case xWindow
        case tokenizerContract
        case textEvents
        case eventTexts
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            apiVersion: try container.decode(Int.self, forKey: .apiVersion),
            modelIdentifier: try container.decode(String.self, forKey: .modelIdentifier),
            framework: try container.decode(MLFramework.self, forKey: .framework),
            dataHasVolume: try container.decodeIfPresent(Bool.self, forKey: .dataHasVolume) ?? false,
            horizonMinutes: try container.decodeIfPresent(Int.self, forKey: .horizonMinutes) ?? 1,
            sequenceBars: try container.decodeIfPresent(Int.self, forKey: .sequenceBars) ?? 1,
            priceCostPoints: try container.decodeIfPresent(Double.self, forKey: .priceCostPoints) ?? 0.0,
            minMovePoints: try container.decodeIfPresent(Double.self, forKey: .minMovePoints) ?? 0.0,
            x: try container.decode([Double].self, forKey: .x),
            xWindow: try container.decodeIfPresent([[Double]].self, forKey: .xWindow) ?? [],
            tokenizerContract: try container.decodeIfPresent(PluginTokenizerContractV4.self, forKey: .tokenizerContract) ?? PluginTokenizerContractV4(),
            textEvents: try container.decodeIfPresent([PluginTextEventV4].self, forKey: .textEvents) ?? [],
            eventTexts: try container.decodeIfPresent([String].self, forKey: .eventTexts) ?? []
        )
    }

    public func validateLatestAPI() throws {
        guard apiVersion == FXDataEngineConstants.latestPluginAPIVersion else {
            throw FXDataEngineError.validation("mlPayload.apiVersion")
        }
        guard !modelIdentifier.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("mlPayload.modelIdentifier")
        }
        guard horizonMinutes > 0 else {
            throw FXDataEngineError.validation("mlPayload.horizonMinutes")
        }
        guard (1...FXDataEngineConstants.maxSequenceBars).contains(sequenceBars) else {
            throw FXDataEngineError.validation("mlPayload.sequenceBars")
        }
        guard priceCostPoints.isFinite, priceCostPoints >= 0 else {
            throw FXDataEngineError.validation("mlPayload.priceCostPoints")
        }
        guard minMovePoints.isFinite, minMovePoints >= 0 else {
            throw FXDataEngineError.validation("mlPayload.minMovePoints")
        }
        try PredictRequestV4.validateInput(x)
        guard xWindow.count <= max(sequenceBars - 1, 0) else {
            throw FXDataEngineError.validation("mlPayload.xWindowSequence")
        }
        if sequenceBars > 1 {
            guard !xWindow.isEmpty else { throw FXDataEngineError.validation("mlPayload.xWindowPayload") }
        }
        for row in xWindow {
            try PredictRequestV4.validateInput(row)
        }
        try tokenizerContract.validate()
        guard textEvents.count <= PluginContextV4.maxTextEvents else {
            throw FXDataEngineError.validation("mlPayload.textEvents.count")
        }
        for event in textEvents {
            try event.validate()
        }
        guard eventTexts.count <= PluginContextV4.maxTextEvents else {
            throw FXDataEngineError.validation("mlPayload.eventTexts.count")
        }
    }
}

public struct MLTrainingPayload: Codable, Hashable, Sendable {
    public let inference: MLInferencePayload
    public let labelClass: LabelClass
    public let movePoints: Double
    public let sampleWeight: Double
    public let nextVolumeTarget: Double

    public init(inference: MLInferencePayload, request: TrainRequestV4) {
        self.inference = inference
        self.labelClass = request.labelClass
        self.movePoints = request.movePoints
        self.sampleWeight = request.sampleWeight
        self.nextVolumeTarget = request.nextVolumeTarget
    }

    public func validateLatestAPI() throws {
        try inference.validateLatestAPI()
        guard movePoints.isFinite else {
            throw FXDataEngineError.validation("mlTraining.movePoints")
        }
        guard sampleWeight.isFinite, sampleWeight >= 0 else {
            throw FXDataEngineError.validation("mlTraining.sampleWeight")
        }
        guard nextVolumeTarget.isFinite, nextVolumeTarget >= 0 else {
            throw FXDataEngineError.validation("mlTraining.nextVolumeTarget")
        }
    }
}

public struct MLTensorContextDescriptor: Codable, Hashable, Sendable {
    public var modelDim: Int
    public var hiddenDim: Int
    public var headCount: Int
    public var headDim: Int
    public var sequenceCapacity: Int
    public var stride: Int
    public var patchSize: Int
    public var dilation: Int
    public var positionStepPenalty: Double

    public init(
        modelDim: Int,
        hiddenDim: Int,
        headCount: Int,
        sequenceCapacity: Int,
        stride: Int = 1,
        patchSize: Int = 1,
        dilation: Int = 1,
        positionStepPenalty: Double = 0.06
    ) {
        self.modelDim = min(max(modelDim, 1), FXDataEngineConstants.aiFeatures)
        self.hiddenDim = min(max(hiddenDim, 1), FXDataEngineConstants.aiMLPHidden)
        self.headCount = min(max(headCount, 1), FXDataEngineConstants.aiFeatures)
        self.headDim = max(self.modelDim / self.headCount, 1)
        self.sequenceCapacity = min(max(sequenceCapacity, 1), FXDataEngineConstants.maxSequenceBars)
        self.stride = max(stride, 1)
        self.patchSize = max(patchSize, 1)
        self.dilation = max(dilation, 1)
        self.positionStepPenalty = fxClamp(positionStepPenalty, 0.0, 2.0)
    }
}

public struct MLSequenceRuntimeDescriptor: Codable, Hashable, Sendable {
    public var maxSteps: Int
    public var stride: Int
    public var patchSize: Int
    public var normalize: Bool
    public var includeCurrent: Bool
    public var positionStepPenalty: Double

    public init(
        maxSteps: Int,
        stride: Int = 1,
        patchSize: Int = 1,
        normalize: Bool = true,
        includeCurrent: Bool = true,
        positionStepPenalty: Double = 0.06
    ) {
        self.maxSteps = min(max(maxSteps, 1), FXDataEngineConstants.maxSequenceBars)
        self.stride = max(stride, 1)
        self.patchSize = max(patchSize, 1)
        self.normalize = normalize
        self.includeCurrent = includeCurrent
        self.positionStepPenalty = fxClamp(positionStepPenalty, 0.0, 2.0)
    }
}

public enum MLTensorBridgeTools {
    public static func clippedCurrentInput(_ x: [Double]) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            let value = index < x.count ? fxSafeFinite(x[index]) : 0.0
            output[index] = index == 0 ? 1.0 : PluginSupportTools.clipSymmetric(value, limit: 8.0)
        }
        return output
    }

    public static func contextDescriptor(
        style: SequenceStyle,
        maxSteps: Int = FXDataEngineConstants.maxSequenceBars,
        horizonMinutes: Int = 1
    ) -> MLTensorContextDescriptor {
        let cap = max(min(maxSteps, FXDataEngineConstants.maxSequenceBars), 4)
        var modelDim = 16
        var heads = 2
        var stride = 1
        var patch = 1
        var dilation = 1
        var positionPenalty = 0.06

        switch style {
        case .recurrent:
            modelDim = min(18, FXDataEngineConstants.aiFeatures)
            heads = 2
        case .convolutional:
            modelDim = min(18, FXDataEngineConstants.aiFeatures)
            patch = 2
        case .transformer:
            modelDim = min(24, FXDataEngineConstants.aiFeatures)
            heads = 4
            patch = cap >= 24 ? 2 : 1
        case .stateSpace:
            modelDim = min(20, FXDataEngineConstants.aiFeatures)
            dilation = 2
        case .world:
            modelDim = min(22, FXDataEngineConstants.aiFeatures)
            heads = 4
            stride = cap >= 32 ? 2 : 1
            positionPenalty = 0.04
        case .generic:
            modelDim = min(16, FXDataEngineConstants.aiFeatures)
        }

        if horizonMinutes >= 60 {
            stride = max(stride, 2)
        }
        return MLTensorContextDescriptor(
            modelDim: modelDim,
            hiddenDim: FXDataEngineConstants.aiMLPHidden,
            headCount: heads,
            sequenceCapacity: cap,
            stride: stride,
            patchSize: patch,
            dilation: dilation,
            positionStepPenalty: positionPenalty
        )
    }

    public static func sequenceRuntimeDescriptor(
        dims: MLTensorContextDescriptor,
        normalize: Bool = true,
        includeCurrent: Bool = true
    ) -> MLSequenceRuntimeDescriptor {
        MLSequenceRuntimeDescriptor(
            maxSteps: dims.sequenceCapacity,
            stride: dims.stride,
            patchSize: dims.patchSize,
            normalize: normalize,
            includeCurrent: includeCurrent,
            positionStepPenalty: dims.positionStepPenalty
        )
    }
}

public protocol ExternalMLBackend: Sendable {
    var descriptor: MLBackendDescriptor { get }
    func predict(_ payload: MLInferencePayload) async throws -> PredictionV4
    func train(_ payload: MLTrainingPayload) async throws
}

public struct RemoteRPCMLBackendConfiguration: Codable, Hashable, Sendable {
    public let endpoint: String
    public let authToken: String?
    public let timeoutSeconds: TimeInterval

    public init(
        endpoint: String,
        authToken: String? = nil,
        timeoutSeconds: TimeInterval = 10.0
    ) {
        self.endpoint = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedToken = authToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.authToken = trimmedToken?.isEmpty == false ? trimmedToken : nil
        self.timeoutSeconds = timeoutSeconds.isFinite && timeoutSeconds > 0.0 ? min(timeoutSeconds, 300.0) : 10.0
    }

    public func validatedURL() throws -> URL {
        guard !endpoint.isEmpty else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint must not be empty")
        }
        guard endpoint.rangeOfCharacter(from: .newlines) == nil, !endpoint.contains("\0") else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint contains invalid control characters")
        }
        guard let url = URL(string: endpoint), let scheme = url.scheme?.lowercased() else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint is not a valid URL")
        }
        guard scheme == "https" || scheme == "http" else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint must use http or https")
        }
        guard url.host?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint must include a host")
        }
        guard url.fragment == nil else {
            throw FXDataEngineError.externalBackend("Remote RPC endpoint must not include a URL fragment")
        }
        return url
    }

    public func validateAuthToken() throws {
        guard let authToken else { return }
        guard authToken.rangeOfCharacter(from: .newlines) == nil, !authToken.contains("\0") else {
            throw FXDataEngineError.externalBackend("Remote RPC auth token contains invalid control characters")
        }
    }
}

public struct RemoteRPCMLBackendRequest: Codable, Hashable, Sendable {
    public let apiVersion: Int
    public let operation: String
    public let inference: MLInferencePayload

    public init(
        operation: String = "predict",
        inference: MLInferencePayload,
        apiVersion: Int = FXDataEngineConstants.latestPluginAPIVersion
    ) {
        self.apiVersion = apiVersion
        self.operation = operation
        self.inference = inference
    }
}

public struct RemoteRPCMLBackendResponse: Codable, Hashable, Sendable {
    public let apiVersion: Int
    public let ok: Bool
    public let prediction: PredictionV4?
    public let error: String?
    public let modelIdentifier: String?
    public let modelVersion: String?
    public let modelSha256: String?
    public let latencyMilliseconds: Double?
    public let serverTimestampUTC: Int64?

    public init(
        apiVersion: Int = FXDataEngineConstants.latestPluginAPIVersion,
        ok: Bool,
        prediction: PredictionV4?,
        error: String? = nil,
        modelIdentifier: String? = nil,
        modelVersion: String? = nil,
        modelSha256: String? = nil,
        latencyMilliseconds: Double? = nil,
        serverTimestampUTC: Int64? = nil
    ) {
        self.apiVersion = apiVersion
        self.ok = ok
        self.prediction = prediction
        self.error = error
        self.modelIdentifier = modelIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.modelVersion = modelVersion?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.modelSha256 = modelSha256?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.latencyMilliseconds = latencyMilliseconds.map { max(0.0, fxSafeFinite($0)) }
        self.serverTimestampUTC = serverTimestampUTC
    }
}

public protocol RemoteRPCMLBackendTransport: Sendable {
    func send(
        _ request: RemoteRPCMLBackendRequest,
        configuration: RemoteRPCMLBackendConfiguration
    ) throws -> RemoteRPCMLBackendResponse
}

public struct URLSessionRemoteRPCMLBackendTransport: RemoteRPCMLBackendTransport {
    public init() {}

    public func send(
        _ request: RemoteRPCMLBackendRequest,
        configuration: RemoteRPCMLBackendConfiguration
    ) throws -> RemoteRPCMLBackendResponse {
        let url = try configuration.validatedURL()
        try configuration.validateAuthToken()
        let body = try JSONEncoder().encode(request)
        var urlRequest = URLRequest(url: url, timeoutInterval: configuration.timeoutSeconds)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Accept")
        if let authToken = configuration.authToken {
            urlRequest.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.httpBody = body

        let resultBox = RemoteRPCTransportResultBox()
        let termination = DispatchSemaphore(value: 0)
        let task = URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            if let error {
                resultBox.set(.failure(error))
            } else {
                resultBox.set(.success((data ?? Data(), response)))
            }
            termination.signal()
        }
        task.resume()
        if termination.wait(timeout: .now() + configuration.timeoutSeconds) == .timedOut {
            task.cancel()
            throw FXDataEngineError.externalBackend(
                "Remote RPC inference timed out after \(String(format: "%.2f", configuration.timeoutSeconds)) seconds"
            )
        }

        let (responseData, urlResponse) = try resultBox.result().get()
        guard let httpResponse = urlResponse as? HTTPURLResponse else {
            throw FXDataEngineError.externalBackend("Remote RPC inference did not return an HTTP response")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let message = String(data: responseData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .prefix(512)
                .description
                .nilIfEmpty
                ?? "Remote RPC inference failed with HTTP status \(httpResponse.statusCode)"
            throw FXDataEngineError.externalBackend(message)
        }
        return try JSONDecoder().decode(RemoteRPCMLBackendResponse.self, from: responseData)
    }
}

private final class RemoteRPCTransportResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Result<(Data, URLResponse?), Error>?

    func set(_ result: Result<(Data, URLResponse?), Error>) {
        lock.lock()
        storage = result
        lock.unlock()
    }

    func result() throws -> Result<(Data, URLResponse?), Error> {
        lock.lock()
        defer { lock.unlock() }
        guard let storage else {
            throw FXDataEngineError.externalBackend("Remote RPC inference did not complete")
        }
        return storage
    }
}

public struct RemoteRPCMLBackendBridge: ExternalMLBackend {
    public let descriptor: MLBackendDescriptor
    private let configuration: RemoteRPCMLBackendConfiguration
    private let transport: any RemoteRPCMLBackendTransport
    private let configurationError: FXDataEngineError?

    public init(
        modelIdentifier: String,
        configuration: RemoteRPCMLBackendConfiguration,
        transport: any RemoteRPCMLBackendTransport = URLSessionRemoteRPCMLBackendTransport()
    ) {
        self.configuration = configuration
        self.transport = transport
        self.configurationError = Self.configurationError(configuration: configuration)
        self.descriptor = MLBackendDescriptor(
            mode: .remoteRPC(endpoint: configuration.endpoint),
            modelIdentifier: modelIdentifier,
            supportsTraining: false,
            supportsInference: true,
            usesVolumeFeatures: true
        )
    }

    public func predict(_ payload: MLInferencePayload) async throws -> PredictionV4 {
        try await Task.detached(priority: .utility) {
            try self.predictSynchronously(payload)
        }.value
    }

    public func train(_ payload: MLTrainingPayload) async throws {
        try trainSynchronously(payload)
    }

    public func predictSynchronously(_ payload: MLInferencePayload) throws -> PredictionV4 {
        try validateConfiguration()
        try payload.validateLatestAPI()
        guard payload.framework == .remoteRPC else {
            throw FXDataEngineError.validation("remoteRPCPayload.framework")
        }
        guard payload.modelIdentifier == descriptor.modelIdentifier else {
            throw FXDataEngineError.validation("remoteRPCPayload.modelIdentifier")
        }
        let request = RemoteRPCMLBackendRequest(inference: payload)
        let response = try transport.send(request, configuration: configuration)
        try Self.validateLatestAPI(response)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Remote RPC backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
        if let responseModel = response.modelIdentifier, responseModel != descriptor.modelIdentifier {
            throw FXDataEngineError.externalBackend(
                "Remote RPC backend returned modelIdentifier \(responseModel); expected \(descriptor.modelIdentifier)"
            )
        }
        guard let prediction = response.prediction else {
            throw FXDataEngineError.externalBackend("Remote RPC backend did not return a prediction")
        }
        try prediction.validate()
        return prediction
    }

    public func trainSynchronously(_ payload: MLTrainingPayload) throws {
        try validateConfiguration()
        try payload.validateLatestAPI()
        throw FXDataEngineError.externalBackend("Remote RPC training is not supported")
    }

    private func validateConfiguration() throws {
        if let configurationError {
            throw configurationError
        }
    }

    private static func configurationError(configuration: RemoteRPCMLBackendConfiguration) -> FXDataEngineError? {
        do {
            _ = try configuration.validatedURL()
            try configuration.validateAuthToken()
            return nil
        } catch let error as FXDataEngineError {
            return error
        } catch {
            return .externalBackend(String(describing: error))
        }
    }

    private static func validateLatestAPI(_ response: RemoteRPCMLBackendResponse) throws {
        guard response.apiVersion == FXDataEngineConstants.latestPluginAPIVersion else {
            throw FXDataEngineError.externalBackend(
                "Remote RPC backend API version \(response.apiVersion) is not supported; expected \(FXDataEngineConstants.latestPluginAPIVersion)"
            )
        }
    }
}

public struct PythonMLBackendBridge: ExternalMLBackend {
    public let descriptor: MLBackendDescriptor
    private let executable: String
    private let module: String
    private let environment: [String: String]
    private let timeoutSeconds: TimeInterval
    private let configurationError: FXDataEngineError?

    public init(
        framework: MLFramework,
        executable: String = "python3",
        module: String,
        modelIdentifier: String,
        environment: [String: String] = [:],
        timeoutSeconds: TimeInterval = 30.0
    ) {
        self.executable = executable
        self.module = module
        self.environment = environment
        self.timeoutSeconds = timeoutSeconds.isFinite && timeoutSeconds > 0.0 ? timeoutSeconds : 30.0
        self.configurationError = Self.configurationError(
            framework: framework,
            executable: executable,
            module: module,
            environment: environment
        )
        self.descriptor = MLBackendDescriptor(
            mode: .externalPython(framework: framework, executable: executable, module: module),
            modelIdentifier: modelIdentifier
        )
    }

    public func predict(_ payload: MLInferencePayload) async throws -> PredictionV4 {
        try validateConfiguration()
        try payload.validateLatestAPI()
        let command = PythonMLBackendCommand(operation: "predict", inference: payload, training: nil)
        let response = try await run(command)
        try Self.validateLatestAPI(response)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
        guard let prediction = response.prediction else {
            throw FXDataEngineError.externalBackend("Python backend did not return a prediction")
        }
        try prediction.validate()
        return prediction
    }

    public func train(_ payload: MLTrainingPayload) async throws {
        try validateConfiguration()
        try payload.validateLatestAPI()
        let command = PythonMLBackendCommand(operation: "train", inference: nil, training: payload)
        let response = try await run(command)
        try Self.validateLatestAPI(response)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
    }

    public func predictSynchronously(_ payload: MLInferencePayload) throws -> PredictionV4 {
        try validateConfiguration()
        try payload.validateLatestAPI()
        let command = PythonMLBackendCommand(operation: "predict", inference: payload, training: nil)
        let response = try runSynchronously(command)
        try Self.validateLatestAPI(response)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
        guard let prediction = response.prediction else {
            throw FXDataEngineError.externalBackend("Python backend did not return a prediction")
        }
        try prediction.validate()
        return prediction
    }

    public func trainSynchronously(_ payload: MLTrainingPayload) throws {
        try validateConfiguration()
        try payload.validateLatestAPI()
        let command = PythonMLBackendCommand(operation: "train", inference: nil, training: payload)
        let response = try runSynchronously(command)
        try Self.validateLatestAPI(response)
        guard response.ok else {
            throw FXDataEngineError.externalBackend(response.error ?? "Python backend returned failure")
        }
        if let error = response.error {
            throw FXDataEngineError.externalBackend(error)
        }
    }

    private func validateConfiguration() throws {
        if let configurationError {
            throw configurationError
        }
    }

    private static func configurationError(
        framework: MLFramework,
        executable: String,
        module: String,
        environment: [String: String]
    ) -> FXDataEngineError? {
        guard framework == .pyTorch || framework == .tensorFlow || framework == .foundationNLP || framework == .onnxRuntime else {
            return .externalBackend(
                "Python bridge does not support \(framework.rawValue); use pyTorch, tensorFlow, foundationNLP, or onnxRuntime"
            )
        }
        let trimmedExecutable = executable.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedModule = module.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedExecutable.isEmpty else {
            return .externalBackend("Python bridge executable must not be empty")
        }
        guard trimmedExecutable.rangeOfCharacter(from: .newlines) == nil, !trimmedExecutable.contains("\0") else {
            return .externalBackend("Python bridge executable contains invalid control characters")
        }
        guard !trimmedModule.isEmpty else {
            return .externalBackend("Python bridge module must not be empty")
        }
        guard trimmedModule.rangeOfCharacter(from: .newlines) == nil, !trimmedModule.contains("\0") else {
            return .externalBackend("Python bridge module contains invalid control characters")
        }
        if trimmedModule.contains("/") || trimmedModule.hasSuffix(".py") {
            guard FileManager.default.fileExists(atPath: trimmedModule) else {
                return .externalBackend("Python bridge module path does not exist: \(trimmedModule)")
            }
        }
        for key in environment.keys.sorted() {
            let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedKey.isEmpty, trimmedKey == key, !trimmedKey.contains("="), !trimmedKey.contains("\0") else {
                return .externalBackend("Python bridge environment key is invalid: \(key)")
            }
        }
        for value in environment.values where value.contains("\0") {
            return .externalBackend("Python bridge environment value contains invalid control characters")
        }
        return nil
    }

    private func run(_ command: PythonMLBackendCommand) async throws -> PythonMLBackendResponse {
        try await Task.detached(priority: .utility) {
            try self.runProcess(command)
        }.value
    }

    private func runSynchronously(_ command: PythonMLBackendCommand) throws -> PythonMLBackendResponse {
        let input = try JSONEncoder().encode(command)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        if module.contains("/") || module.hasSuffix(".py") {
            process.arguments = [executable, module, command.operation]
        } else {
            process.arguments = [executable, "-m", module, command.operation]
        }
        process.environment = subprocessEnvironment()

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        let stdoutBuffer = PythonProcessOutputBuffer()
        let stderrBuffer = PythonProcessOutputBuffer()
        stdout.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            stdoutBuffer.append(data)
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            stderrBuffer.append(data)
        }
        let termination = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in
            termination.signal()
        }
        defer {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            process.terminationHandler = nil
        }

        try process.run()
        stdin.fileHandleForWriting.write(input)
        try stdin.fileHandleForWriting.close()
        try waitForExit(process, termination: termination)

        let output = stdoutBuffer.snapshot(appending: stdout.fileHandleForReading.readDataToEndOfFile())
        let errorData = stderrBuffer.snapshot(appending: stderr.fileHandleForReading.readDataToEndOfFile())
        guard process.terminationStatus == 0 else {
            let stderrMessage = String(data: errorData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let stdoutMessage = String(data: output, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = [stderrMessage, stdoutMessage]
                .compactMap { $0 }
                .first { !$0.isEmpty }
                ?? "Python backend failed with exit=\(process.terminationStatus)"
            throw FXDataEngineError.externalBackend(message)
        }
        return try JSONDecoder().decode(PythonMLBackendResponse.self, from: output)
    }

    private func runProcess(_ command: PythonMLBackendCommand) throws -> PythonMLBackendResponse {
        try runSynchronously(command)
    }

    private func subprocessEnvironment() -> [String: String] {
        let parent = ProcessInfo.processInfo.environment
        var resolved: [String: String] = [:]
        for key in ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"] {
            if let value = parent[key], !value.isEmpty {
                resolved[key] = value
            }
        }
        if resolved["PATH"]?.isEmpty != false {
            resolved["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        }
        for (key, value) in environment {
            resolved[key] = value
        }
        return resolved
    }

    private func waitForExit(_ process: Process, termination: DispatchSemaphore) throws {
        if termination.wait(timeout: .now() + timeoutSeconds) == .success {
            return
        }

        process.terminate()
        if termination.wait(timeout: .now() + 1.0) == .timedOut {
            #if canImport(Darwin)
            Darwin.kill(process.processIdentifier, SIGKILL)
            #endif
            _ = termination.wait(timeout: .now() + 1.0)
        }
        throw FXDataEngineError.externalBackend(
            "Python backend timed out after \(String(format: "%.2f", timeoutSeconds)) seconds"
        )
    }

    private static func validateLatestAPI(_ response: PythonMLBackendResponse) throws {
        guard response.apiVersion == FXDataEngineConstants.latestPluginAPIVersion else {
            throw FXDataEngineError.externalBackend(
                "Python backend API version \(response.apiVersion) is not supported; expected \(FXDataEngineConstants.latestPluginAPIVersion)"
            )
        }
    }
}

private final class PythonProcessOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        lock.unlock()
    }

    func snapshot(appending chunk: Data) -> Data {
        lock.lock()
        data.append(chunk)
        let snapshot = data
        lock.unlock()
        return snapshot
    }
}

private struct PythonMLBackendCommand: Codable, Sendable {
    let apiVersion: Int
    let operation: String
    let inference: MLInferencePayload?
    let training: MLTrainingPayload?

    init(
        operation: String,
        inference: MLInferencePayload?,
        training: MLTrainingPayload?,
        apiVersion: Int = FXDataEngineConstants.latestPluginAPIVersion
    ) {
        self.apiVersion = apiVersion
        self.operation = operation
        self.inference = inference
        self.training = training
    }
}

private struct PythonMLBackendResponse: Codable, Sendable {
    let apiVersion: Int
    let ok: Bool
    let prediction: PredictionV4?
    let error: String?
}

public enum MLBackendFactory {
    public static func inferencePayload(
        descriptor: MLBackendDescriptor,
        request: PredictRequestV4
    ) -> MLInferencePayload {
        let framework: MLFramework
        switch descriptor.mode {
        case .inProcess(let resolved):
            framework = resolved
        case .externalPython(let resolved, _, _):
            framework = resolved
        case .remoteRPC:
            framework = .remoteRPC
        }
        return MLInferencePayload(
            modelIdentifier: descriptor.modelIdentifier,
            framework: framework,
            dataHasVolume: request.context.dataHasVolume,
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            priceCostPoints: request.context.priceCostPoints,
            minMovePoints: request.context.minMovePoints,
            x: request.x,
            xWindow: request.xWindow,
            tokenizerContract: request.context.tokenizerContract,
            textEvents: request.context.textEvents
        )
    }

    public static func trainingPayload(
        descriptor: MLBackendDescriptor,
        request: TrainRequestV4
    ) -> MLTrainingPayload {
        let inference = MLInferencePayload(
            modelIdentifier: descriptor.modelIdentifier,
            framework: framework(for: descriptor),
            dataHasVolume: request.context.dataHasVolume,
            horizonMinutes: request.context.horizonMinutes,
            sequenceBars: request.context.sequenceBars,
            priceCostPoints: request.context.priceCostPoints,
            minMovePoints: request.context.minMovePoints,
            x: request.x,
            xWindow: request.xWindow,
            tokenizerContract: request.context.tokenizerContract,
            textEvents: request.context.textEvents
        )
        return MLTrainingPayload(inference: inference, request: request)
    }

    private static func framework(for descriptor: MLBackendDescriptor) -> MLFramework {
        switch descriptor.mode {
        case .inProcess(let resolved):
            return resolved
        case .externalPython(let resolved, _, _):
            return resolved
        case .remoteRPC:
            return .remoteRPC
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
