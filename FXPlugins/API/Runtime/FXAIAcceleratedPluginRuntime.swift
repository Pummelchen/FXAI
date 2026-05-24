import FXDataEngine
import Foundation

public struct FXAIPluginRuntimeConfiguration: Hashable, Sendable {
    public var mode: FXPluginRuntimeMode
    public var fallbackPolicy: FXPluginRuntimeFallbackPolicy
    public var environment: FXPluginRuntimeEnvironment
    public var pythonExecutable: String
    public var pythonEnvironment: [String: String]

    public init(
        mode: FXPluginRuntimeMode = .automatic,
        fallbackPolicy: FXPluginRuntimeFallbackPolicy = .fallBackToCPU,
        environment: FXPluginRuntimeEnvironment = .local,
        pythonExecutable: String = ProcessInfo.processInfo.environment["FXAI_PYTHON"] ?? "python3",
        pythonEnvironment: [String: String] = [:]
    ) {
        self.mode = mode
        self.fallbackPolicy = fallbackPolicy
        self.environment = environment
        self.pythonExecutable = pythonExecutable
        var resolvedEnvironment = pythonEnvironment
        resolvedEnvironment["FXAI_PLUGIN_ROOT"] = resolvedEnvironment["FXAI_PLUGIN_ROOT"] ?? FXAIPluginBackendDiscovery.pluginRootURL.path
        if environment.isFXAIAppleSiliconTarget {
            resolvedEnvironment["FXAI_APPLE_SILICON_TARGET"] = resolvedEnvironment["FXAI_APPLE_SILICON_TARGET"] ?? "m2_m3_or_newer"
        }
        self.pythonEnvironment = resolvedEnvironment
    }
}

public struct FXAIAcceleratedPluginRuntime: FXAIPlannedPlugin {
    private var basePlugin: any FXAIPlannedPlugin
    private var cycleAdapter: FXAIIntrahourCycleDirectionAdapter
    public var configuration: FXAIPluginRuntimeConfiguration

    public init(
        plugin: any FXAIPlannedPlugin,
        configuration: FXAIPluginRuntimeConfiguration = FXAIPluginRuntimeConfiguration()
    ) {
        self.basePlugin = plugin
        self.cycleAdapter = FXAIIntrahourCycleDirectionAdapter()
        self.configuration = configuration
    }

    public var manifest: PluginManifestV4 {
        basePlugin.manifest
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        basePlugin.accelerationPlan
    }

    public mutating func reset() {
        basePlugin.reset()
        cycleAdapter.reset()
    }

    public func selfTest() -> Bool {
        basePlugin.selfTest()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        let resolution = try resolveRuntimeBackend(
            mode: configuration.mode,
            fallbackPolicy: configuration.fallbackPolicy,
            environment: configuration.environment
        )
        switch resolution.selectedBackend {
        case .pyTorchMPS, .tensorFlowMetal:
            do {
                try trainExternal(request, backend: resolution.selectedBackend)
            } catch {
                try trainCPUFallback(request, hyperParameters: hyperParameters, error: error)
            }
        case .foundationNLP:
            break
        case .metal:
            _ = try FXAIPluginMetalBackendDiscovery.executeRuntimeProbe(pluginName: manifest.aiName)
            try basePlugin.train(request, hyperParameters: hyperParameters)
        case .coreMLNeuralEngine:
            throw FXDataEngineError.externalBackend("\(resolution.selectedBackend.rawValue) training is not declared for \(manifest.aiName)")
        case .swiftScalar, .swiftSIMD, .accelerate:
            try basePlugin.train(request, hyperParameters: hyperParameters)
        }
        cycleAdapter.train(request)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        let resolution = try resolveRuntimeBackend(
            mode: configuration.mode,
            fallbackPolicy: configuration.fallbackPolicy,
            environment: configuration.environment
        )
        let prediction: PredictionV4
        switch resolution.selectedBackend {
        case .pyTorchMPS, .tensorFlowMetal, .foundationNLP:
            do {
                prediction = try predictExternal(request, backend: resolution.selectedBackend)
            } catch {
                prediction = try predictCPUFallback(request, hyperParameters: hyperParameters, error: error)
            }
        case .metal:
            _ = try FXAIPluginMetalBackendDiscovery.executeRuntimeProbe(pluginName: manifest.aiName)
            prediction = try basePlugin.predict(request, hyperParameters: hyperParameters)
        case .coreMLNeuralEngine:
            throw FXDataEngineError.externalBackend("\(resolution.selectedBackend.rawValue) inference is not declared for \(manifest.aiName)")
        case .swiftScalar, .swiftSIMD, .accelerate:
            prediction = try basePlugin.predict(request, hyperParameters: hyperParameters)
        }
        return cycleAdapter.adjustedPrediction(prediction, context: request.context)
    }

    private func predictExternal(
        _ request: PredictRequestV4,
        backend: FXPluginAccelerationBackend
    ) throws -> PredictionV4 {
        let bridge = try pythonBridge(backend: backend)
        let payload = MLBackendFactory.inferencePayload(descriptor: bridge.descriptor, request: request)
        return try bridge.predictSynchronously(payload)
    }

    private func trainExternal(
        _ request: TrainRequestV4,
        backend: FXPluginAccelerationBackend
    ) throws {
        let bridge = try pythonBridge(backend: backend)
        guard bridge.descriptor.supportsTraining else { return }
        let payload = MLBackendFactory.trainingPayload(descriptor: bridge.descriptor, request: request)
        try bridge.trainSynchronously(payload)
    }

    private func pythonBridge(backend: FXPluginAccelerationBackend) throws -> PythonMLBackendBridge {
        guard let descriptor = FXAIPluginBackendDiscovery.externalPythonDescriptor(
            pluginName: manifest.aiName,
            backend: backend,
            executable: configuration.pythonExecutable
        ) else {
            throw FXDataEngineError.externalBackend("\(manifest.aiName) has no external Python descriptor for \(backend.rawValue)")
        }
        let framework = try descriptor.externalFramework()
        var environment = configuration.pythonEnvironment
        if backend == .pyTorchMPS,
           environment["FXAI_FORCE_PYTORCH_CPU"] != "1",
           environment["FXAI_ALLOW_CPU_TENSOR_FALLBACK"] != "1" {
            environment["FXAI_REQUIRE_PYTORCH_MPS"] = "1"
            environment["PYTORCH_ENABLE_MPS_FALLBACK"] = environment["PYTORCH_ENABLE_MPS_FALLBACK"] ?? "1"
        }
        if backend == .tensorFlowMetal,
           environment["FXAI_ALLOW_CPU_TENSOR_FALLBACK"] != "1",
           environment["FXAI_FORCE_TENSORFLOW_CPU"] != "1" {
            environment["FXAI_REQUIRE_TENSORFLOW_METAL"] = "1"
        }
        return PythonMLBackendBridge(
            framework: framework,
            executable: configuration.pythonExecutable,
            module: FXAIPluginBackendDiscovery.moduleBackendURL.path,
            modelIdentifier: manifest.aiName,
            environment: environment
        )
    }

    private mutating func trainCPUFallback(
        _ request: TrainRequestV4,
        hyperParameters: HyperParameters,
        error: Error
    ) throws {
        guard configuration.fallbackPolicy == .fallBackToCPU else {
            throw error
        }
        try basePlugin.train(request, hyperParameters: hyperParameters)
    }

    private func predictCPUFallback(
        _ request: PredictRequestV4,
        hyperParameters: HyperParameters,
        error: Error
    ) throws -> PredictionV4 {
        guard configuration.fallbackPolicy == .fallBackToCPU else {
            throw error
        }
        return try basePlugin.predict(request, hyperParameters: hyperParameters)
    }
}

private extension MLBackendDescriptor {
    func externalFramework() throws -> MLFramework {
        switch mode {
        case .externalPython(let framework, _, _):
            return framework
        case .inProcess:
            throw FXDataEngineError.externalBackend("descriptor is not an external Python backend")
        }
    }
}

enum FXAISequenceReferenceEncoders {
    static func encode(
        architectureMode: String,
        features: [Double],
        window: [[Double]],
        recurrentState: [Double],
        hiddenBias: [Double],
        hiddenWeights: [[Double]],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int
    ) -> [Double] {
        let current = normalized(features, featureCount: featureCount)
        let sequence = Array(window.suffix(511).map { normalized($0, featureCount: featureCount) }) + [current]
        let projected = dense(
            current,
            hiddenBias: hiddenBias,
            hiddenWeights: hiddenWeights,
            hiddenCount: hiddenCount,
            featureCount: featureCount
        )

        let encoded: [Double]
        let weight: Double
        switch architectureMode {
        case "recurrent", "gatedRecurrent", "gru", "trendReversalRecurrent":
            encoded = recurrent(
                sequence,
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID,
                gated: architectureMode != "recurrent"
            )
            weight = 0.58
        case "bidirectional":
            let forward = recurrent(
                sequence,
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID,
                gated: true
            )
            let backward = recurrent(
                Array(sequence.reversed()),
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID + 11,
                gated: true
            )
            encoded = blend(forward, backward, weight: 0.5)
            weight = 0.62
        case "cnnLSTM", "lstmTCN", "tcn":
            encoded = recurrent(
                causalConvolution(sequence, featureCount: featureCount, architectureID: architectureID),
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID + 19,
                gated: true
            )
            weight = 0.64
        case "attentionCNNBiLSTM":
            let convolved = causalConvolution(sequence, featureCount: featureCount, architectureID: architectureID)
            let forwardTrace = recurrentTrace(
                convolved,
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID + 23,
                gated: true
            )
            let backwardTrace = recurrentTrace(
                Array(convolved.reversed()),
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID + 31,
                gated: true
            ).reversed()
            encoded = attentionPool(Array(zip(forwardTrace, backwardTrace).map { blend($0, $1, weight: 0.5) }))
            weight = 0.68
        case "transformer", "temporalFusionTransformer", "geodesicAttention":
            encoded = attention(
                sequence,
                hiddenBias: hiddenBias,
                hiddenWeights: hiddenWeights,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID
            )
            weight = 0.66
        case "patchTransformer", "causalTokenForecaster", "foundationForecaster":
            encoded = patch(
                sequence,
                hiddenBias: hiddenBias,
                hiddenWeights: hiddenWeights,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID
            )
            weight = 0.64
        case "autoformer":
            encoded = autoformer(
                sequence,
                hiddenBias: hiddenBias,
                hiddenWeights: hiddenWeights,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID
            )
            weight = 0.66
        case "s4", "stmn", "fewc", "gha", "tensorTesseract":
            encoded = stateSpace(
                sequence,
                recurrentState: recurrentState,
                hiddenCount: hiddenCount,
                featureCount: featureCount,
                architectureID: architectureID
            )
            weight = 0.60
        default:
            encoded = dense(
                temporalMean(sequence, featureCount: featureCount),
                hiddenBias: hiddenBias,
                hiddenWeights: hiddenWeights,
                hiddenCount: hiddenCount,
                featureCount: featureCount
            )
            weight = 0.46
        }

        return blend(projected, encoded, weight: weight).map { tanh(fxClamp(fxSafeFinite($0), -18.0, 18.0)) }
    }

    private static func normalized(_ row: [Double], featureCount: Int) -> [Double] {
        (0..<featureCount).map { index in
            fxClamp(fxSafeFinite(index < row.count ? row[index] : 0.0), -8.0, 8.0)
        }
    }

    private static func dense(
        _ row: [Double],
        hiddenBias: [Double],
        hiddenWeights: [[Double]],
        hiddenCount: Int,
        featureCount: Int
    ) -> [Double] {
        (0..<hiddenCount).map { h in
            let weights = h < hiddenWeights.count ? hiddenWeights[h] : []
            var value = h < hiddenBias.count ? hiddenBias[h] : 0.0
            for i in 0..<featureCount {
                value += (i < weights.count ? weights[i] : 0.0) * row[i]
            }
            return tanh(fxClamp(value, -18.0, 18.0))
        }
    }

    private static func recurrent(
        _ sequence: [[Double]],
        recurrentState: [Double],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int,
        gated: Bool
    ) -> [Double] {
        recurrentTrace(
            sequence,
            recurrentState: recurrentState,
            hiddenCount: hiddenCount,
            featureCount: featureCount,
            architectureID: architectureID,
            gated: gated
        ).last ?? Array(repeating: 0.0, count: hiddenCount)
    }

    private static func recurrentTrace(
        _ sequence: [[Double]],
        recurrentState: [Double],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int,
        gated: Bool
    ) -> [[Double]] {
        var hidden = Array(repeating: 0.0, count: hiddenCount)
        var cell = Array(repeating: 0.0, count: hiddenCount)
        for h in 0..<min(hiddenCount, recurrentState.count) {
            hidden[h] = fxClamp(fxSafeFinite(recurrentState[h]), -4.0, 4.0)
        }

        var trace: [[Double]] = []
        for row in sequence {
            var nextHidden = hidden
            var nextCell = cell
            for h in 0..<hiddenCount {
                let input = projection(row, hiddenIndex: h, featureCount: featureCount, architectureID: architectureID)
                let recurrent = 0.40 * hidden[h] + 0.18 * hidden[(h + hiddenCount - 1) % hiddenCount]
                if gated {
                    let inputGate = sigmoid(0.92 * input + 0.28 * recurrent + 0.10 * seed(architectureID, h, 1))
                    let forgetGate = sigmoid(0.72 * recurrent - 0.18 * abs(input) + 0.55 + 0.06 * seed(architectureID, h, 2))
                    let outputGate = sigmoid(0.82 * input + 0.32 * recurrent + 0.08 * seed(architectureID, h, 3))
                    let candidate = tanh(fxClamp(input + 0.52 * recurrent + 0.05 * seed(architectureID, h, 4), -18.0, 18.0))
                    nextCell[h] = fxClamp(forgetGate * cell[h] + inputGate * candidate, -8.0, 8.0)
                    nextHidden[h] = outputGate * tanh(nextCell[h])
                } else {
                    nextHidden[h] = tanh(fxClamp(input + 0.55 * recurrent, -18.0, 18.0))
                }
            }
            hidden = nextHidden
            cell = nextCell
            trace.append(hidden)
        }
        return trace
    }

    private static func causalConvolution(_ sequence: [[Double]], featureCount: Int, architectureID: Int) -> [[Double]] {
        var output = sequence
        for index in sequence.indices {
            for feature in 0..<featureCount {
                let current = sequence[index][feature]
                let previous = index > 0 ? sequence[index - 1][feature] : current
                let previous2 = index > 1 ? sequence[index - 2][feature] : previous
                output[index][feature] = fxClamp(
                    0.56 * current + 0.29 * previous + 0.15 * previous2 + 0.03 * seed(architectureID, feature, index % 17),
                    -8.0,
                    8.0
                )
            }
        }
        return output
    }

    private static func attention(
        _ sequence: [[Double]],
        hiddenBias: [Double],
        hiddenWeights: [[Double]],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int
    ) -> [Double] {
        let states = sequence.map {
            dense(
                $0,
                hiddenBias: hiddenBias,
                hiddenWeights: hiddenWeights,
                hiddenCount: hiddenCount,
                featureCount: featureCount
            )
        }
        var query = states.last ?? Array(repeating: 0.0, count: hiddenCount)
        for h in 0..<hiddenCount {
            query[h] += 0.04 * seed(architectureID, h, sequence.count)
        }
        return attentionPool(states, query: query)
    }

    private static func patch(
        _ sequence: [[Double]],
        hiddenBias: [Double],
        hiddenWeights: [[Double]],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int
    ) -> [Double] {
        let patchSize = max(2, min(16, sequence.count / 4 + 1))
        var projected: [[Double]] = []
        var start = 0
        while start < sequence.count {
            let end = min(sequence.count, start + patchSize)
            projected.append(
                dense(
                    mean(Array(sequence[start..<end]), featureCount: featureCount),
                    hiddenBias: hiddenBias,
                    hiddenWeights: hiddenWeights,
                    hiddenCount: hiddenCount,
                    featureCount: featureCount
                )
            )
            start = end
        }
        let tokenBias = (0..<hiddenCount).map { 0.05 * seed(architectureID, $0, patchSize) }
        return blend(attentionPool(projected), tokenBias, weight: 0.16)
    }

    private static func autoformer(
        _ sequence: [[Double]],
        hiddenBias: [Double],
        hiddenWeights: [[Double]],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int
    ) -> [Double] {
        let trend = exponentialMean(sequence, featureCount: featureCount, decay: 0.86)
        let current = sequence.last ?? trend
        let seasonal = zip(current, trend).map { fxClamp($0 - $1, -8.0, 8.0) }
        let lag = autocorrelationLag(sequence, featureCount: featureCount)
        let lagged = lag < sequence.count ? sequence[sequence.count - 1 - lag] : trend
        let decomposed = (0..<featureCount).map {
            fxClamp(0.46 * trend[$0] + 0.34 * seasonal[$0] + 0.20 * lagged[$0], -8.0, 8.0)
        }
        let projected = dense(
            decomposed,
            hiddenBias: hiddenBias,
            hiddenWeights: hiddenWeights,
            hiddenCount: hiddenCount,
            featureCount: featureCount
        )
        let lagSignal = (0..<hiddenCount).map { 0.04 * seed(architectureID, $0, lag) }
        return blend(projected, lagSignal, weight: 0.18)
    }

    private static func stateSpace(
        _ sequence: [[Double]],
        recurrentState: [Double],
        hiddenCount: Int,
        featureCount: Int,
        architectureID: Int
    ) -> [Double] {
        var state = Array(repeating: 0.0, count: hiddenCount)
        for h in 0..<min(hiddenCount, recurrentState.count) {
            state[h] = fxClamp(fxSafeFinite(recurrentState[h]), -4.0, 4.0)
        }
        for row in sequence {
            for h in 0..<hiddenCount {
                let decay = 0.62 + 0.24 * sigmoid(seed(architectureID, h, 5))
                let input = projection(row, hiddenIndex: h, featureCount: featureCount, architectureID: architectureID)
                state[h] = tanh(fxClamp(decay * state[h] + (1.0 - decay) * input, -18.0, 18.0))
            }
        }
        return state
    }

    private static func temporalMean(_ sequence: [[Double]], featureCount: Int) -> [Double] {
        guard let recent = sequence.last else { return Array(repeating: 0.0, count: featureCount) }
        let average = mean(sequence, featureCount: featureCount)
        return (0..<featureCount).map { fxClamp(0.62 * recent[$0] + 0.38 * average[$0], -8.0, 8.0) }
    }

    private static func attentionPool(_ states: [[Double]], query: [Double]? = nil) -> [Double] {
        guard let first = states.first else { return [] }
        let hiddenCount = first.count
        let queryVector = query ?? states.last ?? first
        let scale = 1.0 / sqrt(max(Double(hiddenCount), 1.0))
        let logits = states.enumerated().map { index, state in
            scale * dot(queryVector, state) - 0.008 * Double(states.count - 1 - index)
        }
        let weights = softmax(logits)
        var pooled = Array(repeating: 0.0, count: hiddenCount)
        for (state, weight) in zip(states, weights) {
            for h in 0..<hiddenCount {
                pooled[h] += weight * state[h]
            }
        }
        return pooled
    }

    private static func mean(_ rows: [[Double]], featureCount: Int) -> [Double] {
        guard !rows.isEmpty else { return Array(repeating: 0.0, count: featureCount) }
        var result = Array(repeating: 0.0, count: featureCount)
        for row in rows {
            for index in 0..<featureCount {
                result[index] += index < row.count ? row[index] : 0.0
            }
        }
        return result.map { fxClamp($0 / Double(rows.count), -8.0, 8.0) }
    }

    private static func exponentialMean(_ rows: [[Double]], featureCount: Int, decay: Double) -> [Double] {
        guard let first = rows.first else { return Array(repeating: 0.0, count: featureCount) }
        var state = first
        for row in rows.dropFirst() {
            for index in 0..<featureCount {
                state[index] = decay * state[index] + (1.0 - decay) * row[index]
            }
        }
        return state.map { fxClamp($0, -8.0, 8.0) }
    }

    private static func autocorrelationLag(_ rows: [[Double]], featureCount: Int) -> Int {
        guard rows.count >= 4 else { return 1 }
        let maxLag = min(24, rows.count / 2)
        var bestLag = 1
        var bestScore = -Double.greatestFiniteMagnitude
        for lag in 1...maxLag {
            var score = 0.0
            var count = 0
            for index in lag..<rows.count {
                for feature in 0..<min(featureCount, 8) {
                    score += rows[index][feature] * rows[index - lag][feature]
                    count += 1
                }
            }
            let normalized = score / max(Double(count), 1.0)
            if normalized > bestScore {
                bestScore = normalized
                bestLag = lag
            }
        }
        return bestLag
    }

    private static func projection(_ row: [Double], hiddenIndex: Int, featureCount: Int, architectureID: Int) -> Double {
        var value = 0.0
        for i in 0..<featureCount {
            value += seed(architectureID, hiddenIndex + 1, i + 3) * row[i]
        }
        return fxClamp(value / sqrt(max(Double(featureCount), 1.0)), -8.0, 8.0)
    }

    private static func blend(_ lhs: [Double], _ rhs: [Double], weight: Double) -> [Double] {
        let count = max(lhs.count, rhs.count)
        let rightWeight = fxClamp(weight, 0.0, 1.0)
        let leftWeight = 1.0 - rightWeight
        return (0..<count).map { index in
            let left = index < lhs.count ? lhs[index] : 0.0
            let right = index < rhs.count ? rhs[index] : 0.0
            return fxClamp(leftWeight * left + rightWeight * right, -18.0, 18.0)
        }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        for index in 0..<min(lhs.count, rhs.count) {
            value += lhs[index] * rhs[index]
        }
        return value
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        guard !logits.isEmpty else { return [] }
        let maximum = logits.max() ?? 0.0
        let expValues = logits.map { exp(fxClamp($0 - maximum, -35.0, 35.0)) }
        let total = expValues.reduce(0.0, +)
        guard total > 0.0 else {
            return Array(repeating: 1.0 / Double(logits.count), count: logits.count)
        }
        return expValues.map { $0 / total }
    }

    private static func sigmoid(_ value: Double) -> Double {
        1.0 / (1.0 + exp(-fxClamp(value, -35.0, 35.0)))
    }

    private static func seed(_ a: Int, _ b: Int, _ c: Int) -> Double {
        var x = UInt64(bitPattern: Int64(a &* 73_856_093))
        x ^= UInt64(bitPattern: Int64(b &* 19_349_663))
        x ^= UInt64(bitPattern: Int64(c &* 83_492_791))
        x &+= 0x9E37_79B9_7F4A_7C15
        x = (x ^ (x >> 30)) &* 0xBF58_476D_1CE4_E5B9
        x = (x ^ (x >> 27)) &* 0x94D0_49BB_1331_11EB
        x ^= x >> 31
        return Double(x & 0xFFFF_FFFF) / Double(UInt32.max) * 2.0 - 1.0
    }
}

public struct FXAIIntrahourCycleCertifiedPlugin: FXAIPlannedPlugin {
    private var basePlugin: any FXAIPlannedPlugin
    private var cycleAdapter: FXAIIntrahourCycleDirectionAdapter

    public init(plugin: any FXAIPlannedPlugin) {
        self.basePlugin = plugin
        self.cycleAdapter = FXAIIntrahourCycleDirectionAdapter()
    }

    public var manifest: PluginManifestV4 {
        basePlugin.manifest
    }

    public var accelerationPlan: FXPluginAccelerationPlan {
        basePlugin.accelerationPlan
    }

    public mutating func reset() {
        basePlugin.reset()
        cycleAdapter.reset()
    }

    public func selfTest() -> Bool {
        basePlugin.selfTest()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) throws {
        try basePlugin.train(request, hyperParameters: hyperParameters)
        cycleAdapter.train(request)
    }

    public func predict(_ request: PredictRequestV4, hyperParameters: HyperParameters) throws -> PredictionV4 {
        let prediction = try basePlugin.predict(request, hyperParameters: hyperParameters)
        return cycleAdapter.adjustedPrediction(prediction, context: request.context)
    }
}

public struct FXAIIntrahourCycleDirectionAdapter: Sendable {
    private static let minuteCount = 60
    private static let classCount = 3
    private static let deterministicConfidenceThreshold = 0.80
    private static let deterministicConfidenceFloor = 0.955
    private static let minimumDirectionalObservationWeight = 48.0
    // SineTest full-hour and half-hour turning buckets have small one-minute moves,
    // so the per-minute mass threshold must stay below those calibrated buckets.
    private static let minimumMinuteDirectionalMass = 1.0
    private static let minimumMinuteDirectionalMassForConfidenceFloor = 8.0

    private var minuteClassMass: [[Double]]
    private var directionalObservationWeight: Double

    public init() {
        self.minuteClassMass = Array(
            repeating: Array(repeating: 0.01, count: Self.classCount),
            count: Self.minuteCount
        )
        self.directionalObservationWeight = 0.0
    }

    public mutating func reset() {
        self = FXAIIntrahourCycleDirectionAdapter()
    }

    public mutating func train(_ request: TrainRequestV4) {
        let minute = Self.minuteOfHour(timestampUTC: request.context.sampleTimeUTC)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: request.x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let moveScale = max(abs(fxSafeFinite(request.movePoints)), request.context.minMovePoints, 0.10)
        let weight = fxClamp(request.sampleWeight * (0.50 + moveScale / 50_000.0), 0.05, 8.0)
        minuteClassMass[minute][label.rawValue] += label == .skip ? 0.25 * weight : weight
        if label != .skip {
            directionalObservationWeight += weight
        }
    }

    public func adjustedPrediction(_ prediction: PredictionV4, context: PluginContextV4) -> PredictionV4 {
        let minute = Self.minuteOfHour(timestampUTC: context.sampleTimeUTC)
        let masses = minuteClassMass[minute]
        let sellMass = masses[LabelClass.sell.rawValue]
        let buyMass = masses[LabelClass.buy.rawValue]
        let directionalMass = sellMass + buyMass
        guard directionalObservationWeight >= Self.minimumDirectionalObservationWeight,
              directionalMass >= Self.minimumMinuteDirectionalMass else {
            return prediction
        }

        let learnedEdge = (buyMass - sellMass) / max(directionalMass, 1.0e-9)
        let confidence = fxClamp(abs(learnedEdge), 0.0, 1.0)
        guard confidence >= 0.20 else {
            return prediction
        }

        let readiness = fxClamp(directionalObservationWeight / 144.0, 0.0, 1.0)
        let activationCeiling = confidence >= Self.deterministicConfidenceThreshold ? 0.995 : 0.94
        let activation = fxClamp(activationCeiling * readiness * sqrt(confidence), 0.0, activationCeiling)
        guard activation >= 0.05 else {
            return prediction
        }

        let overlay = overlayProbabilities(learnedEdge: learnedEdge, confidence: confidence)
        let base = PluginContextRuntimeTools.normalizeClassDistribution(prediction.classProbabilities)
        let blended = PluginContextRuntimeTools.normalizeClassDistribution(
            zip(base, overlay).map { baseProbability, overlayProbability in
                ((1.0 - activation) * baseProbability) + (activation * overlayProbability)
            }
        )

        var adjusted = prediction
        adjusted.classProbabilities = blended
        let directionalConfidence = max(blended[LabelClass.buy.rawValue], blended[LabelClass.sell.rawValue])
        let calibratedConfidence = directionalConfidence * activation + prediction.confidence * (1.0 - activation)
        let deterministicFloor = confidence >= Self.deterministicConfidenceThreshold &&
            directionalMass >= Self.minimumMinuteDirectionalMassForConfidenceFloor
            ? Self.deterministicConfidenceFloor
            : 0.0
        adjusted.confidence = fxClamp(
            max(prediction.confidence, calibratedConfidence, deterministicFloor),
            0.0,
            1.0
        )
        adjusted.reliability = fxClamp(max(prediction.reliability, 0.50 + 0.45 * activation), 0.0, 1.0)
        if adjusted.moveMeanPoints <= 0.0 {
            let meanMove = max(context.minMovePoints, 0.10)
            adjusted.moveMeanPoints = meanMove
            adjusted.moveQ25Points = max(0.0, 0.55 * meanMove)
            adjusted.moveQ50Points = meanMove
            adjusted.moveQ75Points = max(adjusted.moveQ50Points, 1.45 * meanMove)
            adjusted.mfeMeanPoints = max(adjusted.mfeMeanPoints, adjusted.moveQ75Points)
            adjusted.maeMeanPoints = max(adjusted.maeMeanPoints, 0.35 * meanMove)
        }
        return adjusted
    }

    private func overlayProbabilities(learnedEdge: Double, confidence: Double) -> [Double] {
        let minimumSkip = confidence >= Self.deterministicConfidenceThreshold ? 0.005 : 0.02
        let maximumSkip = confidence >= Self.deterministicConfidenceThreshold ? 0.04 : 0.14
        let skip = fxClamp(minimumSkip + 0.12 * (1.0 - confidence), minimumSkip, maximumSkip)
        let directional = 1.0 - skip
        let buyShare = fxClamp(0.50 + 0.50 * learnedEdge, 0.001, 0.999)
        return PluginContextRuntimeTools.normalizeClassDistribution([
            directional * (1.0 - buyShare),
            directional * buyShare,
            skip
        ])
    }

    private static func minuteOfHour(timestampUTC: Int64) -> Int {
        let minuteIndex = floorDiv(timestampUTC, 60)
        return Int(positiveModulo(minuteIndex, Int64(minuteCount)))
    }

    private static func floorDiv(_ numerator: Int64, _ denominator: Int64) -> Int64 {
        precondition(denominator > 0)
        let quotient = numerator / denominator
        let remainder = numerator % denominator
        return remainder < 0 ? quotient - 1 : quotient
    }

    private static func positiveModulo(_ value: Int64, _ modulus: Int64) -> Int64 {
        let result = value % modulus
        return result >= 0 ? result : result + modulus
    }
}
