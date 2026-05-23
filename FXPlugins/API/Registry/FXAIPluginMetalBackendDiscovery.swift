import FXDataEngine
import Foundation

public struct FXAIPluginMetalKernelBundle: Codable, Hashable, Sendable {
    public let pluginName: String
    public let sourcePath: String
    public let kernelSource: String
    public let functionNames: [String]

    public init(pluginName: String, sourcePath: String, kernelSource: String, functionNames: [String]) {
        self.pluginName = pluginName
        self.sourcePath = sourcePath
        self.kernelSource = kernelSource
        self.functionNames = functionNames
    }
}

public struct FXAIPluginMetalRuntimeProbeResult: Codable, Hashable, Sendable {
    public let pluginName: String
    public let sourcePath: String
    public let functionName: String
    public let executionResult: MetalFloatKernelExecutionResult
    public let expectedOutput: [Float]

    public init(
        pluginName: String,
        sourcePath: String,
        functionName: String,
        executionResult: MetalFloatKernelExecutionResult,
        expectedOutput: [Float]
    ) {
        self.pluginName = pluginName
        self.sourcePath = sourcePath
        self.functionName = functionName
        self.executionResult = executionResult
        self.expectedOutput = expectedOutput
    }
}

public enum FXAIPluginMetalBackendDiscovery {
    public static func metalDirectoryURL(pluginName: String) -> URL {
        FXAIPluginBackendDiscovery.pluginRootURL
            .appendingPathComponent(pluginName)
            .appendingPathComponent("Metal")
    }

    public static func metalKernelBundles(pluginName: String) throws -> [FXAIPluginMetalKernelBundle] {
        let directory = metalDirectoryURL(pluginName: pluginName)
        guard FileManager.default.fileExists(atPath: directory.path) else {
            return []
        }
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
            .filter { $0.pathExtension == "swift" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        return try files.compactMap { file in
            let text = try String(contentsOf: file, encoding: .utf8)
            guard let kernelSource = extractKernelSource(from: text) else {
                return nil
            }
            let functions = extractKernelFunctionNames(from: kernelSource)
            guard !functions.isEmpty else {
                throw FXDataEngineError.validation("metal.\(pluginName).\(file.lastPathComponent).noKernelFunctions")
            }
            return FXAIPluginMetalKernelBundle(
                pluginName: pluginName,
                sourcePath: file.path,
                kernelSource: kernelSource,
                functionNames: functions
            )
        }
    }

    public static func metalKernelBundlesForDeclaredPlugins() throws -> [FXAIPluginMetalKernelBundle] {
        try FXAIPluginRegistry.accelerationPlans()
            .filter { $0.declares(.metal) }
            .flatMap { try metalKernelBundles(pluginName: $0.pluginName) }
            .sorted {
                if $0.pluginName == $1.pluginName {
                    return $0.sourcePath < $1.sourcePath
                }
                return $0.pluginName < $1.pluginName
            }
    }

    public static func compileDeclaredPluginKernels() throws -> [MetalKernelCompilationResult] {
        try metalKernelBundlesForDeclaredPlugins().flatMap { bundle in
            try bundle.functionNames.map { functionName in
                try MetalKernelCompiler.compile(
                    source: bundle.kernelSource,
                    functionName: functionName,
                    sourceLabel: "\(bundle.pluginName).\(functionName)"
                )
            }
        }
    }

    public static func executeRuntimeProbe(pluginName: String) throws -> MetalFloatKernelExecutionResult {
        try executePluginKernelProbe(pluginName: pluginName).executionResult
    }

    public static func executeGenericRuntimeProbe(pluginName: String) throws -> MetalFloatKernelExecutionResult {
        try MetalKernelCompiler.executeUnaryFloatKernel(
            source: runtimeProbeKernelSource,
            functionName: "fxai_plugin_runtime_probe",
            input: [0.25, -1.0, 2.0, 4.0],
            sourceLabel: "\(pluginName).generic-runtime-probe"
        )
    }

    public static func executePluginKernelProbe(pluginName: String) throws -> FXAIPluginMetalRuntimeProbeResult {
        let bundles = try metalKernelBundles(pluginName: pluginName)
        guard !bundles.isEmpty else {
            throw FXDataEngineError.validation("metal.\(pluginName).noKernelBundles")
        }
        for bundle in bundles {
            for functionName in bundle.functionNames.sorted(by: probeSort) {
                if let result = try executeKnownProbe(bundle: bundle, functionName: functionName) {
                    return result
                }
            }
        }
        throw FXDataEngineError.validation("metal.\(pluginName).noRunnableProbeKernel")
    }

    private static func extractKernelSource(from text: String) -> String? {
        guard let marker = text.range(of: "kernelSource = \"\"\"") else {
            return nil
        }
        let remainder = text[marker.upperBound...]
        guard let end = remainder.range(of: "\"\"\"") else {
            return nil
        }
        return String(remainder[..<end.lowerBound])
    }

    private static func extractKernelFunctionNames(from source: String) -> [String] {
        let pattern = #"kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        let range = NSRange(source.startIndex..<source.endIndex, in: source)
        return regex.matches(in: source, range: range).compactMap { match in
            guard let functionRange = Range(match.range(at: 1), in: source) else {
                return nil
            }
            return String(source[functionRange])
        }
    }

    private static func probeSort(_ lhs: String, _ rhs: String) -> Bool {
        if probePriority(lhs) == probePriority(rhs) {
            return lhs < rhs
        }
        return probePriority(lhs) < probePriority(rhs)
    }

    private static func probePriority(_ functionName: String) -> Int {
        if functionName.contains("softmax3") || functionName.contains("sigmoid") { return 0 }
        if functionName == "dist_quantile_class_score" { return 1 }
        if functionName.contains("sequence_logits") ||
            functionName.contains("linear_logits") ||
            functionName.contains("linear_scores") ||
            functionName == "ai_mlp_class_logits" {
            return 2
        }
        if functionName.hasSuffix("_scores") { return 3 }
        if functionName.contains("pair_products") || functionName.contains("distances") { return 4 }
        if functionName.contains("window_score") || functionName.contains("mode_proxy") { return 5 }
        if functionName.contains("moving_average_cross") || functionName.contains("rule_m1sync") { return 6 }
        if functionName == "tree_rf_margin" { return 7 }
        return 100
    }

    private static func executeKnownProbe(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String
    ) throws -> FXAIPluginMetalRuntimeProbeResult? {
        if functionName.contains("softmax3") {
            let logits: [Float] = [0.25, 1.0, -0.5]
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray(logits), .outputFloat(count: 3)],
                outputArgumentIndex: 1,
                threadGrid: MetalKernelThreadGrid(width: 1),
                expected: softmax3(logits)
            )
        }
        if functionName.contains("sigmoid") {
            let margin: Float = 0.75
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray([margin]), .outputFloat(count: 1)],
                outputArgumentIndex: 1,
                threadGrid: MetalKernelThreadGrid(width: 1),
                expected: [1.0 / (1.0 + exp(-margin))]
            )
        }
        if functionName == "ai_mlp_class_logits" {
            let hidden: [Float] = [0.25, -0.5, 0.75, 1.0]
            let weights: [Float] = [
                0.10, 0.20, -0.10, 0.30, 0.05,
                -0.20, 0.10, 0.25, -0.15, 0.35,
                0.05, -0.30, 0.20, 0.15, -0.10
            ]
            let expected = (0..<3).map { classIndex in
                var value = weights[classIndex * (hidden.count + 1)]
                for i in 0..<hidden.count {
                    value += weights[classIndex * (hidden.count + 1) + i + 1] * hidden[i]
                }
                return clamp(value, -35, 35)
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [
                    .floatArray(hidden),
                    .floatArray(weights),
                    .outputFloat(count: 3),
                    .uintScalar(UInt32(hidden.count))
                ],
                outputArgumentIndex: 2,
                threadGrid: MetalKernelThreadGrid(width: 3),
                expected: expected
            )
        }
        if functionName.contains("sequence_logits") ||
            functionName.contains("linear_logits") ||
            functionName.contains("linear_scores") {
            let featureCount = 4
            let features: [Float] = [1.0, 0.5, -0.25, 0.75]
            let weights: [Float] = [
                0.10, 0.20, 0.30, 0.40,
                -0.20, 0.10, 0.50, -0.10,
                0.00, -0.40, 0.20, 0.30
            ]
            let usesFirstWeightAsBias = bundle.kernelSource.contains("float value = weights[classIndex * featureCount];")
            let expected = (0..<3).map { classIndex in
                var value = usesFirstWeightAsBias ? weights[classIndex * featureCount] : 0.0
                for i in 0..<featureCount {
                    value += weights[classIndex * featureCount + i] * features[i]
                }
                return clamp(value, -35, 35)
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [
                    .floatArray(features),
                    .floatArray(weights),
                    .outputFloat(count: 3),
                    .uintScalar(UInt32(featureCount))
                ],
                outputArgumentIndex: 2,
                threadGrid: MetalKernelThreadGrid(width: 3),
                expected: expected
            )
        }
        if functionName == "dist_quantile_class_score" {
            let features: [Float] = [0.2, -0.1, 0.4, 0.5, -0.3, 0.7, 0.1, -0.2]
            let weights: [Float] = [
                0.3, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.15,
                -0.2, 0.4, 0.1, -0.1, 0.2, 0.3, -0.05, 0.25,
                0.1, -0.3, 0.2, 0.35, -0.15, 0.05, 0.4, -0.2
            ]
            let expected = (0..<3).map { classIndex in
                var sum: Float = 0
                for i in 0..<8 {
                    sum += weights[classIndex * 8 + i] * features[i]
                }
                return clamp(sum, -30, 30)
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray(features), .floatArray(weights), .outputFloat(count: 3)],
                outputArgumentIndex: 2,
                threadGrid: MetalKernelThreadGrid(width: 3),
                expected: expected
            )
        }
        if functionName.hasSuffix("_scores") && bundle.kernelSource.contains("const uint stateDim = 16") {
            let state = (0..<16).map { Float($0 + 1) / 16.0 }
            let weights = (0..<48).map { Float(($0 % 7) - 3) * 0.04 }
            let expected = (0..<3).map { classIndex in
                var sum: Float = 0
                for i in 0..<16 {
                    sum += weights[classIndex * 16 + i] * state[i]
                }
                return clamp(sum, -12, 12)
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray(state), .floatArray(weights), .outputFloat(count: 3)],
                outputArgumentIndex: 2,
                threadGrid: MetalKernelThreadGrid(width: 3),
                expected: expected
            )
        }
        if functionName.contains("pair_products") {
            let featureCount = 4
            let features: [Float] = [1.0, 0.5, -0.25, 0.75]
            let expected = (0..<(featureCount * featureCount)).map { gid in
                let i = gid / featureCount
                let j = gid % featureCount
                return i < j && i > 0 ? features[i] * features[j] : 0.0
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray(features), .outputFloat(count: expected.count), .uintScalar(UInt32(featureCount))],
                outputArgumentIndex: 1,
                threadGrid: MetalKernelThreadGrid(width: expected.count),
                expected: expected
            )
        }
        if functionName == "mem_retrdiff_distances" {
            let dimension = 4
            let query: [Float] = [0.1, 0.3, -0.2, 0.4]
            let memory: [Float] = [
                0.1, 0.2, -0.1, 0.5,
                -0.3, 0.4, 0.2, 0.1,
                0.2, 0.3, -0.2, 0.4
            ]
            let expected = (0..<3).map { memoryIndex in
                var sum: Float = 0
                for i in 0..<dimension {
                    let d = query[i] - memory[memoryIndex * dimension + i]
                    sum += d * d
                }
                return sum
            }
            return try executeAndValidate(
                bundle: bundle,
                functionName: functionName,
                arguments: [.floatArray(query), .floatArray(memory), .outputFloat(count: 3), .uintScalar(UInt32(dimension))],
                outputArgumentIndex: 2,
                threadGrid: MetalKernelThreadGrid(width: 3),
                expected: expected
            )
        }
        if functionName.contains("mode_proxy") || functionName.contains("window_score") {
            return try executeWindowProbe(bundle: bundle, functionName: functionName)
        }
        if functionName == "moving_average_cross_signal_grid" {
            return try executeMovingAverageProbe(bundle: bundle, functionName: functionName)
        }
        if functionName == "rule_m1sync_chain_scan" {
            return try executeM1SyncProbe(bundle: bundle, functionName: functionName)
        }
        if functionName == "tree_rf_margin" {
            return try executeTreeRFProbe(bundle: bundle, functionName: functionName)
        }
        return nil
    }

    private static func executeWindowProbe(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String
    ) throws -> FXAIPluginMetalRuntimeProbeResult {
        let featureCount = max(FXDataEngineConstants.aiWeights, 48)
        let windowLength = 24
        var features = Array(repeating: Float(0), count: featureCount)
        features[1] = 0.35
        features[2] = 0.12
        features[6] = 0.50
        features[40] = 0.70
        var window = Array(repeating: Float(0), count: windowLength * featureCount)
        for row in 0..<windowLength {
            let value = 0.42 - Float(row) * 0.012
            window[row * featureCount + 1] = value
            window[row * featureCount + 2] = value * 0.5
            window[row * featureCount + 6] = 0.45
            window[row * featureCount + 40] = 0.65
        }
        let expected: [Float]
        if functionName.contains("trend_tsmom_vol") {
            expected = trendTSMOMExpected(features: features, window: window, featureCount: featureCount, windowLength: windowLength)
        } else if functionName.contains("trend_vol_breakout") {
            expected = trendVolBreakoutExpected(features: features, window: window, featureCount: featureCount, windowLength: windowLength)
        } else if functionName.contains("stat_vmd") {
            expected = statVMDExpected(features: features, window: window, featureCount: featureCount, windowLength: windowLength)
        } else {
            expected = statEMDHHTExpected(features: features, window: window, featureCount: featureCount, windowLength: windowLength)
        }
        return try executeAndValidate(
            bundle: bundle,
            functionName: functionName,
            arguments: [
                .floatArray(features),
                .floatArray(window),
                .outputFloat(count: 4),
                .uintScalar(1),
                .uintScalar(UInt32(featureCount)),
                .uintScalar(UInt32(windowLength)),
                .boolScalar(true)
            ],
            outputArgumentIndex: 2,
            threadGrid: MetalKernelThreadGrid(width: 1),
            expected: expected,
            tolerance: 0.0005
        )
    }

    private static func executeMovingAverageProbe(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String
    ) throws -> FXAIPluginMetalRuntimeProbeResult {
        let close: [Float] = [1.0, 1.01, 1.02, 1.04, 1.07, 1.09]
        let volumeSignal = Array(repeating: Float(0.4), count: close.count)
        let fastPeriods: [UInt32] = [2]
        let slowPeriods: [UInt32] = [4]
        let expected = movingAverageScores(close: close, volumeSignal: volumeSignal, fast: 2, slow: 4, hasVolume: true)
        return try executeAndValidate(
            bundle: bundle,
            functionName: functionName,
            arguments: [
                .floatArray(close),
                .floatArray(volumeSignal),
                .uintArray(fastPeriods),
                .uintArray(slowPeriods),
                .outputFloat(count: expected.count),
                .outputFloat(count: expected.count),
                .uintScalar(UInt32(close.count)),
                .uintScalar(1),
                .boolScalar(true)
            ],
            outputArgumentIndex: 4,
            threadGrid: MetalKernelThreadGrid(width: close.count, height: 1),
            expected: expected
        )
    }

    private static func executeM1SyncProbe(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String
    ) throws -> FXAIPluginMetalRuntimeProbeResult {
        let utc: [Int64] = [100, 160, 220, 280, 340]
        let open: [Float] = [1.00, 1.01, 1.02, 1.03, 1.05]
        let close: [Float] = [1.00, 1.01, 1.02, 1.04, 1.06]
        let volume: [Float] = [100, 110, 120, 140, 150]
        return try executeAndValidate(
            bundle: bundle,
            functionName: functionName,
            arguments: [
                .int64Array(utc),
                .floatArray(open),
                .floatArray(close),
                .floatArray(volume),
                .int64Array([280]),
                .outputFloat(count: 1),
                .outputFloat(count: 1),
                .outputFloat(count: 1),
                .outputFloat(count: 1),
                .uintScalar(UInt32(utc.count)),
                .uintScalar(1),
                .uintScalar(3),
                .floatScalar(0.0),
                .floatScalar(0.1),
                .floatScalar(0.0001),
                .boolScalar(true)
            ],
            outputArgumentIndex: 5,
            threadGrid: MetalKernelThreadGrid(width: 1),
            expected: [1.0]
        )
    }

    private static func executeTreeRFProbe(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String
    ) throws -> FXAIPluginMetalRuntimeProbeResult {
        let state = Array(repeating: Float(1), count: 16)
        let splitFeatures = Array(repeating: Int32(0), count: 13 * 3)
        let splitThresholds = Array(repeating: Float(0), count: 13 * 3)
        var leafMass: [Float] = []
        for _ in 0..<(13 * 8) {
            leafMass.append(contentsOf: [1.0, 3.0, 1.0])
        }
        let vote0 = Float(0.001) + 13.0 * (1.0 / 5.0)
        let vote1 = Float(0.001) + 13.0 * (3.0 / 5.0)
        let vote2 = Float(0.001) + 13.0 * (1.0 / 5.0)
        let denominator = max(vote0 + vote1 + vote2, 1.0)
        let expected = [
            clamp((vote1 - vote0) / denominator * 5.0, -8, 8),
            clamp(vote1 / denominator, 0, 1)
        ]
        return try executeAndValidate(
            bundle: bundle,
            functionName: functionName,
            arguments: [
                .floatArray(state),
                .intArray(splitFeatures),
                .floatArray(splitThresholds),
                .floatArray(leafMass),
                .outputFloat(count: 2)
            ],
            outputArgumentIndex: 4,
            threadGrid: MetalKernelThreadGrid(width: 1),
            expected: expected
        )
    }

    private static func executeAndValidate(
        bundle: FXAIPluginMetalKernelBundle,
        functionName: String,
        arguments: [MetalKernelArgument],
        outputArgumentIndex: Int,
        threadGrid: MetalKernelThreadGrid,
        expected: [Float],
        tolerance: Float = 0.0001
    ) throws -> FXAIPluginMetalRuntimeProbeResult {
        let execution = try MetalKernelCompiler.executeFloatKernel(
            source: bundle.kernelSource,
            functionName: functionName,
            arguments: arguments,
            outputArgumentIndex: outputArgumentIndex,
            threadGrid: threadGrid,
            sourceLabel: "\(bundle.pluginName).\(functionName)"
        )
        try validateMetalOutput(execution.output, expected: expected, tolerance: tolerance, label: "\(bundle.pluginName).\(functionName)")
        return FXAIPluginMetalRuntimeProbeResult(
            pluginName: bundle.pluginName,
            sourcePath: bundle.sourcePath,
            functionName: functionName,
            executionResult: execution,
            expectedOutput: expected
        )
    }

    private static func validateMetalOutput(
        _ actual: [Float],
        expected: [Float],
        tolerance: Float,
        label: String
    ) throws {
        guard actual.count == expected.count else {
            throw FXDataEngineError.validation("metal.\(label).outputCount")
        }
        for (index, pair) in zip(actual, expected).enumerated() {
            guard pair.0.isFinite, pair.1.isFinite else {
                throw FXDataEngineError.validation("metal.\(label).finite.\(index)")
            }
            guard abs(pair.0 - pair.1) <= tolerance else {
                throw FXDataEngineError.validation("metal.\(label).parity.\(index)")
            }
        }
    }

    private static func softmax3(_ logits: [Float]) -> [Float] {
        let m = max(logits[0], max(logits[1], logits[2]))
        let e0 = exp(clamp(logits[0] - m, -30, 30))
        let e1 = exp(clamp(logits[1] - m, -30, 30))
        let e2 = exp(clamp(logits[2] - m, -30, 30))
        let s = max(e0 + e1 + e2, 0.000001)
        return [e0 / s, e1 / s, e2 / s]
    }

    private static func clamp(_ value: Float, _ lower: Float, _ upper: Float) -> Float {
        min(max(value, lower), upper)
    }

    private static func read(_ values: [Float], _ count: Int, _ index: Int) -> Float {
        guard index >= 0, index < count else { return 0 }
        return clamp(values[index], -50, 50)
    }

    private static func windowFeature(_ window: [Float], _ featureCount: Int, _ row: Int, _ feature: Int) -> Float {
        read(Array(window[(row * featureCount)..<((row + 1) * featureCount)]), featureCount, feature)
    }

    private static func volumeSignal(_ features: [Float], featureCount: Int) -> Float {
        clamp(0.65 * read(features, featureCount, 40) + 0.35 * read(features, featureCount, 6), -8, 8)
    }

    private static func statEMDHHTExpected(
        features: [Float],
        window: [Float],
        featureCount: Int,
        windowLength: Int
    ) -> [Float] {
        let recentCount = min(windowLength, 8)
        var modeMean: Float = 0
        var recentMean: Float = 0
        var variance: Float = 0
        for i in 0..<windowLength { modeMean += windowFeature(window, featureCount, i, 1) }
        modeMean = windowLength > 0 ? modeMean / Float(windowLength) : read(features, featureCount, 1)
        for i in 0..<recentCount { recentMean += windowFeature(window, featureCount, i, 1) }
        recentMean = recentCount > 0 ? recentMean / Float(recentCount) : modeMean
        for i in 0..<windowLength {
            let d = windowFeature(window, featureCount, i, 1) - modeMean
            variance += d * d
        }
        let volatility = max(sqrt(variance / Float(max(windowLength, 1))), 0.01)
        let recent = recentCount > 0 ? windowFeature(window, featureCount, 0, 1) : modeMean
        let older = recentCount > 1 ? windowFeature(window, featureCount, recentCount - 1, 1) : recent
        let recentDelta = recent - older
        let meanShift = recentMean - modeMean
        let v = volumeSignal(features, featureCount: featureCount)
        let margin = clamp(0.35 * recentDelta + 0.20 * meanShift + 0.45 * read(features, featureCount, 1), -8, 8)
        let strength = min((abs(recentDelta) + abs(meanShift)) / volatility, 2) * 0.50
        let confidence = clamp(0.36 + 0.36 * strength + 0.16 / (1 + volatility) + 0.06 * abs(v), 0, 1)
        return [margin, confidence, volatility, v]
    }

    private static func statVMDExpected(
        features: [Float],
        window: [Float],
        featureCount: Int,
        windowLength: Int
    ) -> [Float] {
        var fastMode = windowLength > 0 ? windowFeature(window, featureCount, windowLength - 1, 1) : read(features, featureCount, 1)
        var slowMode = windowLength > 0 ? windowFeature(window, featureCount, windowLength - 1, 2) : read(features, featureCount, 2)
        var mean: Float = 0
        var variance: Float = 0
        if windowLength > 0 {
            for i in stride(from: windowLength - 1, through: 0, by: -1) {
                fastMode = 0.55 * windowFeature(window, featureCount, i, 1) + 0.45 * fastMode
                slowMode = 0.85 * windowFeature(window, featureCount, i, 2) + 0.15 * slowMode
                mean += windowFeature(window, featureCount, i, 1)
            }
        }
        mean = windowLength > 0 ? mean / Float(windowLength) : read(features, featureCount, 1)
        for i in 0..<windowLength {
            let d = windowFeature(window, featureCount, i, 1) - mean
            variance += d * d
        }
        let volatility = max(sqrt(variance / Float(max(windowLength, 1))), 0.01)
        let slope = windowLength > 1
            ? windowFeature(window, featureCount, 0, 1) - windowFeature(window, featureCount, min(windowLength - 1, 15), 1)
            : read(features, featureCount, 1)
        let v = volumeSignal(features, featureCount: featureCount)
        let margin = clamp(0.30 * fastMode + 0.30 * slowMode + 0.40 * slope, -8, 8)
        let decompositionStrength = min((abs(fastMode) + abs(slowMode) + 0.50 * abs(slope)) / volatility, 2) * 0.50
        let confidence = clamp(0.36 + 0.36 * decompositionStrength + 0.16 / (1 + volatility) + 0.06 * abs(v), 0, 1)
        return [margin, confidence, volatility, v]
    }

    private static func trendTSMOMExpected(
        features: [Float],
        window: [Float],
        featureCount: Int,
        windowLength: Int
    ) -> [Float] {
        let count16 = min(windowLength, 16)
        let count8 = min(windowLength, 8)
        var mean: Float = 0
        var variance: Float = 0
        for i in 0..<windowLength { mean += windowFeature(window, featureCount, i, 1) }
        mean = windowLength > 0 ? mean / Float(windowLength) : read(features, featureCount, 1)
        for i in 0..<windowLength {
            let d = windowFeature(window, featureCount, i, 1) - mean
            variance += d * d
        }
        let volatility = max(sqrt(variance / Float(max(windowLength, 1))), 0.01)
        let recent = windowLength > 0 ? windowFeature(window, featureCount, 0, 1) : read(features, featureCount, 1)
        let older = count16 > 1 ? windowFeature(window, featureCount, count16 - 1, 1) : recent
        let delta16 = recent - older
        let older8 = count8 > 1 ? windowFeature(window, featureCount, count8 - 1, 1) : recent
        let delta8 = recent - older8
        var slope: Float = 0
        var weightSum: Float = 0
        for i in 0..<windowLength {
            let weight = Float(windowLength - i)
            slope += weight * windowFeature(window, featureCount, i, 1)
            weightSum += weight
        }
        slope = weightSum > 0 ? slope / weightSum - mean : read(features, featureCount, 1)
        let v = volumeSignal(features, featureCount: featureCount)
        let margin = clamp((0.45 * delta16 + 0.35 * slope + 0.20 * delta8) / volatility, -8, 8)
        let confidence = clamp(0.38 + 0.17 * min(abs(delta16) / volatility, 2) + 0.17 * min(abs(slope) / volatility, 2) + 0.16 / (1 + volatility) + 0.06 * abs(v), 0, 1)
        return [margin, confidence, volatility, v]
    }

    private static func trendVolBreakoutExpected(
        features: [Float],
        window: [Float],
        featureCount: Int,
        windowLength: Int
    ) -> [Float] {
        func range(start: Int, count: Int) -> Float {
            guard start < windowLength, count > 0 else { return 0 }
            let end = min(windowLength, start + count)
            var low = Float.greatestFiniteMagnitude
            var high = -Float.greatestFiniteMagnitude
            for i in start..<end {
                let value = windowFeature(window, featureCount, i, 1)
                low = min(low, value)
                high = max(high, value)
            }
            return max(high - low, 0)
        }
        let recentRange = range(start: 0, count: 8)
        let baselineRange = range(start: 8, count: 24)
        let expansion = recentRange - baselineRange
        var mean: Float = 0
        var variance: Float = 0
        for i in 0..<windowLength { mean += windowFeature(window, featureCount, i, 1) }
        mean = windowLength > 0 ? mean / Float(windowLength) : read(features, featureCount, 1)
        for i in 0..<windowLength {
            let d = windowFeature(window, featureCount, i, 1) - mean
            variance += d * d
        }
        let volatility = max(sqrt(variance / Float(max(windowLength, 1))), 0.01)
        let recent = windowLength > 0 ? windowFeature(window, featureCount, 0, 1) : read(features, featureCount, 1)
        let older = windowLength > 1 ? windowFeature(window, featureCount, min(windowLength - 1, 15), 1) : recent
        let slope = recent - older
        let breakoutDirection: Float = expansion >= 0 ? 1 : -1
        let v = volumeSignal(features, featureCount: featureCount)
        let margin = clamp(breakoutDirection * slope / volatility, -8, 8)
        let confidence = clamp(0.35 + 0.38 * min(abs(expansion) / volatility, 1) + 0.15 * abs(slope) + 0.12 / (1 + volatility) + 0.06 * abs(v), 0, 1)
        return [margin, confidence, volatility, v]
    }

    private static func movingAverageScores(
        close: [Float],
        volumeSignal: [Float],
        fast: Int,
        slow: Int,
        hasVolume: Bool
    ) -> [Float] {
        close.indices.map { bar in
            if bar + 1 < slow { return 0 }
            let fastMA = close[(bar - fast + 1)...bar].reduce(0, +) / Float(fast)
            let slowMA = close[(bar - slow + 1)...bar].reduce(0, +) / Float(slow)
            let edge = fastMA - slowMA
            let priceScale = max(abs(slowMA), 0.000001)
            let normalizedEdge = edge / priceScale
            let volumeBoost = hasVolume ? 0.10 * clamp(abs(volumeSignal[bar]), 0, 1) : 0
            return clamp(abs(normalizedEdge) * 32 + volumeBoost, 0, 1)
        }
    }

    private static let runtimeProbeKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fxai_plugin_runtime_probe(
        device const float *input [[buffer(0)]],
        device float *output [[buffer(1)]],
        uint index [[thread_position_in_grid]]
    ) {
        output[index] = input[index] * 2.0f + 1.0f;
    }
    """
}
