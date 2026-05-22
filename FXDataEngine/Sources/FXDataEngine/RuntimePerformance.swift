import Foundation

public struct RuntimePerformanceState: Codable, Sendable {
    public private(set) var ready: Bool
    public private(set) var lastActiveModels: Int
    public private(set) var lastTimeUTC: Int64
    public private(set) var stageMeanMS: [Double]
    public private(set) var stageMaxMS: [Double]
    public private(set) var stageObservations: [Int]
    public private(set) var pluginPredictMeanMS: [Double]
    public private(set) var pluginPredictMaxMS: [Double]
    public private(set) var pluginPredictObservations: [Int]
    public private(set) var pluginUpdateMeanMS: [Double]
    public private(set) var pluginUpdateMaxMS: [Double]
    public private(set) var pluginUpdateObservations: [Int]
    public private(set) var pluginWorkingSetKB: [Double]

    public init() {
        let stageCount = RuntimeStage.allCases.count
        let pluginCount = FXDataEngineConstants.aiCount
        self.ready = false
        self.lastActiveModels = 0
        self.lastTimeUTC = 0
        self.stageMeanMS = Array(repeating: 0.0, count: stageCount)
        self.stageMaxMS = Array(repeating: 0.0, count: stageCount)
        self.stageObservations = Array(repeating: 0, count: stageCount)
        self.pluginPredictMeanMS = Array(repeating: 0.0, count: pluginCount)
        self.pluginPredictMaxMS = Array(repeating: 0.0, count: pluginCount)
        self.pluginPredictObservations = Array(repeating: 0, count: pluginCount)
        self.pluginUpdateMeanMS = Array(repeating: 0.0, count: pluginCount)
        self.pluginUpdateMaxMS = Array(repeating: 0.0, count: pluginCount)
        self.pluginUpdateObservations = Array(repeating: 0, count: pluginCount)
        self.pluginWorkingSetKB = Array(repeating: 0.0, count: pluginCount)
    }

    public mutating func reset() {
        self = RuntimePerformanceState()
    }

    public mutating func recordStage(
        _ stage: RuntimeStage,
        elapsedMS: Double,
        sampleTimeUTC: Int64 = 0
    ) {
        ensureStorage()
        let index = stage.rawValue
        let milliseconds = max(elapsedMS, 0.0)
        let observations = stageObservations[index]
        stageMeanMS[index] = Self.blend(previous: stageMeanMS[index], value: milliseconds, observations: observations)
        stageMaxMS[index] = max(stageMaxMS[index], milliseconds)
        stageObservations[index] = observations + 1
        markReady(sampleTimeUTC: sampleTimeUTC)
    }

    public mutating func recordPluginPredict(
        aiID: Int,
        elapsedMS: Double,
        sampleTimeUTC: Int64 = 0
    ) {
        guard isValidPlugin(aiID) else { return }
        ensureStorage()
        let milliseconds = max(elapsedMS, 0.0)
        let observations = pluginPredictObservations[aiID]
        pluginPredictMeanMS[aiID] = Self.blend(previous: pluginPredictMeanMS[aiID], value: milliseconds, observations: observations)
        pluginPredictMaxMS[aiID] = max(pluginPredictMaxMS[aiID], milliseconds)
        pluginPredictObservations[aiID] = observations + 1
        markReady(sampleTimeUTC: sampleTimeUTC)
    }

    public mutating func recordPluginUpdate(
        aiID: Int,
        elapsedMS: Double,
        sampleTimeUTC: Int64 = 0
    ) {
        guard isValidPlugin(aiID) else { return }
        ensureStorage()
        let milliseconds = max(elapsedMS, 0.0)
        let observations = pluginUpdateObservations[aiID]
        pluginUpdateMeanMS[aiID] = Self.blend(previous: pluginUpdateMeanMS[aiID], value: milliseconds, observations: observations)
        pluginUpdateMaxMS[aiID] = max(pluginUpdateMaxMS[aiID], milliseconds)
        pluginUpdateObservations[aiID] = observations + 1
        markReady(sampleTimeUTC: sampleTimeUTC)
    }

    public mutating func setPluginWorkingSetKB(aiID: Int, workingSetKB: Double) {
        guard isValidPlugin(aiID) else { return }
        ensureStorage()
        pluginWorkingSetKB[aiID] = max(pluginWorkingSetKB[aiID], max(workingSetKB, 0.0))
    }

    public mutating func setActiveModels(_ activeModels: Int) {
        lastActiveModels = max(0, activeModels)
    }

    public func budgetPressure(budgetMS: Double) -> Double {
        let budget = max(budgetMS, 0.0)
        guard budget > 0.0,
              RuntimeStage.total.rawValue < stageObservations.count,
              stageObservations[RuntimeStage.total.rawValue] > 0 else {
            return 0.0
        }
        return fxClamp(
            (HorizonTools.value(in: stageMeanMS, index: RuntimeStage.total.rawValue, default: 0.0) - budget) / max(budget, 1e-6),
            0.0,
            2.0
        )
    }

    public func stageManifestRows(activeModels: Int? = nil) -> [RuntimePerformanceManifestRow] {
        RuntimeStage.allCases.compactMap { stage in
            let index = stage.rawValue
            guard index < stageObservations.count, stageObservations[index] > 0 else { return nil }
            return RuntimePerformanceManifestRow.stage(
                stage,
                meanMS: HorizonTools.value(in: stageMeanMS, index: index, default: 0.0),
                maxMS: HorizonTools.value(in: stageMaxMS, index: index, default: 0.0),
                observations: stageObservations[index],
                activeModels: activeModels ?? lastActiveModels
            )
        }
    }

    public func pluginManifestRows(
        manifests: [PluginManifestV4],
        activeModels: Int? = nil
    ) -> [RuntimePerformanceManifestRow] {
        manifests.compactMap { manifest in
            let aiID = manifest.aiID
            guard isValidPlugin(aiID) else { return nil }
            let predictObs = intValue(in: pluginPredictObservations, index: aiID)
            let updateObs = intValue(in: pluginUpdateObservations, index: aiID)
            let workingSet = HorizonTools.value(in: pluginWorkingSetKB, index: aiID, default: 0.0)
            guard predictObs > 0 || updateObs > 0 || workingSet > 0.0 else { return nil }
            return RuntimePerformanceManifestRow.plugin(
                aiID: aiID,
                aiName: manifest.aiName,
                predictMeanMS: HorizonTools.value(in: pluginPredictMeanMS, index: aiID, default: 0.0),
                predictMaxMS: HorizonTools.value(in: pluginPredictMaxMS, index: aiID, default: 0.0),
                predictObservations: predictObs,
                updateMeanMS: HorizonTools.value(in: pluginUpdateMeanMS, index: aiID, default: 0.0),
                updateMaxMS: HorizonTools.value(in: pluginUpdateMaxMS, index: aiID, default: 0.0),
                updateObservations: updateObs,
                workingSetKB: workingSet,
                activeModels: activeModels ?? lastActiveModels
            )
        }
    }

    public static func blend(previous: Double, value: Double, observations: Int) -> Double {
        observations <= 0 ? value : ((1.0 - 0.12) * previous + 0.12 * value)
    }

    public static func estimatePluginWorkingSetKB(
        manifest: PluginManifestV4,
        sequenceBars: Int
    ) -> Double {
        let sequence = min(max(sequenceBars, 1), FXDataEngineConstants.maxSequenceBars)
        let payloadBytes = Double(FXDataEngineConstants.aiWeights) * 8.0
        let contextBytes = payloadBytes * Double(sequence)
        var stateMultiplier = 1.0
        if manifest.capabilityMask.contains(.stateful) {
            stateMultiplier += 0.75
        }
        if manifest.capabilityMask.contains(.windowContext) {
            stateMultiplier += 0.35
        }
        if manifest.capabilityMask.contains(.multiHorizon) {
            stateMultiplier += 0.20
        }
        return max((payloadBytes + contextBytes) * stateMultiplier / 1024.0, 1.0)
    }

    public static func stageName(stageID: Int) -> String {
        RuntimeStage(rawValue: stageID)?.name ?? "unknown"
    }

    private mutating func markReady(sampleTimeUTC: Int64) {
        ready = true
        if sampleTimeUTC > 0 {
            lastTimeUTC = sampleTimeUTC
        }
    }

    private mutating func ensureStorage() {
        ensureVector(&stageMeanMS, count: RuntimeStage.allCases.count)
        ensureVector(&stageMaxMS, count: RuntimeStage.allCases.count)
        ensureVector(&stageObservations, count: RuntimeStage.allCases.count)
        ensureVector(&pluginPredictMeanMS, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginPredictMaxMS, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginPredictObservations, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginUpdateMeanMS, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginUpdateMaxMS, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginUpdateObservations, count: FXDataEngineConstants.aiCount)
        ensureVector(&pluginWorkingSetKB, count: FXDataEngineConstants.aiCount)
        lastActiveModels = max(0, lastActiveModels)
        if lastTimeUTC < 0 {
            lastTimeUTC = 0
        }
    }

    private func isValidPlugin(_ aiID: Int) -> Bool {
        aiID >= 0 && aiID < FXDataEngineConstants.aiCount
    }

    private func intValue(in vector: [Int], index: Int, default defaultValue: Int = 0) -> Int {
        guard index >= 0, index < vector.count else { return defaultValue }
        return vector[index]
    }

    private func ensureVector(_ vector: inout [Double], count: Int) {
        if vector.count < count {
            vector.append(contentsOf: Array(repeating: 0.0, count: count - vector.count))
        } else if vector.count > count {
            vector.removeLast(vector.count - count)
        }
    }

    private func ensureVector(_ vector: inout [Int], count: Int) {
        if vector.count < count {
            vector.append(contentsOf: Array(repeating: 0, count: count - vector.count))
        } else if vector.count > count {
            vector.removeLast(vector.count - count)
        }
    }
}
