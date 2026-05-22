import Foundation

public struct RuntimeModelCandidate: Codable, Hashable, Sendable {
    public var aiID: Int
    public var pluginName: String
    public var family: AIFamily
    public var available: Bool
    public var pruned: Bool
    public var metaScore: Double
    public var voteWeight: Double
    public var regimeObservations: Int

    public init(
        aiID: Int,
        pluginName: String,
        family: AIFamily = .other,
        available: Bool = true,
        pruned: Bool = false,
        metaScore: Double = 0.0,
        voteWeight: Double = 0.0,
        regimeObservations: Int = 0
    ) {
        self.aiID = aiID
        self.pluginName = pluginName
        self.family = family
        self.available = available
        self.pruned = pruned
        self.metaScore = max(0.0, fxSafeFinite(metaScore))
        self.voteWeight = fxSafeFinite(voteWeight)
        self.regimeObservations = max(0, regimeObservations)
    }
}

public struct RuntimeModelSelectionResult: Codable, Hashable, Sendable {
    public var activeAIIDs: [Int]
    public var preCapAIIDs: [Int]
    public var fallbackUsed: Bool
    public var explorationAdded: Bool
    public var runtimeModelCap: Int
    public var budgetPressure: Double

    public init(
        activeAIIDs: [Int],
        preCapAIIDs: [Int],
        fallbackUsed: Bool,
        explorationAdded: Bool,
        runtimeModelCap: Int,
        budgetPressure: Double
    ) {
        self.activeAIIDs = RuntimeModelStageTools.uniqueValidIDs(activeAIIDs)
        self.preCapAIIDs = RuntimeModelStageTools.uniqueValidIDs(preCapAIIDs)
        self.fallbackUsed = fallbackUsed
        self.explorationAdded = explorationAdded
        self.runtimeModelCap = Int(fxClamp(Double(runtimeModelCap), 1.0, Double(FXDataEngineConstants.aiCount)))
        self.budgetPressure = fxClamp(budgetPressure, 0.0, 1.0)
    }
}

public struct RuntimeModelRouteRecord: Codable, Hashable, Sendable {
    public var aiID: Int
    public var pluginName: String
    public var family: AIFamily
    public var selected: Bool
    public var routedMetaWeight: Double
    public var adaptiveSuitability: Double
    public var adaptiveStatus: AdaptiveRouterRuntimeStatus
    public var studentAllowed: Bool
    public var reason: String

    public init(
        aiID: Int,
        pluginName: String,
        family: AIFamily,
        selected: Bool = true,
        routedMetaWeight: Double = -1.0,
        adaptiveSuitability: Double = 1.0,
        adaptiveStatus: AdaptiveRouterRuntimeStatus = .active,
        studentAllowed: Bool = true,
        reason: String = "selected"
    ) {
        self.aiID = Int(fxClamp(Double(aiID), 0.0, Double(FXDataEngineConstants.aiCount - 1)))
        self.pluginName = pluginName
        self.family = family
        self.selected = selected
        self.routedMetaWeight = fxSafeFinite(routedMetaWeight)
        self.adaptiveSuitability = fxClamp(adaptiveSuitability, 0.0, 3.0)
        self.adaptiveStatus = adaptiveStatus
        self.studentAllowed = studentAllowed
        self.reason = reason
    }
}

public struct RuntimeModelRouteResult: Codable, Hashable, Sendable {
    public var selectedAIIDs: [Int]
    public var routeCutoff: Double
    public var adaptiveRouterActive: Bool
    public var records: [RuntimeModelRouteRecord]

    public init(
        selectedAIIDs: [Int],
        routeCutoff: Double = -1.0,
        adaptiveRouterActive: Bool = false,
        records: [RuntimeModelRouteRecord] = []
    ) {
        self.selectedAIIDs = RuntimeModelStageTools.uniqueValidIDs(selectedAIIDs)
        self.routeCutoff = fxSafeFinite(routeCutoff)
        self.adaptiveRouterActive = adaptiveRouterActive
        self.records = records
    }
}

public enum RuntimeModelStageTools {
    public static func selectActiveModels(
        candidates: [RuntimeModelCandidate],
        ensembleMode: Bool,
        aiType: Int,
        signalBarUTC: Int64,
        regimeID: Int,
        explorePercent: Double,
        deployProfile: LiveDeploymentProfile,
        performance: RuntimePerformanceState = RuntimePerformanceState()
    ) -> RuntimeModelSelectionResult {
        let validCandidates = uniqueValidCandidates(candidates)
        let byID = candidateMap(validCandidates)
        var active: [Int] = []
        var fallbackUsed = false
        var explorationAdded = false

        if !ensembleMode {
            if let candidate = byID[aiType], candidate.available {
                active = [candidate.aiID]
            }
        } else {
            let eligible = validCandidates.filter { $0.available && !$0.pruned && $0.metaScore > 0.0 }
            if !eligible.isEmpty {
                var topK = eligible.count
                if topK > 10 {
                    topK = 10
                }
                if topK < 2, eligible.count >= 2 {
                    topK = 2
                }

                var used = Array(repeating: false, count: eligible.count)
                for _ in 0..<topK {
                    var bestIndex = -1
                    var bestScore = -Double.greatestFiniteMagnitude
                    for index in eligible.indices where !used[index] {
                        if eligible[index].metaScore > bestScore {
                            bestScore = eligible[index].metaScore
                            bestIndex = index
                        }
                    }
                    guard bestIndex >= 0 else { break }
                    used[bestIndex] = true
                    active.append(eligible[bestIndex].aiID)
                }

                let shouldExplore = eligible.count > active.count &&
                    NormalizationMetaSupportTools.shouldSampleByPercent(
                        barTimeUTC: signalBarUTC,
                        salt: regimeID + 17,
                        percent: explorePercent
                    )
                if shouldExplore {
                    var exploreIndex = -1
                    var exploreScore = -Double.greatestFiniteMagnitude
                    for index in eligible.indices where !used[index] {
                        let coldBonus = 1.0 / sqrt(1.0 + Double(eligible[index].regimeObservations))
                        let score = eligible[index].metaScore * (1.0 + 0.35 * coldBonus)
                        if score > exploreScore {
                            exploreScore = score
                            exploreIndex = index
                        }
                    }
                    if exploreIndex >= 0, !active.contains(eligible[exploreIndex].aiID) {
                        active.append(eligible[exploreIndex].aiID)
                        explorationAdded = true
                    }
                }
            } else {
                fallbackUsed = true
                var fallbackID = -1
                var fallbackWeight = -Double.greatestFiniteMagnitude
                for candidate in validCandidates where candidate.available {
                    if candidate.voteWeight > fallbackWeight {
                        fallbackWeight = candidate.voteWeight
                        fallbackID = candidate.aiID
                    }
                }
                if fallbackID >= 0 {
                    active = [fallbackID]
                }
            }
        }

        let preCap = uniqueValidIDs(active)
        let capped = RuntimeStageTools.applyPerformanceModelCap(
            activeAIIDs: preCap,
            deployProfile: deployProfile,
            performance: performance
        )
        return RuntimeModelSelectionResult(
            activeAIIDs: capped.activeAIIDs,
            preCapAIIDs: preCap,
            fallbackUsed: fallbackUsed,
            explorationAdded: explorationAdded,
            runtimeModelCap: capped.runtimeModelCap,
            budgetPressure: capped.budgetPressure
        )
    }

    public static func routeActiveModels(
        activeAIIDs: [Int],
        candidates: [RuntimeModelCandidate],
        ensembleMode: Bool,
        studentRouter: StudentRouterProfile = StudentRouterProfile(),
        adaptiveProfile: AdaptiveRouterProfile = AdaptiveRouterProfile(),
        adaptiveRegimeState: AdaptiveRegimeState = .reset,
        adaptiveRouterEnabled: Bool = true
    ) -> RuntimeModelRouteResult {
        let byID = candidateMap(uniqueValidCandidates(candidates))
        let active = uniqueValidIDs(activeAIIDs)
        let adaptiveActive = adaptiveRouterEnabled &&
            adaptiveProfile.ready &&
            adaptiveProfile.enabled &&
            (!adaptiveProfile.fallbackToStudentRouterOnly || studentRouter.ready)

        guard ensembleMode, (studentRouter.ready || adaptiveActive) else {
            let records = active.compactMap { id -> RuntimeModelRouteRecord? in
                guard let candidate = byID[id], candidate.available else { return nil }
                return RuntimeModelRouteRecord(
                    aiID: candidate.aiID,
                    pluginName: candidate.pluginName,
                    family: candidate.family,
                    selected: true,
                    routedMetaWeight: candidate.metaScore
                )
            }
            return RuntimeModelRouteResult(
                selectedAIIDs: records.map(\.aiID),
                adaptiveRouterActive: adaptiveActive,
                records: records
            )
        }

        var recordsByID: [Int: RuntimeModelRouteRecord] = [:]
        var topScores = Array(repeating: -1.0, count: FXDataEngineConstants.aiCount)
        var candidateCount = 0

        for id in active {
            guard let candidate = byID[id], candidate.available else { continue }
            var record = RuntimeModelRouteRecord(
                aiID: candidate.aiID,
                pluginName: candidate.pluginName,
                family: candidate.family,
                selected: true,
                routedMetaWeight: -1.0,
                adaptiveSuitability: 1.0,
                adaptiveStatus: .active,
                studentAllowed: true
            )

            if studentRouter.ready, !studentRouter.allowsPlugin(pluginName: candidate.pluginName, family: candidate.family) {
                record.selected = false
                record.studentAllowed = false
                record.reason = "student_router_blocked"
                recordsByID[id] = record
                continue
            }

            let familyWeight = studentRouter.ready ? studentRouter.familyWeight(candidate.family) : 1.0
            let pluginWeight = studentRouter.ready
                ? studentRouter.pluginWeight(pluginName: candidate.pluginName, family: candidate.family)
                : 1.0
            let adaptiveFactor = adaptiveActive
                ? AdaptiveRouterRuntimeTools.pluginSuitability(
                    profile: adaptiveProfile,
                    state: adaptiveRegimeState,
                    pluginName: candidate.pluginName
                )
                : 1.0
            let adaptiveStatus = adaptiveActive
                ? AdaptiveRouterRuntimeTools.suitabilityStatus(profile: adaptiveProfile, suitability: adaptiveFactor)
                : AdaptiveRouterRuntimeStatus.active

            record.adaptiveSuitability = adaptiveFactor
            record.adaptiveStatus = adaptiveStatus
            if adaptiveActive, adaptiveStatus == .suppressed {
                record.selected = false
                record.routedMetaWeight = 0.0
                record.reason = "adaptive_router_suppressed"
                recordsByID[id] = record
                continue
            }

            let routedWeight = candidate.metaScore * familyWeight * pluginWeight * adaptiveFactor
            record.routedMetaWeight = routedWeight
            let minMetaWeight = studentRouter.ready ? studentRouter.minMetaWeight : 0.0
            if routedWeight + 1e-12 < minMetaWeight {
                record.selected = false
                record.reason = "below_min_meta_weight"
                recordsByID[id] = record
                continue
            }

            candidateCount += 1
            insertDescending(routedWeight, into: &topScores)
            recordsByID[id] = record
        }

        var routeCutoff = -1.0
        var maxActiveModels = studentRouter.ready ? studentRouter.maxActiveModels : active.count
        if maxActiveModels < 1 {
            maxActiveModels = 1
        }
        if candidateCount > maxActiveModels {
            routeCutoff = topScores[maxActiveModels - 1]
        }

        let minMetaWeight = studentRouter.ready ? studentRouter.minMetaWeight : 0.0
        var records: [RuntimeModelRouteRecord] = []
        var selected: [Int] = []
        for id in active {
            guard var record = recordsByID[id] else { continue }
            if record.routedMetaWeight < minMetaWeight - 1e-12 {
                let wasSelected = record.selected
                record.selected = false
                if wasSelected || record.reason == "selected" {
                    record.reason = "below_min_meta_weight"
                }
            }
            if routeCutoff >= 0.0,
               record.routedMetaWeight >= 0.0,
               record.routedMetaWeight + 1e-12 < routeCutoff {
                let wasSelected = record.selected
                record.selected = false
                if wasSelected || record.reason == "selected" {
                    record.reason = "below_route_cutoff"
                }
            }
            if record.selected {
                selected.append(id)
            }
            records.append(record)
        }

        return RuntimeModelRouteResult(
            selectedAIIDs: selected,
            routeCutoff: routeCutoff,
            adaptiveRouterActive: adaptiveActive,
            records: records
        )
    }

    public static func uniqueValidIDs(_ aiIDs: [Int]) -> [Int] {
        var seen = Set<Int>()
        var output: [Int] = []
        for aiID in aiIDs where aiID >= 0 && aiID < FXDataEngineConstants.aiCount {
            if seen.insert(aiID).inserted {
                output.append(aiID)
            }
        }
        return output
    }

    private static func candidateMap(_ candidates: [RuntimeModelCandidate]) -> [Int: RuntimeModelCandidate] {
        var output: [Int: RuntimeModelCandidate] = [:]
        for candidate in candidates where candidate.aiID >= 0 &&
            candidate.aiID < FXDataEngineConstants.aiCount &&
            output[candidate.aiID] == nil {
            output[candidate.aiID] = candidate
        }
        return output
    }

    private static func uniqueValidCandidates(_ candidates: [RuntimeModelCandidate]) -> [RuntimeModelCandidate] {
        var seen = Set<Int>()
        var output: [RuntimeModelCandidate] = []
        for candidate in candidates where candidate.aiID >= 0 && candidate.aiID < FXDataEngineConstants.aiCount {
            if seen.insert(candidate.aiID).inserted {
                output.append(candidate)
            }
        }
        return output
    }

    private static func insertDescending(_ value: Double, into scores: inout [Double]) {
        for index in scores.indices where value > scores[index] {
            if scores.count > 1, index < scores.count - 1 {
                for shifted in stride(from: scores.count - 1, through: index + 1, by: -1) {
                    scores[shifted] = scores[shifted - 1]
                }
            }
            scores[index] = value
            return
        }
    }
}
