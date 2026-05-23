import FXDataEngine
import Foundation

public struct MixLoffmCPUModel: Sendable {
    private static let expertCount = 4
    private static let derivedCount = 10
    private static let stateCount = 6
    private static let latentCount = 12
    private static let moveFeatureCount = 8
    private static let replayCount = 12
    private static let classCount = 3

    private var steps: Int
    private var gateWeights: [[Double]]
    private var directionWeights: [[Double]]
    private var directionG2: [[Double]]
    private var moveHeadWeights: [[Double]]
    private var moveG2: [[Double]]
    private var skipWeights: [[Double]]
    private var skipG2: [[Double]]
    private var latent: [[Double]]
    private var confidenceEMA: [Double]
    private var edgeEMA: [Double]
    private var hitEMA: [Double]
    private var expertMass: [Double]
    private var usageEMA: [Double]
    private var globalState: [Double]
    private var globalEdgeEMA: Double
    private var globalHitEMA: Double
    private var replayDerived: [[[Double]]]
    private var replayMove: [[Double]]
    private var replayLabel: [[LabelClass]]
    private var replayHead: [Int]
    private var replayFilled: [Int]
    private var calibrator: PluginTernaryCalibrator
    private var moveReady: Bool
    private var moveEMAAbs: Double
    private var qualityBank: PluginQualityBank

    public init() {
        self.steps = 0
        self.gateWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.derivedCount),
            count: Self.expertCount
        )
        self.directionWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.latentCount),
            count: Self.expertCount
        )
        self.directionG2 = Array(
            repeating: Array(repeating: 0.0, count: Self.latentCount),
            count: Self.expertCount
        )
        self.moveHeadWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.moveFeatureCount),
            count: Self.expertCount
        )
        self.moveG2 = Array(
            repeating: Array(repeating: 0.0, count: Self.moveFeatureCount),
            count: Self.expertCount
        )
        self.skipWeights = Array(
            repeating: Array(repeating: 0.0, count: Self.derivedCount),
            count: Self.expertCount
        )
        self.skipG2 = Array(
            repeating: Array(repeating: 0.0, count: Self.derivedCount),
            count: Self.expertCount
        )
        self.latent = Array(
            repeating: Array(repeating: 0.0, count: Self.stateCount),
            count: Self.expertCount
        )
        self.confidenceEMA = Array(repeating: 0.50, count: Self.expertCount)
        self.edgeEMA = Array(repeating: 0.0, count: Self.expertCount)
        self.hitEMA = Array(repeating: 0.50, count: Self.expertCount)
        self.expertMass = Array(repeating: 0.0, count: Self.expertCount)
        self.usageEMA = Array(repeating: 1.0 / Double(Self.expertCount), count: Self.expertCount)
        self.globalState = Array(repeating: 0.0, count: Self.stateCount)
        self.globalEdgeEMA = 0.0
        self.globalHitEMA = 0.50
        self.replayDerived = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0, count: Self.derivedCount),
                count: Self.replayCount
            ),
            count: Self.expertCount
        )
        self.replayMove = Array(
            repeating: Array(repeating: 0.0, count: Self.replayCount),
            count: Self.expertCount
        )
        self.replayLabel = Array(
            repeating: Array(repeating: .skip, count: Self.replayCount),
            count: Self.expertCount
        )
        self.replayHead = Array(repeating: 0, count: Self.expertCount)
        self.replayFilled = Array(repeating: 0, count: Self.expertCount)
        self.calibrator = PluginTernaryCalibrator()
        self.moveReady = false
        self.moveEMAAbs = 0.0
        self.qualityBank = PluginQualityBank()
        seedWeights()
    }

    public mutating func reset() {
        self = MixLoffmCPUModel()
    }

    public mutating func train(_ request: TrainRequestV4, hyperParameters: HyperParameters) {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let label = PluginContextRuntimeTools.normalizeClassLabel(
            rawLabel: request.labelClass.rawValue,
            x: x,
            movePoints: request.movePoints,
            priceCostPoints: request.context.priceCostPoints
        )
        let derived = buildDerived(x: x, context: request.context)
        let gates = softmaxExperts(derived)
        let mixture = rawMixture(derived: derived, gates: gates, context: request.context)
        let cost = PluginContextRuntimeTools.inputPriceCostPoints(x, explicitCostPoints: request.context.priceCostPoints)
        let targetMove = max(0.0, abs(request.movePoints) - max(cost, 0.0))
        let sampleWeight = fxClamp(
            request.sampleWeight *
                PluginSupportTools.moveSampleWeight(
                    x: x,
                    movePoints: request.movePoints,
                    priceCostPoints: cost,
                    minMovePoints: request.context.minMovePoints,
                    qualityTargets: PluginQualityTargets(request: request)
                ),
            0.25,
            4.0
        )
        calibrator.update(
            rawProbabilities: mixture.rawProbabilities,
            labelClass: label,
            sampleWeight: sampleWeight,
            learningRate: hyperParameters.learningRate
        )
        updateLatentState(derived: derived, gates: gates, movePoints: request.movePoints)
        updateMoveEMA(targetMove)
        globalHitEMA = 0.98 * globalHitEMA + 0.02 * (label == .skip ? 0.50 : 1.0)

        var bestExpert = 0
        var bestGate = gates[0]
        for expert in 0..<Self.expertCount {
            usageEMA[expert] = 0.985 * usageEMA[expert] + 0.015 * gates[expert]
            if gates[expert] > bestGate {
                bestGate = gates[expert]
                bestExpert = expert
            }
        }

        for expert in 0..<Self.expertCount {
            trainExpertSample(
                expert: expert,
                derived: derived,
                label: label,
                movePoints: request.movePoints,
                targetMove: targetMove,
                sampleWeight: sampleWeight,
                hyperParameters: hyperParameters,
                gateWeight: gates[expert],
                adaptGate: true
            )
        }
        storeReplay(expert: bestExpert, derived: derived, label: label, movePoints: request.movePoints)
        replayExpert(bestExpert, hyperParameters: hyperParameters)
        qualityBank.update(request: request, sampleWeight: sampleWeight)
        steps += 1
    }

    public func predict(_ request: PredictRequestV4, hyperParameters _: HyperParameters) -> PredictionV4 {
        let x = Self.preparedFeatures(request.x, dataHasVolume: request.context.dataHasVolume)
        let window = Self.preparedWindow(request.xWindow, dataHasVolume: request.context.dataHasVolume)
        let derived = buildDerived(x: x, context: request.context)
        let gates = softmaxExperts(derived)
        let mixture = rawMixture(derived: derived, gates: gates, context: request.context)
        let probabilities = calibrator.calibrated(mixture.rawProbabilities)
        let expectedMove = fxClamp(max(mixture.edgeHat + request.context.priceCostPoints, moveReady ? moveEMAAbs : mixture.expectedMove), 0.0, 5000.0)
        let sigma = max(
            0.10,
            0.35 * abs(mixture.expectedMove) +
                0.60 * mixture.disagreement +
                0.20 * (1.0 - globalHitEMA)
        )
        let confidence = fxClamp(
            0.60 * max(probabilities[LabelClass.buy.rawValue], probabilities[LabelClass.sell.rawValue]) +
                0.25 * (1.0 - mixture.disagreement) +
                0.15 * (1.0 - probabilities[LabelClass.skip.rawValue]),
            0.0,
            1.0
        )
        let reliability = fxClamp(
            0.45 +
                0.20 * globalHitEMA +
                0.20 * fxClamp(globalEdgeEMA / max(expectedMove + 0.10, 0.10), 0.0, 1.0) +
                0.15 * (1.0 - mixture.disagreement),
            0.0,
            1.0
        )
        let q25 = max(0.0, expectedMove - 0.50 * sigma)
        let q50 = max(q25, expectedMove)
        let q75 = max(q50, expectedMove + 0.50 * sigma)
        let baseOutput = PluginModelOutputV4(
            classProbabilities: probabilities,
            moveMeanPoints: expectedMove,
            moveQ25Points: q25,
            moveQ50Points: q50,
            moveQ75Points: q75,
            mfeMeanPoints: max(q75, expectedMove),
            maeMeanPoints: max(0.0, 0.35 * expectedMove),
            hitTimeFraction: fxClamp(0.68 - 0.24 * confidence + 0.16 * probabilities[LabelClass.skip.rawValue], 0.0, 1.0),
            pathRisk: fxClamp(0.30 * probabilities[LabelClass.skip.rawValue] + 0.25 * mixture.disagreement + 0.20 * (1.0 - reliability), 0.0, 1.0),
            fillRisk: fxClamp(request.context.priceCostPoints / max(expectedMove + request.context.minMovePoints, 0.25), 0.0, 1.0),
            confidence: confidence,
            reliability: reliability,
            hasQuantiles: true,
            hasConfidence: true,
            hasPathQuality: true
        )
        let output = PluginPathQualityTools.populatedOutput(
            baseOutput,
            x: x,
            window: window,
            context: request.context,
            family: .mixture,
            activityGate: 1.0 - probabilities[LabelClass.skip.rawValue],
            structuralQuality: reliability,
            qualityPriors: qualityBank.priors(context: request.context),
            declaredWindowSize: request.windowSize
        )
        return PluginContextRuntimeTools.fillPrediction(
            modelOutput: output,
            calibratedMoveMeanPoints: expectedMove,
            context: request.context
        )
    }

    private mutating func seedWeights() {
        gateWeights[0][0] = 1.20; gateWeights[0][1] = 0.80; gateWeights[0][2] = -0.25
        gateWeights[0][3] = 0.40; gateWeights[0][4] = 0.15; gateWeights[0][5] = 0.30
        gateWeights[0][6] = -0.20; gateWeights[0][7] = 0.25; gateWeights[0][8] = 0.10; gateWeights[0][9] = 0.10

        gateWeights[1][0] = -0.85; gateWeights[1][1] = 0.25; gateWeights[1][2] = 0.70
        gateWeights[1][3] = 0.10; gateWeights[1][4] = -0.25; gateWeights[1][5] = 0.05
        gateWeights[1][6] = 0.25; gateWeights[1][7] = -0.10; gateWeights[1][8] = 0.15; gateWeights[1][9] = -0.10

        gateWeights[2][0] = 0.45; gateWeights[2][1] = 1.10; gateWeights[2][2] = -0.20
        gateWeights[2][3] = 0.90; gateWeights[2][4] = 0.50; gateWeights[2][5] = 0.10
        gateWeights[2][6] = -0.35; gateWeights[2][7] = 0.20; gateWeights[2][8] = 0.05; gateWeights[2][9] = 0.05

        gateWeights[3][0] = -0.20; gateWeights[3][1] = -0.35; gateWeights[3][2] = 0.25
        gateWeights[3][3] = 0.10; gateWeights[3][4] = 0.15; gateWeights[3][5] = 0.55
        gateWeights[3][6] = 0.80; gateWeights[3][7] = -0.10; gateWeights[3][8] = 0.20; gateWeights[3][9] = 0.0

        for expert in 0..<Self.expertCount {
            directionWeights[expert][0] = 0.05
            directionWeights[expert][1] = expert == 0 ? 0.35 : (expert == 1 ? -0.30 : (expert == 2 ? 0.22 : 0.05))
            directionWeights[expert][2] = expert == 2 ? 0.22 : 0.10
            directionWeights[expert][3] = expert == 1 ? 0.18 : 0.08
            directionWeights[expert][4] = expert == 3 ? -0.22 : -0.06
            directionWeights[expert][5] = expert == 0 ? 0.16 : (expert == 1 ? -0.16 : 0.08)
            directionWeights[expert][6] = expert == 0 ? 0.14 : (expert == 2 ? 0.10 : -0.08)
            directionWeights[expert][7] = expert == 1 ? 0.12 : 0.04
            directionWeights[expert][8] = expert == 3 ? -0.18 : -0.04
            directionWeights[expert][9] = 0.06
            directionWeights[expert][10] = 0.04
            directionWeights[expert][11] = expert == 2 ? 0.18 : 0.06

            skipWeights[expert][0] = expert == 3 ? 0.25 : -0.10
            skipWeights[expert][5] = expert == 3 ? 0.18 : 0.05
            skipWeights[expert][8] = expert == 3 ? 0.20 : 0.08
        }
    }

    private func rawMixture(
        derived: [Double],
        gates: [Double],
        context: PluginContextV4
    ) -> (rawProbabilities: [Double], expectedMove: Double, edgeHat: Double, disagreement: Double) {
        var buy = 0.0
        var sell = 0.0
        var skip = 0.0
        var expectedMove = 0.0
        var meanUp = 0.0
        for expert in 0..<Self.expertCount {
            let up = predictExpertUp(expert, derived)
            let move = predictExpertMove(expert, derived)
            let skipProbability = predictExpertSkip(expert, derived)
            buy += gates[expert] * (1.0 - skipProbability) * up
            sell += gates[expert] * (1.0 - skipProbability) * (1.0 - up)
            skip += gates[expert] * skipProbability
            expectedMove += gates[expert] * move
            meanUp += gates[expert] * up
        }
        var disagreement = 0.0
        for expert in 0..<Self.expertCount {
            disagreement += gates[expert] * abs(predictExpertUp(expert, derived) - meanUp)
        }

        let minimumMove = max(context.minMovePoints, max(context.priceCostPoints, 0.10))
        let edgeHat = max(0.0, expectedMove)
        let tradableRatio = edgeHat / max(minimumMove, 0.10)
        let stress = fxClamp(0.55 * abs(derived[5]) + 0.45 * abs(derived[8]), 0.0, 8.0)
        let confidencePenalty = fxClamp(0.70 * disagreement + 0.12 * stress, 0.0, 0.95)
        let active = fxClamp(
            (0.45 * PluginSupportTools.sigmoid(1.20 * tradableRatio) +
                0.40 * (1.0 - confidencePenalty) +
                0.15 * (1.0 - skip)) * (1.0 - 0.35 * skip),
            0.0,
            1.0
        )
        let directionalTotal = max(buy + sell, 1.0e-9)
        let buyShare = buy / directionalTotal
        let sellShare = sell / directionalTotal
        let raw = PluginContextRuntimeTools.normalizeClassDistribution([
            Self.clampProbability(active * sellShare),
            Self.clampProbability(active * buyShare),
            Self.clampProbability(max(skip, 1.0 - active + 0.15 * disagreement))
        ])
        return (raw, expectedMove, edgeHat, disagreement)
    }

    private func buildDerived(x: [Double], context: PluginContextV4) -> [Double] {
        let f1 = safeX(x, 1)
        let f2 = safeX(x, 2)
        let f3 = safeX(x, 3)
        let f4 = safeX(x, 4)
        let f5 = safeX(x, 5)
        let f6 = safeX(x, 6)
        let f7 = safeX(x, 7)
        let f8 = safeX(x, 8)
        let f9 = safeX(x, 9)
        let f10 = safeX(x, 10)
        let f11 = safeX(x, 11)
        let f12 = safeX(x, 12)
        let g1 = avgAbsRange(x, 13, 20)
        let g2 = avgAbsRange(x, 21, 32)
        let g3 = avgAbsRange(x, 33, 48)
        let g4 = avgAbsRange(x, 49, 62)
        let directionalImpulse = PluginSupportTools.clipSymmetric(0.48 * f1 + 0.34 * f2 + 0.20 * f3 - 0.10 * f4 + 0.08 * f12, limit: 6.0)
        let volatilityPressure = PluginSupportTools.clipSymmetric(0.70 * abs(f6) + 0.45 * abs(f7) + 0.25 * g1, limit: 6.0)
        let reversionBias = PluginSupportTools.clipSymmetric(-0.45 * f1 + 0.35 * f5 - 0.20 * f9 + 0.10 * f10, limit: 6.0)
        let breakoutPotential = PluginSupportTools.clipSymmetric(0.55 * abs(f2 - f5) + 0.35 * abs(f3 - f4) + 0.15 * g2, limit: 6.0)
        let liquidityStress = PluginSupportTools.clipSymmetric(0.90 * abs(f7) + 0.35 * abs(f8) + 0.10 * g3, limit: 6.0)
        let asymmetryProxy = PluginSupportTools.clipSymmetric(0.45 * f10 - 0.35 * f11 + 0.25 * f12, limit: 6.0)
        let smoothTrend = PluginSupportTools.clipSymmetric(0.65 * f1 + 0.25 * f2 - 0.12 * f5 + 0.08 * g4, limit: 6.0)
        let noiseProxy = PluginSupportTools.clipSymmetric(0.35 * abs(f3 - f2) + 0.35 * abs(f5 - f4) + 0.20 * g1 + 0.10 * g4, limit: 6.0)
        let parityStrain = PluginSupportTools.clipSymmetric(0.40 * f8 + 0.25 * f11 + 0.12 * g2 - 0.10 * g3, limit: 6.0)
        let sessionBias: Double
        switch context.sessionBucket {
        case 1:
            sessionBias = 0.20
        case 2:
            sessionBias = 0.35
        case 3:
            sessionBias = 0.15
        default:
            sessionBias = -0.10
        }
        return [
            1.0,
            directionalImpulse,
            volatilityPressure,
            reversionBias,
            breakoutPotential,
            liquidityStress,
            asymmetryProxy,
            smoothTrend,
            noiseProxy,
            parityStrain + sessionBias
        ]
    }

    private func buildLatentInput(_ derived: [Double], expert: Int) -> [Double] {
        [
            1.0,
            derived[1],
            derived[2],
            derived[3],
            derived[4],
            derived[5],
            derived[6],
            derived[7],
            derived[8],
            latent[expert][0],
            latent[expert][1],
            latent[expert][2] + 0.50 * globalState[0] - 0.30 * globalState[4]
        ]
    }

    private func buildMoveInput(_ derived: [Double], expert: Int) -> [Double] {
        [
            1.0,
            abs(derived[1]),
            abs(derived[2]),
            abs(derived[4]),
            abs(latent[expert][0]),
            abs(latent[expert][1]),
            abs(edgeEMA[expert]),
            abs(globalEdgeEMA)
        ]
    }

    private func softmaxExperts(_ derived: [Double]) -> [Double] {
        var logits = Array(repeating: 0.0, count: Self.expertCount)
        var maximum = -Double.greatestFiniteMagnitude
        for expert in 0..<Self.expertCount {
            var value = Self.dot(gateWeights[expert], derived)
            value -= 0.35 * (usageEMA[expert] - 1.0 / Double(Self.expertCount))
            value += 0.15 * hitEMA[expert] - 0.10 * confidenceEMA[expert]
            value = PluginSupportTools.clipSymmetric(value, limit: 20.0)
            logits[expert] = value
            maximum = max(maximum, value)
        }
        var values = Array(repeating: 0.0, count: Self.expertCount)
        var sum = 0.0
        for expert in 0..<Self.expertCount {
            let value = exp(PluginSupportTools.clipSymmetric(logits[expert] - maximum, limit: 30.0))
            values[expert] = value
            sum += value
        }
        guard sum > 0.0 else {
            return Array(repeating: 1.0 / Double(Self.expertCount), count: Self.expertCount)
        }
        return values.map { $0 / sum }
    }

    private func predictExpertSkip(_ expert: Int, _ derived: [Double]) -> Double {
        var value = Self.dot(skipWeights[expert], derived)
        value += 0.22 * abs(derived[5]) + 0.14 * abs(derived[8]) - 0.18 * hitEMA[expert] + 0.10 * confidenceEMA[expert]
        return Self.clampProbability(PluginSupportTools.sigmoid(PluginSupportTools.clipSymmetric(value, limit: 20.0)))
    }

    private func predictExpertUp(_ expert: Int, _ derived: [Double]) -> Double {
        let latentInput = buildLatentInput(derived, expert: expert)
        var value = Self.dot(directionWeights[expert], latentInput)
        value += 0.10 * (hitEMA[expert] - 0.50) - 0.08 * confidenceEMA[expert]
        return Self.clampProbability(PluginSupportTools.sigmoid(PluginSupportTools.clipSymmetric(value, limit: 20.0)))
    }

    private func predictExpertMove(_ expert: Int, _ derived: [Double]) -> Double {
        let moveInput = buildMoveInput(derived, expert: expert)
        let fitted = Self.dot(moveHeadWeights[expert], moveInput)
        let base = 0.45 * abs(derived[1]) + 0.35 * abs(derived[4]) + 0.20 * abs(latent[expert][0])
        return fxClamp(max(0.0, base + fitted), 0.0, 5000.0)
    }

    private mutating func updateLatentState(derived: [Double], gates: [Double], movePoints: Double) {
        let direction = PluginSupportTools.clipSymmetric(movePoints, limit: 10.0)
        let magnitude = abs(direction)
        for expert in 0..<Self.expertCount {
            let fast = 0.08 + 0.04 * gates[expert]
            let slow = 0.02 + 0.02 * gates[expert]
            latent[expert][0] = (1.0 - fast) * latent[expert][0] + fast * derived[1]
            latent[expert][1] = (1.0 - slow) * latent[expert][1] + slow * derived[7]
            latent[expert][2] = (1.0 - fast) * latent[expert][2] + fast * derived[3]
            latent[expert][3] = (1.0 - slow) * latent[expert][3] + slow * derived[2]
            latent[expert][4] = (1.0 - fast) * latent[expert][4] + fast * derived[5]
            latent[expert][5] = (1.0 - slow) * latent[expert][5] + slow * direction
            confidenceEMA[expert] = 0.97 * confidenceEMA[expert] + 0.03 * abs(derived[8])
            edgeEMA[expert] = 0.95 * edgeEMA[expert] + 0.05 * magnitude
            expertMass[expert] = min(1.0e6, expertMass[expert] + gates[expert])
        }
        for index in 0..<Self.stateCount {
            globalState[index] *= 0.97
        }
        globalState[0] += 0.03 * derived[1]
        globalState[1] += 0.03 * derived[2]
        globalState[2] += 0.03 * derived[3]
        globalState[3] += 0.03 * derived[4]
        globalState[4] += 0.03 * derived[5]
        globalState[5] += 0.03 * direction
    }

    private mutating func trainExpertSample(
        expert: Int,
        derived: [Double],
        label: LabelClass,
        movePoints: Double,
        targetMove: Double,
        sampleWeight: Double,
        hyperParameters: HyperParameters,
        gateWeight: Double,
        adaptGate: Bool
    ) {
        let directionTarget = Self.targetDirection(label: label, movePoints: movePoints)
        let y01 = directionTarget > 0 ? 1.0 : 0.0
        let up = predictExpertUp(expert, derived)
        let move = predictExpertMove(expert, derived)
        let skip = predictExpertSkip(expert, derived)
        let l2 = fxClamp(hyperParameters.l2, 0.0, 0.05)
        let boundedGate = fxClamp(gateWeight, 0.15, 1.50)

        if directionTarget != 0 {
            let error = y01 - up
            let latentInput = buildLatentInput(derived, expert: expert)
            let learningRate = fxClamp(0.25 * hyperParameters.learningRate * sampleWeight * boundedGate, 0.0002, 0.05)
            for index in 0..<Self.latentCount {
                let gradient = error * latentInput[index] - l2 * directionWeights[expert][index]
                directionG2[expert][index] += gradient * gradient
                let step = learningRate / sqrt(1.0 + directionG2[expert][index])
                directionWeights[expert][index] = PluginSupportTools.clipSymmetric(directionWeights[expert][index] + step * gradient, limit: 4.0)
            }
        }

        let targetSkip = label == .skip ? 1.0 : 0.0
        let skipError = targetSkip - skip
        let skipLearningRate = fxClamp(0.18 * hyperParameters.learningRate * sampleWeight * (0.40 + boundedGate), 0.0002, 0.03)
        for index in 0..<Self.derivedCount {
            let gradient = skipError * derived[index] - 0.50 * l2 * skipWeights[expert][index]
            skipG2[expert][index] += gradient * gradient
            let step = skipLearningRate / sqrt(1.0 + skipG2[expert][index])
            skipWeights[expert][index] = PluginSupportTools.clipSymmetric(skipWeights[expert][index] + step * gradient, limit: 3.0)
        }

        let moveInput = buildMoveInput(derived, expert: expert)
        let moveError = targetMove - move
        let moveLearningRate = fxClamp(0.20 * hyperParameters.learningRate * sampleWeight * (0.40 + boundedGate), 0.0002, 0.03)
        for index in 0..<Self.moveFeatureCount {
            let gradient = moveError * moveInput[index] - 0.25 * l2 * moveHeadWeights[expert][index]
            moveG2[expert][index] += gradient * gradient
            let step = moveLearningRate / sqrt(1.0 + moveG2[expert][index])
            moveHeadWeights[expert][index] = PluginSupportTools.clipSymmetric(moveHeadWeights[expert][index] + step * gradient, limit: 8.0)
        }

        if adaptGate {
            var align = 0.0
            if directionTarget != 0 {
                align = directionTarget > 0 ? (up - 0.5) : (0.5 - up)
            }
            let overload = usageEMA[expert] - 1.0 / Double(Self.expertCount)
            let reward = PluginSupportTools.clipSymmetric(
                0.70 * align + 0.20 * (targetMove > 0.0 ? 1.0 : -0.5) - 0.35 * overload - 0.20 * targetSkip,
                limit: 1.2
            )
            let gateLearningRate = fxClamp(0.05 * hyperParameters.learningRate * sampleWeight, 0.0001, 0.01)
            for index in 0..<Self.derivedCount {
                gateWeights[expert][index] = PluginSupportTools.clipSymmetric(
                    gateWeights[expert][index] + gateLearningRate * reward * derived[index],
                    limit: 2.5
                )
            }
        }

        if directionTarget != 0 {
            let hit = (directionTarget > 0 && up >= 0.5) || (directionTarget < 0 && up < 0.5) ? 1.0 : 0.0
            hitEMA[expert] = 0.98 * hitEMA[expert] + 0.02 * hit
        } else {
            hitEMA[expert] = 0.985 * hitEMA[expert] + 0.015 * 0.50
        }
        edgeEMA[expert] = 0.97 * edgeEMA[expert] + 0.03 * targetMove
    }

    private mutating func storeReplay(expert: Int, derived: [Double], label: LabelClass, movePoints: Double) {
        let slot = replayHead[expert]
        replayDerived[expert][slot] = derived
        replayMove[expert][slot] = movePoints
        replayLabel[expert][slot] = label
        replayHead[expert] = (slot + 1) % Self.replayCount
        replayFilled[expert] = min(replayFilled[expert] + 1, Self.replayCount)
    }

    private mutating func replayExpert(_ expert: Int, hyperParameters: HyperParameters) {
        guard replayFilled[expert] > 0 else { return }
        var slot = replayHead[expert] - 1
        if slot < 0 {
            slot += Self.replayCount
        }
        trainExpertSample(
            expert: expert,
            derived: replayDerived[expert][slot],
            label: replayLabel[expert][slot],
            movePoints: replayMove[expert][slot],
            targetMove: abs(replayMove[expert][slot]),
            sampleWeight: 0.45,
            hyperParameters: hyperParameters,
            gateWeight: 0.55,
            adaptGate: false
        )
    }

    private mutating func updateMoveEMA(_ targetMove: Double) {
        let value = abs(fxSafeFinite(targetMove))
        if !moveReady {
            globalEdgeEMA = value
            moveEMAAbs = value
            moveReady = true
        } else {
            globalEdgeEMA = 0.97 * globalEdgeEMA + 0.03 * value
            moveEMAAbs = 0.95 * moveEMAAbs + 0.05 * value
        }
    }

    private func safeX(_ x: [Double], _ index: Int) -> Double {
        guard index >= 0, index < x.count else { return 0.0 }
        return PluginSupportTools.clipSymmetric(fxSafeFinite(x[index]), limit: 8.0)
    }

    private func avgAbsRange(_ x: [Double], _ start: Int, _ end: Int) -> Double {
        guard start <= end else { return 0.0 }
        var sum = 0.0
        var count = 0
        for index in start...end {
            sum += abs(safeX(x, index))
            count += 1
        }
        return count > 0 ? sum / Double(count) : 0.0
    }

    private static func targetDirection(label: LabelClass, movePoints: Double) -> Int {
        if label == .buy {
            return 1
        }
        if label == .sell {
            return -1
        }
        if movePoints > 0.0 {
            return 1
        }
        if movePoints < 0.0 {
            return -1
        }
        return 0
    }

    private static func clampProbability(_ probability: Double) -> Double {
        fxClamp(probability, 0.0005, 0.9995)
    }

    private static func preparedFeatures(_ x: [Double], dataHasVolume: Bool) -> [Double] {
        var output = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        for index in 0..<FXDataEngineConstants.aiWeights {
            output[index] = fxClamp(index < x.count ? fxSafeFinite(x[index]) : 0.0, -50.0, 50.0)
        }
        if !dataHasVolume {
            zeroVolumeFeatures(&output)
        }
        return output
    }

    private static func preparedWindow(_ window: [[Double]], dataHasVolume: Bool) -> [[Double]] {
        window.map { preparedFeatures($0, dataHasVolume: dataHasVolume) }
    }

    private static func zeroVolumeFeatures(_ features: inout [Double]) {
        for index in [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83] where index < features.count {
            features[index] = 0.0
        }
    }

    private static func dot(_ lhs: [Double], _ rhs: [Double]) -> Double {
        var value = 0.0
        let count = min(lhs.count, rhs.count)
        for index in 0..<count {
            value += lhs[index] * rhs[index]
        }
        return value
    }
}
