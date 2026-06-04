import Foundation

public struct MLFinancialLossSpec: Codable, Hashable, Sendable {
    public static let defaultVersion = "fxai-financial-loss-v1"

    public var version: String
    public var classificationWeight: Double
    public var moveWeight: Double
    public var quantileWeight: Double
    public var adverseTailWeight: Double
    public var costRiskWeight: Double
    public var activityWeight: Double
    public var downsideUtilityWeight: Double
    public var maxTailMultiplier: Double
    public var targetTradeProbability: Double
    public var utilityEpsilon: Double

    public init(
        version: String = Self.defaultVersion,
        classificationWeight: Double = 1.0,
        moveWeight: Double = 0.05,
        quantileWeight: Double = 0.02,
        adverseTailWeight: Double = 0.18,
        costRiskWeight: Double = 0.10,
        activityWeight: Double = 0.04,
        downsideUtilityWeight: Double = 0.03,
        maxTailMultiplier: Double = 4.0,
        targetTradeProbability: Double = 0.35,
        utilityEpsilon: Double = 1e-6
    ) {
        let trimmedVersion = version.trimmingCharacters(in: .whitespacesAndNewlines)
        self.version = trimmedVersion.isEmpty ? Self.defaultVersion : trimmedVersion
        self.classificationWeight = fxClamp(fxSafeFinite(classificationWeight, fallback: 1.0), 0.0, 10.0)
        self.moveWeight = fxClamp(fxSafeFinite(moveWeight, fallback: 0.05), 0.0, 5.0)
        self.quantileWeight = fxClamp(fxSafeFinite(quantileWeight, fallback: 0.02), 0.0, 5.0)
        self.adverseTailWeight = fxClamp(fxSafeFinite(adverseTailWeight, fallback: 0.18), 0.0, 5.0)
        self.costRiskWeight = fxClamp(fxSafeFinite(costRiskWeight, fallback: 0.10), 0.0, 5.0)
        self.activityWeight = fxClamp(fxSafeFinite(activityWeight, fallback: 0.04), 0.0, 5.0)
        self.downsideUtilityWeight = fxClamp(fxSafeFinite(downsideUtilityWeight, fallback: 0.03), 0.0, 5.0)
        self.maxTailMultiplier = fxClamp(fxSafeFinite(maxTailMultiplier, fallback: 4.0), 1.0, 20.0)
        self.targetTradeProbability = fxClamp(fxSafeFinite(targetTradeProbability, fallback: 0.35), 0.0, 1.0)
        self.utilityEpsilon = fxClamp(fxSafeFinite(utilityEpsilon, fallback: 1e-6), 1e-12, 1.0)
    }

    public static var defaultFinancialUtility: MLFinancialLossSpec {
        MLFinancialLossSpec()
    }

    public func validate() throws {
        guard version == Self.defaultVersion else {
            throw FXDataEngineError.validation("mlFinancialLoss.version")
        }
        for (name, value) in [
            ("classificationWeight", classificationWeight),
            ("moveWeight", moveWeight),
            ("quantileWeight", quantileWeight),
            ("adverseTailWeight", adverseTailWeight),
            ("costRiskWeight", costRiskWeight),
            ("activityWeight", activityWeight),
            ("downsideUtilityWeight", downsideUtilityWeight),
            ("maxTailMultiplier", maxTailMultiplier),
            ("targetTradeProbability", targetTradeProbability),
            ("utilityEpsilon", utilityEpsilon)
        ] {
            guard value.isFinite, value >= 0.0 else {
                throw FXDataEngineError.validation("mlFinancialLoss.\(name)")
            }
        }
        guard maxTailMultiplier >= 1.0 else {
            throw FXDataEngineError.validation("mlFinancialLoss.maxTailMultiplier")
        }
        guard targetTradeProbability <= 1.0 else {
            throw FXDataEngineError.validation("mlFinancialLoss.targetTradeProbability")
        }
        guard utilityEpsilon > 0.0 else {
            throw FXDataEngineError.validation("mlFinancialLoss.utilityEpsilon")
        }
    }
}

public struct MLFinancialTrainingTargets: Codable, Hashable, Sendable {
    public var labelClass: LabelClass
    public var movePoints: Double
    public var sampleWeight: Double
    public var mfePoints: Double
    public var maePoints: Double
    public var timeToHitFraction: Double
    public var pathFlags: Int
    public var pathRisk: Double
    public var fillRisk: Double
    public var maskedStepTarget: Double
    public var nextVolumeTarget: Double
    public var regimeShiftTarget: Double
    public var contextLeadTarget: Double
    public var priceCostPoints: Double
    public var minMovePoints: Double

    public init(
        labelClass: LabelClass = .skip,
        movePoints: Double = 0.0,
        sampleWeight: Double = 1.0,
        mfePoints: Double = 0.0,
        maePoints: Double = 0.0,
        timeToHitFraction: Double = 1.0,
        pathFlags: Int = 0,
        pathRisk: Double = 0.0,
        fillRisk: Double = 0.0,
        maskedStepTarget: Double = 0.0,
        nextVolumeTarget: Double = 0.0,
        regimeShiftTarget: Double = 0.0,
        contextLeadTarget: Double = 0.5,
        priceCostPoints: Double = 0.0,
        minMovePoints: Double = 0.0
    ) {
        self.labelClass = labelClass
        self.movePoints = fxSafeFinite(movePoints)
        self.sampleWeight = max(0.0, fxSafeFinite(sampleWeight, fallback: 1.0))
        self.mfePoints = max(0.0, fxSafeFinite(mfePoints))
        self.maePoints = max(0.0, fxSafeFinite(maePoints))
        self.timeToHitFraction = fxClamp(fxSafeFinite(timeToHitFraction, fallback: 1.0), 0.0, 1.0)
        self.pathFlags = pathFlags
        self.pathRisk = fxClamp(fxSafeFinite(pathRisk), 0.0, 1.0)
        self.fillRisk = fxClamp(fxSafeFinite(fillRisk), 0.0, 1.0)
        self.maskedStepTarget = fxSafeFinite(maskedStepTarget)
        self.nextVolumeTarget = max(0.0, fxSafeFinite(nextVolumeTarget))
        self.regimeShiftTarget = fxClamp(fxSafeFinite(regimeShiftTarget), 0.0, 1.0)
        self.contextLeadTarget = fxClamp(fxSafeFinite(contextLeadTarget, fallback: 0.5), 0.0, 1.0)
        self.priceCostPoints = max(0.0, fxSafeFinite(priceCostPoints))
        self.minMovePoints = max(0.0, fxSafeFinite(minMovePoints))
    }

    public init(request: TrainRequestV4) {
        self.init(
            labelClass: request.labelClass,
            movePoints: request.movePoints,
            sampleWeight: request.sampleWeight,
            mfePoints: request.mfePoints,
            maePoints: request.maePoints,
            timeToHitFraction: request.timeToHitFraction,
            pathFlags: request.pathFlags,
            pathRisk: request.pathRisk,
            fillRisk: request.fillRisk,
            maskedStepTarget: request.maskedStepTarget,
            nextVolumeTarget: request.nextVolumeTarget,
            regimeShiftTarget: request.regimeShiftTarget,
            contextLeadTarget: request.contextLeadTarget,
            priceCostPoints: request.context.priceCostPoints,
            minMovePoints: request.context.minMovePoints
        )
    }

    public var absoluteMovePoints: Double {
        abs(movePoints)
    }

    public func validate() throws {
        guard movePoints.isFinite else {
            throw FXDataEngineError.validation("mlFinancialTargets.movePoints")
        }
        guard sampleWeight.isFinite, sampleWeight >= 0.0 else {
            throw FXDataEngineError.validation("mlFinancialTargets.sampleWeight")
        }
        guard mfePoints.isFinite, mfePoints >= 0.0,
              maePoints.isFinite, maePoints >= 0.0 else {
            throw FXDataEngineError.validation("mlFinancialTargets.pathExcursions")
        }
        guard timeToHitFraction.isFinite, (0.0...1.0).contains(timeToHitFraction) else {
            throw FXDataEngineError.validation("mlFinancialTargets.timeToHitFraction")
        }
        guard pathRisk.isFinite, (0.0...1.0).contains(pathRisk) else {
            throw FXDataEngineError.validation("mlFinancialTargets.pathRisk")
        }
        guard fillRisk.isFinite, (0.0...1.0).contains(fillRisk) else {
            throw FXDataEngineError.validation("mlFinancialTargets.fillRisk")
        }
        guard maskedStepTarget.isFinite else {
            throw FXDataEngineError.validation("mlFinancialTargets.maskedStepTarget")
        }
        guard nextVolumeTarget.isFinite, nextVolumeTarget >= 0.0 else {
            throw FXDataEngineError.validation("mlFinancialTargets.nextVolumeTarget")
        }
        guard regimeShiftTarget.isFinite, (0.0...1.0).contains(regimeShiftTarget) else {
            throw FXDataEngineError.validation("mlFinancialTargets.regimeShiftTarget")
        }
        guard contextLeadTarget.isFinite, (0.0...1.0).contains(contextLeadTarget) else {
            throw FXDataEngineError.validation("mlFinancialTargets.contextLeadTarget")
        }
        guard priceCostPoints.isFinite, priceCostPoints >= 0.0 else {
            throw FXDataEngineError.validation("mlFinancialTargets.priceCostPoints")
        }
        guard minMovePoints.isFinite, minMovePoints >= 0.0 else {
            throw FXDataEngineError.validation("mlFinancialTargets.minMovePoints")
        }
    }
}
