import Foundation

public enum ExecutionProfileID: Int, Codable, Sendable, CaseIterable {
    case defaultProfile = 0
    case tightFX
    case primeECN
    case retailFX
    case stress
}

public struct ExecutionProfile: Codable, Hashable, Sendable {
    public var profileID: ExecutionProfileID
    public var commissionPerLotSide: Double
    public var costBufferPoints: Double
    public var slippagePoints: Double
    public var fillPenaltyPoints: Double
    public var slippageCostWeight: Double
    public var slippageStressWeight: Double
    public var slippageHorizonWeight: Double
    public var dualHitPenalty: Double
    public var slowHitPenalty: Double
    public var liquidityShockPenalty: Double
    public var partialFillPenalty: Double
    public var latencyPenaltyPoints: Double
    public var allowedDeviationPoints: Double

    public init(
        profileID: ExecutionProfileID = .defaultProfile,
        commissionPerLotSide: Double = 0.0,
        costBufferPoints: Double = 2.0,
        slippagePoints: Double = 0.0,
        fillPenaltyPoints: Double = 0.0,
        slippageCostWeight: Double = 0.04,
        slippageStressWeight: Double = 0.18,
        slippageHorizonWeight: Double = 0.02,
        dualHitPenalty: Double = 0.12,
        slowHitPenalty: Double = 0.10,
        liquidityShockPenalty: Double = 0.25,
        partialFillPenalty: Double = 0.0,
        latencyPenaltyPoints: Double = 0.0,
        allowedDeviationPoints: Double = 2.0
    ) {
        self.profileID = profileID
        self.commissionPerLotSide = max(0.0, fxSafeFinite(commissionPerLotSide))
        self.costBufferPoints = max(0.0, fxSafeFinite(costBufferPoints))
        self.slippagePoints = max(0.0, fxSafeFinite(slippagePoints))
        self.fillPenaltyPoints = max(0.0, fxSafeFinite(fillPenaltyPoints))
        self.slippageCostWeight = fxSafeFinite(slippageCostWeight)
        self.slippageStressWeight = fxSafeFinite(slippageStressWeight)
        self.slippageHorizonWeight = fxSafeFinite(slippageHorizonWeight)
        self.dualHitPenalty = max(0.0, fxSafeFinite(dualHitPenalty))
        self.slowHitPenalty = max(0.0, fxSafeFinite(slowHitPenalty))
        self.liquidityShockPenalty = max(0.0, fxSafeFinite(liquidityShockPenalty))
        self.partialFillPenalty = max(0.0, fxSafeFinite(partialFillPenalty))
        self.latencyPenaltyPoints = max(0.0, fxSafeFinite(latencyPenaltyPoints))
        self.allowedDeviationPoints = max(0.0, fxSafeFinite(allowedDeviationPoints))
    }

    public static func preset(_ profileID: ExecutionProfileID) -> ExecutionProfile {
        var profile = ExecutionProfile(profileID: profileID)
        switch profileID {
        case .defaultProfile:
            return profile
        case .tightFX:
            profile.costBufferPoints = 1.5
            profile.slippagePoints = 0.10
            profile.fillPenaltyPoints = 0.10
            profile.allowedDeviationPoints = 2.0
        case .primeECN:
            profile.commissionPerLotSide = 3.5
            profile.costBufferPoints = 1.5
            profile.slippagePoints = 0.20
            profile.fillPenaltyPoints = 0.15
            profile.allowedDeviationPoints = 2.5
        case .retailFX:
            profile.costBufferPoints = 2.5
            profile.slippagePoints = 0.40
            profile.fillPenaltyPoints = 0.25
            profile.slippageCostWeight = 0.05
            profile.allowedDeviationPoints = 4.0
        case .stress:
            profile.commissionPerLotSide = 5.0
            profile.costBufferPoints = 3.5
            profile.slippagePoints = 1.0
            profile.fillPenaltyPoints = 0.50
            profile.slippageCostWeight = 0.06
            profile.slippageStressWeight = 0.25
            profile.slippageHorizonWeight = 0.03
            profile.dualHitPenalty = 0.20
            profile.slowHitPenalty = 0.16
            profile.liquidityShockPenalty = 0.55
            profile.partialFillPenalty = 0.25
            profile.latencyPenaltyPoints = 0.20
            profile.allowedDeviationPoints = 8.0
        }
        return profile
    }
}

public struct ExecutionReplayFrame: Codable, Hashable, Sendable {
    public var slippageMultiplier: Double
    public var fillMultiplier: Double
    public var latencyAddPoints: Double
    public var rejectProbability: Double
    public var partialFillProbability: Double
    public var driftPenaltyPoints: Double
    public var eventFlags: SamplePathFlags

    public init(
        slippageMultiplier: Double = 1.0,
        fillMultiplier: Double = 1.0,
        latencyAddPoints: Double = 0.0,
        rejectProbability: Double = 0.0,
        partialFillProbability: Double = 0.0,
        driftPenaltyPoints: Double = 0.0,
        eventFlags: SamplePathFlags = []
    ) {
        self.slippageMultiplier = max(0.0, fxSafeFinite(slippageMultiplier))
        self.fillMultiplier = max(0.0, fxSafeFinite(fillMultiplier))
        self.latencyAddPoints = max(0.0, fxSafeFinite(latencyAddPoints))
        self.rejectProbability = fxClamp(rejectProbability, 0.0, 1.0)
        self.partialFillProbability = fxClamp(partialFillProbability, 0.0, 1.0)
        self.driftPenaltyPoints = max(0.0, fxSafeFinite(driftPenaltyPoints))
        self.eventFlags = eventFlags
    }
}

public struct ExecutionTraceStats: Codable, Hashable, Sendable {
    public var liquidityMeanRatio: Double
    public var liquidityPeakRatio: Double
    public var rangeMeanRatio: Double
    public var bodyEfficiency: Double
    public var gapRatio: Double
    public var reversalRatio: Double
    public var sessionTransitionExposure: Double
    public var rolloverExposure: Double

    public init(
        liquidityMeanRatio: Double = 1.0,
        liquidityPeakRatio: Double = 1.0,
        rangeMeanRatio: Double = 1.0,
        bodyEfficiency: Double = 0.5,
        gapRatio: Double = 0.0,
        reversalRatio: Double = 0.0,
        sessionTransitionExposure: Double = 0.0,
        rolloverExposure: Double = 0.0
    ) {
        self.liquidityMeanRatio = fxClamp(liquidityMeanRatio, 0.5, 6.0)
        self.liquidityPeakRatio = fxClamp(liquidityPeakRatio, 1.0, 10.0)
        self.rangeMeanRatio = fxClamp(rangeMeanRatio, 0.25, 8.0)
        self.bodyEfficiency = fxClamp(bodyEfficiency, 0.0, 1.0)
        self.gapRatio = fxClamp(gapRatio, 0.0, 8.0)
        self.reversalRatio = fxClamp(reversalRatio, 0.0, 1.0)
        self.sessionTransitionExposure = fxClamp(sessionTransitionExposure, 0.0, 1.0)
        self.rolloverExposure = fxClamp(rolloverExposure, 0.0, 1.0)
    }
}

public struct BrokerExecutionStats: Codable, Hashable, Sendable {
    public var coverage: Double
    public var slippagePoints: Double
    public var latencyPoints: Double
    public var rejectProbability: Double
    public var partialFillProbability: Double
    public var fillRatioMean: Double
    public var eventBurstPenalty: Double

    public init(
        coverage: Double = 0.0,
        slippagePoints: Double = 0.0,
        latencyPoints: Double = 0.0,
        rejectProbability: Double = 0.0,
        partialFillProbability: Double = 0.0,
        fillRatioMean: Double = 1.0,
        eventBurstPenalty: Double = 0.0
    ) {
        self.coverage = fxClamp(coverage, 0.0, 1.0)
        self.slippagePoints = max(0.0, fxSafeFinite(slippagePoints))
        self.latencyPoints = max(0.0, fxSafeFinite(latencyPoints))
        self.rejectProbability = fxClamp(rejectProbability, 0.0, 1.0)
        self.partialFillProbability = fxClamp(partialFillProbability, 0.0, 1.0)
        self.fillRatioMean = fxClamp(fillRatioMean, 0.0, 1.0)
        self.eventBurstPenalty = fxClamp(eventBurstPenalty, 0.0, 1.0)
    }
}

public enum ExecutionReplayTools {
    public static func entryCostPoints(
        priceCostPoints: Double = 0.0,
        commissionPoints: Double,
        baseCostBufferPoints: Double,
        profile: ExecutionProfile
    ) -> Double {
        max(0.0, priceCostPoints) +
            max(0.0, commissionPoints) +
            max(0.0, baseCostBufferPoints) +
            max(0.0, profile.costBufferPoints) +
            max(0.0, profile.slippagePoints) +
            max(0.0, profile.fillPenaltyPoints) +
            max(0.0, profile.latencyPenaltyPoints)
    }

    public static func slippagePoints(
        profile: ExecutionProfile,
        roundTripCostPoints: Double,
        horizonMinutes: Int,
        liquidityStressPoints: Double = 0.0,
        pathFlags: SamplePathFlags = []
    ) -> Double {
        let stress = fxClamp(liquidityStressPoints, 0.0, 4.0)
        var slippage = max(profile.slippagePoints, 0.0) +
            profile.slippageCostWeight * max(roundTripCostPoints, 0.0) +
            profile.slippageStressWeight * stress +
            profile.slippageHorizonWeight * sqrt(Double(max(horizonMinutes, 1))) +
            max(profile.latencyPenaltyPoints, 0.0)
        if pathFlags.contains(.dualHit) {
            slippage += max(profile.dualHitPenalty, 0.0) +
                0.12 * max(roundTripCostPoints, 0.0) +
                0.12 * stress
        }
        if pathFlags.contains(.slowHit) {
            slippage += max(profile.slowHitPenalty, 0.0)
        }
        if pathFlags.contains(.spreadStress) {
            slippage += max(profile.liquidityShockPenalty, 0.0)
        }
        return min(slippage, 12.0)
    }

    public static func replaySlippagePoints(
        profile: ExecutionProfile,
        frame: ExecutionReplayFrame,
        roundTripCostPoints: Double,
        horizonMinutes: Int,
        liquidityStressPoints: Double = 0.0,
        pathFlags: SamplePathFlags = []
    ) -> Double {
        let slippage = slippagePoints(
            profile: profile,
            roundTripCostPoints: roundTripCostPoints,
            horizonMinutes: horizonMinutes,
            liquidityStressPoints: liquidityStressPoints,
            pathFlags: pathFlags.union(frame.eventFlags)
        )
        return min(
            slippage * fxClamp(frame.slippageMultiplier, 0.50, 3.00) +
                max(frame.latencyAddPoints, 0.0) +
                max(frame.driftPenaltyPoints, 0.0),
            18.0
        )
    }

    public static func fillPenaltyPoints(
        profile: ExecutionProfile,
        roundTripCostPoints: Double,
        liquidityStressPoints: Double = 0.0,
        pathFlags: SamplePathFlags = []
    ) -> Double {
        let stress = fxClamp(liquidityStressPoints, 0.0, 4.0)
        var fillPenalty = max(profile.fillPenaltyPoints, 0.0) +
            max(profile.partialFillPenalty, 0.0) * (0.20 + 0.20 * stress)
        if pathFlags.contains(.spreadStress) {
            fillPenalty += 0.10 * max(roundTripCostPoints, 0.0) + 0.12 * stress
        }
        if pathFlags.contains(.dualHit) {
            fillPenalty += 0.08 * max(roundTripCostPoints, 0.0)
        }
        return min(fillPenalty, 10.0)
    }

    public static func replayFillPenaltyPoints(
        profile: ExecutionProfile,
        frame: ExecutionReplayFrame,
        roundTripCostPoints: Double,
        liquidityStressPoints: Double = 0.0,
        pathFlags: SamplePathFlags = []
    ) -> Double {
        var fillPenalty = fillPenaltyPoints(
            profile: profile,
            roundTripCostPoints: roundTripCostPoints,
            liquidityStressPoints: liquidityStressPoints,
            pathFlags: pathFlags.union(frame.eventFlags)
        )
        fillPenalty *= fxClamp(frame.fillMultiplier, 0.50, 3.00)
        fillPenalty += max(profile.partialFillPenalty, 0.0) * fxClamp(frame.partialFillProbability, 0.0, 1.0)
        fillPenalty += 0.50 * max(frame.driftPenaltyPoints, 0.0)
        return min(fillPenalty, 15.0)
    }

    public static func allowedDeviationPoints(
        profile: ExecutionProfile,
        pathRisk: Double,
        fillRisk: Double
    ) -> Double {
        min(
            max(profile.allowedDeviationPoints, 0.0) +
                2.5 * fxClamp(pathRisk, 0.0, 1.0) +
                3.0 * fxClamp(fillRisk, 0.0, 1.0),
            25.0
        )
    }

    public static func buildReplayFrame(
        profile: ExecutionProfile,
        sampleTimeUTC: Int64,
        horizonMinutes: Int,
        liquidityStressPoints: Double = 0.0,
        pathFlags: SamplePathFlags = [],
        scenarioID: Int = 0,
        trace: ExecutionTraceStats = ExecutionTraceStats(),
        brokerStats: BrokerExecutionStats = BrokerExecutionStats()
    ) -> ExecutionReplayFrame {
        let dateParts = utcDateParts(timestamp: sampleTimeUTC)
        let stress = fxClamp(liquidityStressPoints, 0.0, 4.0)
        let horizonScale = sqrt(Double(max(horizonMinutes, 1)))
        let minuteNorm = (Double(dateParts.minute) + Double(dateParts.second) / 60.0) / 60.0
        let sessionEdge = fxClamp(
            1.0 - min(abs(Double(dateParts.hour) - 8.0), abs(Double(dateParts.hour) - 16.0)) / 8.0,
            0.0,
            1.0
        )
        let rolloverEdge = fxClamp(
            1.0 - min(abs(Double(dateParts.hour) - 23.0), 6.0) / 6.0,
            0.0,
            1.0
        )
        let pulse = 0.5 + 0.5 * sin(2.0 * Double.pi * minuteNorm)
        let traceLiquidity = fxClamp(
            0.55 * (trace.liquidityMeanRatio - 1.0) +
                0.45 * (trace.liquidityPeakRatio - 1.0),
            0.0,
            6.0
        )
        let traceRange = fxClamp(trace.rangeMeanRatio - 1.0, -0.5, 4.0)
        let traceGap = fxClamp(trace.gapRatio, 0.0, 6.0)
        let traceReversal = fxClamp(trace.reversalRatio, 0.0, 1.0)
        let traceSession = fxClamp(0.55 * sessionEdge + 0.45 * trace.sessionTransitionExposure, 0.0, 1.0)
        let traceRoll = fxClamp(0.55 * rolloverEdge + 0.45 * trace.rolloverExposure, 0.0, 1.0)
        let bodyPenalty = fxClamp(1.0 - trace.bodyEfficiency, 0.0, 1.0)

        var frame = ExecutionReplayFrame(
            slippageMultiplier: 1.0 +
                0.04 * stress +
                0.05 * traceLiquidity +
                0.03 * max(traceRange, 0.0) +
                0.04 * traceGap +
                0.04 * traceSession +
                0.03 * traceRoll +
                0.015 * horizonScale +
                0.02 * pulse,
            fillMultiplier: 1.0 +
                0.03 * stress +
                0.05 * traceReversal +
                0.03 * bodyPenalty +
                0.03 * traceSession +
                0.02 * traceRoll,
            latencyAddPoints: max(profile.latencyPenaltyPoints, 0.0) *
                (0.32 + 0.22 * traceLiquidity + 0.18 * traceGap + 0.16 * traceSession + 0.12 * traceRoll + 0.08 * stress),
            rejectProbability: fxClamp(
                0.002 +
                    0.008 * stress +
                    0.012 * traceLiquidity +
                    0.010 * traceGap +
                    0.009 * traceSession +
                    0.006 * traceRoll,
                0.0,
                0.35
            ),
            partialFillProbability: fxClamp(
                0.01 +
                    0.030 * stress +
                    0.040 * traceLiquidity +
                    0.030 * traceReversal +
                    0.020 * bodyPenalty +
                    0.020 * traceSession,
                0.0,
                0.95
            ),
            driftPenaltyPoints: 0.05 * max(profile.costBufferPoints, 0.0) *
                (traceSession + traceRoll + 0.35 * traceLiquidity),
            eventFlags: []
        )

        if brokerStats.coverage > 1e-6 {
            let slipRef = max(profile.slippagePoints + 0.25, 0.25)
            let fillShortfall = fxClamp(1.0 - brokerStats.fillRatioMean, 0.0, 1.0)
            let burstPenalty = fxClamp(brokerStats.eventBurstPenalty, 0.0, 1.0)
            let brokerSlipMultiplier = 1.0 +
                0.35 * brokerStats.coverage * fxClamp(brokerStats.slippagePoints / slipRef, 0.0, 3.0)
            frame.slippageMultiplier *= brokerSlipMultiplier
            frame.fillMultiplier *= 1.0 + brokerStats.coverage *
                (0.18 * brokerStats.partialFillProbability + 0.12 * fillShortfall + 0.08 * burstPenalty)
            frame.latencyAddPoints += brokerStats.coverage * brokerStats.latencyPoints
            frame.rejectProbability = fxClamp(
                (1.0 - 0.55 * brokerStats.coverage) * frame.rejectProbability +
                    0.55 * brokerStats.coverage * max(
                        frame.rejectProbability,
                        fxClamp(brokerStats.rejectProbability + 0.20 * burstPenalty, 0.0, 1.0)
                    ),
                0.0,
                0.75
            )
            frame.partialFillProbability = fxClamp(
                (1.0 - 0.55 * brokerStats.coverage) * frame.partialFillProbability +
                    0.55 * brokerStats.coverage * max(
                        frame.partialFillProbability,
                        fxClamp(brokerStats.partialFillProbability + 0.35 * fillShortfall, 0.0, 1.0)
                    ),
                0.0,
                0.99
            )
            frame.driftPenaltyPoints += 0.20 * brokerStats.coverage * brokerStats.slippagePoints +
                0.12 * brokerStats.coverage * burstPenalty * max(profile.costBufferPoints, 1.0)
        }

        if pathFlags.contains(.dualHit) {
            frame.eventFlags.insert(.dualHit)
        }
        if pathFlags.contains(.spreadStress) || trace.liquidityPeakRatio > 1.35 {
            frame.eventFlags.insert(.spreadStress)
        }

        if scenarioID == 11 {
            frame.slippageMultiplier += 0.10
            frame.fillMultiplier += 0.08
            frame.rejectProbability = fxClamp(frame.rejectProbability + 0.05, 0.0, 0.45)
            frame.partialFillProbability = fxClamp(frame.partialFillProbability + 0.10, 0.0, 0.98)
            frame.eventFlags.insert(.slowHit)
        } else if scenarioID == 12 {
            frame.slippageMultiplier += 0.12 + 0.03 * traceLiquidity
            frame.fillMultiplier += 0.10 + 0.02 * traceLiquidity
            frame.latencyAddPoints += 0.20 + 0.10 * stress + 0.06 * traceGap
            frame.rejectProbability = fxClamp(frame.rejectProbability + 0.08, 0.0, 0.50)
            frame.partialFillProbability = fxClamp(frame.partialFillProbability + 0.14, 0.0, 0.99)
            frame.eventFlags.insert(.spreadStress)
        } else if scenarioID == 13 {
            frame.slippageMultiplier += 0.04 + 0.02 * traceReversal
            frame.fillMultiplier += 0.03 + 0.02 * bodyPenalty
            frame.rejectProbability = fxClamp(frame.rejectProbability + 0.03, 0.0, 0.40)
        }

        return frame
    }

    private static func utcDateParts(timestamp: Int64) -> (hour: Int, minute: Int, second: Int) {
        let date = Date(timeIntervalSince1970: TimeInterval(max(timestamp, 0)))
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
        let components = calendar.dateComponents([.hour, .minute, .second], from: date)
        return (components.hour ?? 0, components.minute ?? 0, components.second ?? 0)
    }
}
