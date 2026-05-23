import Foundation

public enum TreeXGBFastReference {
    public struct SplitGain: Equatable, Sendable {
        public let leftGain: Double
        public let rightGain: Double
        public let parentGain: Double
        public let totalGain: Double
    }

    public static func leafWeight(gradient: Double, hessian: Double, lambda: Double) -> Double {
        -gradient / max(hessian + lambda, 1.0e-12)
    }

    public static func gain(
        leftGradient: Double,
        leftHessian: Double,
        rightGradient: Double,
        rightHessian: Double,
        lambda: Double,
        gamma: Double
    ) -> SplitGain {
        let parentGradient = leftGradient + rightGradient
        let parentHessian = leftHessian + rightHessian
        let left = score(gradient: leftGradient, hessian: leftHessian, lambda: lambda)
        let right = score(gradient: rightGradient, hessian: rightHessian, lambda: lambda)
        let parent = score(gradient: parentGradient, hessian: parentHessian, lambda: lambda)
        return SplitGain(leftGain: left, rightGain: right, parentGain: parent, totalGain: 0.5 * (left + right - parent) - gamma)
    }

    public static func missingGoesLeft(
        leftGradient: Double,
        leftHessian: Double,
        rightGradient: Double,
        rightHessian: Double,
        missingGradient: Double,
        missingHessian: Double,
        lambda: Double,
        gamma: Double
    ) -> Bool {
        let leftGain = gain(
            leftGradient: leftGradient + missingGradient,
            leftHessian: leftHessian + missingHessian,
            rightGradient: rightGradient,
            rightHessian: rightHessian,
            lambda: lambda,
            gamma: gamma
        ).totalGain
        let rightGain = gain(
            leftGradient: leftGradient,
            leftHessian: leftHessian,
            rightGradient: rightGradient + missingGradient,
            rightHessian: rightHessian + missingHessian,
            lambda: lambda,
            gamma: gamma
        ).totalGain
        return leftGain >= rightGain
    }

    private static func score(gradient: Double, hessian: Double, lambda: Double) -> Double {
        gradient * gradient / max(hessian + lambda, 1.0e-12)
    }
}
