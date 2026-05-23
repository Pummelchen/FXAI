import CoreGraphics
import SwiftUI

public struct ShadowLightSource {
    public let normalizedPosition: CGPoint
    public let referenceDirection: CGVector
    public let lateralResponse: CGFloat
    public let verticalResponse: CGFloat
    public let poolingResponse: CGFloat

    public init(
        normalizedPosition: CGPoint,
        referenceDirection: CGVector,
        lateralResponse: CGFloat = 0.64,
        verticalResponse: CGFloat = 0.56,
        poolingResponse: CGFloat = 1.12
    ) {
        self.normalizedPosition = normalizedPosition
        self.referenceDirection = ShadowProjector.normalized(referenceDirection, fallback: CGVector(dx: -0.78, dy: 0.62))
        self.lateralResponse = lateralResponse
        self.verticalResponse = verticalResponse
        self.poolingResponse = poolingResponse
    }

    public func position(in stageSize: CGSize) -> CGPoint {
        CGPoint(
            x: normalizedPosition.x * stageSize.width,
            y: normalizedPosition.y * stageSize.height
        )
    }
}

public struct ShadowLayer: Identifiable, Hashable {
    public let id = UUID()
    public let color: Color
    public let radius: CGFloat
    public let x: CGFloat
    public let y: CGFloat
    public let opacity: Double
    public let lightInfluence: CGFloat
    public let lateralResponse: CGFloat
    public let verticalResponse: CGFloat
    public let poolingBias: CGFloat

    public init(
        color: Color = .black,
        radius: CGFloat,
        x: CGFloat,
        y: CGFloat,
        opacity: Double,
        lightInfluence: CGFloat = 0.88,
        lateralResponse: CGFloat = 0.60,
        verticalResponse: CGFloat = 0.48,
        poolingBias: CGFloat = 0.20
    ) {
        self.color = color
        self.radius = radius
        self.x = x
        self.y = y
        self.opacity = opacity
        self.lightInfluence = lightInfluence
        self.lateralResponse = lateralResponse
        self.verticalResponse = verticalResponse
        self.poolingBias = poolingBias
    }
}

public struct ShadowStack {
    public let layers: [ShadowLayer]

    public init(_ layers: [ShadowLayer]) {
        self.layers = layers
    }
}

public struct ShadowProjectionContext {
    public let frame: CGRect
    public let stageSize: CGSize
    public let lightSource: ShadowLightSource

    public init(frame: CGRect, stageSize: CGSize, lightSource: ShadowLightSource) {
        self.frame = frame
        self.stageSize = stageSize
        self.lightSource = lightSource
    }
}

public struct ProjectedShadowLayer {
    public let color: Color
    public let blurRadius: CGFloat
    public let offset: CGSize
    public let opacity: Double

    public init(color: Color, blurRadius: CGFloat, offset: CGSize, opacity: Double) {
        self.color = color
        self.blurRadius = blurRadius
        self.offset = offset
        self.opacity = opacity
    }
}

public enum ShadowProjector {
    public static func resolve(
        stack: ShadowStack,
        context: ShadowProjectionContext?,
        scale: CGFloat
    ) -> [ProjectedShadowLayer] {
        guard
            let context,
            context.stageSize.width > 0,
            context.stageSize.height > 0
        else {
            return stack.layers.map { layer in
                ProjectedShadowLayer(
                    color: layer.color,
                    blurRadius: layer.radius * scale,
                    offset: CGSize(width: layer.x * scale, height: layer.y * scale),
                    opacity: layer.opacity
                )
            }
        }

        let lightPosition = context.lightSource.position(in: context.stageSize)
        let center = CGPoint(x: context.frame.midX, y: context.frame.midY)
        let vector = CGVector(dx: center.x - lightPosition.x, dy: center.y - lightPosition.y)
        let actualDirection = normalized(vector, fallback: context.lightSource.referenceDirection)
        let referenceDirection = context.lightSource.referenceDirection
        let distance = hypot(vector.dx, vector.dy)
        let maxDistance = max(context.stageSize.width, context.stageSize.height)
        let distanceFactor = max(0.72, min(1.15, maxDistance > 0 ? distance / maxDistance : 1))
        let horizontalDelta = actualDirection.dx - referenceDirection.dx
        let positiveVerticalDelta = max(0, actualDirection.dy - referenceDirection.dy)
        let negativeVerticalDelta = min(0, actualDirection.dy - referenceDirection.dy) * 0.18

        return stack.layers.map { layer in
            let xDelta = abs(layer.x) * horizontalDelta * layer.lightInfluence * context.lightSource.lateralResponse * layer.lateralResponse * distanceFactor
            let verticalDelta = abs(layer.y) * (
                (positiveVerticalDelta * layer.lightInfluence * context.lightSource.verticalResponse * layer.verticalResponse) +
                (negativeVerticalDelta * layer.lightInfluence * 0.35)
            )
            let poolingDelta = positiveVerticalDelta * layer.poolingBias * layer.radius * context.lightSource.poolingResponse * distanceFactor
            let blurBoost = 1 + (positiveVerticalDelta * 0.08) + ((distanceFactor - 0.72) * 0.06)
            let opacityBoost = 1 + (positiveVerticalDelta * 0.07)

            return ProjectedShadowLayer(
                color: layer.color,
                blurRadius: layer.radius * blurBoost * scale,
                offset: CGSize(
                    width: (layer.x + xDelta) * scale,
                    height: (layer.y + verticalDelta + poolingDelta) * scale
                ),
                opacity: max(0, min(1, layer.opacity * opacityBoost))
            )
        }
    }

    static func normalized(_ vector: CGVector, fallback: CGVector) -> CGVector {
        let length = hypot(vector.dx, vector.dy)
        guard length > 0.0001 else { return fallback }
        return CGVector(dx: vector.dx / length, dy: vector.dy / length)
    }
}

public struct ThemeShadows {
    public let lightSource: ShadowLightSource
    public let kpiCard: ShadowStack
    public let pendingCard: ShadowStack
    public let smallCard: ShadowStack
    public let gaugeCard: ShadowStack
    public let amountOwed: ShadowStack
    public let chartBarPrimary: ShadowStack
    public let chartBarDefault: ShadowStack
    public let footer: ShadowStack

    public init(
        lightSource: ShadowLightSource,
        kpiCard: ShadowStack,
        pendingCard: ShadowStack,
        smallCard: ShadowStack,
        gaugeCard: ShadowStack,
        amountOwed: ShadowStack,
        chartBarPrimary: ShadowStack,
        chartBarDefault: ShadowStack,
        footer: ShadowStack
    ) {
        self.lightSource = lightSource
        self.kpiCard = kpiCard
        self.pendingCard = pendingCard
        self.smallCard = smallCard
        self.gaugeCard = gaugeCard
        self.amountOwed = amountOwed
        self.chartBarPrimary = chartBarPrimary
        self.chartBarDefault = chartBarDefault
        self.footer = footer
    }
}
