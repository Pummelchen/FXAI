import CoreGraphics
import FXAIGUICore
import Testing

struct ShadowProjectionTests {
    @Test
    func lowerCardsPoolShadowFurtherDownFromLight() throws {
        let theme = FinancialDashboardThemeV1()
        let stageSize = theme.layoutMetrics.referenceCanvasSize
        let upperFrame = CGRect(x: 186, y: 229, width: 300, height: 212)
        let lowerFrame = CGRect(x: 186, y: 604, width: 300, height: 212)

        let upper = ShadowProjector.resolve(
            stack: theme.shadows.kpiCard,
            context: ShadowProjectionContext(frame: upperFrame, stageSize: stageSize, lightSource: theme.shadows.lightSource),
            scale: 1
        )
        let lower = ShadowProjector.resolve(
            stack: theme.shadows.kpiCard,
            context: ShadowProjectionContext(frame: lowerFrame, stageSize: stageSize, lightSource: theme.shadows.lightSource),
            scale: 1
        )

        let upperDeepest = try #require(upper.last)
        let lowerDeepest = try #require(lower.last)
        #expect(lowerDeepest.offset.height > upperDeepest.offset.height)
    }

    @Test
    func cardsCloserToLightPullShadowBackHorizontally() throws {
        let theme = FinancialDashboardThemeV1()
        let stageSize = theme.layoutMetrics.referenceCanvasSize
        let leftFrame = CGRect(x: 186, y: 229, width: 300, height: 212)
        let rightFrame = CGRect(x: 1290, y: 229, width: 300, height: 212)

        let left = ShadowProjector.resolve(
            stack: theme.shadows.kpiCard,
            context: ShadowProjectionContext(frame: leftFrame, stageSize: stageSize, lightSource: theme.shadows.lightSource),
            scale: 1
        )
        let right = ShadowProjector.resolve(
            stack: theme.shadows.kpiCard,
            context: ShadowProjectionContext(frame: rightFrame, stageSize: stageSize, lightSource: theme.shadows.lightSource),
            scale: 1
        )

        let leftNearest = try #require(left.first)
        let rightNearest = try #require(right.first)
        #expect(abs(leftNearest.offset.width) > abs(rightNearest.offset.width))
    }

    @Test
    func baselineProjectionWithoutContextPreservesThemeOffsets() throws {
        let stack = ShadowStack([
            ShadowLayer(radius: 24, x: -25, y: 16, opacity: 0.28)
        ])

        let resolved = ShadowProjector.resolve(stack: stack, context: nil, scale: 1)
        let layer = try #require(resolved.first)
        #expect(layer.offset.width == -25)
        #expect(layer.offset.height == 16)
        #expect(layer.blurRadius == 24)
    }
}
