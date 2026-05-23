import Foundation

public struct ThemeMaterials {
    public let glassOpacity: Double
    public let glassStrokeOpacity: Double
    public let glassBlurRadius: CGFloat
    public let footerOpacity: Double
    public let dividerOpacity: Double
    public let metalIntensity: Double
    public let reducedEffectsMetalIntensity: Double

    public init(
        glassOpacity: Double,
        glassStrokeOpacity: Double,
        glassBlurRadius: CGFloat,
        footerOpacity: Double,
        dividerOpacity: Double,
        metalIntensity: Double,
        reducedEffectsMetalIntensity: Double
    ) {
        self.glassOpacity = glassOpacity
        self.glassStrokeOpacity = glassStrokeOpacity
        self.glassBlurRadius = glassBlurRadius
        self.footerOpacity = footerOpacity
        self.dividerOpacity = dividerOpacity
        self.metalIntensity = metalIntensity
        self.reducedEffectsMetalIntensity = reducedEffectsMetalIntensity
    }
}
