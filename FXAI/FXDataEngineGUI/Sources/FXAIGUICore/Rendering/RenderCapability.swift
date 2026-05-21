import Foundation

public struct RenderCapability: OptionSet, Sendable {
    public let rawValue: Int

    public static let shadowStacks = RenderCapability(rawValue: 1 << 0)
    public static let bloomGlow = RenderCapability(rawValue: 1 << 1)
    public static let glassComposite = RenderCapability(rawValue: 1 << 2)
    public static let vectorOverlay = RenderCapability(rawValue: 1 << 3)
    public static let chartPrecision = RenderCapability(rawValue: 1 << 4)
    public static let gaugePrecision = RenderCapability(rawValue: 1 << 5)
    public static let realtimeResizing = RenderCapability(rawValue: 1 << 6)
    public static let metalCompositing = RenderCapability(rawValue: 1 << 7)

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
}
