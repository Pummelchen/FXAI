import CoreGraphics

public struct GUIValidationScenario: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let windowSize: CGSize
    public let backingScaleFactor: CGFloat
    public let expectedLayoutClass: DashboardLayoutClass
    public let expectsReducedDecorativeGlow: Bool
    public let strictStageContainment: Bool

    public init(
        id: String,
        title: String,
        windowSize: CGSize,
        backingScaleFactor: CGFloat,
        expectedLayoutClass: DashboardLayoutClass,
        expectsReducedDecorativeGlow: Bool,
        strictStageContainment: Bool = true
    ) {
        self.id = id
        self.title = title
        self.windowSize = windowSize
        self.backingScaleFactor = backingScaleFactor
        self.expectedLayoutClass = expectedLayoutClass
        self.expectsReducedDecorativeGlow = expectsReducedDecorativeGlow
        self.strictStageContainment = strictStageContainment
    }

    public static let compactEdge = GUIValidationScenario(
        id: "compact-edge",
        title: "Compact Edge",
        windowSize: CGSize(width: 1180, height: 820),
        backingScaleFactor: 2,
        expectedLayoutClass: .compactDesktop,
        expectsReducedDecorativeGlow: true,
        strictStageContainment: false
    )

    public static let macBook14 = GUIValidationScenario(
        id: "macbook-14",
        title: "MacBook Pro 14\"",
        windowSize: CGSize(width: 1512, height: 982),
        backingScaleFactor: 2,
        expectedLayoutClass: .standardDesktop,
        expectsReducedDecorativeGlow: false
    )

    public static let standardDesktop = GUIValidationScenario(
        id: "standard-desktop",
        title: "Standard Desktop",
        windowSize: CGSize(width: 1728, height: 1117),
        backingScaleFactor: 2,
        expectedLayoutClass: .standardDesktop,
        expectsReducedDecorativeGlow: false
    )

    public static let wideDesktop = GUIValidationScenario(
        id: "wide-desktop",
        title: "Wide Desktop",
        windowSize: CGSize(width: 2200, height: 1280),
        backingScaleFactor: 2,
        expectedLayoutClass: .wideDesktop,
        expectsReducedDecorativeGlow: false
    )

    public static let ultraWide4K = GUIValidationScenario(
        id: "ultrawide-4k",
        title: "Ultra-Wide 4K",
        windowSize: CGSize(width: 3440, height: 1440),
        backingScaleFactor: 1,
        expectedLayoutClass: .ultraWideDesktop,
        expectsReducedDecorativeGlow: false
    )

    public static let ultraWide8K = GUIValidationScenario(
        id: "ultrawide-8k",
        title: "8K-Like",
        windowSize: CGSize(width: 5120, height: 2160),
        backingScaleFactor: 1,
        expectedLayoutClass: .ultraWideDesktop,
        expectsReducedDecorativeGlow: false
    )

    public static let layoutAuditMatrix: [GUIValidationScenario] = [
        .compactEdge,
        .macBook14,
        .standardDesktop,
        .wideDesktop,
        .ultraWide4K,
        .ultraWide8K
    ]

    public static let screenshotMatrix: [GUIValidationScenario] = [
        .compactEdge,
        .macBook14,
        .standardDesktop,
        .wideDesktop,
        .ultraWide4K,
        .ultraWide8K
    ]
}
