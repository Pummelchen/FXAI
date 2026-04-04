import Foundation

public enum RenderingTier: String, CaseIterable, Codable, Sendable {
    case swiftUI
    case coreGraphics
    case coreAnimation
    case metal
}
