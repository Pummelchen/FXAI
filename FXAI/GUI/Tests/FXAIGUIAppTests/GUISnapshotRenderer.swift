import AppKit
@testable import FXAIGUIApp
import FXAIGUICore
import Foundation
import SwiftUI

@MainActor
struct GUISnapshotStatistics {
    let opaquePixelRatio: Double
    let luminanceVariance: Double
    let bucketCount: Int
}

@MainActor
enum GUISnapshotRenderer {
    static func captureRootView(
        selection: SidebarDestination,
        scenario: GUIValidationScenario,
        outputDirectory: URL?
    ) throws -> GUISnapshotStatistics {
        let themeEnvironment = ThemeEnvironment(
            registry: ThemeRegistry(themes: [FinancialDashboardThemeV1()]),
            initialThemeID: .financialDashboardV1
        )
        let model = FXAIGUIModel.validationFixture(selection: selection)

        let rootView = FXAIRootView()
            .environmentObject(model)
            .environmentObject(themeEnvironment)
            .frame(width: scenario.windowSize.width, height: scenario.windowSize.height)

        let renderer = ImageRenderer(content: rootView)
        renderer.scale = scenario.backingScaleFactor
        renderer.proposedSize = ProposedViewSize(scenario.windowSize)
        renderer.isOpaque = false

        guard let nsImage = renderer.nsImage else {
            throw SnapshotError.captureFailed("Could not render NSImage")
        }
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw SnapshotError.captureFailed("Could not extract CGImage")
        }

        if let outputDirectory {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            let fileURL = outputDirectory.appendingPathComponent("\(selection.rawValue)-\(scenario.id).png")
            try writePNG(cgImage: cgImage, to: fileURL)
        }
        return analyze(cgImage: cgImage)
    }

    private static func analyze(cgImage: CGImage) -> GUISnapshotStatistics {
        guard
            let dataProvider = cgImage.dataProvider,
            let data = dataProvider.data,
            let pointer = CFDataGetBytePtr(data)
        else {
            return GUISnapshotStatistics(opaquePixelRatio: 0, luminanceVariance: 0, bucketCount: 0)
        }

        let bytesPerPixel = max(4, cgImage.bitsPerPixel / 8)
        let bytesPerRow = cgImage.bytesPerRow
        let sampleStep = max(1, min(cgImage.width, cgImage.height) / 180)

        var sampleCount = 0
        var opaqueCount = 0
        var luminances: [Double] = []
        var buckets = Set<Int>()

        for y in stride(from: 0, to: cgImage.height, by: sampleStep) {
            for x in stride(from: 0, to: cgImage.width, by: sampleStep) {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let red = Double(pointer[offset]) / 255.0
                let green = Double(pointer[offset + 1]) / 255.0
                let blue = Double(pointer[offset + 2]) / 255.0
                let alpha = Double(pointer[offset + 3]) / 255.0
                let luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)

                sampleCount += 1
                if alpha > 0.85 {
                    opaqueCount += 1
                }
                luminances.append(luminance)
                buckets.insert(Int(red * 12) << 8 | Int(green * 12) << 4 | Int(blue * 12))
            }
        }

        let mean = luminances.reduce(0, +) / Double(max(1, luminances.count))
        let variance = luminances.reduce(0) { partial, value in
            let delta = value - mean
            return partial + delta * delta
        } / Double(max(1, luminances.count))

        return GUISnapshotStatistics(
            opaquePixelRatio: Double(opaqueCount) / Double(max(1, sampleCount)),
            luminanceVariance: variance,
            bucketCount: buckets.count
        )
    }

    private static func writePNG(cgImage: CGImage, to url: URL) throws {
        let rep = NSBitmapImageRep(cgImage: cgImage)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw SnapshotError.captureFailed("Could not encode PNG")
        }
        try data.write(to: url)
    }
}

enum SnapshotError: Error {
    case captureFailed(String)
}
