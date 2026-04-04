import CoreGraphics
import Foundation

public struct ParsedSVGReference: Equatable {
    public let canvasSize: CGSize
    public let viewBox: CGRect
    public let containsMainPanelRect: Bool
    public let containsTooltipRect: Bool
    public let containsAugBar: Bool
}

public enum SVGParserSupport {
    public static func parseReferenceAsset() -> ParsedSVGReference? {
        guard let text = SVGAssetCatalog.referenceSVGString() else { return nil }
        let width = captureDouble(pattern: #"width="([0-9.]+)""#, in: text) ?? Double(SVGMetrics.canvasSize.width)
        let height = captureDouble(pattern: #"height="([0-9.]+)""#, in: text) ?? Double(SVGMetrics.canvasSize.height)
        let viewBox = captureViewBox(in: text) ?? CGRect(origin: .zero, size: SVGMetrics.canvasSize)
        return ParsedSVGReference(
            canvasSize: CGSize(width: width, height: height),
            viewBox: viewBox,
            containsMainPanelRect: text.contains(#"x="80" width="1648" height="1117" rx="20""#),
            containsTooltipRect: text.contains(#"x="1424" y="534" width="92.3265" height="24.2653""#),
            containsAugBar: text.contains(#"x="1448" y="586" width="44" height="238""#)
        )
    }

    public static func validateReferenceAsset() -> [String] {
        guard let parsed = parseReferenceAsset() else {
            return ["Missing GUI.svg reference asset in bundle."]
        }

        var issues: [String] = []
        if parsed.canvasSize != SVGMetrics.canvasSize {
            issues.append("Reference canvas size \(parsed.canvasSize) does not match expected \(SVGMetrics.canvasSize).")
        }
        if !parsed.containsMainPanelRect {
            issues.append("Reference asset does not contain the expected main panel geometry.")
        }
        if !parsed.containsTooltipRect {
            issues.append("Reference asset does not contain the expected tooltip geometry.")
        }
        if !parsed.containsAugBar {
            issues.append("Reference asset does not contain the expected Aug chart bar geometry.")
        }
        return issues
    }

    private static func captureDouble(pattern: String, in text: String) -> Double? {
        guard let match = text.range(of: pattern, options: .regularExpression) else { return nil }
        let matched = String(text[match])
        guard let value = matched.split(separator: "\"").dropFirst().first else { return nil }
        return Double(value)
    }

    private static func captureViewBox(in text: String) -> CGRect? {
        guard let match = text.range(of: #"viewBox="([0-9.\s]+)""#, options: .regularExpression) else {
            return nil
        }
        let matched = String(text[match])
        guard let raw = matched.split(separator: "\"").dropFirst().first else { return nil }
        let values = raw.split(separator: " ").compactMap { Double($0) }
        guard values.count == 4 else { return nil }
        return CGRect(x: values[0], y: values[1], width: values[2], height: values[3])
    }
}
