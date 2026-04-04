import Foundation

public enum SVGAssetCatalog {
    public static func referenceSVGURL() -> URL? {
        Bundle.module.url(forResource: "GUI", withExtension: "svg")
    }

    public static func referencePNGURL() -> URL? {
        Bundle.module.url(forResource: "GUI-reference", withExtension: "png")
    }

    public static func referenceSVGData() -> Data? {
        guard let url = referenceSVGURL() else { return nil }
        return try? Data(contentsOf: url)
    }

    public static func referenceSVGString() -> String? {
        guard let data = referenceSVGData() else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
