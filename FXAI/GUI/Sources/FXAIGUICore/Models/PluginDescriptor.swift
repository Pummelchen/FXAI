import Foundation

public struct PluginDescriptor: Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let family: String
    public let sourcePath: URL
    public let sourceKind: SourceKind

    public enum SourceKind: String, Hashable, Sendable {
        case folder
        case file
    }

    public init(name: String, family: String, sourcePath: URL, sourceKind: SourceKind) {
        self.id = "\(family)::\(name)"
        self.name = name
        self.family = family
        self.sourcePath = sourcePath
        self.sourceKind = sourceKind
    }
}
