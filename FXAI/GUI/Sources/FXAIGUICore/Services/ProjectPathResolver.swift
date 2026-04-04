import Foundation

public enum ProjectPathResolver {
    public static func isProjectRoot(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: url.appendingPathComponent("FXAI.mq5").path)
            && fm.fileExists(atPath: url.appendingPathComponent("Plugins").path)
            && fm.fileExists(atPath: url.appendingPathComponent("Tools").path)
    }

    public static func defaultProjectRoot() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        if let explicitRoot = environment["FXAI_PROJECT_ROOT"], !explicitRoot.isEmpty {
            let url = URL(fileURLWithPath: explicitRoot, isDirectory: true)
            if isProjectRoot(url) {
                return url
            }
        }

        let candidates = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
            Bundle.main.bundleURL
        ]

        for candidate in candidates {
            if let resolved = resolveProjectRoot(from: candidate) {
                return resolved
            }
        }

        return nil
    }

    public static func resolveProjectRoot(from url: URL) -> URL? {
        var current = url.standardizedFileURL
        let fm = FileManager.default

        while current.path != "/" {
            if isProjectRoot(current) {
                return current
            }

            if current.lastPathComponent == "GUI" {
                let parent = current.deletingLastPathComponent()
                if isProjectRoot(parent) {
                    return parent
                }
            }

            let parent = current.deletingLastPathComponent()
            if parent == current || !fm.fileExists(atPath: parent.path) {
                break
            }
            current = parent
        }

        return nil
    }
}
