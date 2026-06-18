import Foundation
import os.log

/// Centralized logger for FXGUI file I/O operations.
///
/// Artifact readers use ``loggedTry(_:_:file:line:)`` to replace silent `try?`
/// with a version that logs failures through `os_log`, making file corruption
/// and I/O errors observable without changing return-type contracts.
enum FileIOLogger {
    static let shared = Logger(subsystem: "com.fxai.fxgui", category: "file-io")
}

/// Executes a throwing expression and returns its value, or logs the error and returns `nil`.
///
/// Drop-in replacement for `try?` that records failures via `os_log`.
///
/// ```swift
/// // Before:
/// guard let data = try? Data(contentsOf: url) else { return nil }
///
/// // After:
/// guard let data = loggedTry({ try Data(contentsOf: url) }, "read") else { return nil }
/// ```
func loggedTry<T>(
    _ body: () throws -> T,
    _ context: String = "",
    file: String = #fileID,
    line: Int = #line
) -> T? {
    do {
        return try body()
    } catch {
        if context.isEmpty {
            FileIOLogger.shared.warning("IO failure at \(file, privacy: .public):\(line): \(error, privacy: .public)")
        } else {
            FileIOLogger.shared.warning("IO failure [\(context, privacy: .public)] at \(file, privacy: .public):\(line): \(error, privacy: .public)")
        }
        return nil
    }
}
