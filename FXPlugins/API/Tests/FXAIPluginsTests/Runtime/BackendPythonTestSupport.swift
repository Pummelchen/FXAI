import Foundation

enum BackendPythonTestSupport {
    private static let tensorFlowMetalPythonMajor = 3
    private static let tensorFlowMetalPythonMinor = 12

    static func requireAnyPython() throws -> String {
        var attempts: [PythonAttempt] = []
        for candidate in generalPythonCandidates() {
            let result = runPython(candidate, script: python312ProbeScript)
            attempts.append(PythonAttempt(executable: candidate, result: result))
            if result.succeeded {
                return candidate
            }
        }
        throw BackendPythonTestSupportError(
            "No usable Python executable was found for FXPlugin backend tests.\n\(formatAttempts(attempts))"
        )
    }

    static func requirePythonImporting(_ module: String) throws -> String {
        var attempts: [PythonAttempt] = []
        for candidate in generalPythonCandidates() {
            let result = runPython(candidate, script: "\(python312ProbeScript)\nimport \(module)")
            attempts.append(PythonAttempt(executable: candidate, result: result))
            if result.succeeded {
                return candidate
            }
        }
        throw BackendPythonTestSupportError(
            "No Python executable could import \(module) for FXPlugin backend tests.\n\(formatAttempts(attempts))"
        )
    }

    static func requireTensorFlowMetalPython() throws -> String {
        var attempts: [PythonAttempt] = []
        for candidate in tensorFlowMetalPythonCandidates() {
            let result = runPython(candidate, script: tensorFlowMetalProbeScript)
            attempts.append(PythonAttempt(executable: candidate, result: result))
            if result.succeeded {
                return candidate
            }
        }
        throw BackendPythonTestSupportError(
            """
            No Python \(tensorFlowMetalPythonMajor).\(tensorFlowMetalPythonMinor) executable with TensorFlow Metal GPU support was found.
            FXPlugin TensorFlow backend tests require tensorflow, tensorflow-metal, and at least one TensorFlow GPU device.
            \(formatAttempts(attempts))
            """
        )
    }

    private static func tensorFlowMetalPythonCandidates() -> [String] {
        uniqueCandidates([
            ProcessInfo.processInfo.environment["FXAI_PYTHON"],
            "/opt/homebrew/opt/python@3.12/libexec/bin/python3",
            "/opt/homebrew/opt/python@3.12/bin/python3.12",
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "python3.12"
        ])
    }

    private static func generalPythonCandidates() -> [String] {
        uniqueCandidates([
            ProcessInfo.processInfo.environment["FXAI_PYTHON"],
            "/opt/homebrew/opt/python@3.12/libexec/bin/python3",
            "/opt/homebrew/opt/python@3.12/bin/python3.12",
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "python3.12"
        ])
    }

    private static func uniqueCandidates(_ candidates: [String?]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for candidate in candidates {
            guard let trimmed = candidate?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !trimmed.isEmpty,
                  seen.insert(trimmed).inserted
            else {
                continue
            }
            result.append(trimmed)
        }
        return result
    }

    private static var python312ProbeScript: String {
        """
        import sys
        print(f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        if sys.version_info.major != \(tensorFlowMetalPythonMajor) or sys.version_info.minor != \(tensorFlowMetalPythonMinor):
            raise SystemExit("expected Python \(tensorFlowMetalPythonMajor).\(tensorFlowMetalPythonMinor) for FXAI backend tests")
        """
    }

    private static var tensorFlowMetalProbeScript: String {
        """
        import importlib.metadata
        import sys

        print(f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        if sys.version_info.major != \(tensorFlowMetalPythonMajor) or sys.version_info.minor != \(tensorFlowMetalPythonMinor):
            raise SystemExit("expected Python \(tensorFlowMetalPythonMajor).\(tensorFlowMetalPythonMinor) for tensorflow-metal")

        import tensorflow as tf

        print(f"tensorflow={tf.__version__}")
        print("tensorflow-metal=" + importlib.metadata.version("tensorflow-metal"))
        gpu_devices = tf.config.list_physical_devices("GPU")
        print("tensorflow_gpu_count=" + str(len(gpu_devices)))
        if not gpu_devices:
            raise SystemExit("tensorflow imported, but no GPU device was reported")
        """
    }

    private static func runPython(_ executable: String, script: String) -> PythonRunResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable, "-c", script]
        var environment = ProcessInfo.processInfo.environment
        environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
        process.environment = environment

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        let semaphore = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in
            semaphore.signal()
        }

        do {
            try process.run()
        } catch {
            return PythonRunResult(status: nil, stdout: "", stderr: "failed to launch: \(error)")
        }

        if semaphore.wait(timeout: .now() + .seconds(45)) == .timedOut {
            process.terminate()
            process.waitUntilExit()
            return PythonRunResult(status: process.terminationStatus, stdout: "", stderr: "probe timed out")
        }

        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        return PythonRunResult(
            status: process.terminationStatus,
            stdout: String(data: stdoutData, encoding: .utf8) ?? "",
            stderr: String(data: stderrData, encoding: .utf8) ?? ""
        )
    }

    private static func formatAttempts(_ attempts: [PythonAttempt]) -> String {
        attempts.map { attempt in
            let status = attempt.result.status.map(String.init) ?? "launch-failed"
            let details = [attempt.result.stdout, attempt.result.stderr]
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            if details.isEmpty {
                return "- \(attempt.executable): status \(status)"
            }
            return "- \(attempt.executable): status \(status)\n\(details)"
        }
        .joined(separator: "\n")
    }
}

private struct PythonRunResult {
    var status: Int32?
    var stdout: String
    var stderr: String

    var succeeded: Bool {
        status == 0
    }
}

private struct PythonAttempt {
    var executable: String
    var result: PythonRunResult
}

private struct BackendPythonTestSupportError: Error, CustomStringConvertible {
    var description: String

    init(_ description: String) {
        self.description = description
    }
}
