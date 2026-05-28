import Foundation
import FXBacktestAPI

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct CommandResult: Codable, Sendable {
    var componentId: String
    var componentType: String
    var command: [String]
    var status: String
    var durationSeconds: Double
    var outputHash: String
    var detail: String

    var evidenceDTO: FXAICertificationComponentResultDTO {
        FXAICertificationComponentResultDTO(
            componentId: componentId,
            componentType: componentType,
            status: status,
            durationSeconds: durationSeconds,
            evidenceHash: outputHash,
            detail: detail
        )
    }
}

struct CertificationRunner {
    let root: URL
    let runTests: Bool

    let packagePaths = [
        "FXImporter",
        "FXDatabase",
        "FXDataEngine",
        "FXPlugins",
        "FXBacktest",
        "FXGUI",
        "FXBacktestAgent",
        "FXDemoAgent",
        "FXLiveAgent",
        "FXExecutionContracts"
    ]

    func run() throws -> FXAICertificationEvidenceRequest {
        let started = Int64(Date().timeIntervalSince1970)
        var components: [CommandResult] = []

        components.append(capture("git.commit", type: "environment", command: ["git", "rev-parse", "HEAD"]))
        components.append(capture("git.status", type: "environment", command: ["git", "status", "--porcelain"]))
        components.append(capture("swift.version", type: "environment", command: ["swift", "--version"]))
        components.append(capture("xcode.version", type: "environment", command: ["xcodebuild", "-version"]))
        components.append(capture("macos.version", type: "environment", command: ["sw_vers"]))
        components.append(capture("hardware", type: "environment", command: ["uname", "-m"]))
        components.append(capture("python.version", type: "environment", command: ["python3", "--version"]))
        components.append(capture(
            "pytorch.status",
            type: "environment",
            command: [
                "python3", "-c",
                """
                import torch
                mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
                print(torch.__version__)
                print('mps=' + str(mps))
                raise SystemExit(0 if mps else 1)
                """
            ]
        ))
        components.append(capture(
            "tensorflow.status",
            type: "environment",
            command: [
                "python3", "-c",
                """
                import tensorflow as tf
                gpu = tf.config.list_physical_devices('GPU')
                print(tf.__version__)
                print('gpu=' + str(gpu))
                raise SystemExit(0 if gpu else 1)
                """
            ]
        ))

        for packagePath in packagePaths {
            components.append(capture("\(packagePath).build", type: "swift_build", command: ["swift", "build", "--package-path", packagePath]))
            if runTests {
                components.append(capture("\(packagePath).test", type: "swift_test", command: ["swift", "test", "--package-path", packagePath]))
            }
        }

        components.append(capture(
            "api.clickhouse_boundary",
            type: "boundary_scan",
            command: [
                "bash", "-lc",
                "if rg -n 'import ClickHouse|ClickHouseHTTPClient|clickhouse://|:8123|8123/' FXBacktest FXDataEngine FXPlugins FXImporter FXGUI FXBacktestAgent FXDemoAgent FXLiveAgent FXExecutionContracts --glob '*.swift' --glob '!**/.build/**' --glob '!FXPlugins/API/Registry/FXAIPluginCertificationRegistry.swift' --glob '!FXBacktest/Tests/FXBacktestCoreTests/FXDatabaseGatekeeperTests.swift'; then exit 1; else code=$?; if [ \"$code\" -eq 1 ]; then exit 0; else exit \"$code\"; fi; fi"
            ]
        ))

        let completed = Int64(Date().timeIntervalSince1970)
        let gitCommit = trimmedOutput(for: "git.commit", in: components, fallback: "unknown")
        let gitStatus = trimmedOutput(for: "git.status", in: components, fallback: "")
        let swiftVersion = trimmedOutput(for: "swift.version", in: components, fallback: "unknown")
        let xcodeVersion = trimmedOutput(for: "xcode.version", in: components, fallback: "unknown")
        let macOSVersion = trimmedOutput(for: "macos.version", in: components, fallback: "unknown")
        let hardware = trimmedOutput(for: "hardware", in: components, fallback: "unknown")
        let pythonVersion = trimmedOutput(for: "python.version", in: components, fallback: "unknown")
        let pyTorchStatus = trimmedOutput(for: "pytorch.status", in: components, fallback: "unavailable")
        let tensorflowStatus = trimmedOutput(for: "tensorflow.status", in: components, fallback: "unavailable")
        let optionalInBuildOnly = Set(["pytorch.status", "tensorflow.status"])
        let requiredComponents = runTests ? components : components.filter { !optionalInBuildOnly.contains($0.componentId) }
        let overall = requiredComponents.allSatisfy { $0.status == "passed" } ? "passed" : "failed"
        let evidenceHash = StableHash.hash(components.map { "\($0.componentId)=\($0.status)=\($0.outputHash)" })

        return FXAICertificationEvidenceRequest(
            certificationRunId: "fxai-cert-\(started)-\(evidenceHash.prefix(12))",
            gitCommit: gitCommit,
            workingTreeClean: gitStatus.isEmpty,
            hostHardwareClass: hardware,
            macOSVersion: macOSVersion,
            xcodeVersion: xcodeVersion,
            swiftVersion: swiftVersion,
            metalDeviceName: "see-swift-metal-runtime-gates",
            pythonVersion: pythonVersion,
            pyTorchStatus: pyTorchStatus,
            tensorflowStatus: tensorflowStatus,
            startedAtUTC: started,
            completedAtUTC: max(completed, started),
            overallStatus: overall,
            evidenceHash: evidenceHash,
            componentResults: components.map(\.evidenceDTO)
        )
    }

    private func capture(_ id: String, type: String, command: [String]) -> CommandResult {
        let started = Date()
        let outcome = ProcessRunner.run(command, in: root)
        let elapsed = Date().timeIntervalSince(started)
        let output = outcome.output.trimmingCharacters(in: .whitespacesAndNewlines)
        let detail = output.isEmpty ? "exit=\(outcome.exitCode)" : String(output.prefix(2_000))
        return CommandResult(
            componentId: id,
            componentType: type,
            command: command,
            status: outcome.exitCode == 0 ? "passed" : "failed",
            durationSeconds: elapsed,
            outputHash: StableHash.hash([command.joined(separator: " "), outcome.output]),
            detail: detail
        )
    }

    private func trimmedOutput(for id: String, in components: [CommandResult], fallback: String) -> String {
        components.first { $0.componentId == id }?.detail
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty ?? fallback
    }
}

enum ProcessRunner {
    static func run(_ command: [String], in root: URL) -> (exitCode: Int32, output: String) {
        guard let executable = command.first else { return (127, "empty command") }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable] + command.dropFirst()
        process.currentDirectoryURL = root

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let outputBuffer = ProcessOutputBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            outputBuffer.append(data)
        }
        do {
            try process.run()
            process.waitUntilExit()
            pipe.fileHandleForReading.readabilityHandler = nil
            let remainingData = pipe.fileHandleForReading.readDataToEndOfFile()
            let data = outputBuffer.snapshot(appending: remainingData)
            return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            return (127, "\(error)")
        }
    }
}

final class ProcessOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) {
        lock.lock()
        data.append(chunk)
        lock.unlock()
    }

    func snapshot(appending chunk: Data) -> Data {
        lock.lock()
        data.append(chunk)
        let snapshot = data
        lock.unlock()
        return snapshot
    }
}

enum StableHash {
    static func hash(_ parts: [String]) -> String {
        let input = parts.joined(separator: "\u{1F}")
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(format: "%016llx", hash)
    }
}

extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

func postEvidenceIfConfigured(_ evidence: FXAICertificationEvidenceRequest) {
    let environment = ProcessInfo.processInfo.environment
    let rawURL = environment["FXDATABASE_CERTIFICATION_EVIDENCE_URL"]
        ?? environment["FXDATABASE_API_URL"].map {
            $0.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                + FXBacktestAPIV1.certificationEvidencePath
        }
    guard let rawURL, let url = URL(string: rawURL) else {
        return
    }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try? JSONEncoder().encode(evidence)

    let semaphore = DispatchSemaphore(value: 0)
    URLSession.shared.dataTask(with: request) { _, _, _ in
        semaphore.signal()
    }.resume()
    _ = semaphore.wait(timeout: .now() + 30)
}

let args = Array(CommandLine.arguments.dropFirst())
guard args.first == "certify" else {
    FileHandle.standardError.write(Data("usage: fxai certify [--all|--build-only]\n".utf8))
    Foundation.exit(2)
}

let runTests = args.contains("--all")
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let evidence = try CertificationRunner(root: root, runTests: runTests).run()
try evidence.validate()
postEvidenceIfConfigured(evidence)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(evidence)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
Foundation.exit(evidence.overallStatus == "passed" ? 0 : 1)
