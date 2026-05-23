import Foundation

#if os(macOS)
import Darwin
#endif

public struct AppleSiliconHardware: Codable, Hashable, Sendable {
    public let architecture: String
    public let cpuBrand: String

    public init(architecture: String, cpuBrand: String) {
        self.architecture = architecture
        self.cpuBrand = cpuBrand
    }

    public var isAppleSilicon: Bool {
        architecture == "arm64" && cpuBrand.contains("Apple")
    }

    public var isM2M3OrNewer: Bool {
        guard isAppleSilicon else { return false }
        return ["Apple M2", "Apple M3", "Apple M4", "Apple M5", "Apple M6", "Apple M7", "Apple M8", "Apple M9"].contains {
            cpuBrand.contains($0)
        }
    }

    public var fxaiSupportSummary: String {
        if isM2M3OrNewer {
            return "supported Apple Silicon host (\(cpuBrand))"
        }
        if isAppleSilicon {
            return "unsupported Apple Silicon host for FXAI acceleration target (\(cpuBrand))"
        }
        return "unsupported non-Apple-Silicon host (\(architecture), \(cpuBrand))"
    }

    public static func probe() -> AppleSiliconHardware {
        AppleSiliconHardware(
            architecture: currentArchitecture(),
            cpuBrand: currentCPUBrand()
        )
    }

    private static func currentArchitecture() -> String {
        #if os(macOS)
        var systemInfo = utsname()
        uname(&systemInfo)
        let mirror = Mirror(reflecting: systemInfo.machine)
        let bytes = mirror.children.compactMap { child -> UInt8? in
            guard let value = child.value as? Int8, value != 0 else { return nil }
            return UInt8(value)
        }
        return String(decoding: bytes, as: UTF8.self)
        #else
        return "unknown"
        #endif
    }

    private static func currentCPUBrand() -> String {
        #if os(macOS)
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else {
            return systemProfilerChipName() ?? "unknown"
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else {
            return systemProfilerChipName() ?? "unknown"
        }
        let brandBytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let brand = String(decoding: brandBytes, as: UTF8.self)
        if brand.isEmpty || brand == "Apple processor" {
            return systemProfilerChipName() ?? brand
        }
        return brand
        #else
        return "unknown"
        #endif
    }

    private static func systemProfilerChipName() -> String? {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
        process.arguments = ["SPHardwareDataType"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: data, encoding: .utf8) else { return nil }
        for line in output.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("Chip:") else { continue }
            return String(trimmed.dropFirst("Chip:".count)).trimmingCharacters(in: .whitespaces)
        }
        return nil
        #else
        return nil
        #endif
    }
}
