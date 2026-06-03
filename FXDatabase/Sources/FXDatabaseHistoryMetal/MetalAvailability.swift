import Foundation

#if os(macOS)
import Darwin
#endif

#if canImport(Metal)
import Metal
#endif

public struct MetalAvailability: Sendable {
    public let isAvailable: Bool
    public let deviceName: String?
    public let supportsUnifiedMemory: Bool
    public let hardwareSummary: String
    public let optimizedForAppleSiliconM2M3: Bool

    public init() {
        let hardware = FXDatabaseAppleSiliconHardware.probe()
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            self.isAvailable = true
            self.deviceName = device.name
            self.supportsUnifiedMemory = device.hasUnifiedMemory
            self.hardwareSummary = hardware.summary
            self.optimizedForAppleSiliconM2M3 = device.hasUnifiedMemory && hardware.isM2M3OrNewer
        } else {
            self.isAvailable = false
            self.deviceName = nil
            self.supportsUnifiedMemory = false
            self.hardwareSummary = hardware.summary
            self.optimizedForAppleSiliconM2M3 = false
        }
        #else
        self.isAvailable = false
        self.deviceName = nil
        self.supportsUnifiedMemory = false
        self.hardwareSummary = hardware.summary
        self.optimizedForAppleSiliconM2M3 = false
        #endif
    }
}

private struct FXDatabaseAppleSiliconHardware {
    let architecture: String
    let cpuBrand: String

    var isAppleSilicon: Bool {
        architecture == "arm64" && cpuBrand.contains("Apple")
    }

    var isM2M3OrNewer: Bool {
        guard isAppleSilicon else { return false }
        return ["Apple M2", "Apple M3", "Apple M4", "Apple M5", "Apple M6", "Apple M7", "Apple M8", "Apple M9"].contains {
            cpuBrand.contains($0)
        }
    }

    var summary: String {
        "\(architecture) \(cpuBrand)"
    }

    static func probe() -> FXDatabaseAppleSiliconHardware {
        FXDatabaseAppleSiliconHardware(architecture: currentArchitecture(), cpuBrand: currentCPUBrand())
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
