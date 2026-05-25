import Foundation

public enum FXBacktestAgentCapabilityProbe {
    public static func localCapabilities(
        supportedPluginIds: [String] = [],
        supportedAcceleratorBackends: [String] = ["swiftScalar"],
        localCertificationRunId: String? = nil
    ) -> FXBacktestAgentCapabilities {
        let environment = ProcessInfo.processInfo.environment
        return FXBacktestAgentCapabilities(
            hostName: Host.current().localizedName ?? "unknown",
            cpuCoreCount: ProcessInfo.processInfo.activeProcessorCount,
            memoryBytes: UInt64(ProcessInfo.processInfo.physicalMemory),
            hardwareClass: environment["FXAI_HARDWARE_CLASS"] ?? machineHardwareName(),
            metalDeviceName: environment["FXAI_METAL_DEVICE"] ?? "probe-required",
            pyTorchMPSAvailable: environment["FXAI_ENABLE_PYTORCH_MPS"] == "1",
            tensorFlowMetalAvailable: environment["FXAI_ENABLE_TENSORFLOW_METAL"] == "1",
            foundationNLPAvailable: environment["FXAI_ENABLE_FOUNDATION_NLP"] == "1",
            supportedPluginIds: supportedPluginIds,
            supportedAcceleratorBackends: supportedAcceleratorBackends,
            localCertificationRunId: localCertificationRunId,
            currentLoad: 0.0
        )
    }

    public static func uncertifiedStatus(reason: String = "local certification has not run") -> FXBacktestAgentCertificationStatus {
        FXBacktestAgentCertificationStatus(
            passed: false,
            certificationRunId: nil,
            sineTestPassed: false,
            evidenceHash: stableHash(["uncertified", reason])
        )
    }

    private static func machineHardwareName() -> String {
        #if arch(arm64)
        return "apple-silicon"
        #else
        return "unsupported"
        #endif
    }

    private static func stableHash(_ parts: [String]) -> String {
        let input = parts.joined(separator: "\u{1F}")
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(format: "%016llx", hash)
    }
}
