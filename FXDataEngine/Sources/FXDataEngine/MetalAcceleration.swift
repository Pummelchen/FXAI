import Foundation

#if canImport(Metal)
import Metal
#endif

public struct MetalAccelerationDevice: Codable, Hashable, Sendable {
    public let available: Bool
    public let deviceName: String?
    public let supportsUnifiedMemory: Bool

    public init(available: Bool, deviceName: String?, supportsUnifiedMemory: Bool) {
        self.available = available
        self.deviceName = deviceName
        self.supportsUnifiedMemory = supportsUnifiedMemory
    }

    public static func probe() -> MetalAccelerationDevice {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false)
        }
        return MetalAccelerationDevice(
            available: true,
            deviceName: device.name,
            supportsUnifiedMemory: device.hasUnifiedMemory
        )
        #else
        return MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false)
        #endif
    }
}

public struct MetalFeatureKernelDescriptor: Codable, Hashable, Sendable {
    public let name: String
    public let featureCount: Int
    public let weightCount: Int
    public let usesOHLCV: Bool

    public init(
        name: String = "FXDataEngineFeatureKernel",
        featureCount: Int = FXDataEngineConstants.aiFeatures,
        weightCount: Int = FXDataEngineConstants.aiWeights,
        usesOHLCV: Bool = true
    ) {
        self.name = name
        self.featureCount = featureCount
        self.weightCount = weightCount
        self.usesOHLCV = usesOHLCV
    }
}
