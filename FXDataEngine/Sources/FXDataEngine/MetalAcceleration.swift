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

public struct MetalKernelCompilationResult: Codable, Hashable, Sendable {
    public let deviceName: String
    public let functionName: String
    public let sourceByteCount: Int

    public init(deviceName: String, functionName: String, sourceByteCount: Int) {
        self.deviceName = deviceName
        self.functionName = functionName
        self.sourceByteCount = sourceByteCount
    }
}

public enum MetalKernelCompiler {
    public static func compile(
        source: String,
        functionName: String,
        sourceLabel: String = "plugin-kernel"
    ) throws -> MetalKernelCompilationResult {
        guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).source")
        }
        guard !functionName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).function")
        }

        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FXDataEngineError.externalBackend("Metal device is not available")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        guard library.makeFunction(name: functionName) != nil else {
            throw FXDataEngineError.externalBackend("Metal function \(functionName) is missing in \(sourceLabel)")
        }
        return MetalKernelCompilationResult(
            deviceName: device.name,
            functionName: functionName,
            sourceByteCount: source.utf8.count
        )
        #else
        throw FXDataEngineError.externalBackend("Metal framework is not available")
        #endif
    }
}
