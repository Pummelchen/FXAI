import Foundation

#if canImport(Metal)
import Metal
#endif

public struct MetalAccelerationDevice: Codable, Hashable, Sendable {
    public let available: Bool
    public let deviceName: String?
    public let supportsUnifiedMemory: Bool
    public let hardware: AppleSiliconHardware
    public let optimizedForFXAIAppleSilicon: Bool

    public init(
        available: Bool,
        deviceName: String?,
        supportsUnifiedMemory: Bool,
        hardware: AppleSiliconHardware = .probe(),
        optimizedForFXAIAppleSilicon: Bool? = nil
    ) {
        self.available = available
        self.deviceName = deviceName
        self.supportsUnifiedMemory = supportsUnifiedMemory
        self.hardware = hardware
        self.optimizedForFXAIAppleSilicon = optimizedForFXAIAppleSilicon ?? (available && supportsUnifiedMemory && hardware.isM2M3OrNewer)
    }

    public static func probe() -> MetalAccelerationDevice {
        let hardware = AppleSiliconHardware.probe()
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false, hardware: hardware)
        }
        return MetalAccelerationDevice(
            available: true,
            deviceName: device.name,
            supportsUnifiedMemory: device.hasUnifiedMemory,
            hardware: hardware
        )
        #else
        return MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false, hardware: hardware)
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

public struct MetalFloatKernelExecutionResult: Codable, Hashable, Sendable {
    public let deviceName: String
    public let functionName: String
    public let threadCount: Int
    public let output: [Float]

    public init(deviceName: String, functionName: String, threadCount: Int, output: [Float]) {
        self.deviceName = deviceName
        self.functionName = functionName
        self.threadCount = threadCount
        self.output = output
    }
}

public enum MetalKernelArgument: Sendable {
    case floatArray([Float])
    case uintArray([UInt32])
    case intArray([Int32])
    case int64Array([Int64])
    case floatScalar(Float)
    case uintScalar(UInt32)
    case boolScalar(Bool)
    case outputFloat(count: Int)
}

public struct MetalKernelThreadGrid: Codable, Hashable, Sendable {
    public let width: Int
    public let height: Int
    public let depth: Int

    public init(width: Int, height: Int = 1, depth: Int = 1) {
        self.width = max(1, width)
        self.height = max(1, height)
        self.depth = max(1, depth)
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

    public static func executeUnaryFloatKernel(
        source: String,
        functionName: String,
        input: [Float],
        outputCount: Int? = nil,
        threadCount: Int? = nil,
        sourceLabel: String = "plugin-kernel"
    ) throws -> MetalFloatKernelExecutionResult {
        guard !input.isEmpty else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).input")
        }
        let resolvedOutputCount = outputCount ?? input.count
        let resolvedThreadCount = threadCount ?? resolvedOutputCount
        guard resolvedOutputCount > 0 else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).outputCount")
        }
        guard resolvedThreadCount > 0 else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).threadCount")
        }

        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FXDataEngineError.externalBackend("Metal device is not available")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            throw FXDataEngineError.externalBackend("Metal function \(functionName) is missing in \(sourceLabel)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let commandQueue = device.makeCommandQueue() else {
            throw FXDataEngineError.externalBackend("Metal command queue is unavailable")
        }

        let inputByteCount = input.count * MemoryLayout<Float>.stride
        let outputByteCount = resolvedOutputCount * MemoryLayout<Float>.stride
        guard let inputBuffer = device.makeBuffer(bytes: input, length: inputByteCount, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outputByteCount, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FXDataEngineError.externalBackend("Metal buffer or command encoder allocation failed")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        let threadsPerThreadgroup = MTLSize(
            width: min(max(pipeline.threadExecutionWidth, 1), resolvedThreadCount),
            height: 1,
            depth: 1
        )
        let threadsPerGrid = MTLSize(width: resolvedThreadCount, height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw FXDataEngineError.externalBackend("Metal command buffer failed: \(error.localizedDescription)")
        }

        let rawPointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        let output = Array(UnsafeBufferPointer(start: rawPointer, count: resolvedOutputCount))
        return MetalFloatKernelExecutionResult(
            deviceName: device.name,
            functionName: functionName,
            threadCount: resolvedThreadCount,
            output: output
        )
        #else
        throw FXDataEngineError.externalBackend("Metal framework is not available")
        #endif
    }

    public static func executeFloatKernel(
        source: String,
        functionName: String,
        arguments: [MetalKernelArgument],
        outputArgumentIndex: Int,
        threadGrid: MetalKernelThreadGrid,
        sourceLabel: String = "plugin-kernel"
    ) throws -> MetalFloatKernelExecutionResult {
        guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).source")
        }
        guard !functionName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).function")
        }
        guard arguments.indices.contains(outputArgumentIndex) else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).outputArgumentIndex")
        }
        guard case .outputFloat(let outputCount) = arguments[outputArgumentIndex], outputCount > 0 else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).outputArgument")
        }

        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FXDataEngineError.externalBackend("Metal device is not available")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: functionName) else {
            throw FXDataEngineError.externalBackend("Metal function \(functionName) is missing in \(sourceLabel)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FXDataEngineError.externalBackend("Metal command queue or encoder allocation failed")
        }

        var retainedBuffers: [MTLBuffer] = []
        var outputBuffer: MTLBuffer?

        func makeBuffer<T>(values: [T], label: String) throws -> MTLBuffer {
            guard !values.isEmpty else {
                throw FXDataEngineError.validation("metal.\(sourceLabel).\(label).empty")
            }
            var copy = values
            guard let buffer = copy.withUnsafeMutableBytes({ rawBuffer -> MTLBuffer? in
                guard let baseAddress = rawBuffer.baseAddress else { return nil }
                return device.makeBuffer(bytes: baseAddress, length: rawBuffer.count, options: .storageModeShared)
            }) else {
                throw FXDataEngineError.externalBackend("Metal input buffer allocation failed for \(label)")
            }
            return buffer
        }

        func makeScalarBuffer<T>(value: T, label: String) throws -> MTLBuffer {
            var scalar = value
            guard let buffer = withUnsafeBytes(of: &scalar, { rawBuffer -> MTLBuffer? in
                guard let baseAddress = rawBuffer.baseAddress else { return nil }
                return device.makeBuffer(bytes: baseAddress, length: rawBuffer.count, options: .storageModeShared)
            }) else {
                throw FXDataEngineError.externalBackend("Metal scalar buffer allocation failed for \(label)")
            }
            return buffer
        }

        for (index, argument) in arguments.enumerated() {
            let buffer: MTLBuffer
            switch argument {
            case .floatArray(let values):
                buffer = try makeBuffer(values: values, label: "arg\(index)")
            case .uintArray(let values):
                buffer = try makeBuffer(values: values, label: "arg\(index)")
            case .intArray(let values):
                buffer = try makeBuffer(values: values, label: "arg\(index)")
            case .int64Array(let values):
                buffer = try makeBuffer(values: values, label: "arg\(index)")
            case .floatScalar(let value):
                buffer = try makeScalarBuffer(value: value, label: "arg\(index)")
            case .uintScalar(let value):
                buffer = try makeScalarBuffer(value: value, label: "arg\(index)")
            case .boolScalar(let value):
                var byte: UInt8 = value ? 1 : 0
                guard let scalarBuffer = withUnsafeBytes(of: &byte, { rawBuffer -> MTLBuffer? in
                    guard let baseAddress = rawBuffer.baseAddress else { return nil }
                    return device.makeBuffer(bytes: baseAddress, length: rawBuffer.count, options: .storageModeShared)
                }) else {
                    throw FXDataEngineError.externalBackend("Metal bool buffer allocation failed for arg\(index)")
                }
                buffer = scalarBuffer
            case .outputFloat(let count):
                guard count > 0 else {
                    throw FXDataEngineError.validation("metal.\(sourceLabel).arg\(index).outputCount")
                }
                let zeros = Array(repeating: Float(0), count: count)
                buffer = try makeBuffer(values: zeros, label: "arg\(index)")
                if index == outputArgumentIndex {
                    outputBuffer = buffer
                }
            }
            retainedBuffers.append(buffer)
            encoder.setBuffer(buffer, offset: 0, index: index)
        }

        guard let resolvedOutputBuffer = outputBuffer else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).outputBuffer")
        }

        encoder.setComputePipelineState(pipeline)
        let threadsPerThreadgroup = MTLSize(
            width: min(max(pipeline.threadExecutionWidth, 1), threadGrid.width),
            height: 1,
            depth: 1
        )
        let threadsPerGrid = MTLSize(width: threadGrid.width, height: threadGrid.height, depth: threadGrid.depth)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw FXDataEngineError.externalBackend("Metal command buffer failed: \(error.localizedDescription)")
        }

        guard case .outputFloat(let outputCount) = arguments[outputArgumentIndex] else {
            throw FXDataEngineError.validation("metal.\(sourceLabel).outputArgument")
        }
        let rawPointer = resolvedOutputBuffer.contents().assumingMemoryBound(to: Float.self)
        let output = Array(UnsafeBufferPointer(start: rawPointer, count: outputCount))
        return MetalFloatKernelExecutionResult(
            deviceName: device.name,
            functionName: functionName,
            threadCount: threadGrid.width * threadGrid.height * threadGrid.depth,
            output: output
        )
        #else
        throw FXDataEngineError.externalBackend("Metal framework is not available")
        #endif
    }
}
