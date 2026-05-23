import XCTest
@testable import FXDataEngine

final class MetalKernelCompilerTests: XCTestCase {
    func testCompilesSimpleKernelWhenMetalIsAvailable() throws {
        #if canImport(Metal)
        guard MetalAccelerationDevice.probe().available else {
            throw XCTSkip("Metal device is not available on this runner")
        }
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void fxai_copy_kernel(
            device const float *input [[buffer(0)]],
            device float *output [[buffer(1)]],
            uint index [[thread_position_in_grid]]
        ) {
            output[index] = input[index];
        }
        """

        let result = try MetalKernelCompiler.compile(
            source: source,
            functionName: "fxai_copy_kernel",
            sourceLabel: "test"
        )

        XCTAssertEqual(result.functionName, "fxai_copy_kernel")
        XCTAssertGreaterThan(result.sourceByteCount, 0)
        XCTAssertFalse(result.deviceName.isEmpty)
        #else
        throw XCTSkip("Metal framework is not available")
        #endif
    }

    func testExecutesUnaryFloatKernelAndMatchesCPUReferenceWhenMetalIsAvailable() throws {
        #if canImport(Metal)
        guard MetalAccelerationDevice.probe().available else {
            throw XCTSkip("Metal device is not available on this runner")
        }
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void fxai_affine_kernel(
            device const float *input [[buffer(0)]],
            device float *output [[buffer(1)]],
            uint index [[thread_position_in_grid]]
        ) {
            output[index] = (input[index] * 1.5f) - 0.25f;
        }
        """
        let input: [Float] = [-2.0, -0.5, 0.0, 1.0, 3.5]
        let result = try MetalKernelCompiler.executeUnaryFloatKernel(
            source: source,
            functionName: "fxai_affine_kernel",
            input: input,
            sourceLabel: "affine-test"
        )
        let expected = input.map { ($0 * 1.5) - 0.25 }

        XCTAssertEqual(result.functionName, "fxai_affine_kernel")
        XCTAssertEqual(result.threadCount, input.count)
        XCTAssertEqual(result.output.count, expected.count)
        for (actual, reference) in zip(result.output, expected) {
            XCTAssertEqual(actual, reference, accuracy: 0.000_001)
        }
        #else
        throw XCTSkip("Metal framework is not available")
        #endif
    }
}
