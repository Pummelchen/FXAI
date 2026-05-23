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
}
