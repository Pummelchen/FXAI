import FXBacktestCore
import FXBacktestPlugins
import XCTest

final class PluginAccelerationTests: XCTestCase {
    func testFX7DeclaresValidAccelerationDescriptor() throws {
        let plugin = FX7()
        let descriptor = plugin.accelerationDescriptor

        XCTAssertTrue(descriptor.supportedBackends.contains(.metal))
        XCTAssertEqual(descriptor.metalEntryPoint, "fx7_core_v1")
        XCTAssertEqual(descriptor.ir?.version, "fxbacktest.plugin-ir.v1")
        XCTAssertNoThrow(try PluginAccelerationPipeline().validate(descriptor))
    }
}
