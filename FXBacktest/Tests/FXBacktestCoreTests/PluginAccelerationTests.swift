import FXBacktestCore
import FXBacktestPlugins
import XCTest

final class PluginAccelerationTests: XCTestCase {
    func testFX7DeclaresValidAccelerationDescriptor() throws {
        let plugin = FX7()
        let descriptor = plugin.accelerationDescriptor

        XCTAssertTrue(descriptor.supportedBackends.contains(.metal))
        XCTAssertEqual(descriptor.apiVersion, PluginAccelerationAPIV1.latestVersion)
        XCTAssertEqual(descriptor.metalEntryPoint, "fx7_core_v1")
        XCTAssertEqual(descriptor.ir?.version, PluginAccelerationAPIV1.latestIRVersion)
        XCTAssertNoThrow(try PluginAccelerationPipeline().validate(descriptor))
    }

    func testAccelerationDescriptorRejectsStaleAPIVersions() {
        let pipeline = PluginAccelerationPipeline()
        let staleAPI = PluginAccelerationDescriptor(
            pluginIdentifier: "test",
            apiVersion: "fxbacktest.plugin-acceleration.v0"
        )
        let staleIR = PluginAccelerationDescriptor(
            pluginIdentifier: "test",
            ir: PluginAccelerationIR(
                version: "fxbacktest.plugin-ir.v0",
                requiredColumns: [PluginAccelerationInputColumn(field: "close")],
                operations: [PluginAccelerationIROperation(opcode: "copy")]
            )
        )

        XCTAssertThrowsError(try pipeline.validate(staleAPI)) { error in
            XCTAssertTrue(String(describing: error).contains(PluginAccelerationAPIV1.latestVersion))
        }
        XCTAssertThrowsError(try pipeline.validate(staleIR)) { error in
            XCTAssertTrue(String(describing: error).contains(PluginAccelerationAPIV1.latestIRVersion))
        }
    }
}
