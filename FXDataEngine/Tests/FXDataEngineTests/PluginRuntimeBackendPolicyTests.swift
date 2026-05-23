import XCTest
@testable import FXDataEngine

final class PluginRuntimeBackendPolicyTests: XCTestCase {
    func testAutomaticResolutionPrefersAvailableNonCPUBackend() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_metal",
            primaryBackends: [.swiftScalar, .metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: true, deviceName: "Test GPU", supportsUnifiedMemory: true),
            pythonExecutable: nil
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)

        XCTAssertEqual(resolution.selectedBackend, .metal)
        XCTAssertEqual(resolution.mlFramework, .metal)
        XCTAssertFalse(resolution.didFallback)
        XCTAssertTrue(resolution.usesVolumeWhenAvailable)
    }

    func testAutomaticResolutionUsesPythonBackendOnlyWhenEnabled() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_pytorch",
            primaryBackends: [.swiftScalar, .pyTorchMPS, .metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: true, deviceName: "Test GPU", supportsUnifiedMemory: true),
            pythonExecutable: "python3",
            pyTorchMPSAvailable: true
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)

        XCTAssertEqual(resolution.selectedBackend, .pyTorchMPS)
        XCTAssertEqual(resolution.mlFramework, .pyTorch)
        XCTAssertTrue(resolution.requiresExternalPython)
    }

    func testForcedUnavailableBackendFallsBackToCPUWhenAllowed() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_fallback",
            primaryBackends: [.swiftScalar],
            candidateBackends: [.metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
            pythonExecutable: nil
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, mode: .metal, environment: environment)

        XCTAssertEqual(resolution.selectedBackend, .swiftScalar)
        XCTAssertEqual(resolution.fallbackBackend, .swiftScalar)
        XCTAssertTrue(resolution.didFallback)
        XCTAssertEqual(resolution.requestedMode, .metal)
    }

    func testStrictForcedUnavailableBackendThrows() {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_strict",
            primaryBackends: [.swiftScalar],
            candidateBackends: [.metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(available: false, deviceName: nil, supportsUnifiedMemory: false),
            pythonExecutable: nil
        )

        XCTAssertThrowsError(
            try FXPluginRuntimeResolver.resolve(
                plan: plan,
                mode: .metal,
                fallbackPolicy: .strict,
                environment: environment
            )
        )
    }

    func testUndeclaredBackendThrowsBeforeAvailabilityCheck() {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_undeclared",
            primaryBackends: [.swiftScalar],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )

        XCTAssertThrowsError(
            try FXPluginRuntimeResolver.resolve(
                plan: plan,
                mode: .pyTorchMPS,
                environment: FXPluginRuntimeEnvironment(pythonExecutable: "python3", pyTorchMPSAvailable: true)
            )
        )
    }

    func testCPUOnlyModeAllowsAccelerateAsCPUFallback() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_cpu",
            primaryBackends: [.accelerate],
            candidateBackends: [.metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, mode: .cpuOnly)

        XCTAssertEqual(resolution.selectedBackend, .accelerate)
        XCTAssertEqual(resolution.mlFramework, .nativeSwift)
    }
}
