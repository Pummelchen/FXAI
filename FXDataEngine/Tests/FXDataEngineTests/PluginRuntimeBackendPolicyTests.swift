import XCTest
@testable import FXDataEngine

final class PluginRuntimeBackendPolicyTests: XCTestCase {
    private let appleM2 = AppleSiliconHardware(architecture: "arm64", cpuBrand: "Apple M2 Pro")

    func testAutomaticResolutionPrefersAvailableNonCPUBackend() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_metal",
            primaryBackends: [.swiftScalar, .metal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(
                available: true,
                deviceName: "Test GPU",
                supportsUnifiedMemory: true,
                hardware: appleM2
            ),
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
            metalDevice: MetalAccelerationDevice(
                available: true,
                deviceName: "Test GPU",
                supportsUnifiedMemory: true,
                hardware: appleM2
            ),
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

    func testAppleSiliconTargetRejectsM1AndIntelAccelerators() {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_hardware_gate",
            primaryBackends: [.swiftScalar],
            candidateBackends: [.metal, .pyTorchMPS, .tensorFlowMetal],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let unsupportedHosts = [
            AppleSiliconHardware(architecture: "arm64", cpuBrand: "Apple M1 Max"),
            AppleSiliconHardware(architecture: "x86_64", cpuBrand: "Intel Core i9")
        ]

        for host in unsupportedHosts {
            let environment = FXPluginRuntimeEnvironment(
                metalDevice: MetalAccelerationDevice(
                    available: true,
                    deviceName: "Test GPU",
                    supportsUnifiedMemory: true,
                    hardware: host
                ),
                pythonExecutable: "python3",
                pyTorchMPSAvailable: true,
                tensorFlowMetalAvailable: true
            )

            XCTAssertFalse(environment.supports(.metal), host.cpuBrand)
            XCTAssertFalse(environment.supports(.pyTorchMPS), host.cpuBrand)
            XCTAssertFalse(environment.supports(.tensorFlowMetal), host.cpuBrand)
            let resolution = try? FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)
            XCTAssertEqual(resolution?.selectedBackend, .swiftScalar, host.cpuBrand)
        }
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
                environment: FXPluginRuntimeEnvironment(
                    metalDevice: MetalAccelerationDevice(
                        available: true,
                        deviceName: "Test GPU",
                        supportsUnifiedMemory: true,
                        hardware: appleM2
                    ),
                    pythonExecutable: "python3",
                    pyTorchMPSAvailable: true
                )
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
