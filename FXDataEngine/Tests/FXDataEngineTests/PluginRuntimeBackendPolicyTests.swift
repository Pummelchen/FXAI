import XCTest
@testable import FXDataEngine

final class PluginRuntimeBackendPolicyTests: XCTestCase {
    private let appleM2 = AppleSiliconHardware(architecture: "arm64", cpuBrand: "Apple M2 Pro")

    func testLocalRuntimeEnvironmentDefaultsToPython312() {
        XCTAssertEqual(FXPluginRuntimeEnvironment.defaultPythonExecutable(environment: [:]), "python3.12")
        XCTAssertEqual(
            FXPluginRuntimeEnvironment.defaultPythonExecutable(environment: ["FXAI_PYTHON": " /opt/homebrew/bin/python3.12 "]),
            "/opt/homebrew/bin/python3.12"
        )
        XCTAssertEqual(
            FXPluginRuntimeEnvironment.local(environment: ["FXAI_ENABLE_ONNX_RUNTIME": "1"]).pythonExecutable,
            "python3.12"
        )
    }

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
            pythonExecutable: "python3.12",
            pyTorchMPSAvailable: true
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)

        XCTAssertEqual(resolution.selectedBackend, .pyTorchMPS)
        XCTAssertEqual(resolution.mlFramework, .pyTorch)
        XCTAssertTrue(resolution.requiresExternalPython)
    }

    func testAutomaticResolutionPrefersDeclaredRemoteRPCWhenConfigured() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_remote",
            primaryBackends: [.swiftScalar, .onnxRuntime, .remoteRPC, .pyTorchMPS],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let remoteEndpointKey = "FXAI_REMOTE_INFERENCE_ENDPOINT"
        let environment = FXPluginRuntimeEnvironment(
            metalDevice: MetalAccelerationDevice(
                available: true,
                deviceName: "Test GPU",
                supportsUnifiedMemory: true,
                hardware: appleM2
            ),
            pythonExecutable: "python3.12",
            pyTorchMPSAvailable: true,
            onnxRuntimeAvailable: true,
            remoteInferenceAvailable: true,
            remoteInferenceEndpoint: "https://inference.example.test/fxai/predict",
            remoteInferenceAuthToken: "token",
            remoteInferenceTimeoutSeconds: 4.0
        )

        let resolution = try FXPluginRuntimeResolver.resolve(plan: plan, environment: environment)

        XCTAssertEqual(remoteEndpointKey, "FXAI_REMOTE_INFERENCE_ENDPOINT")
        XCTAssertEqual(resolution.selectedBackend, .remoteRPC)
        XCTAssertEqual(resolution.mlFramework, .remoteRPC)
        XCTAssertFalse(resolution.requiresExternalPython)
        XCTAssertEqual(environment.remoteRPCConfiguration()?.endpoint, "https://inference.example.test/fxai/predict")
        XCTAssertEqual(environment.remoteRPCConfiguration()?.authToken, "token")
        XCTAssertEqual(environment.remoteRPCConfiguration()?.timeoutSeconds ?? 0.0, 4.0, accuracy: 0.0)
    }

    func testONNXRuntimeRequiresPythonAndExplicitEnablement() throws {
        let plan = FXPluginAccelerationPlan(
            pluginName: "test_onnx",
            primaryBackends: [.swiftScalar],
            candidateBackends: [.onnxRuntime],
            usesVolumeWhenAvailable: true,
            notes: "test"
        )
        let enabledEnvironment = FXPluginRuntimeEnvironment(
            pythonExecutable: "python3.12",
            onnxRuntimeAvailable: true
        )
        let missingPythonEnvironment = FXPluginRuntimeEnvironment(
            pythonExecutable: nil,
            onnxRuntimeAvailable: true
        )
        let disabledEnvironment = FXPluginRuntimeEnvironment(
            pythonExecutable: "python3.12",
            onnxRuntimeAvailable: false
        )
        let onnxEnablementKey = "FXAI_ENABLE_ONNX_RUNTIME"

        let enabled = try FXPluginRuntimeResolver.resolve(plan: plan, environment: enabledEnvironment)
        let missingPython = try FXPluginRuntimeResolver.resolve(plan: plan, environment: missingPythonEnvironment)
        let disabled = try FXPluginRuntimeResolver.resolve(plan: plan, environment: disabledEnvironment)

        XCTAssertEqual(onnxEnablementKey, "FXAI_ENABLE_ONNX_RUNTIME")
        XCTAssertEqual(enabled.selectedBackend, .onnxRuntime)
        XCTAssertEqual(enabled.mlFramework, .onnxRuntime)
        XCTAssertTrue(enabled.requiresExternalPython)
        XCTAssertEqual(missingPython.selectedBackend, .swiftScalar)
        XCTAssertEqual(disabled.selectedBackend, .swiftScalar)
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
                pythonExecutable: "python3.12",
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
                    pythonExecutable: "python3.12",
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
