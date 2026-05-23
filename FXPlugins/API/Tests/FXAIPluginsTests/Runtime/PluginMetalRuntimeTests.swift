import FXDataEngine
import XCTest
@testable import FXAIPlugins

final class PluginMetalRuntimeTests: XCTestCase {
    func testEveryDeclaredMetalPluginHasDiscoverableKernelSourceAndFunctions() throws {
        let metalPlans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.metal) }
        XCTAssertFalse(metalPlans.isEmpty)

        for plan in metalPlans {
            let bundles = try FXAIPluginMetalBackendDiscovery.metalKernelBundles(pluginName: plan.pluginName)
            XCTAssertFalse(bundles.isEmpty, "\(plan.pluginName) declares Metal but has no discoverable kernel bundle")
            for bundle in bundles {
                XCTAssertFalse(bundle.kernelSource.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, plan.pluginName)
                XCTAssertFalse(bundle.functionNames.isEmpty, plan.pluginName)
            }
        }
    }

    func testEveryDeclaredMetalKernelCompilesWhenMetalIsAvailable() throws {
        #if canImport(Metal)
        guard MetalAccelerationDevice.probe().available else {
            throw XCTSkip("Metal device is not available on this runner")
        }
        let results = try FXAIPluginMetalBackendDiscovery.compileDeclaredPluginKernels()
        let declaredFunctionCount = try FXAIPluginMetalBackendDiscovery.metalKernelBundlesForDeclaredPlugins()
            .reduce(0) { $0 + $1.functionNames.count }

        XCTAssertEqual(results.count, declaredFunctionCount)
        XCTAssertGreaterThan(results.count, 0)
        XCTAssertTrue(results.allSatisfy { !$0.deviceName.isEmpty })
        #else
        throw XCTSkip("Metal framework is not available")
        #endif
    }

    func testStrictMetalRuntimeExecutesLiveProbeBeforeCPUPredictionWhenMetalIsAvailable() throws {
        #if canImport(Metal)
        let device = MetalAccelerationDevice.probe()
        guard device.available else {
            throw XCTSkip("Metal device is not available on this runner")
        }
        let plugin = try XCTUnwrap(
            FXAIPluginRegistry.availablePlugins()
                .first { $0.manifest.aiName == "fxbacktest_moving_average_cross" } as? any FXAIPlannedPlugin
        )
        let runtime = FXAIAcceleratedPluginRuntime(
            plugin: plugin,
            configuration: FXAIPluginRuntimeConfiguration(
                mode: .metal,
                fallbackPolicy: .strict,
                environment: FXPluginRuntimeEnvironment(metalDevice: device)
            )
        )

        let prediction = try runtime.predict(Self.predictRequest(), hyperParameters: HyperParameters())

        try prediction.validate()
        XCTAssertGreaterThan(prediction.classProbabilities[LabelClass.buy.rawValue], 0.5)
        #else
        throw XCTSkip("Metal framework is not available")
        #endif
    }

    func testEveryDeclaredMetalPluginExecutesPluginLocalKernelWithCPUParityFixtureWhenMetalIsAvailable() throws {
        #if canImport(Metal)
        guard MetalAccelerationDevice.probe().available else {
            throw XCTSkip("Metal device is not available on this runner")
        }
        let metalPlans = FXAIPluginRegistry.accelerationPlans().filter { $0.declares(.metal) }
        XCTAssertFalse(metalPlans.isEmpty)

        for plan in metalPlans {
            let result = try FXAIPluginMetalBackendDiscovery.executePluginKernelProbe(pluginName: plan.pluginName)
            XCTAssertEqual(result.pluginName, plan.pluginName)
            XCTAssertFalse(result.functionName.isEmpty)
            XCTAssertEqual(result.executionResult.output.count, result.expectedOutput.count)
            for (actual, expected) in zip(result.executionResult.output, result.expectedOutput) {
                XCTAssertEqual(actual, expected, accuracy: 0.0005, plan.pluginName)
            }
        }
        #else
        throw XCTSkip("Metal framework is not available")
        #endif
    }

    private static func predictRequest() -> PredictRequestV4 {
        var x = Array(repeating: 0.0, count: FXDataEngineConstants.aiWeights)
        x[6] = 0.8
        x[7] = 0.35
        x[8] = -0.05
        return PredictRequestV4(
            valid: true,
            context: PluginContextV4(
                horizonMinutes: 15,
                priceCostPoints: 0.5,
                minMovePoints: 1.0,
                dataHasVolume: true
            ),
            x: x
        )
    }
}
