import FXDataEngine
import Foundation

public enum FXAIPluginBackendDiscovery {
    public static var pluginRootURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    public static var moduleBackendURL: URL {
        pluginRootURL
            .appendingPathComponent("API")
            .appendingPathComponent("Backends")
            .appendingPathComponent("Python")
            .appendingPathComponent("fxai_plugin_module_backend.py")
    }

    public static func pluginBackendURL(pluginName: String, backend: FXPluginAccelerationBackend) -> URL? {
        let folder: String
        let suffix: String
        switch backend {
        case .pyTorchMPS:
            folder = "PyTorch"
            suffix = "_torch.py"
        case .tensorFlowMetal:
            folder = "TensorFlow"
            suffix = "_tensorflow.py"
        case .foundationNLP:
            folder = "NLP"
            suffix = "_nlp.py"
        case .onnxRuntime:
            folder = "ONNX"
            suffix = ".onnx"
        case .swiftScalar, .swiftSIMD, .accelerate, .metal, .coreMLNeuralEngine, .remoteRPC:
            return nil
        }
        return pluginRootURL
            .appendingPathComponent(pluginName)
            .appendingPathComponent(folder)
            .appendingPathComponent("\(pluginName)\(suffix)")
    }

    public static func externalPythonDescriptor(
        pluginName: String,
        backend: FXPluginAccelerationBackend,
        executable: String = FXAIPluginPythonRuntime.defaultExecutable()
    ) -> MLBackendDescriptor? {
        let framework: MLFramework
        let supportsTraining: Bool
        switch backend {
        case .pyTorchMPS:
            framework = .pyTorch
            supportsTraining = true
        case .tensorFlowMetal:
            framework = .tensorFlow
            supportsTraining = true
        case .foundationNLP:
            framework = .foundationNLP
            supportsTraining = false
        case .onnxRuntime:
            framework = .onnxRuntime
            supportsTraining = false
        case .swiftScalar, .swiftSIMD, .accelerate, .metal, .coreMLNeuralEngine, .remoteRPC:
            return nil
        }
        return MLBackendDescriptor(
            mode: .externalPython(
                framework: framework,
                executable: executable,
                module: moduleBackendURL.path
            ),
            modelIdentifier: pluginName,
            supportsTraining: supportsTraining,
            supportsInference: true,
            usesVolumeFeatures: true
        )
    }

    public static func externalPythonDescriptors(plan: FXPluginAccelerationPlan) -> [MLBackendDescriptor] {
        plan.declaredBackends.compactMap {
            externalPythonDescriptor(pluginName: plan.pluginName, backend: $0)
        }
    }
}
