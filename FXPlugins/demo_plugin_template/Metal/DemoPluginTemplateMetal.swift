import FXDataEngine
import Foundation

public enum DemoPluginTemplateMetal {
    public static let descriptor = FXPluginAccelerationPlan(
        pluginName: "replace_with_plugin_name",
        primaryBackends: [.metal],
        candidateBackends: [.swiftScalar],
        usesVolumeWhenAvailable: true,
        notes: "Template Metal descriptor. Replace the no-op kernel with plugin-specific scoring or training kernels."
    )

    public static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct DemoPluginTemplateConfig {
        uint feature_count;
        float confidence_floor;
        uint use_volume_when_available;
    };

    kernel void demo_plugin_template_score(
        device const float *features [[buffer(0)]],
        constant DemoPluginTemplateConfig &config [[buffer(1)]],
        device float *class_probabilities [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint offset = id * 3u;
        class_probabilities[offset + 0u] = 0.0f;
        class_probabilities[offset + 1u] = 0.0f;
        class_probabilities[offset + 2u] = 1.0f;
    }
    """
}
