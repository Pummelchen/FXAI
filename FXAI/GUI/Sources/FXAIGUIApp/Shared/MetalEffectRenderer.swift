import Metal
import MetalKit
import SwiftUI

struct MetalEffectRenderer: NSViewRepresentable {
    let intensity: Double

    func makeCoordinator() -> Coordinator {
        Coordinator(intensity: intensity)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.framebufferOnly = false
        view.clearColor = MTLClearColorMake(0, 0, 0, 0)
        view.colorPixelFormat = .bgra8Unorm
        view.delegate = context.coordinator
        context.coordinator.configureIfNeeded(view: view)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.intensity = intensity
        context.coordinator.configureIfNeeded(view: nsView)
        nsView.needsDisplay = true
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        private var commandQueue: MTLCommandQueue?
        private var pipeline: MTLRenderPipelineState?
        private var device: MTLDevice?
        var intensity: Double

        init(intensity: Double) {
            self.intensity = intensity
        }

        func configureIfNeeded(view: MTKView) {
            guard pipeline == nil, let device = view.device else {
                return
            }
            self.device = device
            commandQueue = device.makeCommandQueue()

            let source = """
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut {
                float4 position [[position]];
                float2 uv;
            };
            vertex VertexOut vertex_main(uint vertexID [[vertex_id]]) {
                float2 positions[6] = {
                    float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0),
                    float2(-1.0, 1.0), float2(1.0, -1.0), float2(1.0, 1.0)
                };
                float2 uv[6] = {
                    float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0),
                    float2(0.0, 0.0), float2(1.0, 1.0), float2(1.0, 0.0)
                };
                VertexOut out;
                out.position = float4(positions[vertexID], 0.0, 1.0);
                out.uv = uv[vertexID];
                return out;
            }
            fragment float4 fragment_main(VertexOut in [[stage_in]], constant float &intensity [[buffer(0)]]) {
                float2 uv = in.uv;
                float2 center = float2(0.55, 0.42);
                float dist = distance(uv, center);
                float vignette = smoothstep(1.1, 0.15, dist) * 0.12;
                float bloom = smoothstep(0.55, 0.0, dist) * 0.08;
                float noise = fract(sin(dot(uv * 1431.231, float2(12.9898, 78.233))) * 43758.5453) * 0.035;
                float3 color = float3(0.89, 0.93, 0.34) * bloom + float3(0.80, 0.84, 0.88) * vignette;
                return float4(color + noise, (vignette + bloom * 1.6) * intensity);
            }
            """

            do {
                let library = try device.makeLibrary(source: source, options: nil)
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.vertexFunction = library.makeFunction(name: "vertex_main")
                descriptor.fragmentFunction = library.makeFunction(name: "fragment_main")
                descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
                descriptor.colorAttachments[0].isBlendingEnabled = true
                descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
                descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
                descriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
                descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
                pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            } catch {
                pipeline = nil
            }
        }

        nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        nonisolated func draw(in view: MTKView) {
            Task { @MainActor in
                guard
                    let pipeline,
                    let drawable = view.currentDrawable,
                    let descriptor = view.currentRenderPassDescriptor,
                    let commandQueue
                else {
                    return
                }

                let commandBuffer = commandQueue.makeCommandBuffer()
                let encoder = commandBuffer?.makeRenderCommandEncoder(descriptor: descriptor)
                var intensityValue = Float(intensity)
                encoder?.setRenderPipelineState(pipeline)
                encoder?.setFragmentBytes(&intensityValue, length: MemoryLayout<Float>.size, index: 0)
                encoder?.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                encoder?.endEncoding()
                commandBuffer?.present(drawable)
                commandBuffer?.commit()
            }
        }
    }
}
