import AppKit
import FXAIGUICore
import Metal
import MetalKit
import SwiftUI
import simd

private struct HeatmapVertex {
    var position: SIMD2<Float>
    var color: SIMD4<Float>
}

struct MetalHeatmapPanel: View {
    let heatmap: VisualizationHeatmap

    private let cellWidth: CGFloat = 84
    private let cellHeight: CGFloat = 34
    private let rowLabelWidth: CGFloat = 120

    var body: some View {
        FXAIVisualEffectSurface {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(heatmap.title)
                        .font(.headline)
                        .foregroundStyle(FXAITheme.textPrimary)
                    Text(heatmap.subtitle)
                        .font(.caption)
                        .foregroundStyle(FXAITheme.textSecondary)
                }

                ScrollView([.horizontal, .vertical]) {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 0) {
                            Color.clear
                                .frame(width: rowLabelWidth, height: 18)
                            HStack(spacing: 0) {
                                ForEach(Array(heatmap.columnLabels.enumerated()), id: \.offset) { _, label in
                                    Text(label)
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textSecondary)
                                        .frame(width: cellWidth, height: 18)
                                        .lineLimit(2)
                                        .multilineTextAlignment(.center)
                                }
                            }
                        }

                        HStack(alignment: .top, spacing: 0) {
                            VStack(alignment: .leading, spacing: 0) {
                                ForEach(Array(heatmap.rowLabels.enumerated()), id: \.offset) { _, label in
                                    Text(label)
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(FXAITheme.textSecondary)
                                        .frame(width: rowLabelWidth, height: cellHeight, alignment: .leading)
                                }
                            }

                            MetalHeatmapGridView(heatmap: heatmap)
                                .frame(
                                    width: max(CGFloat(heatmap.columnLabels.count) * cellWidth, 200),
                                    height: max(CGFloat(heatmap.rowLabels.count) * cellHeight, 100)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        }
                    }
                }

                HStack {
                    legendSwatch("Low", color: Color(nsColor: NSColor(srgbRed: 0.10, green: 0.28, blue: 0.56, alpha: 1)))
                    legendSwatch("Neutral", color: Color(nsColor: NSColor(srgbRed: 0.18, green: 0.53, blue: 0.63, alpha: 1)))
                    legendSwatch("High", color: Color(nsColor: NSColor(srgbRed: 0.90, green: 0.46, blue: 0.24, alpha: 1)))
                    Spacer()
                    Text("Range \(String(format: "%.3f", heatmap.valueRange.lowerBound)) → \(String(format: "%.3f", heatmap.valueRange.upperBound))")
                        .font(.caption2)
                        .foregroundStyle(FXAITheme.textMuted)
                }
            }
        }
    }

    private func legendSwatch(_ title: String, color: Color) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 4, style: .continuous)
                .fill(color)
                .frame(width: 18, height: 12)
            Text(title)
                .font(.caption2)
                .foregroundStyle(FXAITheme.textMuted)
        }
    }
}

private struct MetalHeatmapGridView: NSViewRepresentable {
    let heatmap: VisualizationHeatmap

    func makeCoordinator() -> Coordinator {
        Coordinator(heatmap: heatmap)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: context.coordinator.renderer.device)
        view.delegate = context.coordinator.renderer
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.framebufferOnly = false
        view.clearColor = MTLClearColorMake(0.07, 0.08, 0.10, 1.0)
        view.colorPixelFormat = .bgra8Unorm
        context.coordinator.renderer.attach(view: view)
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.renderer.update(heatmap: heatmap)
        nsView.setNeedsDisplay(nsView.bounds)
    }

    final class Coordinator {
        let renderer: MetalHeatmapRenderer

        init(heatmap: VisualizationHeatmap) {
            self.renderer = MetalHeatmapRenderer(heatmap: heatmap)
        }
    }
}

private final class MetalHeatmapRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice

    private var heatmap: VisualizationHeatmap
    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    private weak var view: MTKView?

    init(heatmap: VisualizationHeatmap) {
        self.heatmap = heatmap
        self.device = MTLCreateSystemDefaultDevice() ?? {
            fatalError("Metal is required for Phase 5 heatmaps.")
        }()
        super.init()
        self.commandQueue = device.makeCommandQueue()
        self.pipelineState = makePipelineState()
    }

    func attach(view: MTKView) {
        self.view = view
    }

    func update(heatmap: VisualizationHeatmap) {
        self.heatmap = heatmap
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard
            let descriptor = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let commandQueue,
            let pipelineState
        else {
            return
        }

        let vertices = buildVertices(for: heatmap)
        guard !vertices.isEmpty else { return }

        guard let vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: MemoryLayout<HeatmapVertex>.stride * vertices.count
        ) else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        else {
            return
        }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func makePipelineState() -> MTLRenderPipelineState? {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexIn {
            float2 position;
            float4 color;
        };

        struct VertexOut {
            float4 position [[position]];
            float4 color;
        };

        vertex VertexOut heatmap_vertex(const device VertexIn *vertices [[buffer(0)]], uint vid [[vertex_id]]) {
            VertexOut out;
            out.position = float4(vertices[vid].position, 0.0, 1.0);
            out.color = vertices[vid].color;
            return out;
        }

        fragment float4 heatmap_fragment(VertexOut in [[stage_in]]) {
            return in.color;
        }
        """

        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = library.makeFunction(name: "heatmap_vertex")
            descriptor.fragmentFunction = library.makeFunction(name: "heatmap_fragment")
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            assertionFailure("Failed to build Metal heatmap pipeline: \(error)")
            return nil
        }
    }

    private func buildVertices(for heatmap: VisualizationHeatmap) -> [HeatmapVertex] {
        let rows = heatmap.rowLabels.count
        let columns = heatmap.columnLabels.count
        guard rows > 0, columns > 0 else { return [] }

        let range = heatmap.valueRange
        let xStep = 2.0 / Float(columns)
        let yStep = 2.0 / Float(rows)
        let paddingX = xStep * 0.08
        let paddingY = yStep * 0.12

        var vertices: [HeatmapVertex] = []
        vertices.reserveCapacity(rows * columns * 6)

        for row in 0..<rows {
            for column in 0..<columns {
                let value = heatmap.values[row][column]
                let color = colorForValue(value, range: range)

                let x0 = -1.0 + Float(column) * xStep + paddingX
                let x1 = -1.0 + Float(column + 1) * xStep - paddingX
                let yTop = 1.0 - Float(row) * yStep - paddingY
                let yBottom = 1.0 - Float(row + 1) * yStep + paddingY

                let topLeft = SIMD2<Float>(x0, yTop)
                let topRight = SIMD2<Float>(x1, yTop)
                let bottomLeft = SIMD2<Float>(x0, yBottom)
                let bottomRight = SIMD2<Float>(x1, yBottom)

                vertices.append(HeatmapVertex(position: topLeft, color: color))
                vertices.append(HeatmapVertex(position: bottomLeft, color: color))
                vertices.append(HeatmapVertex(position: topRight, color: color))
                vertices.append(HeatmapVertex(position: topRight, color: color))
                vertices.append(HeatmapVertex(position: bottomLeft, color: color))
                vertices.append(HeatmapVertex(position: bottomRight, color: color))
            }
        }

        return vertices
    }

    private func colorForValue(_ value: Double, range: ClosedRange<Double>) -> SIMD4<Float> {
        guard !value.isNaN else {
            return SIMD4<Float>(0.12, 0.13, 0.16, 1.0)
        }

        let minValue = range.lowerBound
        let maxValue = range.upperBound
        let normalized: Double
        if maxValue == minValue {
            normalized = 0.5
        } else {
            normalized = min(max((value - minValue) / (maxValue - minValue), 0), 1)
        }

        let cool = SIMD3<Float>(0.10, 0.28, 0.56)
        let mid = SIMD3<Float>(0.18, 0.53, 0.63)
        let warm = SIMD3<Float>(0.90, 0.46, 0.24)

        let color: SIMD3<Float>
        if normalized < 0.5 {
            let local = Float(normalized / 0.5)
            color = simd_mix(cool, mid, SIMD3<Float>(repeating: local))
        } else {
            let local = Float((normalized - 0.5) / 0.5)
            color = simd_mix(mid, warm, SIMD3<Float>(repeating: local))
        }

        return SIMD4<Float>(color.x, color.y, color.z, 1.0)
    }
}
