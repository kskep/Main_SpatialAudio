import { vec3 } from 'gl-matrix';
import { FrequencyBands } from './ray';

export class RayRenderer {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private vertexBuffer: GPUBuffer;
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;

    private lastViewProjection: Float32Array | null = null;
    private hasRendered: boolean = false;

    constructor(device: GPUDevice) {
        this.device = device;

        // Updated shader to handle wave properties
        const shader = this.device.createShaderModule({
            code: `
                struct Uniforms {
                    viewProjection: mat4x4f,
                };

                struct VertexInput {
                    @location(0) position: vec3f,
                    @location(1) energy: f32,
                    @location(2) low_band: f32,
                    @location(3) mid_band: f32,
                    @location(4) high_band: f32,
                    @location(5) phase: f32,
                };

                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) energy: f32,
                    @location(1) low_band: f32,
                    @location(2) mid_band: f32,
                    @location(3) high_band: f32,
                    @location(4) phase: f32,
                };

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vertexMain(input: VertexInput) -> VertexOutput {
                    var output: VertexOutput;
                    output.position = uniforms.viewProjection * vec4f(input.position, 1.0);
                    output.energy = input.energy;
                    output.low_band = input.low_band;
                    output.mid_band = input.mid_band;
                    output.high_band = input.high_band;
                    output.phase = input.phase;
                    return output;
                }

                @fragment
                fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                    // Modulate color based on phase
                    let phaseColor = (sin(input.phase) + 1.0) * 0.5;

                    // Combine with frequency band colors
                    let color = vec3f(
                        input.high_band * 2.0 * phaseColor,
                        input.mid_band * 2.0 * phaseColor,
                        input.low_band * 2.0 * phaseColor
                    );

                    let alpha = mix(0.3, 1.0, input.energy * phaseColor);
                    return vec4f(color, alpha);
                }
            `
        });

        // Create pipeline with updated vertex layout
        this.pipeline = device.createRenderPipeline({
            vertex: {
                module: shader,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 32, // vec3f + 5 floats (3*4 + 5*4 = 32 bytes)
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                        { shaderLocation: 1, offset: 12, format: 'float32' },   // energy
                        { shaderLocation: 2, offset: 16, format: 'float32' },   // low_band
                        { shaderLocation: 3, offset: 20, format: 'float32' },   // mid_band
                        { shaderLocation: 4, offset: 24, format: 'float32' },   // high_band
                        { shaderLocation: 5, offset: 28, format: 'float32' },   // phase
                    ]
                }]
            },
            fragment: {
                module: shader,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: 'bgra8unorm',
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        },
                    },
                }]
            },
            primitive: {
                topology: 'line-list'
            },
            layout: 'auto', // let WebGPU infer the layout
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            },
        });

        // Create uniform buffer
        this.uniformBuffer = device.createBuffer({
            size: 64, // mat4x4
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Get the bind group layout from the pipeline
        const uniformBindGroupLayout = this.pipeline.getBindGroupLayout(0);

        // Create bind group using the layout from the pipeline
        this.uniformBindGroup = device.createBindGroup({
            layout: uniformBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });

        // Create empty vertex buffer (will be updated with ray data)
        this.vertexBuffer = device.createBuffer({
            size: 1024, // Initial size, will be recreated as needed
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
    }

    private calculateFrequencyWeightedEnergy(energies: FrequencyBands): {value: number, bandDistribution: number[]} {
        // Weights for visualization (emphasize different frequency ranges)
        const weights = {
            low: 0.7,    // 125-500 Hz
            mid: 1.0,    // 1k-4k Hz
            high: 0.8    // 8k-16k Hz
        };

        const lowFreqs = [
            energies.energy125Hz * weights.low,
            energies.energy250Hz * weights.low,
            energies.energy500Hz * weights.low
        ];

        const midFreqs = [
            energies.energy1kHz * weights.mid,
            energies.energy2kHz * weights.mid,
            energies.energy4kHz * weights.mid
        ];

        const highFreqs = [
            energies.energy8kHz * weights.high,
            energies.energy16kHz * weights.high
        ];

        const lowAvg = lowFreqs.reduce((a, b) => a + b, 0) / lowFreqs.length;
        const midAvg = midFreqs.reduce((a, b) => a + b, 0) / midFreqs.length;
        const highAvg = highFreqs.reduce((a, b) => a + b, 0) / highFreqs.length;

        return {
            value: (lowAvg + midAvg + highAvg) / 3,
            bandDistribution: [lowAvg, midAvg, highAvg]
        };
    }

    public render(
        pass: GPURenderPassEncoder,
        viewProjection: Float32Array,
        rays: { origin: vec3, direction: vec3, energies: FrequencyBands }[],
        roomDimensions: { width: number, height: number, depth: number }
    ): void {
        // If we've already rendered and the view hasn't changed, skip rendering
        if (this.hasRendered && this.lastViewProjection && 
            this.arraysEqual(this.lastViewProjection, viewProjection)) {
            return;
        }

        // Update uniform buffer with view projection matrix
        this.device.queue.writeBuffer(this.uniformBuffer, 0, viewProjection);

        // Create vertex data for rays with wave properties
        const vertices = new Float32Array(rays.length * 16); // 2 vertices per ray * 8 floats per vertex
        let vertexOffset = 0;

        const { width, height, depth } = roomDimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;

        for (const ray of rays) {
            const { value: energy, bandDistribution } = this.calculateFrequencyWeightedEnergy(ray.energies);
            const phase = Math.random() * Math.PI * 2; // Temporary phase value, should come from ray data

            // Start point
            vertices[vertexOffset++] = ray.origin[0];
            vertices[vertexOffset++] = ray.origin[1];
            vertices[vertexOffset++] = ray.origin[2];
            vertices[vertexOffset++] = energy;
            vertices[vertexOffset++] = bandDistribution[0];
            vertices[vertexOffset++] = bandDistribution[1];
            vertices[vertexOffset++] = bandDistribution[2];
            vertices[vertexOffset++] = phase;

            // Calculate intersection with room boundaries
            let minT = Infinity;
            let validIntersection = false;

            // Check X planes (left/right walls)
            if (Math.abs(ray.direction[0]) > 0.0001) {
                const tx1 = (-halfWidth - ray.origin[0]) / ray.direction[0];
                const tx2 = (halfWidth - ray.origin[0]) / ray.direction[0];
                const tx = tx1 > 0 ? tx1 : tx2 > 0 ? tx2 : Infinity;
                if (tx > 0) {
                    minT = Math.min(minT, tx);
                    validIntersection = true;
                }
            }

            // Check Y planes (floor/ceiling)
            if (Math.abs(ray.direction[1]) > 0.0001) {
                const ty1 = (0 - ray.origin[1]) / ray.direction[1];
                const ty2 = (height - ray.origin[1]) / ray.direction[1];
                let ty = Infinity;

                if (ty1 > 0 && ray.origin[1] + ty1 * ray.direction[1] <= height) {
                    ty = ty1;
                }
                if (ty2 > 0 && ray.origin[1] + ty2 * ray.direction[1] >= 0 && ty2 < ty) {
                    ty = ty2;
                }

                if (ty < Infinity) {
                    minT = Math.min(minT, ty);
                    validIntersection = true;
                }
            }

            // Check Z planes (front/back walls)
            if (Math.abs(ray.direction[2]) > 0.0001) {
                const tz1 = (-halfDepth - ray.origin[2]) / ray.direction[2];
                const tz2 = (halfDepth - ray.origin[2]) / ray.direction[2];
                const tz = tz1 > 0 ? tz1 : tz2 > 0 ? tz2 : Infinity;
                if (tz > 0) {
                    minT = Math.min(minT, tz);
                    validIntersection = true;
                }
            }

            // Use the intersection distance for ray length
            const rayLength = validIntersection ? minT : 50.0;
            const endPoint = vec3.scaleAndAdd(vec3.create(), ray.origin, ray.direction, rayLength);

            // End point
            vertices[vertexOffset++] = endPoint[0];
            vertices[vertexOffset++] = endPoint[1];
            vertices[vertexOffset++] = endPoint[2];
            vertices[vertexOffset++] = energy * 0.2;
            vertices[vertexOffset++] = bandDistribution[0] * 0.2;
            vertices[vertexOffset++] = bandDistribution[1] * 0.2;
            vertices[vertexOffset++] = bandDistribution[2] * 0.2;
            vertices[vertexOffset++] = phase + Math.PI; // Phase at end point
        }

        // Update or recreate vertex buffer if needed
        if (this.vertexBuffer.size < vertices.byteLength) {
            this.vertexBuffer.destroy();
            this.vertexBuffer = this.device.createBuffer({
                size: vertices.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
        }

        // Update vertex buffer and draw
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.draw(rays.length * 2, 1, 0, 0);

        // Store the current view projection
        this.lastViewProjection = new Float32Array(viewProjection);
        this.hasRendered = true;
    }

    // Helper method to compare Float32Arrays
    private arraysEqual(a: Float32Array, b: Float32Array): boolean {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 0.0001) return false; // Using epsilon for float comparison
        }
        return true;
    }

    // Add method to reset the render state
    public resetRender(): void {
        this.hasRendered = false;
        this.lastViewProjection = null;
    }
}