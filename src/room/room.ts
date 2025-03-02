import { WallMaterial } from './room-materials';

export interface RoomConfig {
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    materials: {
        walls: WallMaterial;
        ceiling: WallMaterial;
        floor: WallMaterial;
    };
}

export class Room {
    public config: RoomConfig;
    private device: GPUDevice;
    private uniformBuffer: GPUBuffer;
    private vertexBuffer: GPUBuffer;
    private pipeline: GPURenderPipeline;
    private uniformBindGroup: GPUBindGroup;

    constructor(device: GPUDevice, config: RoomConfig) {
        this.device = device;
        this.config = config;

        // Create uniform buffer
        this.uniformBuffer = device.createBuffer({
            size: 64, // 4x4 matrix
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create vertex buffer
        this.vertexBuffer = this.createVertexBuffer();

        // Create render pipeline
        this.pipeline = this.createRenderPipeline();

        // Create bind group
        this.uniformBindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });
    }

    public getUniformBuffer(): GPUBuffer {
        return this.uniformBuffer;
    }

    private createVertexBuffer(): GPUBuffer {
        const { width, height, depth } = this.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;

        // Define vertices for a room (12 lines, 2 points per line)
        const vertices = new Float32Array([
            // Floor
            -halfWidth, 0, -halfDepth,
            -halfWidth, 0, halfDepth,
            -halfWidth, 0, halfDepth,
            halfWidth, 0, halfDepth,
            halfWidth, 0, halfDepth,
            halfWidth, 0, -halfDepth,
            halfWidth, 0, -halfDepth,
            -halfWidth, 0, -halfDepth,

            // Ceiling
            -halfWidth, height, -halfDepth,
            -halfWidth, height, halfDepth,
            -halfWidth, height, halfDepth,
            halfWidth, height, halfDepth,
            halfWidth, height, halfDepth,
            halfWidth, height, -halfDepth,
            halfWidth, height, -halfDepth,
            -halfWidth, height, -halfDepth,

            // Vertical edges
            -halfWidth, 0, -halfDepth,
            -halfWidth, height, -halfDepth,
            -halfWidth, 0, halfDepth,
            -halfWidth, height, halfDepth,
            halfWidth, 0, halfDepth,
            halfWidth, height, halfDepth,
            halfWidth, 0, -halfDepth,
            halfWidth, height, -halfDepth,
        ]);

        const buffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        new Float32Array(buffer.getMappedRange()).set(vertices);
        buffer.unmap();

        return buffer;
    }

    private createRenderPipeline(): GPURenderPipeline {
        const shader = this.device.createShaderModule({
            code: `
                struct Uniforms {
                    viewProjection: mat4x4f,
                }

                @group(0) @binding(0) var<uniform> uniforms: Uniforms;

                @vertex
                fn vertexMain(@location(0) position: vec3f) -> @builtin(position) vec4f {
                    return uniforms.viewProjection * vec4f(position, 1.0);
                }

                @fragment
                fn fragmentMain() -> @location(0) vec4f {
                    return vec4f(0.8, 0.8, 0.8, 1.0); // Light gray color for room
                }
            `
        });

        return this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [
                    this.device.createBindGroupLayout({
                        entries: [{
                            binding: 0,
                            visibility: GPUShaderStage.VERTEX,
                            buffer: { type: 'uniform' }
                        }]
                    })
                ]
            }),
            vertex: {
                module: shader,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 12, // 3 floats * 4 bytes
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x3'
                    }]
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
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });
    }

    public render(pass: GPURenderPassEncoder): void {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.draw(24, 1, 0, 0); // 12 lines * 2 vertices per line
    }

    public getVolume(): number {
        const { width, height, depth } = this.config.dimensions;
        return width * height * depth;
    }

    public getSurfaceArea(): number {
        const { width, height, depth } = this.config.dimensions;
        return 2 * (width * height + width * depth + height * depth);
    }

    public getTemperature(): number {
        return 20; // Default room temperature in Celsius
    }

    public getHumidity(): number {
        return 50; // Default humidity percentage
    }

    public getClosestValidPosition(position: [number, number, number]): [number, number, number] {
        const { width, height, depth } = this.config.dimensions;
        const halfWidth = width / 2;
        const halfDepth = depth / 2;
        const margin = 0.1; // Minimum distance from walls

        return [
            Math.max(-halfWidth + margin, Math.min(halfWidth - margin, position[0])),
            Math.max(margin, Math.min(height - margin, position[1])),
            Math.max(-halfDepth + margin, Math.min(halfDepth - margin, position[2]))
        ];
    }
}