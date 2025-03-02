import { vec3 } from 'gl-matrix';
import { Camera } from '../camera/camera';
import { FrequencyBands } from '../raytracer/ray';
import { Room } from '../room/room';

interface RoomAcoustics {
    rt60: {
        band125Hz: number;
        band250Hz: number;
        band500Hz: number;
        band1kHz: number;
        band2kHz: number;
        band4kHz: number;
        band8kHz: number;
        band16kHz: number;
    };
    airAbsorption: {
        band125Hz: number;
        band250Hz: number;
        band500Hz: number;
        band1kHz: number;
        band2kHz: number;
        band4kHz: number;
        band8kHz: number;
        band16kHz: number;
    };
    earlyReflectionTime: number;
    roomVolume: number;
    totalSurfaceArea: number;
}

export interface RayHit {
    position: vec3;
    time: number;
    energies: FrequencyBands;
    bounces: number;
    phase: number;
    frequency: number;
    dopplerShift: number;
}

interface WaveProperties {
    phase: number;
    frequency: number;
    dopplerShift: number;
}

export class SpatialAudioProcessor {
    private device: GPUDevice;
    private computePipeline!: GPUComputePipeline;
    private bindGroup!: GPUBindGroup;
    private listenerBuffer: GPUBuffer;
    private rayHitsBuffer!: GPUBuffer;
    private spatialIRBuffer!: GPUBuffer;
    private paramsBuffer: GPUBuffer;
    private acousticsBuffer: GPUBuffer;
    private wavePropertiesBuffer: GPUBuffer;
    private outputBuffer: GPUBuffer;
    private sampleRate: number;
    private readonly WORKGROUP_SIZE = 256;

    constructor(device: GPUDevice, sampleRate: number = 44100) {
        this.device = device;
        this.sampleRate = sampleRate;

        this.listenerBuffer = device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Listener Data Buffer'
        });

        this.paramsBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Audio Params Buffer'
        });

        this.acousticsBuffer = device.createBuffer({
            size: 256, // Increased size for 8 frequency bands
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Room Acoustics Buffer'
        });

        this.wavePropertiesBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Wave Properties Buffer'
        });

        this.outputBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'Output Buffer'
        });

        this.initializeAsync();
    }

    private async initializeAsync(): Promise<void> {
        await this.createPipeline();
    }

    private async createPipeline(): Promise<void> {
        const shaderModule = this.device.createShaderModule({
            code: await fetch('/src/raytracer/shaders/spatial_audio.wgsl').then(r => r.text()),
            label: 'Spatial Audio Shader'
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 64 }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', minBindingSize: 96 }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 48 }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform', minBindingSize: 256 }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage', minBindingSize: 128 }
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage', minBindingSize: 16 }
                }
            ],
            label: 'Spatial Audio Bind Group Layout'
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
            label: 'Spatial Audio Pipeline Layout'
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            },
            label: 'Spatial Audio Pipeline'
        });
    }

    private createOrResizeBuffers(hitCount: number): void {
        // Calculate buffer sizes with proper alignment
        const rayHitSize = 96; // Increased size for 8 frequency bands
        const rayHitsSize = Math.max(hitCount * rayHitSize, rayHitSize);
        const wavePropsSize = Math.max(hitCount * 16, 16);
        // Each IR sample needs: left[1], right[1], frequency[1], time[1] = 4 floats
        const spatialIRSize = Math.max(hitCount * 4 * 4, 64); // 4 floats * 4 bytes per float, minimum 64 bytes

        // Ensure minimum workgroup alignment
        if (!this.rayHitsBuffer || this.rayHitsBuffer.size < rayHitsSize) {
            if (this.rayHitsBuffer) this.rayHitsBuffer.destroy();
            this.rayHitsBuffer = this.device.createBuffer({
                // Each ray hit needs: position[3], time[1], energies[8], phase[1], frequency[1], dopplerShift[1], padding[1]
                size: Math.max(hitCount * 16 * 4, 64), // 16 float32s per hit
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Ray Hits Buffer'
            });
        }

        if (!this.wavePropertiesBuffer || this.wavePropertiesBuffer.size < wavePropsSize) {
            if (this.wavePropertiesBuffer) this.wavePropertiesBuffer.destroy();
            this.wavePropertiesBuffer = this.device.createBuffer({
                size: wavePropsSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                label: 'Wave Properties Buffer'
            });
        }

        if (!this.spatialIRBuffer || this.spatialIRBuffer.size < spatialIRSize) {
            if (this.spatialIRBuffer) this.spatialIRBuffer.destroy();
            this.spatialIRBuffer = this.device.createBuffer({
                size: spatialIRSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                label: 'Spatial IR Buffer'
            });

            const zeros = new Float32Array(hitCount * 4);
            this.device.queue.writeBuffer(this.spatialIRBuffer, 0, zeros);
        }

        this.updateBindGroup();
    }

    private updateBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.listenerBuffer } },
                { binding: 1, resource: { buffer: this.rayHitsBuffer } },
                { binding: 2, resource: { buffer: this.spatialIRBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
                { binding: 4, resource: { buffer: this.acousticsBuffer } },
                { binding: 5, resource: { buffer: this.wavePropertiesBuffer } },
                { binding: 6, resource: { buffer: this.outputBuffer } }
            ],
            label: 'Spatial Audio Bind Group'
        });
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: RayHit[],
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        if (rayHits.length === 0) {
            console.warn('No ray hits to process');
            return [new Float32Array(0), new Float32Array(0)];
        }

        this.createOrResizeBuffers(rayHits.length);

        // Prepare ray hits data
        const rayHitsData = new Float32Array(rayHits.length * 16); // 16 floats per hit
        const wavePropsData = new Float32Array(rayHits.length * 4);

        rayHits.forEach((hit, i) => {
            const baseIndex = i * 16; // Updated for new layout

            const position = hit.position || vec3.create();
            rayHitsData[baseIndex] = position[0];
            rayHitsData[baseIndex + 1] = position[1];
            rayHitsData[baseIndex + 2] = position[2];
            rayHitsData[baseIndex + 3] = Math.max(hit.time || 0, 0);

            // Time value
            rayHitsData[baseIndex + 3] = Math.max(hit.time || 0, 0);

            // Get energies with fallback
            const energies = hit.energies || {
                energy125Hz: 0, energy250Hz: 0, energy500Hz: 0, energy1kHz: 0,
                energy2kHz: 0, energy4kHz: 0, energy8kHz: 0, energy16kHz: 0
            };

            // Store energies (match shader struct layout)
            rayHitsData[baseIndex + 4] = Math.max(energies.energy125Hz || 0, 0);
            rayHitsData[baseIndex + 5] = Math.max(energies.energy250Hz || 0, 0);
            rayHitsData[baseIndex + 6] = Math.max(energies.energy500Hz || 0, 0);
            rayHitsData[baseIndex + 7] = Math.max(energies.energy1kHz || 0, 0);
            rayHitsData[baseIndex + 8] = Math.max(energies.energy2kHz || 0, 0);
            rayHitsData[baseIndex + 9] = Math.max(energies.energy4kHz || 0, 0);
            rayHitsData[baseIndex + 10] = Math.max(energies.energy8kHz || 0, 0);
            rayHitsData[baseIndex + 11] = Math.max(energies.energy16kHz || 0, 0);

            // Store wave properties
            rayHitsData[baseIndex + 12] = hit.phase || 0;
            rayHitsData[baseIndex + 13] = Math.max(hit.frequency || 440, 20);
            rayHitsData[baseIndex + 14] = Math.max(hit.dopplerShift || 1, 0.1);
            rayHitsData[baseIndex + 15] = 1.0; // Padding/alignment

            const waveBaseIndex = i * 4;
            wavePropsData[waveBaseIndex] = hit.phase || 0;
            wavePropsData[waveBaseIndex + 1] = Math.max(hit.frequency || 440, 20);
            wavePropsData[waveBaseIndex + 2] = Math.max(hit.dopplerShift || 1, 0.1);
            wavePropsData[waveBaseIndex + 3] = 1.0;
        });

        // Write data to GPU buffers
        this.device.queue.writeBuffer(this.rayHitsBuffer, 0, rayHitsData);
        this.device.queue.writeBuffer(this.wavePropertiesBuffer, 0, wavePropsData);

        // Update listener data
        const front = camera.getFront();
        const position = camera.getPosition();
        const up = camera.getUp();
        const right = vec3.create();
        vec3.cross(right, front, up);
        vec3.normalize(right, right);

        const listenerData = new Float32Array([
            position[0], position[1], position[2], 0,
            front[0], front[1], front[2], 0,
            up[0], up[1], up[2], 0,
            right[0], right[1], right[2], 0
        ]);
        this.device.queue.writeBuffer(this.listenerBuffer, 0, listenerData);

        // Execute compute pipeline
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindGroup);

        const workgroupCount = Math.ceil(rayHits.length / this.WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount);
        computePass.end();

        try {
            // Read results (4 floats per sample: left, right, frequency, time)
            const bufferSize = Math.max(rayHits.length * 4 * 4, 256); // Ensure minimum buffer size
            const readbackBuffer = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            // Copy the entire IR buffer
            commandEncoder.copyBufferToBuffer(
                this.spatialIRBuffer,
                0,
                readbackBuffer,
                0,
                bufferSize
            );

            this.device.queue.submit([commandEncoder.finish()]);

            await readbackBuffer.mapAsync(GPUMapMode.READ);
            const results = new Float32Array(readbackBuffer.getMappedRange());

            const leftChannel = new Float32Array(rayHits.length);
            const rightChannel = new Float32Array(rayHits.length);

            // Validate and process results
            let validSamples = 0;
            for (let i = 0; i < rayHits.length; i++) {
                const left = results[i * 4];
                const right = results[i * 4 + 1];

                if (isFinite(left) && !isNaN(left) && isFinite(right) && !isNaN(right)) {
                    leftChannel[i] = left;
                    rightChannel[i] = right;
                    validSamples++;
                } else {
                    leftChannel[i] = 0;
                    rightChannel[i] = 0;
                }
            }

            readbackBuffer.unmap();
            readbackBuffer.destroy();

            if (validSamples === 0) {
                console.error('No valid samples in IR calculation');
                return [new Float32Array(1), new Float32Array(1)]; // Return minimal valid buffers
            }

            console.log(`Processed ${validSamples} valid IR samples out of ${rayHits.length}`);
            return [leftChannel, rightChannel];
        } catch (error) {
            console.error('Error processing spatial audio:', error);
            return [new Float32Array(1), new Float32Array(1)]; // Return minimal valid buffers
        }
    }
}