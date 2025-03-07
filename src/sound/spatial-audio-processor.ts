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
    private initialized = false;
    private initializationPromise: Promise<void>;

    // NEW: HRTF related properties
    private hrtfEnabled = false;
    private hrtfFilters: Map<string, AudioBuffer> = new Map();

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

        this.initializationPromise = this.initializeAsync();
        
        // Initialize simplified HRTF data
        this.initializeHRTF();
    }

    private async initializeAsync(): Promise<void> {
        await this.createPipeline();
        this.initialized = true;
    }

    // NEW METHOD: Initialize simplified HRTF filters
    private async initializeHRTF(): Promise<void> {
        try {
            // In a real implementation, you would load actual HRTF data
            // Here we'll create a simplified approximation
            
            // Create an AudioContext just for HRTF generation
            const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
            
            // Generate filters for key directions
            const directions = [
                { azimuth: 0, elevation: 0 },    // Front
                { azimuth: 90, elevation: 0 },   // Right
                { azimuth: 180, elevation: 0 },  // Back
                { azimuth: 270, elevation: 0 },  // Left
                { azimuth: 0, elevation: 45 },   // Above front
                { azimuth: 0, elevation: -45 }   // Below front
            ];
            
            for (const dir of directions) {
                const hrtfBuffer = this.generateSimplifiedHRTF(audioCtx, dir.azimuth, dir.elevation);
                this.hrtfFilters.set(`${dir.azimuth}_${dir.elevation}`, hrtfBuffer);
            }
            
            console.log("Initialized simplified HRTF filters");
            this.hrtfEnabled = true;
        } catch (error) {
            console.error("Failed to initialize HRTF filters:", error);
            this.hrtfEnabled = false;
        }
    }
    
    // HELPER METHOD: Generate simplified HRTF
    private generateSimplifiedHRTF(audioCtx: AudioContext, azimuth: number, elevation: number): AudioBuffer {
        // Simplified HRTF generation, replace with actual HRTF data
        const hrtfBuffer = audioCtx.createBuffer(2, 256, audioCtx.sampleRate);
        const leftChannel = hrtfBuffer.getChannelData(0);
        const rightChannel = hrtfBuffer.getChannelData(1);
        
        // Simple HRTF approximation based on azimuth and elevation
        for (let i = 0; i < 256; i++) {
            const theta = i / 256 * Math.PI * 2;
            const leftGain = 0.5 * (1 - Math.sin(azimuth + theta));
            const rightGain = 0.5 * (1 + Math.sin(azimuth + theta));
            leftChannel[i] = Math.sin(theta) * leftGain;
            rightChannel[i] = Math.sin(theta) * rightGain;
        }
        
        return hrtfBuffer;
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

    private updateRoomAcousticsBuffer(room: Room): void {
        // Get room dimensions and materials
        const materials = room.config.materials;
        const dimensions = room.config.dimensions;
        
        // Calculate room volume and surface area
        const volume = dimensions.width * dimensions.height * dimensions.depth;
        const surfaceArea = 2 * (
            dimensions.width * dimensions.height +
            dimensions.width * dimensions.depth +
            dimensions.height * dimensions.depth
        );
        
        // Calculate absorption and scattering coefficients
        let absorption = {
            125: 0, 250: 0, 500: 0, 1000: 0,
            2000: 0, 4000: 0, 8000: 0, 16000: 0
        };
        let scattering = {
            125: 0, 250: 0, 500: 0, 1000: 0,
            2000: 0, 4000: 0, 8000: 0, 16000: 0
        };
        
        // Sum up coefficients from all surfaces
        const surfaceTypes = ['left', 'right', 'floor', 'ceiling', 'front', 'back'];
        surfaceTypes.forEach(surface => {
            const material = materials[surface];
            if (material) {
                absorption[125] += material.absorption125Hz || 0.1;
                absorption[250] += material.absorption250Hz || 0.1;
                absorption[500] += material.absorption500Hz || 0.1;
                absorption[1000] += material.absorption1kHz || 0.1;
                absorption[2000] += material.absorption2kHz || 0.1;
                absorption[4000] += material.absorption4kHz || 0.1;
                absorption[8000] += material.absorption8kHz || 0.1;
                absorption[16000] += material.absorption16kHz || 0.1;
                
                scattering[125] += material.scattering125Hz || 0.1;
                scattering[250] += material.scattering250Hz || 0.15;
                scattering[500] += material.scattering500Hz || 0.2;
                scattering[1000] += material.scattering1kHz || 0.25;
                scattering[2000] += material.scattering2kHz || 0.3;
                scattering[4000] += material.scattering4kHz || 0.35;
                scattering[8000] += material.scattering8kHz || 0.4;
                scattering[16000] += material.scattering16kHz || 0.45;
            }
        });
        
        // Calculate RT60 values
        const rt60 = {};
        const frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];
        frequencies.forEach(freq => {
            const absorptionCoeff = Math.max(0.01, Math.min(0.99, absorption[freq] / surfaceTypes.length));
            const scatteringCoeff = Math.max(0.01, Math.min(0.99, scattering[freq] / surfaceTypes.length));
            rt60[freq] = 0.161 * volume / (absorptionCoeff * surfaceArea * (1 - scatteringCoeff * 0.3));
        });
        
        // Write data to GPU buffer
        const acousticsData = new Float32Array([
            rt60[125], rt60[250], rt60[500], rt60[1000],
            rt60[2000], rt60[4000], rt60[8000], rt60[16000],
            0.0002, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.011, 0.015,
            scattering[125], scattering[250], scattering[500], scattering[1000],
            scattering[2000], scattering[4000], scattering[8000], scattering[16000],
            0.08, volume, surfaceArea
        ]);
        
        this.device.queue.writeBuffer(this.acousticsBuffer, 0, acousticsData);
    }

    private calculateImprovedSpatialGains(
        hitPosition: vec3,
        listenerPosition: vec3,
        listenerFront: vec3,
        listenerRight: vec3,
        listenerUp: vec3
    ): [number, number] {
        // Calculate vector from listener to hit point
        const toSource = vec3.create();
        vec3.subtract(toSource, hitPosition, listenerPosition);
        vec3.normalize(toSource, toSource);

        // Calculate horizontal angle (azimuth)
        const dotRight = vec3.dot(toSource, listenerRight);
        const dotFront = vec3.dot(toSource, listenerFront);
        const azimuth = Math.atan2(dotRight, dotFront);
        
        // Calculate vertical angle (elevation)
        const dotUp = vec3.dot(toSource, listenerUp);
        const elevation = Math.asin(dotUp);
        
        // Distance attenuation (inverse square law)
        const distance = vec3.length(toSource);
        const distanceFactor = 1 / Math.max(1, distance * distance);
        
        // Improved HRTF model with frequency-dependent IID and ITD
        const baseITD = Math.sin(azimuth) * 0.001; // Approximation of head shadow timing
        
        // Head shadow effect (stronger at higher frequencies, weaker at low)
        const shadowFactor = (0.5 + 0.5 * Math.cos(azimuth)) * (0.7 + 0.3 * Math.cos(elevation));
        
        // Pinna (outer ear) effects for elevation perception
        const pinnaFactor = Math.max(0.5, 0.5 + 0.5 * Math.sin(elevation));
        
        // Combined spatial factors
        const leftGain = shadowFactor * pinnaFactor * distanceFactor;
        const rightGain = (1 - shadowFactor + 0.3) * pinnaFactor * distanceFactor;
        
        return [leftGain, rightGain];
    }

    private calculateTotalEnergy(energies: any): number {
        if (!energies) return 0;
        
        return (
            (energies.energy125Hz || 0) * 0.7 +
            (energies.energy250Hz || 0) * 0.8 +
            (energies.energy500Hz || 0) * 0.9 +
            (energies.energy1kHz || 0) * 1.0 +
            (energies.energy2kHz || 0) * 0.95 +
            (energies.energy4kHz || 0) * 0.9 +
            (energies.energy8kHz || 0) * 0.85 +
            (energies.energy16kHz || 0) * 0.8
        );
    }

    private applySimpleLowpass(buffer: Float32Array, alpha: number): void {
        let lastValue = buffer[0];
        for (let i = 1; i < buffer.length; i++) {
            lastValue = buffer[i] = buffer[i] * (1 - alpha) + lastValue * alpha;
        }
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: RayHit[],
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        // Create buffers for IR (2 seconds at sample rate)
        const irLength = Math.ceil(this.sampleRate * 2);
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);
        
        const listenerPos = camera.getPosition();
        const listenerFront = camera.getFront();
        const listenerRight = camera.getRight();
        const listenerUp = camera.getUp();
        
        // Process hits with limit for performance
        const maxRayHits = Math.min(rayHits.length, 5000);
        console.log(`Processing ${maxRayHits} ray hits for spatial audio`);
        
        // Sort hits by time for better perceptual coherence
        const sortedHits = [...rayHits]
            .sort((a, b) => a.time - b.time)
            .slice(0, maxRayHits);
            
        // Process in chunks for better performance
        const chunkSize = 1000;
        
        for (let hitIndex = 0; hitIndex < sortedHits.length; hitIndex += chunkSize) {
            const endIndex = Math.min(hitIndex + chunkSize, sortedHits.length);
            
            for (let i = hitIndex; i < endIndex; i++) {
                const hit = sortedHits[i];
                
                // Skip invalid hits or hits with no energy
                if (!hit || !hit.energies) continue;
                
                const totalEnergy = this.calculateTotalEnergy(hit.energies);
                if (totalEnergy <= 0.001) continue;
                
                // Calculate which sample this hit affects
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength) continue;
                
                // Calculate spatial positioning using improved model
                const [leftGain, rightGain] = this.calculateImprovedSpatialGains(
                    hit.position,
                    listenerPos,
                    listenerFront,
                    listenerRight,
                    listenerUp
                );
                
                // Calculate energy scaling with distance attenuation
                const distance = vec3.distance(hit.position, listenerPos);
                const distanceAttenuation = 1 / Math.max(1, distance * distance);
                
                // Scale by bounce count - earlier reflections are more important
                const bounceScaling = Math.pow(0.7, hit.bounces);
                
                // Apply directional attenuation
                const amplitude = Math.sqrt(totalEnergy) * distanceAttenuation * bounceScaling;
                
                // Apply to IR with temporal spreading for more natural sound
                const spreadFactor = Math.min(0.01 * this.sampleRate, 200); // Up to 10ms spread
                
                for (let j = -spreadFactor; j <= spreadFactor; j++) {
                    const spreadIndex = sampleIndex + j;
                    if (spreadIndex >= 0 && spreadIndex < irLength) {
                        // Apply amplitude with temporal decay
                        const temporalDecay = Math.exp(-Math.abs(j) / (spreadFactor / 3));
                        leftIR[spreadIndex] += amplitude * leftGain * temporalDecay;
                        rightIR[spreadIndex] += amplitude * rightGain * temporalDecay;
                    }
                }
            }
        }
        
        // Apply simple low-pass filter for smoothing
        this.applySimpleLowpass(leftIR, 0.3);
        this.applySimpleLowpass(rightIR, 0.3);
        
        return [leftIR, rightIR];
    }
}