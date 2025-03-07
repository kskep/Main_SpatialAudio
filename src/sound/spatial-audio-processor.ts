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

    /**
     * Enhanced HRTF approximation with better spatial cues
     */
    public calculateImprovedHRTF(
        sourcePosition: vec3,
        listenerPosition: vec3, 
        listenerForward: vec3,
        listenerRight: vec3,
        listenerUp: vec3
    ): [number, number] {
        // Create vector from listener to source
        const toSource = vec3.create();
        vec3.subtract(toSource, sourcePosition, listenerPosition);
        const distance = vec3.length(toSource);
        
        // Normalize only if non-zero distance
        if (distance > 0.001) {
            vec3.scale(toSource, toSource, 1/distance);
        } else {
            vec3.copy(toSource, listenerForward); // Default to forward if too close
        }
        
        // Calculate azimuth angle (horizontal plane)
        const projectedDir = vec3.fromValues(
            vec3.dot(toSource, listenerRight),
            0,
            vec3.dot(toSource, listenerForward)
        );
        vec3.normalize(projectedDir, projectedDir);
        
        const azimuth = Math.atan2(projectedDir[0], projectedDir[2]);
        
        // Calculate elevation angle (vertical plane)
        const elevation = Math.asin(vec3.dot(toSource, listenerUp));
        
        // Enhanced directional cues (interaural level differences)
        let leftGain = 0, rightGain = 0;
        
        // Apply azimuth-based gain (horizontal localization)
        // This creates a more natural stereo field with smooth left-right transitions
        if (azimuth < 0) { // Source is to the left
            leftGain = 0.9 - 0.4 * azimuth/Math.PI; // Left gain increases
            rightGain = 0.4 + 0.5 * (1 + azimuth/Math.PI); // Right gain decreases
        } else { // Source is to the right
            leftGain = 0.4 + 0.5 * (1 - azimuth/Math.PI); // Left gain decreases
            rightGain = 0.9 + 0.4 * azimuth/Math.PI; // Right gain increases
        }
        
        // Apply elevation effects (pinna filtering)
        const elevationFactor = 0.7 + 0.3 * Math.cos(elevation);
        leftGain *= elevationFactor;
        rightGain *= elevationFactor;
        
        // Apply distance attenuation
        const distanceFactor = 1.0 / Math.max(1, distance * distance * 0.1);
        
        // Apply front-back disambiguation
        const frontFactor = 0.7 + 0.3 * vec3.dot(toSource, listenerForward);
        
        // Final gains
        leftGain *= distanceFactor * frontFactor;
        rightGain *= distanceFactor * frontFactor;
        
        return [leftGain, rightGain];
    }

    private calculateEnhancedHRTF(
        sourcePosition: vec3,
        listenerPosition: vec3,
        listenerForward: vec3,
        listenerRight: vec3,
        listenerUp: vec3
    ): [number, number, number] { // Returns [leftGain, rightGain, directionalFactor]
        // Vector from listener to source
        const toSource = vec3.create();
        vec3.subtract(toSource, sourcePosition, listenerPosition);
        
        // Distance to source (for distance cues)
        const distance = vec3.length(toSource);
        vec3.normalize(toSource, toSource);
        
        // Calculate azimuth (horizontal angle)
        const rightComponent = vec3.dot(toSource, listenerRight);
        const forwardComponent = vec3.dot(toSource, listenerForward);
        const azimuth = Math.atan2(rightComponent, forwardComponent);
        
        // Calculate elevation (vertical angle)
        const upComponent = vec3.dot(toSource, listenerUp);
        const elevation = Math.asin(Math.max(-1, Math.min(1, upComponent)));
        
        // Head shadow effect - more pronounced at higher frequencies
        // Stronger attenuation on the opposite ear
        const shadowFactor = 1.0 - 0.6 * Math.abs(Math.sin(azimuth));
        
        // Interaural time difference - affects phase for frequencies below 1500 Hz
        const ITD = 0.0007 * Math.sin(azimuth); // Approximate ITD in seconds
        
        // HRTF frequency-dependent effects
        // For low frequencies: mostly level differences
        // For high frequencies: complex spectral shaping and stronger ILD
        const pinna_effect = Math.abs(Math.sin(elevation)) * 0.3;
        
        // Enhanced ILD (Interaural Level Difference)
        let leftGain = 0.5;
        let rightGain = 0.5;
        
        if (azimuth < 0) { // Sound from the left
            leftGain = 0.8 + 0.2 * Math.cos(azimuth);
            rightGain = Math.max(0.1, shadowFactor);
        } else { // Sound from the right
            rightGain = 0.8 + 0.2 * Math.cos(azimuth);
            leftGain = Math.max(0.1, shadowFactor);
        }
        
        // Elevation effects
        const elevationFactor = Math.abs(Math.sin(elevation));
        leftGain *= (1.0 - elevationFactor * 0.3);
        rightGain *= (1.0 - elevationFactor * 0.3);
        
        // Distance attenuation
        const distanceFactor = 1.0 / (1.0 + distance * 0.3);
        
        // Directional factor depends on how much the sound is in front of the listener
        const directionalFactor = Math.max(0.3, (forwardComponent + 1.0) * 0.5);
        
        return [
            leftGain * distanceFactor, 
            rightGain * distanceFactor,
            directionalFactor
        ];
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: RayHit[],
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        // Create IR buffers
        const irLength = Math.ceil(this.sampleRate * 2); // 2 seconds of IR
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);
        
        const listenerPos = camera.getPosition();
        const listenerFront = camera.getFront();
        const listenerRight = camera.getRight();
        const listenerUp = camera.getUp();
        
        // Sort hits by time for better perceptual coherence
        const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);
        
        // Process each hit to generate impulse response samples
        for (const hit of sortedHits) {
            // Skip invalid hits
            if (!hit || !hit.energies) continue;
            
            // Calculate which sample this hit affects
            const sampleIndex = Math.floor(hit.time * this.sampleRate);
            if (sampleIndex < 0 || sampleIndex >= irLength) continue;
            
            // Get enhanced spatial gains for this ray hit
            const [leftGain, rightGain, directionalFactor] = this.calculateEnhancedHRTF(
                hit.position,
                listenerPos,
                listenerFront,
                listenerRight,
                listenerUp
            );
            
            // Apply energy scaling
            const energyScale = this.calculateTotalEnergy(hit.energies);
            
            // Apply bounce attenuation - earlier bounces are more important
            const bounceScaling = Math.pow(0.8, hit.bounces);
            
            // Early reflections should be stronger
            const isEarly = hit.time < 0.1;
            const timeScaling = isEarly ? 2.0 : Math.exp(-hit.time * 3);
            
            // Calculate total amplitude with all factors
            const amplitude = Math.sqrt(energyScale) * bounceScaling * timeScaling * directionalFactor;
            
            // Apply to IR with temporal spreading for more natural sound
            const spreadFactor = Math.min(0.005 * this.sampleRate, 100); // 5ms spread
            
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
        
        // Apply a gentle lowpass filter to make sound more natural
        this.applyLowPassFilter(leftIR, 0.2);
        this.applyLowPassFilter(rightIR, 0.2);
        
        return [leftIR, rightIR];
    }

    private applyLowPassFilter(buffer: Float32Array, alpha: number): void {
        let lastValue = buffer[0];
        for (let i = 1; i < buffer.length; i++) {
            lastValue = buffer[i] = buffer[i] * (1 - alpha) + lastValue * alpha;
        }
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
}