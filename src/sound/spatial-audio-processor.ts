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

    private generateImpulseResponse(leftIR: Float32Array, rightIR: Float32Array, rayHits: RayHit[], camera: Camera): void {
        // Clear the arrays
        leftIR.fill(0);
        rightIR.fill(0);
        
        const timeStep = 1 / this.sampleRate;
        
        // For each ray hit, add an impulse at the arrival time
        for (const hit of rayHits) {
            // Calculate sample index for this hit
            const sampleIndex = Math.floor(hit.time / timeStep);
            if (sampleIndex >= 0 && sampleIndex < leftIR.length) {
                // Calculate total energy across all frequency bands
                const energyFactor = this.calculateTotalEnergy(hit.energies);
                
                // Calculate distance from hit point to listener
                const listenerPos = camera.getPosition();
                const dx = hit.position[0] - listenerPos[0];
                const dy = hit.position[1] - listenerPos[1];
                const dz = hit.position[2] - listenerPos[2];
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                
                // Apply inverse square law attenuation
                const distanceAttenuation = 1 / (4 * Math.PI * distance * distance);
                
                // Calculate directional gains for HRTF approximation
                const [leftGain, rightGain] = this.calculateSpatialGains(
                    hit.position, 
                    camera.getPosition(), 
                    camera.getFront(), 
                    camera.getRight()
                );
                
                // Add the impulse with appropriate amplitude
                const amplitude = energyFactor * distanceAttenuation;
                leftIR[sampleIndex] += amplitude * leftGain;
                rightIR[sampleIndex] += amplitude * rightGain;
            }
        }
        
        // Apply a gentle low-pass filter to smooth the impulse response
        this.applyLowPassFilter(leftIR);
        this.applyLowPassFilter(rightIR);
    }

    private calculateTotalEnergy(energies: FrequencyBands): number {
        // Sum up energy across all frequency bands
        return (
            energies.energy125Hz +
            energies.energy250Hz +
            energies.energy500Hz +
            energies.energy1kHz +
            energies.energy2kHz +
            energies.energy4kHz +
            energies.energy8kHz +
            energies.energy16kHz
        ) / 8; // Average across bands
    }

    private calculateSpatialGains(
        hitPosition: vec3,
        listenerPosition: vec3,
        listenerFront: vec3,
        listenerRight: vec3
    ): [number, number] {
        // Calculate vector from listener to hit point
        const toSource = vec3.create();
        vec3.subtract(toSource, hitPosition, listenerPosition);
        vec3.normalize(toSource, toSource);

        // Calculate azimuth angle (horizontal plane)
        const dotRight = vec3.dot(toSource, listenerRight);
        const dotFront = vec3.dot(toSource, listenerFront);
        const azimuth = Math.atan2(dotRight, dotFront);

        // Simple HRTF approximation based on azimuth
        // Positive azimuth = sound from right
        // Negative azimuth = sound from left
        const leftGain = 0.5 * (1 - Math.sin(azimuth));
        const rightGain = 0.5 * (1 + Math.sin(azimuth));

        return [leftGain, rightGain];
    }

    private applyLowPassFilter(buffer: Float32Array): void {
        const alpha = 0.1; // Smoothing factor (0-1), higher = more smoothing
        let lastValue = buffer[0];

        for (let i = 1; i < buffer.length; i++) {
            lastValue = buffer[i] = lastValue * (1 - alpha) + buffer[i] * alpha;
        }
    }

    private calculateRoomModes(room: Room): number[] {
        const { width, height, depth } = room.config.dimensions;
        const modes: number[] = [];
        const c = 343; // Speed of sound in m/s

        // Calculate axial modes (most significant)
        for (let l = 1; l <= 3; l++) {
            modes.push((c * l) / (2 * width));  // Width modes
            modes.push((c * l) / (2 * height)); // Height modes
            modes.push((c * l) / (2 * depth));  // Depth modes
        }

        return modes.sort((a, b) => a - b);
    }

    private addRoomModes(leftIR: Float32Array, rightIR: Float32Array, modes: number[]): void {
        // Use FFT to transform to frequency domain
        const leftFreq = this.fftForward(leftIR);
        const rightFreq = this.fftForward(rightIR);
        
        // Enhance amplitudes at room mode frequencies
        modes.forEach(freq => {
            const binIndex = Math.round(freq * leftIR.length / this.sampleRate);
            if (binIndex > 0 && binIndex < leftFreq.length / 2) {
                // Amplify this frequency bin
                leftFreq[binIndex] *= 1.5;  
                rightFreq[binIndex] *= 1.5;
                
                // Also slightly amplify neighboring bins for smoother response
                if (binIndex > 1) {
                    leftFreq[binIndex - 1] *= 1.2;
                    rightFreq[binIndex - 1] *= 1.2;
                }
                if (binIndex < leftFreq.length / 2 - 1) {
                    leftFreq[binIndex + 1] *= 1.2;
                    rightFreq[binIndex + 1] *= 1.2;
                }
            }
        });
        
        // Transform back to time domain
        this.fftInverse(leftFreq, leftIR);
        this.fftInverse(rightFreq, rightIR);
    }

    private fftForward(timeData: Float32Array): Float32Array {
        const size = this.nextPowerOf2(timeData.length);
        const real = new Float32Array(size);
        const imag = new Float32Array(size);
        
        // Copy input data and apply Hanning window
        for (let i = 0; i < timeData.length; i++) {
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (timeData.length - 1)));
            real[i] = timeData[i] * window;
        }
        
        // Perform FFT
        this.fft(real, imag, false);
        
        // Convert to magnitude spectrum
        const magnitudes = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }
        
        return magnitudes;
    }

    private fftInverse(freqData: Float32Array, timeData: Float32Array): void {
        const size = freqData.length;
        const real = new Float32Array(size);
        const imag = new Float32Array(size);
        
        // Convert magnitude spectrum back to complex form
        for (let i = 0; i < size; i++) {
            real[i] = freqData[i];
            imag[i] = 0;
        }
        
        // Perform inverse FFT
        this.fft(real, imag, true);
        
        // Copy result back to time domain buffer
        const scale = 1 / size;
        for (let i = 0; i < timeData.length; i++) {
            timeData[i] = real[i] * scale;
        }
    }

    private fft(real: Float32Array, imag: Float32Array, inverse: boolean): void {
        const n = real.length;
        
        // Bit reversal
        for (let i = 0; i < n; i++) {
            const j = this.reverseBits(i, Math.log2(n));
            if (j > i) {
                [real[i], real[j]] = [real[j], real[i]];
                [imag[i], imag[j]] = [imag[j], imag[i]];
            }
        }
        
        // Cooley-Tukey FFT
        for (let size = 2; size <= n; size *= 2) {
            const halfsize = size / 2;
            const tablestep = n / size;
            
            for (let i = 0; i < n; i += size) {
                for (let j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
                    const thetaT = (inverse ? 2 : -2) * Math.PI * k / n;
                    const cos = Math.cos(thetaT);
                    const sin = Math.sin(thetaT);
                    
                    const a_real = real[j + halfsize];
                    const a_imag = imag[j + halfsize];
                    const b_real = real[j];
                    const b_imag = imag[j];
                    
                    real[j + halfsize] = b_real - (cos * a_real - sin * a_imag);
                    imag[j + halfsize] = b_imag - (cos * a_imag + sin * a_real);
                    real[j] = b_real + (cos * a_real - sin * a_imag);
                    imag[j] = b_imag + (cos * a_imag + sin * a_real);
                }
            }
        }
    }

    private reverseBits(x: number, bits: number): number {
        let result = 0;
        for (let i = 0; i < bits; i++) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }

    private nextPowerOf2(n: number): number {
        return Math.pow(2, Math.ceil(Math.log2(n)));
    }

    // NEW METHOD: Calculate frequency-dependent energy factors
    private calculateFrequencyEnergyFactors(energies: FrequencyBands): {
        lowBand: number;
        midBand: number;
        highBand: number;
        average: number;
    } {
        const lowBand = (energies.energy125Hz + energies.energy250Hz) / 2;
        const midBand = (energies.energy500Hz + energies.energy1kHz) / 2;
        const highBand = (energies.energy2kHz + energies.energy4kHz + energies.energy8kHz + energies.energy16kHz) / 4;
        const average = (lowBand + midBand + highBand) / 3;
        
        return {
            lowBand,
            midBand,
            highBand,
            average
        };
    }
    
    // NEW METHOD: Calculate distance attenuation with bounce factor
    private calculateDistanceAttenuation(distance: number, bounces: number): number {
        const falloff = 1 / (distance * distance);
        const bounceScaling = Math.pow(0.7, bounces);
        
        return falloff * bounceScaling;
    }
    
    // NEW METHOD: Add temporal spreading for more natural reverberation
    private addTemporalSpreading(
        leftIR: Float32Array, 
        rightIR: Float32Array, 
        centerIndex: number, 
        baseAmplitude: number,
        leftGain: number,
        rightGain: number,
        bounces: number,
        distance: number
    ): void {
        const spreadingFactor = 0.1; // Adjust this value to control spreading amount
        const spreadingSize = Math.floor(leftIR.length * spreadingFactor);
        
        for (let i = -spreadingSize; i <= spreadingSize; i++) {
            const index = centerIndex + i;
            if (index >= 0 && index < leftIR.length) {
                const amplitude = baseAmplitude * Math.pow(0.7, Math.abs(i));
                leftIR[index] += amplitude * leftGain;
                rightIR[index] += amplitude * rightGain;
            }
        }
    }
    
    // NEW METHOD: Improved spatial gains using direction and distance
    private calculateImprovedSpatialGains(
        azimuth: number, 
        elevation: number, 
        distance: number,
        bounces: number
    ): [number, number] {
        const distanceFactor = 1 / (distance * distance);
        const bounceFactor = Math.pow(0.7, bounces);
        
        const leftGain = 0.5 * (1 - Math.sin(azimuth)) * distanceFactor * bounceFactor;
        const rightGain = 0.5 * (1 + Math.sin(azimuth)) * distanceFactor * bounceFactor;
        
        return [leftGain, rightGain];
    }
    
    // NEW METHOD: Simple smoothing for IR
    private smoothImpulseResponse(ir: Float32Array): void {
        const alpha = 0.1; // Smoothing factor (0-1), higher = more smoothing
        let lastValue = ir[0];

        for (let i = 1; i < ir.length; i++) {
            lastValue = ir[i] = lastValue * (1 - alpha) + ir[i] * alpha;
        }
    }

    public async processSpatialAudio(
        camera: Camera,
        rayHits: RayHit[],
        params: any,
        room: Room
    ): Promise<[Float32Array, Float32Array]> {
        // Wait for initialization to complete first
        if (!this.initialized) {
            await this.initializationPromise;
        }

        if (rayHits.length === 0) {
            console.warn('No ray hits to process');
            return [new Float32Array(0), new Float32Array(0)];
        }

        // Update room acoustics parameters before processing
        this.updateRoomAcousticsBuffer(room);

        // Create IR buffers
        const irLength = Math.ceil(this.sampleRate * 2); // 2 seconds of IR
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);

        // Generate basic impulse response
        this.generateImpulseResponse(leftIR, rightIR, rayHits, camera);

        // Calculate and apply room modes
        const modes = this.calculateRoomModes(room);
        this.addRoomModes(leftIR, rightIR, modes);

        return [leftIR, rightIR];
    }
}