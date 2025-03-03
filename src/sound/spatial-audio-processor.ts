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
    }

    private async initializeAsync(): Promise<void> {
        await this.createPipeline();
        this.initialized = true;
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
        
        // Ensure valid room dimensions
        const width = Math.max(dimensions.width, 0.1);  // Minimum 10cm
        const height = Math.max(dimensions.height, 0.1);
        const depth = Math.max(dimensions.depth, 0.1);
        
        const volume = width * height * depth;
        const surfaceArea = 2 * (
            width * height +
            width * depth +
            height * depth
        );
        
        // Initialize absorption values
        let avg125Hz = 0, avg250Hz = 0, avg500Hz = 0, avg1kHz = 0, 
            avg2kHz = 0, avg4kHz = 0, avg8kHz = 0, avg16kHz = 0;
        let validSurfaces = 0;
        
        // List all surface types
        const materialKeys = ['left', 'right', 'floor', 'ceiling', 'front', 'back'];
        
        // Sum up absorption values for each surface
        for (const key of materialKeys) {
            const material = materials[key];
            if (material) {
                // Debug log this material's properties
                console.log(`Material ${key}:`, {
                    a125: material.absorption125Hz,
                    a1k: material.absorption1kHz
                });
                
                // Ensure we have valid values (default to 0.1 if not)
                const a125 = isFinite(material.absorption125Hz) ? material.absorption125Hz : 0.1;
                const a250 = isFinite(material.absorption250Hz) ? material.absorption250Hz : 0.1;
                const a500 = isFinite(material.absorption500Hz) ? material.absorption500Hz : 0.1;
                const a1k = isFinite(material.absorption1kHz) ? material.absorption1kHz : 0.1;
                const a2k = isFinite(material.absorption2kHz) ? material.absorption2kHz : 0.1;
                const a4k = isFinite(material.absorption4kHz) ? material.absorption4kHz : 0.1;
                const a8k = isFinite(material.absorption8kHz) ? material.absorption8kHz : 0.1;
                const a16k = isFinite(material.absorption16kHz) ? material.absorption16kHz : 0.1;
                
                avg125Hz += a125;
                avg250Hz += a250;
                avg500Hz += a500;
                avg1kHz += a1k;
                avg2kHz += a2k;
                avg4kHz += a4k;
                avg8kHz += a8k;
                avg16kHz += a16k;
                
                validSurfaces++;
            }
        }
        
        // Use default values if no valid surfaces found
        if (validSurfaces === 0) {
            console.warn("No valid surfaces found, using default absorption");
            avg125Hz = 0.1;
            avg250Hz = 0.1;
            avg500Hz = 0.1;
            avg1kHz = 0.1;
            avg2kHz = 0.1;
            avg4kHz = 0.1;
            avg8kHz = 0.1;
            avg16kHz = 0.1;
            validSurfaces = 1;
        }
        
        // Average by number of valid surfaces
        avg125Hz /= validSurfaces;
        avg250Hz /= validSurfaces;
        avg500Hz /= validSurfaces;
        avg1kHz /= validSurfaces;
        avg2kHz /= validSurfaces;
        avg4kHz /= validSurfaces;
        avg8kHz /= validSurfaces;
        avg16kHz /= validSurfaces;
        
        // Calculate RT60 values using Sabine formula with safety checks
        const calcRT60 = (absorption: number) => {
            // Ensure absorption is non-zero and between 0.01 and 0.99
            const safeAbsorption = Math.max(0.01, Math.min(0.99, absorption));
            const rt60 = 0.161 * volume / (safeAbsorption * surfaceArea);
            
            // Clamp to reasonable RT60 range (0.1s to 10s)
            return Math.max(0.1, Math.min(10.0, rt60));
        };
        
        const rt60_125 = calcRT60(avg125Hz);
        const rt60_250 = calcRT60(avg250Hz);
        const rt60_500 = calcRT60(avg500Hz);
        const rt60_1k = calcRT60(avg1kHz);
        const rt60_2k = calcRT60(avg2kHz);
        const rt60_4k = calcRT60(avg4kHz);
        const rt60_8k = calcRT60(avg8kHz);
        const rt60_16k = calcRT60(avg16kHz);
        
        // Create buffer data with RT60 values for each frequency band
        const acousticsData = new Float32Array([
            // RT60 values for each frequency band
            rt60_125, rt60_250, rt60_500, rt60_1k, rt60_2k, rt60_4k, rt60_8k, rt60_16k,
            
            // Air absorption coefficients
            0.0002, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.011, 0.015,
            
            // Scattering coefficients
            0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            
            // Room characteristics
            0.08, // earlyReflectionTime (80ms)
            volume,
            surfaceArea
        ]);
        
        // Write to the GPU buffer
        this.device.queue.writeBuffer(this.acousticsBuffer, 0, acousticsData);
        
        // Log the data sent to the GPU for debugging
        console.log("Updated room acoustics:", {
            rt60: {
                "125Hz": rt60_125, 
                "1kHz": rt60_1k, 
                "4kHz": rt60_4k, 
                "16kHz": rt60_16k
            },
            surfaceArea,
            volume
        });
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