import { Camera } from '../camera/camera';
import { SpatialAudioProcessor, RayHit as SpatialRayHit } from './spatial-audio-processor';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { vec3 } from 'gl-matrix';
import { FrequencyBands } from '../raytracer/ray';

// Use the RayHit interface from spatial-audio-processor
export type RayHit = SpatialRayHit;

interface RoomMode {
    frequency: number;
    decayTime: number;
    rt60: number;
}

export class AudioProcessor {
    private audioCtx: AudioContext;
    private sampleRate: number;
    private impulseResponseBuffer: AudioBuffer | null;
    private lastImpulseData: Float32Array | null;
    private spatialProcessor: SpatialAudioProcessor;
    private room: Room;
    private lastRayHits: RayHit[] | null = null;

    constructor(device: GPUDevice, room: Room, sampleRate: number = 44100) {
        this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.sampleRate = this.audioCtx.sampleRate || sampleRate;
        this.impulseResponseBuffer = null;
        this.lastImpulseData = null;
        this.spatialProcessor = new SpatialAudioProcessor(device, this.sampleRate);
        this.room = room;
    }

    public normalizeAndApplyEnvelope(leftIR: Float32Array, rightIR: Float32Array, envelope: Float32Array): void {
        let maxAmplitude = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxAmplitude = Math.max(maxAmplitude, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }

        if (maxAmplitude > 0) {
            // Target a higher amplitude level (0.7 instead of normalization to 1.0)
            const targetAmplitude = 0.7;
            const normalizationFactor = targetAmplitude / maxAmplitude;
            
            for (let i = 0; i < leftIR.length; i++) {
                // Use a more gentle envelope (square root makes decay more gradual)
                const gentleEnvelope = Math.sqrt(envelope[i]);
                leftIR[i] = leftIR[i] * normalizationFactor * gentleEnvelope;
                rightIR[i] = rightIR[i] * normalizationFactor * gentleEnvelope;
            }
            
            console.log(`Normalized IR with factor: ${normalizationFactor}, max amplitude was: ${maxAmplitude}`);
        }
    }

    async processRayHits(
        rayHits: RayHit[],
        camera: Camera,
        maxTime: number = 0.5,
        params = {
            speedOfSound: 343,
            maxDistance: 20,
            minDistance: 1,
            temperature: 20,
            humidity: 50,
            sourcePower: 0
        }
    ): Promise<void> {
        try {
            // Store the ray hits for later use in decay curve calculation
            this.lastRayHits = rayHits;

            console.log(`Processing ${rayHits.length} ray hits for IR calculation`);
            
            // Sort ray hits by time for proper processing
            const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);

            // Process spatial audio in chunks if there are too many hits
            const chunkSize = 1000;
            const chunks = [];
            for (let i = 0; i < sortedHits.length; i += chunkSize) {
                const chunk = sortedHits.slice(i, i + chunkSize);
                chunks.push(chunk);
            }

            console.log(`Processing ${chunks.length} chunks of ray hits`);

            let leftIR: Float32Array, rightIR: Float32Array;
            if (chunks.length > 1) {
                // Process chunks and combine results
                const results = await Promise.all(chunks.map(chunk =>
                    this.spatialProcessor.processSpatialAudio(camera, chunk, params, this.room)
                ));

                // Combine results
                const maxLength = Math.max(...results.map(([left]) => left.length));
                leftIR = new Float32Array(maxLength);
                rightIR = new Float32Array(maxLength);

                results.forEach(([left, right]) => {
                    for (let i = 0; i < left.length; i++) {
                        leftIR[i] += left[i] || 0;
                        rightIR[i] += right[i] || 0;
                    }
                });
            } else {
                [leftIR, rightIR] = await this.spatialProcessor.processSpatialAudio(
                    camera,
                    sortedHits,
                    params,
                    this.room
                );
            }

            console.log(`Generated initial IR buffers: L=${leftIR.length}, R=${rightIR.length}`);

            if (!this.validateIRBuffers(leftIR, rightIR)) {
                throw new Error('Invalid IR buffers generated');
            }

            const sampleCount = leftIR.length;
            const envelope = this.generateEnvelope(sampleCount, sortedHits);
            const roomModes = this.calculateRoomModes();

            console.log('Applying room acoustics processing...');
            this.addRoomModes(leftIR, rightIR, roomModes);
            this.applyWaveInterference(leftIR, rightIR, sortedHits);
            this.normalizeAndApplyEnvelope(leftIR, rightIR, envelope);

            if (!this.validateIRBuffers(leftIR, rightIR)) {
                throw new Error('Invalid IR buffers after acoustic processing');
            }

            this.setupImpulseResponseBuffer(leftIR, rightIR);

            // Create interleaved array with both channels
            this.lastImpulseData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                this.lastImpulseData[i * 2] = leftIR[i] || 0;
                this.lastImpulseData[i * 2 + 1] = rightIR[i] || 0;
            }

            console.log("Impulse response processed successfully");
        } catch (error) {
            console.error("Error processing ray hits:", error);
            throw error;
        }
    }

    private applyWaveInterference(leftIR: Float32Array, rightIR: Float32Array, rayHits: RayHit[]): void {
        const timeStep = 1 / this.sampleRate;

        for (let i = 0; i < leftIR.length; i++) {
            const currentTime = i * timeStep;
            let leftSum = 0;
            let rightSum = 0;

            for (const hit of rayHits) {
                if (hit.time <= currentTime) {
                    // Emphasize early reflections
                    const arrivalTime = hit.time;
                    const isEarlyReflection = arrivalTime < 0.1; // First 100ms
                    
                    // Stronger emphasis on early reflections
                    const amplitudeScale = isEarlyReflection ? 
                        3.0 * Math.exp(-arrivalTime * 5) : // Early reflections decay
                        0.7 * Math.exp(-arrivalTime * 2);  // Late reflections decay

                    const frequency = Math.max(hit.frequency || 440, 20);
                    const dopplerShift = Math.max(hit.dopplerShift || 1, 0.1);
                    const phase = hit.phase || 0;

                    const timeSinceArrival = Math.max(currentTime - hit.time, 0);
                    const instantPhase = phase +
                        2 * Math.PI * frequency * (1 + dopplerShift) * timeSinceArrival;

                    // Calculate amplitude using weighted contributions from all frequency bands
                    const frequencyWeights = {
                        energy125Hz: 0.7,  // Bass frequencies (more weight)
                        energy250Hz: 0.8,
                        energy500Hz: 0.9,
                        energy1kHz: 1.0,   // Mid frequencies (full weight)
                        energy2kHz: 0.95,
                        energy4kHz: 0.9,
                        energy8kHz: 0.85,
                        energy16kHz: 0.8   // High frequencies (less weight due to air absorption)
                    };

                    // Calculate weighted energy
                    let totalEnergy = 0;
                    let totalWeight = 0;
                    for (const [band, weight] of Object.entries(frequencyWeights)) {
                        if (hit.energies && hit.energies[band]) {
                            totalEnergy += hit.energies[band] * weight;
                            totalWeight += weight;
                        }
                    }

                    // Normalize energy
                    const normalizedEnergy = totalWeight > 0 ? totalEnergy / totalWeight : 0;
                    
                    // Calculate final amplitude with distance attenuation and scaling
                    const amplitude = Math.sqrt(normalizedEnergy) / (4 * Math.PI * hit.distance);
                    const contribution = amplitude * amplitudeScale * Math.sin(instantPhase);

                    leftSum += contribution;
                    rightSum += contribution;
                }
            }

            leftIR[i] = leftSum;
            rightIR[i] = rightSum;
        }
    }

    private calculateSpatialGains(position: vec3, listenerPos: vec3, listenerForward: vec3, listenerRight: vec3): [number, number] {
        // Calculate direction vector from listener to sound source
        const direction = vec3.create();
        vec3.subtract(direction, position, listenerPos);
        vec3.normalize(direction, direction);
        
        // Calculate azimuth angle (horizontal plane)
        const dot = vec3.dot(direction, listenerRight);
        const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
        
        // More realistic HRTF approximation
        const leftGain = 0.5 + 0.5 * Math.cos(angle); 
        const rightGain = 0.5 + 0.5 * Math.cos(Math.PI - angle);
        
        return [leftGain, rightGain];
    }

    private calculateRT60(rayHits: RayHit[]): number {
        if (!rayHits.length) return 0.5; // Default RT60

        // Find the time when energy drops by 60dB
        const initialEnergy = rayHits[0].energy;
        for (const hit of rayHits) {
            if (hit.energy <= initialEnergy * 0.001) { // -60dB = 10^(-60/20) ≈ 0.001
                return hit.time;
            }
        }

        // If we don't find a 60dB drop, estimate based on last hit
        return Math.max(0.5, rayHits[rayHits.length - 1].time);
    }

    private generateEnvelope(sampleCount: number, rayHits: RayHit[]): Float32Array {
        const envelope = new Float32Array(sampleCount);
        const rt60 = Math.max(0.5, this.calculateRT60(rayHits)); // Minimum 0.5s reverb
        
        // Define key time points
        const directSoundTime = 0.005; // 5ms
        const earlyReflectionsEnd = 0.08; // 80ms
        const directSoundSamples = Math.floor(directSoundTime * this.sampleRate);
        const earlyReflectionSamples = Math.floor(earlyReflectionsEnd * this.sampleRate);
        
        for (let i = 0; i < sampleCount; i++) {
            const t = i / this.sampleRate;
            
            if (i < directSoundSamples) {
                // Direct sound (full strength)
                envelope[i] = 1.0;
            } else if (i < earlyReflectionSamples) {
                // Early reflections (gentle decay)
                const normalizedPos = (i - directSoundSamples) / (earlyReflectionSamples - directSoundSamples);
                envelope[i] = 0.9 - 0.2 * normalizedPos;
            } else {
                // Late reflections (exponential decay)
                const lateTime = t - earlyReflectionsEnd;
                envelope[i] = 0.7 * Math.exp(-6.91 * lateTime / rt60);
            }
        }
        
        return envelope;
    }

    private calculateAverageAbsorption(): number {
        const materials = this.room.config.materials;
        let totalAbsorption = 0;
        let count = 0;

        // Calculate average absorption across all materials and frequencies
        for (const key of ['left', 'right', 'floor', 'ceiling', 'front', 'back']) {
            const material = materials[key];
            if (material) {
                // Ensure we have valid absorption values
                const absorptions = [
                    isFinite(material.absorption125Hz) ? material.absorption125Hz : 0.1,
                    isFinite(material.absorption250Hz) ? material.absorption250Hz : 0.1,
                    isFinite(material.absorption500Hz) ? material.absorption500Hz : 0.1,
                    isFinite(material.absorption1kHz) ? material.absorption1kHz : 0.1,
                    isFinite(material.absorption2kHz) ? material.absorption2kHz : 0.1,
                    isFinite(material.absorption4kHz) ? material.absorption4kHz : 0.1,
                    isFinite(material.absorption8kHz) ? material.absorption8kHz : 0.1,
                    isFinite(material.absorption16kHz) ? material.absorption16kHz : 0.1
                ];

                // Weight mid-frequencies more heavily for room modes
                const weights = [0.5, 0.7, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3];
                
                for (let i = 0; i < absorptions.length; i++) {
                    totalAbsorption += absorptions[i] * weights[i];
                    count += weights[i];
                }
            }
        }

        // Return average, defaulting to 0.1 if no valid materials
        return count > 0 ? totalAbsorption / count : 0.1;
    }

    private calculateMode(l: number, m: number, n: number): number {
        const dimensions = this.room.config.dimensions;
        const c = 343; // Speed of sound in m/s
        
        // Ensure dimensions are valid
        const width = Math.max(dimensions.width, 0.1);
        const height = Math.max(dimensions.height, 0.1);
        const depth = Math.max(dimensions.depth, 0.1);

        // Calculate mode frequency
        const freq = (c/2) * Math.sqrt(
            Math.pow(l/width, 2) + 
            Math.pow(m/height, 2) + 
            Math.pow(n/depth, 2)
        );

        // Calculate mode amplitude based on room absorption
        const avgAbsorption = this.calculateAverageAbsorption();
        const modeAmplitude = 1.0 - avgAbsorption;

        return freq * modeAmplitude;
    }

    private calculateRoomModes(): number[] {
        const modes: number[] = [];
        
        // Calculate first few axial modes (most significant)
        for (let l = 0; l <= 2; l++) {
            for (let m = 0; m <= 2; m++) {
                for (let n = 0; n <= 2; n++) {
                    if (l + m + n > 0) { // Skip (0,0,0)
                        const modeFreq = this.calculateMode(l, m, n);
                        if (modeFreq >= 20 && modeFreq <= 20000) { // Only include audible frequencies
                            modes.push(modeFreq);
                        }
                    }
                }
            }
        }

        return modes.sort((a, b) => a - b);
    }

    private addRoomModes(leftIR: Float32Array, rightIR: Float32Array, modes: number[]): void {
        const modeAmplitude = 0.1;

        modes.forEach(mode => {
            const freq = mode;
            const decay = Math.exp(-3 * 0.5 / 0.5);

            for (let t = 0; t < leftIR.length; t++) {
                const sample = modeAmplitude * decay *
                    Math.sin(2 * Math.PI * freq * t / this.sampleRate);
                leftIR[t] += sample;
                rightIR[t] += sample;
            }
        });
    }

    private validateIRBuffers(leftIR: Float32Array, rightIR: Float32Array): boolean {
        if (!leftIR || !rightIR || leftIR.length === 0 || rightIR.length === 0) {
            console.error('Empty IR buffers');
            return false;
        }

        if (leftIR.length !== rightIR.length) {
            console.error('IR buffer length mismatch:', leftIR.length, rightIR.length);
            return false;
        }

        // Check for invalid values
        const hasInvalidValues = (buffer: Float32Array) => {
            for (let i = 0; i < buffer.length; i++) {
                if (isNaN(buffer[i]) || !isFinite(buffer[i])) {
                    console.error(`Invalid value at index ${i}:`, buffer[i]);
                    return true;
                }
            }
            return false;
        };

        if (hasInvalidValues(leftIR) || hasInvalidValues(rightIR)) {
            return false;
        }

        return true;
    }

    private setupImpulseResponseBuffer(leftIR: Float32Array, rightIR: Float32Array): void {
        const length = leftIR.length;
        const decayCurve = this.calculateDecayCurve(length);
        
        this.impulseResponseBuffer = this.audioCtx.createBuffer(2, length, this.audioCtx.sampleRate);
        
        // Apply decay to both channels
        const leftChannel = this.impulseResponseBuffer.getChannelData(0);
        const rightChannel = this.impulseResponseBuffer.getChannelData(1);
        
        for (let i = 0; i < length; i++) {
            leftChannel[i] = leftIR[i] * decayCurve[i];
            rightChannel[i] = rightIR[i] * decayCurve[i];
        }

        // Normalize to prevent clipping
        this.normalizeBuffer(this.impulseResponseBuffer);
    }

    private normalizeBuffer(buffer: AudioBuffer): void {
        const leftChannel = buffer.getChannelData(0);
        const rightChannel = buffer.getChannelData(1);
        
        // Find peak amplitude
        let maxAmplitude = 0;
        for (let i = 0; i < buffer.length; i++) {
            maxAmplitude = Math.max(maxAmplitude, 
                Math.abs(leftChannel[i]), 
                Math.abs(rightChannel[i]));
        }
        
        // Normalize if needed
        if (maxAmplitude > 1.0) {
            const scalar = 0.99 / maxAmplitude;
            for (let i = 0; i < buffer.length; i++) {
                leftChannel[i] *= scalar;
                rightChannel[i] *= scalar;
            }
        }
    }

    /**
     * Returns the last impulse response data as a Float32Array.
     */
    public getImpulseResponseData(): Float32Array | null {
        return this.lastImpulseData;
    }

    public async visualizeImpulseResponse(renderer: WaveformRenderer): Promise<void> {
        if (this.lastImpulseData) {
            await renderer.drawWaveformWithFFT(this.lastImpulseData);
        }
    }

    public async debugPlaySineWave(): Promise<void> {
        try {
            const duration = 2; // Duration in seconds
            const frequency = 440; // A4 note
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;
            
            const buffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = buffer.getChannelData(0);
            
            for (let i = 0; i < samples; i++) {
                channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
            }
            
            const source = this.audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(this.audioCtx.destination);
            source.start();
        } catch (error) {
            console.error("Error playing sine wave:", error);
        }
    }

    public async playConvolvedSineWave(dryWetMix = 1): Promise<void> {
        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer available');
            return;
        }

        try {
            // Create sine wave with decay
            const duration = 2;
            const frequency = 440;
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;
            
            const sineBuffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = sineBuffer.getChannelData(0);
            
            // Simple envelope for the source sound (attack and release)
            const attackTime = 0.01; // seconds
            const releaseTime = 0.05; // seconds
            const attackSamples = Math.floor(attackTime * sampleRate);
            const releaseSamples = Math.floor(releaseTime * sampleRate);
            
            // Apply sine wave with envelope
            for (let i = 0; i < samples; i++) {
                let amplitude = 1.0;
                
                // Apply attack
                if (i < attackSamples) {
                    amplitude = i / attackSamples;
                }
                
                // Apply release
                if (i > samples - releaseSamples) {
                    amplitude = (samples - i) / releaseSamples;
                }
                
                channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * amplitude;
            }

            // Create the audio graph with both dry and wet paths
            const source = this.audioCtx.createBufferSource();
            source.buffer = sineBuffer;
            
            // Set up wet path (through convolver)
            const convolver = this.audioCtx.createConvolver();
            convolver.buffer = this.impulseResponseBuffer;
            const wetGain = this.audioCtx.createGain();
            wetGain.gain.value = dryWetMix; // Wet level (0.7 = 70% wet)
            
            // Set up dry path (direct sound)
            const dryGain = this.audioCtx.createGain();
            dryGain.gain.value = 1 - dryWetMix; // Dry level (0.3 = 30% dry)
            
            // Create master gain for overall volume
            const masterGain = this.audioCtx.createGain();
            masterGain.gain.value = 0.3; // Overall volume
            
            // Connect everything
            source.connect(convolver);    // Wet path: source → convolver
            convolver.connect(wetGain);   // Wet path: convolver → wet gain
            source.connect(dryGain);      // Dry path: source → dry gain
            
            wetGain.connect(masterGain);  // Both paths connect to master
            dryGain.connect(masterGain);
            
            masterGain.connect(this.audioCtx.destination);
            
            // Start playback
            source.start();
            
            // Schedule the source to stop after duration
            source.stop(this.audioCtx.currentTime + duration);
        } catch (error) {
            console.error("Error playing convolved sine wave:", error);
        }
    }

    public async playNoiseWithIR(): Promise<void> {
        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer available');
            return;
        }

        try {
            // Create white noise with decay based on ray energy
            const duration = 2;
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;
            
            const noiseBuffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = noiseBuffer.getChannelData(0);
            
            // Calculate decay curve based on ray energy
            const decayFactor = this.calculateDecayCurve(samples);
            
            // Apply noise with decay
            for (let i = 0; i < samples; i++) {
                channelData[i] = (Math.random() * 2 - 1) * decayFactor[i];
            }

            // Create convolver
            const convolver = this.audioCtx.createConvolver();
            convolver.buffer = this.impulseResponseBuffer;

            // Create gain to control volume
            const gainNode = this.audioCtx.createGain();
            gainNode.gain.value = 0.1;

            // Create source and connect
            const source = this.audioCtx.createBufferSource();
            source.buffer = noiseBuffer;
            source.connect(convolver);
            convolver.connect(gainNode);
            gainNode.connect(this.audioCtx.destination);
            source.start();
        } catch (error) {
            console.error("Error playing noise with IR:", error);
        }
    }

    private calculateDecayCurve(numSamples: number): Float32Array {
        const decayCurve = new Float32Array(numSamples);
        const sampleRate = this.audioCtx.sampleRate;
        
        if (!this.lastRayHits || this.lastRayHits.length === 0) {
            console.warn('No ray hits available for decay curve calculation');
            return decayCurve.fill(1.0); // Return flat curve instead of zeros
        }

        // Sort ray hits by time
        const sortedHits = [...this.lastRayHits].sort((a, b) => a.time - b.time);
        
        // Find the time range of hits
        const startTime = sortedHits[0].time;
        const endTime = sortedHits[sortedHits.length - 1].time;
        const timeRange = endTime - startTime;

        console.log(`Decay curve time range: ${timeRange}s, Hits: ${sortedHits.length}`);

        // Create time bins for energy accumulation
        const numBins = Math.min(200, numSamples); // More bins for better resolution
        const binDuration = timeRange / numBins;
        const energyBins = new Array(numBins).fill(0);
        
        // Accumulate energy in bins
        let maxBinEnergy = 0;
        for (const hit of sortedHits) {
            const binIndex = Math.floor((hit.time - startTime) / binDuration);
            if (binIndex >= 0 && binIndex < numBins) {
                // Safely access energies with validation
                const energies = hit.energies || {};
                
                // Sum energy across frequency bands
                const totalEnergy = 
                    (energies.energy125Hz || 0) * 0.7 +
                    (energies.energy250Hz || 0) * 0.8 +
                    (energies.energy500Hz || 0) * 0.9 +
                    (energies.energy1kHz || 0) * 1.0 +
                    (energies.energy2kHz || 0) * 0.95 +
                    (energies.energy4kHz || 0) * 0.9 +
                    (energies.energy8kHz || 0) * 0.85 +
                    (energies.energy16kHz || 0) * 0.8;
                
                energyBins[binIndex] += totalEnergy;
                maxBinEnergy = Math.max(maxBinEnergy, energyBins[binIndex]);
            }
        }

        console.log(`Max bin energy: ${maxBinEnergy}`);

        // Normalize energy bins
        if (maxBinEnergy > 0) {
            for (let i = 0; i < numBins; i++) {
                energyBins[i] = Math.sqrt(energyBins[i] / maxBinEnergy); // Square root for smoother decay
            }
        }

        // Create smooth decay curve from energy bins
        for (let i = 0; i < numSamples; i++) {
            const timeInSeconds = i / sampleRate;
            const relativeBinPosition = (timeInSeconds - startTime) / binDuration;
            const binIndex = Math.floor(relativeBinPosition);
            
            if (binIndex < 0) {
                decayCurve[i] = 1.0; // Before first reflection
            } else if (binIndex >= numBins) {
                decayCurve[i] = energyBins[numBins - 1] || 0; // Use last bin's energy
            } else {
                // Interpolate between bins for smoother curve
                const nextBin = Math.min(binIndex + 1, numBins - 1);
                const fraction = relativeBinPosition - binIndex;
                decayCurve[i] = energyBins[binIndex] * (1 - fraction) + 
                               energyBins[nextBin] * fraction;
            }
        }

        // Ensure the curve starts at 1.0 and smoothly transitions
        const rampSamples = Math.min(100, numSamples);
        for (let i = 0; i < rampSamples; i++) {
            const rampFactor = i / rampSamples;
            decayCurve[i] = 1.0 * (1 - rampFactor) + decayCurve[i] * rampFactor;
        }

        // Add some minimum energy to prevent complete silence
        const minimumLevel = 0.001;
        for (let i = 0; i < numSamples; i++) {
            decayCurve[i] = Math.max(decayCurve[i], minimumLevel);
        }

        console.log(`Decay curve range: ${Math.min(...decayCurve)} to ${Math.max(...decayCurve)}`);
        return decayCurve;
    }

    // You might also want to add a method to clear the ray hits when needed
    public clearRayHits(): void {
        this.lastRayHits = null;
    }

    public generateMultibandImpulseResponse(rayHits: RayHit[], responseLength: number): Float32Array[] {
        // Create an array of 8 impulse responses, one per frequency band
        const impulseResponses: Float32Array[] = [];
        
        for (let band = 0; band < 8; band++) {
            const ir = new Float32Array(responseLength);
            impulseResponses.push(ir);
        }
        
        // Get sample rate from audio context
        const sampleRate = this.audioCtx.sampleRate;
        
        // Process each ray hit
        rayHits.forEach(hit => {
            // Calculate sample index based on time
            const sampleIndex = Math.floor(hit.time * sampleRate);
            
            // Skip if out of range
            if (sampleIndex < 0 || sampleIndex >= responseLength) return;
            
            // Extract energies for all bands
            const energies = [
                hit.energy125Hz,
                hit.energy250Hz,
                hit.energy500Hz,
                hit.energy1kHz,
                hit.energy2kHz,
                hit.energy4kHz,
                hit.energy8kHz,
                hit.energy16kHz
            ];
            
            // Add energy to the appropriate sample in each band's IR
            for (let band = 0; band < 8; band++) {
                impulseResponses[band][sampleIndex] += energies[band];
            }
        });
        
        // Normalize each impulse response
        for (let band = 0; band < 8; band++) {
            this.normalizeImpulseResponse(impulseResponses[band]);
        }
        
        return impulseResponses;
    }

    // Helper function to normalize an impulse response
    private normalizeImpulseResponse(ir: Float32Array): void {
        // Find the maximum absolute value
        let maxAbs = 0;
        for (let i = 0; i < ir.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(ir[i]));
        }
        
        // Normalize if non-zero
        if (maxAbs > 0) {
            const scale = 0.95 / maxAbs; // Leave a bit of headroom
            for (let i = 0; i < ir.length; i++) {
                ir[i] *= scale;
            }
        }
    }

    async applyMultibandConvolution(audioBuffer: AudioBuffer, impulseResponses: Float32Array[]): Promise<AudioBuffer> {
        const context = this.audioCtx;
        const sampleRate = context.sampleRate;
        const channels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        
        // Create band-pass filters for each frequency band
        const filterBands = [
            { lowFreq: 88, highFreq: 177 },   // 125Hz band
            { lowFreq: 177, highFreq: 354 },  // 250Hz band
            { lowFreq: 354, highFreq: 707 },  // 500Hz band
            { lowFreq: 707, highFreq: 1414 }, // 1kHz band
            { lowFreq: 1414, highFreq: 2828 },// 2kHz band
            { lowFreq: 2828, highFreq: 5657 },// 4kHz band
            { lowFreq: 5657, highFreq: 11314 },// 8kHz band
            { lowFreq: 11314, highFreq: 20000 }// 16kHz band
        ];
        
        // Create 8 separate convolver nodes
        const convolvers = impulseResponses.map(ir => {
            const convolver = context.createConvolver();
            const irBuffer = context.createBuffer(1, ir.length, sampleRate);
            irBuffer.getChannelData(0).set(ir);
            convolver.buffer = irBuffer;
            return convolver;
        });
        
        // Process audio through band filters and convolvers
        const offlineContext = new OfflineAudioContext(channels, length, sampleRate);
        
        // Split input audio into frequency bands
        const bandInputs = filterBands.map((band, i) => {
            const input = offlineContext.createBufferSource();
            input.buffer = audioBuffer;
            
            // Create band-pass filter
            const filter = offlineContext.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.value = Math.sqrt(band.lowFreq * band.highFreq);
            filter.Q.value = 1.0;
            
            input.connect(filter);
            return filter;
        });
        
        // Connect each band to its convolver
        const merger = offlineContext.createChannelMerger(8);
        bandInputs.forEach((filter, i) => {
            // Create offline convolver
            const convolver = offlineContext.createConvolver();
            const irBuffer = offlineContext.createBuffer(1, impulseResponses[i].length, sampleRate);
            irBuffer.getChannelData(0).set(impulseResponses[i]);
            convolver.buffer = irBuffer;
            
            // Connect filter → convolver → merger
            filter.connect(convolver);
            convolver.connect(merger);
        });
        
        // Connect merger to destination
        merger.connect(offlineContext.destination);
        
        // Start all sources
        bandInputs.forEach(filter => {
            const source = filter.context.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(filter);
            source.start();
        });
        
        // Render and return
        return await offlineContext.startRendering();
    }
}