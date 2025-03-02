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
            for (let i = 0; i < leftIR.length; i++) {
                leftIR[i] = (leftIR[i] / maxAmplitude) * envelope[i];
                rightIR[i] = (rightIR[i] / maxAmplitude) * envelope[i];
            }
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
            const roomModes = this.calculateRoomModes(this.room.config.dimensions);

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

                    try {
                        let totalEnergy = 0;
                        let weightedSum = 0;
                        let validBands = 0;

                        // Safely process energy bands with validation
                        for (const [band, energy] of Object.entries(hit.energies)) {
                            if (typeof energy === 'number' && isFinite(energy) && energy >= 0) {
                                const weight = frequencyWeights[band as keyof typeof frequencyWeights];
                                if (typeof weight === 'number') {
                                    weightedSum += energy * weight;
                                    totalEnergy += energy;
                                    validBands++;
                                }
                            }
                        }

                        if (validBands > 0 && totalEnergy > 0) {
                            // Normalize the amplitude to prevent overflow
                            const amplitude = Math.min(1.0, Math.sqrt(weightedSum / validBands));
                            const contribution = amplitude * Math.sin(instantPhase);
                            const position = hit.position || [0, 0, 0];
                            const [leftGain, rightGain] = this.calculateSpatialGains(position);

                            if (!isNaN(contribution) && isFinite(contribution)) {
                                // Scale down contributions to prevent accumulation overflow
                                const scale = 1.0 / Math.sqrt(rayHits.length);
                                leftSum += contribution * leftGain * scale;
                                rightSum += contribution * rightGain * scale;
                            }
                        }
                    } catch (error) {
                        console.warn('Error processing ray hit:', error);
                        continue;
                    }
                }
            }

            leftIR[i] = isFinite(leftSum) ? leftSum : 0;
            rightIR[i] = isFinite(rightSum) ? rightSum : 0;
        }
    }

    private calculateSpatialGains(position: vec3): [number, number] {
        const x = position[0];
        const maxPan = 0.8;
        const pan = Math.max(-maxPan, Math.min(maxPan, x / 5));

        const leftGain = Math.cos((pan + 1) * Math.PI / 4);
        const rightGain = Math.sin((pan + 1) * Math.PI / 4);

        return [leftGain, rightGain];
    }

    private calculateRT60(rayHits: RayHit[]): number {
        if (rayHits.length === 0) {
            return 1.0;
        }

        const sortedHits = [...rayHits].sort((a, b) => a.time - b.time);
        const times: number[] = [];
        const energies: number[] = [];
        let totalEnergy = 0;

        sortedHits.forEach(hit => {
            times.push(hit.time);
            const energyValues = Object.values(hit.energies);
            const avgEnergy = energyValues.reduce((sum, energy) => sum + energy, 0) / energyValues.length;
            totalEnergy += avgEnergy;
            energies.push(totalEnergy);
        });

        const maxEnergy = Math.max(...energies);
        const normalizedEnergies = energies.map(e => e / maxEnergy);

        let rt60Time = times[times.length - 1];
        for (let i = 0; i < normalizedEnergies.length; i++) {
            if (normalizedEnergies[i] <= 0.001) {
                rt60Time = times[i];
                break;
            }
        }

        const volume = this.room.getVolume();
        const surfaceArea = this.room.getSurfaceArea();
        const avgAbsorption = this.calculateAverageAbsorption();
        const sabineRT60 = 0.161 * volume / (avgAbsorption * surfaceArea);

        return (rt60Time + sabineRT60) / 2;
    }

    private calculateAverageAbsorption(): number {
        const materials = this.room.config.materials;
        let totalAbsorption = 0;
        let count = 0;

        Object.values(materials).forEach(material => {
            totalAbsorption += material.absorption125Hz;
            totalAbsorption += material.absorption250Hz;
            totalAbsorption += material.absorption500Hz;
            totalAbsorption += material.absorption1kHz;
            totalAbsorption += material.absorption2kHz;
            totalAbsorption += material.absorption4kHz;
            totalAbsorption += material.absorption8kHz;
            totalAbsorption += material.absorption16kHz;
            count += 8; // Eight frequency bands
        });

        return totalAbsorption / count;
    }

    private generateEnvelope(sampleCount: number, rayHits: RayHit[]): Float32Array {
        const envelope = new Float32Array(sampleCount);
        const rt60 = this.calculateRT60(rayHits);

        for (let i = 0; i < sampleCount; i++) {
            const t = i / this.sampleRate;

            if (t < 0.05) {
                envelope[i] = Math.exp(-3 * t);
            } else {
                envelope[i] = Math.exp(-6.91 * t / rt60);
            }
        }

        return envelope;
    }

    private calculateRoomModes(dimensions: { width: number, height: number, depth: number }): RoomMode[] {
        const modes: RoomMode[] = [];
        const c = 343; // Speed of sound in m/s

        // Calculate axial modes
        const calculateMode = (l: number, m: number, n: number): RoomMode => {
            const frequency = (c/2) * Math.sqrt(
                Math.pow(l/dimensions.width, 2) +
                Math.pow(m/dimensions.height, 2) +
                Math.pow(n/dimensions.depth, 2)
            );

            const volume = dimensions.width * dimensions.height * dimensions.depth;
            const surfaceArea = 2 * (
                dimensions.width * dimensions.height +
                dimensions.width * dimensions.depth +
                dimensions.height * dimensions.depth
            );

            const rt60 = 0.161 * volume / (this.calculateAverageAbsorption() * surfaceArea);
            const decayTime = rt60 * 0.8;

            return { frequency, decayTime, rt60 };
        };

        // Add first few axial modes
        for (let i = 0; i <= 2; i++) {
            for (let j = 0; j <= 2; j++) {
                for (let k = 0; k <= 2; k++) {
                    if (i + j + k > 0) {
                        modes.push(calculateMode(i, j, k));
                    }
                }
            }
        }

        return modes;
    }

    private addRoomModes(leftIR: Float32Array, rightIR: Float32Array, modes: RoomMode[]): void {
        const modeAmplitude = 0.1;

        modes.forEach(mode => {
            const freq = mode.frequency;
            const decay = Math.exp(-3 * mode.decayTime / mode.rt60);

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