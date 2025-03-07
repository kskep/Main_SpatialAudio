import { Camera } from '../camera/camera';
import { SpatialAudioProcessor, RayHit as SpatialRayHit } from './spatial-audio-processor';
import { Room } from '../room/room';
import { WaveformRenderer } from '../visualization/waveform-renderer';
import { vec3 } from 'gl-matrix';
import { FrequencyBands } from '../raytracer/ray';
import { FeedbackDelayNetwork } from './feedback-delay-network';
import { DiffuseFieldModel } from './diffuse-field-model';

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
    private fdn: FeedbackDelayNetwork | null = null;
    private diffuseFieldModel: DiffuseFieldModel | null = null;

    constructor(device: GPUDevice, room: Room, sampleRate: number = 44100) {
        this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.sampleRate = this.audioCtx.sampleRate || sampleRate;
        this.impulseResponseBuffer = null;
        this.lastImpulseData = null;
        this.spatialProcessor = new SpatialAudioProcessor(device, this.sampleRate);
        this.room = room;
        
        this.fdn = new FeedbackDelayNetwork(this.audioCtx, 16);
        this.diffuseFieldModel = new DiffuseFieldModel(this.sampleRate, room.config);
    }

    public normalizeAndApplyEnvelope(leftIR: Float32Array, rightIR: Float32Array, envelope: Float32Array): void {
        let maxAmplitude = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxAmplitude = Math.max(maxAmplitude, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }

        if (maxAmplitude > 0) {
            const targetAmplitude = 0.7;
            const normalizationFactor = targetAmplitude / maxAmplitude;
            
            for (let i = 0; i < leftIR.length; i++) {
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
            this.lastRayHits = rayHits;

            // Limit the number of ray hits for processing
            const maxRayHits = 10000;
            const hitCount = Math.min(rayHits.length, maxRayHits);
            console.log(`Processing ${hitCount} ray hits for IR calculation (limited from ${rayHits.length})`);
            
            // Sort by time and limit
            const sortedHits = [...rayHits]
                .sort((a, b) => a.time - b.time)
                .slice(0, hitCount);

            let leftIR: Float32Array, rightIR: Float32Array;
            
            // Process in smaller chunks for better performance
            const chunkSize = 700;
            const chunks = [];
            for (let i = 0; i < sortedHits.length; i += chunkSize) {
                const chunk = sortedHits.slice(i, i + chunkSize);
                chunks.push(chunk);
            }

            console.log(`Processing ${chunks.length} chunks of ray hits`);

            if (chunks.length > 1) {
                const results = await Promise.all(chunks.map(chunk =>
                    this.spatialProcessor.processSpatialAudio(camera, chunk, params, this.room)
                ));

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

            // Always validate and fix the buffers
            this.fixInvalidValues(leftIR);
            this.fixInvalidValues(rightIR);

            console.log(`Generated initial IR buffers: L=${leftIR.length}, R=${rightIR.length}`);

            // Use the validateIRBuffers function which now fixes rather than fails
            if (!this.validateIRBuffers(leftIR, rightIR)) {
                throw new Error('Invalid IR buffers generated');
            }

            const sampleCount = leftIR.length;
            const envelope = this.generateEnvelope(sampleCount, sortedHits);
            
            // Limit number of room modes for performance
            const roomModes = this.calculateRoomModes().slice(0, 10);

            console.log('Applying room acoustics processing...');
            this.addRoomModes(leftIR, rightIR, roomModes);
            
            // Only do wave interference if requested and performance allows
            //this.applyWaveInterference(leftIR, rightIR, sortedHits);
            
            this.normalizeAndApplyEnvelope(leftIR, rightIR, envelope);

            this.setupImpulseResponseBuffer(leftIR, rightIR);

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

    private addRoomModes(leftIR: Float32Array, rightIR: Float32Array, modes: number[]): void {
        // Limit the number of modes to process
        const maxModes = Math.min(modes.length, 10); // Process only up to 10 most significant modes
        const modeAmplitude = 0.1;
        const decayBase = Math.exp(-3 * 0.5 / 0.5);
        
        // Process in chunks for better performance
        const chunkSize = 1000;
        
        for (let m = 0; m < maxModes; m++) {
            const freq = modes[m];
            const decay = decayBase;
            
            // Process in chunks to avoid long-running loops
            for (let start = 0; start < leftIR.length; start += chunkSize) {
                const end = Math.min(start + chunkSize, leftIR.length);
                
                for (let t = start; t < end; t++) {
                    const sample = modeAmplitude * decay *
                        Math.sin(2 * Math.PI * freq * t / this.sampleRate);
                    leftIR[t] += sample;
                    rightIR[t] += sample;
                }
            }
        }
    }

    private calculateDecayCurve(numSamples: number): Float32Array {
        const decayCurve = new Float32Array(numSamples);
        const sampleRate = this.audioCtx.sampleRate;
        
        if (!this.lastRayHits || this.lastRayHits.length === 0) {
            console.warn('No ray hits available for decay curve calculation');
            return decayCurve.fill(1.0); 
        }

        // Limit the number of ray hits to process for the curve
        const maxHits = Math.min(this.lastRayHits.length, 1000);
        const sortedHits = [...this.lastRayHits]
            .sort((a, b) => a.time - b.time)
            .slice(0, maxHits);
        
        const startTime = sortedHits[0].time;
        const endTime = sortedHits[sortedHits.length - 1].time;
        const timeRange = endTime - startTime;

        // Reduce number of bins for faster calculations
        const numBins = Math.min(100, numSamples); 
        const binDuration = timeRange / numBins;
        const energyBins = new Array(numBins).fill(0);
        
        let maxBinEnergy = 0;
        
        // Process in batches for better performance
        const hitBatchSize = 100;
        for (let hitIndex = 0; hitIndex < sortedHits.length; hitIndex += hitBatchSize) {
            const endIndex = Math.min(hitIndex + hitBatchSize, sortedHits.length);
            
            for (let i = hitIndex; i < endIndex; i++) {
                const hit = sortedHits[i];
                const binIndex = Math.floor((hit.time - startTime) / binDuration);
                if (binIndex >= 0 && binIndex < numBins) {
                    const energies = hit.energies || {};
                    
                    // Simpler energy calculation for better performance
                    const totalEnergy = 
                        (energies.energy1kHz || 0) * 1.0 + // Use 1kHz as a representative sample
                        (energies.energy500Hz || 0) * 0.5;
                    
                    energyBins[binIndex] += totalEnergy;
                    maxBinEnergy = Math.max(maxBinEnergy, energyBins[binIndex]);
                }
            }
        }

        // Normalize and apply the curve more efficiently
        if (maxBinEnergy > 0) {
            for (let i = 0; i < numBins; i++) {
                energyBins[i] = Math.sqrt(energyBins[i] / maxBinEnergy);
            }
        }

        // Efficient curve application
        for (let i = 0; i < numSamples; i++) {
            const timeInSeconds = i / sampleRate;
            const relativeBinPosition = (timeInSeconds - startTime) / binDuration;
            const binIndex = Math.floor(relativeBinPosition);
            
            if (binIndex < 0) {
                decayCurve[i] = 1.0; 
            } else if (binIndex >= numBins) {
                decayCurve[i] = energyBins[numBins - 1] || 0; 
            } else {
                decayCurve[i] = energyBins[binIndex];
            }
        }

        return decayCurve;
    }

    private fixInvalidValues(buffer: Float32Array): void {
        for (let i = 0; i < buffer.length; i++) {
            if (isNaN(buffer[i]) || !isFinite(buffer[i])) {
                buffer[i] = 0;
            }
        }
    }

    private applyWaveInterference(leftIR: Float32Array, rightIR: Float32Array, rayHits: RayHit[]): void {
        const timeStep = 1 / this.sampleRate;

        for (let i = 0; i < leftIR.length; i++) {
            const currentTime = i * timeStep;
            let leftSum = 0;
            let rightSum = 0;

            for (const hit of rayHits) {
                if (!hit || !hit.energies) continue; // Skip invalid hits
                
                if (hit.time <= currentTime) {
                    // Ensure all values are valid with fallbacks
                    const arrivalTime = isFinite(hit.time) ? hit.time : 0;
                    const isEarlyReflection = arrivalTime < 0.1; 
                    
                    // Apply safer amplitude calculation
                    const amplitudeScale = isFinite(arrivalTime) ? 
                        (isEarlyReflection ? 
                            3.0 * Math.exp(-Math.min(arrivalTime * 5, 10)) : 
                            0.7 * Math.exp(-Math.min(arrivalTime * 2, 10))   
                        ) : 0;

                    // Sanitize frequency and phase values
                    const frequency = Math.max(hit.frequency || 440, 20);
                    const dopplerShift = Math.max(hit.dopplerShift || 1, 0.1);
                    const phase = isFinite(hit.phase) ? hit.phase : 0;

                    // Safer calculation
                    const timeSinceArrival = Math.max(currentTime - arrivalTime, 0);
                    const instantPhase = phase +
                        2 * Math.PI * frequency * (1 + dopplerShift) * timeSinceArrival;

                    // Validate energy values and use defaults if needed
                    const frequencyWeights = {
                        energy125Hz: 0.7,  
                        energy250Hz: 0.8,
                        energy500Hz: 0.9,
                        energy1kHz: 1.0,   
                        energy2kHz: 0.95,
                        energy4kHz: 0.9,
                        energy8kHz: 0.85,
                        energy16kHz: 0.8   
                    };

                    let totalEnergy = 0;
                    let totalWeight = 0;
                    for (const [band, weight] of Object.entries(frequencyWeights)) {
                        // Only add valid energy values
                        if (hit.energies[band] !== undefined && 
                            isFinite(hit.energies[band])) {
                            totalEnergy += hit.energies[band] * weight;
                            totalWeight += weight;
                        }
                    }

                    const normalizedEnergy = totalWeight > 0 ? totalEnergy / totalWeight : 0;
                    
                    // Validate distance
                    const distance = isFinite(hit.distance) ? Math.max(hit.distance || 1, 0.1) : 1;
                    
                    // Final amplitude calculation with full validation
                    const amplitude = isFinite(normalizedEnergy) ? 
                        Math.sqrt(Math.max(0, normalizedEnergy)) / (4 * Math.PI * distance) : 0;

                    // Avoid NaN with full validation
                    let contribution = 0;
                    if (isFinite(amplitude) && isFinite(amplitudeScale) && isFinite(instantPhase)) {
                        contribution = amplitude * amplitudeScale * Math.sin(instantPhase);
                        // Final NaN check
                        if (!isFinite(contribution)) contribution = 0;
                    }

                    leftSum += contribution;
                    rightSum += contribution;
                }
            }

            // Ensure we're storing finite values
            leftIR[i] = isFinite(leftSum) ? leftSum : 0;
            rightIR[i] = isFinite(rightSum) ? rightSum : 0;
        }
    }

    private calculateSpatialGains(position: vec3, listenerPos: vec3, listenerForward: vec3, listenerRight: vec3): [number, number] {
        const direction = vec3.create();
        vec3.subtract(direction, position, listenerPos);
        vec3.normalize(direction, direction);
        
        const dot = vec3.dot(direction, listenerRight);
        const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
        
        const leftGain = 0.5 + 0.5 * Math.cos(angle); 
        const rightGain = 0.5 + 0.5 * Math.cos(Math.PI - angle);
        
        return [leftGain, rightGain];
    }

    private calculateRT60(rayHits: RayHit[]): number {
        if (!rayHits.length) return 0.5; 

        const initialEnergy = rayHits[0].energy;
        for (const hit of rayHits) {
            if (hit.energy <= initialEnergy * 0.001) { 
                return hit.time;
            }
        }

        return Math.max(0.5, rayHits[rayHits.length - 1].time);
    }

    private generateEnvelope(sampleCount: number, rayHits: RayHit[]): Float32Array {
        const envelope = new Float32Array(sampleCount);
        const rt60 = Math.max(0.5, this.calculateRT60(rayHits)); 
        
        const directSoundTime = 0.005; 
        const earlyReflectionsEnd = 0.08; 
        const directSoundSamples = Math.floor(directSoundTime * this.sampleRate);
        const earlyReflectionSamples = Math.floor(earlyReflectionsEnd * this.sampleRate);
        
        for (let i = 0; i < sampleCount; i++) {
            const t = i / this.sampleRate;
            
            if (i < directSoundSamples) {
                envelope[i] = 1.0;
            } else if (i < earlyReflectionSamples) {
                const normalizedPos = (i - directSoundSamples) / (earlyReflectionSamples - directSoundSamples);
                envelope[i] = 0.9 - 0.2 * normalizedPos;
            } else {
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

        for (const key of ['left', 'right', 'floor', 'ceiling', 'front', 'back']) {
            const material = materials[key];
            if (material) {
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

                const weights = [0.5, 0.7, 1.0, 1.0, 1.0, 0.7, 0.5, 0.3];
                
                for (let i = 0; i < absorptions.length; i++) {
                    totalAbsorption += absorptions[i] * weights[i];
                    count += weights[i];
                }
            }
        }

        return count > 0 ? totalAbsorption / count : 0.1;
    }

    private calculateMode(l: number, m: number, n: number): number {
        const dimensions = this.room.config.dimensions;
        const c = 343; 
        
        const width = Math.max(dimensions.width, 0.1);
        const height = Math.max(dimensions.height, 0.1);
        const depth = Math.max(dimensions.depth, 0.1);

        const freq = (c/2) * Math.sqrt(
            Math.pow(l/width, 2) + 
            Math.pow(m/height, 2) + 
            Math.pow(n/depth, 2)
        );

        const avgAbsorption = this.calculateAverageAbsorption();
        const modeAmplitude = 1.0 - avgAbsorption;

        return freq * modeAmplitude;
    }

    private calculateRoomModes(): number[] {
        const modes: number[] = [];
        
        for (let l = 0; l <= 2; l++) {
            for (let m = 0; m <= 2; m++) {
                for (let n = 0; n <= 2; n++) {
                    if (l + m + n > 0) { 
                        const modeFreq = this.calculateMode(l, m, n);
                        if (modeFreq >= 20 && modeFreq <= 20000) { 
                            modes.push(modeFreq);
                        }
                    }
                }
            }
        }

        return modes.sort((a, b) => a - b);
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

        const hasInvalidValues = (buffer: Float32Array) => {
            for (let i = 0; i < buffer.length; i++) {
                if (isNaN(buffer[i]) || !isFinite(buffer[i])) {
                    console.error(`Invalid value at index ${i}:`, buffer[i]);
                    buffer[i] = 0; // Fix invalid values
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
        
        const leftChannel = this.impulseResponseBuffer.getChannelData(0);
        const rightChannel = this.impulseResponseBuffer.getChannelData(1);
        
        for (let i = 0; i < length; i++) {
            leftChannel[i] = leftIR[i] * decayCurve[i];
            rightChannel[i] = rightIR[i] * decayCurve[i];
        }

        this.normalizeBuffer(this.impulseResponseBuffer);
    }

    private normalizeBuffer(buffer: AudioBuffer): void {
        const leftChannel = buffer.getChannelData(0);
        const rightChannel = buffer.getChannelData(1);
        
        let maxAmplitude = 0;
        for (let i = 0; i < buffer.length; i++) {
            maxAmplitude = Math.max(maxAmplitude, 
                Math.abs(leftChannel[i]), 
                Math.abs(rightChannel[i]));
        }
        
        if (maxAmplitude > 1.0) {
            const scalar = 0.99 / maxAmplitude;
            for (let i = 0; i < buffer.length; i++) {
                leftChannel[i] *= scalar;
                rightChannel[i] *= scalar;
            }
        }
    }

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
            const duration = 2; 
            const frequency = 440; 
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
            const duration = 2;
            const frequency = 440;
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;
            
            const sineBuffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = sineBuffer.getChannelData(0);
            
            const attackTime = 0.01; 
            const releaseTime = 0.05; 
            const attackSamples = Math.floor(attackTime * sampleRate);
            const releaseSamples = Math.floor(releaseTime * sampleRate);
            
            for (let i = 0; i < samples; i++) {
                let amplitude = 1.0;
                
                if (i < attackSamples) {
                    amplitude = i / attackSamples;
                }
                
                if (i > samples - releaseSamples) {
                    amplitude = (samples - i) / releaseSamples;
                }
                
                channelData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * amplitude;
            }

            const source = this.audioCtx.createBufferSource();
            source.buffer = sineBuffer;
            
            const convolver = this.audioCtx.createConvolver();
            convolver.buffer = this.impulseResponseBuffer;
            const wetGain = this.audioCtx.createGain();
            wetGain.gain.value = dryWetMix; 
            
            const dryGain = this.audioCtx.createGain();
            dryGain.gain.value = 1 - dryWetMix; 
            
            const masterGain = this.audioCtx.createGain();
            masterGain.gain.value = 0.3; 
            
            source.connect(convolver);    
            convolver.connect(wetGain);   
            source.connect(dryGain);      
            
            wetGain.connect(masterGain);  
            dryGain.connect(masterGain);
            
            masterGain.connect(this.audioCtx.destination);
            
            source.start();
            
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
            const duration = 2;
            const sampleRate = this.audioCtx.sampleRate;
            const samples = duration * sampleRate;
            
            const noiseBuffer = this.audioCtx.createBuffer(1, samples, sampleRate);
            const channelData = noiseBuffer.getChannelData(0);
            
            const decayFactor = this.calculateDecayCurve(samples);
            
            for (let i = 0; i < samples; i++) {
                channelData[i] = (Math.random() * 2 - 1) * decayFactor[i];
            }

            const convolver = this.audioCtx.createConvolver();
            convolver.buffer = this.impulseResponseBuffer;

            const gainNode = this.audioCtx.createGain();
            gainNode.gain.value = 0.1;

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

    public clearRayHits(): void {
        this.lastRayHits = null;
    }

    public generateMultibandImpulseResponse(rayHits: RayHit[], responseLength: number): Float32Array[] {
        const impulseResponses: Float32Array[] = [];
        
        for (let band = 0; band < 8; band++) {
            const ir = new Float32Array(responseLength);
            impulseResponses.push(ir);
        }
        
        const sampleRate = this.audioCtx.sampleRate;
        
        rayHits.forEach(hit => {
            const sampleIndex = Math.floor(hit.time * sampleRate);
            
            if (sampleIndex < 0 || sampleIndex >= responseLength) return;
            
            const energies = hit.energies || {};
            
            const totalEnergy = 
                (energies.energy125Hz || 0) * 0.7 +
                (energies.energy250Hz || 0) * 0.8 +
                (energies.energy500Hz || 0) * 0.9 +
                (energies.energy1kHz || 0) * 1.0 +
                (energies.energy2kHz || 0) * 0.95 +
                (energies.energy4kHz || 0) * 0.9 +
                (energies.energy8kHz || 0) * 0.85 +
                (energies.energy16kHz || 0) * 0.8;
            
            for (let band = 0; band < 8; band++) {
                impulseResponses[band][sampleIndex] += totalEnergy;
            }
        });
        
        for (let band = 0; band < 8; band++) {
            this.normalizeImpulseResponse(impulseResponses[band]);
        }
        
        return impulseResponses;
    }

    private normalizeImpulseResponse(ir: Float32Array): void {
        let maxAbs = 0;
        for (let i = 0; i < ir.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(ir[i]));
        }
        
        if (maxAbs > 0) {
            const scale = 0.95 / maxAbs; 
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
        
        const filterBands = [
            { lowFreq: 88, highFreq: 177 },   
            { lowFreq: 177, highFreq: 354 },  
            { lowFreq: 354, highFreq: 707 },  
            { lowFreq: 707, highFreq: 1414 }, 
            { lowFreq: 1414, highFreq: 2828 }, 
            { lowFreq: 2828, highFreq: 5657 }, 
            { lowFreq: 5657, highFreq: 11314 }, 
            { lowFreq: 11314, highFreq: 20000 } 
        ];
        
        const convolvers = impulseResponses.map(ir => {
            const convolver = context.createConvolver();
            const irBuffer = context.createBuffer(1, ir.length, sampleRate);
            irBuffer.getChannelData(0).set(ir);
            convolver.buffer = irBuffer;
            return convolver;
        });
        
        const offlineContext = new OfflineAudioContext(channels, length, sampleRate);
        
        const bandInputs = filterBands.map((band, i) => {
            const input = offlineContext.createBufferSource();
            input.buffer = audioBuffer;
            
            const filter = offlineContext.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.value = Math.sqrt(band.lowFreq * band.highFreq);
            filter.Q.value = 1.0;
            
            input.connect(filter);
            return filter;
        });
        
        const merger = offlineContext.createChannelMerger(8);
        bandInputs.forEach((filter, i) => {
            const convolver = offlineContext.createConvolver();
            const irBuffer = offlineContext.createBuffer(1, impulseResponses[i].length, sampleRate);
            irBuffer.getChannelData(0).set(impulseResponses[i]);
            convolver.buffer = irBuffer;
            
            filter.connect(convolver);
            convolver.connect(merger);
        });
        
        merger.connect(offlineContext.destination);
        
        bandInputs.forEach(filter => {
            const source = filter.context.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(filter);
            source.start();
        });
        
        return await offlineContext.startRendering();
    }
}