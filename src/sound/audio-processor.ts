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
    private camera: Camera;
    private device: GPUDevice;

    constructor(device: GPUDevice, room: Room, camera: Camera, sampleRate: number = 44100) {
        this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.sampleRate = this.audioCtx.sampleRate || sampleRate;
        this.impulseResponseBuffer = null;
        this.lastImpulseData = null;
        this.room = room;
        this.camera = camera;
        this.device = device;
        
        // Initialize spatial audio processor
        this.spatialProcessor = new SpatialAudioProcessor(this.sampleRate);
        
        // Initialize FDN and diffuse field model
        this.fdn = new FeedbackDelayNetwork(this.audioCtx, 16);
        this.diffuseFieldModel = new DiffuseFieldModel(this.sampleRate, {
            dimensions: {
                width: room.width,
                height: room.height,
                depth: room.depth
            },
            materials: room.materials || {
                walls: { absorption125Hz: 0.1, absorption250Hz: 0.1, absorption500Hz: 0.1, absorption1kHz: 0.1, absorption2kHz: 0.1, absorption4kHz: 0.1, absorption8kHz: 0.1, absorption16kHz: 0.1 },
                ceiling: { absorption125Hz: 0.15, absorption250Hz: 0.15, absorption500Hz: 0.15, absorption1kHz: 0.15, absorption2kHz: 0.15, absorption4kHz: 0.15, absorption8kHz: 0.15, absorption16kHz: 0.15 },
                floor: { absorption125Hz: 0.05, absorption250Hz: 0.05, absorption500Hz: 0.05, absorption1kHz: 0.05, absorption2kHz: 0.05, absorption4kHz: 0.05, absorption8kHz: 0.05, absorption16kHz: 0.05 }
            }
        });
    }

    async processRayHits(
        rayHits: RayHit[],
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
            // Validate input
            if (!rayHits || !Array.isArray(rayHits) || rayHits.length === 0) {
                console.warn('No valid ray hits to process');
                return;
            }

            // Filter out invalid hits
            const validHits = rayHits.filter(hit => hit && hit.position && hit.energies);
            if (validHits.length === 0) {
                console.warn('No valid ray hits after filtering');
                return;
            }

            const [leftIR, rightIR] = this.processRayHitsInternal(validHits);

            // Create interleaved stereo data for visualization
            const stereoData = new Float32Array(leftIR.length * 2);
            for (let i = 0; i < leftIR.length; i++) {
                stereoData[i * 2] = leftIR[i];
                stereoData[i * 2 + 1] = rightIR[i];
            }
            this.lastImpulseData = stereoData;

            // Set up the final impulse response buffer
            await this.setupImpulseResponseBuffer(leftIR, rightIR);
        } catch (error) {
            console.error('Error processing ray hits:', error);
            throw error; // Re-throw to allow caller to handle
        }
    }

    private processRayHitsInternal(hits: RayHit[]): [Float32Array, Float32Array] {
        const irLength = Math.ceil(this.sampleRate * 2); // 2 seconds IR
        const leftIR = new Float32Array(irLength);
        const rightIR = new Float32Array(irLength);
        
        try {
            // Sort hits by time for temporal coherence
            const sortedHits = [...hits].sort((a, b) => a.time - b.time);
            
            // Process early reflections
            const earlyHits = sortedHits.filter(hit => hit.time < 0.1); // First 100ms
            for (const hit of earlyHits) {
                const sampleIndex = Math.floor(hit.time * this.sampleRate);
                if (sampleIndex < 0 || sampleIndex >= irLength) continue;
                
                // Calculate spatial gains using HRTF
                const [leftGain, rightGain] = this.spatialProcessor.calculateImprovedHRTF(
                    hit.position,
                    this.camera.getPosition(),
                    this.camera.getFront(),
                    this.camera.getRight(),
                    this.camera.getUp()
                );
                
                // Calculate energy across frequency bands
                const totalEnergy = Object.values(hit.energies).reduce((sum, e) => sum + (typeof e === 'number' ? e : 0), 0);
                const amplitude = Math.sqrt(totalEnergy) * Math.exp(-hit.bounces * 0.5);
                
                // Apply temporal spreading for natural sound
                const spreadSamples = Math.min(0.005 * this.sampleRate, 50); // 5ms spread
                for (let j = -spreadSamples; j <= spreadSamples; j++) {
                    const idx = sampleIndex + j;
                    if (idx >= 0 && idx < irLength) {
                        const spread = Math.exp(-Math.abs(j) / (spreadSamples / 2));
                        leftIR[idx] += amplitude * leftGain * spread;
                        rightIR[idx] += amplitude * rightGain * spread;
                    }
                }
            }
            
            // Process late reflections using diffuse field model
            const lateHits = sortedHits.filter(hit => hit.time >= 0.1);
            if (lateHits.length > 0 && this.diffuseFieldModel) {
                const roomConfig = {
                    dimensions: {
                        width: this.room.width || 10,
                        height: this.room.height || 3,
                        depth: this.room.depth || 10
                    },
                    materials: this.room.materials || {
                        walls: { 
                            absorption125Hz: 0.1, absorption250Hz: 0.1, absorption500Hz: 0.1,
                            absorption1kHz: 0.1, absorption2kHz: 0.1, absorption4kHz: 0.1,
                            absorption8kHz: 0.1, absorption16kHz: 0.1 
                        },
                        ceiling: { 
                            absorption125Hz: 0.15, absorption250Hz: 0.15, absorption500Hz: 0.15,
                            absorption1kHz: 0.15, absorption2kHz: 0.15, absorption4kHz: 0.15,
                            absorption8kHz: 0.15, absorption16kHz: 0.15 
                        },
                        floor: { 
                            absorption125Hz: 0.05, absorption250Hz: 0.05, absorption500Hz: 0.05,
                            absorption1kHz: 0.05, absorption2kHz: 0.05, absorption4kHz: 0.05,
                            absorption8kHz: 0.05, absorption16kHz: 0.05 
                        }
                    }
                };

                try {
                    const [lateDiffuseL, lateDiffuseR] = this.diffuseFieldModel.processLateReverberation(
                        lateHits,
                        this.camera,
                        roomConfig,
                        this.sampleRate
                    );
                    
                    if (lateDiffuseL && lateDiffuseR && lateDiffuseL.length > 0 && lateDiffuseR.length > 0) {
                        // Crossfade and combine early reflections with late reverberation
                        const crossfadeStart = Math.floor(0.08 * this.sampleRate); // Start at 80ms
                        const crossfadeEnd = Math.floor(0.12 * this.sampleRate);   // End at 120ms
                        
                        for (let i = crossfadeStart; i < Math.min(irLength, lateDiffuseL.length); i++) {
                            let alpha = 1.0;
                            if (i < crossfadeEnd) {
                                alpha = (i - crossfadeStart) / (crossfadeEnd - crossfadeStart);
                            }
                            
                            leftIR[i] = leftIR[i] * (1 - alpha) + lateDiffuseL[i] * alpha;
                            rightIR[i] = rightIR[i] * (1 - alpha) + lateDiffuseR[i] * alpha;
                        }
                    } else {
                        console.warn('Invalid late reverberation response');
                    }
                } catch (error) {
                    console.error('Error processing late reverberation:', error);
                }
            }
            
            // Normalize the final impulse response
            this.sanitizeIRBuffers(leftIR, rightIR);
            
        } catch (error) {
            console.error('Error in processRayHitsInternal:', error);
            // Return empty buffers in case of error
            return [new Float32Array(irLength), new Float32Array(irLength)];
        }
        
        return [leftIR, rightIR];
    }

    /**
     * Ensures IR buffers are valid and properly normalized
     */
    private sanitizeIRBuffers(leftIR: Float32Array, rightIR: Float32Array): void {
        // First pass: fix any NaN or infinity values
        for (let i = 0; i < leftIR.length; i++) {
            if (!isFinite(leftIR[i])) leftIR[i] = 0;
            if (!isFinite(rightIR[i])) rightIR[i] = 0;
        }
        
        // Second pass: find maximum value for normalization
        let maxValue = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxValue = Math.max(maxValue, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }
        
        // Only normalize if we need to (values too high or too low)
        if (maxValue > 1.0 || maxValue < 0.1) {
            const targetPeak = 0.8; // Target 80% of maximum for headroom
            const gainFactor = (maxValue > 0) ? targetPeak / maxValue : 1.0;
            
            for (let i = 0; i < leftIR.length; i++) {
                leftIR[i] *= gainFactor;
                rightIR[i] *= gainFactor;
            }
            
            console.log(`Normalized IR buffers by factor ${gainFactor}, peak was ${maxValue}`);
        }
        
        // Apply gentle fade-in and fade-out to avoid clicks
        const fadeLength = Math.min(leftIR.length * 0.01, 100); // 1% or 100 samples max
        
        // Fade in
        for (let i = 0; i < fadeLength; i++) {
            const fadeGain = i / fadeLength;
            leftIR[i] *= fadeGain;
            rightIR[i] *= fadeGain;
        }
        
        // Fade out 
        for (let i = 0; i < fadeLength; i++) {
            const index = leftIR.length - 1 - i;
            const fadeGain = i / fadeLength;
            leftIR[index] *= fadeGain;
            rightIR[index] *= fadeGain;
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
        const listenerPos = this.camera.getPosition();
        const listenerFront = this.camera.getFront();
        const listenerRight = this.camera.getRight();
        const listenerUp = this.camera.getUp();

        // Create arrays to hold the data for efficiency
        const hitTimes: number[] = [];
        const hitPositions: vec3[] = [];
        const hitEnergies: number[] = [];
        const hitBounces: number[] = [];
        
        // Preprocess ray hits to avoid repeated calculations
        for (const hit of rayHits) {
            if (!hit || !hit.energies) continue;
            
            // Get average energy across frequency bands
            const energy = this.calculateWeightedEnergy(hit.energies);
            if (energy < 0.001) continue; // Skip very low energy hits
            
            hitTimes.push(hit.time);
            hitPositions.push(hit.position);
            hitEnergies.push(energy);
            hitBounces.push(hit.bounces);
        }
        
        // Process each sample in the impulse response
        for (let i = 0; i < leftIR.length; i++) {
            const currentTime = i * timeStep;
            
            // Find all hits that contribute to this time sample
            for (let j = 0; j < hitTimes.length; j++) {
                const hitTime = hitTimes[j];
                
                // Only consider hits that have arrived by this time
                if (hitTime <= currentTime) {
                    const position = hitPositions[j];
                    const energy = hitEnergies[j];
                    const bounces = hitBounces[j];
                    
                    // Calculate spatial position relative to listener
                    const toSource = vec3.create();
                    vec3.subtract(toSource, position, listenerPos);
                    const distance = vec3.length(toSource);
                    vec3.normalize(toSource, toSource);
                    
                    // Calculate spatial gains
                    const dotRight = vec3.dot(toSource, listenerRight);
                    const dotFront = vec3.dot(toSource, listenerFront);
                    const dotUp = vec3.dot(toSource, listenerUp);
                    
                    const azimuth = Math.atan2(dotRight, dotFront);
                    const elevation = Math.asin(Math.max(-1, Math.min(1, dotUp)));
                    
                    // Enhanced directional filtering
                    let leftGain = 0.5, rightGain = 0.5;
                    
                    // Strong directional effect for nearby sounds, less for distant
                    const distanceFactor = 1.0 / (1.0 + distance * 0.2);
                    
                    // Apply ILD (Interaural Level Difference)
                    if (azimuth < 0) { // Sound from left
                        leftGain = 0.7 + 0.3 * Math.cos(azimuth);
                        rightGain = 0.7 - 0.5 * Math.sin(azimuth);
                    } else { // Sound from right
                        rightGain = 0.7 + 0.3 * Math.cos(azimuth);
                        leftGain = 0.7 - 0.5 * Math.sin(azimuth);
                    }
                    
                    // Elevation affects sound intensity
                    const elevationFactor = 1.0 - 0.3 * Math.abs(elevation);
                    leftGain *= elevationFactor;
                    rightGain *= elevationFactor;
                    
                    // Early reflections are stronger
                    const timeScaling = Math.exp(-(currentTime - hitTime) * 3);
                    
                    // Bounce attenuation
                    const bounceScaling = Math.pow(0.8, bounces);
                    
                    // Calculate final amplitude
                    const amplitude = energy * timeScaling * bounceScaling * distanceFactor;
                    
                    // Add to impulse response
                    leftIR[i] += amplitude * leftGain;
                    rightIR[i] += amplitude * rightGain;
                }
            }
        }
        
        // Normalize to avoid clipping
        this.normalizeIR(leftIR, rightIR);
    }

    private normalizeIR(leftIR: Float32Array, rightIR: Float32Array): void {
        // Find maximum absolute value
        let maxAbs = 0;
        for (let i = 0; i < leftIR.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(leftIR[i]), Math.abs(rightIR[i]));
        }
        
        // Normalize if needed
        if (maxAbs > 1.0) {
            const scalar = 0.9 / maxAbs;
            for (let i = 0; i < leftIR.length; i++) {
                leftIR[i] *= scalar;
                rightIR[i] *= scalar;
            }
        }
    }

    private calculateWeightedEnergy(energies: FrequencyBands): number {
        // Apply perceptual weighting based on frequency importance
        return (
            energies.energy125Hz * 0.7 +
            energies.energy250Hz * 0.8 +
            energies.energy500Hz * 0.9 +
            energies.energy1kHz * 1.0 +
            energies.energy2kHz * 0.95 +
            energies.energy4kHz * 0.9 +
            energies.energy8kHz * 0.85 +
            energies.energy16kHz * 0.8
        ) / 8.0;
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
        
        // Create a new audio buffer with the proper sample rate
        this.impulseResponseBuffer = this.audioCtx.createBuffer(2, length, this.audioCtx.sampleRate);
        
        // Get the channel data for writing
        const leftChannel = this.impulseResponseBuffer.getChannelData(0);
        const rightChannel = this.impulseResponseBuffer.getChannelData(1);
        
        // Create interleaved stereo data for visualization
        this.lastImpulseData = new Float32Array(length * 2);
        
        // Copy the calculated IR to the audio buffer and create interleaved data
        for (let i = 0; i < length; i++) {
            // Ensure values are finite and in valid range
            const leftSample = isFinite(leftIR[i]) ? Math.max(-1, Math.min(1, leftIR[i])) : 0;
            const rightSample = isFinite(rightIR[i]) ? Math.max(-1, Math.min(1, rightIR[i])) : 0;
            
            // Store in audio buffer
            leftChannel[i] = leftSample;
            rightChannel[i] = rightSample;
            
            // Store in interleaved data
            this.lastImpulseData[i * 2] = leftSample;
            this.lastImpulseData[i * 2 + 1] = rightSample;
        }
        
        // Apply gentle fade-in and fade-out
        const fadeLength = Math.min(length * 0.01, 100); // 1% of total length or 100 samples
        
        // Fade in
        for (let i = 0; i < fadeLength; i++) {
            const fadeGain = i / fadeLength;
            const leftIndex = i * 2;
            const rightIndex = leftIndex + 1;
            
            leftChannel[i] *= fadeGain;
            rightChannel[i] *= fadeGain;
            this.lastImpulseData[leftIndex] *= fadeGain;
            this.lastImpulseData[rightIndex] *= fadeGain;
        }
        
        // Fade out
        for (let i = 0; i < fadeLength; i++) {
            const bufferIndex = length - i - 1;
            const fadeGain = i / fadeLength;
            const leftIndex = bufferIndex * 2;
            const rightIndex = leftIndex + 1;
            
            leftChannel[bufferIndex] *= fadeGain;
            rightChannel[bufferIndex] *= fadeGain;
            this.lastImpulseData[leftIndex] *= fadeGain;
            this.lastImpulseData[rightIndex] *= fadeGain;
        }
        
        console.log(`Created impulse response buffer: ${length} samples at ${this.audioCtx.sampleRate}Hz`);
    }

    /**
     * Play a test impulse through the current impulse response for testing
     */
    public async playTestImpulseResponse(): Promise<void> {
        if (!this.impulseResponseBuffer) {
            console.warn('No impulse response buffer available');
            return;
        }

        try {
            // Create a short click/impulse
            const clickDuration = 0.01; // 10ms
            const clickBuffer = this.audioCtx.createBuffer(1, this.audioCtx.sampleRate * clickDuration, this.audioCtx.sampleRate);
            const clickData = clickBuffer.getChannelData(0);
            
            // Create an impulse (single sample spike)
            clickData[0] = 1.0;
            
            // Create convolver node
            const convolver = this.audioCtx.createConvolver();
            convolver.buffer = this.impulseResponseBuffer;
            
            // Create gain to avoid clipping
            const outputGain = this.audioCtx.createGain();
            outputGain.gain.value = 0.5;
            
            // Create a source for the click
            const source = this.audioCtx.createBufferSource();
            source.buffer = clickBuffer;
            
            // Connect nodes
            source.connect(convolver);
            convolver.connect(outputGain);
            outputGain.connect(this.audioCtx.destination);
            
            // Play the sound
            source.start();
            console.log("Playing test impulse response");
        } catch (error) {
            console.error("Error playing test impulse:", error);
        }
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